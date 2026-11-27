"""Hardened Colab CLI runner for P1 (OLMo) -> P2a (Gemma 4B) -> P2b (Gemma 4B vs 12B).

Features:
  - Can attach to an existing user-provided session (--config + --session + --keep-session)
    OR allocate a fresh session with bounded retries (max 3 G4, max 2 A100; L4 only if --skip-p1).
  - Hashes complete upload bundle before running.
  - Probes remote GPU VRAM before running stages (so standalone --skip-p1 --skip-p2a works).
  - Checks explicit success markers in remote stdout (since colab-cli 0.8.0 can exit 0 on
    remote Python tracebacks!) and enforces absolute wall-clock deadlines.
  - Validates hardware, pinned revisions, and required keys on every downloaded JSON.
  - Verifies kernel restarts and unconditionally releases any session it allocated.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
STATE_DIR = Path(os.environ.get("XDG_STATE_HOME", Path.home() / ".local" / "state")) / "persona-plan"
STATE_FILE = STATE_DIR / "session.json"

TRANSIENT = re.compile(
    r"\b(?:503|502|504)\b|service unavailable|bad gateway|gateway timeout|"
    r"read timed out|connect timed out|no backend with gpu available|cannot assign|unavailable",
    re.IGNORECASE,
)
ACCOUNT_BLOCK = re.compile(
    r"\b(?:401|403|412|429)\b|quota|compute units|permission denied|"
    r"authentication required|too many (?:assignments|requests|active sessions)|"
    r"precondition failed|rate.limit|resource_exhausted",
    re.IGNORECASE,
)


def _colab_cmd(config_path: str | None, *args: str) -> list[str]:
    cmd = ["colab"]
    if config_path:
        cmd.extend(["--config", config_path])
    cmd.extend(args)
    return cmd


def run_colab(*args: str, config_path: str | None = None, timeout: int = 180) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env.update(COLAB_CLI_ASSIGN_RETRIES="0", COLAB_CLI_REQUEST_TIMEOUT="90")
    return subprocess.run(
        _colab_cmd(config_path, *args),
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=timeout,
        check=False,
    )


def list_sessions_ok(config_path: str | None = None) -> tuple[bool, str]:
    try:
        p = run_colab("sessions", config_path=config_path, timeout=60)
        return (p.returncode == 0), p.stdout
    except subprocess.TimeoutExpired:
        return False, ""


def persist_state(data: dict[str, Any]) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True, mode=0o700)
    STATE_FILE.write_text(json.dumps(data, indent=2) + "\n")


def hash_bundle(files: list[Path]) -> dict[str, str]:
    out = {}
    for f in files:
        assert f.is_file(), f"Missing bundle file: {f}"
        out[str(f.relative_to(ROOT))] = hashlib.sha256(f.read_bytes()).hexdigest()
    return out


def parse_endpoints(sessions_txt: str) -> set[str]:
    return set(re.findall(r"\[[^\]]+\]\s+(\S+)\s+\|", sessions_txt))


def acquire_session(
    session_id: str,
    gpu_schedule: list[tuple[str, int]],
    interval: int,
    config_path: str | None = None,
) -> tuple[str, str]:
    ok, pre_sessions = list_sessions_ok(config_path)
    if not ok:
        raise RuntimeError("Could not query `colab sessions` before allocation.")
    pre_endpoints = parse_endpoints(pre_sessions)
    if session_id in pre_sessions:
        raise RuntimeError(f"Session {session_id} already exists before allocation.")

    for gpu, max_tries in gpu_schedule:
        for attempt_idx in range(1, max_tries + 1):
            t0 = time.monotonic()
            persist_state({
                "session_id": session_id,
                "requested_gpu": gpu,
                "attempt": attempt_idx,
                "timestamp": time.time(),
                "status": "requesting",
                "owned": True,
            })
            print(f"[colab] Requesting {gpu} (attempt {attempt_idx}/{max_tries}) as {session_id}...", flush=True)
            proc = None
            try:
                proc = run_colab("new", "-s", session_id, "--gpu", gpu, config_path=config_path, timeout=150)
            except subprocess.TimeoutExpired:
                ok_post, post = list_sessions_ok(config_path)
                if not ok_post:
                    raise RuntimeError("Allocation timed out and `colab sessions` failed; refusing to retry.")
                new_eps = parse_endpoints(post) - pre_endpoints
                if session_id in post and len(new_eps) == 1:
                    ep = next(iter(new_eps))
                    persist_state({"session_id": session_id, "endpoint": ep, "gpu": gpu, "status": "active", "owned": True})
                    return gpu, ep
                if new_eps:
                    raise RuntimeError(f"Ambiguous backend assignment after timeout ({new_eps}); refusing to retry.")

            if proc is not None:
                if proc.returncode == 0:
                    ok_post, post = list_sessions_ok(config_path)
                    new_eps = (parse_endpoints(post) - pre_endpoints) if ok_post else set()
                    ep = next(iter(new_eps)) if len(new_eps) == 1 else ""
                    persist_state({"session_id": session_id, "endpoint": ep, "gpu": gpu, "status": "active", "owned": True})
                    print(f"[colab] Assigned {gpu} to {session_id} (endpoint={ep or 'unknown'}).", flush=True)
                    return gpu, ep
                tail = " ".join(proc.stdout.splitlines()[-4:])
                print(f"[colab] {gpu} attempt {attempt_idx} failed (rc={proc.returncode}): {tail}", flush=True)
                if ACCOUNT_BLOCK.search(tail):
                    raise RuntimeError(f"Account/quota/rate limit hit; stopping allocation: {tail}")
                ok_post, post = list_sessions_ok(config_path)
                if ok_post and (session_id in post or (parse_endpoints(post) - pre_endpoints)):
                    raise RuntimeError(f"Unexpected backend assignment after error; stopping:\n{post}")
                if not TRANSIENT.search(tail):
                    print(f"[colab] Non-transient error on {gpu}; advancing to next GPU tier.", flush=True)
                    break

            wait = max(0.0, interval - (time.monotonic() - t0))
            if wait > 0:
                time.sleep(wait)

    raise RuntimeError("Exhausted all bounded GPU allocation attempts.")


def upload_file(session_id: str, local_path: Path, remote_path: str, config_path: str | None = None) -> None:
    p = run_colab("upload", "-s", session_id, str(local_path), remote_path, config_path=config_path, timeout=180)
    if p.returncode != 0:
        raise RuntimeError(f"Upload failed ({local_path} -> {remote_path}): {p.stdout}")


def restart_kernel(session_id: str, config_path: str | None = None) -> None:
    p = run_colab("restart-kernel", "-s", session_id, config_path=config_path, timeout=120)
    if p.returncode != 0:
        raise RuntimeError(f"Kernel restart failed for {session_id}: {p.stdout}")
    time.sleep(5)
    print(f"[colab] Kernel restarted cleanly for {session_id}.", flush=True)


def probe_remote_gpu(session_id: str, config_path: str | None = None) -> tuple[str, float]:
    import tempfile
    code = (
        "import json, torch\n"
        "assert torch.cuda.is_available()\n"
        "print('PROBE_JSON:' + json.dumps({'name': torch.cuda.get_device_name(0), "
        "'vram_gb': round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2)}), flush=True)\n"
    )
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as tf:
        tf.write(code)
        probe_path = tf.name
    try:
        last_err = ""
        for attempt in range(1, 4):
            try:
                proc = subprocess.run(
                    _colab_cmd(config_path, "exec", "-s", session_id, "--timeout", "30", "-f", probe_path),
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    timeout=50,
                    check=False,
                )
                m = re.search(r"PROBE_JSON:(\{.*\})", proc.stdout)
                if proc.returncode == 0 and m:
                    info = json.loads(m.group(1))
                    return str(info["name"]), float(info["vram_gb"])
                last_err = f"rc={proc.returncode} out={proc.stdout[-300:]}"
            except subprocess.TimeoutExpired:
                last_err = f"probe attempt {attempt} timed out after 50s"
            time.sleep(3)
        raise RuntimeError(f"Remote GPU probe failed after 3 attempts: {last_err}")
    finally:
        Path(probe_path).unlink(missing_ok=True)


def exec_file(
    session_id: str,
    local_script: Path,
    *,
    success_marker: str,
    wall_timeout: int,
    idle_timeout: int = 900,
    config_path: str | None = None,
) -> None:
    print(f"[colab] Running {local_script.name} (wall={wall_timeout}s, idle={idle_timeout}s)...", flush=True)
    proc = subprocess.Popen(
        _colab_cmd(config_path, "exec", "-s", session_id, "--timeout", str(idle_timeout), "-f", str(local_script)),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    saw_marker = False
    t0 = time.monotonic()
    assert proc.stdout is not None
    try:
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            if success_marker in line:
                saw_marker = True
            if time.monotonic() - t0 > wall_timeout:
                proc.kill()
                raise RuntimeError(f"Wall-clock timeout ({wall_timeout}s) exceeded for {local_script.name}")
        rc = proc.wait(timeout=60)
    finally:
        if proc.poll() is None:
            proc.kill()
    if rc != 0 or not saw_marker:
        raise RuntimeError(
            f"Execution failed for {local_script.name} (rc={rc}, saw_marker={saw_marker})"
        )


def download_and_verify(
    session_id: str,
    remote_path: str,
    local_dest: Path,
    required_keys: tuple[str, ...],
    *,
    expected_revs: dict[str, str] | None = None,
    config_path: str | None = None,
) -> dict[str, Any]:
    tmp = local_dest.with_suffix(".tmp.json")
    p = run_colab("download", "-s", session_id, remote_path, str(tmp), config_path=config_path, timeout=300)
    if p.returncode != 0:
        raise RuntimeError(f"Download failed ({remote_path}): {p.stdout}")
    data = json.loads(tmp.read_text())
    for k in required_keys:
        if k not in data:
            raise RuntimeError(f"Downloaded {remote_path} missing required key {k!r}")
    if expected_revs:
        if "revisions" in data:
            for k, v in expected_revs.items():
                if data["revisions"].get(k) != v:
                    raise RuntimeError(f"Revision mismatch in {remote_path}: {k}={data['revisions'].get(k)} != {v}")
        elif "revision" in data and "revision" in expected_revs:
            if data["revision"] != expected_revs["revision"]:
                raise RuntimeError(f"Revision mismatch in {remote_path}: {data['revision']} != {expected_revs['revision']}")
    tmp.replace(local_dest)
    print(f"[colab] Verified & saved {local_dest} (hw={data.get('hardware')}, vram={data.get('vram_gb')} GB)", flush=True)
    return data


def release_session(session_id: str, endpoint: str = "", config_path: str | None = None) -> None:
    for attempt in range(1, 4):
        print(f"[colab] Stopping {session_id} (attempt {attempt}/3)...", flush=True)
        try:
            stop_p = run_colab("stop", "-s", session_id, config_path=config_path, timeout=90)
            stop_rc = stop_p.returncode
        except subprocess.TimeoutExpired:
            stop_rc = -1
            print(f"[colab] Stop command timed out on attempt {attempt}/3.", flush=True)
        time.sleep(3)
        ok, sess_txt = list_sessions_ok(config_path)
        if not ok:
            print(f"[colab] Could not verify release (colab sessions failed on attempt {attempt}/3).", flush=True)
            continue
        still_present = (session_id in sess_txt) or (bool(endpoint) and endpoint in sess_txt)
        if not still_present:
            persist_state({"session_id": session_id, "endpoint": endpoint, "status": "released", "timestamp": time.time()})
            print(f"[colab] Verified release of {session_id}.", flush=True)
            return
        print(f"[colab] Session still visible after stop (rc={stop_rc}): {sess_txt[:200]}", flush=True)
    raise RuntimeError(f"FAILED TO VERIFY RELEASE OF COLAB SESSION {session_id}! Stop manually immediately.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", type=int, default=45)
    ap.add_argument("--config", type=str, default=None, help="Optional colab --config state file path.")
    ap.add_argument("--session", type=str, default=None, help="Reuse an already-active session instead of allocating.")
    ap.add_argument("--keep-session", action="store_true", help="Do not stop the session on exit (default when --session is passed).")
    ap.add_argument("--skip-p1", action="store_true")
    ap.add_argument("--skip-p2a", action="store_true")
    ap.add_argument("--skip-p2b", action="store_true")
    args = ap.parse_args()

    assert shutil.which("colab"), "colab CLI not found."
    bundle_files = [
        ROOT / "experiments" / "p1_cases_manifest.json",
        ROOT / "experiments" / "p1_olmo_realizations.py",
        ROOT / "experiments" / "p2a_gemma_restoration.py",
        ROOT / "experiments" / "p2b_gemma_scale.py",
    ]
    bundle_hashes = hash_bundle(bundle_files)
    print("[colab] Bundle SHA-256 hashes:", json.dumps(bundle_hashes, indent=2), flush=True)

    reusing_external = args.session is not None
    session_id = args.session or f"persona-plan-{uuid.uuid4().hex[:10]}"
    keep_session = args.keep_session or reusing_external
    # Exclude L4 when P1 is enabled (P1 peaks ~27 GiB during Gram + head build)
    gpu_schedule = [("G4", 3), ("A100", 2)] if not args.skip_p1 else [("G4", 3), ("A100", 2), ("L4", 2)]

    t_session0 = time.monotonic()
    P1_P2A_CAP_SEC = 3600  # shared 60-min budget for P1 + P2a
    SESSION_CAP_SEC = 5400  # 90-min total ceiling

    allocated_here = False
    endpoint = ""
    try:
        if not reusing_external:
            _, endpoint = acquire_session(session_id, gpu_schedule, args.interval, config_path=args.config)
            allocated_here = True
        else:
            print(f"[colab] Reusing existing session {session_id} (keep_session={keep_session}).", flush=True)
            # Clear GPU memory cleanly without risking a wedged restart websocket
            run_colab("restart-kernel", "-s", session_id, config_path=args.config, timeout=90)
            time.sleep(6)

        gpu_name, vram_gb = probe_remote_gpu(session_id, config_path=args.config)
        print(f"[colab] Remote GPU verified: {gpu_name} ({vram_gb:.2f} GB VRAM)", flush=True)
        if not args.skip_p1 and vram_gb < 30.0:
            raise RuntimeError(f"P1 requires >=30 GB VRAM for full-precision head build; got {gpu_name} ({vram_gb} GB)")

        upload_file(session_id, ROOT / "experiments" / "p1_cases_manifest.json", "/content/p1_cases_manifest.json", config_path=args.config)

        if not args.skip_p1:
            rem_p12 = P1_P2A_CAP_SEC - (time.monotonic() - t_session0)
            exec_file(
                session_id,
                ROOT / "experiments" / "p1_olmo_realizations.py",
                success_marker="P1_SUCCESS_MARKER",
                wall_timeout=min(2400, max(600, int(rem_p12 - 600))),
                config_path=args.config,
            )
            download_and_verify(
                session_id,
                "/content/p1_olmo_realizations.json",
                ROOT / "results" / "p1_olmo_realizations.json",
                ("hardware", "revisions", "head_stats", "parity", "teacher_forcing", "generations"),
                expected_revs={
                    "base": "7df9a82518afdecae4e8c026b27adccc8c1f0032",
                    "sft": "1de02c0175118a9de5854aec80a1f970e701e928",
                    "tokenizer": "470b1fba1ae01581f270116362ee4aa1b97f4c84",
                },
                config_path=args.config,
            )
            restart_kernel(session_id, config_path=args.config)

        if not args.skip_p2a:
            rem_p12 = P1_P2A_CAP_SEC - (time.monotonic() - t_session0)
            if rem_p12 < 480:
                raise RuntimeError(f"Insufficient P1+P2a budget remaining ({rem_p12:.0f}s) for P2a.")
            exec_file(
                session_id,
                ROOT / "experiments" / "p2a_gemma_restoration.py",
                success_marker="P2A_SUCCESS_MARKER",
                wall_timeout=max(420, int(rem_p12)),
                config_path=args.config,
            )
            download_and_verify(
                session_id,
                "/content/p2a_gemma_restoration.json",
                ROOT / "results" / "p2a_gemma_restoration.json",
                ("hardware", "revision", "hook_parity", "loso_discovery_gates", "interventions"),
                expected_revs={"revision": "bf46152c47f5dd20b896357cb51abc4c03b8ee8c"},
                config_path=args.config,
            )
            restart_kernel(session_id, config_path=args.config)

        if not args.skip_p2b:
            remaining = SESSION_CAP_SEC - (time.monotonic() - t_session0)
            if vram_gb >= 36.0 and remaining >= 1200:
                print(f"[colab] VRAM={vram_gb:.1f} GB >= 36 GB and {remaining/60:.1f} min left -> running P2b (4B vs 12B)!", flush=True)
                exec_file(
                    session_id,
                    ROOT / "experiments" / "p2b_gemma_scale.py",
                    success_marker="P2B_SUCCESS_MARKER",
                    wall_timeout=min(1200, int(remaining - 120)),
                    config_path=args.config,
                )
                download_and_verify(
                    session_id,
                    "/content/p2b_gemma_scale.json",
                    ROOT / "results" / "p2b_gemma_scale.json",
                    ("hardware", "gemma_4b", "gemma_12b"),
                    config_path=args.config,
                )
            else:
                print(f"[colab] Skipping P2b (vram_gb={vram_gb:.1f}, remaining={remaining/60:.1f} min).", flush=True)

        print(f"[colab] All eligible GPU runs completed in {(time.monotonic()-t_session0)/60:.1f} min!", flush=True)
    finally:
        if allocated_here and not keep_session:
            release_session(session_id, endpoint=endpoint, config_path=args.config)


if __name__ == "__main__":
    main()
