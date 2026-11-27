"""Hardened Colab CLI runner for P4 (Gemma paradox, ~3 min) -> P3 (5-model lm_head generalization, ~15-25 min).

Reuses hardened allocation, VRAM probing, explicit success-marker verification, atomic local
downloads, and unconditional verified session release from scripts/run_colab_plan.py.
Supports G4, A100, and L4 (both P4 and chunked-Gram P3 fit in <= 22 GB VRAM).
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from run_colab_plan import (  # noqa: E402
    acquire_session,
    download_and_verify,
    exec_file,
    hash_bundle,
    probe_remote_gpu,
    release_session,
    restart_kernel,
    run_colab,
    upload_file,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", type=int, default=45)
    ap.add_argument("--config", type=str, default=None)
    ap.add_argument("--session", type=str, default=None)
    ap.add_argument("--keep-session", action="store_true")
    ap.add_argument("--skip-p4", action="store_true")
    ap.add_argument("--skip-p3", action="store_true")
    args = ap.parse_args()

    assert shutil.which("colab"), "colab CLI not found."
    manifest_path = ROOT / "experiments" / "p1_cases_manifest.json"
    if not manifest_path.exists():
        import subprocess
        subprocess.check_call([sys.executable, str(ROOT / "experiments" / "p1_cases_manifest.py")])

    bundle_files = [
        manifest_path,
        ROOT / "experiments" / "p4_gemma_paradox.py",
        ROOT / "experiments" / "p3_head_generalization.py",
    ]
    print("[colab-p3p4] Bundle SHA-256:", json.dumps(hash_bundle(bundle_files), indent=2), flush=True)

    reusing = args.session is not None
    session_id = args.session or f"p3p4-{uuid.uuid4().hex[:10]}"
    keep_session = args.keep_session or reusing
    gpu_schedule = [("G4", 2), ("A100", 2), ("L4", 2)]

    t0 = time.monotonic()
    allocated_here = False
    endpoint = ""
    try:
        if not reusing:
            _, endpoint = acquire_session(session_id, gpu_schedule, args.interval, config_path=args.config)
            allocated_here = True
        else:
            print(f"[colab-p3p4] Reusing session {session_id} (keep={keep_session})", flush=True)
            run_colab("restart-kernel", "-s", session_id, config_path=args.config, timeout=90)
            time.sleep(6)

        gpu_name, vram_gb = probe_remote_gpu(session_id, config_path=args.config)
        print(f"[colab-p3p4] Remote GPU: {gpu_name} ({vram_gb:.2f} GB VRAM)", flush=True)
        if vram_gb < 20.0:
            raise RuntimeError(f"Need >=20 GB VRAM (L4/A100/G4); got {gpu_name} ({vram_gb:.1f} GB)")

        # Pre-install transformers==4.51.3 cleanly once and restart kernel BEFORE loading models
        import tempfile
        setup_code = (
            "import importlib.metadata, subprocess, sys\n"
            "try:\n"
            "    v = importlib.metadata.version('transformers')\n"
            "except Exception:\n"
            "    v = None\n"
            "print('transformers_before:', v, flush=True)\n"
            "if v != '4.51.3':\n"
            "    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'transformers==4.51.3'])\n"
            "print('SETUP_OK_MARKER', flush=True)\n"
        )
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as tf:
            tf.write(setup_code)
            setup_path = Path(tf.name)
        try:
            exec_file(session_id, setup_path, success_marker="SETUP_OK_MARKER", wall_timeout=300, idle_timeout=180, config_path=args.config)
        finally:
            setup_path.unlink(missing_ok=True)
        restart_kernel(session_id, config_path=args.config)

        upload_file(session_id, manifest_path, "/content/p1_cases_manifest.json", config_path=args.config)

        if not args.skip_p4:
            exec_file(
                session_id,
                ROOT / "experiments" / "p4_gemma_paradox.py",
                success_marker="P4_SUCCESS_MARKER",
                wall_timeout=1200,
                idle_timeout=600,
                config_path=args.config,
            )
            download_and_verify(
                session_id,
                "/content/p4_gemma_paradox.json",
                ROOT / "results" / "p4_gemma_paradox.json",
                ("hardware", "revision", "hook_parity", "layer_sweep_2x2", "subspace_decomp", "open_kv_patching"),
                expected_revs={"revision": "bf46152c47f5dd20b896357cb51abc4c03b8ee8c"},
                config_path=args.config,
            )
            restart_kernel(session_id, config_path=args.config)

        if not args.skip_p3:
            exec_file(
                session_id,
                ROOT / "experiments" / "p3_head_generalization.py",
                success_marker="P3_SUCCESS_MARKER",
                wall_timeout=4200,
                idle_timeout=900,
                config_path=args.config,
            )
            download_and_verify(
                session_id,
                "/content/p3_head_generalization.json",
                ROOT / "results" / "p3_head_generalization.json",
                ("hardware", "manifest_sha256", "models"),
                config_path=args.config,
            )

        print(f"[colab-p3p4] Completed in {(time.monotonic()-t0)/60:.1f} min!", flush=True)
    finally:
        if allocated_here and not keep_session:
            release_session(session_id, endpoint=endpoint, config_path=args.config)


if __name__ == "__main__":
    main()
