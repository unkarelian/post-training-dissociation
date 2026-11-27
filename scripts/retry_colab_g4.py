"""Acquire one supported Colab GPU and run the synthetic routing experiments."""

import argparse
import fcntl
from itertools import cycle
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import tempfile
import time


ROOT = Path(__file__).resolve().parents[1]
SESSION = "persona-g4-research"
GPU_NAMES = {"G4": ("6000", "Blackwell"), "A100": ("A100",), "L4": ("L4",)}
RESULTS = ("olmo_head_timing.json", "gemma_routing_binding.json")
TRANSIENT = re.compile(
    r"\b(?:503|502|504)\b|service unavailable|bad gateway|gateway timeout|"
    r"read timed out|connect timed out|no backend with gpu available",
    re.IGNORECASE,
)
ACCOUNT_BLOCK = re.compile(
    r"\b(?:401|403|412|429)\b|quota|compute units|permission denied|"
    r"authentication required|too many (?:assignments|requests)|rate.limit|resource_exhausted",
    re.IGNORECASE,
)


def run_colab(*args, timeout=180):
    env = os.environ.copy()
    env.update(COLAB_CLI_ASSIGN_RETRIES="0", COLAB_CLI_REQUEST_TIMEOUT="90")
    return subprocess.run(
        ["colab", *args],
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=timeout,
        check=False,
    )


def acquire(interval, accelerators):
    for gpu in cycle(accelerators):
        started = time.monotonic()
        print(f"Requesting one Colab {gpu} via the official CLI.", flush=True)
        try:
            attempt = run_colab("new", "-s", SESSION, "--gpu", gpu)
        except subprocess.TimeoutExpired:
            # A timed-out request may have been accepted. Reconcile through
            # the CLI before making any further allocation requests.
            status = run_colab("status", "-s", SESSION)
            if status.returncode == 0 and f"[{SESSION}]" in status.stdout:
                return gpu
            raise RuntimeError("Allocation command timed out without a known outcome.") from None

        if attempt.returncode == 0:
            return gpu
        # Match the request error, not traceback source lines or a notebook ID.
        tail = " ".join(attempt.stdout.splitlines()[-3:])
        if ACCOUNT_BLOCK.search(tail):
            raise RuntimeError("Colab reported an account, quota, or rate limit. Stopping allocation.")
        if not TRANSIENT.search(tail):
            raise RuntimeError(f"Colab exited {attempt.returncode} without a retryable capacity error.")
        delay = max(0, interval - (time.monotonic() - started))
        print(f"{gpu} allocation unavailable. Next attempt in {delay:.0f} seconds.", flush=True)
        time.sleep(delay)


def execute(*args):
    subprocess.run(["colab", *args], stdin=subprocess.DEVNULL, check=True, timeout=7200)


def save_result(filename, directory, gpu):
    tmp = directory / filename
    execute("download", "-s", SESSION, f"/content/{filename}", str(tmp))
    with tmp.open() as f:
        data = json.load(f)
    hardware = data.get("hardware", "")
    if not all(part in hardware for part in GPU_NAMES[gpu]):
        raise RuntimeError(f"{filename} reports unexpected hardware: {hardware!r}")
    tmp.replace(ROOT / "results" / filename)
    print(f"Saved results/{filename}.", flush=True)


def run_experiments(directory, gpu):
    notebook = directory / "olmo_head_timing.ipynb"
    shutil.copyfile(ROOT / "experiments" / notebook.name, notebook)
    execute("exec", "-s", SESSION, "--timeout", "1200", "-f", str(notebook))
    save_result(RESULTS[0], directory, gpu)

    # A fresh kernel frees OLMo weights before loading the Gemma checkpoint.
    execute("restart-kernel", "-s", SESSION)
    execute("exec", "-s", SESSION, "--timeout", "1200", "-f",
            str(ROOT / "experiments" / "gemma_routing_binding.py"))
    save_result(RESULTS[1], directory, gpu)


def stop_on_signal(_signum, _frame):
    raise SystemExit(130)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interval", type=int, default=60, help="Minimum seconds between allocation attempts")
    parser.add_argument("--gpu", action="append", choices=tuple(GPU_NAMES),
                        help="Accelerator to try. Repeat the flag to rotate in order.")
    args = parser.parse_args()
    if args.interval <= 0:
        parser.error("--interval must be positive")
    for name in ("olmo_head_timing.ipynb", "gemma_routing_binding.py"):
        if not (ROOT / "experiments" / name).is_file():
            parser.error(f"Missing experiments/{name}")
    if not shutil.which("colab"):
        parser.error("Install the official google-colab-cli first.")

    state_dir = Path(os.environ.get("XDG_STATE_HOME", Path.home() / ".local" / "state")) / "persona-g4"
    state_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    signal.signal(signal.SIGTERM, stop_on_signal)
    with (state_dir / "retry.lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error("A Colab retry job is already active.")
        status = run_colab("status", "-s", SESSION)
        if status.returncode == 0 and f"[{SESSION}]" in status.stdout:
            raise RuntimeError(f"A Colab runtime named {SESSION!r} already exists.")
        try:
            gpu = acquire(args.interval, args.gpu or ["G4"])
            print(f"{gpu} assigned. Starting the notebook.", flush=True)
            with tempfile.TemporaryDirectory(prefix=".persona-g4-", dir=ROOT / "results") as tmp:
                run_experiments(Path(tmp), gpu)
            print("Both GPU experiments completed.", flush=True)
        finally:
            status = run_colab("status", "-s", SESSION)
            if status.returncode == 0 and f"[{SESSION}]" in status.stdout:
                stopped = run_colab("stop", "-s", SESSION)
                if stopped.returncode:
                    raise RuntimeError(f"Could not release Colab runtime {SESSION!r}.")
                print("Released the Colab runtime.", flush=True)


if __name__ == "__main__":
    main()
