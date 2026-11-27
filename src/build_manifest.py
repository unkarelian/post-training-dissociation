"""Generate results/manifest.json pinning git commit, scripts, result JSONs, SHA-256 hashes, and revisions."""

from __future__ import annotations

import hashlib
import json
import subprocess
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    git_head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    git_dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip())

    pinned_revisions = {
        "allenai/OLMo-2-1124-7B": "7df9a82518afdecae4e8c026b27adccc8c1f0032",
        "allenai/OLMo-2-1124-7B-SFT": "1de02c0175118a9de5854aec80a1f970e701e928",
        "allenai/OLMo-2-1124-7B-Instruct": "470b1fba1ae01581f270116362ee4aa1b97f4c84",
        "unsloth/gemma-3-4b-it": "bf46152c47f5dd20b896357cb51abc4c03b8ee8c",
        "unsloth/gemma-3-12b-it": "9478e665381f42974aa06177b019352fb6291876",
        "allenai/olmo-2-hard-coded": "7e7685797806d1210bb939cd3ef2eb77d6291682",
        "allenai/tulu-3-sft-olmo-2-mixture": "9b9737bf4fd1ae4f24028728596d396524ddbb29",
    }

    scripts: dict[str, str] = {}
    for folder in ("src", "experiments", "scripts", "tests"):
        d = ROOT / folder
        if not d.exists():
            continue
        for p in sorted(d.glob("*.py")):
            scripts[str(p.relative_to(ROOT))] = sha256_file(p)
        for p in sorted(d.glob("*.json")):
            scripts[str(p.relative_to(ROOT))] = sha256_file(p)

    results: dict[str, dict[str, Any]] = {}
    res_dir = ROOT / "results"
    for p in sorted(res_dir.glob("*.json")):
        if p.name == "manifest.json":
            continue
        results[str(p.relative_to(ROOT))] = {
            "sha256": sha256_file(p),
            "bytes": p.stat().st_size,
        }

    manifest = {
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_head": git_head,
        "git_dirty": git_dirty,
        "pinned_revisions": pinned_revisions,
        "scripts_and_manifests": scripts,
        "results": results,
    }
    out = res_dir / "manifest.json"
    out.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote {out} ({len(scripts)} code/manifest files, {len(results)} result JSONs)")


if __name__ == "__main__":
    main()
