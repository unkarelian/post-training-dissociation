#!/usr/bin/env bash
set -euo pipefail
cd /workspace/foa
pip uninstall -y torchvision torchaudio || true
pip install -q -U transformers accelerate peft datasets sentencepiece protobuf jinja2 filelock
export HF_HUB_DISABLE_XET=1
export PYTHONPATH="/workspace/foa/src:${PYTHONPATH:-}"
mkdir -p results

python3 -u src/exp7_sft_ablation.py --pack data/pack.json --out-dir results --n-per-arm 2000 2>&1 | tee results/exp7_sft_ablation.log
echo "=== V4_ABLATION_COMPLETE ==="
