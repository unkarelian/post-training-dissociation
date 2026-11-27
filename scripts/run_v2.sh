#!/usr/bin/env bash
set -euo pipefail
cd /workspace/foa
pip uninstall -y torchvision torchaudio || true
pip install -q -U transformers accelerate sentencepiece protobuf jinja2 filelock
export HF_HUB_DISABLE_XET=1
export PYTHONPATH="/workspace/foa/src:${PYTHONPATH:-}"
mkdir -p results

for FAM in gemma llama qwen mistral; do
  echo "================ V2 FAMILY: $FAM ================"
  python3 -u src/exp4_v2_verify.py "$FAM" --pack data/pack.json --out-dir results 2>&1 | tee "results/exp4_${FAM}.log"
  rm -rf ~/.cache/huggingface
  df -h / | tail -n 1
done
echo "=== V2_ALL_COMPLETE ==="
