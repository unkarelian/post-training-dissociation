#!/usr/bin/env bash
set -euo pipefail
cd /workspace/foa
pip uninstall -y torchvision torchaudio || true
pip install -q -U transformers accelerate sentencepiece protobuf jinja2 filelock
export HF_HUB_DISABLE_XET=1
export PYTHONPATH="/workspace/foa/src:${PYTHONPATH:-}"
mkdir -p results

echo "================ EXP 5: OLMo-2-1124-7B STAGES + GRAFTS ================"
python3 -u src/exp5_olmo_stages.py --pack data/pack.json --out-dir results --clear-mid 2>&1 | tee results/exp5_olmo.log
rm -rf ~/.cache/huggingface
df -h / | tail -n 1

echo "================ EXP 6: ROLEPLAY GRAFT STRESS-TEST (LLAMA) ================"
python3 -u src/exp6_roleplay_graft.py llama --pack data/pack.json --out-dir results 2>&1 | tee results/exp6_llama.log
rm -rf ~/.cache/huggingface
df -h / | tail -n 1

echo "================ EXP 6: ROLEPLAY GRAFT STRESS-TEST (MISTRAL) ================"
python3 -u src/exp6_roleplay_graft.py mistral --pack data/pack.json --out-dir results 2>&1 | tee results/exp6_mistral.log
rm -rf ~/.cache/huggingface
df -h / | tail -n 1

echo "=== V3_ALL_COMPLETE ==="
