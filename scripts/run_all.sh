#!/usr/bin/env bash
# Full replication runner for "The Face of Another" experiments (Exp 1, 2, 3)
# Usage:
#   bash scripts/run_all.sh                 # keeps HuggingFace cache
#   bash scripts/run_all.sh --clear-cache   # clears HF cache between families (for <=70GB disks)
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
mkdir -p results

CLEAR_CACHE="${1:-}"
export HF_HUB_DISABLE_XET=1
export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"

run_family() {
  local FAM=$1
  local BASE=$2
  local IT=$3
  echo "================ FAMILY: $FAM ================"
  python3 -u src/exp1_clothes.py "$BASE" "$IT" --out-dir results
  python3 -u src/exp2_inversion.py "$BASE" "$IT" --pack data/pack.json --out-dir results
  python3 -u src/exp3_graft.py "$FAM" --pack data/pack.json --out-dir results
  if [ "$CLEAR_CACHE" = "--clear-cache" ]; then
    echo "=== Clearing HuggingFace cache after $FAM ==="
    rm -rf ~/.cache/huggingface
  fi
}

run_family gemma gemma_base gemma_it
run_family llama llama_base llama_it
run_family qwen qwen_base qwen_it

python3 src/analyze.py --results-dir results | tee results/analysis_summary.txt
echo "=== ALL_EXPERIMENTS_COMPLETE ==="
