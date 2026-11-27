#!/usr/bin/env bash
# Run Phase 1 (OLMo-2 lm_head decomposition) and Phase 2 (Gemma referent probe gate + conditional patching).
# Expects PyTorch 2.6+ and Transformers 4.51+ on a 48GB+ GPU (e.g., A40 / L40S / H100).
# Set NANOGPT_API_KEY (or pass --score-jev-only locally afterwards) for Jev classification.
set -euo pipefail

cd "$(dirname "$0")/.."

echo "=== CPU preflight unit tests (including tensor ridge/gate verification) ==="
python3 src/test_exp10.py

echo "=== Phase 1: OLMo-2 lm_head 2x2 teacher forcing + ΔW weight surgery ==="
python3 src/exp10_olmo_head.py \
  --out results/exp10_olmo_head.json \
  --batch-size 32 \
  --include-exp9 \
  ${EXP10_SCORE_JEV:+--score-jev}

echo "=== Phase 2: Gemma-3-4B order-invariant referent probe gate + behavioral + conditional patching ==="
python3 src/exp10_gemma_referent.py \
  --out results/exp10_gemma_referent.json \
  --batch-size 24 \
  ${EXP10_SCORE_JEV:+--score-jev}

echo "=== Done! Outputs saved to results/exp10_olmo_head.json and results/exp10_gemma_referent.json ==="
