#!/bin/bash
# B2 v4_w10 multi-seed validation (MSE, unmasked, w=10)
# seed=42 already done (runs/B2_dt_v4_w10)
# Usage: bash scripts/run_b2_multiseed.sh
set -e

for SEED in 123 456; do
  echo "============================================"
  echo "Starting B2_dt_v4_w10_s${SEED}"
  echo "============================================"
  python scripts/train_b2.py \
    --data-root data/raw \
    --output "runs/B2_dt_v4_w10_s${SEED}" \
    --skel-weight 10.0 \
    --skel-loss-type mse \
    --skel-unmask \
    --seed "${SEED}"
  echo "Finished seed=${SEED}"
  echo ""
done

echo "All seeds complete. Results:"
echo "--- seed=42 (existing) ---"
grep -E "best_miou_fg|delta_vs_b0" runs/B2_dt_v4_w10/summary.json
for SEED in 123 456; do
  echo "--- seed=${SEED} ---"
  grep -E "best_miou_fg|delta_vs_b0" "runs/B2_dt_v4_w10_s${SEED}/summary.json"
done
