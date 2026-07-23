#!/bin/bash
# B2 DT regression weight sweep — v4 unmasked MSE
# Usage: bash scripts/run_b2_sweep.sh
set -e

for W in 8 15 20; do
  echo "============================================"
  echo "Starting B2_dt_v4_w${W} (MSE, unmasked, weight=${W})"
  echo "============================================"
  python scripts/train_b2.py \
    --data-root data/raw \
    --output "runs/B2_dt_v4_w${W}" \
    --skel-weight "${W}" \
    --skel-loss-type mse \
    --skel-unmask
  echo "Finished B2_dt_v4_w${W}"
  echo ""
done

echo "All runs complete. Results:"
for W in 8 15 20; do
  echo "--- w=${W} ---"
  cat "runs/B2_dt_v4_w${W}/summary.json" | grep -E "best_miou_fg|delta_vs_b0|weight"
done
