#!/bin/bash
# B2 v4 unmasked fine weight sweep around w=10 peak
# Usage: bash scripts/run_b2_fine_sweep.sh
set -e

for W in 9 11 13; do
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
for W in 9 11 13; do
  echo "--- w=${W} ---"
  grep -E "best_miou_fg|delta_vs_b0" "runs/B2_dt_v4_w${W}/summary.json"
done
