#!/bin/bash
# B2 Wave 2: schedule + deep head on v4_w10 base (MSE, unmasked, w=10)
# Usage: bash scripts/run_b2_wave2.sh
set -e

# v5: delayed start (let encoder learn seg for 20 epochs, then ramp skel over 10)
echo "============================================"
echo "Starting B2_dt_v5 (schedule: start=20, ramp=10)"
echo "============================================"
python scripts/train_b2.py \
  --data-root data/raw \
  --output runs/B2_dt_v5 \
  --skel-weight 10.0 \
  --skel-loss-type mse \
  --skel-unmask \
  --skel-start-epoch 20 \
  --skel-ramp-epochs 10
echo "Finished B2_dt_v5"
echo ""

# v6: deeper skeleton head (256->128->64->1, ~450K params)
echo "============================================"
echo "Starting B2_dt_v6 (deep head)"
echo "============================================"
python scripts/train_b2.py \
  --data-root data/raw \
  --output runs/B2_dt_v6 \
  --skel-weight 10.0 \
  --skel-loss-type mse \
  --skel-unmask \
  --skel-head-deep
echo "Finished B2_dt_v6"
echo ""

echo "All runs complete. Results:"
for V in v5 v6; do
  echo "--- ${V} ---"
  grep -E "best_miou_fg|delta_vs_b0|weight|head_deep|start_epoch|ramp_epochs" "runs/B2_dt_${V}/summary.json"
done
