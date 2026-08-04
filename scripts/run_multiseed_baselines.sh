#!/bin/bash
# Multi-seed baseline experiments: B0 + B2_best × 3 seeds = 6 runs
# Run on RunPod after git pull
#
# B2_best config: weight=10, MSE, unmasked, start=0, ramp=0
# (from B2_dt_v4_w10, mIoU=0.683 on seed=42)

set -e

SEEDS=(42 123 7)
DATA="data/raw"

for SEED in "${SEEDS[@]}"; do
    echo "========== Seed ${SEED} =========="

    # B0: mask-only baseline
    python scripts/train_b0.py --data-root $DATA --seed $SEED \
        --output runs/B0_s${SEED} --epochs 100

    # B2_best: skeleton DT (weight=10, MSE, unmasked)
    python scripts/train_b2.py --data-root $DATA --seed $SEED \
        --output runs/B2_best_s${SEED} --epochs 100 \
        --skel-weight 10.0 --skel-loss-type mse --skel-unmask

    echo "========== Seed ${SEED} done =========="
done

echo "All 6 runs complete."
echo ""
echo "Next: python scripts/aggregate_baseline_results.py --runs-dir runs"
