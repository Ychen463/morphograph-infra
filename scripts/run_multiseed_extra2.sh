#!/bin/bash
# Extra 2 seeds for B0, B1a, B2_best (total 5 seeds: 42, 123, 7, 0, 99)
# Run on RunPod after git pull

set -e

SEEDS=(0 99)
DATA="data/raw"

for SEED in "${SEEDS[@]}"; do
    echo "========== Seed ${SEED} =========="

    # B0: mask-only baseline
    python scripts/train_b0.py --data-root $DATA --seed $SEED \
        --output runs/B0_s${SEED} --epochs 100

    # B1a: clDice
    python scripts/train_b1a.py --data-root $DATA --seed $SEED \
        --output runs/B1a_s${SEED} --epochs 100

    # B2_best: skeleton DT (weight=10, MSE, unmasked)
    python scripts/train_b2.py --data-root $DATA --seed $SEED \
        --output runs/B2_best_s${SEED} --epochs 100 \
        --skel-weight 10.0 --skel-loss-type mse --skel-unmask

    echo "========== Seed ${SEED} done =========="
done

echo "All 6 runs (2 seeds × 3 configs) complete."
echo ""
echo "Next: python scripts/aggregate_baseline_results.py --runs-dir runs"
