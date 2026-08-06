#!/bin/bash
# Multi-seed B1a (clDice) experiments: 3 seeds
# Run on RunPod after git pull

set -e

SEEDS=(42 123 7)
DATA="data/raw"

for SEED in "${SEEDS[@]}"; do
    echo "========== B1a Seed ${SEED} =========="

    python scripts/train_b1a.py --data-root $DATA --seed $SEED \
        --output runs/B1a_s${SEED} --epochs 100

    echo "========== B1a Seed ${SEED} done =========="
done

echo "All 3 B1a runs complete."
echo ""
echo "Next: python scripts/aggregate_baseline_results.py --runs-dir runs"
