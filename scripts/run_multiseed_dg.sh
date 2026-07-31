#!/bin/bash
# Multi-seed DG experiments: 4 methods × 3 seeds = 12 runs
# Run on RunPod after git pull

set -e

SEEDS=(42 123 7)
DATA="data/raw"

for SEED in "${SEEDS[@]}"; do
    echo "========== Seed ${SEED} =========="

    # D1: ERM baseline
    python scripts/train_dg.py --data-root $DATA --seed $SEED \
        --output runs/D1_erm_s${SEED}

    # D2c: MixStyle (best config: alpha=0.1)
    python scripts/train_dg.py --data-root $DATA --seed $SEED \
        --output runs/D2c_mixstyle_s${SEED} \
        --mixstyle --mixstyle-alpha 0.1

    # D2a: CORAL
    python scripts/train_dg.py --data-root $DATA --seed $SEED \
        --output runs/D2a_coral_s${SEED} \
        --coral --coral-weight 1.0

    # D2b: DANN
    python scripts/train_dg.py --data-root $DATA --seed $SEED \
        --output runs/D2b_dann_s${SEED} \
        --dann --dann-weight 0.1

    echo "========== Seed ${SEED} done =========="
done

echo "All 12 runs complete."
