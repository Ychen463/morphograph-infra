#!/bin/bash
# Extra 2 seeds for DG experiments: 4 methods × 2 seeds = 8 runs
# Complements run_multiseed_dg.sh (seeds 42, 123, 7) to reach 5 seeds total
# Run on RunPod after git pull

set -e

EXTRA_SEEDS=(0 99)
DATA="data/raw"

for SEED in "${EXTRA_SEEDS[@]}"; do
    echo "========== DG Seed ${SEED} =========="

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

    echo "========== DG Seed ${SEED} done =========="
done

echo ""
echo "All 8 runs complete (4 methods × 2 extra seeds)."
echo ""
echo "Next: python scripts/aggregate_dg_results.py --runs-dir runs"
