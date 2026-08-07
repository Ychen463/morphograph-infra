#!/bin/bash
# Extra 2 seeds for B0, B1a, B2_best (total 5 seeds: 42, 123, 7, 0, 99)
# + B3 and B5 × 5 seeds (previously only seed=42)
# Run on RunPod after git pull

set -e

DATA="data/raw"

# --- Part 1: Extra 2 seeds for B0, B1a, B2_best ---
EXTRA_SEEDS=(0 99)

for SEED in "${EXTRA_SEEDS[@]}"; do
    echo "========== B0/B1a/B2 Seed ${SEED} =========="

    python scripts/train_b0.py --data-root $DATA --seed $SEED \
        --output runs/B0_s${SEED} --epochs 100

    python scripts/train_b1a.py --data-root $DATA --seed $SEED \
        --output runs/B1a_s${SEED} --epochs 100

    python scripts/train_b2.py --data-root $DATA --seed $SEED \
        --output runs/B2_best_s${SEED} --epochs 100 \
        --skel-weight 10.0 --skel-loss-type mse --skel-unmask

    echo "========== B0/B1a/B2 Seed ${SEED} done =========="
done

# --- Part 2: B3 and B5 × 5 seeds (all seeds) ---
ALL_SEEDS=(42 123 7 0 99)

for SEED in "${ALL_SEEDS[@]}"; do
    echo "========== B3/B5 Seed ${SEED} =========="

    python scripts/train_b345.py --baseline B3 --data-root $DATA --seed $SEED \
        --output runs/B3_s${SEED} --epochs 100

    python scripts/train_b345.py --baseline B5 --data-root $DATA --seed $SEED \
        --output runs/B5_s${SEED} --epochs 100

    echo "========== B3/B5 Seed ${SEED} done =========="
done

echo ""
echo "All runs complete: B0/B1a/B2 × 2 extra seeds + B3/B5 × 5 seeds = 16 runs"
echo ""
echo "Next: python scripts/aggregate_baseline_results.py --runs-dir runs"
