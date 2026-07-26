#!/bin/bash
# Run B3 and B5 baseline ladder (B4 skipped — edge loss conflicts with DT)
# Usage: bash scripts/run_b345.sh
set -e

echo "============================================"
echo "B3: B2_best + endpoint/junction heads"
echo "============================================"
python scripts/train_b345.py \
  --baseline B3 \
  --data-root data/raw \
  --output runs/B3
echo ""

echo "============================================"
echo "B5: B3 + width regression"
echo "============================================"
python scripts/train_b345.py \
  --baseline B5 \
  --data-root data/raw \
  --output runs/B5
echo ""

echo "All done. Results:"
for B in B3 B5; do
  echo "--- ${B} ---"
  grep -E "best_miou_fg|delta_vs_b0|delta_vs_b2" "runs/${B}/summary.json"
done
