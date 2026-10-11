#!/usr/bin/env bash
# Re-run B3 and B5 (5 seeds each) paired with the current B0 re-run,
# then rebuild baseline_multiseed_summary.json with correct pairwise stats,
# then restore perclass fields for B0/B1a/B2_best.
#
# Usage (from repo root on RunPod):
#   bash scripts/rerun_b3b5_and_aggregate.sh
#   bash scripts/rerun_b3b5_and_aggregate.sh --data-root /path/to/data/raw
#
# After the script completes, pull runs/ back to local and the summary JSON
# will have consistent B3/B5 pairwise stats vs the current B0.

set -euo pipefail

DATA_ROOT="${1:-data/raw}"
if [[ "$1" == "--data-root" ]]; then
    DATA_ROOT="$2"
fi

echo "============================================================"
echo " Step 1/3: Train B3 and B5 (5 seeds each = 10 runs)"
echo " Data root: $DATA_ROOT"
echo "============================================================"
python scripts/run_baseline_multiseed.py \
    --data-root "$DATA_ROOT" \
    --baselines B3 B5 \
    --seeds 0 7 42 99 123

echo ""
echo "============================================================"
echo " Step 2/3: Rebuild baseline_multiseed_summary.json"
echo "           (pairwise tests vs current B0 for ALL methods)"
echo "============================================================"
python scripts/aggregate_baseline_results.py --runs-dir runs

echo ""
echo "============================================================"
echo " Step 3/3: Restore perclass metrics for B0 / B1a / B2_best"
echo "           (re-evaluates existing best.pt checkpoints)"
echo "============================================================"
python scripts/recompute_perclass_metrics.py \
    --data-root "$DATA_ROOT" \
    --baselines B0 B1a B2_best

echo ""
echo "All done. Pull runs/baseline_multiseed_summary.json back to local."
