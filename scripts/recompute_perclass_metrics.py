"""Recompute per-class metrics from existing best.pt checkpoints.

Uses the corrected global-confusion-matrix IoU (matching compute_miou used
during training) instead of the previous per-image-averaged IoU.  Then
rebuilds baseline_multiseed_summary.json with the corrected perclass fields
and paired t-test p-values for each class metric.

Usage:
    python scripts/recompute_perclass_metrics.py --data-root data/raw
    python scripts/recompute_perclass_metrics.py --data-root data/raw --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy import stats
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from morphograph.training.utils import (
    DamSegmentDataset, discover_all_samples, eval_perclass_metrics, split_data,
)

BASELINES = ["B0", "B1a", "B2_best"]
SEEDS = [0, 7, 42, 99, 123]
VAL_RATIO = 0.15
BATCH_SIZE = 4
NUM_WORKERS = 4
PERCLASS_METRICS = ["iou_crack", "iou_spalling", "bf1_crack"]


def _paired_ttest(a: list[float], b: list[float]) -> tuple[float, float, float]:
    """Return (mean_delta, std_delta, p_value) for paired test b vs a."""
    diffs = [bi - ai for ai, bi in zip(a, b)]
    mean_d = float(np.mean(diffs))
    std_d = float(np.std(diffs))
    _, p = stats.ttest_rel(b, a)
    return mean_d, std_d, float(p)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    parser.add_argument("--baselines", nargs="+", default=BASELINES)
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    all_pairs = discover_all_samples(args.data_root)
    if not all_pairs:
        print("ERROR: No data found.")
        sys.exit(1)

    # ── Step 1: eval each checkpoint ───────────────────────────────────────
    recomputed: dict[str, dict[str, dict]] = {}

    for baseline in args.baselines:
        recomputed[baseline] = {}
        for seed in args.seeds:
            run_dir = args.runs_dir / f"{baseline}_s{seed}"
            best_pt = run_dir / "best.pt"

            if not best_pt.exists():
                print(f"  [SKIP] {baseline}_s{seed}: no best.pt")
                continue

            if args.dry_run:
                print(f"  [DRY] would eval {best_pt}")
                continue

            print(f"  Evaluating {baseline}_s{seed} ...", flush=True)
            _, val_pairs = split_data(all_pairs, VAL_RATIO, seed)
            val_loader = DataLoader(
                DamSegmentDataset(val_pairs, augment=False),
                batch_size=BATCH_SIZE, shuffle=False,
                num_workers=NUM_WORKERS, pin_memory=True,
            )

            pc = eval_perclass_metrics(best_pt, val_loader, device)
            recomputed[baseline][str(seed)] = pc
            print(f"    iou_crack={pc['iou_crack']:.4f}  "
                  f"iou_spalling={pc['iou_spalling']:.4f}  "
                  f"miou_fg={pc['miou_fg']:.4f}  "
                  f"bf1_crack={pc['bf1_crack']:.4f}")

            # Patch individual summary.json
            summary_path = run_dir / "summary.json"
            if summary_path.exists():
                s = json.loads(summary_path.read_text())
                s["perclass"] = pc
                summary_path.write_text(json.dumps(s, indent=2))

    if args.dry_run:
        return

    # ── Step 2: rebuild baseline_multiseed_summary.json ────────────────────
    summary_path = args.runs_dir / "baseline_multiseed_summary.json"
    if not summary_path.exists():
        print(f"\nERROR: {summary_path} not found; run aggregate_baseline_results.py first")
        sys.exit(1)

    summary = json.loads(summary_path.read_text())

    for baseline in args.baselines:
        if baseline not in recomputed or not recomputed[baseline]:
            continue

        seeds_done = sorted(recomputed[baseline].keys(), key=int)
        pc_by_seed = recomputed[baseline]

        # per_seed_perclass
        summary[baseline]["per_seed_perclass"] = {s: pc_by_seed[s] for s in seeds_done}

        # per-metric mean/std (population ddof=0)
        for m in PERCLASS_METRICS:
            vals = [pc_by_seed[s][m] for s in seeds_done if m in pc_by_seed[s]]
            if vals:
                summary[baseline][f"{m}_mean"] = float(np.mean(vals))
                summary[baseline][f"{m}_std"]  = float(np.std(vals))

    # ── Step 3: perclass pairwise tests vs B0 ──────────────────────────────
    if "B0" not in recomputed or not recomputed["B0"]:
        print("\nWARNING: B0 not recomputed; skipping pairwise perclass tests")
    else:
        b0 = recomputed["B0"]
        if "pairwise_tests" not in summary:
            summary["pairwise_tests"] = {}

        for other in args.baselines:
            if other == "B0" or other not in recomputed or not recomputed[other]:
                continue
            other_pc = recomputed[other]
            common = sorted(set(b0) & set(other_pc), key=int)
            if len(common) < 3:
                continue

            pt = summary["pairwise_tests"].setdefault(other, {})
            for m in PERCLASS_METRICS:
                b0_vals   = [b0[s][m]      for s in common]
                other_vals = [other_pc[s][m] for s in common]
                mean_d, std_d, p = _paired_ttest(b0_vals, other_vals)
                pt[f"{m}_mean_delta"] = mean_d
                pt[f"{m}_std_delta"]  = std_d
                pt[f"{m}_p_value"]    = p
                print(f"  {other} vs B0 {m}: Δ={mean_d:+.4f} p={p:.4f}")

    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\nUpdated {summary_path}")

    # Save raw recomputed data separately for auditing
    raw_out = args.runs_dir / "perclass_recomputed.json"
    raw_out.write_text(json.dumps(recomputed, indent=2))
    print(f"Raw recomputed perclass data saved to {raw_out}")


if __name__ == "__main__":
    main()
