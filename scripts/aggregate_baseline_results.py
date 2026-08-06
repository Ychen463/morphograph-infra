"""Aggregate multi-seed B0/B2 baseline results.

Usage:
    python scripts/aggregate_baseline_results.py --runs-dir runs
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    # Pattern: B0_s{seed} or B2_best_s{seed}
    pattern = re.compile(r"^(B\w+?)_s(\d+)$")
    results_by_method = defaultdict(list)

    for run_dir in sorted(args.runs_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        m = pattern.match(run_dir.name)
        if not m:
            continue
        summary_path = run_dir / "summary.json"
        if not summary_path.exists():
            continue
        method = m.group(1)
        seed = int(m.group(2))
        with open(summary_path) as f:
            data = json.load(f)
        data["seed"] = seed
        results_by_method[method].append(data)

    if not results_by_method:
        print("No multi-seed baseline runs found. Expected: B0_s42, B2_best_s42, etc.")
        sys.exit(1)

    print(f"\n{'='*70}")
    print("Multi-Seed Baseline Results (mean +/- std)")
    print(f"{'='*70}\n")

    header = f"{'Method':<15} {'n':>2}  {'mIoU_fg':>20}  {'seeds'}"
    print(header)
    print("-" * len(header))

    summary = {}
    for method in sorted(results_by_method.keys()):
        runs = results_by_method[method]
        n = len(runs)
        vals = [r["best_miou_fg"] for r in runs]
        seeds = sorted([r["seed"] for r in runs])
        mean_val = np.mean(vals)
        std_val = np.std(vals)
        print(f"{method:<15} {n:>2}  {mean_val:.4f} +/- {std_val:.4f}  {seeds}")

        summary[method] = {
            "n_seeds": n,
            "best_miou_fg_mean": float(mean_val),
            "best_miou_fg_std": float(std_val),
            "per_seed": {str(r["seed"]): r["best_miou_fg"] for r in runs},
        }

    # Pairwise tests: each method vs B0
    b0_key = "B0"
    if b0_key in results_by_method and len(results_by_method[b0_key]) >= 3:
        b0_by_seed = {r["seed"]: r["best_miou_fg"] for r in results_by_method[b0_key]}
        summary["pairwise_tests"] = {}

        for other_key in sorted(results_by_method.keys()):
            if other_key == b0_key or len(results_by_method[other_key]) < 3:
                continue
            other_by_seed = {r["seed"]: r["best_miou_fg"] for r in results_by_method[other_key]}
            common_seeds = sorted(set(b0_by_seed) & set(other_by_seed))
            if len(common_seeds) < 3:
                continue

            b0_vals = [b0_by_seed[s] for s in common_seeds]
            other_vals = [other_by_seed[s] for s in common_seeds]
            diffs = [o - b for b, o in zip(b0_vals, other_vals)]

            # Paired t-test
            t_stat, p_val = stats.ttest_rel(other_vals, b0_vals)
            sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"

            print(f"\n{'='*70}")
            print(f"Paired t-test: {other_key} vs B0 (n={len(common_seeds)} seeds)")
            print(f"{'='*70}\n")
            for s, d in zip(common_seeds, diffs):
                print(f"  Seed {s}: B0={b0_by_seed[s]:.4f}  {other_key}={other_by_seed[s]:.4f}  delta={d:+.4f}")
            print(f"\n  Mean delta: {np.mean(diffs):+.4f} +/- {np.std(diffs):.4f}")
            print(f"  t={t_stat:.3f}, p={p_val:.4f} {sig}")

            # TOST equivalence test (margin = 0.01 mIoU = 1 percentage point)
            margin = 0.01
            mean_diff = np.mean(diffs)
            se_diff = np.std(diffs, ddof=1) / np.sqrt(len(diffs))
            df = len(diffs) - 1
            t_upper = (mean_diff - margin) / se_diff
            t_lower = (mean_diff + margin) / se_diff
            p_upper = stats.t.cdf(t_upper, df)       # H0: diff >= margin
            p_lower = 1.0 - stats.t.cdf(t_lower, df)  # H0: diff <= -margin
            p_tost = max(p_upper, p_lower)
            tost_sig = "EQUIVALENT" if p_tost < 0.05 else "NOT equivalent"

            print(f"\n  TOST equivalence test (margin=+/-{margin}):")
            print(f"    p_upper={p_upper:.4f}, p_lower={p_lower:.4f}")
            print(f"    p_TOST={p_tost:.4f} → {tost_sig}")

            summary["pairwise_tests"][other_key] = {
                "comparison": f"{other_key} vs B0",
                "n_pairs": len(common_seeds),
                "mean_delta": float(np.mean(diffs)),
                "std_delta": float(np.std(diffs)),
                "t_stat": float(t_stat),
                "p_value": float(p_val),
                "significant": bool(p_val < 0.05),
                "tost_margin": margin,
                "tost_p": float(p_tost),
                "tost_equivalent": bool(p_tost < 0.05),
            }

    output_path = args.output or args.runs_dir / "baseline_multiseed_summary.json"
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
