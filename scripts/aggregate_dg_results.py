"""Aggregate multi-seed DG results into a summary table.

Usage:
    python scripts/aggregate_dg_results.py --runs-dir runs
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
    parser = argparse.ArgumentParser(description="Aggregate multi-seed DG results")
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    # Discover runs: pattern D{id}_{method}_s{seed}
    pattern = re.compile(r"^(D\w+?)_s(\d+)$")
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
        data["run_name"] = run_dir.name
        results_by_method[method].append(data)

    if not results_by_method:
        print("No multi-seed runs found. Expected pattern: D{id}_{method}_s{seed}/summary.json")
        sys.exit(1)

    # Aggregate
    metrics = ["final_damseg_miou_fg", "final_s2ds_miou_fg", "domain_gap"]
    class_metrics = {
        "s2ds_crack": ("final_s2ds_per_class", "1"),
        "s2ds_spalling": ("final_s2ds_per_class", "2"),
    }

    print(f"\n{'='*80}")
    print("Multi-Seed DG Results (mean +/- std)")
    print(f"{'='*80}\n")

    table_data = []
    for method in sorted(results_by_method.keys()):
        runs = results_by_method[method]
        n = len(runs)
        dg_method = runs[0].get("dg_method", "?")

        row = {"method": method, "dg": dg_method, "n": n}
        for metric in metrics:
            vals = [r[metric] for r in runs if r.get(metric) is not None]
            if vals:
                row[metric] = (np.mean(vals), np.std(vals))
        for name, (dict_key, class_key) in class_metrics.items():
            vals = [r[dict_key][class_key] for r in runs if r.get(dict_key) and class_key in r[dict_key]]
            if vals:
                row[name] = (np.mean(vals), np.std(vals))

        table_data.append(row)

    # Print table
    header = f"{'Method':<20} {'DG':<12} {'n':>2}  {'DamSeg mIoU':>16}  {'s2ds mIoU':>16}  {'Gap':>16}  {'s2ds crack':>16}  {'s2ds spall':>16}"
    print(header)
    print("-" * len(header))

    for row in table_data:
        def fmt(key):
            if key in row:
                m, s = row[key]
                return f"{m:.4f}+/-{s:.4f}"
            return "N/A"
        print(f"{row['method']:<20} {row['dg']:<12} {row['n']:>2}  "
              f"{fmt('final_damseg_miou_fg'):>16}  {fmt('final_s2ds_miou_fg'):>16}  "
              f"{fmt('domain_gap'):>16}  {fmt('s2ds_crack'):>16}  {fmt('s2ds_spalling'):>16}")

    # Pairwise significance tests (MixStyle vs ERM)
    erm_key = None
    mix_key = None
    for method in results_by_method:
        if "erm" in method.lower():
            erm_key = method
        if "mixstyle" in method.lower():
            mix_key = method

    if erm_key and mix_key and len(results_by_method[erm_key]) >= 3 and len(results_by_method[mix_key]) >= 3:
        print(f"\n{'='*80}")
        print(f"Significance Test: {mix_key} vs {erm_key}")
        print(f"{'='*80}\n")

        for metric in ["final_s2ds_miou_fg", "domain_gap"]:
            erm_vals = [r[metric] for r in results_by_method[erm_key] if r.get(metric) is not None]
            mix_vals = [r[metric] for r in results_by_method[mix_key] if r.get(metric) is not None]
            if len(erm_vals) >= 3 and len(mix_vals) >= 3:
                t_stat, p_val = stats.ttest_ind(mix_vals, erm_vals)
                sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
                print(f"  {metric}: t={t_stat:.3f}, p={p_val:.4f} {sig}")
                print(f"    ERM: {np.mean(erm_vals):.4f}+/-{np.std(erm_vals):.4f}")
                print(f"    Mix: {np.mean(mix_vals):.4f}+/-{np.std(mix_vals):.4f}")

    # Save
    output_path = args.output or args.runs_dir / "dg_multiseed_summary.json"
    summary = {}
    for method, runs in results_by_method.items():
        method_summary = {"n_seeds": len(runs), "dg_method": runs[0].get("dg_method")}
        for metric in metrics:
            vals = [r[metric] for r in runs if r.get(metric) is not None]
            if vals:
                method_summary[f"{metric}_mean"] = float(np.mean(vals))
                method_summary[f"{metric}_std"] = float(np.std(vals))
        for name, (dict_key, class_key) in class_metrics.items():
            vals = [r[dict_key][class_key] for r in runs if r.get(dict_key) and class_key in r[dict_key]]
            if vals:
                method_summary[f"{name}_mean"] = float(np.mean(vals))
                method_summary[f"{name}_std"] = float(np.std(vals))
        summary[method] = method_summary

    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
