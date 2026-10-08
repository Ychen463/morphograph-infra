"""Aggregate per-class multi-seed results into a paper-ready table.

Reads summary.json["perclass"] from each run directory (written by
eval_perclass_multiseed.py) and prints / saves a LaTeX-ready table:
  Baseline | seed0 | seed7 | seed42 | seed99 | seed123 | Mean±Std

Also computes paired t-test and TOST vs B0 for each metric.

Usage:
    python scripts/summarize_perclass_multiseed.py \\
        --runs-dir runs \\
        --baselines B0 B1a B2_best B3 B5 \\
        --seeds 0 7 42 99 123 \\
        --output runs/perclass_multiseed_summary.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import scipy.stats as stats


METRICS = [
    ("iou_crack",    "IoU\\textsubscript{crack}"),
    ("iou_spalling", "IoU\\textsubscript{spalling}"),
    ("miou_fg",      "mIoU\\textsubscript{fg}"),
    ("bf1_crack",    "BF1\\textsubscript{crack}"),
    ("bf1_spalling", "BF1\\textsubscript{spalling}"),
]

DIR_PREFIX = {
    "B0":     "B0",
    "B1a":    "B1a",
    "B2_best":"B2_best",
    "B3":     "B3",
    "B5":     "B5",
}


def load_values(runs_dir: Path, baseline: str, seeds: list[int]) -> dict[str, list]:
    """Return {metric: [val_seed0, val_seed7, ...]} for one baseline."""
    prefix = DIR_PREFIX.get(baseline, baseline)
    result: dict[str, list] = {m: [] for m, _ in METRICS}
    for s in seeds:
        p = runs_dir / f"{prefix}_s{s}" / "summary.json"
        if not p.exists():
            print(f"  WARNING: {p} not found — filling with NaN", file=sys.stderr)
            for m, _ in METRICS:
                result[m].append(float("nan"))
            continue
        d = json.loads(p.read_text())
        pc = d.get("perclass")
        if pc is None:
            print(f"  WARNING: {p} has no 'perclass' key — run eval_perclass_multiseed.py first",
                  file=sys.stderr)
            for m, _ in METRICS:
                result[m].append(float("nan"))
            continue
        for m, _ in METRICS:
            result[m].append(pc.get(m, float("nan")))
    return result


def tost_p(diff: np.ndarray, margin: float = 0.01) -> float:
    """Two one-sided t-tests; return max p-value (conservative)."""
    _, p_lo = stats.ttest_1samp(diff - (-margin), 0, alternative="greater")
    _, p_hi = stats.ttest_1samp(diff - margin,    0, alternative="less")
    return max(p_lo, p_hi)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    parser.add_argument("--baselines", nargs="+", default=["B0", "B1a", "B2_best", "B3", "B5"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 7, 42, 99, 123])
    parser.add_argument("--output", type=Path, default=Path("runs/perclass_multiseed_summary.json"))
    parser.add_argument("--latex", action="store_true", help="Print LaTeX table snippet")
    args = parser.parse_args()

    data: dict[str, dict[str, list]] = {}
    for bl in args.baselines:
        data[bl] = load_values(args.runs_dir, bl, args.seeds)

    b0 = data["B0"]

    # ── Console table ──
    seed_labels = [f"s{s}" for s in args.seeds]
    for metric, label in METRICS:
        print(f"\n{'='*70}")
        print(f"Metric: {label}")
        print(f"{'Baseline':>10}  " +
              "  ".join(f"{sl:>6}" for sl in seed_labels) +
              f"  {'Mean':>6}  {'Std0':>5}  {'|diff|':>6}  {'t p':>6}  {'TOST p':>7}")
        print("-" * 70)
        b0_vals = np.array(b0[metric])
        for bl in args.baselines:
            vals = np.array(data[bl][metric])
            mean = np.nanmean(vals)
            std0 = np.nanstd(vals, ddof=0)
            row = f"{bl:>10}  " + "  ".join(f"{v:6.4f}" for v in vals)
            row += f"  {mean:6.4f}  {std0:5.4f}"
            if bl != "B0":
                diff = vals - b0_vals
                _, p_t = stats.ttest_rel(vals, b0_vals)
                p_tost = tost_p(diff)
                row += f"  {np.nanmean(diff):+6.4f}  {p_t:6.4f}  {p_tost:7.4f}"
            print(row)

    # ── LaTeX snippet ──
    if args.latex:
        print("\n\n% ── LaTeX table (copy into paper) ──")
        for metric, label in METRICS:
            print(f"\n% {label}")
            print(r"\begin{tabular}{lcccccc}")
            print(r"\toprule")
            print(r" & Seed 0 & Seed 7 & Seed 42 & Seed 99 & Seed 123 & Mean $\pm$ Std \\")
            print(r"\midrule")
            for bl in args.baselines:
                vals = np.array(data[bl][metric])
                mean = np.nanmean(vals)
                std0 = np.nanstd(vals, ddof=0)
                cells = " & ".join(f"{v:.3f}" for v in vals)
                print(f"{bl} & {cells} & ${mean:.3f} \\pm {std0:.3f}$ \\\\")
            print(r"\bottomrule")
            print(r"\end{tabular}")

    # ── Save JSON ──
    out = {}
    for bl in args.baselines:
        out[bl] = {}
        for metric, _ in METRICS:
            vals = np.array(data[bl][metric])
            out[bl][metric] = {
                "per_seed": {str(s): float(v) for s, v in zip(args.seeds, vals)},
                "mean": float(np.nanmean(vals)),
                "std_ddof0": float(np.nanstd(vals, ddof=0)),
                "std_ddof1": float(np.nanstd(vals, ddof=1)),
            }
            if bl != "B0":
                b0_vals = np.array(b0[metric])
                diff = vals - b0_vals
                _, p_t = stats.ttest_rel(vals, b0_vals)
                out[bl][metric]["diff_mean"] = float(np.nanmean(diff))
                out[bl][metric]["paired_t_p"] = float(p_t)
                out[bl][metric]["tost_p"] = float(tost_p(diff))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2))
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
