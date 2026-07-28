"""Statistical utilities for evaluation: aggregation and hypothesis tests."""

from __future__ import annotations

import numpy as np
from scipy import stats


def aggregate_metrics(
    per_image: list[dict],
    metric_keys: list[str],
) -> dict[str, dict[str, float]]:
    """Compute mean/std/median for each metric across images."""
    agg = {}
    for key in metric_keys:
        vals = [r[key] for r in per_image if key in r and r[key] is not None]
        if vals:
            agg[key] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
                "median": float(np.median(vals)),
                "n": len(vals),
            }
    return agg


def paired_wilcoxon(
    results_a: list[dict],
    results_b: list[dict],
    key: str,
) -> dict:
    """Paired Wilcoxon signed-rank test on a metric between two result sets."""
    b_by_name = {r["filename"]: r for r in results_b}
    pairs_x, pairs_y = [], []
    for r in results_a:
        if r["filename"] in b_by_name and key in r and key in b_by_name[r["filename"]]:
            pairs_x.append(r[key])
            pairs_y.append(b_by_name[r["filename"]][key])

    if len(pairs_x) < 10:
        return {"n_pairs": len(pairs_x), "statistic": None, "p_value": None}

    x, y = np.array(pairs_x), np.array(pairs_y)
    diff = y - x
    if np.all(diff == 0):
        return {"n_pairs": len(pairs_x), "statistic": 0.0, "p_value": 1.0, "mean_diff": 0.0}

    try:
        stat, p = stats.wilcoxon(diff, alternative="two-sided")
    except ValueError:
        return {"n_pairs": len(pairs_x), "statistic": None, "p_value": None}

    return {
        "n_pairs": len(pairs_x),
        "statistic": float(stat),
        "p_value": float(p),
        "mean_diff": float(np.mean(diff)),
    }
