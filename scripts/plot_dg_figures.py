"""Regenerate paper figures with 5-seed confirmatory data.

Produces:
  paper/final/figures/fig_multiseed_baseline_comparison.png
  paper/final/figures/fig_dg_multiseed_domain_gap.png
  paper/final/figures/fig_single_vs_multiseed_overturned.png
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

RUNS = Path("runs")
OUT = Path("paper/final/figures")


def load_dg_summary():
    with open(RUNS / "dg_multiseed_summary.json") as f:
        return json.load(f)


def plot_dg_domain_gap(summary):
    methods = ["D1_erm", "D2a_coral", "D2b_dann", "D2c_mixstyle"]
    labels = ["ERM", "CORAL", "DANN", "MixStyle"]

    dam_means = [summary[m]["final_damseg_miou_fg_mean"] for m in methods]
    dam_stds = [summary[m]["final_damseg_miou_fg_std"] for m in methods]
    s2ds_means = [summary[m]["final_s2ds_miou_fg_mean"] for m in methods]
    s2ds_stds = [summary[m]["final_s2ds_miou_fg_std"] for m in methods]

    x = np.arange(len(methods))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(x - width / 2, dam_means, width, yerr=dam_stds, label="DamSegment (in-domain)",
           color="#4472C4", capsize=4, error_kw={"linewidth": 1.5})
    ax.bar(x + width / 2, s2ds_means, width, yerr=s2ds_stds, label="s2ds (out-of-domain)",
           color="#C0504D", capsize=4, error_kw={"linewidth": 1.5})

    # ERM reference lines
    ax.axhline(dam_means[0], color="#4472C4", linestyle="--", alpha=0.3, linewidth=1.5)
    ax.axhline(s2ds_means[0], color="#C0504D", linestyle="--", alpha=0.3, linewidth=1.5)

    # Gap annotation
    gap_mean = np.mean([summary[m]["domain_gap_mean"] for m in methods])
    ax.text(1.02, 0.39, f"Gap\n~{gap_mean * 100:.0f}%", transform=ax.get_yaxis_transform(),
            fontsize=12, color="grey", va="center")

    # p-value annotation
    ax.text(0.5, 0.97, "All methods equivalent\n(p=0.941)", transform=ax.transAxes,
            ha="center", va="top", fontsize=11,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="grey", alpha=0.8))

    ax.set_ylabel(r"mIoU$_{fg}$", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylim(0, 0.75)
    ax.legend(fontsize=11, loc="upper left")
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUT / "fig_dg_multiseed_domain_gap.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {OUT / 'fig_dg_multiseed_domain_gap.png'}")


def plot_single_vs_multiseed():
    # Two panels only: (a) B2 skeleton DT, (b) B1a clDice
    # MixStyle panel removed — DG result is reported separately in §4.5

    with open(RUNS / "baseline_multiseed_summary.json") as f:
        bl = json.load(f)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # --- Panel (a): B2 vs B0 ---
    ax = axes[0]
    # Both bars within-confirmatory: single seed 42 vs five-seed mean
    single_seed_delta_b2 = (
        bl["B2_best"]["per_seed"]["42"] - bl["B0"]["per_seed"]["42"]
    ) * 100  # +0.25 pp (new runs)
    multi_mean_b2 = (bl["B2_best"]["best_miou_fg_mean"] - bl["B0"]["best_miou_fg_mean"]) * 100
    multi_std_b2 = bl["pairwise_tests"]["B2_best"]["std_delta"] * 100

    ax.bar(0, single_seed_delta_b2, color="#C0504D", width=0.6)
    ax.bar(1, multi_mean_b2, yerr=multi_std_b2, color="#4472C4", width=0.6,
           capsize=6, error_kw={"linewidth": 2})
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Single seed\n(seed 42)", "Five-seed mean\n(same protocol)"], fontsize=10)
    ax.set_ylabel(r"$\Delta$ mIoU$_{fg}$ (pp)", fontsize=11)
    ax.set_title("(a) B2 skeleton DT vs B0", fontsize=12, fontweight="bold")
    ax.text(1, multi_mean_b2 - multi_std_b2 - 0.15, "p=0.190 (ns)", ha="center", fontsize=10)
    ax.set_ylim(-1.5, 1.5)
    ax.grid(axis="y", alpha=0.3)

    # --- Panel (b): B1a vs B0 ---
    ax = axes[1]
    # Both bars within-confirmatory: single seed 0 vs five-seed mean
    single_seed_delta_b1a = (
        bl["B1a"]["per_seed"]["0"] - bl["B0"]["per_seed"]["0"]
    ) * 100  # +0.90 pp (new runs)
    multi_mean_b1a = (bl["B1a"]["best_miou_fg_mean"] - bl["B0"]["best_miou_fg_mean"]) * 100
    multi_std_b1a = bl["pairwise_tests"]["B1a"]["std_delta"] * 100

    ax.bar(0, single_seed_delta_b1a, color="#C0504D", width=0.6)
    ax.bar(1, multi_mean_b1a, yerr=multi_std_b1a, color="#4472C4", width=0.6,
           capsize=6, error_kw={"linewidth": 2})
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Single seed\n(seed 0)", "Five-seed mean\n(same protocol)"], fontsize=10)
    ax.set_ylabel(r"$\Delta$ mIoU$_{fg}$ (pp)", fontsize=11)
    ax.set_title("(b) B1a clDice vs B0", fontsize=12, fontweight="bold")
    ax.text(1, multi_mean_b1a + multi_std_b1a + 0.1, "p=0.600 (ns)\nTOST: equiv",
            ha="center", fontsize=10, color="green")
    ax.set_ylim(-0.5, 1.5)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUT / "fig_single_vs_multiseed_overturned.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {OUT / 'fig_single_vs_multiseed_overturned.png'}")


def plot_baseline_comparison():
    """Regenerate fig_multiseed_baseline_comparison.png from current JSON."""
    with open(RUNS / "baseline_multiseed_summary.json") as f:
        bl = json.load(f)

    methods = ["B0", "B1a", "B2_best", "B3", "B5"]
    labels = ["B0\n(mask-only)", "B1a\n(+clDice)", "B2_best\n(+skelDT)", "B3\n(+keypoints)", "B5\n(+full joint)"]
    colors = ["#4472C4", "#C0504D", "#70AD47", "#ED7D31", "#7030A0"]
    seeds = [0, 7, 42, 99, 123]

    means = [bl[m]["best_miou_fg_mean"] for m in methods]
    stds  = [bl[m]["best_miou_fg_std"]  for m in methods]

    # Per-seed values for scatter dots
    per_seed = [[bl[m]["per_seed"][str(s)] for s in seeds] for m in methods]

    # p-values and TOST from pairwise_tests (vs B0); B0 has no test
    pvals  = [None] + [bl["pairwise_tests"][m]["p_value"] for m in methods[1:]]
    tost_p = [None] + [bl["pairwise_tests"][m]["tost_p"]  for m in methods[1:]]
    sig    = [None] + [bl["pairwise_tests"][m]["significant"] for m in methods[1:]]

    x = np.arange(len(methods))
    fig, ax = plt.subplots(figsize=(10, 6))

    bars = ax.bar(x, means, color=colors, alpha=0.85, zorder=2)
    ax.errorbar(x, means, yerr=stds, fmt="none", color="black",
                capsize=5, linewidth=1.8, zorder=3)

    # Per-seed dots
    for i, seeds_vals in enumerate(per_seed):
        jitter = np.zeros(len(seeds_vals))
        ax.scatter([x[i]] * len(seeds_vals), seeds_vals,
                   color="black", s=25, zorder=4, alpha=0.8)

    # p-value / TOST annotations
    for i, m in enumerate(methods[1:], 1):
        p = pvals[i]
        tp = tost_p[i]
        is_sig = sig[i]
        y_top = means[i] + stds[i] + 0.003

        if is_sig:
            label = f"p={p:.3f}*\nInferior"
            color = "red"
        else:
            label = f"p={p:.3f} (ns)"
            color = "dimgrey"
        ax.text(x[i], y_top, label, ha="center", va="bottom",
                fontsize=9, color=color, zorder=5)

        # TOST label for equivalent methods
        if tp is not None and tp < 0.05:
            ax.text(x[i], y_top + 0.013, f"TOST: equiv\n(p={tp:.3f})",
                    ha="center", va="bottom", fontsize=8.5,
                    color="green", fontweight="bold", zorder=5)

    # B0 reference line
    b0_mean = means[0]
    ax.axhline(b0_mean, color="#4472C4", linestyle="--", alpha=0.4, linewidth=1.5, zorder=1)
    ax.text(len(methods) - 0.45, b0_mean + 0.001, f"B0 mean={b0_mean:.3f}",
            ha="right", va="bottom", fontsize=9, color="#4472C4", alpha=0.8)

    ax.set_ylabel(r"mIoU$_{fg}$", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_title("Multi-Seed Comparison of Baseline Methods (5 seeds)", fontsize=13)
    ax.set_ylim(0.64, 0.73)
    ax.grid(axis="y", alpha=0.3, zorder=0)
    fig.tight_layout()

    out = OUT / "fig_multiseed_baseline_comparison.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    plot_baseline_comparison()
    plot_dg_domain_gap(load_dg_summary())
    plot_single_vs_multiseed()
