"""Regenerate DG figures with 5-seed data.

Produces:
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
    single_seed_delta_b2 = 1.0  # +1.0% at seed 42 (exploratory)
    multi_mean_b2 = (bl["B2_best"]["best_miou_fg_mean"] - bl["B0"]["best_miou_fg_mean"]) * 100
    multi_std_b2 = np.sqrt(bl["B2_best"]["best_miou_fg_std"] ** 2 + bl["B0"]["best_miou_fg_std"] ** 2) * 100

    ax.bar(0, single_seed_delta_b2, color="#C0504D", width=0.6)
    ax.bar(1, multi_mean_b2, yerr=multi_std_b2, color="#4472C4", width=0.6,
           capsize=6, error_kw={"linewidth": 2})
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Exploratory\n(single run)", "Confirmatory\n(5 seeds)"], fontsize=10)
    ax.set_ylabel(r"$\Delta$ mIoU$_{fg}$ (%)", fontsize=11)
    ax.set_title("(a) B2 skeleton DT vs B0", fontsize=12, fontweight="bold")
    ax.text(1, multi_mean_b2 + multi_std_b2 + 0.1, "p=0.713 (ns)", ha="center", fontsize=10)
    ax.set_ylim(-1.5, 1.5)
    ax.grid(axis="y", alpha=0.3)

    # --- Panel (b): B1a vs B0 ---
    ax = axes[1]
    single_seed_delta_b1a = -1.6  # −1.6% at seed 42 (exploratory)
    multi_mean_b1a = (bl["B1a"]["best_miou_fg_mean"] - bl["B0"]["best_miou_fg_mean"]) * 100
    multi_std_b1a = np.sqrt(bl["B1a"]["best_miou_fg_std"] ** 2 + bl["B0"]["best_miou_fg_std"] ** 2) * 100

    ax.bar(0, single_seed_delta_b1a, color="#C0504D", width=0.6)
    ax.bar(1, multi_mean_b1a, yerr=multi_std_b1a, color="#4472C4", width=0.6,
           capsize=6, error_kw={"linewidth": 2})
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Exploratory\n(single run)", "Confirmatory\n(5 seeds)"], fontsize=10)
    ax.set_ylabel(r"$\Delta$ mIoU$_{fg}$ (%)", fontsize=11)
    ax.set_title("(b) B1a clDice vs B0", fontsize=12, fontweight="bold")
    ax.text(1, 0.8, "p=0.947 (ns)\nTOST: equiv",
            ha="center", fontsize=10, color="green")
    ax.set_ylim(-2.0, 1.5)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUT / "fig_single_vs_multiseed_overturned.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {OUT / 'fig_single_vs_multiseed_overturned.png'}")


if __name__ == "__main__":
    plot_dg_domain_gap(load_dg_summary())
    plot_single_vs_multiseed()
