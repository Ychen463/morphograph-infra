"""Graph evaluation visualization utilities."""

from __future__ import annotations

from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from morphograph.data.graph_targets import CrackGraph


def draw_graph_overlay(
    ax: plt.Axes,
    img: np.ndarray,
    graph: CrackGraph,
    title: str,
) -> None:
    """Draw graph overlaid on image."""
    ax.imshow(img, alpha=0.6)
    for path in graph.edge_paths:
        if len(path) > 1:
            ax.plot(path[:, 1], path[:, 0], "g-", linewidth=1.0, alpha=0.8)
    if len(graph.endpoints) > 0:
        ax.plot(graph.endpoints[:, 1], graph.endpoints[:, 0],
                "ro", markersize=4, markeredgewidth=0.5)
    if len(graph.junctions) > 0:
        ax.plot(graph.junctions[:, 1], graph.junctions[:, 0],
                "bs", markersize=5, markeredgewidth=0.5)
    ax.set_title(title, fontsize=8)
    ax.axis("off")


def plot_summary_bars(
    all_agg: dict[str, dict],
    metric_keys: list[str],
    out_path: Path,
) -> None:
    """Grouped bar chart comparing methods across metrics."""
    labels = list(all_agg.keys())
    n_metrics = len(metric_keys)
    n_labels = len(labels)

    x = np.arange(n_metrics)
    width = 0.8 / n_labels

    fig, ax = plt.subplots(figsize=(max(10, n_metrics * 1.5), 5))
    for i, label in enumerate(labels):
        vals = [all_agg[label].get(k, {}).get("mean", 0.0) for k in metric_keys]
        ax.bar(x + i * width, vals, width, label=label, alpha=0.85)

    ax.set_xticks(x + width * (n_labels - 1) / 2)
    ax.set_xticklabels([k.replace("_", "\n") for k in metric_keys], fontsize=7)
    ax.set_ylabel("Score")
    ax.set_title("Graph Evaluation: Method Comparison")
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
