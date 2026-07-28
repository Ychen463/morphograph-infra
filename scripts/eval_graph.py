"""Graph-level evaluation: compare B0 vs B2_best vs P3 on crack graph extraction quality.

Usage:
    python scripts/eval_graph.py --data-root data/raw \
        --checkpoints runs/B0/best.pt runs/B2_dt_v5/best.pt \
        --labels B0 B2_best --output runs/eval_graph

    # With P3 model (Method D: direct graph decoder):
    python scripts/eval_graph.py --data-root data/raw \
        --checkpoints runs/B2_dt_v5/best.pt runs/P3b/best.pt \
        --labels B2_best P3b --output runs/eval_graph_p3

Computes per-image pixel metrics (mIoU, clDice, ConnR, BF1) AND graph metrics
(endpoint/junction/edge F1, width MAE) for each checkpoint.

Four graph extraction methods:
  A: seg mask -> morphological skeleton -> graph (all models)
  B: predicted DT > threshold -> skeleton -> graph (B2 only)
  C: seg mask + predicted DT -> adaptive ridge -> graph (B2 only, novelty)
  D: direct graph-topology prediction with learned-field edge geometry recovery (P3 only)
     (hybrid: topology from decoder, polylines from DT cost-field routing)
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy import ndimage, stats
from skimage.morphology import skeletonize

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from morphograph.data.schema import decode_rgb_mask, NUM_CLASSES
from morphograph.data.graph_targets import (
    mask_to_graph,
    mask_to_skeleton,
    mask_to_dt_target,
    detect_keypoints,
    build_graph,
    estimate_width,
    CrackGraph,
)
from morphograph.metrics.graph_metrics import (
    compute_graph_metrics,
    GraphMetrics,
    DEFAULT_KEYPOINT_TOLERANCE_PX,
)
from morphograph.metrics.segmentation import (
    compute_iou,
    compute_cldice,
    compute_connectivity_recall,
    compute_boundary_f1,
)
from morphograph.models.morphograph_net import MorphoAuxNet, MorphoGraphNet, BASELINE_HEADS
from morphograph.models.graph_decoder import (
    extract_nodes, build_candidate_pairs, recover_edge_paths,
)
from morphograph.training.utils import set_seed, discover_all_samples, split_data

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ── Model loading (same pattern as eval_topology.py) ──


def load_model(checkpoint_path: Path, device: torch.device):
    """Load model from checkpoint, auto-detecting B0-B5 vs P3."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = ckpt["model_state_dict"]

    # Detect P3 model by presence of fpn128 keys
    is_p3 = any(k.startswith("fpn128.") for k in state)

    if is_p3:
        # Detect whether graph heads are present (P3a/P3b vs P3-Base)
        has_graph_heads = any(k.startswith("node_head.") for k in state)
        model = MorphoGraphNet(
            backbone="mit_b2",
            num_classes=NUM_CLASSES,
            graph_heads=has_graph_heads,
        ).to(device)
        model.load_state_dict(state)
        model.eval()
        return model

    has_skeleton = any(k.startswith("skeleton_head.") for k in state)
    has_endpoint = any(k.startswith("endpoint_head.") for k in state)
    has_junction = any(k.startswith("junction_head.") for k in state)
    has_width = any(k.startswith("width_head.") for k in state)

    heads = {
        "seg_head": True,
        "skeleton_head": has_skeleton,
        "endpoint_head": has_endpoint,
        "junction_head": has_junction,
        "width_head": has_width,
    }

    model = MorphoAuxNet(
        backbone="mit_b2",
        num_classes=NUM_CLASSES,
        heads=heads,
    ).to(device)
    model.load_state_dict(state)
    model.eval()
    return model


# ── Graph extraction methods ──


def extract_graph_a(crack_mask: np.ndarray) -> CrackGraph:
    """Method A: seg mask -> morphological skeleton -> graph."""
    return mask_to_graph(crack_mask, min_branch_length=10, junction_merge_radius=5)


def extract_graph_b(
    dt_pred: np.ndarray,
    threshold: float = 0.5,
) -> CrackGraph:
    """Method B: predicted DT -> threshold -> skeleton -> graph."""
    binary = (dt_pred > threshold).astype(np.uint8)
    if not binary.any():
        return CrackGraph(
            endpoints=np.empty((0, 2), dtype=int),
            junctions=np.empty((0, 2), dtype=int),
        )
    skel = mask_to_skeleton(binary, dilate_radius=0)
    if not skel.any():
        return CrackGraph(
            endpoints=np.empty((0, 2), dtype=int),
            junctions=np.empty((0, 2), dtype=int),
        )
    endpoints, junctions = detect_keypoints(skel)
    return build_graph(skel, endpoints, junctions, binary_mask=binary)


def extract_graph_c(
    crack_mask: np.ndarray,
    dt_pred: np.ndarray,
    ridge_threshold: float = 0.3,
) -> CrackGraph:
    """Method C (novelty): DT-guided graph extraction.

    Per-component adaptive ridge thresholding on predicted DT,
    masked to seg crack region. Produces cleaner skeletons because
    DT peaks naturally follow centerlines.
    """
    dt_masked = dt_pred * crack_mask.astype(np.float32)

    if not crack_mask.any() or dt_masked.max() == 0:
        return CrackGraph(
            endpoints=np.empty((0, 2), dtype=int),
            junctions=np.empty((0, 2), dtype=int),
        )

    # Per-component adaptive thresholding
    labeled, n_comp = ndimage.label(crack_mask)
    ridge = np.zeros_like(crack_mask, dtype=bool)
    for i in range(1, n_comp + 1):
        comp = labeled == i
        local_dt = dt_masked * comp
        local_max = local_dt.max()
        if local_max > 0:
            ridge |= (local_dt > ridge_threshold * local_max) & comp

    if not ridge.any():
        return CrackGraph(
            endpoints=np.empty((0, 2), dtype=int),
            junctions=np.empty((0, 2), dtype=int),
        )

    skel = skeletonize(ridge)
    if not skel.any():
        return CrackGraph(
            endpoints=np.empty((0, 2), dtype=int),
            junctions=np.empty((0, 2), dtype=int),
        )

    endpoints, junctions = detect_keypoints(skel)
    return build_graph(
        skel, endpoints, junctions,
        min_branch_length=10,
        junction_merge_radius=5,
        binary_mask=crack_mask,
    )


def extract_graph_d(
    model: MorphoGraphNet,
    outputs: dict[str, torch.Tensor],
    node_threshold: float = 0.3,
    nms_radius: int = 3,
    max_nodes: int = 50,
    knn_k: int = 8,
    edge_threshold: float = 0.5,
) -> CrackGraph:
    """Method D: direct graph decoder (P3 only, no skeleton extraction)."""
    import torch.nn.functional as Fd

    hm = torch.sigmoid(outputs["node_heatmap"])
    detected = extract_nodes(hm, threshold=node_threshold,
                             nms_radius=nms_radius, max_nodes=max_nodes)[0]

    if len(detected.coords) < 1:
        return CrackGraph(
            endpoints=np.empty((0, 2), dtype=int),
            junctions=np.empty((0, 2), dtype=int),
        )

    # Scale to 512-space
    coords_512 = detected.coords.cpu().numpy() * 4.0

    # Split by type
    types_np = detected.types.cpu().numpy()
    ep_mask = types_np == 0
    jn_mask = types_np == 1
    endpoints = coords_512[ep_mask].astype(int) if ep_mask.any() else np.empty((0, 2), dtype=int)
    junctions = coords_512[jn_mask].astype(int) if jn_mask.any() else np.empty((0, 2), dtype=int)

    if len(detected.coords) < 2:
        return CrackGraph(endpoints=endpoints, junctions=junctions)

    # Edge prediction
    dt_128 = Fd.interpolate(
        torch.sigmoid(outputs["skeleton"]),
        size=(128, 128), mode="bilinear", align_corners=False,
    )
    candidates = build_candidate_pairs(detected.coords, k=knn_k)

    if len(candidates) == 0:
        return CrackGraph(endpoints=endpoints, junctions=junctions)

    edge_logits = model.edge_classifier(
        outputs["_fpn"][:1],
        dt_128[:1],
        detected.coords,
        detected.types,
        detected.scores,
        candidates,
    )
    pred_edge_mask = torch.sigmoid(edge_logits) > edge_threshold
    pred_edges = candidates[pred_edge_mask].cpu().tolist()
    pred_edges = [(min(a, b), max(a, b)) for a, b in pred_edges]

    # P3c: recover edge polylines via Dijkstra
    dt_np = torch.sigmoid(outputs["skeleton"])[0, 0].cpu().numpy()
    edge_paths = recover_edge_paths(dt_np, coords_512, pred_edges)

    return CrackGraph(
        endpoints=endpoints,
        junctions=junctions,
        edges=pred_edges,
        edge_paths=edge_paths,
    )


# ── Structural metrics (computed inline) ──


def approx_graph_edit_distance(
    pred_graph: CrackGraph,
    gt_graph: CrackGraph,
) -> float:
    """Approximate GED: (node ins + del + edge ins + del) / GT size."""
    gt_nodes = gt_graph.num_nodes
    gt_edges = gt_graph.num_edges
    gt_size = gt_nodes + gt_edges
    if gt_size == 0:
        return 0.0 if pred_graph.num_nodes + pred_graph.num_edges == 0 else 1.0

    node_diff = abs(pred_graph.num_nodes - gt_nodes)
    edge_diff = abs(pred_graph.num_edges - gt_edges)
    return (node_diff + edge_diff) / gt_size


def path_continuity(
    pred_graph: CrackGraph,
    gt_graph: CrackGraph,
    tolerance_px: float = DEFAULT_KEYPOINT_TOLERANCE_PX,
) -> float:
    """For each GT edge, check if matched pred nodes are connected.

    Returns fraction of GT edges with a continuous predicted path.
    """
    if gt_graph.num_edges == 0:
        return 1.0
    if pred_graph.num_nodes == 0:
        return 0.0

    from scipy.optimize import linear_sum_assignment
    from scipy.spatial.distance import cdist

    gt_all = gt_graph.all_nodes
    pred_all = pred_graph.all_nodes
    if len(gt_all) == 0 or len(pred_all) == 0:
        return 0.0

    dists = cdist(gt_all, pred_all)
    row_ind, col_ind = linear_sum_assignment(dists)
    gt_to_pred = {}
    for r, c in zip(row_ind, col_ind):
        if dists[r, c] <= tolerance_px:
            gt_to_pred[r] = c

    # Build pred adjacency
    adj: dict[int, set[int]] = {i: set() for i in range(pred_graph.num_nodes)}
    for a, b in pred_graph.edges:
        adj[a].add(b)
        adj[b].add(a)

    def connected(src: int, dst: int) -> bool:
        if src == dst:
            return True
        visited = {src}
        queue = [src]
        while queue:
            node = queue.pop(0)
            for nb in adj.get(node, set()):
                if nb == dst:
                    return True
                if nb not in visited:
                    visited.add(nb)
                    queue.append(nb)
        return False

    continuous = 0
    for a, b in gt_graph.edges:
        if a in gt_to_pred and b in gt_to_pred:
            if connected(gt_to_pred[a], gt_to_pred[b]):
                continuous += 1

    return continuous / gt_graph.num_edges


def degree_distribution_kl(
    pred_graph: CrackGraph,
    gt_graph: CrackGraph,
    max_degree: int = 6,
) -> float:
    """KL divergence between predicted and GT node degree histograms."""

    def _degree_hist(edges, num_nodes):
        if num_nodes == 0:
            return np.ones(max_degree + 1) / (max_degree + 1)
        degrees = np.zeros(num_nodes, dtype=int)
        for a, b in edges:
            if a < num_nodes:
                degrees[a] += 1
            if b < num_nodes:
                degrees[b] += 1
        hist = np.zeros(max_degree + 1, dtype=float)
        for d in degrees:
            hist[min(d, max_degree)] += 1
        hist = hist / hist.sum() if hist.sum() > 0 else np.ones_like(hist) / len(hist)
        # Laplace smoothing
        hist = (hist + 1e-6)
        hist = hist / hist.sum()
        return hist

    p = _degree_hist(gt_graph.edges, gt_graph.num_nodes)
    q = _degree_hist(pred_graph.edges, pred_graph.num_nodes)
    return float(np.sum(p * np.log(p / q)))


# ── Per-image evaluation ──


def _graph_metrics_dict(
    pred: CrackGraph, gt: CrackGraph, tolerance: float, prefix: str,
) -> dict:
    """Compute graph metrics and return as prefixed dict."""
    gm = compute_graph_metrics(
        pred_endpoints=pred.endpoints,
        pred_junctions=pred.junctions,
        pred_edges=pred.edges,
        target_endpoints=gt.endpoints,
        target_junctions=gt.junctions,
        target_edges=gt.edges,
        pred_width=pred.width_at_nodes,
        target_width=gt.width_at_nodes,
        tolerance_px=tolerance,
    )
    d = {f"{prefix}_{k}": v for k, v in asdict(gm).items()}
    d[f"{prefix}_ged"] = approx_graph_edit_distance(pred, gt)
    d[f"{prefix}_path_cont"] = path_continuity(pred, gt, tolerance)
    d[f"{prefix}_degree_kl"] = degree_distribution_kl(pred, gt)
    d[f"{prefix}_num_nodes"] = pred.num_nodes
    d[f"{prefix}_num_edges"] = pred.num_edges
    return d


def evaluate_single_image(
    model,
    img_path: Path,
    mask_path: Path,
    device: torch.device,
    has_dt: bool,
    kp_tolerance: float,
    ridge_threshold: float,
    is_p3: bool = False,
    img_size: int = 512,
) -> dict | None:
    """Evaluate one image, returning all metrics or None if no crack."""
    img = np.array(
        Image.open(img_path).convert("RGB").resize(
            (img_size, img_size), Image.BILINEAR
        )
    )
    mask_raw = np.array(
        Image.open(mask_path).resize((img_size, img_size), Image.NEAREST)
    )
    if mask_raw.ndim == 3:
        gt = decode_rgb_mask(mask_raw)
    else:
        gt = mask_raw.astype(np.uint8)

    gt_crack = (gt == 1).astype(np.uint8)

    # Skip images with no crack in GT
    if not gt_crack.any():
        return None

    # Inference
    img_t = torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    with torch.no_grad(), torch.amp.autocast("cuda", enabled=device.type == "cuda"):
        outputs = model(img_t.to(device))

    pred_seg = outputs["seg"].argmax(dim=1)[0].cpu().numpy()
    pred_crack = (pred_seg == 1).astype(np.uint8)

    dt_pred = None
    if has_dt or is_p3:
        dt_pred = torch.sigmoid(outputs["skeleton"])[0, 0].cpu().numpy()

    # ── Pixel metrics ──
    ious = compute_iou(pred_seg, gt)
    fg_vals = [v for c, v in ious.items() if c > 0]
    result = {
        "filename": img_path.name,
        "crack_px": int(gt_crack.sum()),
        "miou_fg": float(np.mean(fg_vals)) if fg_vals else 0.0,
        "iou_crack": ious.get(1, 0.0),
        "cldice": compute_cldice(pred_crack.astype(bool), gt_crack.astype(bool)),
        "connr": compute_connectivity_recall(
            pred_crack.astype(bool), gt_crack.astype(bool)
        ),
        "bf1_crack": compute_boundary_f1(pred_seg, gt).get(1, 0.0),
    }

    # ── GT graph ──
    gt_graph = mask_to_graph(gt_crack, min_branch_length=10, junction_merge_radius=5)
    result["gt_num_nodes"] = gt_graph.num_nodes
    result["gt_num_edges"] = gt_graph.num_edges

    # ── Method A (all models) ──
    pred_graph_a = extract_graph_a(pred_crack)
    result.update(
        _graph_metrics_dict(pred_graph_a, gt_graph, kp_tolerance, "A")
    )

    # ── Methods B & C (B2 only) ──
    if has_dt and dt_pred is not None and not is_p3:
        pred_graph_b = extract_graph_b(dt_pred, threshold=0.5)
        result.update(
            _graph_metrics_dict(pred_graph_b, gt_graph, kp_tolerance, "B")
        )

        pred_graph_c = extract_graph_c(pred_crack, dt_pred, ridge_threshold)
        result.update(
            _graph_metrics_dict(pred_graph_c, gt_graph, kp_tolerance, "C")
        )

    # ── Method D (P3 only) ──
    if is_p3:
        pred_graph_d = extract_graph_d(model, outputs)
        result.update(
            _graph_metrics_dict(pred_graph_d, gt_graph, kp_tolerance, "D")
        )

    return result


# ── Visualization ──


def draw_graph_overlay(
    ax: plt.Axes,
    img: np.ndarray,
    graph: CrackGraph,
    title: str,
) -> None:
    """Draw graph overlaid on image."""
    ax.imshow(img, alpha=0.6)

    # Edges
    for path in graph.edge_paths:
        if len(path) > 1:
            ax.plot(path[:, 1], path[:, 0], "g-", linewidth=1.0, alpha=0.8)

    # Endpoints
    if len(graph.endpoints) > 0:
        ax.plot(
            graph.endpoints[:, 1],
            graph.endpoints[:, 0],
            "ro",
            markersize=4,
            markeredgewidth=0.5,
        )

    # Junctions
    if len(graph.junctions) > 0:
        ax.plot(
            graph.junctions[:, 1],
            graph.junctions[:, 0],
            "bs",
            markersize=5,
            markeredgewidth=0.5,
        )

    ax.set_title(title, fontsize=8)
    ax.axis("off")


def visualize_sample(
    img_path: Path,
    mask_path: Path,
    model: MorphoAuxNet,
    device: torch.device,
    has_dt: bool,
    ridge_threshold: float,
    out_path: Path,
    img_size: int = 512,
) -> None:
    """Generate 6-panel visualization for one sample."""
    img = np.array(
        Image.open(img_path).convert("RGB").resize(
            (img_size, img_size), Image.BILINEAR
        )
    )
    mask_raw = np.array(
        Image.open(mask_path).resize((img_size, img_size), Image.NEAREST)
    )
    if mask_raw.ndim == 3:
        gt = decode_rgb_mask(mask_raw)
    else:
        gt = mask_raw.astype(np.uint8)
    gt_crack = (gt == 1).astype(np.uint8)

    img_t = torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    with torch.no_grad(), torch.amp.autocast("cuda", enabled=device.type == "cuda"):
        outputs = model(img_t.to(device))

    pred_seg = outputs["seg"].argmax(dim=1)[0].cpu().numpy()
    pred_crack = (pred_seg == 1).astype(np.uint8)

    gt_graph = mask_to_graph(gt_crack)
    pred_graph_a = extract_graph_a(pred_crack)

    ncols = 6 if has_dt else 5
    fig, axes = plt.subplots(1, ncols, figsize=(4 * ncols, 4))

    axes[0].imshow(img)
    axes[0].set_title("Image", fontsize=8)
    axes[0].axis("off")

    axes[1].imshow(gt_crack, cmap="gray")
    axes[1].set_title("GT Crack", fontsize=8)
    axes[1].axis("off")

    axes[2].imshow(pred_crack, cmap="gray")
    axes[2].set_title("Pred Crack", fontsize=8)
    axes[2].axis("off")

    draw_graph_overlay(axes[3], img, gt_graph, "GT Graph")
    draw_graph_overlay(axes[4], img, pred_graph_a, "Pred Graph (A)")

    if has_dt:
        dt_pred = torch.sigmoid(outputs["skeleton"])[0, 0].cpu().numpy()
        pred_graph_c = extract_graph_c(pred_crack, dt_pred, ridge_threshold)
        draw_graph_overlay(axes[5], img, pred_graph_c, "Pred Graph (C)")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── Aggregation and statistics ──


def aggregate_metrics(
    per_image: list[dict],
    metric_keys: list[str],
) -> dict[str, dict[str, float]]:
    """Compute mean/std/median for each metric."""
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
    # Match by filename
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


def ridge_threshold_sweep(
    model: MorphoAuxNet,
    val_pairs: list[tuple[Path, Path]],
    device: torch.device,
    kp_tolerance: float,
    thresholds: list[float],
    max_images: int = 50,
) -> dict:
    """Sweep ridge_threshold for Method C, report edge F1."""
    results = {t: [] for t in thresholds}

    for img_path, mask_path in val_pairs[:max_images]:
        mask_raw = np.array(
            Image.open(mask_path).resize((512, 512), Image.NEAREST)
        )
        if mask_raw.ndim == 3:
            gt = decode_rgb_mask(mask_raw)
        else:
            gt = mask_raw.astype(np.uint8)
        gt_crack = (gt == 1).astype(np.uint8)
        if not gt_crack.any():
            continue

        img = np.array(
            Image.open(img_path).convert("RGB").resize((512, 512), Image.BILINEAR)
        )
        img_t = torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0) / 255.0
        with torch.no_grad(), torch.amp.autocast(
            "cuda", enabled=device.type == "cuda"
        ):
            outputs = model(img_t.to(device))

        pred_seg = outputs["seg"].argmax(dim=1)[0].cpu().numpy()
        pred_crack = (pred_seg == 1).astype(np.uint8)
        dt_pred = torch.sigmoid(outputs["skeleton"])[0, 0].cpu().numpy()

        gt_graph = mask_to_graph(gt_crack)

        for t in thresholds:
            pg = extract_graph_c(pred_crack, dt_pred, ridge_threshold=t)
            gm = compute_graph_metrics(
                pg.endpoints, pg.junctions, pg.edges,
                gt_graph.endpoints, gt_graph.junctions, gt_graph.edges,
                tolerance_px=kp_tolerance,
            )
            results[t].append(gm.edge_f1)

    summary = {}
    for t in thresholds:
        vals = results[t]
        summary[str(t)] = {
            "mean_edge_f1": float(np.mean(vals)) if vals else 0.0,
            "mean_ep_f1": 0.0,  # simplified for sweep
            "n_images": len(vals),
        }
    return summary


# ── Summary bar chart ──


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
        vals = [
            all_agg[label].get(k, {}).get("mean", 0.0) for k in metric_keys
        ]
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


# ── Main ──


def main() -> None:
    parser = argparse.ArgumentParser(description="Graph-level evaluation")
    parser.add_argument("--data-root", type=Path, default=Path("data/raw"))
    parser.add_argument(
        "--checkpoints", nargs="+", type=Path, required=True,
        help="Checkpoint paths, e.g. runs/B0/best.pt runs/B2/best.pt",
    )
    parser.add_argument(
        "--labels", nargs="+", type=str, default=None,
        help="Labels for each checkpoint",
    )
    parser.add_argument("--output", type=Path, default=Path("runs/eval_graph"))
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device", type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--kp-tolerance", type=float, default=DEFAULT_KEYPOINT_TOLERANCE_PX)
    parser.add_argument("--ridge-threshold", type=float, default=0.3)
    parser.add_argument("--num-vis", type=int, default=20)
    args = parser.parse_args()

    set_seed(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "vis").mkdir(exist_ok=True)
    device = torch.device(args.device)

    labels = args.labels or [p.parent.name for p in args.checkpoints]
    assert len(labels) == len(args.checkpoints), "labels must match checkpoints"

    # ── Data ──
    all_pairs = discover_all_samples(args.data_root)
    _, val_pairs = split_data(all_pairs, args.val_ratio, args.seed)
    print(f"Evaluating on {len(val_pairs)} validation images\n")

    # Pixel metric keys (shared across all methods)
    pixel_keys = ["miou_fg", "iou_crack", "cldice", "connr", "bf1_crack"]
    # Graph metric keys per method
    graph_core = [
        "endpoint_f1", "junction_f1", "edge_f1", "width_mae",
        "ged", "path_cont", "degree_kl",
        # Multi-tier tolerance (from GraphMetrics)
        "endpoint_f1_relaxed", "junction_f1_relaxed", "edge_f1_relaxed",
        "endpoint_f1_lenient", "junction_f1_lenient", "edge_f1_lenient",
        "edge_f1_soft",
    ]

    # ── Evaluate each checkpoint ──
    all_checkpoint_results = {}  # label -> list[dict]
    dt_model_label = None  # track which model has DT for threshold sweep

    for label, ckpt_path in zip(labels, args.checkpoints):
        print(f"{'=' * 60}")
        print(f"Evaluating: {label} ({ckpt_path})")
        print(f"{'=' * 60}")

        model = load_model(ckpt_path, device)
        is_p3 = isinstance(model, MorphoGraphNet) and model.has_graph_heads
        has_dt = isinstance(model, MorphoGraphNet) or any(
            k.startswith("skeleton_head.") for k in model.state_dict()
        )
        if has_dt and not is_p3:
            dt_model_label = label

        per_image = []
        for i, (img_path, mask_path) in enumerate(val_pairs):
            result = evaluate_single_image(
                model, img_path, mask_path, device,
                has_dt=has_dt,
                kp_tolerance=args.kp_tolerance,
                ridge_threshold=args.ridge_threshold,
                is_p3=is_p3,
            )
            if result is not None:
                per_image.append(result)

            if (i + 1) % 50 == 0:
                print(f"  [{i + 1}/{len(val_pairs)}]")

        print(f"  {len(per_image)} images with crack (skipped {len(val_pairs) - len(per_image)} no-crack)\n")

        all_checkpoint_results[label] = per_image

        # Determine which method keys to aggregate
        methods = ["A"]
        if is_p3:
            methods.append("D")
        elif has_dt:
            methods.extend(["B", "C"])

        for method in methods:
            method_keys = [f"{method}_{k}" for k in graph_core]
            agg = aggregate_metrics(per_image, pixel_keys + method_keys)
            print(f"  Method {method}:")
            for k in method_keys:
                if k in agg:
                    s = agg[k]
                    print(f"    {k:30s}: {s['mean']:.4f} +/- {s['std']:.4f}")
            print()

        # Pixel metrics (same across methods)
        agg_px = aggregate_metrics(per_image, pixel_keys)
        print("  Pixel metrics:")
        for k in pixel_keys:
            if k in agg_px:
                s = agg_px[k]
                print(f"    {k:20s}: {s['mean']:.4f} +/- {s['std']:.4f}")
        print()

        # Visualizations
        vis_count = 0
        for img_path, mask_path in val_pairs:
            if vis_count >= args.num_vis:
                break
            mask_raw = np.array(
                Image.open(mask_path).resize((512, 512), Image.NEAREST)
            )
            if mask_raw.ndim == 3:
                gt_check = decode_rgb_mask(mask_raw)
            else:
                gt_check = mask_raw
            if not (gt_check == 1).any():
                continue
            out_vis = args.output / "vis" / f"{label}_{vis_count:03d}.png"
            visualize_sample(
                img_path, mask_path, model, device,
                has_dt=has_dt,
                ridge_threshold=args.ridge_threshold,
                out_path=out_vis,
            )
            vis_count += 1
        print(f"  Saved {vis_count} visualizations\n")

        del model
        torch.cuda.empty_cache()

    # ── Ridge threshold sweep (for DT models) ──
    sweep_result = None
    if dt_model_label is not None:
        print("=" * 60)
        print("Ridge Threshold Sweep (Method C)")
        print("=" * 60)
        ckpt_idx = labels.index(dt_model_label)
        model = load_model(args.checkpoints[ckpt_idx], device)
        sweep_result = ridge_threshold_sweep(
            model, val_pairs, device,
            kp_tolerance=args.kp_tolerance,
            thresholds=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        )
        for t, s in sweep_result.items():
            print(f"  alpha={t}: edge_f1={s['mean_edge_f1']:.4f} (n={s['n_images']})")
        print()
        del model
        torch.cuda.empty_cache()

    # ── Comparison table ──
    print("=" * 60)
    print("COMPARISON TABLE")
    print("=" * 60)

    # Build aggregated results for bar chart
    bar_agg = {}

    for label in labels:
        per_image = all_checkpoint_results[label]
        has_dt = any("C_edge_f1" in r for r in per_image)
        has_d = any("D_edge_f1" in r for r in per_image)
        methods = ["A"]
        if has_d:
            methods.append("D")
        elif has_dt:
            methods.extend(["B", "C"])

        for method in methods:
            tag = f"{label}-{method}"
            method_keys = [f"{method}_{k}" for k in graph_core]
            bar_agg[tag] = aggregate_metrics(per_image, pixel_keys + method_keys)
            # Flatten method-prefixed keys to unprefixed for chart
            for k in graph_core:
                mk = f"{method}_{k}"
                if mk in bar_agg[tag]:
                    bar_agg[tag][k] = bar_agg[tag][mk]

    # Print table
    compare_keys = pixel_keys + graph_core  # all graph metrics
    tags = list(bar_agg.keys())
    header = f"{'Metric':25s}" + "".join(f" | {t:>14s}" for t in tags)
    print(header)
    print("-" * len(header))
    for key in compare_keys:
        row = f"{key:25s}"
        values = [bar_agg[t].get(key, {}).get("mean", 0.0) for t in tags]
        best_val = max(values) if values else 0
        for t in tags:
            val = bar_agg[t].get(key, {}).get("mean", 0.0)
            marker = " *" if val == best_val and len(tags) > 1 and val > 0 else "  "
            row += f" | {val:12.4f}{marker}"
        print(row)
    print()

    # ── Statistical tests ──
    # Pre-specified primary endpoint: edge F1 @10px (relaxed tolerance).
    # All other comparisons are exploratory.
    PRIMARY_METRIC = "D_edge_f1_relaxed"  # or A_/C_ depending on method
    if len(labels) >= 2:
        print("=" * 60)
        print("STATISTICAL TESTS (Wilcoxon signed-rank)")
        print(f"  Primary endpoint: edge F1 @10px (relaxed tolerance)")
        print("=" * 60)
        base_label = labels[0]
        stat_results = {}
        for other_label in labels[1:]:
            stat_results[f"{other_label}_vs_{base_label}"] = {}
            # Compare Method A (strict + relaxed + lenient + soft)
            test_keys = [
                "A_endpoint_f1", "A_junction_f1", "A_edge_f1", "A_path_cont",
                "A_edge_f1_relaxed", "A_edge_f1_lenient", "A_edge_f1_soft",
            ]
            # Add Method C if available
            if any("C_edge_f1" in r for r in all_checkpoint_results[other_label]):
                test_keys.extend([
                    "C_endpoint_f1", "C_junction_f1", "C_edge_f1", "C_path_cont",
                    "C_edge_f1_relaxed", "C_edge_f1_lenient", "C_edge_f1_soft",
                ])
            # Add Method D if available
            if any("D_edge_f1" in r for r in all_checkpoint_results[other_label]):
                test_keys.extend([
                    "D_endpoint_f1", "D_junction_f1", "D_edge_f1", "D_path_cont",
                    "D_edge_f1_relaxed", "D_edge_f1_lenient", "D_edge_f1_soft",
                ])

            for key in test_keys:
                wt = paired_wilcoxon(
                    all_checkpoint_results[base_label],
                    all_checkpoint_results[other_label],
                    key,
                )
                stat_results[f"{other_label}_vs_{base_label}"][key] = wt
                sig = ""
                if wt["p_value"] is not None:
                    sig = " ***" if wt["p_value"] < 0.001 else " **" if wt["p_value"] < 0.01 else " *" if wt["p_value"] < 0.05 else ""
                p_str = f"{wt['p_value']:.4f}" if wt["p_value"] is not None else "N/A"
                print(
                    f"  {key:30s}: diff={wt.get('mean_diff', 0):.4f}, "
                    f"p={p_str}{sig}"
                )
        print()

    # ── Summary bar chart ──
    chart_metric_suffixes = ["endpoint_f1", "junction_f1", "edge_f1", "path_cont"]
    chart_agg = {}
    for label in labels:
        per_image = all_checkpoint_results[label]
        has_dt = any("C_edge_f1" in r for r in per_image)
        has_d = any("D_edge_f1" in r for r in per_image)
        methods = ["A"]
        if has_d:
            methods.append("D")
        elif has_dt:
            methods.append("C")
        for method in methods:
            tag = f"{label}-{method}"
            mkeys = [f"{method}_{k}" for k in chart_metric_suffixes]
            chart_agg[tag] = aggregate_metrics(per_image, mkeys)

    first_method = list(chart_agg.keys())[0].split("-")[1]
    plot_summary_bars(
        chart_agg,
        [f"{first_method}_{k}" for k in chart_metric_suffixes],
        args.output / "vis" / "summary_bars.png",
    )

    # ── Save outputs ──

    # Per-image CSV
    csv_path = args.output / "per_image_results.csv"
    all_rows = []
    for label, per_image in all_checkpoint_results.items():
        for r in per_image:
            row = {"model": label}
            row.update(r)
            all_rows.append(row)

    if all_rows:
        fieldnames = list(all_rows[0].keys())
        # Ensure all keys present
        for row in all_rows:
            for k in row:
                if k not in fieldnames:
                    fieldnames.append(k)

        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"Per-image results: {csv_path}")

    # Summary JSON
    summary = {
        "val_samples": len(val_pairs),
        "primary_endpoint": "edge_f1_relaxed (@10px)",
        "params": {
            "kp_tolerance": args.kp_tolerance,
            "ridge_threshold": args.ridge_threshold,
            "seed": args.seed,
        },
        "per_model": {},
    }
    for label in labels:
        per_image = all_checkpoint_results[label]
        has_dt = any("C_edge_f1" in r for r in per_image)
        has_d = any("D_edge_f1" in r for r in per_image)
        methods = ["A"]
        if has_d:
            methods.append("D")
        elif has_dt:
            methods.extend(["B", "C"])
        model_summary = {"n_crack_images": len(per_image)}
        model_summary["pixel"] = aggregate_metrics(per_image, pixel_keys)
        for method in methods:
            mkeys = [f"{method}_{k}" for k in graph_core]
            model_summary[f"method_{method}"] = aggregate_metrics(per_image, mkeys)
        summary["per_model"][label] = model_summary

    if sweep_result:
        summary["ridge_threshold_sweep"] = sweep_result
        with open(args.output / "ridge_threshold_sweep.json", "w") as f:
            json.dump(sweep_result, f, indent=2)
        print(f"Ridge sweep: {args.output / 'ridge_threshold_sweep.json'}")
    if len(labels) >= 2:
        summary["statistical_tests"] = stat_results

    with open(args.output / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary: {args.output / 'summary.json'}")

    # Comparison table text
    table_path = args.output / "comparison_table.txt"
    with open(table_path, "w") as f:
        f.write(header + "\n")
        f.write("-" * len(header) + "\n")
        for key in compare_keys:
            row = f"{key:25s}"
            for t in tags:
                val = bar_agg[t].get(key, {}).get("mean", 0.0)
                row += f" | {val:12.4f}  "
            f.write(row + "\n")
    print(f"Comparison table: {table_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
