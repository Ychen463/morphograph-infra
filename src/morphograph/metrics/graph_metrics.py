"""Graph-level evaluation metrics for crack morphology.

Evaluation protocol parameters (must be frozen before test evaluation):
    - keypoint_tolerance_px: matching distance (default 5px at 512x512)
    - keypoint_matching: greedy nearest-neighbor, one-to-one
    - edge_matching: both endpoint nodes must match within tolerance
    - width_mae_scope: computed only at matched skeleton pixels
    - spur_threshold: minimum branch length for false-spur counting
    - gt_source: auto-derived labels for training, gold subset for final eval

Final graph metrics MUST be evaluated on the locked gold test set
(100-200 images, see DATA_CONTRACT.md). Auto mask-to-graph labels
are acceptable for large-scale training but not as sole final GT.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist


# Default evaluation protocol values. Freeze before test evaluation.
DEFAULT_KEYPOINT_TOLERANCE_PX = 5.0
DEFAULT_SPUR_THRESHOLD_PX = 10


RELAXED_TOLERANCE_PX = 10.0
LENIENT_TOLERANCE_PX = 15.0


@dataclass
class GraphMetrics:
    """Container for graph evaluation metrics."""
    endpoint_precision: float = 0.0
    endpoint_recall: float = 0.0
    endpoint_f1: float = 0.0
    junction_precision: float = 0.0
    junction_recall: float = 0.0
    junction_f1: float = 0.0
    edge_precision: float = 0.0
    edge_recall: float = 0.0
    edge_f1: float = 0.0
    width_mae: float = 0.0  # at matched skeleton pixels only
    false_spur_rate: float = 0.0
    # Relaxed tier (10px) — for development/ablation
    endpoint_f1_relaxed: float = 0.0
    junction_f1_relaxed: float = 0.0
    edge_f1_relaxed: float = 0.0
    # Lenient tier (15px) — upper bound / sanity check
    endpoint_f1_lenient: float = 0.0
    junction_f1_lenient: float = 0.0
    edge_f1_lenient: float = 0.0
    # Soft edge matching (partial credit at 15px)
    edge_f1_soft: float = 0.0


def approx_graph_edit_distance(
    pred_num_nodes: int,
    pred_num_edges: int,
    gt_num_nodes: int,
    gt_num_edges: int,
) -> float:
    """Approximate GED: (node ins + del + edge ins + del) / GT size."""
    gt_size = gt_num_nodes + gt_num_edges
    if gt_size == 0:
        return 0.0 if pred_num_nodes + pred_num_edges == 0 else 1.0
    node_diff = abs(pred_num_nodes - gt_num_nodes)
    edge_diff = abs(pred_num_edges - gt_num_edges)
    return (node_diff + edge_diff) / gt_size


def path_continuity(
    pred_nodes: np.ndarray,
    pred_edges: list[tuple[int, int]],
    gt_nodes: np.ndarray,
    gt_edges: list[tuple[int, int]],
    tolerance_px: float = DEFAULT_KEYPOINT_TOLERANCE_PX,
) -> float:
    """Fraction of GT edges whose matched pred nodes are connected."""
    if len(gt_edges) == 0:
        return 1.0
    if len(pred_nodes) == 0 or len(gt_nodes) == 0:
        return 0.0

    dists = cdist(gt_nodes, pred_nodes)
    row_ind, col_ind = linear_sum_assignment(dists)
    gt_to_pred = {}
    for r, c in zip(row_ind, col_ind):
        if dists[r, c] <= tolerance_px:
            gt_to_pred[r] = c

    adj: dict[int, set[int]] = {i: set() for i in range(len(pred_nodes))}
    for a, b in pred_edges:
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
    for a, b in gt_edges:
        if a in gt_to_pred and b in gt_to_pred:
            if connected(gt_to_pred[a], gt_to_pred[b]):
                continuous += 1

    return continuous / len(gt_edges)


def degree_distribution_kl(
    pred_edges: list[tuple[int, int]],
    pred_num_nodes: int,
    gt_edges: list[tuple[int, int]],
    gt_num_nodes: int,
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
        hist = (hist + 1e-6)
        hist = hist / hist.sum()
        return hist

    p = _degree_hist(gt_edges, gt_num_nodes)
    q = _degree_hist(pred_edges, pred_num_nodes)
    return float(np.sum(p * np.log(p / q)))


def _keypoint_prf(
    pred: np.ndarray,
    target: np.ndarray,
    tolerance_px: float = DEFAULT_KEYPOINT_TOLERANCE_PX,
) -> tuple[float, float, float]:
    """Precision, recall, F1 for keypoints with one-to-one matching.

    Uses the Hungarian algorithm for optimal one-to-one assignment,
    then filters matches within tolerance. This prevents a single
    GT point from being matched by multiple predictions.

    Args:
        pred: (N, 2) predicted keypoint coordinates (row, col).
        target: (M, 2) ground-truth keypoint coordinates.
        tolerance_px: maximum distance for a valid match.

    Returns:
        (precision, recall, f1).
    """
    if len(pred) == 0 and len(target) == 0:
        return 1.0, 1.0, 1.0
    if len(pred) == 0:
        return 0.0, 0.0, 0.0
    if len(target) == 0:
        return 0.0, 0.0, 0.0

    dists = cdist(pred, target)

    # Hungarian assignment (one-to-one)
    row_ind, col_ind = linear_sum_assignment(dists)
    tp = sum(1 for r, c in zip(row_ind, col_ind)
             if dists[r, c] <= tolerance_px)

    precision = tp / len(pred)
    recall = tp / len(target)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    return float(precision), float(recall), float(f1)


def _edge_prf(
    pred_nodes: np.ndarray,
    target_nodes: np.ndarray,
    pred_edges: list[tuple[int, int]],
    target_edges: list[tuple[int, int]],
    tolerance_px: float,
) -> tuple[float, float, float, int]:
    """Edge precision, recall, F1 at a given node tolerance.

    Returns (precision, recall, f1, tp_count).
    """
    edge_tp = 0
    if len(pred_nodes) > 0 and len(target_nodes) > 0 and len(pred_edges) > 0:
        node_dists = cdist(pred_nodes, target_nodes)
        row_ind, col_ind = linear_sum_assignment(node_dists)
        pred_to_target = {}
        for r, c in zip(row_ind, col_ind):
            if node_dists[r, c] <= tolerance_px:
                pred_to_target[r] = c

        target_edge_set = set()
        for a, b in target_edges:
            target_edge_set.add((min(a, b), max(a, b)))

        for a, b in pred_edges:
            if a in pred_to_target and b in pred_to_target:
                mapped = (
                    min(pred_to_target[a], pred_to_target[b]),
                    max(pred_to_target[a], pred_to_target[b]),
                )
                if mapped in target_edge_set:
                    edge_tp += 1

    edge_p = edge_tp / max(len(pred_edges), 1)
    edge_r = edge_tp / max(len(target_edges), 1)
    edge_f = 2 * edge_p * edge_r / (edge_p + edge_r + 1e-8)
    return float(edge_p), float(edge_r), float(edge_f), edge_tp


def _soft_edge_score(
    pred_nodes: np.ndarray,
    target_nodes: np.ndarray,
    pred_edges: list[tuple[int, int]],
    target_edges: list[tuple[int, int]],
    strict_tolerance: float = DEFAULT_KEYPOINT_TOLERANCE_PX,
    soft_tolerance: float = LENIENT_TOLERANCE_PX,
) -> tuple[float, float, float]:
    """Soft edge matching: full credit for strict match, 0.5 for soft match.

    For each predicted edge, if strict match fails, check whether both
    nodes are within soft_tolerance of ANY pair of GT nodes that share
    an edge. Award 0.5 partial credit.

    Returns (precision, recall, f1).
    """
    if len(pred_edges) == 0 and len(target_edges) == 0:
        return 1.0, 1.0, 1.0
    if len(pred_edges) == 0 or len(pred_nodes) == 0:
        return 0.0, 0.0, 0.0
    if len(target_edges) == 0 or len(target_nodes) == 0:
        return 0.0, 0.0, 0.0

    node_dists = cdist(pred_nodes, target_nodes)

    # Strict one-to-one matching
    row_ind, col_ind = linear_sum_assignment(node_dists)
    strict_map = {}
    for r, c in zip(row_ind, col_ind):
        if node_dists[r, c] <= strict_tolerance:
            strict_map[r] = c

    target_edge_set = set()
    for a, b in target_edges:
        target_edge_set.add((min(a, b), max(a, b)))

    score_sum = 0.0
    for a, b in pred_edges:
        # Try strict match first
        if a in strict_map and b in strict_map:
            mapped = (
                min(strict_map[a], strict_map[b]),
                max(strict_map[a], strict_map[b]),
            )
            if mapped in target_edge_set:
                score_sum += 1.0
                continue

        # Soft match: check if both pred nodes are near any GT edge's nodes
        found_soft = False
        for gt_a, gt_b in target_edges:
            dist_a_gta = node_dists[a, gt_a] if a < node_dists.shape[0] and gt_a < node_dists.shape[1] else float("inf")
            dist_b_gtb = node_dists[b, gt_b] if b < node_dists.shape[0] and gt_b < node_dists.shape[1] else float("inf")
            dist_a_gtb = node_dists[a, gt_b] if a < node_dists.shape[0] and gt_b < node_dists.shape[1] else float("inf")
            dist_b_gta = node_dists[b, gt_a] if b < node_dists.shape[0] and gt_a < node_dists.shape[1] else float("inf")
            if (dist_a_gta <= soft_tolerance and dist_b_gtb <= soft_tolerance) or \
               (dist_a_gtb <= soft_tolerance and dist_b_gta <= soft_tolerance):
                found_soft = True
                break
        if found_soft:
            score_sum += 0.5

    precision = score_sum / len(pred_edges)
    recall = score_sum / len(target_edges)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    return float(precision), float(recall), float(f1)


def compute_graph_metrics(
    pred_endpoints: np.ndarray,
    pred_junctions: np.ndarray,
    pred_edges: list[tuple[int, int]],
    target_endpoints: np.ndarray,
    target_junctions: np.ndarray,
    target_edges: list[tuple[int, int]],
    pred_width: np.ndarray | None = None,
    target_width: np.ndarray | None = None,
    tolerance_px: float = DEFAULT_KEYPOINT_TOLERANCE_PX,
) -> GraphMetrics:
    """Compute all graph-level metrics.

    Matching protocol:
    1. Endpoints and junctions are matched independently using
       Hungarian assignment with distance tolerance.
    2. An edge is a true positive if both its endpoint nodes are
       matched to GT nodes that form a GT edge.
    3. Width MAE is computed only at matched node pairs.

    Args:
        pred_endpoints: (N, 2) predicted endpoint coordinates.
        pred_junctions: (M, 2) predicted junction coordinates.
        pred_edges: edge list as (node_i, node_j) index pairs.
            Node indices refer to the concatenation [endpoints; junctions].
        target_endpoints: (N', 2) GT endpoint coordinates.
        target_junctions: (M', 2) GT junction coordinates.
        target_edges: GT edge list.
        pred_width: optional (K,) per-node width estimates.
        target_width: optional (K',) per-node GT width.
        tolerance_px: distance tolerance for matching.

    Returns:
        GraphMetrics instance.
    """
    # Strict tier (default tolerance)
    ep, er, ef = _keypoint_prf(pred_endpoints, target_endpoints, tolerance_px)
    jp, jr, jf = _keypoint_prf(pred_junctions, target_junctions, tolerance_px)

    pred_n_all = (
        np.concatenate([pred_endpoints, pred_junctions])
        if len(pred_endpoints) + len(pred_junctions) > 0
        else np.empty((0, 2))
    )
    target_n_all = (
        np.concatenate([target_endpoints, target_junctions])
        if len(target_endpoints) + len(target_junctions) > 0
        else np.empty((0, 2))
    )

    # Edge matching at strict tolerance
    edge_p, edge_r, edge_f, _ = _edge_prf(
        pred_n_all, target_n_all, pred_edges, target_edges, tolerance_px,
    )

    # Relaxed tier (10px)
    _, _, ef_relaxed = _keypoint_prf(pred_endpoints, target_endpoints, RELAXED_TOLERANCE_PX)
    _, _, jf_relaxed = _keypoint_prf(pred_junctions, target_junctions, RELAXED_TOLERANCE_PX)
    _, _, edge_f_relaxed, _ = _edge_prf(
        pred_n_all, target_n_all, pred_edges, target_edges, RELAXED_TOLERANCE_PX,
    )

    # Lenient tier (15px)
    _, _, ef_lenient = _keypoint_prf(pred_endpoints, target_endpoints, LENIENT_TOLERANCE_PX)
    _, _, jf_lenient = _keypoint_prf(pred_junctions, target_junctions, LENIENT_TOLERANCE_PX)
    _, _, edge_f_lenient, _ = _edge_prf(
        pred_n_all, target_n_all, pred_edges, target_edges, LENIENT_TOLERANCE_PX,
    )

    # Soft edge matching
    _, _, edge_f_soft = _soft_edge_score(
        pred_n_all, target_n_all, pred_edges, target_edges,
        strict_tolerance=tolerance_px, soft_tolerance=LENIENT_TOLERANCE_PX,
    )

    # Width MAE at matched nodes only
    width_mae = 0.0
    if (pred_width is not None and target_width is not None
            and len(pred_n_all) > 0 and len(target_n_all) > 0):
        node_dists = cdist(pred_n_all, target_n_all)
        row_ind, col_ind = linear_sum_assignment(node_dists)
        matched_errors = []
        for r, c in zip(row_ind, col_ind):
            if (node_dists[r, c] <= tolerance_px
                    and r < len(pred_width) and c < len(target_width)):
                matched_errors.append(abs(pred_width[r] - target_width[c]))
        if matched_errors:
            width_mae = float(np.mean(matched_errors))

    return GraphMetrics(
        endpoint_precision=ep, endpoint_recall=er, endpoint_f1=ef,
        junction_precision=jp, junction_recall=jr, junction_f1=jf,
        edge_precision=edge_p, edge_recall=edge_r, edge_f1=edge_f,
        width_mae=width_mae,
        false_spur_rate=0.0,
        endpoint_f1_relaxed=ef_relaxed,
        junction_f1_relaxed=jf_relaxed,
        edge_f1_relaxed=edge_f_relaxed,
        endpoint_f1_lenient=ef_lenient,
        junction_f1_lenient=jf_lenient,
        edge_f1_lenient=edge_f_lenient,
        edge_f1_soft=edge_f_soft,
    )
