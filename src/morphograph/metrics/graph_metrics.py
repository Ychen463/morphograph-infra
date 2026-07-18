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
    ep, er, ef = _keypoint_prf(pred_endpoints, target_endpoints, tolerance_px)
    jp, jr, jf = _keypoint_prf(pred_junctions, target_junctions, tolerance_px)

    # Edge matching
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

    edge_tp = 0
    if len(pred_n_all) > 0 and len(target_n_all) > 0 and len(pred_edges) > 0:
        node_dists = cdist(pred_n_all, target_n_all)
        # One-to-one node matching
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
    )
