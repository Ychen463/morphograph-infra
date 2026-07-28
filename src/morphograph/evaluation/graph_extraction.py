"""Graph extraction methods A/B/C/D for evaluation.

A: seg mask -> morphological skeleton -> graph (all models)
B: predicted DT > threshold -> skeleton -> graph (B2 only)
C: seg mask + predicted DT -> adaptive ridge -> graph (B2 only, novelty)
D: direct graph-topology prediction (P3 only, hybrid: topology from decoder,
   polylines from DT cost-field routing)
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage
from skimage.morphology import skeletonize

from morphograph.data.graph_targets import (
    CrackGraph, mask_to_graph, mask_to_skeleton, detect_keypoints, build_graph,
)
from morphograph.models.graph_decoder import (
    extract_nodes, build_candidate_pairs, recover_edge_paths,
)


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
    """Method C: DT-guided per-component adaptive ridge thresholding."""
    dt_masked = dt_pred * crack_mask.astype(np.float32)

    if not crack_mask.any() or dt_masked.max() == 0:
        return CrackGraph(
            endpoints=np.empty((0, 2), dtype=int),
            junctions=np.empty((0, 2), dtype=int),
        )

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
    model,
    outputs: dict[str, torch.Tensor],
    node_threshold: float = 0.3,
    nms_radius: int = 2,
    max_nodes: int = 50,
    knn_k: int = 8,
    edge_threshold: float = 0.5,
) -> CrackGraph:
    """Method D: direct graph decoder (P3 only)."""
    hm = torch.sigmoid(outputs["node_heatmap"])
    detected = extract_nodes(hm, threshold=node_threshold,
                             nms_radius=nms_radius, max_nodes=max_nodes)[0]

    if len(detected.coords) < 1:
        return CrackGraph(
            endpoints=np.empty((0, 2), dtype=int),
            junctions=np.empty((0, 2), dtype=int),
        )

    coords_512 = detected.coords.cpu().numpy() * 4.0
    types_np = detected.types.cpu().numpy()
    ep_mask = types_np == 0
    jn_mask = types_np == 1
    endpoints = coords_512[ep_mask].astype(int) if ep_mask.any() else np.empty((0, 2), dtype=int)
    junctions = coords_512[jn_mask].astype(int) if jn_mask.any() else np.empty((0, 2), dtype=int)

    if len(detected.coords) < 2:
        return CrackGraph(endpoints=endpoints, junctions=junctions)

    dt_128 = F.interpolate(
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

    dt_np = torch.sigmoid(outputs["skeleton"])[0, 0].cpu().numpy()
    edge_paths = recover_edge_paths(dt_np, coords_512, pred_edges)

    return CrackGraph(
        endpoints=endpoints,
        junctions=junctions,
        edges=pred_edges,
        edge_paths=edge_paths,
    )
