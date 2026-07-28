"""Direct graph prediction heads for P3.

Direct graph-topology prediction with learned-field edge geometry recovery.

NodeHeatmapHead: predicts endpoint/junction heatmaps at 128x128.
EdgeClassifier: predicts pairwise adjacency from node features + corridor path evidence.
extract_nodes: NMS-based node extraction from predicted heatmaps.
recover_edge_paths: Dijkstra on normalized DT cost field for edge polyline recovery (P3c).
audit_candidate_recall: measures KNN candidate recall against GT edges.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class DetectedNodes:
    """Per-image detected nodes from heatmap."""
    coords: torch.Tensor  # (N, 2) in 128-space, float [row, col]
    types: torch.Tensor  # (N,) 0=endpoint, 1=junction
    scores: torch.Tensor  # (N,) detection confidence


class NodeHeatmapHead(nn.Module):
    """Predict endpoint/junction heatmaps at 128x128.

    Input: (B, 256, 128, 128) FPN features.
    Output: (B, 2, 128, 128) raw logits — ch0=endpoints, ch1=junctions.
    """

    def __init__(self, fpn_dim: int = 256) -> None:
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(fpn_dim, 128, 3, padding=1, bias=False),
            nn.GroupNorm(16, 128),
            nn.GELU(),
            nn.Conv2d(128, 64, 3, padding=1, bias=False),
            nn.GroupNorm(16, 64),
            nn.GELU(),
            nn.Conv2d(64, 2, 1),
        )

    def forward(self, fpn_features: torch.Tensor) -> torch.Tensor:
        return self.head(fpn_features)


def extract_nodes(
    heatmap: torch.Tensor,
    threshold: float = 0.3,
    nms_radius: int = 2,
    max_nodes: int = 50,
    refine: bool = True,
) -> list[DetectedNodes]:
    """Extract node coordinates from heatmap via NMS + sub-pixel refinement.

    Args:
        heatmap: (B, 2, H, W) after sigmoid.
        threshold: minimum score to keep.
        nms_radius: max-pool kernel radius for NMS.
        max_nodes: maximum nodes per image.
        refine: if True, apply weighted-centroid sub-pixel refinement.

    Returns:
        List of DetectedNodes, one per batch element.
    """
    # Cast to float32 — AMP may produce float16 where the equality
    # check (heatmap == pooled) fails due to reduced precision.
    heatmap = heatmap.float()

    B, C, H, W = heatmap.shape
    assert C == 2

    kernel = 2 * nms_radius + 1
    pooled = F.max_pool2d(heatmap, kernel_size=kernel, stride=1, padding=nms_radius)
    is_peak = (heatmap == pooled) & (heatmap > threshold)

    results = []
    for b in range(B):
        coords_list = []
        types_list = []
        scores_list = []
        for c in range(C):
            peaks = is_peak[b, c].nonzero(as_tuple=False)  # (K, 2) [row, col]
            if len(peaks) > 0:
                peak_scores = heatmap[b, c, peaks[:, 0], peaks[:, 1]]

                if refine:
                    refined = _refine_peaks(heatmap[b, c], peaks)
                else:
                    refined = peaks.float()

                coords_list.append(refined)
                types_list.append(torch.full((len(peaks),), c, device=heatmap.device))
                scores_list.append(peak_scores)

        if coords_list:
            all_coords = torch.cat(coords_list, dim=0)
            all_types = torch.cat(types_list, dim=0)
            all_scores = torch.cat(scores_list, dim=0)

            if len(all_coords) > max_nodes:
                topk = all_scores.topk(max_nodes)
                all_coords = all_coords[topk.indices]
                all_types = all_types[topk.indices]
                all_scores = topk.values

            results.append(DetectedNodes(
                coords=all_coords,
                types=all_types,
                scores=all_scores,
            ))
        else:
            results.append(DetectedNodes(
                coords=torch.empty((0, 2), device=heatmap.device),
                types=torch.empty((0,), dtype=torch.long, device=heatmap.device),
                scores=torch.empty((0,), device=heatmap.device),
            ))

    return results


def _refine_peaks(
    channel_hm: torch.Tensor,
    peaks: torch.Tensor,
    radius: int = 2,
) -> torch.Tensor:
    """Weighted-centroid sub-pixel refinement around each NMS peak.

    Reduces coordinate quantization error from the 512→128→512 round-trip.
    """
    H, W = channel_hm.shape
    refined = []
    for peak in peaks:
        r, c = int(peak[0].item()), int(peak[1].item())
        r0, r1 = max(0, r - radius), min(H, r + radius + 1)
        c0, c1 = max(0, c - radius), min(W, c + radius + 1)
        patch = channel_hm[r0:r1, c0:c1]
        total = patch.sum()
        if total < 1e-8:
            refined.append([float(r), float(c)])
            continue
        rr = torch.arange(r0, r1, device=channel_hm.device).float()
        cc = torch.arange(c0, c1, device=channel_hm.device).float()
        grid_r, grid_c = torch.meshgrid(rr, cc, indexing="ij")
        wr = (patch * grid_r).sum() / total
        wc = (patch * grid_c).sum() / total
        refined.append([wr.item(), wc.item()])
    return torch.tensor(refined, device=channel_hm.device)


def _sample_path_evidence(
    dt_map: torch.Tensor,
    coord_i: torch.Tensor,
    coord_j: torch.Tensor,
    num_samples: int = 20,
    corridor_width: float = 3.0,
    num_perp: int = 5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample DT values along a corridor between two nodes.

    Instead of sampling only along the straight line (which penalizes curved
    cracks), we build a corridor of width `corridor_width` and at each sample
    position along the line, take the max DT value across perpendicular offsets.
    This allows the evidence to capture cracks that curve away from the direct
    line between nodes.

    Args:
        dt_map: (1, 1, H, W) normalized DT field.
        coord_i: (2,) [row, col] in map space.
        coord_j: (2,) [row, col] in map space.
        num_samples: number of points along the line.
        corridor_width: half-width of perpendicular corridor in pixels.
        num_perp: number of perpendicular samples per position (each side).

    Returns:
        (q_mean, q_min, q_frac) all scalar tensors, computed from
        corridor-max values at each sample position.
    """
    device = dt_map.device
    H, W = dt_map.shape[2], dt_map.shape[3]

    t = torch.linspace(0, 1, num_samples, device=device)
    center_rows = coord_i[0] + t * (coord_j[0] - coord_i[0])
    center_cols = coord_i[1] + t * (coord_j[1] - coord_i[1])

    # Direction vector and perpendicular normal
    dr = coord_j[0] - coord_i[0]
    dc = coord_j[1] - coord_i[1]
    length = torch.sqrt(dr ** 2 + dc ** 2).clamp(min=1e-6)
    # Normal: (-dc, dr) / length (perpendicular to the line)
    nr = -dc / length
    nc = dr / length

    # Build corridor: for each sample point, sample across perpendicular
    perp_offsets = torch.linspace(-corridor_width, corridor_width, 2 * num_perp + 1, device=device)
    # (num_samples, 2*num_perp+1) grid
    all_rows = center_rows.unsqueeze(1) + nr * perp_offsets.unsqueeze(0)
    all_cols = center_cols.unsqueeze(1) + nc * perp_offsets.unsqueeze(0)

    total_pts = num_samples * (2 * num_perp + 1)
    grid_x = 2.0 * all_cols.reshape(-1) / (W - 1) - 1.0
    grid_y = 2.0 * all_rows.reshape(-1) / (H - 1) - 1.0
    grid = torch.stack([grid_x, grid_y], dim=-1).view(1, 1, total_pts, 2)

    sampled = F.grid_sample(
        dt_map.float(), grid.float(), mode="bilinear", padding_mode="zeros", align_corners=True
    ).view(num_samples, 2 * num_perp + 1)

    # Take max across perpendicular direction at each position
    corridor_max, _ = sampled.max(dim=1)  # (num_samples,)

    q_mean = corridor_max.mean()
    q_min = corridor_max.min()
    q_frac = (corridor_max > 0.3).float().mean()

    return q_mean, q_min, q_frac


class EdgeClassifier(nn.Module):
    """Predict pairwise adjacency from node features + path evidence.

    Symmetric feature construction ensures p(i,j) = p(j,i).
    """

    def __init__(self, fpn_dim: int = 256) -> None:
        super().__init__()
        # Feature dim: fpn_dim*3 (sum, abs-diff, hadamard)
        #   + 4 (node metadata) + 5 (geometry) + 3 (path evidence)
        input_dim = fpn_dim * 3 + 4 + 5 + 3
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.GroupNorm(16, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
        )

    def forward(
        self,
        fpn_features: torch.Tensor,
        dt_map: torch.Tensor,
        node_coords: torch.Tensor,
        node_types: torch.Tensor,
        node_scores: torch.Tensor,
        candidate_pairs: torch.Tensor,
    ) -> torch.Tensor:
        """Predict edge logits for candidate pairs.

        Args:
            fpn_features: (1, C, H, W) single-image FPN features at 128x128.
            dt_map: (1, 1, H, W) normalized DT field at 128x128 (sigmoid output).
            node_coords: (N, 2) node positions [row, col] in 128-space.
            node_types: (N,) 0=endpoint, 1=junction.
            node_scores: (N,) detection confidence.
            candidate_pairs: (E, 2) index pairs into node_coords.

        Returns:
            (E,) edge logits (raw, before sigmoid).
        """
        if len(candidate_pairs) == 0:
            return torch.empty(0, device=fpn_features.device)

        N = len(node_coords)
        C = fpn_features.shape[1]
        H, W = fpn_features.shape[2], fpn_features.shape[3]

        # Extract per-node features via grid_sample
        # Convert [row, col] to grid_sample format
        grid_x = 2.0 * node_coords[:, 1] / (W - 1) - 1.0
        grid_y = 2.0 * node_coords[:, 0] / (H - 1) - 1.0
        grid = torch.stack([grid_x, grid_y], dim=-1).view(1, 1, N, 2)
        node_feats = F.grid_sample(
            fpn_features, grid, mode="bilinear",
            padding_mode="zeros", align_corners=True,
        ).view(C, N).t()  # (N, C)

        idx_i = candidate_pairs[:, 0]
        idx_j = candidate_pairs[:, 1]
        feat_i = node_feats[idx_i]  # (E, C)
        feat_j = node_feats[idx_j]  # (E, C)

        # Symmetric visual features
        feat_sum = feat_i + feat_j
        feat_diff = (feat_i - feat_j).abs()
        feat_prod = feat_i * feat_j

        # Node metadata (symmetric)
        type_i = node_types[idx_i].float().unsqueeze(1)
        type_j = node_types[idx_j].float().unsqueeze(1)
        score_i = node_scores[idx_i].unsqueeze(1)
        score_j = node_scores[idx_j].unsqueeze(1)

        type_sum = type_i + type_j
        type_diff = (type_i - type_j).abs()
        score_sum = score_i + score_j
        score_min = torch.min(score_i, score_j)

        # Geometry (symmetric)
        coord_i = node_coords[idx_i]  # (E, 2)
        coord_j = node_coords[idx_j]  # (E, 2)
        dx = (coord_i[:, 1] - coord_j[:, 1]).abs().unsqueeze(1) / H
        dy = (coord_i[:, 0] - coord_j[:, 0]).abs().unsqueeze(1) / H
        dist = torch.sqrt(
            (coord_i[:, 0] - coord_j[:, 0]) ** 2 +
            (coord_i[:, 1] - coord_j[:, 1]) ** 2
        ).unsqueeze(1) / H
        mid_x = ((coord_i[:, 1] + coord_j[:, 1]) / 2).unsqueeze(1) / H
        mid_y = ((coord_i[:, 0] + coord_j[:, 0]) / 2).unsqueeze(1) / H

        # Path evidence from normalized DT field
        E = len(candidate_pairs)
        q_means = torch.zeros(E, 1, device=fpn_features.device)
        q_mins = torch.zeros(E, 1, device=fpn_features.device)
        q_fracs = torch.zeros(E, 1, device=fpn_features.device)

        for e in range(E):
            q_mean, q_min, q_frac = _sample_path_evidence(
                dt_map, coord_i[e], coord_j[e],
            )
            q_means[e, 0] = q_mean
            q_mins[e, 0] = q_min
            q_fracs[e, 0] = q_frac

        # Concatenate all features
        pair_feat = torch.cat([
            feat_sum, feat_diff, feat_prod,
            type_sum, type_diff, score_sum, score_min,
            dx, dy, dist, mid_x, mid_y,
            q_means, q_mins, q_fracs,
        ], dim=1)

        return self.mlp(pair_feat).squeeze(1)


def build_candidate_pairs(
    node_coords: torch.Tensor,
    k: int = 8,
    gt_edges: list[tuple[int, int]] | None = None,
    pred_to_gt: dict[int, int] | None = None,
) -> torch.Tensor:
    """Build candidate edge pairs using KNN + GT guarantee.

    Args:
        node_coords: (N, 2) positions.
        k: number of nearest neighbors per node.
        gt_edges: GT edges (indices into GT nodes) — included if pred_to_gt provided.
        pred_to_gt: mapping from pred node index to GT node index.

    Returns:
        (E, 2) candidate index pairs (undirected, no duplicates).
    """
    N = len(node_coords)
    if N < 2:
        return torch.empty((0, 2), dtype=torch.long, device=node_coords.device)

    # KNN candidates
    dists = torch.cdist(node_coords.unsqueeze(0), node_coords.unsqueeze(0))[0]
    dists.fill_diagonal_(float("inf"))
    k_actual = min(k, N - 1)
    _, knn_idx = dists.topk(k_actual, dim=1, largest=False)

    pairs_set: set[tuple[int, int]] = set()
    for i in range(N):
        for j in knn_idx[i].tolist():
            pairs_set.add((min(i, j), max(i, j)))

    # GT guarantee: include all edges whose endpoints are both matched
    if gt_edges is not None and pred_to_gt is not None:
        gt_to_pred = {v: k for k, v in pred_to_gt.items()}
        for ga, gb in gt_edges:
            if ga in gt_to_pred and gb in gt_to_pred:
                pa, pb = gt_to_pred[ga], gt_to_pred[gb]
                pairs_set.add((min(pa, pb), max(pa, pb)))

    if not pairs_set:
        return torch.empty((0, 2), dtype=torch.long, device=node_coords.device)

    pairs = torch.tensor(sorted(pairs_set), dtype=torch.long, device=node_coords.device)
    return pairs


def recover_edge_paths(
    dt_map: np.ndarray,
    node_coords_512: np.ndarray,
    edges: list[tuple[int, int]],
    cost_offset: float = 0.01,
    bg_penalty: float = 5.0,
) -> list[np.ndarray]:
    """Recover edge polylines via Dijkstra on learned cost field (P3c).

    This is hybrid reconstruction: graph topology comes from the direct
    decoder (EdgeClassifier), while edge geometry is recovered from the
    learned DT field via shortest-path routing.

    Cost function: C(p) = -log(DT(p) + eps) + bg_penalty * (1 - DT(p))
    This penalizes background pixels (DT~0) much more heavily than the
    simple 1-DT formulation, producing tighter path adherence to cracks.

    Args:
        dt_map: (H, W) normalized DT field in [0,1] at 512x512.
        node_coords_512: (N, 2) node positions [row, col] at 512-space.
        edges: list of (i, j) node index pairs.
        cost_offset: small epsilon for log stability.
        bg_penalty: penalty weight for background pixels (DT near 0).

    Returns:
        List of (L, 2) path arrays in 512-space, one per edge.
    """
    import heapq

    H, W = dt_map.shape
    # Improved cost: log-based + linear background penalty
    eps = cost_offset
    cost_map = -np.log(dt_map + eps) + bg_penalty * (1.0 - dt_map)

    neighbors = [(-1, -1), (-1, 0), (-1, 1),
                 (0, -1),           (0, 1),
                 (1, -1),  (1, 0),  (1, 1)]
    diag_cost = np.sqrt(2)

    edge_paths = []
    for src_idx, dst_idx in edges:
        sr, sc = int(node_coords_512[src_idx, 0]), int(node_coords_512[src_idx, 1])
        dr, dc = int(node_coords_512[dst_idx, 0]), int(node_coords_512[dst_idx, 1])

        sr = max(0, min(sr, H - 1))
        sc = max(0, min(sc, W - 1))
        dr = max(0, min(dr, H - 1))
        dc = max(0, min(dc, W - 1))

        # Dijkstra with bounded search region
        margin = 30
        r_min = max(0, min(sr, dr) - margin)
        r_max = min(H, max(sr, dr) + margin + 1)
        c_min = max(0, min(sc, dc) - margin)
        c_max = min(W, max(sc, dc) + margin + 1)

        dist_map = np.full((H, W), np.inf, dtype=np.float64)
        dist_map[sr, sc] = 0.0
        prev = {}
        heap = [(0.0, sr, sc)]

        found = False
        while heap:
            d, r, c = heapq.heappop(heap)
            if r == dr and c == dc:
                found = True
                break
            if d > dist_map[r, c]:
                continue
            for nr_off, nc_off in neighbors:
                nr, nc = r + nr_off, c + nc_off
                if not (r_min <= nr < r_max and c_min <= nc < c_max):
                    continue
                step = diag_cost if (nr_off != 0 and nc_off != 0) else 1.0
                new_dist = d + cost_map[nr, nc] * step
                if new_dist < dist_map[nr, nc]:
                    dist_map[nr, nc] = new_dist
                    prev[(nr, nc)] = (r, c)
                    heapq.heappush(heap, (new_dist, nr, nc))

        if found:
            # Trace back path
            path = [(dr, dc)]
            cur = (dr, dc)
            while cur != (sr, sc) and cur in prev:
                cur = prev[cur]
                path.append(cur)
            path.reverse()
            edge_paths.append(np.array(path, dtype=np.int64))
        else:
            # Fallback: straight line
            n_pts = max(int(np.sqrt((sr - dr) ** 2 + (sc - dc) ** 2)), 2)
            rows = np.linspace(sr, dr, n_pts).astype(np.int64)
            cols = np.linspace(sc, dc, n_pts).astype(np.int64)
            edge_paths.append(np.stack([rows, cols], axis=1))

    return edge_paths


def audit_candidate_recall(
    pred_coords: torch.Tensor,
    pred_to_gt: dict[int, int],
    gt_edges: list[tuple[int, int]],
    candidate_pairs: torch.Tensor,
) -> dict[str, float]:
    """Measure how many GT edges are recoverable from candidate pairs.

    Must be run BEFORE training to verify that KNN k is sufficient.
    If recall < 0.95, increase k or add radius-based candidates.

    Args:
        pred_coords: (N, 2) predicted node positions.
        pred_to_gt: mapping from pred node index -> GT node index.
        gt_edges: GT edges as (gt_idx_a, gt_idx_b) pairs.
        candidate_pairs: (E, 2) candidate index pairs into pred_coords.

    Returns:
        Dict with 'recall', 'n_gt_recoverable', 'n_gt_total', 'n_candidates'.
    """
    gt_to_pred = {v: k for k, v in pred_to_gt.items()}

    # GT edges where both endpoints are matched to predicted nodes
    recoverable = 0
    recovered = 0
    candidate_set = set()
    for e in range(len(candidate_pairs)):
        a, b = candidate_pairs[e].tolist()
        candidate_set.add((min(a, b), max(a, b)))

    for ga, gb in gt_edges:
        if ga in gt_to_pred and gb in gt_to_pred:
            recoverable += 1
            pa, pb = gt_to_pred[ga], gt_to_pred[gb]
            if (min(pa, pb), max(pa, pb)) in candidate_set:
                recovered += 1

    recall = recovered / max(recoverable, 1)
    return {
        "recall": recall,
        "n_gt_recoverable": recoverable,
        "n_gt_recovered": recovered,
        "n_gt_total": len(gt_edges),
        "n_candidates": len(candidate_pairs),
    }
