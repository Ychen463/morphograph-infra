"""Crack mask to morphological graph conversion pipeline.

Pipeline stages:
    binary mask -> skeleton -> endpoint/junction detection ->
    junction merging -> branch tracing -> spur pruning -> width estimation
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage
from scipy.spatial.distance import cdist
from skimage.morphology import skeletonize


@dataclass
class CrackGraph:
    """Graph representation of crack morphology in a single image."""

    endpoints: np.ndarray  # (N, 2) array of (row, col) endpoint coordinates
    junctions: np.ndarray  # (M, 2) array of (row, col) junction coordinates
    edges: list[tuple[int, int]] = field(default_factory=list)
    edge_paths: list[np.ndarray] = field(default_factory=list)  # pixel coords per edge
    width_at_nodes: np.ndarray | None = None  # per-node width estimate

    @property
    def num_nodes(self) -> int:
        return len(self.endpoints) + len(self.junctions)

    @property
    def num_edges(self) -> int:
        return len(self.edges)

    @property
    def all_nodes(self) -> np.ndarray:
        """All nodes (endpoints first, then junctions) as (K, 2) array."""
        parts = [p for p in (self.endpoints, self.junctions) if len(p) > 0]
        if not parts:
            return np.empty((0, 2), dtype=np.int64)
        return np.concatenate(parts, axis=0)


def mask_to_skeleton(binary_mask: np.ndarray) -> np.ndarray:
    """Skeletonize a binary crack mask.

    Args:
        binary_mask: HxW boolean or uint8 array (nonzero = crack).

    Returns:
        HxW boolean skeleton.
    """
    return skeletonize(binary_mask.astype(bool))


def _neighbor_count(skeleton: np.ndarray) -> np.ndarray:
    """Count 8-connected neighbors at each skeleton pixel."""
    kernel = np.ones((3, 3), dtype=np.uint8)
    kernel[1, 1] = 0
    counts = ndimage.convolve(
        skeleton.astype(np.uint8), kernel, mode="constant", cval=0
    )
    return counts * skeleton.astype(np.uint8)


def detect_keypoints(skeleton: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Detect endpoints and junctions on a skeleton image.

    An endpoint has exactly 1 neighbor in the 8-connected skeleton.
    A junction has 3 or more neighbors.

    Args:
        skeleton: HxW boolean skeleton image.

    Returns:
        (endpoints, junctions) each as (N, 2) arrays of (row, col).
    """
    counts = _neighbor_count(skeleton)

    endpoint_mask = (counts == 1) & skeleton
    junction_mask = (counts >= 3) & skeleton

    endpoints = np.argwhere(endpoint_mask)
    junctions = np.argwhere(junction_mask)

    return endpoints, junctions


def merge_junctions(
    junctions: np.ndarray,
    merge_radius: int = 5,
) -> np.ndarray:
    """Merge junction pixel clusters into single representative points.

    Multiple skeleton pixels near a branch point all have >=3 neighbors,
    forming a cluster. This merges clusters within merge_radius into
    their centroid.

    Args:
        junctions: (M, 2) junction coordinates.
        merge_radius: maximum distance to merge.

    Returns:
        (M', 2) merged junction coordinates (M' <= M).
    """
    if len(junctions) == 0:
        return junctions

    # Label connected clusters via distance threshold
    dists = cdist(junctions, junctions)
    visited = np.zeros(len(junctions), dtype=bool)
    merged = []

    for i in range(len(junctions)):
        if visited[i]:
            continue
        cluster = dists[i] <= merge_radius
        cluster &= ~visited
        visited |= cluster
        centroid = junctions[cluster].mean(axis=0).round().astype(int)
        merged.append(centroid)

    return np.array(merged) if merged else np.empty((0, 2), dtype=int)


def _trace_branches(
    skeleton: np.ndarray,
    keypoints: np.ndarray,
) -> list[tuple[int, int, np.ndarray]]:
    """Trace skeleton branches between keypoints.

    Walks along skeleton pixels from each keypoint, collecting the path
    until another keypoint or a dead end is reached.

    Args:
        skeleton: HxW boolean skeleton.
        keypoints: (K, 2) all keypoint coords (endpoints + junctions).

    Returns:
        List of (start_idx, end_idx, path_coords) tuples.
        Indices refer to positions in the keypoints array.
    """
    if len(keypoints) == 0 or not skeleton.any():
        return []

    h, w = skeleton.shape
    # Build a lookup from (row, col) -> keypoint index
    kp_map = {}
    for idx, (r, c) in enumerate(keypoints):
        kp_map[(int(r), int(c))] = idx

    # Track visited pixels to avoid tracing the same branch twice
    visited = np.zeros_like(skeleton, dtype=bool)
    branches: list[tuple[int, int, np.ndarray]] = []

    neighbors_offsets = [(-1, -1), (-1, 0), (-1, 1),
                         (0, -1),           (0, 1),
                         (1, -1),  (1, 0),  (1, 1)]

    for start_idx, (sr, sc) in enumerate(keypoints):
        sr, sc = int(sr), int(sc)
        # Try walking in each unvisited skeleton direction
        for dr, dc in neighbors_offsets:
            nr, nc = sr + dr, sc + dc
            if not (0 <= nr < h and 0 <= nc < w):
                continue
            if not skeleton[nr, nc] or visited[nr, nc]:
                continue

            # Trace this branch
            path = [(sr, sc), (nr, nc)]
            visited[nr, nc] = True
            cr, cc = nr, nc

            while True:
                # Check if we reached another keypoint
                if (cr, cc) in kp_map and (cr, cc) != (sr, sc):
                    end_idx = kp_map[(cr, cc)]
                    branches.append((
                        start_idx,
                        end_idx,
                        np.array(path, dtype=np.int64),
                    ))
                    break

                # Find next unvisited skeleton neighbor
                found_next = False
                for dr2, dc2 in neighbors_offsets:
                    nr2, nc2 = cr + dr2, cc + dc2
                    if not (0 <= nr2 < h and 0 <= nc2 < w):
                        continue
                    if not skeleton[nr2, nc2] or visited[nr2, nc2]:
                        # Allow reaching a keypoint even if "visited"
                        if (nr2, nc2) in kp_map and (nr2, nc2) != (sr, sc):
                            path.append((nr2, nc2))
                            end_idx = kp_map[(nr2, nc2)]
                            branches.append((
                                start_idx,
                                end_idx,
                                np.array(path, dtype=np.int64),
                            ))
                            found_next = True
                            break
                        continue
                    visited[nr2, nc2] = True
                    path.append((nr2, nc2))
                    cr, cc = nr2, nc2
                    found_next = True
                    break

                if not found_next:
                    break  # dead end (dangling spur without keypoint match)

    return branches


def estimate_width(
    binary_mask: np.ndarray,
    skeleton: np.ndarray,
) -> np.ndarray:
    """Estimate crack width at each skeleton pixel using distance transform.

    Width = 2 * distance_transform_value at the skeleton pixel, since the
    distance transform gives the radius to the nearest background pixel.

    Args:
        binary_mask: HxW binary crack mask.
        skeleton: HxW boolean skeleton.

    Returns:
        HxW float array with width values at skeleton pixels (0 elsewhere).
    """
    dt = ndimage.distance_transform_edt(binary_mask.astype(bool))
    width_map = np.zeros_like(dt)
    width_map[skeleton] = 2.0 * dt[skeleton]
    return width_map


def build_graph(
    skeleton: np.ndarray,
    endpoints: np.ndarray,
    junctions: np.ndarray,
    min_branch_length: int = 10,
    junction_merge_radius: int = 5,
    binary_mask: np.ndarray | None = None,
) -> CrackGraph:
    """Construct a CrackGraph from skeleton and detected keypoints.

    Steps:
        1. Merge junction clusters
        2. Trace branches between keypoints
        3. Prune short branches (spurs)
        4. Optionally estimate width at nodes

    Args:
        skeleton: HxW boolean skeleton.
        endpoints: (N, 2) endpoint coordinates.
        junctions: (M, 2) junction coordinates.
        min_branch_length: minimum branch length in pixels to keep.
        junction_merge_radius: radius for merging junction clusters.
        binary_mask: optional original mask for width estimation.

    Returns:
        CrackGraph with nodes, edges, and optional width estimates.
    """
    junctions = merge_junctions(junctions, junction_merge_radius)
    all_kps = np.concatenate([endpoints, junctions], axis=0) if len(endpoints) + len(junctions) > 0 else np.empty((0, 2), dtype=int)

    if len(all_kps) == 0:
        return CrackGraph(
            endpoints=np.empty((0, 2), dtype=int),
            junctions=np.empty((0, 2), dtype=int),
        )

    raw_branches = _trace_branches(skeleton, all_kps)

    # Prune short branches (spurs): short branches ending at an endpoint
    n_ep = len(endpoints)
    edges = []
    edge_paths = []
    for start, end, path in raw_branches:
        branch_len = len(path)
        is_spur = (start < n_ep or end < n_ep) and branch_len < min_branch_length
        if not is_spur:
            edges.append((start, end))
            edge_paths.append(path)

    # Deduplicate edges (a-b same as b-a)
    seen: set[tuple[int, int]] = set()
    unique_edges = []
    unique_paths = []
    for (a, b), path in zip(edges, edge_paths):
        key = (min(a, b), max(a, b))
        if key not in seen:
            seen.add(key)
            unique_edges.append(key)
            unique_paths.append(path)

    # Width estimation
    width_at_nodes = None
    if binary_mask is not None:
        width_map = estimate_width(binary_mask, skeleton)
        width_at_nodes = np.array(
            [width_map[int(r), int(c)] for r, c in all_kps]
        )

    return CrackGraph(
        endpoints=endpoints,
        junctions=junctions,
        edges=unique_edges,
        edge_paths=unique_paths,
        width_at_nodes=width_at_nodes,
    )


def mask_to_graph(
    binary_mask: np.ndarray,
    min_branch_length: int = 10,
    junction_merge_radius: int = 5,
) -> CrackGraph:
    """Full pipeline: binary crack mask -> morphological graph.

    Args:
        binary_mask: HxW binary crack mask.
        min_branch_length: prune branches shorter than this.
        junction_merge_radius: radius for merging junction clusters.

    Returns:
        CrackGraph instance with edges, paths, and width estimates.
    """
    skeleton = mask_to_skeleton(binary_mask)
    endpoints, junctions = detect_keypoints(skeleton)
    return build_graph(
        skeleton, endpoints, junctions,
        min_branch_length=min_branch_length,
        junction_merge_radius=junction_merge_radius,
        binary_mask=binary_mask,
    )
