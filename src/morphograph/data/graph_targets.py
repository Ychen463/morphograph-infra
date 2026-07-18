"""Crack mask to morphological graph conversion pipeline.

Pipeline stages:
    binary mask -> skeleton -> endpoint/junction detection -> graph construction

The skeleton and keypoint detection stages are implemented;
graph construction is stubbed for now.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage
from skimage.morphology import skeletonize


@dataclass
class CrackGraph:
    """Graph representation of crack morphology in a single image."""

    endpoints: np.ndarray  # (N, 2) array of (row, col) endpoint coordinates
    junctions: np.ndarray  # (M, 2) array of (row, col) junction coordinates
    edges: list[tuple[int, int]] = field(default_factory=list)  # adjacency list
    width_at_nodes: np.ndarray | None = None  # per-node width estimate


def mask_to_skeleton(binary_mask: np.ndarray) -> np.ndarray:
    """Skeletonize a binary crack mask.

    Args:
        binary_mask: HxW boolean or uint8 array (nonzero = crack).

    Returns:
        HxW boolean skeleton.
    """
    return skeletonize(binary_mask.astype(bool))


def detect_keypoints(skeleton: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Detect endpoints and junctions on a skeleton image.

    An endpoint has exactly 1 neighbor in the 8-connected skeleton.
    A junction has 3 or more neighbors.

    Args:
        skeleton: HxW boolean skeleton image.

    Returns:
        (endpoints, junctions) each as (N, 2) arrays of (row, col).
    """
    kernel = np.ones((3, 3), dtype=np.uint8)
    kernel[1, 1] = 0
    neighbor_count = ndimage.convolve(
        skeleton.astype(np.uint8), kernel, mode="constant", cval=0
    )

    # Only count neighbors at skeleton pixels
    neighbor_count = neighbor_count * skeleton.astype(np.uint8)

    endpoint_mask = (neighbor_count == 1) & skeleton
    junction_mask = (neighbor_count >= 3) & skeleton

    endpoints = np.argwhere(endpoint_mask)
    junctions = np.argwhere(junction_mask)

    return endpoints, junctions


def build_graph(
    skeleton: np.ndarray,
    endpoints: np.ndarray,
    junctions: np.ndarray,
    min_branch_length: int = 10,
) -> CrackGraph:
    """Construct a CrackGraph from skeleton and detected keypoints.

    Traces skeleton branches between keypoints to build an adjacency graph.
    Short branches (< min_branch_length) are pruned.

    Args:
        skeleton: HxW boolean skeleton.
        endpoints: (N, 2) endpoint coordinates.
        junctions: (M, 2) junction coordinates.
        min_branch_length: minimum branch length in pixels to keep.

    Returns:
        CrackGraph with nodes, edges, and optional width estimates.
    """
    # TODO: implement branch tracing and graph construction
    raise NotImplementedError("Graph construction from skeleton not yet implemented")


def mask_to_graph(
    binary_mask: np.ndarray,
    min_branch_length: int = 10,
) -> CrackGraph:
    """Full pipeline: binary crack mask -> morphological graph.

    Args:
        binary_mask: HxW binary crack mask.
        min_branch_length: prune branches shorter than this.

    Returns:
        CrackGraph instance.
    """
    skeleton = mask_to_skeleton(binary_mask)
    endpoints, junctions = detect_keypoints(skeleton)
    return build_graph(skeleton, endpoints, junctions, min_branch_length)
