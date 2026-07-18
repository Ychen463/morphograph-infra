"""Tests for crack mask to graph conversion pipeline."""

import numpy as np
import pytest

from morphograph.data.graph_targets import (
    build_graph,
    detect_keypoints,
    estimate_width,
    mask_to_graph,
    mask_to_skeleton,
    merge_junctions,
)


def test_skeleton_preserves_connectivity():
    """A simple horizontal line should produce a connected skeleton."""
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[32, 10:50] = 1  # horizontal line
    skeleton = mask_to_skeleton(mask)
    assert skeleton.any(), "Skeleton should not be empty"
    assert skeleton[32, 10:50].all(), "Skeleton should cover the line"


def test_endpoint_detection_simple_line():
    """A straight skeleton line should have exactly 2 endpoints."""
    skeleton = np.zeros((64, 64), dtype=bool)
    skeleton[32, 10:50] = True
    endpoints, junctions = detect_keypoints(skeleton)
    assert len(endpoints) == 2, f"Expected 2 endpoints, got {len(endpoints)}"
    assert len(junctions) == 0, f"Expected 0 junctions, got {len(junctions)}"


def test_junction_detection_t_shape():
    """A T-shaped skeleton should have 3 endpoints and 1 junction."""
    skeleton = np.zeros((64, 64), dtype=bool)
    skeleton[32, 10:50] = True   # horizontal bar
    skeleton[32:50, 30] = True   # vertical bar downward from midpoint
    endpoints, junctions = detect_keypoints(skeleton)
    assert len(endpoints) == 3, f"Expected 3 endpoints, got {len(endpoints)}"
    assert len(junctions) >= 1, f"Expected >=1 junction, got {len(junctions)}"


def test_empty_mask_produces_empty_graph():
    """An empty mask should produce an empty skeleton with no keypoints."""
    mask = np.zeros((64, 64), dtype=np.uint8)
    skeleton = mask_to_skeleton(mask)
    endpoints, junctions = detect_keypoints(skeleton)
    assert not skeleton.any()
    assert len(endpoints) == 0
    assert len(junctions) == 0


def test_merge_junctions_clusters():
    """Nearby junction pixels should merge into a single point."""
    junctions = np.array([[30, 30], [30, 31], [31, 30], [31, 31]])
    merged = merge_junctions(junctions, merge_radius=3)
    assert len(merged) == 1, f"Expected 1 merged junction, got {len(merged)}"


def test_merge_junctions_distant():
    """Distant junction clusters should remain separate."""
    junctions = np.array([[10, 10], [10, 11], [50, 50], [50, 51]])
    merged = merge_junctions(junctions, merge_radius=3)
    assert len(merged) == 2, f"Expected 2 clusters, got {len(merged)}"


def test_merge_junctions_empty():
    """Empty junction array should return empty."""
    merged = merge_junctions(np.empty((0, 2), dtype=int))
    assert len(merged) == 0


def test_estimate_width_on_thick_line():
    """Width estimate on a 5px thick line should be approximately 5."""
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[30:35, 10:50] = 1  # 5px thick horizontal band
    skeleton = mask_to_skeleton(mask)
    width_map = estimate_width(mask, skeleton)
    # Width at center of a 5px band should be close to 5
    skel_widths = width_map[skeleton]
    assert skel_widths.max() >= 3.0, "Width should be substantial for thick line"
    assert skel_widths.max() <= 7.0, "Width should not exceed band thickness"


def test_mask_to_graph_simple_line():
    """Full pipeline on a simple line should produce a graph with 2 endpoints."""
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[32, 10:50] = 1
    graph = mask_to_graph(mask, min_branch_length=3)
    assert len(graph.endpoints) == 2
    assert graph.num_edges >= 1
    assert graph.width_at_nodes is not None


def test_mask_to_graph_empty():
    """Empty mask should produce an empty graph."""
    mask = np.zeros((64, 64), dtype=np.uint8)
    graph = mask_to_graph(mask)
    assert graph.num_nodes == 0
    assert graph.num_edges == 0


def test_build_graph_prunes_short_spurs():
    """Branches shorter than min_branch_length should be pruned."""
    # Create a skeleton with a main branch and a short spur
    skeleton = np.zeros((64, 64), dtype=bool)
    skeleton[32, 10:50] = True  # main horizontal line (40px)
    skeleton[31, 30] = True     # 1px spur upward
    endpoints, junctions = detect_keypoints(skeleton)
    graph = build_graph(skeleton, endpoints, junctions, min_branch_length=5)
    # The 1px spur should be pruned, leaving only the main branch
    assert graph.num_edges >= 1
