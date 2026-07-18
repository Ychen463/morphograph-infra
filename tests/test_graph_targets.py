"""Tests for crack mask to graph conversion pipeline."""

import numpy as np
import pytest

from morphograph.data.graph_targets import (
    detect_keypoints,
    mask_to_skeleton,
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
