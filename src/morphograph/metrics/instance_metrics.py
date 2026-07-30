"""Spalling instance metrics via connected-component analysis.

Given a semantic segmentation mask (pred and GT), extract spalling
instances as connected components and compute instance-level metrics:
  - Instance P/R/F1: bipartite matching by IoU
  - Mean matched IoU: average IoU of matched pairs
  - Count error: |n_pred - n_gt|
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage
from scipy.optimize import linear_sum_assignment


@dataclass
class InstanceMetrics:
    """Per-image spalling instance metrics."""
    n_gt: int = 0
    n_pred: int = 0
    tp: int = 0
    fp: int = 0
    fn: int = 0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    mean_matched_iou: float = 0.0


def extract_instances(
    mask: np.ndarray,
    class_id: int = 2,
    min_area_px: int = 25,
) -> tuple[np.ndarray, int]:
    """Extract connected components for a given class.

    Args:
        mask: HxW semantic mask with class IDs.
        class_id: which class to extract instances for.
        min_area_px: minimum component area to keep (filters noise).

    Returns:
        (label_map, n_instances): labeled array and count.
    """
    binary = (mask == class_id).astype(np.uint8)
    labels, n = ndimage.label(binary)
    if min_area_px > 0 and n > 0:
        # Filter small components
        areas = ndimage.sum(binary, labels, range(1, n + 1))
        for i, area in enumerate(areas, start=1):
            if area < min_area_px:
                labels[labels == i] = 0
        # Re-label to make IDs contiguous
        labels, n = ndimage.label(labels > 0)
    return labels, n


def compute_instance_metrics(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    class_id: int = 2,
    iou_threshold: float = 0.5,
    min_area_px: int = 25,
) -> InstanceMetrics:
    """Compute instance-level P/R/F1 for a single image.

    Instances are extracted via connected components. Matching uses
    Hungarian algorithm on IoU cost matrix with a threshold.

    Args:
        pred_mask: HxW predicted semantic labels.
        gt_mask: HxW ground-truth semantic labels.
        class_id: class to evaluate (default 2 = spalling).
        iou_threshold: minimum IoU for a valid match.
        min_area_px: minimum component area.

    Returns:
        InstanceMetrics for this image.
    """
    gt_labels, n_gt = extract_instances(gt_mask, class_id, min_area_px)
    pred_labels, n_pred = extract_instances(pred_mask, class_id, min_area_px)

    if n_gt == 0 and n_pred == 0:
        return InstanceMetrics(precision=1.0, recall=1.0, f1=1.0, mean_matched_iou=1.0)
    if n_gt == 0:
        return InstanceMetrics(n_pred=n_pred, fp=n_pred)
    if n_pred == 0:
        return InstanceMetrics(n_gt=n_gt, fn=n_gt)

    # Build IoU matrix: (n_gt, n_pred)
    iou_matrix = np.zeros((n_gt, n_pred), dtype=np.float64)
    for gi in range(1, n_gt + 1):
        gt_comp = gt_labels == gi
        for pi in range(1, n_pred + 1):
            pred_comp = pred_labels == pi
            inter = (gt_comp & pred_comp).sum()
            union = (gt_comp | pred_comp).sum()
            if union > 0:
                iou_matrix[gi - 1, pi - 1] = inter / union

    # Hungarian matching (maximize IoU = minimize -IoU)
    row_ind, col_ind = linear_sum_assignment(-iou_matrix)

    matched_ious = []
    tp = 0
    for r, c in zip(row_ind, col_ind):
        if iou_matrix[r, c] >= iou_threshold:
            tp += 1
            matched_ious.append(iou_matrix[r, c])

    fp = n_pred - tp
    fn = n_gt - tp
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    mean_iou = float(np.mean(matched_ious)) if matched_ious else 0.0

    return InstanceMetrics(
        n_gt=n_gt, n_pred=n_pred,
        tp=tp, fp=fp, fn=fn,
        precision=precision, recall=recall, f1=f1,
        mean_matched_iou=mean_iou,
    )
