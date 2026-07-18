"""Segmentation and topology evaluation metrics.

Metrics:
    IoU:    per-class intersection over union
    Dice:   per-class F1 score
    BF1:    boundary F1 with configurable pixel tolerance
    clDice: centerline Dice via morphological skeletonization
    ConnR:  connectivity recall (fraction of GT components preserved)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage
from skimage.morphology import skeletonize


@dataclass
class SegMetrics:
    """Container for per-image segmentation metrics."""
    iou_per_class: dict[int, float] = field(default_factory=dict)
    dice_per_class: dict[int, float] = field(default_factory=dict)
    pixel_acc: float = 0.0

    @property
    def miou_fg(self) -> float:
        """Mean IoU over foreground classes only (crack + spalling)."""
        fg = [v for k, v in self.iou_per_class.items() if k > 0]
        return float(np.mean(fg)) if fg else 0.0

    @property
    def miou_all(self) -> float:
        """Mean IoU over all classes including background."""
        vals = list(self.iou_per_class.values())
        return float(np.mean(vals)) if vals else 0.0


def compute_iou(
    pred: np.ndarray,
    target: np.ndarray,
    num_classes: int = 3,
    ignore_index: int = 255,
) -> dict[int, float]:
    """Per-class IoU.

    Args:
        pred: HxW predicted class labels.
        target: HxW ground-truth class labels.
        num_classes: number of classes (excluding ignore).
        ignore_index: label to exclude.

    Returns:
        Dict mapping class_id -> IoU.
    """
    valid = target != ignore_index
    result = {}
    for c in range(num_classes):
        pred_c = (pred == c) & valid
        target_c = (target == c) & valid
        intersection = (pred_c & target_c).sum()
        union = (pred_c | target_c).sum()
        result[c] = float(intersection / (union + 1e-8))
    return result


def compute_dice(
    pred: np.ndarray,
    target: np.ndarray,
    num_classes: int = 3,
    ignore_index: int = 255,
) -> dict[int, float]:
    """Per-class Dice coefficient."""
    valid = target != ignore_index
    result = {}
    for c in range(num_classes):
        pred_c = (pred == c) & valid
        target_c = (target == c) & valid
        intersection = (pred_c & target_c).sum()
        total = pred_c.sum() + target_c.sum()
        result[c] = float(2 * intersection / (total + 1e-8))
    return result


def compute_boundary_f1(
    pred: np.ndarray,
    target: np.ndarray,
    tolerance_px: int = 2,
    ignore_index: int = 255,
) -> dict[int, float]:
    """Boundary F1 score per foreground class.

    Extracts boundaries via morphological gradient, then computes
    precision/recall within a distance tolerance.

    Args:
        pred: HxW predicted class labels.
        target: HxW ground-truth class labels.
        tolerance_px: distance tolerance for boundary matching.
        ignore_index: label to exclude.

    Returns:
        Dict mapping class_id -> BF1 (foreground classes only).
    """
    valid = target != ignore_index
    struct = ndimage.generate_binary_structure(2, 1)
    result = {}

    for c in range(1, pred.max() + 1):  # foreground only
        pred_c = (pred == c) & valid
        target_c = (target == c) & valid

        pred_boundary = pred_c ^ ndimage.binary_erosion(pred_c, struct)
        target_boundary = target_c ^ ndimage.binary_erosion(target_c, struct)

        if not target_boundary.any() and not pred_boundary.any():
            result[c] = 1.0
            continue
        if not target_boundary.any() or not pred_boundary.any():
            result[c] = 0.0
            continue

        # Distance transform for matching
        dt_target = ndimage.distance_transform_edt(~target_boundary)
        dt_pred = ndimage.distance_transform_edt(~pred_boundary)

        precision = (dt_target[pred_boundary] <= tolerance_px).mean()
        recall = (dt_pred[target_boundary] <= tolerance_px).mean()

        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        result[c] = float(f1)

    return result


def compute_cldice(
    pred_binary: np.ndarray,
    target_binary: np.ndarray,
) -> float:
    """Centerline Dice (clDice) for a single binary class.

    clDice = 2 * |S_pred ∩ V_target| * |S_target ∩ V_pred| /
             (|S_pred| * |V_target| + |S_target| * |V_pred|)

    where S = skeleton, V = volume (binary mask).

    Args:
        pred_binary: HxW boolean predicted mask for one class.
        target_binary: HxW boolean GT mask for one class.

    Returns:
        clDice score in [0, 1].
    """
    if not target_binary.any():
        return 1.0 if not pred_binary.any() else 0.0
    if not pred_binary.any():
        return 0.0

    skel_pred = skeletonize(pred_binary)
    skel_target = skeletonize(target_binary)

    # Topology precision: skeleton of pred inside GT volume
    tprec = skel_pred & target_binary
    # Topology sensitivity: skeleton of GT inside pred volume
    tsens = skel_target & pred_binary

    tprec_sum = tprec.sum()
    tsens_sum = tsens.sum()
    skel_pred_sum = skel_pred.sum()
    skel_target_sum = skel_target.sum()

    if skel_pred_sum == 0 and skel_target_sum == 0:
        return 1.0

    precision = tprec_sum / (skel_pred_sum + 1e-8)
    recall = tsens_sum / (skel_target_sum + 1e-8)

    return float(2 * precision * recall / (precision + recall + 1e-8))


def compute_connectivity_recall(
    pred_binary: np.ndarray,
    target_binary: np.ndarray,
) -> float:
    """Connectivity recall (ConnR): fraction of GT connected components
    that are preserved (not split) in the prediction.

    A GT component is "preserved" if all its skeleton pixels fall within
    a single predicted connected component.

    Args:
        pred_binary: HxW boolean predicted mask.
        target_binary: HxW boolean GT mask.

    Returns:
        ConnR in [0, 1].
    """
    if not target_binary.any():
        return 1.0

    target_labels, n_target = ndimage.label(target_binary)
    if n_target == 0:
        return 1.0

    pred_labels, _ = ndimage.label(pred_binary)
    preserved = 0

    for comp_id in range(1, n_target + 1):
        comp_mask = target_labels == comp_id
        skel = skeletonize(comp_mask)
        if not skel.any():
            preserved += 1
            continue

        pred_ids_on_skel = pred_labels[skel]
        pred_ids_on_skel = pred_ids_on_skel[pred_ids_on_skel > 0]

        if len(pred_ids_on_skel) > 0 and len(set(pred_ids_on_skel)) == 1:
            preserved += 1

    return preserved / n_target


def compute_seg_metrics(
    pred: np.ndarray,
    target: np.ndarray,
    num_classes: int = 3,
    ignore_index: int = 255,
    boundary_tolerance: int = 2,
) -> SegMetrics:
    """Compute all segmentation metrics for a single image.

    Args:
        pred: HxW predicted class labels.
        target: HxW ground-truth class labels.
        num_classes: number of classes.
        ignore_index: label to exclude.
        boundary_tolerance: BF1 tolerance in pixels.

    Returns:
        SegMetrics with IoU, Dice, and pixel accuracy.
    """
    valid = target != ignore_index
    pixel_acc = float((pred[valid] == target[valid]).mean()) if valid.any() else 0.0

    return SegMetrics(
        iou_per_class=compute_iou(pred, target, num_classes, ignore_index),
        dice_per_class=compute_dice(pred, target, num_classes, ignore_index),
        pixel_acc=pixel_acc,
    )
