"""Loss functions for P3 graph prediction.

NodeHeatmapLoss: CenterNet-style modified focal loss for heatmaps.
EdgeBCELoss: focal BCE with hard negative mining for edge classification.
build_edge_labels: Hungarian-matched three-state edge label assignment.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

from morphograph.losses.composite import LossSchedule


class NodeHeatmapLoss(nn.Module):
    """CenterNet-style modified focal loss for node heatmaps.

    Positive pixels: -(1-p)^alpha * log(p)
    Negative pixels: -(1-Y)^beta * p^alpha * log(1-p)
    where Y = Gaussian GT heatmap value.

    Handles ~10-30 positive pixels per 16K map naturally.
    """

    def __init__(self, alpha: float = 2.0, beta: float = 4.0) -> None:
        super().__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(
        self,
        pred_logits: torch.Tensor,
        target_heatmap: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            pred_logits: (B, 2, H, W) raw logits.
            target_heatmap: (B, 2, H, W) Gaussian heatmaps in [0, 1].

        Returns:
            Scalar loss.
        """
        pred = torch.sigmoid(pred_logits)
        pred = pred.clamp(1e-6, 1.0 - 1e-6)

        pos_mask = target_heatmap.ge(0.999)
        neg_mask = ~pos_mask

        # Positive loss
        pos_loss = -((1.0 - pred) ** self.alpha) * torch.log(pred)
        pos_loss = pos_loss * pos_mask.float()

        # Negative loss with Gaussian-weighted penalty reduction
        neg_loss = (
            -((1.0 - target_heatmap) ** self.beta)
            * (pred ** self.alpha)
            * torch.log(1.0 - pred)
        )
        neg_loss = neg_loss * neg_mask.float()

        num_pos = pos_mask.float().sum()
        if num_pos == 0:
            return neg_loss.sum() / max(pred.numel(), 1)

        return (pos_loss.sum() + neg_loss.sum()) / num_pos


class EdgeBCELoss(nn.Module):
    """Focal BCE loss for edge classification with dynamic pos_weight.

    Handles variable positive/negative ratios per batch.
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: float = 0.25,
        max_pos_weight: float = 10.0,
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.max_pos_weight = max_pos_weight

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        loss_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            logits: (E,) edge logits.
            labels: (E,) binary labels (0 or 1).
            loss_weight: (E,) per-edge loss weights.
                0 = ignore, 1 = full weight, intermediate = scaled.
                If None, all edges weighted equally.

        Returns:
            Scalar loss.
        """
        if len(logits) == 0:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        if loss_weight is not None:
            active = loss_weight > 0
            if not active.any():
                return torch.tensor(0.0, device=logits.device, requires_grad=True)
            logits = logits[active]
            labels = labels[active]
            per_edge_w = loss_weight[active]
        else:
            per_edge_w = torch.ones_like(logits)

        # Dynamic pos_weight
        num_pos = labels.sum()
        num_neg = len(labels) - num_pos
        if num_pos > 0:
            pos_weight = min(num_neg / num_pos, self.max_pos_weight)
        else:
            pos_weight = 1.0

        # Focal BCE
        bce = F.binary_cross_entropy_with_logits(
            logits, labels.float(), reduction="none",
        )
        p = torch.sigmoid(logits)
        pt = p * labels + (1 - p) * (1 - labels)
        focal_weight = (1 - pt) ** self.gamma

        # Alpha weighting
        alpha_t = self.alpha * labels + (1 - self.alpha) * (1 - labels)

        # Apply pos_weight to positive samples
        sample_weight = torch.where(
            labels.bool(),
            torch.tensor(pos_weight, device=logits.device),
            torch.ones(1, device=logits.device),
        )

        loss = alpha_t * focal_weight * bce * sample_weight * per_edge_w
        return loss.sum() / per_edge_w.sum().clamp(min=1.0)


def build_edge_labels(
    pred_coords: torch.Tensor,
    pred_types: torch.Tensor,
    gt_coords: torch.Tensor,
    gt_types: torch.Tensor,
    gt_edges: list[tuple[int, int]],
    candidate_pairs: torch.Tensor,
    max_match_dist: float = 2.5,
    type_match_strict: bool = True,
    unmatched_far_threshold: float = 5.0,
    unmatched_neg_weight: float = 0.3,
) -> tuple[torch.Tensor, torch.Tensor, dict[int, int]]:
    """Assign edge labels via type-aware Hungarian matching.

    All coordinates are in 128-space. max_match_dist=2.5 corresponds
    to 10px in 512-space (2.5 * 4 = 10), matching the relaxed eval
    tolerance. For stricter training, use 1.25 (= 5px in 512-space).

    Three-state label assignment:
      - Both endpoints matched: label from GT adjacency (pos/neg), weight=1.0
      - Clearly unmatched (min dist to GT > unmatched_far_threshold):
        negative edge, weight=unmatched_neg_weight
      - Near matching threshold (ambiguous): IGNORE (weight=0.0)

    Returns:
        labels: (E,) binary labels (0/1).
        loss_weight: (E,) per-edge loss weights (0=ignore, 0.3=unmatched neg, 1.0=matched).
        pred_to_gt: dict mapping pred node idx -> GT node idx.
    """
    device = pred_coords.device
    E = len(candidate_pairs)
    labels = torch.zeros(E, device=device)
    loss_weight = torch.zeros(E, device=device)
    pred_to_gt: dict[int, int] = {}

    if len(pred_coords) == 0 or len(gt_coords) == 0:
        return labels, loss_weight, pred_to_gt

    pc = pred_coords.detach().cpu().numpy()
    gc = gt_coords.detach().cpu().numpy()
    pt = pred_types.detach().cpu().numpy()
    gt = gt_types.detach().cpu().numpy()

    dists = cdist(pc, gc)

    if type_match_strict:
        type_mismatch = (pt[:, None].astype(int) != gt[None, :].astype(int))
        cost = dists.copy()
        cost[type_mismatch] = 1e6
    else:
        type_mismatch = np.abs(pt[:, None].astype(float) - gt[None, :].astype(float))
        cost = dists + 5.0 * type_mismatch

    row_ind, col_ind = linear_sum_assignment(cost)
    for r, c in zip(row_ind, col_ind):
        if dists[r, c] <= max_match_dist:
            pred_to_gt[int(r)] = int(c)

    min_dist_to_gt = dists.min(axis=1)
    is_clearly_unmatched = {}
    for i in range(len(pc)):
        if i not in pred_to_gt:
            is_clearly_unmatched[i] = min_dist_to_gt[i] > unmatched_far_threshold

    gt_edge_set = set()
    for a, b in gt_edges:
        gt_edge_set.add((min(a, b), max(a, b)))

    for e_idx in range(E):
        a, b = candidate_pairs[e_idx].tolist()
        a_matched = a in pred_to_gt
        b_matched = b in pred_to_gt

        if a_matched and b_matched:
            loss_weight[e_idx] = 1.0
            mapped = (
                min(pred_to_gt[a], pred_to_gt[b]),
                max(pred_to_gt[a], pred_to_gt[b]),
            )
            if mapped in gt_edge_set:
                labels[e_idx] = 1.0
        elif (not a_matched and is_clearly_unmatched.get(a, False)) or \
             (not b_matched and is_clearly_unmatched.get(b, False)):
            labels[e_idx] = 0.0
            loss_weight[e_idx] = unmatched_neg_weight
        else:
            loss_weight[e_idx] = 0.0

    return labels, loss_weight, pred_to_gt


@dataclass
class P3aLossConfig:
    """P3a: B2 + node heatmap loss."""
    ce_weight: float = 0.5
    dice_weight: float = 0.5
    ce_class_weights: list[float] = field(
        default_factory=lambda: [0.2, 2.0, 3.0]
    )
    ignore_index: int = 255
    skeleton_weight: float = 10.0
    node_heatmap: LossSchedule = field(default_factory=lambda: LossSchedule(
        weight=1.0, start_epoch=0, ramp_epochs=5,
    ))


@dataclass
class P3bLossConfig(P3aLossConfig):
    """P3b: P3a + edge loss with scheduled sampling."""
    edge: LossSchedule = field(default_factory=lambda: LossSchedule(
        weight=1.0, start_epoch=10, ramp_epochs=10,
    ))
    ss_warmup_epoch: int = 10
    ss_anneal_end_epoch: int = 60


def scheduled_sampling_prob(
    epoch: int,
    warmup: int = 10,
    anneal_end: int = 60,
) -> float:
    """Probability of using GT nodes (teacher forcing).

    1.0 if epoch < warmup, linearly anneals to 0.0 at anneal_end.
    """
    if epoch < warmup:
        return 1.0
    if epoch >= anneal_end:
        return 0.0
    return max(0.0, 1.0 - (epoch - warmup) / (anneal_end - warmup))
