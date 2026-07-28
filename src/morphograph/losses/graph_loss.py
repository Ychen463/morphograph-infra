"""Loss functions for P3 graph prediction.

NodeHeatmapLoss: CenterNet-style modified focal loss for heatmaps.
EdgeBCELoss: focal BCE with hard negative mining for edge classification.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

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

        pos_mask = target_heatmap.eq(1.0)
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
