"""Multi-task loss functions for the baseline ladder.

Each baseline configuration (B0-B5) has a specific loss set.
Losses must NOT be mixed across baselines — otherwise the
ablation cannot attribute improvement to a specific supervision.

Loss assignments per baseline:
    B0:   CE + Dice
    B1a:  CE + Dice + clDice (scheduled)
    B1b:  CE + Dice + SRL
    B2:   CE + Dice + skeleton BCE+Dice
    B3:   B2 + endpoint BCE + junction BCE
    B4:   B3 + edge connectivity loss
    B5:   B4 + width regression loss

Tversky and boundary losses are ablation variants, not defaults.

All heads output raw logits. Binary heads use BCEWithLogitsLoss.
clDice soft-skeletonization must run in float32 (fp16 unstable).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Schedule
# ---------------------------------------------------------------------------

@dataclass
class LossSchedule:
    """Controls when an auxiliary loss term activates."""
    weight: float = 1.0
    start_epoch: int = 0
    ramp_epochs: int = 0

    def effective_weight(self, current_epoch: int) -> float:
        if current_epoch < self.start_epoch:
            return 0.0
        elapsed = current_epoch - self.start_epoch
        if self.ramp_epochs > 0 and elapsed < self.ramp_epochs:
            return self.weight * (elapsed / self.ramp_epochs)
        return self.weight


# ---------------------------------------------------------------------------
# Per-baseline loss configs
# ---------------------------------------------------------------------------

@dataclass
class B0LossConfig:
    """B0: mask-only baseline."""
    ce_weight: float = 0.5
    dice_weight: float = 0.5
    ce_class_weights: list[float] = field(
        default_factory=lambda: [0.2, 2.0, 3.0]
    )
    ignore_index: int = 255


@dataclass
class B1aLossConfig(B0LossConfig):
    """B1a: B0 + clDice (scheduled)."""
    cldice: LossSchedule = field(default_factory=lambda: LossSchedule(
        weight=0.15, start_epoch=40, ramp_epochs=5,
    ))


@dataclass
class B2LossConfig(B0LossConfig):
    """B2: B0 + skeleton supervision."""
    skeleton_weight: float = 1.0
    skeleton_pos_weight: float = 50.0  # skeleton ~0.5% of crack pixels


@dataclass
class B3LossConfig(B2LossConfig):
    """B3: B2 + endpoint + junction supervision."""
    endpoint_weight: float = 0.5
    endpoint_pos_weight: float = 100.0  # <0.1% of pixels
    junction_weight: float = 0.5
    junction_pos_weight: float = 100.0


@dataclass
class B4LossConfig(B3LossConfig):
    """B4: B3 + edge/connectivity loss."""
    edge_weight: float = 1.0


@dataclass
class B5LossConfig(B4LossConfig):
    """B5: B4 + width regression."""
    width_weight: float = 0.5


# ---------------------------------------------------------------------------
# Segmentation losses
# ---------------------------------------------------------------------------

class WeightedCEDiceLoss(nn.Module):
    """Weighted cross-entropy + foreground-only Dice loss.

    Handles BG 96.6% / Crack 2.2% / Spalling 1.2% imbalance via:
    1. Per-class CE weights [0.2, 2.0, 3.0]
    2. Dice computed only over foreground classes
    """

    def __init__(
        self,
        ce_weight: float = 0.5,
        dice_weight: float = 0.5,
        class_weights: list[float] | None = None,
        ignore_index: int = 255,
        num_classes: int = 3,
    ) -> None:
        super().__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.num_classes = num_classes
        self.ignore_index = ignore_index

        if class_weights is not None:
            self.register_buffer(
                "class_weights",
                torch.tensor(class_weights, dtype=torch.float32),
            )
        else:
            self.class_weights = None

    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            logits: (B, C, H, W) raw class logits.
            targets: (B, H, W) integer class labels.
        Returns:
            Dict with 'ce', 'dice', 'total'.
        """
        ce = F.cross_entropy(
            logits, targets,
            weight=self.class_weights,
            ignore_index=self.ignore_index,
        )

        # Foreground-only Dice
        probs = F.softmax(logits, dim=1)
        valid = (targets != self.ignore_index).float()
        dice_sum = torch.tensor(0.0, device=logits.device)
        fg_count = 0

        for c in range(1, self.num_classes):
            pred_c = probs[:, c]
            target_c = (targets == c).float()
            intersection = (pred_c * target_c * valid).sum()
            union = (pred_c * valid).sum() + (target_c * valid).sum()
            dice_sum = dice_sum + 1.0 - (2.0 * intersection + 1e-6) / (union + 1e-6)
            fg_count += 1

        dice = dice_sum / max(fg_count, 1)
        total = self.ce_weight * ce + self.dice_weight * dice
        return {"ce": ce, "dice": dice, "total": total}


class TverskyLoss(nn.Module):
    """Tversky loss for a single class. Ablation variant, not default.

    With alpha=0.3, beta=0.7, false negatives are penalized 2.3x more
    than false positives — useful for thin crack recall.
    """

    def __init__(
        self,
        alpha: float = 0.3,
        beta: float = 0.7,
        target_class: int = 1,
        ignore_index: int = 255,
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.target_class = target_class
        self.ignore_index = ignore_index

    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor,
    ) -> torch.Tensor:
        probs = F.softmax(logits, dim=1)[:, self.target_class]
        target_bin = (targets == self.target_class).float()
        valid = (targets != self.ignore_index).float()

        probs = probs * valid
        target_bin = target_bin * valid

        tp = (probs * target_bin).sum()
        fp = (probs * (1 - target_bin)).sum()
        fn = ((1 - probs) * target_bin).sum()

        tversky = (tp + 1e-6) / (tp + self.alpha * fp + self.beta * fn + 1e-6)
        return 1.0 - tversky


# ---------------------------------------------------------------------------
# Binary head losses (skeleton, endpoint, junction)
# All heads output raw logits; use BCEWithLogitsLoss.
# ---------------------------------------------------------------------------

class BinaryHeadLoss(nn.Module):
    """BCE + Dice loss for sparse binary targets.

    Handles extreme sparsity via pos_weight in BCE.
    For skeleton: ~0.5% of crack pixels are skeleton.
    For endpoints: <0.1% of skeleton pixels.

    Args:
        pos_weight: weight for positive class in BCE. Higher values
            increase recall at the cost of precision.
        dice_weight: relative weight of Dice term vs BCE.
    """

    def __init__(
        self,
        pos_weight: float = 50.0,
        dice_weight: float = 0.5,
    ) -> None:
        super().__init__()
        self.register_buffer(
            "pos_weight_tensor",
            torch.tensor([pos_weight], dtype=torch.float32),
        )
        self.dice_weight = dice_weight

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            logits: (B, 1, H, W) raw logits.
            targets: (B, 1, H, W) binary GT (0 or 1).
            mask: optional (B, 1, H, W) valid-pixel mask.
        """
        if mask is not None:
            # Only compute loss on valid pixels
            logits = logits[mask.bool()]
            targets = targets[mask.bool()]
            if logits.numel() == 0:
                return torch.tensor(0.0, device=logits.device, requires_grad=True)

        bce = F.binary_cross_entropy_with_logits(
            logits, targets.float(),
            pos_weight=self.pos_weight_tensor.to(logits.device),
        )

        # Dice on sigmoid output
        probs = torch.sigmoid(logits)
        intersection = (probs * targets).sum()
        union = probs.sum() + targets.sum()
        dice = 1.0 - (2.0 * intersection + 1e-6) / (union + 1e-6)

        return bce + self.dice_weight * dice


# ---------------------------------------------------------------------------
# Width regression
# ---------------------------------------------------------------------------

class WidthRegressionLoss(nn.Module):
    """Smooth L1 loss for per-pixel width, masked to skeleton.

    Width target = 2 * distance_transform (full width, not radius).
    Only skeleton pixels have meaningful targets; loss excludes all others.

    Width values can optionally be log-transformed for stability
    on the wide range (1-20px).
    """

    def __init__(self, use_log: bool = False) -> None:
        super().__init__()
        self.use_log = use_log

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        skeleton_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            pred: (B, 1, H, W) predicted width (non-negative).
            target: (B, 1, H, W) GT width = 2 * DT at skeleton.
            skeleton_mask: (B, 1, H, W) binary skeleton mask.
        """
        valid = skeleton_mask.bool()
        if not valid.any():
            return torch.tensor(0.0, device=pred.device, requires_grad=True)

        p = pred[valid]
        t = target[valid]

        if self.use_log:
            p = torch.log1p(p)
            t = torch.log1p(t)

        return F.smooth_l1_loss(p, t)
