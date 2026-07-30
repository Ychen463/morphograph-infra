"""DANN: Domain-Adversarial Neural Network for domain generalization.

Ganin et al., "Domain-Adversarial Training of Neural Networks", JMLR 2016.

Trains a domain discriminator on encoder features with gradient reversal,
forcing the encoder to learn domain-invariant representations.
In single-source DG, we use DamSegment difficulty tiers as pseudo-domains.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.autograd import Function


class GradientReversalFunction(Function):
    """Gradient reversal layer: identity in forward, negates gradient in backward."""

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None


def grad_reverse(x: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
    """Apply gradient reversal."""
    return GradientReversalFunction.apply(x, alpha)


class DomainDiscriminator(nn.Module):
    """MLP domain classifier with gradient reversal.

    Args:
        in_dim: input feature dimension.
        hidden_dim: hidden layer dimension.
        num_domains: number of pseudo-domains to classify.
    """

    def __init__(self, in_dim: int, hidden_dim: int = 256, num_domains: int = 3):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, num_domains),
        )

    def forward(
        self, feat: torch.Tensor, alpha: float = 1.0,
    ) -> torch.Tensor:
        """
        Args:
            feat: (B, D) features (already pooled).
            alpha: gradient reversal strength (ramp up during training).

        Returns:
            (B, num_domains) domain logits.
        """
        feat_rev = grad_reverse(feat, alpha)
        return self.classifier(feat_rev)


class DANNRegularizer(nn.Module):
    """DANN regularizer: domain discriminator on encoder features.

    Usage in training loop:
        dann_reg = DANNRegularizer(in_dim=320, num_domains=3).to(device)
        features = encoder(images, output_hidden_states=True).hidden_states
        dann_loss = dann_reg(features, domain_ids, epoch, max_epochs)
        total_loss += dann_weight * dann_loss
    """

    # MiT-B2 hidden dimensions per stage
    MIT_B2_DIMS = (64, 128, 320, 512)

    def __init__(
        self, stage: int = 2, hidden_dim: int = 256, num_domains: int = 3,
    ):
        """
        Args:
            stage: which encoder stage to discriminate (0-indexed).
            hidden_dim: discriminator hidden layer size.
            num_domains: number of pseudo-domains.
        """
        super().__init__()
        self.stage = stage
        in_dim = self.MIT_B2_DIMS[stage]
        self.discriminator = DomainDiscriminator(in_dim, hidden_dim, num_domains)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(
        self,
        hidden_states: list[torch.Tensor],
        domain_ids: torch.Tensor,
        epoch: int = 1,
        max_epochs: int = 100,
    ) -> torch.Tensor:
        """Compute DANN adversarial loss.

        Args:
            hidden_states: list of encoder stage outputs.
            domain_ids: (B,) integer domain labels.
            epoch: current training epoch (for alpha scheduling).
            max_epochs: total epochs (for alpha scheduling).

        Returns:
            Scalar domain classification loss (with reversed gradients).
        """
        feat = hidden_states[self.stage]

        # Global average pool
        if feat.dim() == 3:
            feat = feat.mean(dim=1)  # (B, C)
        elif feat.dim() == 4:
            feat = feat.mean(dim=[2, 3])  # (B, C)

        # GRL alpha schedule: 0 → 1 over training (Ganin et al. Eq. 4)
        p = epoch / max_epochs
        alpha = 2.0 / (1.0 + torch.exp(torch.tensor(-10.0 * p)).item()) - 1.0

        domain_logits = self.discriminator(feat, alpha=alpha)
        return self.loss_fn(domain_logits, domain_ids)
