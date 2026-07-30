"""CORAL: CORrelation ALignment for domain generalization.

Sun & Saenko, "Deep CORAL: Correlation Alignment for Deep Domain Adaptation", ECCV 2016.

Minimizes the distance between second-order statistics (covariance matrices)
of features from different domains. In single-source DG, we use DamSegment
difficulty tiers (Easy/Medium/Hard) as pseudo-domains.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def coral_loss(feat_a: torch.Tensor, feat_b: torch.Tensor) -> torch.Tensor:
    """CORAL loss: Frobenius norm of covariance difference.

    Args:
        feat_a: (N_a, D) features from domain A.
        feat_b: (N_b, D) features from domain B.

    Returns:
        Scalar CORAL loss.
    """
    d = feat_a.size(1)

    # Covariance matrices
    mean_a = feat_a.mean(dim=0, keepdim=True)
    mean_b = feat_b.mean(dim=0, keepdim=True)
    ca = feat_a - mean_a
    cb = feat_b - mean_b
    cov_a = (ca.T @ ca) / max(feat_a.size(0) - 1, 1)
    cov_b = (cb.T @ cb) / max(feat_b.size(0) - 1, 1)

    # Frobenius norm of difference, normalized by dimension
    loss = (cov_a - cov_b).pow(2).sum() / (4 * d * d)
    return loss


class CORALRegularizer(nn.Module):
    """CORAL regularizer that extracts features from encoder and computes
    alignment loss between pseudo-domains (difficulty tiers).

    Usage in training loop:
        coral_reg = CORALRegularizer()
        # After encoder forward:
        features = encoder(images, output_hidden_states=True).hidden_states
        coral_loss = coral_reg(features, domain_ids)
        total_loss += coral_weight * coral_loss
    """

    def __init__(self, stage: int = 2):
        """
        Args:
            stage: which encoder stage's features to align (0-indexed).
                   Stage 2 is a good default (mid-level features).
        """
        super().__init__()
        self.stage = stage

    def forward(
        self, hidden_states: list[torch.Tensor], domain_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Compute CORAL loss between domains.

        Args:
            hidden_states: list of encoder stage outputs.
                Stage outputs are (B, H*W, C) or (B, C, H, W).
            domain_ids: (B,) integer domain IDs per sample.

        Returns:
            Scalar CORAL loss (0 if fewer than 2 domains in batch).
        """
        feat = hidden_states[self.stage]

        # Flatten spatial dims: (B, ...) -> (B, D)
        if feat.dim() == 3:
            # (B, H*W, C) -> global average -> (B, C)
            feat = feat.mean(dim=1)
        elif feat.dim() == 4:
            # (B, C, H, W) -> global average -> (B, C)
            feat = feat.mean(dim=[2, 3])

        unique_domains = domain_ids.unique()
        if len(unique_domains) < 2:
            return torch.tensor(0.0, device=feat.device, requires_grad=True)

        # Pairwise CORAL loss between all domain pairs
        total_loss = torch.tensor(0.0, device=feat.device)
        n_pairs = 0
        for i in range(len(unique_domains)):
            for j in range(i + 1, len(unique_domains)):
                mask_i = domain_ids == unique_domains[i]
                mask_j = domain_ids == unique_domains[j]
                if mask_i.sum() >= 2 and mask_j.sum() >= 2:
                    total_loss = total_loss + coral_loss(feat[mask_i], feat[mask_j])
                    n_pairs += 1

        return total_loss / max(n_pairs, 1)
