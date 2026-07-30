"""MixStyle: domain generalization via feature statistics mixing.

Zhou et al., "Domain Generalization with MixStyle", ICLR 2021.

Randomly mixes instance-level feature statistics (mean and variance)
between samples in a mini-batch during training. No domain labels needed.
Applied after shallow encoder stages (e.g., stages 1 and 2 of MiT-B2).
"""

from __future__ import annotations

import random

import torch
import torch.nn as nn


class MixStyle(nn.Module):
    """MixStyle layer: shuffle-and-mix feature statistics.

    Args:
        p: probability of applying MixStyle per forward pass.
        alpha: parameter for Beta(alpha, alpha) distribution.
        eps: small constant for numerical stability.
    """

    def __init__(self, p: float = 0.5, alpha: float = 0.1, eps: float = 1e-6):
        super().__init__()
        self.p = p
        self.alpha = alpha
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or random.random() > self.p:
            return x

        B = x.size(0)
        if B < 2:
            return x

        # Instance normalization statistics
        mu = x.mean(dim=[2, 3], keepdim=True)
        var = x.var(dim=[2, 3], keepdim=True)
        sig = (var + self.eps).sqrt()
        x_normed = (x - mu) / sig

        # Random permutation for mixing partners
        perm = torch.randperm(B)
        mu2 = mu[perm]
        sig2 = sig[perm]

        # Mix statistics with random interpolation weight
        lmda = torch.distributions.Beta(self.alpha, self.alpha).sample(
            (B, 1, 1, 1)
        ).to(x.device)
        mu_mix = lmda * mu + (1 - lmda) * mu2
        sig_mix = lmda * sig + (1 - lmda) * sig2

        return x_normed * sig_mix + mu_mix

    def extra_repr(self) -> str:
        return f"p={self.p}, alpha={self.alpha}"


def apply_mixstyle_hooks(
    encoder: nn.Module,
    stages: tuple[int, ...] = (0, 1),
    p: float = 0.5,
    alpha: float = 0.1,
) -> list:
    """Register MixStyle forward hooks on SegformerEncoder stage outputs.

    The HuggingFace SegformerEncoder has ``layer_norm`` as a ModuleList
    with one LayerNorm per stage. We hook after each specified stage's
    LayerNorm to apply MixStyle to the stage output.

    Args:
        encoder: HuggingFace SegformerEncoder module.
        stages: which stages to apply MixStyle after (0-indexed).
        p: MixStyle probability.
        alpha: Beta distribution parameter.

    Returns:
        List of hook handles (for removal if needed).
    """
    mixstyles = []
    handles = []

    for stage_idx in stages:
        ms = MixStyle(p=p, alpha=alpha)
        mixstyles.append(ms)

        # Hook after layer_norm[stage_idx]
        ln = encoder.layer_norm[stage_idx]

        def make_hook(ms_layer):
            def hook_fn(module, input, output):
                # output shape: (B, H*W, C) from LayerNorm
                # Reshape to (B, C, H, W) for MixStyle, then back
                B, HW, C = output.shape
                H = W = int(HW ** 0.5)
                x = output.reshape(B, H, W, C).permute(0, 3, 1, 2)
                x = ms_layer(x)
                return x.permute(0, 2, 3, 1).reshape(B, HW, C)
            return hook_fn

        handle = ln.register_forward_hook(make_hook(ms))
        handles.append(handle)

    # Store mixstyle modules so they participate in .train()/.eval()
    encoder._mixstyle_modules = nn.ModuleList(mixstyles)
    encoder._mixstyle_handles = handles

    return handles
