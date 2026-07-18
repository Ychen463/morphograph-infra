"""Morphology Auxiliary Network: shared encoder + FPN + multi-task heads.

Current scope: dense morphology auxiliary predictions (B0-B5).
NOT yet a graph reconstruction model — see module docstring for roadmap.

Architecture:
    MiT-B2 encoder -> SharedFPN (project + fuse) -> per-task output layers

    All heads share the FPN trunk to:
    - Reduce parameter redundancy vs. per-head projection
    - Simplify capacity-controlled ablation
    - Allow consistent multi-resolution output

Output resolutions:
    seg_head:       full resolution (H, W) — logits
    skeleton_head:  full resolution (H, W) — logits
    endpoint_head:  full resolution (H, W) — logits
    junction_head:  full resolution (H, W) — logits
    width_head:     full resolution (H, W) — raw (non-negative)

All heads output raw logits (no sigmoid/softmax in model).
Use BCEWithLogitsLoss or cross_entropy in the loss module.

SegFormer-B2 (MiT-B2) encoder feature dimensions at 512x512 input:
    Stage 1: (B,  64, 128, 128)  — 1/4
    Stage 2: (B, 128,  64,  64)  — 1/8
    Stage 3: (B, 320,  32,  32)  — 1/16
    Stage 4: (B, 512,  16,  16)  — 1/32

Roadmap (what this module does NOT yet do):
    - Explicit graph decoder (node detection -> pairwise connectivity
      -> edge polyline -> graph pruning) — needed for P3/B5+
    - Spalling instance head (center + boundary + offset) — needed for B6
    - Relation head (pairwise crack-spalling classifier) — needed for P5/B7
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SegformerModel


# MiT-B2 per-stage output channels.
MIT_B2_CHANNELS = (64, 128, 320, 512)

# Shared FPN decode dimension.
FPN_DIM = 256

# Maps internal head name -> output key in forward() dict.
_HEAD_REGISTRY: list[tuple[str, str]] = [
    ("seg_head", "seg"),
    ("skeleton_head", "skeleton"),
    ("endpoint_head", "endpoints"),
    ("junction_head", "junctions"),
    ("width_head", "width"),
]

# Preset head configs matching baseline ladder (BENCHMARK_PROTOCOL.md).
# B1a/B1b: same architecture as B0, differ only in loss (clDice / SRL).
# B6/B7: require spalling instance head / relation head (not yet implemented).
BASELINE_HEADS: dict[str, dict[str, bool]] = {
    "B0": {"seg_head": True},
    "B1a": {"seg_head": True},
    "B1b": {"seg_head": True},
    "B2": {"seg_head": True, "skeleton_head": True},
    "B3": {"seg_head": True, "skeleton_head": True, "endpoint_head": True, "junction_head": True},
    "B4": {"seg_head": True, "skeleton_head": True, "endpoint_head": True, "junction_head": True},
    "B5": {"seg_head": True, "skeleton_head": True, "endpoint_head": True, "junction_head": True, "width_head": True},
}


class SharedFPN(nn.Module):
    """Feature Pyramid Network shared across all heads.

    Projects each encoder stage to FPN_DIM, upsamples to full input
    resolution, and fuses into a single feature map. All heads read
    from this shared representation.

    Using GroupNorm (groups=32) instead of BatchNorm because
    batch_size=4 makes BN statistics unreliable.
    """

    def __init__(
        self,
        in_channels: tuple[int, ...] = MIT_B2_CHANNELS,
        fpn_dim: int = FPN_DIM,
    ) -> None:
        super().__init__()
        # Per-stage lateral projection
        self.laterals = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(ch, fpn_dim, 1, bias=False),
                nn.GroupNorm(32, fpn_dim),
                nn.GELU(),
            )
            for ch in in_channels
        ])
        # Top-down fusion: after upsampling and adding, smooth with 3x3
        self.smooths = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(fpn_dim, fpn_dim, 3, padding=1, bias=False),
                nn.GroupNorm(32, fpn_dim),
                nn.GELU(),
            )
            for _ in in_channels
        ])
        # Final fusion of all levels (all at full resolution)
        self.fuse = nn.Sequential(
            nn.Conv2d(fpn_dim * len(in_channels), fpn_dim, 1, bias=False),
            nn.GroupNorm(32, fpn_dim),
            nn.GELU(),
        )
        self.fpn_dim = fpn_dim

    def forward(
        self,
        features: list[torch.Tensor],
        target_size: tuple[int, int],
    ) -> torch.Tensor:
        """Produce a single fused feature map at target_size.

        Args:
            features: list of 4 encoder stage outputs.
            target_size: (H, W) of the desired output resolution.

        Returns:
            (B, fpn_dim, H, W) fused feature map.
        """
        # Lateral projections
        laterals = [lat(feat) for lat, feat in zip(self.laterals, features)]

        # Top-down pathway (coarse -> fine)
        for i in range(len(laterals) - 1, 0, -1):
            upsampled = F.interpolate(
                laterals[i], size=laterals[i - 1].shape[2:],
                mode="bilinear", align_corners=False,
            )
            laterals[i - 1] = laterals[i - 1] + upsampled

        # Smooth and upsample all to target resolution
        outputs = []
        for lat, smooth in zip(laterals, self.smooths):
            x = smooth(lat)
            if x.shape[2:] != target_size:
                x = F.interpolate(
                    x, size=target_size,
                    mode="bilinear", align_corners=False,
                )
            outputs.append(x)

        return self.fuse(torch.cat(outputs, dim=1))


class SegHead(nn.Module):
    """Semantic segmentation output layer.

    Lightweight: just a 1x1 conv on shared FPN features.
    Output: (B, num_classes, H, W) raw logits at full resolution.
    """

    def __init__(self, fpn_dim: int = FPN_DIM, num_classes: int = 3) -> None:
        super().__init__()
        self.head = nn.Conv2d(fpn_dim, num_classes, 1)

    def forward(self, fpn_features: torch.Tensor) -> torch.Tensor:
        return self.head(fpn_features)


class SkeletonHead(nn.Module):
    """Dense skeleton prediction (binary).

    Predicts whether each pixel lies on a crack skeleton.
    This is a dense auxiliary target, not graph reconstruction.
    Skeleton targets are extremely sparse (~0.5% of crack pixels),
    so the loss should use BCEWithLogitsLoss with pos_weight or
    Dice + focal combination.

    Output: (B, 1, H, W) raw logits (no sigmoid).
    """

    def __init__(self, fpn_dim: int = FPN_DIM) -> None:
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(fpn_dim, fpn_dim // 4, 3, padding=1, bias=False),
            nn.GroupNorm(16, fpn_dim // 4),
            nn.GELU(),
            nn.Conv2d(fpn_dim // 4, 1, 1),
        )

    def forward(self, fpn_features: torch.Tensor) -> torch.Tensor:
        return self.head(fpn_features)


class KeypointHead(nn.Module):
    """Endpoint or junction heatmap prediction.

    Output: (B, 1, H, W) raw logits (no sigmoid).
    Use BCEWithLogitsLoss with pos_weight for the extreme sparsity.
    """

    def __init__(self, fpn_dim: int = FPN_DIM) -> None:
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(fpn_dim, fpn_dim // 4, 3, padding=1, bias=False),
            nn.GroupNorm(16, fpn_dim // 4),
            nn.GELU(),
            nn.Conv2d(fpn_dim // 4, 1, 1),
        )

    def forward(self, fpn_features: torch.Tensor) -> torch.Tensor:
        return self.head(fpn_features)


class WidthHead(nn.Module):
    """Per-pixel crack width regression.

    Width target = 2 * distance_transform at skeleton pixels (full width,
    not radius). Only meaningful at skeleton pixels; loss should be masked.

    Width values are in pixels at training resolution. For DamSegment
    (native 640x640, train at 512x512), a scaling factor of 640/512=1.25
    applies if reporting at native resolution.

    Output: (B, 1, H, W) raw values. Softplus ensures non-negative output
    without the dead-gradient problem of ReLU at zero.
    """

    def __init__(self, fpn_dim: int = FPN_DIM) -> None:
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(fpn_dim, fpn_dim // 4, 3, padding=1, bias=False),
            nn.GroupNorm(16, fpn_dim // 4),
            nn.GELU(),
            nn.Conv2d(fpn_dim // 4, 1, 1),
        )
        self.activation = nn.Softplus()

    def forward(self, fpn_features: torch.Tensor) -> torch.Tensor:
        return self.activation(self.head(fpn_features))


class MorphoAuxNet(nn.Module):
    """Multi-task network with shared FPN and lightweight output heads.

    Covers B0-B5 of the baseline ladder. Honest naming: this is a
    morphology auxiliary network, not yet a graph reconstruction model.

    For graph reconstruction (B5+ in the research sense), a separate
    GraphDecoder module will be needed that takes detected nodes +
    shared features and predicts pairwise connectivity, edge polylines,
    and graph attributes.

    Args:
        backbone: encoder backbone name (only "mit_b2" supported).
        num_classes: number of segmentation classes (3: bg/crack/spalling).
        heads: dict of head_name -> bool indicating which heads to enable.
        fpn_dim: channel width of shared FPN (default 256).
    """

    def __init__(
        self,
        backbone: str = "mit_b2",
        num_classes: int = 3,
        heads: dict[str, bool] | None = None,
        fpn_dim: int = FPN_DIM,
    ) -> None:
        super().__init__()
        self.backbone_name = backbone
        self.num_classes = num_classes
        self.fpn_dim = fpn_dim

        default_heads = {name: False for name, _ in _HEAD_REGISTRY}
        default_heads["seg_head"] = True
        self.active_heads = {**default_heads, **(heads or {})}

        # Shared encoder
        self.encoder = self._build_encoder(backbone)

        # Shared FPN decoder trunk
        self.fpn = SharedFPN(MIT_B2_CHANNELS, fpn_dim)

        # Lightweight output heads (read from shared FPN)
        _builders: dict[str, callable] = {
            "seg_head": lambda: SegHead(fpn_dim, num_classes),
            "skeleton_head": lambda: SkeletonHead(fpn_dim),
            "endpoint_head": lambda: KeypointHead(fpn_dim),
            "junction_head": lambda: KeypointHead(fpn_dim),
            "width_head": lambda: WidthHead(fpn_dim),
        }
        for head_name, _ in _HEAD_REGISTRY:
            if self.active_heads.get(head_name, False):
                setattr(self, head_name, _builders[head_name]())

    @staticmethod
    def _build_encoder(backbone: str) -> nn.Module:
        """Build the shared encoder backbone.

        Uses HuggingFace SegformerModel pretrained on ADE20K.
        Returns the encoder portion which outputs multi-scale features.
        """
        pretrained_map = {
            "mit_b2": "nvidia/segformer-b2-finetuned-ade-512-512",
        }
        if backbone not in pretrained_map:
            raise ValueError(
                f"Unsupported backbone '{backbone}'. "
                f"Available: {list(pretrained_map.keys())}"
            )
        model = SegformerModel.from_pretrained(
            pretrained_map[backbone],
            output_hidden_states=True,
        )
        return model.encoder

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Forward pass: encoder -> FPN -> heads.

        All spatial outputs are at full input resolution (H, W).
        Losses should be computed at full resolution against
        full-resolution GT targets.

        Args:
            x: (B, 3, H, W) input tensor. Expect H=W=512.

        Returns:
            Dict mapping output keys to tensors. All at (B, C, H, W).
            seg: (B, 3, H, W) class logits.
            skeleton: (B, 1, H, W) skeleton logits.
            endpoints: (B, 1, H, W) endpoint logits.
            junctions: (B, 1, H, W) junction logits.
            width: (B, 1, H, W) width values (non-negative).
        """
        input_size = x.shape[2:]  # (H, W)
        # HuggingFace SegformerEncoder with output_hidden_states=True
        # returns exactly 4 stage outputs: (64, 128, 320, 512) channels.
        enc_out = self.encoder(x, output_hidden_states=True, return_dict=True)
        features = list(enc_out.hidden_states)  # 4 stage feature maps

        # Shared FPN features at full input resolution
        fpn_features = self.fpn(features, target_size=input_size)

        outputs: dict[str, torch.Tensor] = {}
        for head_name, output_key in _HEAD_REGISTRY:
            if self.active_heads.get(head_name, False):
                head = getattr(self, head_name)
                outputs[output_key] = head(fpn_features)

        return outputs

    def count_parameters(self) -> dict[str, int]:
        """Parameter breakdown: encoder, FPN, each head.

        Use this for capacity control audits. Each added head
        requires a parameter-matched control without that head's
        supervision signal.
        """
        counts: dict[str, int] = {
            "encoder": sum(p.numel() for p in self.encoder.parameters()),
            "fpn": sum(p.numel() for p in self.fpn.parameters()),
        }
        for head_name, _ in _HEAD_REGISTRY:
            if self.active_heads.get(head_name, False):
                head = getattr(self, head_name)
                counts[head_name] = sum(p.numel() for p in head.parameters())

        total = sum(counts.values())
        trainable = sum(
            p.numel() for p in self.parameters() if p.requires_grad
        )
        counts["total"] = total
        counts["trainable"] = trainable
        return counts

    def get_param_groups(
        self,
        encoder_lr: float = 6e-5,
        head_lr: float = 6e-4,
    ) -> list[dict]:
        """Parameter groups with differential learning rates.

        Encoder uses lower LR (pretrained), new heads use higher LR.

        Args:
            encoder_lr: learning rate for encoder parameters.
            head_lr: learning rate for FPN + head parameters.

        Returns:
            List of param group dicts for optimizer.
        """
        encoder_params = list(self.encoder.parameters())
        other_params = [
            p for name, p in self.named_parameters()
            if not name.startswith("encoder.")
        ]
        return [
            {"params": encoder_params, "lr": encoder_lr},
            {"params": other_params, "lr": head_lr},
        ]
