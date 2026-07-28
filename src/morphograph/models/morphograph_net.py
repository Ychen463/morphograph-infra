"""Morphology Auxiliary/Graph Network: shared encoder + FPN + multi-task heads.

B0-B5 scope: dense morphology auxiliary predictions via MorphoAuxNet.
P3 scope: direct graph prediction via MorphoGraphNet.

Architecture (B0-B5):
    MiT-B2 encoder -> SharedFPN (project + fuse at 512x512) -> per-task heads

Architecture (P3):
    MiT-B2 encoder -> SharedFPN128 (fuse at 128x128) -> SegDecoder/DTDecoder (128->512)
                                                      -> NodeHeatmapHead (128x128)
                                                      -> EdgeClassifier (per-node)

SegFormer-B2 (MiT-B2) encoder feature dimensions at 512x512 input:
    Stage 1: (B,  64, 128, 128)  — 1/4
    Stage 2: (B, 128,  64,  64)  — 1/8
    Stage 3: (B, 320,  32,  32)  — 1/16
    Stage 4: (B, 512,  16,  16)  — 1/32
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


class SkeletonHeadDeep(nn.Module):
    """Deeper skeleton head for DT regression (Wave 2 v6).

    256→128→64→1 (~450K params vs 147K for SkeletonHead).
    Tests whether head capacity is a bottleneck for DT learning.

    Output: (B, 1, H, W) raw logits (no sigmoid).
    """

    def __init__(self, fpn_dim: int = FPN_DIM) -> None:
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(fpn_dim, fpn_dim // 2, 3, padding=1, bias=False),
            nn.GroupNorm(16, fpn_dim // 2),
            nn.GELU(),
            nn.Conv2d(fpn_dim // 2, fpn_dim // 4, 3, padding=1, bias=False),
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


# ======================================================================
# P3: Direct Graph Prediction (MorphoGraphNet)
# ======================================================================


class SharedFPN128(nn.Module):
    """FPN variant that fuses at 128x128 (1/4 res) instead of full resolution.

    Same lateral projections and top-down pathway as SharedFPN.
    All levels upsample to 128x128, saving ~16x memory on the output tensor.
    Conv weights are resolution-agnostic — transfer directly from SharedFPN.
    """

    def __init__(
        self,
        in_channels: tuple[int, ...] = MIT_B2_CHANNELS,
        fpn_dim: int = FPN_DIM,
    ) -> None:
        super().__init__()
        self.laterals = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(ch, fpn_dim, 1, bias=False),
                nn.GroupNorm(32, fpn_dim),
                nn.GELU(),
            )
            for ch in in_channels
        ])
        self.smooths = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(fpn_dim, fpn_dim, 3, padding=1, bias=False),
                nn.GroupNorm(32, fpn_dim),
                nn.GELU(),
            )
            for _ in in_channels
        ])
        self.fuse = nn.Sequential(
            nn.Conv2d(fpn_dim * len(in_channels), fpn_dim, 1, bias=False),
            nn.GroupNorm(32, fpn_dim),
            nn.GELU(),
        )
        self.fpn_dim = fpn_dim

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        """Produce fused feature map at 128x128.

        Args:
            features: list of 4 encoder stage outputs.

        Returns:
            (B, fpn_dim, 128, 128) fused feature map.
        """
        target_size = (128, 128)

        laterals = [lat(feat) for lat, feat in zip(self.laterals, features)]

        for i in range(len(laterals) - 1, 0, -1):
            upsampled = F.interpolate(
                laterals[i], size=laterals[i - 1].shape[2:],
                mode="bilinear", align_corners=False,
            )
            laterals[i - 1] = laterals[i - 1] + upsampled

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


class SegDecoder(nn.Module):
    """Lightweight decoder: 128x128 FPN features -> 512x512 segmentation.

    Conv(256->128, 3x3) + GN + GELU -> Upsample(x4) -> Conv(128->C, 1x1)
    """

    def __init__(self, fpn_dim: int = FPN_DIM, num_classes: int = 3) -> None:
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Conv2d(fpn_dim, fpn_dim // 2, 3, padding=1, bias=False),
            nn.GroupNorm(16, fpn_dim // 2),
            nn.GELU(),
        )
        self.head = nn.Conv2d(fpn_dim // 2, num_classes, 1)

    def forward(self, fpn_features: torch.Tensor) -> torch.Tensor:
        x = self.decoder(fpn_features)
        x = F.interpolate(x, scale_factor=4, mode="bilinear", align_corners=False)
        return self.head(x)


class DTDecoder(nn.Module):
    """Lightweight decoder: 128x128 FPN features -> 512x512 DT prediction.

    Conv(256->128, 3x3) + GN + GELU -> Upsample(x4) -> Conv(128->1, 1x1)
    """

    def __init__(self, fpn_dim: int = FPN_DIM) -> None:
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Conv2d(fpn_dim, fpn_dim // 2, 3, padding=1, bias=False),
            nn.GroupNorm(16, fpn_dim // 2),
            nn.GELU(),
        )
        self.head = nn.Conv2d(fpn_dim // 2, 1, 1)

    def forward(self, fpn_features: torch.Tensor) -> torch.Tensor:
        x = self.decoder(fpn_features)
        x = F.interpolate(x, scale_factor=4, mode="bilinear", align_corners=False)
        return self.head(x)


class MorphoGraphNet(nn.Module):
    """P3 model: shared encoder + FPN at 128x128 + optional graph heads.

    When graph_heads=False, this is the P3-Base architectural control:
    same FPN128 + SegDecoder + DTDecoder, but NO graph supervision.
    This isolates the effect of the decoder change from graph prediction.

    Ablation ladder:
        B2:      SharedFPN@512 + SegHead + SkeletonHead
        P3-Base: SharedFPN128 + SegDecoder + DTDecoder (no graph heads)
        P3a:     P3-Base + NodeHeatmapHead
        P3b:     P3a + EdgeClassifier
        P3c:     P3b + Dijkstra path recovery (inference only)
    """

    def __init__(
        self,
        backbone: str = "mit_b2",
        num_classes: int = 3,
        fpn_dim: int = FPN_DIM,
        graph_heads: bool = True,
    ) -> None:
        super().__init__()
        self.backbone_name = backbone
        self.num_classes = num_classes
        self.fpn_dim = fpn_dim
        self.has_graph_heads = graph_heads

        self.encoder = MorphoAuxNet._build_encoder(backbone)
        self.fpn128 = SharedFPN128(MIT_B2_CHANNELS, fpn_dim)
        self.seg_decoder = SegDecoder(fpn_dim, num_classes)
        self.dt_decoder = DTDecoder(fpn_dim)

        if graph_heads:
            from morphograph.models.graph_decoder import NodeHeatmapHead, EdgeClassifier
            self.node_head = NodeHeatmapHead(fpn_dim)
            self.edge_classifier = EdgeClassifier(fpn_dim)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Forward pass. EdgeClassifier called separately in training loop.

        Returns:
            seg: (B, C, 512, 512) class logits.
            skeleton: (B, 1, 512, 512) DT logits.
            node_heatmap: (B, 2, 128, 128) node logits (only if graph_heads).
            _fpn: (B, 256, 128, 128) for edge classifier (only if graph_heads).
        """
        enc_out = self.encoder(x, output_hidden_states=True, return_dict=True)
        features = list(enc_out.hidden_states)

        fpn = self.fpn128(features)

        outputs = {
            "seg": self.seg_decoder(fpn),
            "skeleton": self.dt_decoder(fpn),
        }
        if self.has_graph_heads:
            outputs["node_heatmap"] = self.node_head(fpn)
            outputs["_fpn"] = fpn

        return outputs

    def count_parameters(self) -> dict[str, int]:
        counts = {
            "encoder": sum(p.numel() for p in self.encoder.parameters()),
            "fpn128": sum(p.numel() for p in self.fpn128.parameters()),
            "seg_decoder": sum(p.numel() for p in self.seg_decoder.parameters()),
            "dt_decoder": sum(p.numel() for p in self.dt_decoder.parameters()),
        }
        if self.has_graph_heads:
            counts["node_head"] = sum(p.numel() for p in self.node_head.parameters())
            counts["edge_classifier"] = sum(p.numel() for p in self.edge_classifier.parameters())
        total = sum(counts.values())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        counts["total"] = total
        counts["trainable"] = trainable
        return counts

    def get_param_groups(
        self,
        encoder_lr: float = 6e-5,
        head_lr: float = 3e-4,
    ) -> list[dict]:
        """Param groups: encoder, fpn+decoders, graph heads (if present)."""
        encoder_params = list(self.encoder.parameters())
        fpn_decoder_params = (
            list(self.fpn128.parameters())
            + list(self.seg_decoder.parameters())
            + list(self.dt_decoder.parameters())
        )
        groups = [
            {"params": encoder_params, "lr": encoder_lr},
            {"params": fpn_decoder_params, "lr": head_lr},
        ]
        if self.has_graph_heads:
            graph_params = (
                list(self.node_head.parameters())
                + list(self.edge_classifier.parameters())
            )
            groups.append({"params": graph_params, "lr": head_lr})
        return groups


def load_b2_into_p3(
    b2_ckpt_path: str,
    p3_model: MorphoGraphNet,
    device: torch.device | str = "cpu",
) -> tuple[list[str], list[str]]:
    """Transfer B2 checkpoint weights into P3 model.

    Transfers (shape-checked, skip on mismatch):
      - Encoder: exact copy (all keys match)
      - FPN laterals/smooths/fuse: exact copy (conv weights are resolution-agnostic)
      - SegHead -> SegDecoder final conv: only if shapes match
        (B2 SegHead is 256->3 1x1, SegDecoder.head is 128->3 1x1 — shape
        mismatch, so this is SKIPPED in practice; SegDecoder re-initializes)
      - SkeletonHead -> DTDecoder: skipped (64-ch vs 128-ch intermediate)

    New heads (NodeHeatmapHead, EdgeClassifier) stay randomly initialized.

    Returns:
        (loaded_keys, skipped_keys) for logging.
    """
    ckpt = torch.load(b2_ckpt_path, map_location=device, weights_only=False)
    b2_state = ckpt["model_state_dict"]

    p3_state = p3_model.state_dict()
    loaded = []
    skipped = []

    for b2_key, b2_val in b2_state.items():
        # Encoder: direct copy
        if b2_key.startswith("encoder."):
            if b2_key in p3_state and p3_state[b2_key].shape == b2_val.shape:
                p3_state[b2_key] = b2_val
                loaded.append(b2_key)
            else:
                skipped.append(b2_key)

        # FPN -> FPN128: laterals, smooths, fuse are resolution-agnostic
        elif b2_key.startswith("fpn."):
            p3_key = b2_key.replace("fpn.", "fpn128.", 1)
            if p3_key in p3_state and p3_state[p3_key].shape == b2_val.shape:
                p3_state[p3_key] = b2_val
                loaded.append(f"{b2_key} -> {p3_key}")
            else:
                skipped.append(b2_key)

        # SegHead -> SegDecoder final conv
        elif b2_key.startswith("seg_head.head."):
            # B2 SegHead: nn.Conv2d(256, 3, 1) keyed as "seg_head.head.weight/bias"
            suffix = b2_key.replace("seg_head.head.", "")
            p3_key = f"seg_decoder.head.{suffix}"
            if p3_key in p3_state and p3_state[p3_key].shape == b2_val.shape:
                p3_state[p3_key] = b2_val
                loaded.append(f"{b2_key} -> {p3_key}")
            else:
                skipped.append(b2_key)

        # SkeletonHead -> DTDecoder
        elif b2_key.startswith("skeleton_head.head."):
            # B2 SkeletonHead: Sequential(Conv(256,64,3)+GN+GELU, Conv(64,1,1))
            # DTDecoder: Sequential(Conv(256,128,3)+GN+GELU), head=Conv(128,1,1)
            # Shapes differ (64 vs 128), so skip — DTDecoder is re-initialized
            skipped.append(b2_key)

        else:
            skipped.append(b2_key)

    p3_model.load_state_dict(p3_state)
    return loaded, skipped
