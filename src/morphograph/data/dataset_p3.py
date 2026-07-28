"""P3 dataset: DamSegment with graph targets (heatmaps + edges)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from morphograph.data.schema import decode_rgb_mask
from morphograph.data.graph_targets import (
    mask_to_dt_target, mask_to_graph, make_gaussian_heatmap,
)


class DamSegmentP3Dataset(Dataset):
    """DamSegment dataset with graph targets for P3."""

    def __init__(
        self,
        pairs: list[tuple[Path, Path]],
        img_size: int = 512,
        augment: bool = False,
        heatmap_sigma: float = 1.5,
    ) -> None:
        self.pairs = pairs
        self.img_size = img_size
        self.augment = augment
        self.heatmap_sigma = heatmap_sigma
        self._transform = None
        if augment:
            self._transform = self._build_augmentation()

    def _build_augmentation(self):
        import albumentations as A
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.1, scale_limit=0.15, rotate_limit=15,
                border_mode=0, p=0.5,
            ),
            A.OneOf([
                A.RandomBrightnessContrast(
                    brightness_limit=0.2, contrast_limit=0.2, p=1.0,
                ),
                A.HueSaturationValue(
                    hue_shift_limit=10, sat_shift_limit=20,
                    val_shift_limit=20, p=1.0,
                ),
            ], p=0.5),
            A.GaussNoise(p=0.2),
        ])

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict:
        img_path, mask_path = self.pairs[idx]

        img = np.array(Image.open(img_path).convert("RGB").resize(
            (self.img_size, self.img_size), Image.BILINEAR,
        ))
        mask_raw = np.array(Image.open(mask_path).resize(
            (self.img_size, self.img_size), Image.NEAREST,
        ))

        if mask_raw.ndim == 3:
            mask = decode_rgb_mask(mask_raw)
        else:
            mask = mask_raw.astype(np.uint8)

        if self._transform is not None:
            transformed = self._transform(image=img, mask=mask)
            img = transformed["image"]
            mask = transformed["mask"]

        crack_binary = (mask == 1).astype(np.uint8)
        dt_target = mask_to_dt_target(crack_binary)

        graph = mask_to_graph(crack_binary, min_branch_length=10, junction_merge_radius=5)

        scale = self.img_size / 128.0
        gt_endpoints_128 = graph.endpoints.astype(np.float32) / scale
        gt_junctions_128 = graph.junctions.astype(np.float32) / scale

        ep_heatmap = make_gaussian_heatmap(
            gt_endpoints_128, sigma=self.heatmap_sigma,
        )
        jn_heatmap = make_gaussian_heatmap(
            gt_junctions_128, sigma=self.heatmap_sigma,
        )
        node_heatmap = np.stack([ep_heatmap, jn_heatmap], axis=0)

        all_coords_128 = np.concatenate(
            [gt_endpoints_128, gt_junctions_128], axis=0
        ) if len(gt_endpoints_128) + len(gt_junctions_128) > 0 else np.empty((0, 2), dtype=np.float32)
        n_ep = len(gt_endpoints_128)
        node_types = np.array(
            [0] * n_ep + [1] * len(gt_junctions_128), dtype=np.int64
        )

        return {
            "image": torch.from_numpy(img).permute(2, 0, 1).float() / 255.0,
            "mask": torch.from_numpy(mask.copy()).long(),
            "dt_target": torch.from_numpy(dt_target.copy()).float().unsqueeze(0),
            "crack_mask": torch.from_numpy(crack_binary.copy()).float().unsqueeze(0),
            "node_heatmap": torch.from_numpy(node_heatmap),
            "gt_node_coords": torch.from_numpy(all_coords_128),
            "gt_node_types": torch.from_numpy(node_types),
            "gt_edges": graph.edges,
            "gt_num_endpoints": n_ep,
        }


def p3_collate_fn(batch: list[dict]) -> dict:
    """Custom collate: stack fixed-size, keep variable-size as lists."""
    result = {}
    for key in ["image", "mask", "dt_target", "crack_mask", "node_heatmap"]:
        result[key] = torch.stack([b[key] for b in batch])
    for key in ["gt_node_coords", "gt_node_types", "gt_edges", "gt_num_endpoints"]:
        result[key] = [b[key] for b in batch]
    return result
