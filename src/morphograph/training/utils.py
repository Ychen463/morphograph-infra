"""Shared training utilities for all baseline scripts.

Extracted from train_b0.py / train_b1a.py to eliminate duplication.
"""

from __future__ import annotations

import math
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from morphograph.data.schema import decode_rgb_mask


# ---------------------------------------------------------------------------
# Seeding
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class DamSegmentDataset(Dataset):
    """DamSegment dataset with on-the-fly augmentation."""

    def __init__(
        self,
        pairs: list[tuple[Path, Path]],
        img_size: int = 512,
        augment: bool = False,
    ) -> None:
        self.pairs = pairs
        self.img_size = img_size
        self.augment = augment
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

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
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

        img_t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        mask_t = torch.from_numpy(mask.copy()).long()

        return {"image": img_t, "mask": mask_t}


# ---------------------------------------------------------------------------
# Data discovery
# ---------------------------------------------------------------------------

def discover_all_samples(data_root: Path) -> list[tuple[Path, Path]]:
    """Find all image-mask pairs from DamSegment."""
    pairs = []
    for tier in ["Easy", "Medium", "Hard"]:
        img_dir = data_root / f"DamSegment/Damage Segmentaion/{tier}/Images"
        mask_dir = data_root / f"DamSegment/Damage Segmentaion/{tier}/Labels/Mask"
        if not img_dir.exists():
            continue
        for img_path in sorted(img_dir.iterdir()):
            if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            for suffix in ["_mask", ""]:
                for ext in [".png", ".jpg", ".jpeg"]:
                    mask_path = mask_dir / (img_path.stem + suffix + ext)
                    if mask_path.exists():
                        pairs.append((img_path, mask_path))
                        break
                else:
                    continue
                break
    return pairs


def split_data(
    pairs: list[tuple[Path, Path]],
    val_ratio: float = 0.15,
    seed: int = 42,
) -> tuple[list[tuple[Path, Path]], list[tuple[Path, Path]]]:
    """Deterministic train/val split."""
    rng = random.Random(seed)
    indices = list(range(len(pairs)))
    rng.shuffle(indices)
    n_val = max(1, int(len(pairs) * val_ratio))
    val_indices = set(indices[:n_val])
    train = [pairs[i] for i in range(len(pairs)) if i not in val_indices]
    val = [pairs[i] for i in range(len(pairs)) if i in val_indices]
    return train, val


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_miou(
    pred: torch.Tensor,
    target: torch.Tensor,
    num_classes: int = 3,
) -> dict[str, float]:
    """Compute per-class IoU and mIoU."""
    ious = {}
    for c in range(num_classes):
        pred_c = (pred == c)
        target_c = (target == c)
        intersection = (pred_c & target_c).sum().item()
        union = (pred_c | target_c).sum().item()
        ious[c] = intersection / union if union > 0 else float("nan")

    valid_ious = [v for v in ious.values() if not math.isnan(v)]
    fg_ious = [v for c, v in ious.items() if c > 0 and not math.isnan(v)]

    return {
        "per_class": ious,
        "mIoU_all": float(np.nanmean(valid_ious)) if valid_ious else 0.0,
        "mIoU_fg": float(np.nanmean(fg_ious)) if fg_ious else 0.0,
    }


# ---------------------------------------------------------------------------
# LR schedule
# ---------------------------------------------------------------------------

def make_cosine_schedule(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    warmup_steps: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Cosine annealing with linear warmup."""
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------

def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_miou_fg: float,
    args,
) -> None:
    """Save model checkpoint with config."""
    # Convert Path objects to str for JSON compatibility
    config = {}
    for k, v in vars(args).items():
        config[k] = str(v) if isinstance(v, Path) else v
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_miou_fg": best_miou_fg,
        "config": config,
    }, path)
