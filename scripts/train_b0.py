"""B0 full training: mask-only SegFormer-B2 baseline.

Usage:
    python scripts/train_b0.py --data-root data/raw --output runs/B0

This is the first baseline in the progressive ladder (B0-B5).
B0 uses only CE+Dice segmentation loss, no auxiliary heads.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from morphograph.data.schema import decode_rgb_mask, NUM_CLASSES, DEFAULT_CE_WEIGHTS
from morphograph.losses.composite import WeightedCEDiceLoss
from morphograph.models.morphograph_net import MorphoAuxNet, BASELINE_HEADS


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
            if not img_path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
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
    """Split data into train/val sets."""
    rng = random.Random(seed)
    indices = list(range(len(pairs)))
    rng.shuffle(indices)
    n_val = max(1, int(len(pairs) * val_ratio))
    val_indices = set(indices[:n_val])
    train = [pairs[i] for i in range(len(pairs)) if i not in val_indices]
    val = [pairs[i] for i in range(len(pairs)) if i in val_indices]
    return train, val


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_miou(pred: torch.Tensor, target: torch.Tensor, num_classes: int = 3) -> dict[str, float]:
    """Compute per-class IoU and mIoU."""
    ious = {}
    for c in range(num_classes):
        pred_c = (pred == c)
        target_c = (target == c)
        intersection = (pred_c & target_c).sum().item()
        union = (pred_c | target_c).sum().item()
        if union > 0:
            ious[c] = intersection / union
        else:
            ious[c] = float("nan")

    valid_ious = [v for v in ious.values() if not math.isnan(v)]
    fg_ious = [v for c, v in ious.items() if c > 0 and not math.isnan(v)]

    return {
        "per_class": ious,
        "mIoU_all": np.nanmean(valid_ious) if valid_ious else 0.0,
        "mIoU_fg": np.nanmean(fg_ious) if fg_ious else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="B0 full training")
    parser.add_argument("--data-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--output", type=Path, default=Path("runs/B0"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--encoder-lr", type=float, default=6e-5)
    parser.add_argument("--head-lr", type=float, default=6e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--amp", action="store_true", default=True, help="Mixed precision training")
    args = parser.parse_args()

    set_seed(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # ── Data ──
    all_pairs = discover_all_samples(args.data_root)
    if not all_pairs:
        print("ERROR: No data found.")
        sys.exit(1)

    train_pairs, val_pairs = split_data(all_pairs, args.val_ratio, args.seed)
    print(f"Data: {len(all_pairs)} total, {len(train_pairs)} train, {len(val_pairs)} val")

    train_ds = DamSegmentDataset(train_pairs, augment=True)
    val_ds = DamSegmentDataset(val_pairs, augment=False)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    # ── Model ──
    print("Loading SegFormer-B2 pretrained encoder...")
    model = MorphoAuxNet(
        backbone="mit_b2",
        num_classes=NUM_CLASSES,
        heads=BASELINE_HEADS["B0"],
    ).to(device)

    param_counts = model.count_parameters()
    print(f"Parameters: {param_counts['total']:,} total, {param_counts['trainable']:,} trainable")
    for k, v in param_counts.items():
        if k not in ("total", "trainable"):
            print(f"  {k}: {v:,}")

    # ── Optimizer + scheduler ──
    param_groups = model.get_param_groups(
        encoder_lr=args.encoder_lr, head_lr=args.head_lr,
    )
    optimizer = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)

    # Cosine schedule with linear warmup
    total_steps = len(train_loader) * args.epochs
    warmup_steps = len(train_loader) * args.warmup_epochs

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ── Loss ──
    loss_fn = WeightedCEDiceLoss(
        class_weights=DEFAULT_CE_WEIGHTS, ignore_index=255,
    ).to(device)

    # ── AMP scaler ──
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")

    # ── Training loop ──
    best_miou_fg = 0.0
    history = {"train_loss": [], "val_loss": [], "val_mIoU_fg": [], "val_mIoU_all": []}

    print(f"\nTraining B0 for {args.epochs} epochs...")
    print(f"  Batches/epoch: {len(train_loader)}")
    print(f"  Warmup: {args.warmup_epochs} epochs ({warmup_steps} steps)")
    print()

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        # ── Train ──
        model.train()
        train_losses = []
        for batch in train_loader:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)

            with torch.amp.autocast("cuda", enabled=args.amp and device.type == "cuda"):
                outputs = model(images)
                loss_dict = loss_fn(outputs["seg"], masks)
                loss = loss_dict["total"]

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            train_losses.append(loss.item())

        avg_train_loss = np.mean(train_losses)

        # ── Validate ──
        model.eval()
        val_losses = []
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(device)
                masks = batch["mask"].to(device)

                with torch.amp.autocast("cuda", enabled=args.amp and device.type == "cuda"):
                    outputs = model(images)
                    loss_dict = loss_fn(outputs["seg"], masks)

                val_losses.append(loss_dict["total"].item())
                preds = outputs["seg"].argmax(dim=1)
                all_preds.append(preds.cpu())
                all_targets.append(masks.cpu())

        avg_val_loss = np.mean(val_losses)
        all_preds = torch.cat(all_preds)
        all_targets = torch.cat(all_targets)
        miou = compute_miou(all_preds, all_targets)

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["val_mIoU_fg"].append(miou["mIoU_fg"])
        history["val_mIoU_all"].append(miou["mIoU_all"])

        elapsed = time.time() - t0
        lr_enc = optimizer.param_groups[0]["lr"]
        lr_head = optimizer.param_groups[1]["lr"]

        # ── Checkpoint ──
        is_best = miou["mIoU_fg"] > best_miou_fg
        if is_best:
            best_miou_fg = miou["mIoU_fg"]
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_miou_fg": best_miou_fg,
                "config": vars(args),
            }, args.output / "best.pt")

        # Save last
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_miou_fg": best_miou_fg,
            "config": vars(args),
        }, args.output / "last.pt")

        # ── Log ──
        per_class = " ".join(
            f"c{c}={v:.3f}" for c, v in sorted(miou["per_class"].items())
        )
        best_marker = " *" if is_best else ""
        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"train={avg_train_loss:.4f} val={avg_val_loss:.4f} | "
            f"mIoU_fg={miou['mIoU_fg']:.4f} mIoU_all={miou['mIoU_all']:.4f} | "
            f"{per_class} | "
            f"lr={lr_enc:.1e}/{lr_head:.1e} | "
            f"{elapsed:.0f}s{best_marker}"
        )

    # ── Save history ──
    with open(args.output / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    # ── Save loss curves ──
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        axes[0].plot(history["train_loss"], label="train")
        axes[0].plot(history["val_loss"], label="val")
        axes[0].set_title("Loss")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(history["val_mIoU_fg"], label="mIoU_fg")
        axes[1].plot(history["val_mIoU_all"], label="mIoU_all")
        axes[1].set_title("Validation mIoU")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        axes[2].text(0.1, 0.7, f"Best mIoU_fg: {best_miou_fg:.4f}", fontsize=14, transform=axes[2].transAxes)
        axes[2].text(0.1, 0.5, f"Epochs: {args.epochs}", fontsize=12, transform=axes[2].transAxes)
        axes[2].text(0.1, 0.3, f"Params: {param_counts['total']:,}", fontsize=12, transform=axes[2].transAxes)
        axes[2].set_title("Summary")
        axes[2].axis("off")

        plt.tight_layout()
        plt.savefig(args.output / "training_curves.png", dpi=150)
        plt.close()
        print(f"\nCurves saved to {args.output / 'training_curves.png'}")
    except Exception as e:
        print(f"Plot failed: {e}")

    # ── Final summary ──
    summary = {
        "baseline": "B0",
        "best_miou_fg": best_miou_fg,
        "final_val_loss": history["val_loss"][-1],
        "epochs": args.epochs,
        "total_params": param_counts["total"],
        "trainable_params": param_counts["trainable"],
        "train_samples": len(train_pairs),
        "val_samples": len(val_pairs),
        "seed": args.seed,
    }
    with open(args.output / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nB0 training complete. Best mIoU_fg = {best_miou_fg:.4f}")
    print(f"Results saved to {args.output}/")


if __name__ == "__main__":
    main()
