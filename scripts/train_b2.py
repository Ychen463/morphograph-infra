"""B2 training: B0 + skeleton head (explicit morphology supervision).

Usage:
    python scripts/train_b2.py --data-root data/raw --output runs/B2

B2 adds a skeleton prediction head with BCE+Dice loss, providing
explicit dense morphology supervision. Same training budget as B0/B1a
(100 epochs). Tests whether explicit skeleton targets improve
segmentation quality vs topology-loss-only (B1a).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy import ndimage
from skimage.morphology import skeletonize
from torch.utils.data import Dataset, DataLoader

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from morphograph.data.schema import decode_rgb_mask, NUM_CLASSES, DEFAULT_CE_WEIGHTS
from morphograph.losses.composite import WeightedCEDiceLoss, BinaryHeadLoss
from morphograph.models.morphograph_net import MorphoAuxNet, BASELINE_HEADS
from morphograph.training.utils import (
    set_seed, discover_all_samples, split_data,
    compute_miou, make_cosine_schedule, save_checkpoint,
)


# ---------------------------------------------------------------------------
# Dataset with skeleton targets
# ---------------------------------------------------------------------------

class DamSegmentSkeletonDataset(Dataset):
    """DamSegment dataset that also generates skeleton targets on-the-fly."""

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
        ], additional_targets={"skeleton": "mask"})

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

        # Generate skeleton from crack class
        crack_binary = (mask == 1).astype(np.uint8)
        if crack_binary.any():
            skel = skeletonize(crack_binary.astype(bool)).astype(np.uint8)
        else:
            skel = np.zeros_like(crack_binary, dtype=np.uint8)

        if self._transform is not None:
            transformed = self._transform(image=img, mask=mask, skeleton=skel)
            img = transformed["image"]
            mask = transformed["mask"]
            skel = transformed["skeleton"]

        img_t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        mask_t = torch.from_numpy(mask.copy()).long()
        skel_t = torch.from_numpy(skel.copy()).float().unsqueeze(0)  # (1, H, W)

        return {"image": img_t, "mask": mask_t, "skeleton": skel_t}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="B2 training: B0 + skeleton head")
    parser.add_argument("--data-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--output", type=Path, default=Path("runs/B2"))
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
    parser.add_argument("--amp", action="store_true", default=True)
    # Skeleton loss config
    parser.add_argument("--skel-weight", type=float, default=1.0)
    parser.add_argument("--skel-pos-weight", type=float, default=200.0)
    parser.add_argument("--skel-dice-weight", type=float, default=0.2)
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

    train_loader = DataLoader(
        DamSegmentSkeletonDataset(train_pairs, augment=True),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        DamSegmentSkeletonDataset(val_pairs, augment=False),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    # ── Model (B2: seg_head + skeleton_head) ──
    print("Loading SegFormer-B2 pretrained encoder...")
    model = MorphoAuxNet(
        backbone="mit_b2",
        num_classes=NUM_CLASSES,
        heads=BASELINE_HEADS["B2"],
    ).to(device)

    param_counts = model.count_parameters()
    print(f"Parameters: {param_counts['total']:,} total")
    for k, v in param_counts.items():
        if k not in ("total", "trainable"):
            print(f"  {k}: {v:,}")

    # ── Optimizer + scheduler ──
    param_groups = model.get_param_groups(
        encoder_lr=args.encoder_lr, head_lr=args.head_lr,
    )
    optimizer = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)
    total_steps = len(train_loader) * args.epochs
    warmup_steps = len(train_loader) * args.warmup_epochs
    scheduler = make_cosine_schedule(optimizer, total_steps, warmup_steps)

    # ── Losses ──
    seg_loss_fn = WeightedCEDiceLoss(
        class_weights=DEFAULT_CE_WEIGHTS, ignore_index=255,
    ).to(device)

    skel_loss_fn = BinaryHeadLoss(
        pos_weight=args.skel_pos_weight,
        dice_weight=args.skel_dice_weight,
    ).to(device)

    print(f"\nSkeleton loss: weight={args.skel_weight}, "
          f"pos_weight={args.skel_pos_weight}, dice_weight={args.skel_dice_weight}")

    # ── AMP ──
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")

    # ── Training ──
    best_miou_fg = 0.0
    history = {
        "train_loss": [], "train_seg_loss": [], "train_skel_loss": [],
        "val_loss": [], "val_mIoU_fg": [], "val_mIoU_all": [],
    }

    print(f"\nTraining B2 for {args.epochs} epochs...")
    print(f"  Batches/epoch: {len(train_loader)}")
    print()

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        # ── Train ──
        model.train()
        epoch_seg_losses = []
        epoch_skel_losses = []
        epoch_total_losses = []

        for batch in train_loader:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            skeletons = batch["skeleton"].to(device)

            with torch.amp.autocast("cuda", enabled=args.amp and device.type == "cuda"):
                outputs = model(images)
                seg_loss = seg_loss_fn(outputs["seg"], masks)["total"]
                skel_loss = skel_loss_fn(outputs["skeleton"], skeletons)
                total_loss = seg_loss + args.skel_weight * skel_loss

            optimizer.zero_grad()
            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            epoch_seg_losses.append(seg_loss.item())
            epoch_skel_losses.append(skel_loss.item())
            epoch_total_losses.append(total_loss.item())

        avg_seg = np.mean(epoch_seg_losses)
        avg_skel = np.mean(epoch_skel_losses)
        avg_total = np.mean(epoch_total_losses)

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
                    val_loss = seg_loss_fn(outputs["seg"], masks)["total"]
                val_losses.append(val_loss.item())
                all_preds.append(outputs["seg"].argmax(dim=1).cpu())
                all_targets.append(masks.cpu())

        avg_val_loss = np.mean(val_losses)
        miou = compute_miou(torch.cat(all_preds), torch.cat(all_targets))

        history["train_loss"].append(avg_total)
        history["train_seg_loss"].append(avg_seg)
        history["train_skel_loss"].append(avg_skel)
        history["val_loss"].append(avg_val_loss)
        history["val_mIoU_fg"].append(miou["mIoU_fg"])
        history["val_mIoU_all"].append(miou["mIoU_all"])

        elapsed = time.time() - t0

        # ── Checkpoint ──
        is_best = miou["mIoU_fg"] > best_miou_fg
        if is_best:
            best_miou_fg = miou["mIoU_fg"]
            save_checkpoint(args.output / "best.pt", model, optimizer, epoch, best_miou_fg, args)
        save_checkpoint(args.output / "last.pt", model, optimizer, epoch, best_miou_fg, args)

        # ── Log ──
        per_class = " ".join(f"c{c}={v:.3f}" for c, v in sorted(miou["per_class"].items()))
        best_marker = " *" if is_best else ""
        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"seg={avg_seg:.4f} skel={avg_skel:.4f} total={avg_total:.4f} | "
            f"val={avg_val_loss:.4f} mIoU_fg={miou['mIoU_fg']:.4f} | "
            f"{per_class} | {elapsed:.0f}s{best_marker}"
        )

    # ── Save history + curves ──
    with open(args.output / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].plot(history["train_seg_loss"], label="seg")
        axes[0].plot(history["train_skel_loss"], label="skeleton")
        axes[0].plot(history["train_loss"], label="total")
        axes[0].set_title("Train Loss")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(history["val_mIoU_fg"], label="mIoU_fg")
        axes[1].plot(history["val_mIoU_all"], label="mIoU_all")
        axes[1].set_title("Validation mIoU")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        axes[2].text(0.1, 0.8, f"Best mIoU_fg: {best_miou_fg:.4f}", fontsize=14, transform=axes[2].transAxes)
        axes[2].text(0.1, 0.6, f"B0 mIoU_fg:  0.673", fontsize=12, transform=axes[2].transAxes)
        axes[2].text(0.1, 0.4, f"B1a mIoU_fg: 0.657", fontsize=12, transform=axes[2].transAxes)
        axes[2].text(0.1, 0.2, f"Delta vs B0: {best_miou_fg - 0.673:+.4f}", fontsize=12, transform=axes[2].transAxes)
        axes[2].set_title("B2 vs B0/B1a")
        axes[2].axis("off")
        plt.tight_layout()
        plt.savefig(args.output / "training_curves.png", dpi=150)
        plt.close()
        print(f"\nCurves saved to {args.output / 'training_curves.png'}")
    except Exception as e:
        print(f"Plot failed: {e}")

    # ── Summary ──
    summary = {
        "baseline": "B2",
        "description": "B0 + skeleton head (explicit morphology supervision)",
        "best_miou_fg": best_miou_fg,
        "b0_miou_fg": 0.673,
        "b1a_miou_fg": 0.657,
        "delta_vs_b0": best_miou_fg - 0.673,
        "delta_vs_b1a": best_miou_fg - 0.657,
        "final_val_loss": history["val_loss"][-1],
        "epochs": args.epochs,
        "total_params": param_counts["total"],
        "skeleton_config": {
            "weight": args.skel_weight,
            "pos_weight": args.skel_pos_weight,
            "dice_weight": args.skel_dice_weight,
        },
        "train_samples": len(train_pairs),
        "val_samples": len(val_pairs),
        "seed": args.seed,
    }
    with open(args.output / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nB2 training complete. Best mIoU_fg = {best_miou_fg:.4f}")
    print(f"Delta vs B0: {best_miou_fg - 0.673:+.4f}")
    print(f"Delta vs B1a: {best_miou_fg - 0.657:+.4f}")
    print(f"Results saved to {args.output}/")


if __name__ == "__main__":
    main()
