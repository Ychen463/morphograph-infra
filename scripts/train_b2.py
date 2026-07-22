"""B2 training: B0 + skeleton head (distance transform regression).

Usage:
    python scripts/train_b2.py --data-root data/raw --output runs/B2

    # Wave 1 variants:
    python scripts/train_b2.py --data-root data/raw --output runs/B2_dt_v2 --skel-weight 5.0
    python scripts/train_b2.py --data-root data/raw --output runs/B2_dt_v3 --skel-weight 5.0 --skel-loss-type mse
    python scripts/train_b2.py --data-root data/raw --output runs/B2_dt_v4 --skel-weight 1.0 --skel-loss-type mse --skel-unmask

B2 adds a skeleton prediction head with DT regression loss.
Target is normalized distance transform of crack mask (centerline=1.0,
boundary=0.0). Supports masked (crack-only) or unmasked (all pixels)
supervision, SmoothL1 or MSE loss, and scheduled weight ramp-up.
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
from torch.utils.data import Dataset, DataLoader

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from morphograph.data.schema import decode_rgb_mask, NUM_CLASSES, DEFAULT_CE_WEIGHTS
from morphograph.data.graph_targets import mask_to_dt_target
from morphograph.losses.composite import WeightedCEDiceLoss, DTRegressionLoss, LossSchedule
from morphograph.models.morphograph_net import MorphoAuxNet, BASELINE_HEADS, SkeletonHeadDeep
from morphograph.training.utils import (
    set_seed, discover_all_samples, split_data,
    compute_miou, make_cosine_schedule, save_checkpoint,
)


# ---------------------------------------------------------------------------
# Dataset with skeleton targets
# ---------------------------------------------------------------------------

class DamSegmentDTDataset(Dataset):
    """DamSegment dataset that generates distance transform targets on-the-fly."""

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
        ], additional_targets={"dt_target": "mask", "crack_mask": "mask"})

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

        # Generate DT target from crack class
        crack_binary = (mask == 1).astype(np.uint8)
        dt_target = mask_to_dt_target(crack_binary)

        if self._transform is not None:
            transformed = self._transform(
                image=img, mask=mask,
                dt_target=dt_target, crack_mask=crack_binary,
            )
            img = transformed["image"]
            mask = transformed["mask"]
            dt_target = transformed["dt_target"]
            crack_binary = transformed["crack_mask"]

        img_t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        mask_t = torch.from_numpy(mask.copy()).long()
        dt_t = torch.from_numpy(dt_target.copy()).float().unsqueeze(0)  # (1, H, W)
        crack_t = torch.from_numpy(crack_binary.copy()).float().unsqueeze(0)  # (1, H, W)

        return {"image": img_t, "mask": mask_t, "dt_target": dt_t, "crack_mask": crack_t}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="B2 training: B0 + skeleton DT regression")
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
    # Skeleton DT loss config
    parser.add_argument("--skel-weight", type=float, default=0.3)
    parser.add_argument("--skel-loss-type", choices=["smooth_l1", "mse"], default="smooth_l1")
    parser.add_argument("--skel-unmask", action="store_true",
                        help="Supervise all pixels (non-crack→0, crack→DT) instead of crack-only")
    parser.add_argument("--skel-start-epoch", type=int, default=0,
                        help="Epoch to start skeleton loss (0 = from beginning)")
    parser.add_argument("--skel-ramp-epochs", type=int, default=0,
                        help="Epochs to linearly ramp skeleton loss weight")
    parser.add_argument("--skel-head-deep", action="store_true",
                        help="Use deeper skeleton head (256->128->64->1, ~450K params)")
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
        DamSegmentDTDataset(train_pairs, augment=True),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        DamSegmentDTDataset(val_pairs, augment=False),
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

    if args.skel_head_deep:
        from morphograph.models.morphograph_net import FPN_DIM
        model.skeleton_head = SkeletonHeadDeep(FPN_DIM).to(device)
        print("Using SkeletonHeadDeep (256->128->64->1)")

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

    skel_loss_fn = DTRegressionLoss(loss_type=args.skel_loss_type).to(device)

    skel_schedule = LossSchedule(
        weight=args.skel_weight,
        start_epoch=args.skel_start_epoch,
        ramp_epochs=args.skel_ramp_epochs,
    )

    print(f"\nSkeleton DT loss: weight={args.skel_weight}, type={args.skel_loss_type}, "
          f"unmask={args.skel_unmask}, start={args.skel_start_epoch}, ramp={args.skel_ramp_epochs}")

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
            dt_targets = batch["dt_target"].to(device)
            crack_masks = batch["crack_mask"].to(device)

            with torch.amp.autocast("cuda", enabled=args.amp and device.type == "cuda"):
                outputs = model(images)
                seg_loss = seg_loss_fn(outputs["seg"], masks)["total"]
                skel_pred = torch.sigmoid(outputs["skeleton"])
                skel_mask = torch.ones_like(crack_masks) if args.skel_unmask else crack_masks
                skel_loss = skel_loss_fn(skel_pred, dt_targets, skel_mask)
                skel_w = skel_schedule.effective_weight(epoch)
                total_loss = seg_loss + skel_w * skel_loss

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
        "description": f"B0 + skeleton DT regression ({args.skel_loss_type}, {'unmasked' if args.skel_unmask else 'crack-masked'})",
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
            "loss_type": args.skel_loss_type,
            "unmasked": args.skel_unmask,
            "start_epoch": args.skel_start_epoch,
            "ramp_epochs": args.skel_ramp_epochs,
            "head_deep": args.skel_head_deep,
            "target": "normalized_distance_transform",
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
