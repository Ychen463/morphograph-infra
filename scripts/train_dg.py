"""Domain generalization training: B2 + DG methods.

Usage:
    # D0: ERM baseline (B0, no skeleton head)
    python scripts/train_dg.py --data-root data/raw --output runs/D0 --no-skeleton

    # D1: ERM baseline (B2, with skeleton head)
    python scripts/train_dg.py --data-root data/raw --output runs/D1

    # D2c: B2 + MixStyle
    python scripts/train_dg.py --data-root data/raw --output runs/D2c_mixstyle \
        --mixstyle --mixstyle-p 0.5 --mixstyle-alpha 0.1

Trains on DamSegment, evaluates on both DamSegment val and s2ds (OOD).
Reports domain gap at end of training.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from morphograph.data.schema import decode_rgb_mask, NUM_CLASSES, DEFAULT_CE_WEIGHTS
from morphograph.data.graph_targets import mask_to_dt_target
from morphograph.losses.composite import WeightedCEDiceLoss, DTRegressionLoss, LossSchedule
from morphograph.models.morphograph_net import MorphoAuxNet, BASELINE_HEADS
from morphograph.training.utils import (
    set_seed, discover_all_samples, split_data,
    compute_miou, make_cosine_schedule, save_checkpoint,
)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

TIER_TO_DOMAIN = {"Easy": 0, "Medium": 1, "Hard": 2}


def path_to_domain_id(img_path: Path) -> int:
    """Extract pseudo-domain ID from DamSegment image path (difficulty tier)."""
    parts = img_path.parts
    for part in parts:
        if part in TIER_TO_DOMAIN:
            return TIER_TO_DOMAIN[part]
    return 0  # default (e.g., s2ds)


class DamSegmentDTDataset(Dataset):
    """DamSegment dataset with DT targets. Handles both RGB and indexed masks."""

    def __init__(self, pairs, img_size=512, augment=False):
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
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0),
                A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=20, p=1.0),
            ], p=0.5),
            A.GaussNoise(p=0.2),
        ], additional_targets={"dt_target": "mask", "crack_mask": "mask"})

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
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

        crack_binary = (mask == 1).astype(np.uint8)
        dt_target = mask_to_dt_target(crack_binary)

        if self._transform is not None:
            transformed = self._transform(
                image=img, mask=mask, dt_target=dt_target, crack_mask=crack_binary,
            )
            img, mask = transformed["image"], transformed["mask"]
            dt_target, crack_binary = transformed["dt_target"], transformed["crack_mask"]

        return {
            "image": torch.from_numpy(img).permute(2, 0, 1).float() / 255.0,
            "mask": torch.from_numpy(mask.copy()).long(),
            "dt_target": torch.from_numpy(dt_target.copy()).float().unsqueeze(0),
            "crack_mask": torch.from_numpy(crack_binary.copy()).float().unsqueeze(0),
            "domain_id": path_to_domain_id(img_path),
        }


def discover_s2ds_samples(data_root: Path) -> list[tuple[Path, Path]]:
    """Find all image-mask pairs from s2ds."""
    img_dir = data_root / "s2ds" / "images"
    mask_dir = data_root / "s2ds" / "masks"
    if not img_dir.exists():
        return []
    pairs = []
    for img_path in sorted(img_dir.iterdir()):
        if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        mask_path = mask_dir / img_path.name
        if mask_path.exists():
            pairs.append((img_path, mask_path))
    return pairs


@torch.no_grad()
def evaluate(model, loader, device):
    """Evaluate segmentation on a dataset."""
    model.eval()
    all_preds, all_targets = [], []
    for batch in loader:
        images = batch["image"].to(device)
        masks = batch["mask"]
        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            outputs = model(images)
        all_preds.append(outputs["seg"].argmax(dim=1).cpu())
        all_targets.append(masks)
    return compute_miou(torch.cat(all_preds), torch.cat(all_targets))


def main():
    parser = argparse.ArgumentParser(description="DG training: B2 + DG methods")
    parser.add_argument("--data-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--encoder-lr", type=float, default=6e-5)
    parser.add_argument("--head-lr", type=float, default=6e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--amp", action="store_true", default=True)
    # Skeleton config
    parser.add_argument("--no-skeleton", action="store_true", help="B0 mode: no skeleton head")
    parser.add_argument("--skel-weight", type=float, default=1.0)
    parser.add_argument("--skel-loss-type", choices=["smooth_l1", "mse"], default="mse")
    # MixStyle
    parser.add_argument("--mixstyle", action="store_true", help="Enable MixStyle")
    parser.add_argument("--mixstyle-p", type=float, default=0.5, help="MixStyle probability")
    parser.add_argument("--mixstyle-alpha", type=float, default=0.1, help="MixStyle Beta dist alpha")
    parser.add_argument("--mixstyle-stages", type=int, nargs="+", default=[0, 1],
                        help="Encoder stages to apply MixStyle (0-indexed)")
    # CORAL
    parser.add_argument("--coral", action="store_true", help="Enable CORAL alignment")
    parser.add_argument("--coral-weight", type=float, default=1.0, help="CORAL loss weight")
    parser.add_argument("--coral-stage", type=int, default=2, help="Encoder stage for CORAL")
    # DANN
    parser.add_argument("--dann", action="store_true", help="Enable DANN adversarial training")
    parser.add_argument("--dann-weight", type=float, default=0.1, help="DANN loss weight")
    parser.add_argument("--dann-stage", type=int, default=2, help="Encoder stage for DANN")
    args = parser.parse_args()

    set_seed(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Data
    all_pairs = discover_all_samples(args.data_root)
    if not all_pairs:
        print("ERROR: No DamSegment data found.")
        sys.exit(1)
    train_pairs, val_pairs = split_data(all_pairs, args.val_ratio, args.seed)
    s2ds_pairs = discover_s2ds_samples(args.data_root)

    print(f"DamSegment: {len(all_pairs)} total, {len(train_pairs)} train, {len(val_pairs)} val")
    print(f"s2ds (OOD): {len(s2ds_pairs)} images")

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
    s2ds_loader = None
    if s2ds_pairs:
        s2ds_loader = DataLoader(
            DamSegmentDTDataset(s2ds_pairs, augment=False),
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=True,
        )

    # Model
    heads = BASELINE_HEADS["B0" if args.no_skeleton else "B2"]
    model = MorphoAuxNet(
        backbone="mit_b2", num_classes=NUM_CLASSES,
        fpn_dim=256, heads=heads,
    ).to(device)

    # DG methods
    dg_method = "ERM"
    coral_reg = None
    dann_reg = None

    if args.mixstyle:
        from morphograph.models.mixstyle import apply_mixstyle_hooks
        apply_mixstyle_hooks(
            model.encoder, stages=tuple(args.mixstyle_stages),
            p=args.mixstyle_p, alpha=args.mixstyle_alpha,
        )
        dg_method = "MixStyle"
        print(f"MixStyle enabled: stages={args.mixstyle_stages}, p={args.mixstyle_p}, alpha={args.mixstyle_alpha}")

    if args.coral:
        from morphograph.models.coral import CORALRegularizer
        coral_reg = CORALRegularizer(stage=args.coral_stage).to(device)
        dg_method = "CORAL" if dg_method == "ERM" else f"{dg_method}+CORAL"
        print(f"CORAL enabled: stage={args.coral_stage}, weight={args.coral_weight}")

    if args.dann:
        from morphograph.models.dann import DANNRegularizer
        dann_reg = DANNRegularizer(stage=args.dann_stage, num_domains=3).to(device)
        dg_method = "DANN" if dg_method == "ERM" else f"{dg_method}+DANN"
        print(f"DANN enabled: stage={args.dann_stage}, weight={args.dann_weight}")

    param_counts = model.count_parameters()
    print(f"Parameters: {param_counts['total']:,}")
    baseline = "B0" if args.no_skeleton else "B2"
    print(f"Config: {baseline} + {dg_method}")

    # Optimizer
    param_groups = model.get_param_groups(encoder_lr=args.encoder_lr, head_lr=args.head_lr)
    if dann_reg is not None:
        param_groups.append({"params": dann_reg.parameters(), "lr": args.head_lr})
    optimizer = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)
    scheduler = make_cosine_schedule(
        optimizer, len(train_loader) * args.epochs, len(train_loader) * args.warmup_epochs,
    )

    # Losses
    seg_loss_fn = WeightedCEDiceLoss(class_weights=DEFAULT_CE_WEIGHTS, ignore_index=255).to(device)
    skel_loss_fn = DTRegressionLoss(loss_type=args.skel_loss_type).to(device)
    skel_schedule = LossSchedule(weight=args.skel_weight, start_epoch=0, ramp_epochs=0)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")

    # Training
    best_miou_fg = 0.0
    history = defaultdict(list)

    print(f"\nTraining {baseline}+{dg_method} for {args.epochs} epochs\n")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        epoch_losses = defaultdict(list)

        need_hidden = coral_reg is not None or dann_reg is not None
        # Capture hidden states via hook if needed
        captured_hidden = [None]
        if need_hidden and not hasattr(model, "_dg_hook"):
            def capture_hook(module, input, output):
                captured_hidden[0] = list(output.hidden_states)
                return output
            model._dg_hook = model.encoder.register_forward_hook(capture_hook)

        for batch in train_loader:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            dt_targets = batch["dt_target"].to(device)
            crack_masks = batch["crack_mask"].to(device)
            domain_ids = batch["domain_id"].to(device) if need_hidden else None

            with torch.amp.autocast("cuda", enabled=args.amp and device.type == "cuda"):
                outputs = model(images)
                seg_loss = seg_loss_fn(outputs["seg"], masks)["total"]
                total_loss = seg_loss

                if not args.no_skeleton:
                    skel_pred = torch.sigmoid(outputs["skeleton"])
                    skel_loss = skel_loss_fn(skel_pred, dt_targets, torch.ones_like(crack_masks))
                    total_loss = total_loss + skel_schedule.effective_weight(epoch) * skel_loss
                    epoch_losses["skel"].append(skel_loss.item())

                # CORAL/DANN losses (use captured hidden states)
                if coral_reg is not None and captured_hidden[0] is not None:
                    c_loss = coral_reg(captured_hidden[0], domain_ids)
                    total_loss = total_loss + args.coral_weight * c_loss
                    epoch_losses["coral"].append(c_loss.item())

                if dann_reg is not None and captured_hidden[0] is not None:
                    d_loss = dann_reg(captured_hidden[0], domain_ids, epoch, args.epochs)
                    total_loss = total_loss + args.dann_weight * d_loss
                    epoch_losses["dann"].append(d_loss.item())

            optimizer.zero_grad()
            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            epoch_losses["seg"].append(seg_loss.item())
            epoch_losses["total"].append(total_loss.item())

        for k, v in epoch_losses.items():
            history[f"train_{k}"].append(float(np.mean(v)))

        # Validate (in-domain)
        dm_miou = evaluate(model, val_loader, device)
        history["val_mIoU_fg"].append(dm_miou["mIoU_fg"])

        # Validate (OOD) every 10 epochs + last epoch
        s2ds_miou_fg = None
        if s2ds_loader and (epoch % 10 == 0 or epoch == args.epochs):
            s2_miou = evaluate(model, s2ds_loader, device)
            s2ds_miou_fg = s2_miou["mIoU_fg"]
            history["s2ds_mIoU_fg"].append(s2ds_miou_fg)
            history["s2ds_eval_epoch"].append(epoch)

        # Checkpoint
        is_best = dm_miou["mIoU_fg"] > best_miou_fg
        if is_best:
            best_miou_fg = dm_miou["mIoU_fg"]
            save_checkpoint(args.output / "best.pt", model, optimizer, epoch, best_miou_fg, args)
        # Skip last.pt to reduce IO on network FS; best.pt is sufficient

        # Log
        elapsed = time.time() - t0
        loss_str = " ".join(f"{k}={np.mean(v):.4f}" for k, v in epoch_losses.items())
        s2ds_str = f" s2ds={s2ds_miou_fg:.4f}" if s2ds_miou_fg is not None else ""
        print(f"Epoch {epoch:3d}/{args.epochs} | {loss_str} | "
              f"mIoU={dm_miou['mIoU_fg']:.4f}{s2ds_str} | "
              f"{elapsed:.0f}s{' *' if is_best else ''}")

    # Final OOD evaluation with best checkpoint
    print(f"\nLoading best checkpoint for final evaluation...")
    best_ckpt = torch.load(args.output / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(best_ckpt["model_state_dict"])

    final_dm = evaluate(model, val_loader, device)
    final_s2ds = evaluate(model, s2ds_loader, device) if s2ds_loader else None

    print(f"\nFinal results (best checkpoint):")
    print(f"  DamSegment val: mIoU_fg={final_dm['mIoU_fg']:.4f}")
    if final_s2ds:
        gap = final_dm["mIoU_fg"] - final_s2ds["mIoU_fg"]
        print(f"  s2ds OOD:       mIoU_fg={final_s2ds['mIoU_fg']:.4f}")
        print(f"  Domain gap:     {gap:+.4f} ({gap/final_dm['mIoU_fg']*100:+.1f}%)")

    # Save
    with open(args.output / "history.json", "w") as f:
        json.dump(dict(history), f, indent=2)

    summary = {
        "baseline": baseline,
        "dg_method": dg_method,
        "best_miou_fg": best_miou_fg,
        "final_damseg_miou_fg": final_dm["mIoU_fg"],
        "final_damseg_per_class": final_dm["per_class"],
        "final_s2ds_miou_fg": final_s2ds["mIoU_fg"] if final_s2ds else None,
        "final_s2ds_per_class": final_s2ds["per_class"] if final_s2ds else None,
        "domain_gap": (final_dm["mIoU_fg"] - final_s2ds["mIoU_fg"]) if final_s2ds else None,
        "epochs": args.epochs,
        "total_params": param_counts["total"],
        "seed": args.seed,
        "mixstyle": {
            "enabled": args.mixstyle,
            "p": args.mixstyle_p,
            "alpha": args.mixstyle_alpha,
            "stages": args.mixstyle_stages,
        } if args.mixstyle else None,
        "coral": {
            "enabled": args.coral,
            "weight": args.coral_weight,
            "stage": args.coral_stage,
        } if args.coral else None,
        "dann": {
            "enabled": args.dann,
            "weight": args.dann_weight,
            "stage": args.dann_stage,
        } if args.dann else None,
    }
    with open(args.output / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        for k in ["seg", "skel", "coral", "dann", "total"]:
            if f"train_{k}" in history:
                axes[0].plot(history[f"train_{k}"], label=k)
        axes[0].set_title("Train Loss"); axes[0].legend(); axes[0].grid(True, alpha=0.3)

        axes[1].plot(history["val_mIoU_fg"], label="DamSegment val")
        if history["s2ds_mIoU_fg"]:
            axes[1].plot(history["s2ds_eval_epoch"], history["s2ds_mIoU_fg"],
                         "o-", label="s2ds OOD", markersize=4)
        axes[1].set_title("mIoU_fg"); axes[1].legend(); axes[1].grid(True, alpha=0.3)

        axes[2].text(0.1, 0.8, f"{baseline} + {dg_method}", fontsize=14, transform=axes[2].transAxes)
        axes[2].text(0.1, 0.6, f"DamSeg: {final_dm['mIoU_fg']:.4f}", fontsize=12, transform=axes[2].transAxes)
        if final_s2ds:
            axes[2].text(0.1, 0.4, f"s2ds:   {final_s2ds['mIoU_fg']:.4f}", fontsize=12, transform=axes[2].transAxes)
            axes[2].text(0.1, 0.2, f"Gap:    {gap:+.4f}", fontsize=12, transform=axes[2].transAxes)
        axes[2].set_title("Summary"); axes[2].axis("off")
        plt.tight_layout()
        plt.savefig(args.output / "training_curves.png", dpi=150)
        plt.close()
    except Exception as e:
        print(f"Plot failed: {e}")

    print(f"\nSaved to {args.output}/")


if __name__ == "__main__":
    main()
