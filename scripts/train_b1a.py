"""B1a training: B0 + clDice topology loss.

Usage:
    python scripts/train_b1a.py --data-root data/raw --output runs/B1a

B1a adds soft clDice loss on the crack class to encourage topological
connectivity. Same architecture as B0 (seg head only), same training
budget (100 epochs). clDice activates at epoch 40 with 5-epoch ramp.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from morphograph.data.schema import NUM_CLASSES, DEFAULT_CE_WEIGHTS
from morphograph.losses.composite import WeightedCEDiceLoss, SoftCLDiceLoss, LossSchedule
from morphograph.models.morphograph_net import MorphoAuxNet, BASELINE_HEADS
from morphograph.training.utils import (
    set_seed, DamSegmentDataset, discover_all_samples, split_data,
    compute_miou, make_cosine_schedule, save_checkpoint,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="B1a training: B0 + clDice")
    parser.add_argument("--data-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--output", type=Path, default=Path("runs/B1a"))
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
    # clDice schedule
    parser.add_argument("--cldice-weight", type=float, default=0.15)
    parser.add_argument("--cldice-start-epoch", type=int, default=40)
    parser.add_argument("--cldice-ramp-epochs", type=int, default=5)
    parser.add_argument("--cldice-iters", type=int, default=10)
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
        DamSegmentDataset(train_pairs, augment=True),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        DamSegmentDataset(val_pairs, augment=False),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    # ── Model (same architecture as B0) ──
    print("Loading SegFormer-B2 pretrained encoder...")
    model = MorphoAuxNet(
        backbone="mit_b2",
        num_classes=NUM_CLASSES,
        heads=BASELINE_HEADS["B1a"],
    ).to(device)

    param_counts = model.count_parameters()
    print(f"Parameters: {param_counts['total']:,} total")

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

    cldice_loss_fn = SoftCLDiceLoss(
        num_iters=args.cldice_iters, target_class=1,
    ).to(device)

    cldice_schedule = LossSchedule(
        weight=args.cldice_weight,
        start_epoch=args.cldice_start_epoch,
        ramp_epochs=args.cldice_ramp_epochs,
    )

    print(f"\nclDice schedule: weight={args.cldice_weight}, "
          f"start_epoch={args.cldice_start_epoch}, ramp={args.cldice_ramp_epochs}")

    # ── AMP ──
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")

    # ── Training ──
    best_miou_fg = 0.0
    history = {
        "train_loss": [], "train_seg_loss": [], "train_cldice_loss": [],
        "val_loss": [], "val_mIoU_fg": [], "val_mIoU_all": [],
    }

    print(f"\nTraining B1a for {args.epochs} epochs...")
    print(f"  Batches/epoch: {len(train_loader)}")
    print()

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        cldice_w = cldice_schedule.effective_weight(epoch)

        # ── Train ──
        model.train()
        epoch_seg_losses = []
        epoch_cldice_losses = []
        epoch_total_losses = []

        for batch in train_loader:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)

            with torch.amp.autocast("cuda", enabled=args.amp and device.type == "cuda"):
                outputs = model(images)
                seg_loss = seg_loss_fn(outputs["seg"], masks)["total"]

                if cldice_w > 0:
                    # SoftCLDiceLoss internally disables autocast for float32 stability
                    cldice_loss = cldice_loss_fn(outputs["seg"], masks)
                    total_loss = seg_loss + cldice_w * cldice_loss
                else:
                    cldice_loss = torch.tensor(0.0, device=device)
                    total_loss = seg_loss

            optimizer.zero_grad()
            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            epoch_seg_losses.append(seg_loss.item())
            epoch_cldice_losses.append(cldice_loss.item())
            epoch_total_losses.append(total_loss.item())

        avg_seg = np.mean(epoch_seg_losses)
        avg_cldice = np.mean(epoch_cldice_losses)
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
        history["train_cldice_loss"].append(avg_cldice)
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
            f"seg={avg_seg:.4f} clD={avg_cldice:.4f}(w={cldice_w:.3f}) "
            f"total={avg_total:.4f} | "
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
        axes[0].plot(history["train_cldice_loss"], label="clDice")
        axes[0].plot(history["train_loss"], label="total")
        axes[0].set_title("Train Loss")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(history["val_mIoU_fg"], label="mIoU_fg")
        axes[1].plot(history["val_mIoU_all"], label="mIoU_all")
        axes[1].set_title("Validation mIoU")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        axes[2].text(0.1, 0.7, f"Best mIoU_fg: {best_miou_fg:.4f}", fontsize=14, transform=axes[2].transAxes)
        axes[2].text(0.1, 0.5, f"B0 mIoU_fg: 0.673", fontsize=12, transform=axes[2].transAxes)
        axes[2].text(0.1, 0.3, f"Delta: {best_miou_fg - 0.673:+.4f}", fontsize=12, transform=axes[2].transAxes)
        axes[2].set_title("B1a vs B0")
        axes[2].axis("off")
        plt.tight_layout()
        plt.savefig(args.output / "training_curves.png", dpi=150)
        plt.close()
        print(f"\nCurves saved to {args.output / 'training_curves.png'}")
    except Exception as e:
        print(f"Plot failed: {e}")

    # ── Summary ──
    summary = {
        "baseline": "B1a",
        "description": "B0 + clDice (topology loss on crack class)",
        "best_miou_fg": best_miou_fg,
        "b0_miou_fg": 0.673,
        "delta_miou_fg": best_miou_fg - 0.673,
        "final_val_loss": history["val_loss"][-1],
        "epochs": args.epochs,
        "total_params": param_counts["total"],
        "cldice_config": {
            "weight": args.cldice_weight,
            "start_epoch": args.cldice_start_epoch,
            "ramp_epochs": args.cldice_ramp_epochs,
            "num_iters": args.cldice_iters,
        },
        "train_samples": len(train_pairs),
        "val_samples": len(val_pairs),
        "seed": args.seed,
    }
    with open(args.output / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nB1a training complete. Best mIoU_fg = {best_miou_fg:.4f}")
    print(f"Delta vs B0: {best_miou_fg - 0.673:+.4f}")
    print(f"Results saved to {args.output}/")


if __name__ == "__main__":
    main()
