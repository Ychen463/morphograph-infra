"""B3-B5 training: progressive graph supervision ladder.

Usage:
    python scripts/train_b345.py --baseline B3 --data-root data/raw --output runs/B3
    python scripts/train_b345.py --baseline B4 --data-root data/raw --output runs/B4
    python scripts/train_b345.py --baseline B5 --data-root data/raw --output runs/B5

B3 = B2_best + endpoint/junction heatmap heads
B4 = B3 + edge path connectivity loss
B5 = B4 + width regression head

B2 skeleton config inherits v4_w10 best: MSE, unmasked, weight=10.0.
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
from scipy import ndimage as ndi
from torch.utils.data import Dataset, DataLoader

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from morphograph.data.schema import decode_rgb_mask, NUM_CLASSES, DEFAULT_CE_WEIGHTS
from morphograph.data.graph_targets import (
    mask_to_dt_target, mask_to_skeleton, detect_keypoints,
    build_graph, estimate_width,
)
from morphograph.losses.composite import (
    WeightedCEDiceLoss, DTRegressionLoss, BinaryHeadLoss, WidthRegressionLoss,
)
from morphograph.models.morphograph_net import MorphoAuxNet, BASELINE_HEADS
from morphograph.training.utils import (
    set_seed, discover_all_samples, split_data,
    compute_miou, make_cosine_schedule, save_checkpoint,
)


# ---------------------------------------------------------------------------
# Target generation helpers
# ---------------------------------------------------------------------------

def _make_gaussian_heatmap(
    coords: np.ndarray, shape: tuple[int, int], sigma: float = 3.0,
) -> np.ndarray:
    """Create a Gaussian heatmap from (N, 2) row-col coordinates."""
    heatmap = np.zeros(shape, dtype=np.float32)
    for r, c in coords:
        heatmap[int(r), int(c)] = 1.0
    if heatmap.any():
        heatmap = ndi.gaussian_filter(heatmap, sigma=sigma)
        heatmap = heatmap / (heatmap.max() + 1e-8)  # normalize to [0, 1]
    return heatmap


def _make_edge_map(
    binary_mask: np.ndarray, skeleton: np.ndarray,
    endpoints: np.ndarray, junctions: np.ndarray,
) -> np.ndarray:
    """Create binary edge map from traced graph branches."""
    graph = build_graph(
        skeleton, endpoints, junctions,
        min_branch_length=5, junction_merge_radius=5,
        binary_mask=binary_mask,
    )
    edge_map = np.zeros_like(skeleton, dtype=np.float32)
    for path in graph.edge_paths:
        for r, c in path:
            if 0 <= r < edge_map.shape[0] and 0 <= c < edge_map.shape[1]:
                edge_map[int(r), int(c)] = 1.0
    return edge_map


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class DamSegmentGraphDataset(Dataset):
    """DamSegment dataset with full graph targets for B3-B5."""

    def __init__(
        self,
        pairs: list[tuple[Path, Path]],
        baseline: str,
        img_size: int = 512,
        augment: bool = False,
    ) -> None:
        self.pairs = pairs
        self.baseline = baseline
        self.img_size = img_size
        self.augment = augment
        self._transform = None
        if augment:
            self._transform = self._build_augmentation()

    def _build_augmentation(self):
        import albumentations as A
        extra_targets = {
            "dt_target": "mask", "crack_mask": "mask",
            "ep_heatmap": "mask", "jn_heatmap": "mask",
        }
        if self.baseline in ("B4", "B5"):
            extra_targets["edge_map"] = "mask"
        if self.baseline == "B5":
            extra_targets["width_map"] = "mask"
            extra_targets["skel_mask"] = "mask"
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
        ], additional_targets=extra_targets)

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

        crack_binary = (mask == 1).astype(np.uint8)

        # B2 targets: DT
        dt_target = mask_to_dt_target(crack_binary)

        # Skeleton (thin, for keypoint detection)
        skeleton = mask_to_skeleton(crack_binary, dilate_radius=0)

        # B3 targets: keypoint heatmaps
        endpoints, junctions = detect_keypoints(skeleton)
        ep_heatmap = _make_gaussian_heatmap(endpoints, mask.shape[:2], sigma=3.0)
        jn_heatmap = _make_gaussian_heatmap(junctions, mask.shape[:2], sigma=3.0)

        result = {
            "dt_target": dt_target,
            "crack_mask": crack_binary,
            "ep_heatmap": ep_heatmap,
            "jn_heatmap": jn_heatmap,
        }

        # B4 targets: edge map
        if self.baseline in ("B4", "B5"):
            edge_map = _make_edge_map(crack_binary, skeleton, endpoints, junctions)
            result["edge_map"] = edge_map

        # B5 targets: width map + skeleton mask
        if self.baseline == "B5":
            width_map = estimate_width(crack_binary, skeleton)
            result["width_map"] = width_map.astype(np.float32)
            result["skel_mask"] = skeleton.astype(np.uint8)

        # Apply augmentation
        if self._transform is not None:
            aug_input = {"image": img, "mask": mask}
            aug_input.update(result)
            transformed = self._transform(**aug_input)
            img = transformed["image"]
            mask = transformed["mask"]
            for k in result:
                result[k] = transformed[k]

        # Convert to tensors
        out = {
            "image": torch.from_numpy(img).permute(2, 0, 1).float() / 255.0,
            "mask": torch.from_numpy(mask.copy()).long(),
            "dt_target": torch.from_numpy(result["dt_target"].copy()).float().unsqueeze(0),
            "crack_mask": torch.from_numpy(result["crack_mask"].copy()).float().unsqueeze(0),
            "ep_heatmap": torch.from_numpy(result["ep_heatmap"].copy()).float().unsqueeze(0),
            "jn_heatmap": torch.from_numpy(result["jn_heatmap"].copy()).float().unsqueeze(0),
        }
        if "edge_map" in result:
            out["edge_map"] = torch.from_numpy(result["edge_map"].copy()).float().unsqueeze(0)
        if "width_map" in result:
            out["width_map"] = torch.from_numpy(result["width_map"].copy()).float().unsqueeze(0)
            out["skel_mask"] = torch.from_numpy(result["skel_mask"].copy()).float().unsqueeze(0)

        return out


# ---------------------------------------------------------------------------
# Edge connectivity loss (B4)
# ---------------------------------------------------------------------------

class EdgeConnectivityLoss(torch.nn.Module):
    """Path recall loss: encourages predicted skeleton to be active along GT edge paths.

    For each GT edge path pixel, we want high skeleton probability.
    Loss = 1 - mean(sigmoid(skeleton_logits[edge_pixels])).
    """

    def forward(
        self, skel_logits: torch.Tensor, edge_map: torch.Tensor,
    ) -> torch.Tensor:
        valid = edge_map.bool()
        if not valid.any():
            return torch.tensor(0.0, device=skel_logits.device, requires_grad=True)
        probs = torch.sigmoid(skel_logits[valid])
        return 1.0 - probs.mean()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="B3-B5 training")
    parser.add_argument("--baseline", choices=["B3", "B4", "B5"], required=True)
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
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--amp", action="store_true", default=True)
    # Skeleton DT (inherited from B2 best: MSE, unmasked, w=10)
    parser.add_argument("--skel-weight", type=float, default=10.0)
    parser.add_argument("--skel-loss-type", choices=["smooth_l1", "mse"], default="mse")
    parser.add_argument("--skel-unmask", action="store_true", default=True)
    # B3: keypoint weights
    parser.add_argument("--ep-weight", type=float, default=0.5)
    parser.add_argument("--ep-pos-weight", type=float, default=100.0)
    parser.add_argument("--jn-weight", type=float, default=0.5)
    parser.add_argument("--jn-pos-weight", type=float, default=100.0)
    # B4: edge connectivity weight
    parser.add_argument("--edge-weight", type=float, default=1.0)
    # B5: width regression weight
    parser.add_argument("--width-weight", type=float, default=0.5)
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
        DamSegmentGraphDataset(train_pairs, args.baseline, augment=True),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        DamSegmentGraphDataset(val_pairs, args.baseline, augment=False),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    # ── Model ──
    print(f"Loading SegFormer-B2 pretrained encoder for {args.baseline}...")
    model = MorphoAuxNet(
        backbone="mit_b2",
        num_classes=NUM_CLASSES,
        heads=BASELINE_HEADS[args.baseline],
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

    skel_loss_fn = DTRegressionLoss(loss_type=args.skel_loss_type).to(device)

    ep_loss_fn = BinaryHeadLoss(pos_weight=args.ep_pos_weight, dice_weight=0.5).to(device)
    jn_loss_fn = BinaryHeadLoss(pos_weight=args.jn_pos_weight, dice_weight=0.5).to(device)

    edge_loss_fn = EdgeConnectivityLoss().to(device) if args.baseline in ("B4", "B5") else None
    width_loss_fn = WidthRegressionLoss().to(device) if args.baseline == "B5" else None

    # Print config
    active_losses = [f"seg", f"skel(w={args.skel_weight})",
                     f"ep(w={args.ep_weight})", f"jn(w={args.jn_weight})"]
    if edge_loss_fn:
        active_losses.append(f"edge(w={args.edge_weight})")
    if width_loss_fn:
        active_losses.append(f"width(w={args.width_weight})")
    print(f"\nLosses: {' + '.join(active_losses)}")

    # ── AMP ──
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")

    # ── Training ──
    best_miou_fg = 0.0
    loss_keys = ["train_loss", "train_seg", "train_skel", "train_ep", "train_jn"]
    if edge_loss_fn:
        loss_keys.append("train_edge")
    if width_loss_fn:
        loss_keys.append("train_width")
    history = {k: [] for k in loss_keys}
    history.update({"val_loss": [], "val_mIoU_fg": [], "val_mIoU_all": []})

    print(f"\nTraining {args.baseline} for {args.epochs} epochs...")
    print(f"  Batches/epoch: {len(train_loader)}")
    print()

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        # ── Train ──
        model.train()
        epoch_losses = {k: [] for k in loss_keys}

        for batch in train_loader:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            dt_targets = batch["dt_target"].to(device)
            crack_masks = batch["crack_mask"].to(device)
            ep_targets = batch["ep_heatmap"].to(device)
            jn_targets = batch["jn_heatmap"].to(device)

            with torch.amp.autocast("cuda", enabled=args.amp and device.type == "cuda"):
                outputs = model(images)

                # Seg loss
                seg_loss = seg_loss_fn(outputs["seg"], masks)["total"]

                # Skeleton DT loss (B2 best config)
                skel_pred = torch.sigmoid(outputs["skeleton"])
                skel_mask = torch.ones_like(crack_masks) if args.skel_unmask else crack_masks
                skel_loss = skel_loss_fn(skel_pred, dt_targets, skel_mask)

                # Endpoint + junction loss (B3+)
                ep_loss = ep_loss_fn(outputs["endpoints"], ep_targets)
                jn_loss = jn_loss_fn(outputs["junctions"], jn_targets)

                total_loss = (seg_loss
                              + args.skel_weight * skel_loss
                              + args.ep_weight * ep_loss
                              + args.jn_weight * jn_loss)

                # Edge connectivity loss (B4+)
                edge_loss_val = torch.tensor(0.0, device=device)
                if edge_loss_fn is not None:
                    edge_map = batch["edge_map"].to(device)
                    edge_loss_val = edge_loss_fn(outputs["skeleton"], edge_map)
                    total_loss = total_loss + args.edge_weight * edge_loss_val

                # Width regression loss (B5)
                width_loss_val = torch.tensor(0.0, device=device)
                if width_loss_fn is not None:
                    width_map = batch["width_map"].to(device)
                    skel_mask_target = batch["skel_mask"].to(device)
                    width_loss_val = width_loss_fn(outputs["width"], width_map, skel_mask_target)
                    total_loss = total_loss + args.width_weight * width_loss_val

            optimizer.zero_grad()
            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            epoch_losses["train_loss"].append(total_loss.item())
            epoch_losses["train_seg"].append(seg_loss.item())
            epoch_losses["train_skel"].append(skel_loss.item())
            epoch_losses["train_ep"].append(ep_loss.item())
            epoch_losses["train_jn"].append(jn_loss.item())
            if edge_loss_fn:
                epoch_losses["train_edge"].append(edge_loss_val.item())
            if width_loss_fn:
                epoch_losses["train_width"].append(width_loss_val.item())

        avgs = {k: np.mean(v) for k, v in epoch_losses.items()}
        for k in loss_keys:
            history[k].append(avgs[k])

        # ── Validate ──
        model.eval()
        val_losses = []
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(device)
                masks_val = batch["mask"].to(device)
                with torch.amp.autocast("cuda", enabled=args.amp and device.type == "cuda"):
                    outputs = model(images)
                    val_loss = seg_loss_fn(outputs["seg"], masks_val)["total"]
                val_losses.append(val_loss.item())
                all_preds.append(outputs["seg"].argmax(dim=1).cpu())
                all_targets.append(masks_val.cpu())

        avg_val_loss = np.mean(val_losses)
        miou = compute_miou(torch.cat(all_preds), torch.cat(all_targets))

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
        loss_str = (f"seg={avgs['train_seg']:.4f} skel={avgs['train_skel']:.4f} "
                    f"ep={avgs['train_ep']:.4f} jn={avgs['train_jn']:.4f}")
        if edge_loss_fn:
            loss_str += f" edge={avgs['train_edge']:.4f}"
        if width_loss_fn:
            loss_str += f" width={avgs['train_width']:.4f}"
        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"{loss_str} total={avgs['train_loss']:.4f} | "
            f"val={avg_val_loss:.4f} mIoU_fg={miou['mIoU_fg']:.4f} | "
            f"{per_class} | {elapsed:.0f}s{best_marker}"
        )

    # ── Save history ──
    with open(args.output / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    # ── Plot ──
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Train losses
        axes[0].plot(history["train_seg"], label="seg")
        axes[0].plot(history["train_skel"], label="skel")
        axes[0].plot(history["train_ep"], label="ep")
        axes[0].plot(history["train_jn"], label="jn")
        if "train_edge" in history:
            axes[0].plot(history["train_edge"], label="edge")
        if "train_width" in history:
            axes[0].plot(history["train_width"], label="width")
        axes[0].plot(history["train_loss"], label="total", linestyle="--")
        axes[0].set_title("Train Loss")
        axes[0].legend(fontsize=8)
        axes[0].grid(True, alpha=0.3)

        # Val mIoU
        axes[1].plot(history["val_mIoU_fg"], label="mIoU_fg")
        axes[1].plot(history["val_mIoU_all"], label="mIoU_all")
        axes[1].set_title("Validation mIoU")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        # Summary text
        axes[2].text(0.1, 0.8, f"Best mIoU_fg: {best_miou_fg:.4f}", fontsize=14, transform=axes[2].transAxes)
        axes[2].text(0.1, 0.6, f"B0 mIoU_fg:  0.673", fontsize=12, transform=axes[2].transAxes)
        axes[2].text(0.1, 0.4, f"B2 best:     0.683", fontsize=12, transform=axes[2].transAxes)
        axes[2].text(0.1, 0.2, f"Delta vs B0: {best_miou_fg - 0.673:+.4f}", fontsize=12, transform=axes[2].transAxes)
        axes[2].set_title(f"{args.baseline} vs B0/B2")
        axes[2].axis("off")
        plt.tight_layout()
        plt.savefig(args.output / "training_curves.png", dpi=150)
        plt.close()
        print(f"\nCurves saved to {args.output / 'training_curves.png'}")
    except Exception as e:
        print(f"Plot failed: {e}")

    # ── Summary ──
    summary = {
        "baseline": args.baseline,
        "best_miou_fg": best_miou_fg,
        "b0_miou_fg": 0.673,
        "b2_best_miou_fg": 0.683,
        "delta_vs_b0": best_miou_fg - 0.673,
        "delta_vs_b2": best_miou_fg - 0.683,
        "final_val_loss": history["val_loss"][-1],
        "epochs": args.epochs,
        "total_params": param_counts["total"],
        "config": {
            "skel_weight": args.skel_weight,
            "skel_loss_type": args.skel_loss_type,
            "skel_unmask": args.skel_unmask,
            "ep_weight": args.ep_weight,
            "ep_pos_weight": args.ep_pos_weight,
            "jn_weight": args.jn_weight,
            "jn_pos_weight": args.jn_pos_weight,
            "edge_weight": args.edge_weight if args.baseline in ("B4", "B5") else None,
            "width_weight": args.width_weight if args.baseline == "B5" else None,
        },
        "train_samples": len(train_pairs),
        "val_samples": len(val_pairs),
        "seed": args.seed,
    }
    with open(args.output / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{args.baseline} training complete. Best mIoU_fg = {best_miou_fg:.4f}")
    print(f"Delta vs B0: {best_miou_fg - 0.673:+.4f}")
    print(f"Delta vs B2: {best_miou_fg - 0.683:+.4f}")
    print(f"Results saved to {args.output}/")


if __name__ == "__main__":
    main()
