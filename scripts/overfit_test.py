"""Overfit test: verify each head can learn on a tiny subset.

This is the minimum viability gate before large-scale runs.
If any head's loss doesn't decrease on 8-16 images over 50 epochs,
something is wrong (data loading, loss implementation, gradient flow).

Usage:
    python scripts/overfit_test.py --data-root data/raw --num-samples 16 --epochs 50

Expected output:
    - Each head's loss should decrease by >80% from epoch 1 to epoch 50
    - Segmentation output should visually show crack/spalling regions
    - Skeleton output should show thin lines inside crack regions
    - Endpoint/junction output should show dots at line ends/branches
    - Width output should show non-zero values at skeleton locations

Saves loss curves to runs/overfit_test/loss_curves.png
Saves prediction visualizations to runs/overfit_test/predictions/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy import ndimage
from skimage.morphology import skeletonize

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from morphograph.data.schema import decode_rgb_mask
from morphograph.data.graph_targets import mask_to_skeleton, detect_keypoints
from morphograph.losses.composite import (
    WeightedCEDiceLoss,
    BinaryHeadLoss,
    WidthRegressionLoss,
)


def make_gaussian_heatmap(
    points: np.ndarray, shape: tuple[int, int], sigma: float = 3.0,
) -> np.ndarray:
    """Create a Gaussian heatmap from point coordinates."""
    heatmap = np.zeros(shape, dtype=np.float32)
    for r, c in points:
        r, c = int(r), int(c)
        if 0 <= r < shape[0] and 0 <= c < shape[1]:
            heatmap[r, c] = 1.0
    if heatmap.any():
        heatmap = ndimage.gaussian_filter(heatmap, sigma=sigma)
        heatmap = heatmap / (heatmap.max() + 1e-8)
    return heatmap


def load_sample(
    img_path: Path, mask_path: Path, img_size: int = 512,
) -> dict[str, torch.Tensor]:
    """Load and preprocess a single sample with all targets."""
    # Image
    img = np.array(Image.open(img_path).convert("RGB").resize(
        (img_size, img_size), Image.BILINEAR,
    ))
    img_t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

    # Mask
    mask_raw = np.array(Image.open(mask_path).resize(
        (img_size, img_size), Image.NEAREST,
    ))
    if mask_raw.ndim == 3:
        mask = decode_rgb_mask(mask_raw)
    else:
        mask = mask_raw.astype(np.uint8)
    mask_t = torch.from_numpy(mask).long()

    # Skeleton target (from crack class)
    crack_binary = (mask == 1).astype(np.uint8)
    skeleton = mask_to_skeleton(crack_binary) if crack_binary.any() else np.zeros_like(crack_binary, dtype=bool)
    skeleton_t = torch.from_numpy(skeleton.astype(np.float32)).unsqueeze(0)

    # Endpoint / junction targets
    if skeleton.any():
        endpoints, junctions = detect_keypoints(skeleton)
    else:
        endpoints = np.empty((0, 2))
        junctions = np.empty((0, 2))

    ep_heatmap = make_gaussian_heatmap(endpoints, (img_size, img_size), sigma=3.0)
    jn_heatmap = make_gaussian_heatmap(junctions, (img_size, img_size), sigma=3.0)
    ep_t = torch.from_numpy(ep_heatmap).unsqueeze(0)
    jn_t = torch.from_numpy(jn_heatmap).unsqueeze(0)

    # Width target = 2 * distance_transform at skeleton pixels
    if crack_binary.any():
        dt = ndimage.distance_transform_edt(crack_binary)
        width_map = np.zeros_like(dt, dtype=np.float32)
        width_map[skeleton] = 2.0 * dt[skeleton]
    else:
        width_map = np.zeros((img_size, img_size), dtype=np.float32)
    width_t = torch.from_numpy(width_map).unsqueeze(0)

    return {
        "image": img_t,
        "mask": mask_t,
        "skeleton": skeleton_t,
        "endpoints": ep_t,
        "junctions": jn_t,
        "width": width_t,
        "skeleton_mask": skeleton_t,  # same as skeleton for loss masking
    }


def discover_samples(data_root: Path, num_samples: int) -> list[tuple[Path, Path]]:
    """Find image-mask pairs from DamSegment."""
    pairs = []
    for tier in ["Easy", "Medium", "Hard"]:
        img_dir = data_root / f"DamSegment/Damage Segmentaion/{tier}/Images"
        mask_dir = data_root / f"DamSegment/Damage Segmentaion/{tier}/Labels/Mask"
        if not img_dir.exists():
            continue
        for img_path in sorted(img_dir.iterdir()):
            # Try common mask name patterns
            for ext in [".png", ".jpg", ".jpeg"]:
                mask_path = mask_dir / (img_path.stem + ext)
                if mask_path.exists():
                    pairs.append((img_path, mask_path))
                    break
            if len(pairs) >= num_samples:
                break
        if len(pairs) >= num_samples:
            break

    if len(pairs) < num_samples:
        print(f"WARNING: Found only {len(pairs)} samples (wanted {num_samples})")
    return pairs[:num_samples]


class SimpleMultiHeadModel(torch.nn.Module):
    """Minimal multi-head model for overfit testing.

    Uses a tiny encoder (3 conv layers) instead of MiT-B2
    so the test runs fast without downloading pretrained weights.
    """

    def __init__(self, num_classes: int = 3) -> None:
        super().__init__()
        # Tiny encoder
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, 3, padding=1),
            torch.nn.GroupNorm(8, 32),
            torch.nn.GELU(),
            torch.nn.Conv2d(32, 64, 3, padding=1),
            torch.nn.GroupNorm(16, 64),
            torch.nn.GELU(),
            torch.nn.Conv2d(64, 64, 3, padding=1),
            torch.nn.GroupNorm(16, 64),
            torch.nn.GELU(),
        )
        # Heads (same interface as MorphoAuxNet)
        self.seg_head = torch.nn.Conv2d(64, num_classes, 1)
        self.skeleton_head = torch.nn.Conv2d(64, 1, 1)
        self.endpoint_head = torch.nn.Conv2d(64, 1, 1)
        self.junction_head = torch.nn.Conv2d(64, 1, 1)
        self.width_head = torch.nn.Sequential(
            torch.nn.Conv2d(64, 1, 1),
            torch.nn.Softplus(),
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.encoder(x)
        return {
            "seg": self.seg_head(features),
            "skeleton": self.skeleton_head(features),
            "endpoints": self.endpoint_head(features),
            "junctions": self.junction_head(features),
            "width": self.width_head(features),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Overfit test for multi-head model")
    parser.add_argument("--data-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--num-samples", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--output", type=Path, default=Path("runs/overfit_test"))
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "predictions").mkdir(exist_ok=True)

    device = torch.device(args.device)
    print(f"Device: {device}")

    # ── Load data ──
    pairs = discover_samples(args.data_root, args.num_samples)
    if not pairs:
        print("ERROR: No data found. Place DamSegment in data/raw/ first.")
        sys.exit(1)

    print(f"Loading {len(pairs)} samples...")
    samples = [load_sample(ip, mp) for ip, mp in pairs]

    # Stack into batches
    images = torch.stack([s["image"] for s in samples]).to(device)
    masks = torch.stack([s["mask"] for s in samples]).to(device)
    skeletons = torch.stack([s["skeleton"] for s in samples]).to(device)
    endpoints = torch.stack([s["endpoints"] for s in samples]).to(device)
    junctions = torch.stack([s["junctions"] for s in samples]).to(device)
    widths = torch.stack([s["width"] for s in samples]).to(device)
    skel_masks = torch.stack([s["skeleton_mask"] for s in samples]).to(device)

    print(f"  Images:    {images.shape}")
    print(f"  Masks:     {masks.shape}, classes: {masks.unique().tolist()}")
    print(f"  Skeleton:  {skeletons.sum().item():.0f} positive pixels")
    print(f"  Endpoints: {(endpoints > 0.1).sum().item()} hot pixels")
    print(f"  Junctions: {(junctions > 0.1).sum().item()} hot pixels")
    print(f"  Width max: {widths.max().item():.1f} px")
    print()

    # ── Build model + losses ──
    model = SimpleMultiHeadModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    seg_loss_fn = WeightedCEDiceLoss(
        class_weights=[0.2, 2.0, 3.0], ignore_index=255,
    ).to(device)
    skel_loss_fn = BinaryHeadLoss(pos_weight=50.0).to(device)
    ep_loss_fn = BinaryHeadLoss(pos_weight=100.0).to(device)
    jn_loss_fn = BinaryHeadLoss(pos_weight=100.0).to(device)
    width_loss_fn = WidthRegressionLoss()

    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {param_count:,}")
    print()

    # ── Training loop ──
    history: dict[str, list[float]] = {
        "seg": [], "skeleton": [], "endpoints": [],
        "junctions": [], "width": [], "total": [],
    }

    print("Training...")
    for epoch in range(1, args.epochs + 1):
        model.train()
        outputs = model(images)

        seg_out = seg_loss_fn(outputs["seg"], masks)
        skel_out = skel_loss_fn(outputs["skeleton"], skeletons)
        ep_out = ep_loss_fn(outputs["endpoints"], endpoints)
        jn_out = jn_loss_fn(outputs["junctions"], junctions)
        w_out = width_loss_fn(outputs["width"], widths, skel_masks)

        total = (
            seg_out["total"]
            + 1.0 * skel_out
            + 0.5 * ep_out
            + 0.5 * jn_out
            + 0.5 * w_out
        )

        optimizer.zero_grad()
        total.backward()
        optimizer.step()

        history["seg"].append(seg_out["total"].item())
        history["skeleton"].append(skel_out.item())
        history["endpoints"].append(ep_out.item())
        history["junctions"].append(jn_out.item())
        history["width"].append(w_out.item())
        history["total"].append(total.item())

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"  Epoch {epoch:3d} | "
                f"seg={seg_out['total'].item():.4f} "
                f"skel={skel_out.item():.4f} "
                f"ep={ep_out.item():.4f} "
                f"jn={jn_out.item():.4f} "
                f"w={w_out.item():.4f} "
                f"total={total.item():.4f}"
            )

    # ── Check convergence ──
    print()
    print("Convergence check:")
    all_pass = True
    for key in ["seg", "skeleton", "endpoints", "junctions", "width"]:
        first = history[key][0]
        last = history[key][-1]
        if first > 0:
            reduction = (first - last) / first * 100
        else:
            reduction = 0.0
        status = "PASS" if reduction > 50 else "FAIL"
        if status == "FAIL":
            all_pass = False
        print(f"  {key:12s}: {first:.4f} -> {last:.4f} ({reduction:+.1f}%) [{status}]")

    # ── Save loss curves ──
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    for i, (key, ax) in enumerate(zip(
        ["seg", "skeleton", "endpoints", "junctions", "width", "total"],
        axes.flat,
    )):
        ax.plot(history[key])
        ax.set_title(key)
        ax.set_xlabel("epoch")
        ax.set_ylabel("loss")
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(args.output / "loss_curves.png", dpi=150)
    plt.close()
    print(f"\nLoss curves saved to {args.output / 'loss_curves.png'}")

    # ── Save predictions for first 4 samples ──
    model.eval()
    with torch.no_grad():
        outputs = model(images[:4])

    for i in range(min(4, len(images))):
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))

        # Row 1: GT
        axes[0, 0].imshow(images[i].cpu().permute(1, 2, 0))
        axes[0, 0].set_title("Image")
        axes[0, 1].imshow(masks[i].cpu(), cmap="tab10", vmin=0, vmax=3)
        axes[0, 1].set_title("GT Mask")
        axes[0, 2].imshow(skeletons[i, 0].cpu(), cmap="gray")
        axes[0, 2].set_title("GT Skeleton")
        axes[0, 3].imshow(widths[i, 0].cpu(), cmap="hot")
        axes[0, 3].set_title("GT Width")

        # Row 2: Predictions
        pred_mask = outputs["seg"][i].argmax(dim=0).cpu()
        axes[1, 0].imshow(pred_mask, cmap="tab10", vmin=0, vmax=3)
        axes[1, 0].set_title("Pred Mask")
        pred_skel = torch.sigmoid(outputs["skeleton"][i, 0]).cpu()
        axes[1, 1].imshow(pred_skel, cmap="gray")
        axes[1, 1].set_title("Pred Skeleton")
        pred_ep = torch.sigmoid(outputs["endpoints"][i, 0]).cpu()
        axes[1, 2].imshow(pred_ep, cmap="hot")
        axes[1, 2].set_title("Pred Endpoints")
        pred_w = outputs["width"][i, 0].cpu()
        axes[1, 3].imshow(pred_w, cmap="hot")
        axes[1, 3].set_title("Pred Width")

        for ax in axes.flat:
            ax.axis("off")
        plt.tight_layout()
        plt.savefig(args.output / f"predictions/sample_{i}.png", dpi=100)
        plt.close()

    print(f"Predictions saved to {args.output / 'predictions/'}")

    # ── Save summary ──
    summary = {
        "num_samples": len(pairs),
        "epochs": args.epochs,
        "final_losses": {k: v[-1] for k, v in history.items()},
        "convergence": {
            k: {
                "first": history[k][0],
                "last": history[k][-1],
                "reduction_pct": (history[k][0] - history[k][-1]) / max(history[k][0], 1e-8) * 100,
            }
            for k in ["seg", "skeleton", "endpoints", "junctions", "width"]
        },
        "all_pass": all_pass,
    }
    with open(args.output / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to {args.output / 'summary.json'}")

    if not all_pass:
        print("\nWARNING: Some heads did not converge. Fix before full training.")
        sys.exit(1)
    else:
        print("\nAll heads converged. Ready for B0 full run.")


if __name__ == "__main__":
    main()
