"""Visualize DT targets on real DamSegment images.

Generates a grid for each sample: [original image | class mask | DT target | skeleton]
Also prints per-image DT statistics (max DT value, mean, crack width estimate).

Usage:
    python scripts/visualize_dt_targets.py --data-root data/raw --output runs/viz_dt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from morphograph.data.schema import decode_rgb_mask
from morphograph.data.graph_targets import mask_to_dt_target, mask_to_skeleton
from morphograph.training.utils import discover_all_samples, split_data


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize DT targets")
    parser.add_argument("--data-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--output", type=Path, default=Path("runs/viz_dt"))
    parser.add_argument("--num-samples", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--img-size", type=int, default=512)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    all_pairs = discover_all_samples(args.data_root)
    _, val_pairs = split_data(all_pairs, 0.15, args.seed)

    # Pick samples WITH crack pixels, spread across tiers
    selected = []
    for img_path, mask_path in val_pairs:
        mask_raw = np.array(Image.open(mask_path).resize(
            (args.img_size, args.img_size), Image.NEAREST,
        ))
        if mask_raw.ndim == 3:
            mask = decode_rgb_mask(mask_raw)
        else:
            mask = mask_raw.astype(np.uint8)
        crack = (mask == 1).astype(np.uint8)
        if crack.sum() > 100:  # at least 100 crack pixels
            selected.append((img_path, mask_path))
        if len(selected) >= args.num_samples:
            break

    print(f"Selected {len(selected)} samples with crack pixels")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy import ndimage

    # Per-sample visualization
    all_stats = []
    for i, (img_path, mask_path) in enumerate(selected):
        img = np.array(Image.open(img_path).convert("RGB").resize(
            (args.img_size, args.img_size), Image.BILINEAR,
        ))
        mask_raw = np.array(Image.open(mask_path).resize(
            (args.img_size, args.img_size), Image.NEAREST,
        ))
        if mask_raw.ndim == 3:
            mask = decode_rgb_mask(mask_raw)
        else:
            mask = mask_raw.astype(np.uint8)

        crack = (mask == 1).astype(np.uint8)
        dt_target = mask_to_dt_target(crack)
        skel = mask_to_skeleton(crack, dilate_radius=0).astype(np.uint8)

        # Raw (unnormalized) DT for width stats
        crack_filled = ndimage.binary_fill_holes(crack.astype(bool))
        dt_raw = ndimage.distance_transform_edt(crack_filled)
        dt_raw_crack = dt_raw[crack.astype(bool)]

        # Stats
        crack_frac = crack.sum() / mask.size
        dt_positive = dt_target[dt_target > 0]
        stats = {
            "idx": i,
            "name": img_path.stem,
            "tier": img_path.parts[-3] if len(img_path.parts) >= 3 else "?",
            "crack_frac": crack_frac,
            "crack_px": int(crack.sum()),
            "dt_max_raw": float(dt_raw_crack.max()) if len(dt_raw_crack) > 0 else 0,
            "dt_mean_raw": float(dt_raw_crack.mean()) if len(dt_raw_crack) > 0 else 0,
            "dt_mean_norm": float(dt_positive.mean()) if len(dt_positive) > 0 else 0,
            "est_width_px": float(2 * dt_raw_crack.max()) if len(dt_raw_crack) > 0 else 0,
            "n_components": int(ndimage.label(crack)[1]),
            "skel_coverage": float((skel & crack).sum() / max(crack.sum(), 1)),
        }
        all_stats.append(stats)

        # Plot 4-panel
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))

        axes[0].imshow(img)
        axes[0].set_title(f"Image ({stats['tier']})")
        axes[0].axis("off")

        # Class mask: BG=black, crack=red, spalling=blue
        mask_vis = np.zeros((*mask.shape, 3), dtype=np.uint8)
        mask_vis[mask == 1] = [255, 0, 0]    # crack = red
        mask_vis[mask == 2] = [0, 0, 255]    # spalling = blue
        axes[1].imshow(mask_vis)
        axes[1].set_title(f"Mask (crack={stats['crack_frac']:.1%}, {stats['n_components']} comp)")
        axes[1].axis("off")

        # DT target heatmap
        im = axes[2].imshow(dt_target, cmap="hot", vmin=0, vmax=1)
        axes[2].set_title(f"DT Target (max_raw={stats['dt_max_raw']:.1f}px, "
                         f"width≈{stats['est_width_px']:.0f}px)")
        axes[2].axis("off")
        plt.colorbar(im, ax=axes[2], fraction=0.046)

        # Skeleton overlay on crack mask
        overlay = np.zeros((*mask.shape, 3), dtype=np.uint8)
        overlay[crack == 1] = [100, 100, 100]  # crack = gray
        overlay[skel == 1] = [0, 255, 0]       # skeleton = green
        axes[3].imshow(overlay)
        axes[3].set_title(f"Skeleton (cov={stats['skel_coverage']:.0%})")
        axes[3].axis("off")

        plt.suptitle(f"Sample {i}: {img_path.stem}", fontsize=12)
        plt.tight_layout()
        plt.savefig(args.output / f"sample_{i:02d}.png", dpi=150, bbox_inches="tight")
        plt.close()

    # Summary stats
    print("\n" + "=" * 80)
    print("DT TARGET STATISTICS (per-sample)")
    print("=" * 80)
    print(f"{'Idx':>3s} {'Tier':>6s} {'Crack%':>7s} {'CrackPx':>8s} {'DTmax':>6s} "
          f"{'DTmean':>7s} {'Width':>6s} {'Comp':>4s} {'SkelCov':>8s} {'DTnorm':>7s}")
    print("-" * 80)
    for s in all_stats:
        print(f"{s['idx']:3d} {s['tier']:>6s} {s['crack_frac']:6.1%} {s['crack_px']:8d} "
              f"{s['dt_max_raw']:6.1f} {s['dt_mean_raw']:7.2f} {s['est_width_px']:6.0f} "
              f"{s['n_components']:4d} {s['skel_coverage']:7.1%} {s['dt_mean_norm']:7.3f}")

    # Aggregate
    print("\n" + "=" * 80)
    print("AGGREGATE")
    print("=" * 80)
    dt_maxes = [s["dt_max_raw"] for s in all_stats]
    dt_means = [s["dt_mean_raw"] for s in all_stats]
    dt_norms = [s["dt_mean_norm"] for s in all_stats]
    widths = [s["est_width_px"] for s in all_stats]
    crack_fracs = [s["crack_frac"] for s in all_stats]
    skel_covs = [s["skel_coverage"] for s in all_stats]

    print(f"  Crack fraction:   {np.mean(crack_fracs):.1%} ± {np.std(crack_fracs):.1%}")
    print(f"  DT max (raw px):  {np.mean(dt_maxes):.1f} ± {np.std(dt_maxes):.1f}  "
          f"(range {np.min(dt_maxes):.1f} - {np.max(dt_maxes):.1f})")
    print(f"  DT mean (raw px): {np.mean(dt_means):.2f} ± {np.std(dt_means):.2f}")
    print(f"  DT mean (norm):   {np.mean(dt_norms):.3f} ± {np.std(dt_norms):.3f}")
    print(f"  Est width (px):   {np.mean(widths):.1f} ± {np.std(widths):.1f}")
    print(f"  Skel coverage:    {np.mean(skel_covs):.1%} ± {np.std(skel_covs):.1%}")

    # Summary plot: DT value distribution across all samples
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].hist(dt_maxes, bins=20, edgecolor="black")
    axes[0].set_xlabel("Max DT (raw px)")
    axes[0].set_title("Crack Width Distribution\n(max DT per image)")
    axes[0].axvline(np.mean(dt_maxes), color="red", linestyle="--",
                    label=f"mean={np.mean(dt_maxes):.1f}")
    axes[0].legend()

    axes[1].hist(dt_norms, bins=20, edgecolor="black")
    axes[1].set_xlabel("Mean normalized DT")
    axes[1].set_title("Normalized DT Distribution\n(mean per image, crack pixels)")
    axes[1].axvline(np.mean(dt_norms), color="red", linestyle="--",
                    label=f"mean={np.mean(dt_norms):.3f}")
    axes[1].legend()

    axes[2].bar(range(len(all_stats)),
                [s["crack_frac"] * 100 for s in all_stats], color="coral")
    axes[2].set_xlabel("Sample index")
    axes[2].set_ylabel("Crack %")
    axes[2].set_title("Crack Area per Sample")

    plt.tight_layout()
    plt.savefig(args.output / "summary_stats.png", dpi=150)
    plt.close()

    print(f"\nVisualizations saved to {args.output}/")
    print(f"  {len(selected)} per-sample panels: sample_00.png - sample_{len(selected)-1:02d}.png")
    print(f"  Summary histogram: summary_stats.png")


if __name__ == "__main__":
    main()
