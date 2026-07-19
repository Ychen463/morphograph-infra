"""Topology evaluation: compare B0/B1a/B2 on structural metrics.

Usage:
    python scripts/eval_topology.py --data-root data/raw --checkpoints runs/B0/best.pt runs/B2/best.pt

Computes per-image and aggregate metrics:
  - mIoU_fg (pixel-level baseline)
  - clDice (centerline Dice, crack class)
  - ConnR (connectivity recall, crack class)
  - BF1 (boundary F1, crack class)
  - Crack IoU (per-class)

Also analyzes skeleton target quality to diagnose whether the
auto-generated skeleton labels are reliable enough for supervision.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy import ndimage
from skimage.morphology import skeletonize
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from morphograph.data.schema import decode_rgb_mask, NUM_CLASSES
from morphograph.models.morphograph_net import MorphoAuxNet, BASELINE_HEADS
from morphograph.metrics.segmentation import (
    compute_iou, compute_cldice, compute_connectivity_recall, compute_boundary_f1,
)
from morphograph.training.utils import (
    set_seed, DamSegmentDataset, discover_all_samples, split_data,
)


def load_model(checkpoint_path: Path, device: torch.device) -> MorphoAuxNet:
    """Load model from checkpoint, auto-detecting head config."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = ckpt["model_state_dict"]

    # Detect which heads are present from state dict keys
    has_skeleton = any(k.startswith("skeleton_head.") for k in state)
    has_endpoint = any(k.startswith("endpoint_head.") for k in state)
    has_junction = any(k.startswith("junction_head.") for k in state)
    has_width = any(k.startswith("width_head.") for k in state)

    heads = {
        "seg_head": True,
        "skeleton_head": has_skeleton,
        "endpoint_head": has_endpoint,
        "junction_head": has_junction,
        "width_head": has_width,
    }

    model = MorphoAuxNet(
        backbone="mit_b2",
        num_classes=NUM_CLASSES,
        heads=heads,
    ).to(device)
    model.load_state_dict(state)
    model.eval()
    return model


def evaluate_model(
    model: MorphoAuxNet,
    val_pairs: list[tuple[Path, Path]],
    device: torch.device,
    img_size: int = 512,
) -> dict[str, list[float]]:
    """Run per-image evaluation on validation set."""
    metrics = {
        "iou_crack": [],
        "iou_spalling": [],
        "miou_fg": [],
        "cldice_crack": [],
        "connr_crack": [],
        "bf1_crack": [],
    }

    for img_path, mask_path in val_pairs:
        # Load and preprocess
        img = np.array(Image.open(img_path).convert("RGB").resize(
            (img_size, img_size), Image.BILINEAR,
        ))
        mask_raw = np.array(Image.open(mask_path).resize(
            (img_size, img_size), Image.NEAREST,
        ))
        if mask_raw.ndim == 3:
            gt = decode_rgb_mask(mask_raw)
        else:
            gt = mask_raw.astype(np.uint8)

        # Predict
        img_t = torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0) / 255.0
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            outputs = model(img_t.to(device))
        pred = outputs["seg"].argmax(dim=1)[0].cpu().numpy()

        # IoU
        ious = compute_iou(pred, gt)
        metrics["iou_crack"].append(ious.get(1, 0.0))
        metrics["iou_spalling"].append(ious.get(2, 0.0))
        fg_vals = [v for c, v in ious.items() if c > 0]
        metrics["miou_fg"].append(np.mean(fg_vals) if fg_vals else 0.0)

        # Topology metrics on crack class
        pred_crack = (pred == 1)
        gt_crack = (gt == 1)

        metrics["cldice_crack"].append(compute_cldice(pred_crack, gt_crack))
        metrics["connr_crack"].append(compute_connectivity_recall(pred_crack, gt_crack))

        bf1 = compute_boundary_f1(pred, gt)
        metrics["bf1_crack"].append(bf1.get(1, 0.0))

    return metrics


def analyze_skeleton_quality(
    val_pairs: list[tuple[Path, Path]],
    img_size: int = 512,
    num_samples: int = 50,
) -> dict[str, float]:
    """Analyze auto-generated skeleton target quality.

    Checks for common issues:
    - Spur ratio: fraction of skeleton that are short spurs
    - Coverage: fraction of crack pixels covered by skeleton DT
    - Fragmentation: ratio of skeleton components to crack components
    """
    spur_ratios = []
    coverage_ratios = []
    frag_ratios = []
    skeleton_pixel_fractions = []
    crack_pixel_fractions = []

    for img_path, mask_path in val_pairs[:num_samples]:
        mask_raw = np.array(Image.open(mask_path).resize(
            (img_size, img_size), Image.NEAREST,
        ))
        if mask_raw.ndim == 3:
            mask = decode_rgb_mask(mask_raw)
        else:
            mask = mask_raw.astype(np.uint8)

        crack = (mask == 1).astype(np.uint8)
        if not crack.any():
            continue

        crack_px = crack.sum()
        total_px = mask.size
        crack_pixel_fractions.append(crack_px / total_px)

        skel = skeletonize(crack.astype(bool))
        skel_px = skel.sum()
        if skel_px == 0:
            continue

        skeleton_pixel_fractions.append(skel_px / total_px)

        # Neighbor count for spur detection
        kernel = np.ones((3, 3), dtype=np.uint8)
        kernel[1, 1] = 0
        counts = ndimage.convolve(skel.astype(np.uint8), kernel, mode="constant", cval=0)
        counts = counts * skel.astype(np.uint8)

        # Endpoints (1 neighbor) and their connected short branches
        endpoints = (counts == 1).sum()
        junctions = (counts >= 3).sum()

        # Spur ratio: endpoints / total skeleton pixels (high = many short branches)
        spur_ratios.append(endpoints / skel_px if skel_px > 0 else 0)

        # Coverage: what fraction of crack area is within 2px of skeleton
        if skel.any():
            dt_skel = ndimage.distance_transform_edt(~skel)
            covered = (dt_skel[crack.astype(bool)] <= 2).mean()
            coverage_ratios.append(covered)

        # Fragmentation: skeleton components vs crack components
        _, n_crack_comp = ndimage.label(crack)
        _, n_skel_comp = ndimage.label(skel)
        if n_crack_comp > 0:
            frag_ratios.append(n_skel_comp / n_crack_comp)

    return {
        "num_analyzed": len(crack_pixel_fractions),
        "avg_crack_pixel_fraction": float(np.mean(crack_pixel_fractions)) if crack_pixel_fractions else 0,
        "avg_skeleton_pixel_fraction": float(np.mean(skeleton_pixel_fractions)) if skeleton_pixel_fractions else 0,
        "avg_spur_ratio": float(np.mean(spur_ratios)) if spur_ratios else 0,
        "avg_coverage": float(np.mean(coverage_ratios)) if coverage_ratios else 0,
        "avg_fragmentation": float(np.mean(frag_ratios)) if frag_ratios else 0,
        "median_fragmentation": float(np.median(frag_ratios)) if frag_ratios else 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Topology evaluation")
    parser.add_argument("--data-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--checkpoints", nargs="+", type=Path, required=True,
                        help="Checkpoint paths, e.g. runs/B0/best.pt runs/B2/best.pt")
    parser.add_argument("--labels", nargs="+", type=str, default=None,
                        help="Labels for each checkpoint, e.g. B0 B2")
    parser.add_argument("--output", type=Path, default=Path("runs/eval_topology"))
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    set_seed(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    labels = args.labels or [p.parent.name for p in args.checkpoints]
    assert len(labels) == len(args.checkpoints)

    # ── Data (same val split as training) ──
    all_pairs = discover_all_samples(args.data_root)
    _, val_pairs = split_data(all_pairs, args.val_ratio, args.seed)
    print(f"Evaluating on {len(val_pairs)} validation images\n")

    # ── Skeleton quality analysis ──
    print("=" * 60)
    print("Skeleton Target Quality Analysis")
    print("=" * 60)
    skel_quality = analyze_skeleton_quality(val_pairs)
    for k, v in skel_quality.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")
    print()

    # Diagnose issues
    issues = []
    if skel_quality["avg_spur_ratio"] > 0.15:
        issues.append(f"High spur ratio ({skel_quality['avg_spur_ratio']:.2f}) — many false short branches in skeleton targets")
    if skel_quality["avg_coverage"] < 0.85:
        issues.append(f"Low coverage ({skel_quality['avg_coverage']:.2f}) — skeleton doesn't represent crack well")
    if skel_quality["avg_fragmentation"] > 2.0:
        issues.append(f"High fragmentation ({skel_quality['avg_fragmentation']:.1f}x) — skeleton has more components than crack mask")
    if skel_quality["avg_skeleton_pixel_fraction"] < 0.001:
        issues.append(f"Extremely sparse skeleton ({skel_quality['avg_skeleton_pixel_fraction']:.5f}) — very weak supervision signal")

    if issues:
        print("POTENTIAL ISSUES:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("No major skeleton quality issues detected.")
    print()

    # ── Evaluate each checkpoint ──
    all_results = {}
    for label, ckpt_path in zip(labels, args.checkpoints):
        print(f"{'=' * 60}")
        print(f"Evaluating: {label} ({ckpt_path})")
        print(f"{'=' * 60}")

        model = load_model(ckpt_path, device)
        metrics = evaluate_model(model, val_pairs, device)

        # Aggregate
        agg = {}
        for k, vals in metrics.items():
            agg[k] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
                "median": float(np.median(vals)),
            }

        all_results[label] = agg

        for k, stats in agg.items():
            print(f"  {k:20s}: {stats['mean']:.4f} ± {stats['std']:.4f} (median {stats['median']:.4f})")
        print()

        # Free GPU memory
        del model
        torch.cuda.empty_cache()

    # ── Comparison table ──
    print("=" * 60)
    print("COMPARISON TABLE")
    print("=" * 60)
    header = f"{'Metric':20s}" + "".join(f" | {l:>12s}" for l in labels)
    print(header)
    print("-" * len(header))

    metric_keys = ["miou_fg", "iou_crack", "cldice_crack", "connr_crack", "bf1_crack"]
    for key in metric_keys:
        row = f"{key:20s}"
        values = [all_results[l][key]["mean"] for l in labels]
        best_val = max(values)
        for l in labels:
            val = all_results[l][key]["mean"]
            marker = " *" if val == best_val and len(labels) > 1 else "  "
            row += f" | {val:10.4f}{marker}"
        print(row)
    print()

    # ── Delta analysis ──
    if len(labels) >= 2:
        base = labels[0]
        print(f"Deltas vs {base}:")
        for other in labels[1:]:
            print(f"  {other}:")
            for key in metric_keys:
                delta = all_results[other][key]["mean"] - all_results[base][key]["mean"]
                print(f"    {key:20s}: {delta:+.4f}")
        print()

    # ── Save results ──
    output = {
        "val_samples": len(val_pairs),
        "skeleton_quality": skel_quality,
        "skeleton_issues": issues,
        "results": all_results,
    }
    with open(args.output / "topology_eval.json", "w") as f:
        json.dump(output, f, indent=2)

    print(f"Results saved to {args.output / 'topology_eval.json'}")


if __name__ == "__main__":
    main()
