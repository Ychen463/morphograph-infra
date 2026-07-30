"""Cross-domain evaluation: DamSegment-trained models on s2ds (OOD).

Establishes ERM baselines (D0/D1) for P4 domain generalization.
Evaluates semantic segmentation (per-class IoU, mIoU) and spalling
instance metrics on the s2ds dataset.

Usage:
    python scripts/eval_cross_domain.py \
        --data-root data/raw \
        --checkpoints runs/B0/best.pt runs/B2_dt_v5/best.pt \
        --labels B0 B2 \
        --output runs/eval_cross_domain
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from morphograph.data.schema import NUM_CLASSES
from morphograph.metrics.segmentation import compute_iou, compute_dice, compute_boundary_f1
from morphograph.metrics.instance_metrics import compute_instance_metrics
from morphograph.training.utils import DamSegmentDataset, discover_all_samples, split_data


def discover_s2ds_samples(data_root: Path) -> list[tuple[Path, Path]]:
    """Find all image-mask pairs from s2ds."""
    img_dir = data_root / "s2ds" / "images"
    mask_dir = data_root / "s2ds" / "masks"
    if not img_dir.exists():
        print(f"ERROR: s2ds images not found at {img_dir}")
        sys.exit(1)
    pairs = []
    for img_path in sorted(img_dir.iterdir()):
        if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        mask_path = mask_dir / img_path.name
        if mask_path.exists():
            pairs.append((img_path, mask_path))
    return pairs


def load_model(checkpoint_path: Path, device: torch.device):
    """Load model from checkpoint, auto-detecting architecture."""
    from morphograph.models.morphograph_net import MorphoAuxNet, MorphoGraphNet, FPN_DIM

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt

    has_graph_heads = any("node_heatmap_head" in k or "edge_classifier" in k for k in state)

    if has_graph_heads:
        model = MorphoGraphNet(
            backbone="mit_b2", num_classes=NUM_CLASSES,
            fpn_dim=FPN_DIM, graph_heads=True,
        )
    else:
        head_flags = {}
        for name in ["seg", "skeleton", "endpoints", "junctions", "width"]:
            head_flags[name] = any(name in k for k in state)
        model = MorphoAuxNet(
            backbone="mit_b2", num_classes=NUM_CLASSES,
            fpn_dim=FPN_DIM, heads=head_flags,
        )

    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def evaluate_on_dataset(
    model, loader, device, dataset_name: str,
) -> dict:
    """Run inference and compute metrics on a dataset."""
    all_iou = defaultdict(list)
    all_dice = defaultdict(list)
    all_bf1 = defaultdict(list)
    inst_metrics = []
    per_image = []

    for batch in loader:
        images = batch["image"].to(device)
        masks = batch["mask"].numpy()

        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            outputs = model(images)

        preds = outputs["seg"].argmax(dim=1).cpu().numpy()

        for i in range(len(images)):
            gt = masks[i]
            pred = preds[i]

            iou = compute_iou(pred, gt)
            dice = compute_dice(pred, gt)
            bf1 = compute_boundary_f1(pred, gt, tolerance_px=2)

            for c, v in iou.items():
                all_iou[c].append(v)
            for c, v in dice.items():
                all_dice[c].append(v)
            for c, v in bf1.items():
                all_bf1[c].append(v)

            # Spalling instance metrics
            im = compute_instance_metrics(pred, gt, class_id=2, iou_threshold=0.5)
            inst_metrics.append(im)

            per_image.append({
                "iou_bg": iou.get(0, 0), "iou_crack": iou.get(1, 0), "iou_spalling": iou.get(2, 0),
                "dice_crack": dice.get(1, 0), "dice_spalling": dice.get(2, 0),
                "bf1_crack": bf1.get(1, 0), "bf1_spalling": bf1.get(2, 0),
                "inst_n_gt": im.n_gt, "inst_n_pred": im.n_pred,
                "inst_f1": im.f1,
            })

    # Aggregate
    class_names = {0: "background", 1: "crack", 2: "spalling"}
    metrics = {"dataset": dataset_name, "n_images": len(per_image)}

    for c in range(NUM_CLASSES):
        name = class_names[c]
        metrics[f"iou_{name}"] = float(np.mean(all_iou[c])) if all_iou[c] else 0.0
        metrics[f"dice_{name}"] = float(np.mean(all_dice[c])) if all_dice[c] else 0.0

    fg_ious = []
    for c in [1, 2]:
        if all_iou[c]:
            fg_ious.append(np.mean(all_iou[c]))
    metrics["mIoU_fg"] = float(np.mean(fg_ious)) if fg_ious else 0.0
    metrics["mIoU_all"] = float(np.mean([np.mean(all_iou[c]) for c in range(NUM_CLASSES) if all_iou[c]]))

    for c in [1, 2]:
        name = class_names[c]
        metrics[f"bf1_{name}"] = float(np.mean(all_bf1[c])) if all_bf1[c] else 0.0

    # Instance metrics
    total_tp = sum(m.tp for m in inst_metrics)
    total_fp = sum(m.fp for m in inst_metrics)
    total_fn = sum(m.fn for m in inst_metrics)
    micro_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    micro_r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    metrics["inst_micro_f1"] = 2 * micro_p * micro_r / (micro_p + micro_r) if (micro_p + micro_r) > 0 else 0.0
    metrics["inst_n_gt_total"] = sum(m.n_gt for m in inst_metrics)
    metrics["inst_n_pred_total"] = sum(m.n_pred for m in inst_metrics)

    return metrics, per_image


def main():
    parser = argparse.ArgumentParser(description="Cross-domain evaluation")
    parser.add_argument("--data-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    parser.add_argument("--labels", type=str, nargs="+", required=True)
    parser.add_argument("--output", type=Path, default=Path("runs/eval_cross_domain"))
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    assert len(args.checkpoints) == len(args.labels), "Must provide one label per checkpoint"
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    # Datasets
    damseg_pairs = discover_all_samples(args.data_root)
    _, damseg_val = split_data(damseg_pairs, args.val_ratio, args.seed)
    s2ds_pairs = discover_s2ds_samples(args.data_root)

    print(f"DamSegment val: {len(damseg_val)} images")
    print(f"s2ds (OOD):     {len(s2ds_pairs)} images")

    damseg_loader = DataLoader(
        DamSegmentDataset(damseg_val, augment=False),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )
    s2ds_loader = DataLoader(
        DamSegmentDataset(s2ds_pairs, augment=False),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    # Evaluate each checkpoint
    all_results = []
    for ckpt_path, label in zip(args.checkpoints, args.labels):
        print(f"\n{'='*60}")
        print(f"Model: {label} ({ckpt_path})")
        print(f"{'='*60}")

        model = load_model(ckpt_path, device)

        # In-domain (DamSegment val)
        print(f"\n  DamSegment val ({len(damseg_val)} images):")
        dm_metrics, dm_per_image = evaluate_on_dataset(model, damseg_loader, device, "damsegment_val")
        dm_metrics["model"] = label
        print(f"    mIoU_fg={dm_metrics['mIoU_fg']:.4f}  "
              f"crack={dm_metrics['iou_crack']:.4f}  spalling={dm_metrics['iou_spalling']:.4f}  "
              f"inst_F1={dm_metrics['inst_micro_f1']:.3f}")

        # Cross-domain (s2ds)
        print(f"\n  s2ds OOD ({len(s2ds_pairs)} images):")
        s2_metrics, s2_per_image = evaluate_on_dataset(model, s2ds_loader, device, "s2ds")
        s2_metrics["model"] = label
        print(f"    mIoU_fg={s2_metrics['mIoU_fg']:.4f}  "
              f"crack={s2_metrics['iou_crack']:.4f}  spalling={s2_metrics['iou_spalling']:.4f}  "
              f"inst_F1={s2_metrics['inst_micro_f1']:.3f}")

        # Domain gap
        gap = dm_metrics["mIoU_fg"] - s2_metrics["mIoU_fg"]
        print(f"\n  Domain gap (mIoU_fg): {gap:+.4f} ({gap/dm_metrics['mIoU_fg']*100:+.1f}%)")

        all_results.append({"label": label, "damsegment": dm_metrics, "s2ds": s2_metrics, "domain_gap_miou_fg": gap})

        # Save per-image
        with open(args.output / f"{label}_damseg_per_image.json", "w") as f:
            json.dump(dm_per_image, f, indent=2)
        with open(args.output / f"{label}_s2ds_per_image.json", "w") as f:
            json.dump(s2_per_image, f, indent=2)

        del model
        torch.cuda.empty_cache()

    # Summary table
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    header = f"{'Model':<10} {'DamSeg mIoU':>12} {'s2ds mIoU':>12} {'Gap':>8} {'DamSeg iF1':>12} {'s2ds iF1':>12}"
    print(header)
    print("-" * len(header))
    for r in all_results:
        print(f"{r['label']:<10} {r['damsegment']['mIoU_fg']:>12.4f} {r['s2ds']['mIoU_fg']:>12.4f} "
              f"{r['domain_gap_miou_fg']:>+8.4f} {r['damsegment']['inst_micro_f1']:>12.3f} "
              f"{r['s2ds']['inst_micro_f1']:>12.3f}")

    # Save
    with open(args.output / "summary.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nSaved to {args.output}/")


if __name__ == "__main__":
    main()
