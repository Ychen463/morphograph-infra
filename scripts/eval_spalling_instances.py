"""Evaluate spalling instance segmentation via post-hoc connected components.

Loads a trained model checkpoint, runs inference on the val set,
extracts spalling instances via connected components from both
predicted and GT semantic masks, and computes instance-level metrics.

Usage:
    python scripts/eval_spalling_instances.py \
        --data-root data/raw \
        --checkpoint runs/B2_dt_v5/best.pt \
        --output runs/eval_spalling_instances
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from morphograph.data.schema import NUM_CLASSES, decode_rgb_mask
from morphograph.metrics.instance_metrics import (
    compute_instance_metrics, extract_instances, InstanceMetrics,
)
from morphograph.metrics.segmentation import compute_iou
from morphograph.training.utils import discover_all_samples, split_data


class SimpleSegDataset(Dataset):
    """Minimal dataset for evaluation — image + mask only."""

    def __init__(self, pairs: list[tuple[Path, Path]], size: int = 512):
        self.pairs = pairs
        self.size = size
        self.to_tensor = transforms.Compose([
            transforms.Resize((size, size), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, mask_path = self.pairs[idx]
        image = Image.open(img_path).convert("RGB")
        mask_rgb = np.array(Image.open(mask_path).convert("RGB").resize(
            (self.size, self.size), Image.NEAREST,
        ))
        mask = decode_rgb_mask(mask_rgb)
        return {
            "image": self.to_tensor(image),
            "mask": torch.from_numpy(mask).long(),
            "sample_id": img_path.stem,
        }


def load_model(checkpoint_path: Path, device: torch.device):
    """Load model from checkpoint, auto-detecting architecture."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt

    # Detect architecture from state dict keys
    has_graph_heads = any("node_heatmap_head" in k or "edge_classifier" in k for k in state)

    if has_graph_heads:
        from morphograph.models.morphograph_net import MorphoGraphNet, FPN_DIM
        model = MorphoGraphNet(
            backbone="mit_b2", num_classes=NUM_CLASSES,
            fpn_dim=FPN_DIM, graph_heads=True,
        )
    else:
        from morphograph.models.morphograph_net import MorphoAuxNet, FPN_DIM, BASELINE_HEADS
        # Detect which heads are present
        head_names = set()
        for k in state:
            for h in BASELINE_HEADS.get("B5", []):
                if h in k:
                    head_names.add(h)
        has_skel = any("skeleton" in k for k in state)
        model = MorphoAuxNet(
            backbone="mit_b2", num_classes=NUM_CLASSES,
            fpn_dim=FPN_DIM, head_names=BASELINE_HEADS.get("B2", ["seg", "skeleton"]),
        )

    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser(description="Evaluate spalling instances")
    parser.add_argument("--data-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("runs/eval_spalling"))
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--iou-threshold", type=float, default=0.5,
                        help="IoU threshold for instance matching")
    parser.add_argument("--min-area", type=int, default=25,
                        help="Minimum instance area in pixels")
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    # Data
    all_pairs = discover_all_samples(args.data_root)
    _, val_pairs = split_data(all_pairs, args.val_ratio, args.seed)
    print(f"Val set: {len(val_pairs)} images")

    val_loader = DataLoader(
        SimpleSegDataset(val_pairs),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    # Model
    model = load_model(args.checkpoint, device)
    print(f"Loaded checkpoint: {args.checkpoint}")

    # Evaluate
    per_image = []
    agg = defaultdict(list)
    gt_spalling_count = 0  # images with GT spalling

    with torch.no_grad():
        for batch in val_loader:
            images = batch["image"].to(device)
            masks = batch["mask"].numpy()

            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                outputs = model(images)

            preds = outputs["seg"].argmax(dim=1).cpu().numpy()

            for i in range(len(images)):
                gt_mask = masks[i]
                pred_mask = preds[i]
                sample_id = batch["sample_id"][i]

                # Check if GT has spalling
                has_gt_spalling = (gt_mask == 2).any()
                if has_gt_spalling:
                    gt_spalling_count += 1

                im = compute_instance_metrics(
                    pred_mask, gt_mask,
                    class_id=2,
                    iou_threshold=args.iou_threshold,
                    min_area_px=args.min_area,
                )

                # Semantic spalling IoU for comparison
                iou_dict = compute_iou(pred_mask, gt_mask)
                spalling_iou = iou_dict.get(2, 0.0)

                record = {
                    "sample_id": sample_id,
                    "has_gt_spalling": bool(has_gt_spalling),
                    "n_gt": im.n_gt,
                    "n_pred": im.n_pred,
                    "tp": im.tp, "fp": im.fp, "fn": im.fn,
                    "precision": im.precision,
                    "recall": im.recall,
                    "f1": im.f1,
                    "mean_matched_iou": im.mean_matched_iou,
                    "semantic_iou": spalling_iou,
                }
                per_image.append(record)

                # Only aggregate images that have GT or pred spalling
                if im.n_gt > 0 or im.n_pred > 0:
                    for k in ["precision", "recall", "f1", "mean_matched_iou",
                              "n_gt", "n_pred", "tp", "fp", "fn"]:
                        agg[k].append(getattr(im, k))
                    agg["semantic_iou"].append(spalling_iou)

    # Aggregate
    total_tp = sum(agg["tp"])
    total_fp = sum(agg["fp"])
    total_fn = sum(agg["fn"])
    micro_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    micro_r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r) if (micro_p + micro_r) > 0 else 0.0

    summary = {
        "checkpoint": str(args.checkpoint),
        "iou_threshold": args.iou_threshold,
        "min_area_px": args.min_area,
        "n_val_images": len(val_pairs),
        "n_images_with_gt_spalling": gt_spalling_count,
        "n_images_with_instances": len(agg["f1"]),
        "total_gt_instances": int(sum(agg["n_gt"])),
        "total_pred_instances": int(sum(agg["n_pred"])),
        "micro_precision": micro_p,
        "micro_recall": micro_r,
        "micro_f1": micro_f1,
        "macro_precision": float(np.mean(agg["precision"])) if agg["precision"] else 0.0,
        "macro_recall": float(np.mean(agg["recall"])) if agg["recall"] else 0.0,
        "macro_f1": float(np.mean(agg["f1"])) if agg["f1"] else 0.0,
        "mean_matched_iou": float(np.mean(agg["mean_matched_iou"])) if agg["mean_matched_iou"] else 0.0,
        "mean_semantic_iou": float(np.mean(agg["semantic_iou"])) if agg["semantic_iou"] else 0.0,
    }

    # Print
    print(f"\n{'='*60}")
    print(f"Spalling Instance Evaluation (IoU threshold={args.iou_threshold})")
    print(f"{'='*60}")
    print(f"Images: {len(val_pairs)} total, {gt_spalling_count} with GT spalling")
    print(f"Instances: {summary['total_gt_instances']} GT, {summary['total_pred_instances']} pred")
    print(f"\nMicro:  P={micro_p:.3f}  R={micro_r:.3f}  F1={micro_f1:.3f}")
    print(f"Macro:  P={summary['macro_precision']:.3f}  R={summary['macro_recall']:.3f}  F1={summary['macro_f1']:.3f}")
    print(f"Mean matched IoU: {summary['mean_matched_iou']:.3f}")
    print(f"Semantic spalling IoU: {summary['mean_semantic_iou']:.3f}")

    # Save
    with open(args.output / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open(args.output / "per_image.json", "w") as f:
        json.dump(per_image, f, indent=2)

    print(f"\nSaved to {args.output}/")


if __name__ == "__main__":
    main()
