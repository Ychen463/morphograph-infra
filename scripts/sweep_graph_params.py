"""Sweep skeleton extraction parameters to find optimal graph extraction config.

Sweeps closing_radius, spur_length, min_component_px, junction_merge_radius,
and min_branch_length independently (one-at-a-time), measuring graph metrics
at strict/relaxed/lenient tolerances.

Usage:
    python scripts/sweep_graph_params.py \
        --data-root data/raw \
        --checkpoint runs/B0/best.pt \
        --output runs/sweep_graph_params \
        --max-images 100 --device cuda
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict
from itertools import product
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from morphograph.data.schema import decode_rgb_mask
from morphograph.data.graph_targets import (
    mask_to_skeleton,
    detect_keypoints,
    build_graph,
    mask_to_graph,
    CrackGraph,
)
from morphograph.metrics.graph_metrics import (
    compute_graph_metrics,
    DEFAULT_KEYPOINT_TOLERANCE_PX,
    RELAXED_TOLERANCE_PX,
    LENIENT_TOLERANCE_PX,
)
from morphograph.models.morphograph_net import MorphoAuxNet
from morphograph.data.schema import NUM_CLASSES
from morphograph.training.utils import set_seed, discover_all_samples, split_data


# ── Default (current) extraction params ──

DEFAULTS = {
    "closing_radius": 1,
    "spur_length": 3,
    "min_component_px": 10,
    "junction_merge_radius": 5,
    "min_branch_length": 10,
}

# ── Sweep ranges (one-at-a-time around defaults) ──

SWEEP_RANGES = {
    "closing_radius": [0, 1, 2, 3],
    "spur_length": [0, 3, 5, 8, 12],
    "min_component_px": [5, 10, 15, 20, 30],
    "junction_merge_radius": [3, 5, 8, 10, 15],
    "min_branch_length": [5, 10, 15, 20, 30],
}


def extract_graph_with_params(
    crack_mask: np.ndarray,
    closing_radius: int,
    spur_length: int,
    min_component_px: int,
    junction_merge_radius: int,
    min_branch_length: int,
) -> CrackGraph:
    """Extract graph with custom skeleton/graph params."""
    skel = mask_to_skeleton(
        crack_mask,
        closing_radius=closing_radius,
        min_component_px=min_component_px,
        spur_length=spur_length,
        dilate_radius=0,
    )
    if not skel.any():
        return CrackGraph(
            endpoints=np.empty((0, 2), dtype=int),
            junctions=np.empty((0, 2), dtype=int),
        )
    endpoints, junctions = detect_keypoints(skel)
    return build_graph(
        skel, endpoints, junctions,
        min_branch_length=min_branch_length,
        junction_merge_radius=junction_merge_radius,
        binary_mask=crack_mask,
    )


def load_model(checkpoint_path: Path, device: torch.device) -> MorphoAuxNet:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = ckpt["model_state_dict"]
    heads = {
        "seg_head": True,
        "skeleton_head": any(k.startswith("skeleton_head.") for k in state),
        "endpoint_head": any(k.startswith("endpoint_head.") for k in state),
        "junction_head": any(k.startswith("junction_head.") for k in state),
        "width_head": any(k.startswith("width_head.") for k in state),
    }
    model = MorphoAuxNet(backbone="mit_b2", num_classes=NUM_CLASSES, heads=heads).to(device)
    model.load_state_dict(state)
    model.eval()
    return model


def evaluate_config(
    pred_cracks: list[np.ndarray],
    gt_graphs: list[CrackGraph],
    gt_cracks: list[np.ndarray],
    params: dict,
) -> dict:
    """Evaluate one parameter config across all images."""
    metrics_strict = []
    metrics_relaxed = []
    metrics_lenient = []
    metrics_soft = []

    for pred_crack, gt_graph, gt_crack in zip(pred_cracks, gt_graphs, gt_cracks):
        pred_graph = extract_graph_with_params(pred_crack, **params)

        gm = compute_graph_metrics(
            pred_endpoints=pred_graph.endpoints,
            pred_junctions=pred_graph.junctions,
            pred_edges=pred_graph.edges,
            target_endpoints=gt_graph.endpoints,
            target_junctions=gt_graph.junctions,
            target_edges=gt_graph.edges,
            pred_width=pred_graph.width_at_nodes,
            target_width=gt_graph.width_at_nodes,
        )
        metrics_strict.append(gm.edge_f1)
        metrics_relaxed.append(gm.edge_f1_relaxed)
        metrics_lenient.append(gm.edge_f1_lenient)
        metrics_soft.append(gm.edge_f1_soft)

    return {
        **params,
        "n_images": len(pred_cracks),
        "edge_f1_strict_mean": float(np.mean(metrics_strict)),
        "edge_f1_strict_median": float(np.median(metrics_strict)),
        "edge_f1_relaxed_mean": float(np.mean(metrics_relaxed)),
        "edge_f1_relaxed_median": float(np.median(metrics_relaxed)),
        "edge_f1_lenient_mean": float(np.mean(metrics_lenient)),
        "edge_f1_lenient_median": float(np.median(metrics_lenient)),
        "edge_f1_soft_mean": float(np.mean(metrics_soft)),
        "edge_f1_soft_median": float(np.median(metrics_soft)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep graph extraction parameters")
    parser.add_argument("--data-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("runs/sweep_graph_params"))
    parser.add_argument("--max-images", type=int, default=100,
                        help="Max crack images to evaluate per config")
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--mode", choices=["one_at_a_time", "grid"], default="one_at_a_time",
                        help="one_at_a_time: sweep each param independently; grid: full grid (slow)")
    args = parser.parse_args()

    set_seed(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    # ── Load model and precompute predictions ──
    print("Loading model...")
    model = load_model(args.checkpoint, device)

    all_pairs = discover_all_samples(args.data_root)
    _, val_pairs = split_data(all_pairs, args.val_ratio, args.seed)

    print(f"Precomputing predictions on up to {args.max_images} crack images...")
    pred_cracks = []
    gt_graphs = []
    gt_cracks = []

    for img_path, mask_path in val_pairs:
        if len(pred_cracks) >= args.max_images:
            break

        mask_raw = np.array(
            Image.open(mask_path).resize((512, 512), Image.NEAREST)
        )
        if mask_raw.ndim == 3:
            gt = decode_rgb_mask(mask_raw)
        else:
            gt = mask_raw.astype(np.uint8)
        gt_crack = (gt == 1).astype(np.uint8)
        if not gt_crack.any():
            continue

        img = np.array(
            Image.open(img_path).convert("RGB").resize((512, 512), Image.BILINEAR)
        )
        img_t = torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0) / 255.0
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            outputs = model(img_t.to(device))

        pred_seg = outputs["seg"].argmax(dim=1)[0].cpu().numpy()
        pred_crack = (pred_seg == 1).astype(np.uint8)

        # GT graph with default params (fixed reference)
        gt_graph = mask_to_graph(gt_crack, min_branch_length=10, junction_merge_radius=5)

        pred_cracks.append(pred_crack)
        gt_graphs.append(gt_graph)
        gt_cracks.append(gt_crack)

    print(f"  {len(pred_cracks)} crack images cached\n")
    del model
    torch.cuda.empty_cache()

    # ── Run sweep ──
    all_results = []

    if args.mode == "one_at_a_time":
        # Evaluate defaults first
        print("Evaluating defaults...")
        default_result = evaluate_config(pred_cracks, gt_graphs, gt_cracks, DEFAULTS)
        default_result["sweep_param"] = "defaults"
        all_results.append(default_result)
        print(f"  defaults: strict={default_result['edge_f1_strict_mean']:.4f}, "
              f"relaxed={default_result['edge_f1_relaxed_mean']:.4f}, "
              f"lenient={default_result['edge_f1_lenient_mean']:.4f}, "
              f"soft={default_result['edge_f1_soft_mean']:.4f}")

        # Sweep each param independently
        for param_name, values in SWEEP_RANGES.items():
            print(f"\nSweeping {param_name}: {values}")
            for val in values:
                params = {**DEFAULTS, param_name: val}
                result = evaluate_config(pred_cracks, gt_graphs, gt_cracks, params)
                result["sweep_param"] = param_name
                all_results.append(result)
                print(f"  {param_name}={val:>3d}: strict={result['edge_f1_strict_mean']:.4f}, "
                      f"relaxed={result['edge_f1_relaxed_mean']:.4f}, "
                      f"lenient={result['edge_f1_lenient_mean']:.4f}, "
                      f"soft={result['edge_f1_soft_mean']:.4f}")

    else:
        # Full grid (can be slow)
        configs = list(product(*SWEEP_RANGES.values()))
        print(f"Grid sweep: {len(configs)} configs")
        for i, vals in enumerate(configs):
            params = dict(zip(SWEEP_RANGES.keys(), vals))
            result = evaluate_config(pred_cracks, gt_graphs, gt_cracks, params)
            result["sweep_param"] = "grid"
            all_results.append(result)
            if (i + 1) % 50 == 0:
                print(f"  [{i + 1}/{len(configs)}] best so far: "
                      f"strict={max(r['edge_f1_strict_mean'] for r in all_results):.4f}")

    # ── Save results ──
    csv_path = args.output / "sweep_results.csv"
    fieldnames = list(all_results[0].keys())
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)
    print(f"\nResults saved to {csv_path}")

    # ── Print summary: best config per metric ──
    print("\n" + "=" * 60)
    print("BEST CONFIGS")
    print("=" * 60)
    for metric in ["edge_f1_strict_mean", "edge_f1_relaxed_mean",
                    "edge_f1_lenient_mean", "edge_f1_soft_mean"]:
        best = max(all_results, key=lambda r: r[metric])
        params_str = ", ".join(f"{k}={best[k]}" for k in DEFAULTS)
        print(f"  {metric:30s}: {best[metric]:.4f}  ({params_str})")

    # Save summary JSON
    summary = {
        "n_images": len(pred_cracks),
        "mode": args.mode,
        "defaults": DEFAULTS,
        "n_configs": len(all_results),
        "best": {},
    }
    for metric in ["edge_f1_strict_mean", "edge_f1_relaxed_mean",
                    "edge_f1_lenient_mean", "edge_f1_soft_mean"]:
        best = max(all_results, key=lambda r: r[metric])
        summary["best"][metric] = {
            "value": best[metric],
            "params": {k: best[k] for k in DEFAULTS},
        }
    with open(args.output / "sweep_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary: {args.output / 'sweep_summary.json'}")
    print("Done.")


if __name__ == "__main__":
    main()
