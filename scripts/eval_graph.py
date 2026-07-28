"""Graph-level evaluation: compare B0 vs B2 vs P3 on crack graph extraction.

Usage:
    python scripts/eval_graph.py --data-root data/raw \
        --checkpoints runs/B0/best.pt runs/B2_dt_v5/best.pt \
        --labels B0 B2_best --output runs/eval_graph

    python scripts/eval_graph.py --data-root data/raw \
        --checkpoints runs/B2_dt_v5/best.pt runs/P3b/best.pt \
        --labels B2_best P3b --output runs/eval_graph_p3

Methods: A (skeleton), B (DT threshold), C (DT-guided ridge), D (learned decoder).
Primary endpoint: edge F1 @10px (relaxed tolerance) — pre-specified before testing.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from morphograph.data.schema import decode_rgb_mask, NUM_CLASSES
from morphograph.data.graph_targets import mask_to_graph, mask_to_dt_target
from morphograph.metrics.graph_metrics import (
    compute_graph_metrics, approx_graph_edit_distance, path_continuity,
    degree_distribution_kl, DEFAULT_KEYPOINT_TOLERANCE_PX,
)
from morphograph.metrics.segmentation import (
    compute_iou, compute_cldice, compute_connectivity_recall, compute_boundary_f1,
)
from morphograph.models.morphograph_net import MorphoAuxNet, MorphoGraphNet, BASELINE_HEADS
from morphograph.evaluation.graph_extraction import (
    extract_graph_a, extract_graph_b, extract_graph_c, extract_graph_d,
)
from morphograph.evaluation.statistics import aggregate_metrics, paired_wilcoxon
from morphograph.evaluation.graph_viz import draw_graph_overlay, plot_summary_bars
from morphograph.training.utils import set_seed, discover_all_samples, split_data

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ── Model loading ──

def load_model(checkpoint_path: Path, device: torch.device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = ckpt["model_state_dict"]
    is_p3 = any(k.startswith("fpn128.") for k in state)

    if is_p3:
        has_graph_heads = any(k.startswith("node_head.") for k in state)
        model = MorphoGraphNet(
            backbone="mit_b2", num_classes=NUM_CLASSES, graph_heads=has_graph_heads,
        ).to(device)
        model.load_state_dict(state)
        model.eval()
        return model

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


# ── Per-image graph metrics helper ──

GRAPH_CORE_KEYS = [
    "endpoint_f1", "junction_f1", "edge_f1", "width_mae",
    "ged", "path_cont", "degree_kl",
    "endpoint_f1_relaxed", "junction_f1_relaxed", "edge_f1_relaxed",
    "endpoint_f1_lenient", "junction_f1_lenient", "edge_f1_lenient",
    "edge_f1_soft",
]


def _graph_metrics_dict(pred, gt, tolerance, prefix):
    gm = compute_graph_metrics(
        pred.endpoints, pred.junctions, pred.edges,
        gt.endpoints, gt.junctions, gt.edges,
        pred_width=pred.width_at_nodes, target_width=gt.width_at_nodes,
        tolerance_px=tolerance,
    )
    d = {f"{prefix}_{k}": v for k, v in asdict(gm).items()}
    d[f"{prefix}_ged"] = approx_graph_edit_distance(
        pred.num_nodes, pred.num_edges, gt.num_nodes, gt.num_edges,
    )
    d[f"{prefix}_path_cont"] = path_continuity(
        pred.all_nodes, pred.edges, gt.all_nodes, gt.edges, tolerance,
    )
    d[f"{prefix}_degree_kl"] = degree_distribution_kl(
        pred.edges, pred.num_nodes, gt.edges, gt.num_nodes,
    )
    d[f"{prefix}_num_nodes"] = pred.num_nodes
    d[f"{prefix}_num_edges"] = pred.num_edges
    return d


# ── Per-image evaluation ──

def evaluate_single_image(
    model, img_path, mask_path, device, has_dt, kp_tolerance,
    ridge_threshold, is_p3=False, img_size=512,
):
    img = np.array(Image.open(img_path).convert("RGB").resize(
        (img_size, img_size), Image.BILINEAR))
    mask_raw = np.array(Image.open(mask_path).resize((img_size, img_size), Image.NEAREST))
    gt = decode_rgb_mask(mask_raw) if mask_raw.ndim == 3 else mask_raw.astype(np.uint8)
    gt_crack = (gt == 1).astype(np.uint8)

    if not gt_crack.any():
        return None

    img_t = torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    with torch.no_grad(), torch.amp.autocast("cuda", enabled=device.type == "cuda"):
        outputs = model(img_t.to(device))

    pred_seg = outputs["seg"].argmax(dim=1)[0].cpu().numpy()
    pred_crack = (pred_seg == 1).astype(np.uint8)
    dt_pred = torch.sigmoid(outputs["skeleton"])[0, 0].cpu().numpy() if (has_dt or is_p3) else None

    # Pixel metrics
    ious = compute_iou(pred_seg, gt)
    fg_vals = [v for c, v in ious.items() if c > 0]
    result = {
        "filename": img_path.name,
        "crack_px": int(gt_crack.sum()),
        "miou_fg": float(np.mean(fg_vals)) if fg_vals else 0.0,
        "iou_crack": ious.get(1, 0.0),
        "cldice": compute_cldice(pred_crack.astype(bool), gt_crack.astype(bool)),
        "connr": compute_connectivity_recall(pred_crack.astype(bool), gt_crack.astype(bool)),
        "bf1_crack": compute_boundary_f1(pred_seg, gt).get(1, 0.0),
    }

    gt_graph = mask_to_graph(gt_crack, min_branch_length=10, junction_merge_radius=5)
    result["gt_num_nodes"] = gt_graph.num_nodes
    result["gt_num_edges"] = gt_graph.num_edges

    result.update(_graph_metrics_dict(extract_graph_a(pred_crack), gt_graph, kp_tolerance, "A"))

    if has_dt and dt_pred is not None and not is_p3:
        result.update(_graph_metrics_dict(extract_graph_b(dt_pred), gt_graph, kp_tolerance, "B"))
        result.update(_graph_metrics_dict(
            extract_graph_c(pred_crack, dt_pred, ridge_threshold), gt_graph, kp_tolerance, "C"))

    if is_p3:
        result.update(_graph_metrics_dict(extract_graph_d(model, outputs), gt_graph, kp_tolerance, "D"))

    return result


# ── Visualization ──

def visualize_sample(img_path, mask_path, model, device, has_dt, ridge_threshold, out_path, img_size=512):
    img = np.array(Image.open(img_path).convert("RGB").resize((img_size, img_size), Image.BILINEAR))
    mask_raw = np.array(Image.open(mask_path).resize((img_size, img_size), Image.NEAREST))
    gt = decode_rgb_mask(mask_raw) if mask_raw.ndim == 3 else mask_raw.astype(np.uint8)
    gt_crack = (gt == 1).astype(np.uint8)

    img_t = torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    with torch.no_grad(), torch.amp.autocast("cuda", enabled=device.type == "cuda"):
        outputs = model(img_t.to(device))

    pred_crack = (outputs["seg"].argmax(dim=1)[0].cpu().numpy() == 1).astype(np.uint8)
    gt_graph = mask_to_graph(gt_crack)
    pred_graph_a = extract_graph_a(pred_crack)

    ncols = 6 if has_dt else 5
    fig, axes = plt.subplots(1, ncols, figsize=(4 * ncols, 4))
    axes[0].imshow(img); axes[0].set_title("Image", fontsize=8); axes[0].axis("off")
    axes[1].imshow(gt_crack, cmap="gray"); axes[1].set_title("GT Crack", fontsize=8); axes[1].axis("off")
    axes[2].imshow(pred_crack, cmap="gray"); axes[2].set_title("Pred Crack", fontsize=8); axes[2].axis("off")
    draw_graph_overlay(axes[3], img, gt_graph, "GT Graph")
    draw_graph_overlay(axes[4], img, pred_graph_a, "Pred Graph (A)")
    if has_dt:
        dt_pred = torch.sigmoid(outputs["skeleton"])[0, 0].cpu().numpy()
        draw_graph_overlay(axes[5], img, extract_graph_c(pred_crack, dt_pred, ridge_threshold), "Pred Graph (C)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── Ridge threshold sweep ──

def ridge_threshold_sweep(model, val_pairs, device, kp_tolerance, thresholds, max_images=50):
    results = {t: [] for t in thresholds}
    for img_path, mask_path in val_pairs[:max_images]:
        mask_raw = np.array(Image.open(mask_path).resize((512, 512), Image.NEAREST))
        gt = decode_rgb_mask(mask_raw) if mask_raw.ndim == 3 else mask_raw.astype(np.uint8)
        gt_crack = (gt == 1).astype(np.uint8)
        if not gt_crack.any():
            continue

        img = np.array(Image.open(img_path).convert("RGB").resize((512, 512), Image.BILINEAR))
        img_t = torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0) / 255.0
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            outputs = model(img_t.to(device))

        pred_crack = (outputs["seg"].argmax(dim=1)[0].cpu().numpy() == 1).astype(np.uint8)
        dt_pred = torch.sigmoid(outputs["skeleton"])[0, 0].cpu().numpy()
        gt_graph = mask_to_graph(gt_crack)

        for t in thresholds:
            gm = compute_graph_metrics(
                *(lambda g: (g.endpoints, g.junctions, g.edges))(
                    extract_graph_c(pred_crack, dt_pred, ridge_threshold=t)),
                gt_graph.endpoints, gt_graph.junctions, gt_graph.edges,
                tolerance_px=kp_tolerance,
            )
            results[t].append(gm.edge_f1)

    return {str(t): {"mean_edge_f1": float(np.mean(v)) if v else 0.0, "n_images": len(v)}
            for t, v in results.items()}


# ── Main ──

def main() -> None:
    parser = argparse.ArgumentParser(description="Graph-level evaluation")
    parser.add_argument("--data-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--checkpoints", nargs="+", type=Path, required=True)
    parser.add_argument("--labels", nargs="+", type=str, default=None)
    parser.add_argument("--output", type=Path, default=Path("runs/eval_graph"))
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--kp-tolerance", type=float, default=DEFAULT_KEYPOINT_TOLERANCE_PX)
    parser.add_argument("--ridge-threshold", type=float, default=0.3)
    parser.add_argument("--num-vis", type=int, default=20)
    args = parser.parse_args()

    set_seed(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "vis").mkdir(exist_ok=True)
    device = torch.device(args.device)
    labels = args.labels or [p.parent.name for p in args.checkpoints]
    assert len(labels) == len(args.checkpoints)

    all_pairs = discover_all_samples(args.data_root)
    _, val_pairs = split_data(all_pairs, args.val_ratio, args.seed)
    print(f"Evaluating on {len(val_pairs)} validation images\n")

    pixel_keys = ["miou_fg", "iou_crack", "cldice", "connr", "bf1_crack"]
    all_checkpoint_results = {}
    dt_model_label = None

    for label, ckpt_path in zip(labels, args.checkpoints):
        print(f"{'=' * 60}\nEvaluating: {label} ({ckpt_path})\n{'=' * 60}")
        model = load_model(ckpt_path, device)
        is_p3 = isinstance(model, MorphoGraphNet) and model.has_graph_heads
        has_dt = isinstance(model, MorphoGraphNet) or any(
            k.startswith("skeleton_head.") for k in model.state_dict())
        if has_dt and not is_p3:
            dt_model_label = label

        per_image = []
        for i, (img_path, mask_path) in enumerate(val_pairs):
            result = evaluate_single_image(
                model, img_path, mask_path, device, has_dt=has_dt,
                kp_tolerance=args.kp_tolerance, ridge_threshold=args.ridge_threshold, is_p3=is_p3,
            )
            if result is not None:
                per_image.append(result)
            if (i + 1) % 50 == 0:
                print(f"  [{i + 1}/{len(val_pairs)}]")

        print(f"  {len(per_image)} crack images\n")
        all_checkpoint_results[label] = per_image

        methods = ["A"] + (["D"] if is_p3 else ["B", "C"] if has_dt else [])
        for method in methods:
            method_keys = [f"{method}_{k}" for k in GRAPH_CORE_KEYS]
            agg = aggregate_metrics(per_image, method_keys)
            print(f"  Method {method}:")
            for k in method_keys:
                if k in agg:
                    print(f"    {k:30s}: {agg[k]['mean']:.4f} +/- {agg[k]['std']:.4f}")
            print()

        agg_px = aggregate_metrics(per_image, pixel_keys)
        print("  Pixel metrics:")
        for k in pixel_keys:
            if k in agg_px:
                print(f"    {k:20s}: {agg_px[k]['mean']:.4f} +/- {agg_px[k]['std']:.4f}")
        print()

        vis_count = 0
        for img_path, mask_path in val_pairs:
            if vis_count >= args.num_vis:
                break
            mask_raw = np.array(Image.open(mask_path).resize((512, 512), Image.NEAREST))
            gt_check = decode_rgb_mask(mask_raw) if mask_raw.ndim == 3 else mask_raw
            if not (gt_check == 1).any():
                continue
            visualize_sample(img_path, mask_path, model, device, has_dt,
                             args.ridge_threshold, args.output / "vis" / f"{label}_{vis_count:03d}.png")
            vis_count += 1
        print(f"  Saved {vis_count} visualizations\n")
        del model
        torch.cuda.empty_cache()

    # Ridge threshold sweep
    sweep_result = None
    if dt_model_label is not None:
        print(f"{'=' * 60}\nRidge Threshold Sweep (Method C)\n{'=' * 60}")
        model = load_model(args.checkpoints[labels.index(dt_model_label)], device)
        sweep_result = ridge_threshold_sweep(
            model, val_pairs, device, args.kp_tolerance, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        for t, s in sweep_result.items():
            print(f"  alpha={t}: edge_f1={s['mean_edge_f1']:.4f} (n={s['n_images']})")
        del model; torch.cuda.empty_cache()
        print()

    # Comparison table
    print(f"{'=' * 60}\nCOMPARISON TABLE\n{'=' * 60}")
    bar_agg = {}
    for label in labels:
        per_image = all_checkpoint_results[label]
        has_d = any("D_edge_f1" in r for r in per_image)
        has_bc = any("C_edge_f1" in r for r in per_image)
        methods = ["A"] + (["D"] if has_d else ["B", "C"] if has_bc else [])
        for method in methods:
            tag = f"{label}-{method}"
            method_keys = [f"{method}_{k}" for k in GRAPH_CORE_KEYS]
            bar_agg[tag] = aggregate_metrics(per_image, pixel_keys + method_keys)
            for k in GRAPH_CORE_KEYS:
                mk = f"{method}_{k}"
                if mk in bar_agg[tag]:
                    bar_agg[tag][k] = bar_agg[tag][mk]

    tags = list(bar_agg.keys())
    compare_keys = pixel_keys + GRAPH_CORE_KEYS
    header = f"{'Metric':25s}" + "".join(f" | {t:>14s}" for t in tags)
    print(header)
    print("-" * len(header))
    for key in compare_keys:
        values = [bar_agg[t].get(key, {}).get("mean", 0.0) for t in tags]
        best_val = max(values) if values else 0
        row = f"{key:25s}"
        for val in values:
            marker = " *" if val == best_val and len(tags) > 1 and val > 0 else "  "
            row += f" | {val:12.4f}{marker}"
        print(row)
    print()

    # Statistical tests
    if len(labels) >= 2:
        print(f"{'=' * 60}\nSTATISTICAL TESTS (Wilcoxon)\n  Primary: edge F1 @10px\n{'=' * 60}")
        stat_results = {}
        base = labels[0]
        for other in labels[1:]:
            stat_results[f"{other}_vs_{base}"] = {}
            test_keys = ["A_endpoint_f1", "A_junction_f1", "A_edge_f1", "A_path_cont",
                         "A_edge_f1_relaxed", "A_edge_f1_lenient", "A_edge_f1_soft"]
            if any("C_edge_f1" in r for r in all_checkpoint_results[other]):
                test_keys += [k.replace("A_", "C_") for k in test_keys]
            if any("D_edge_f1" in r for r in all_checkpoint_results[other]):
                test_keys += [k.replace("A_", "D_") for k in test_keys[:7]]
            for key in test_keys:
                wt = paired_wilcoxon(all_checkpoint_results[base], all_checkpoint_results[other], key)
                stat_results[f"{other}_vs_{base}"][key] = wt
                sig = ""
                if wt["p_value"] is not None:
                    sig = " ***" if wt["p_value"] < 0.001 else " **" if wt["p_value"] < 0.01 else " *" if wt["p_value"] < 0.05 else ""
                p_str = f"{wt['p_value']:.4f}" if wt["p_value"] is not None else "N/A"
                print(f"  {key:30s}: diff={wt.get('mean_diff', 0):.4f}, p={p_str}{sig}")
        print()

    # Bar chart
    chart_suffixes = ["endpoint_f1", "junction_f1", "edge_f1", "path_cont"]
    chart_agg = {}
    for label in labels:
        per_image = all_checkpoint_results[label]
        has_d = any("D_edge_f1" in r for r in per_image)
        has_bc = any("C_edge_f1" in r for r in per_image)
        methods = ["A"] + (["D"] if has_d else ["C"] if has_bc else [])
        for method in methods:
            tag = f"{label}-{method}"
            chart_agg[tag] = aggregate_metrics(per_image, [f"{method}_{k}" for k in chart_suffixes])
    first_method = list(chart_agg.keys())[0].split("-")[1]
    plot_summary_bars(chart_agg, [f"{first_method}_{k}" for k in chart_suffixes],
                      args.output / "vis" / "summary_bars.png")

    # Save outputs
    all_rows = []
    for label, per_image in all_checkpoint_results.items():
        for r in per_image:
            all_rows.append({"model": label, **r})
    if all_rows:
        fieldnames = list(dict.fromkeys(k for row in all_rows for k in row))
        with open(args.output / "per_image_results.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(all_rows)

    summary = {
        "val_samples": len(val_pairs),
        "primary_endpoint": "edge_f1_relaxed (@10px)",
        "params": {"kp_tolerance": args.kp_tolerance, "ridge_threshold": args.ridge_threshold, "seed": args.seed},
        "per_model": {},
    }
    for label in labels:
        per_image = all_checkpoint_results[label]
        has_d = any("D_edge_f1" in r for r in per_image)
        has_bc = any("C_edge_f1" in r for r in per_image)
        methods = ["A"] + (["D"] if has_d else ["B", "C"] if has_bc else [])
        ms = {"n_crack_images": len(per_image), "pixel": aggregate_metrics(per_image, pixel_keys)}
        for method in methods:
            ms[f"method_{method}"] = aggregate_metrics(per_image, [f"{method}_{k}" for k in GRAPH_CORE_KEYS])
        summary["per_model"][label] = ms
    if sweep_result:
        summary["ridge_threshold_sweep"] = sweep_result
    if len(labels) >= 2:
        summary["statistical_tests"] = stat_results

    with open(args.output / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    with open(args.output / "comparison_table.txt", "w") as f:
        f.write(header + "\n" + "-" * len(header) + "\n")
        for key in compare_keys:
            row = f"{key:25s}"
            for t in tags:
                row += f" | {bar_agg[t].get(key, {}).get('mean', 0.0):12.4f}  "
            f.write(row + "\n")

    print(f"\nResults saved to {args.output}/")
    print("Done.")


if __name__ == "__main__":
    main()
