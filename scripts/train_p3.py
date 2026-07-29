"""P3 training: direct node and graph-topology prediction.

Usage:
    python scripts/train_p3.py --phase P3a --data-root data/raw --output runs/P3a
    python scripts/train_p3.py --phase P3b --data-root data/raw --output runs/P3b
    python scripts/train_p3.py --phase P3b --data-root data/raw --output runs/P3b_overfit \
        --overfit 16 --epochs 200
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from morphograph.data.schema import NUM_CLASSES, DEFAULT_CE_WEIGHTS
from morphograph.data.dataset_p3 import DamSegmentP3Dataset, p3_collate_fn
from morphograph.losses.composite import WeightedCEDiceLoss, DTRegressionLoss, LossSchedule
from morphograph.losses.graph_loss import (
    NodeHeatmapLoss, EdgeBCELoss, build_edge_labels, scheduled_sampling_prob,
)
from morphograph.models.morphograph_net import MorphoGraphNet, load_b2_into_p3, FPN_DIM
from morphograph.models.graph_decoder import (
    extract_nodes, build_candidate_pairs, audit_candidate_recall,
)
from morphograph.metrics.graph_metrics import (
    compute_graph_metrics, DEFAULT_KEYPOINT_TOLERANCE_PX,
    _keypoint_prf, _edge_prf,
)
from morphograph.training.utils import (
    set_seed, discover_all_samples, split_data,
    compute_miou, make_cosine_schedule, save_checkpoint,
)
from morphograph.data.graph_targets import mask_to_graph


def parse_args():
    parser = argparse.ArgumentParser(description="P3 training: direct graph prediction")
    parser.add_argument("--phase", choices=["P3-Base", "P3a", "P3b", "P3h"], default="P3b")
    parser.add_argument("--data-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--b2-checkpoint", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--encoder-lr", type=float, default=6e-5)
    parser.add_argument("--head-lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--freeze-encoder-epochs", type=int, default=5)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--skel-weight", type=float, default=10.0)
    parser.add_argument("--node-weight", type=float, default=1.0)
    parser.add_argument("--edge-weight", type=float, default=1.0)
    parser.add_argument("--node-ramp-epochs", type=int, default=5)
    parser.add_argument("--edge-start-epoch", type=int, default=10)
    parser.add_argument("--edge-ramp-epochs", type=int, default=10)
    parser.add_argument("--ss-warmup", type=int, default=10)
    parser.add_argument("--ss-anneal-end", type=int, default=60)
    parser.add_argument("--node-threshold", type=float, default=0.3)
    parser.add_argument("--nms-radius", type=int, default=2)
    parser.add_argument("--max-nodes", type=int, default=50)
    parser.add_argument("--heatmap-sigma", type=float, default=1.0)
    parser.add_argument("--loss-pos-threshold", type=float, default=0.999)
    parser.add_argument("--loss-beta", type=float, default=4.0)
    parser.add_argument("--knn-k", type=int, default=8)
    parser.add_argument("--overfit", type=int, default=0)
    return parser.parse_args()


def train_one_epoch(
    model, train_loader, optimizer, scheduler, scaler, args,
    seg_loss_fn, skel_loss_fn, node_loss_fn, edge_loss_fn,
    node_schedule, edge_schedule, epoch, do_nodes, do_edges, device,
):
    model.train()
    epoch_losses = defaultdict(list)

    for batch in train_loader:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)
        dt_targets = batch["dt_target"].to(device)
        node_heatmaps = batch["node_heatmap"].to(device)

        with torch.amp.autocast("cuda", enabled=args.amp and device.type == "cuda"):
            outputs = model(images)
            seg_loss = seg_loss_fn(outputs["seg"], masks)["total"]
            skel_pred = torch.sigmoid(outputs["skeleton"])
            skel_loss = skel_loss_fn(skel_pred, dt_targets, torch.ones_like(dt_targets))

            node_loss = torch.tensor(0.0, device=device)
            if do_nodes:
                node_loss = node_loss_fn(outputs["node_heatmap"], node_heatmaps)

            node_w = node_schedule.effective_weight(epoch) if do_nodes else 0.0
            total_loss = seg_loss + args.skel_weight * skel_loss + node_w * node_loss

            edge_loss_val = torch.tensor(0.0, device=device)
            if do_edges and edge_schedule.effective_weight(epoch) > 0:
                # P3h: always use GT nodes (no heatmap); P3b: scheduled sampling
                p_gt = 1.0 if not do_nodes else scheduled_sampling_prob(epoch, args.ss_warmup, args.ss_anneal_end)
                batch_edge_loss = []

                for b_idx in range(images.shape[0]):
                    gt_coords = batch["gt_node_coords"][b_idx].to(device)
                    gt_types = batch["gt_node_types"][b_idx].to(device)
                    gt_edges = batch["gt_edges"][b_idx]

                    if len(gt_coords) < 2:
                        continue

                    if random.random() < p_gt:
                        node_coords = gt_coords
                        node_types = gt_types
                        node_scores = torch.ones(len(gt_coords), device=device)
                    else:
                        hm = torch.sigmoid(outputs["node_heatmap"][b_idx:b_idx+1])
                        detected = extract_nodes(
                            hm, threshold=args.node_threshold,
                            nms_radius=args.nms_radius, max_nodes=args.max_nodes,
                        )[0]
                        if len(detected.coords) < 2:
                            continue
                        node_coords = detected.coords
                        node_types = detected.types
                        node_scores = detected.scores

                    dummy_candidates = build_candidate_pairs(node_coords, k=args.knn_k)
                    _, _, pred_to_gt = build_edge_labels(
                        node_coords, node_types, gt_coords, gt_types,
                        gt_edges, dummy_candidates,
                    )
                    candidates = build_candidate_pairs(
                        node_coords, k=args.knn_k,
                        gt_edges=gt_edges, pred_to_gt=pred_to_gt,
                    )
                    if len(candidates) == 0:
                        continue

                    labels, loss_weight, pred_to_gt = build_edge_labels(
                        node_coords, node_types, gt_coords, gt_types,
                        gt_edges, candidates,
                    )
                    dt_128 = F.interpolate(
                        skel_pred[b_idx:b_idx+1],
                        size=(128, 128), mode="bilinear", align_corners=False,
                    )
                    edge_logits = model.edge_classifier(
                        outputs["_fpn"][b_idx:b_idx+1], dt_128,
                        node_coords, node_types, node_scores, candidates,
                    )
                    batch_edge_loss.append(edge_loss_fn(edge_logits, labels, loss_weight))

                if batch_edge_loss:
                    edge_loss_val = torch.stack(batch_edge_loss).mean()
                    total_loss = total_loss + edge_schedule.effective_weight(epoch) * edge_loss_val

        optimizer.zero_grad()
        scaler.scale(total_loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        epoch_losses["total"].append(total_loss.item())
        epoch_losses["seg"].append(seg_loss.item())
        epoch_losses["skel"].append(skel_loss.item())
        if do_nodes:
            epoch_losses["node"].append(node_loss.item())
        if do_edges:
            epoch_losses["edge"].append(edge_loss_val.item())

    return {k: np.mean(v) for k, v in epoch_losses.items()}


@torch.no_grad()
def validate(model, val_loader, args, do_nodes, do_edges, device):
    model.eval()
    val_seg_preds, val_seg_targets = [], []
    val_node_f1s = {t: [] for t in [5, 10, 15]}
    val_ep_f1s = {t: [] for t in [5, 10, 15]}
    val_jn_f1s = {t: [] for t in [5, 10, 15]}
    val_edge_f1s = {t: [] for t in [5, 10, 15]}
    val_edge_gt_f1s = {t: [] for t in [5, 10, 15]}  # Edge F1 using GT nodes
    val_n_pred, val_n_gt = [], []
    val_pred_scores = []  # Detected peak scores
    val_gt_hm_vals = []  # Heatmap values at GT node positions

    for batch in val_loader:
        images = batch["image"].to(device)
        masks_val = batch["mask"].to(device)

        with torch.amp.autocast("cuda", enabled=args.amp and device.type == "cuda"):
            outputs = model(images)

        val_seg_preds.append(outputs["seg"].argmax(dim=1).cpu())
        val_seg_targets.append(masks_val.cpu())

        if not (do_nodes or do_edges):
            continue

        seg_pred_np = outputs["seg"].argmax(dim=1).cpu().numpy()
        hm = torch.sigmoid(outputs["node_heatmap"]) if do_nodes else None
        for b_idx in range(images.shape[0]):
            gt_coords = batch["gt_node_coords"][b_idx]
            gt_types = batch["gt_node_types"][b_idx]
            gt_edges = batch["gt_edges"][b_idx]
            if len(gt_coords) == 0:
                continue

            if do_nodes:
                # P3a/P3b: extract nodes from learned heatmap
                detected = extract_nodes(
                    hm[b_idx:b_idx+1], threshold=args.node_threshold,
                    nms_radius=args.nms_radius, max_nodes=args.max_nodes,
                )[0]
                pred_coords_512 = detected.coords.cpu().numpy() * 4.0
                pred_types_np = detected.types.cpu().numpy()
                pred_scores = detected.scores

                val_n_pred.append(len(detected.coords))
                val_n_gt.append(len(gt_coords))
                if len(detected.scores) > 0:
                    val_pred_scores.extend(detected.scores.cpu().tolist())

                # Diagnostic: heatmap values at GT node positions
                hm_b = hm[b_idx]
                for ni in range(len(gt_coords)):
                    r, c = int(round(gt_coords[ni, 0].item())), int(round(gt_coords[ni, 1].item()))
                    ch = int(gt_types[ni].item())
                    r = max(0, min(r, hm_b.shape[1] - 1))
                    c = max(0, min(c, hm_b.shape[2] - 1))
                    val_gt_hm_vals.append(hm_b[ch, r, c].item())
            else:
                # P3h: extract nodes from predicted seg mask via skeleton
                crack_binary = (seg_pred_np[b_idx] == 1).astype(np.uint8)
                pred_graph = mask_to_graph(crack_binary, min_branch_length=10, junction_merge_radius=5)
                pred_ep_512 = pred_graph.endpoints.astype(np.float64)
                pred_jn_512 = pred_graph.junctions.astype(np.float64)
                pred_coords_512 = pred_graph.all_nodes.astype(np.float64)
                n_ep = len(pred_ep_512)
                pred_types_np = np.array([0] * n_ep + [1] * len(pred_jn_512), dtype=np.int64)
                pred_scores = torch.ones(len(pred_coords_512), device=device)

                val_n_pred.append(len(pred_coords_512))
                val_n_gt.append(len(gt_coords))

            gt_coords_512 = gt_coords.numpy() * 4.0

            pred_ep_mask = pred_types_np == 0
            pred_jn_mask = pred_types_np == 1
            gt_ep_mask = gt_types.numpy() == 0
            gt_jn_mask = gt_types.numpy() == 1

            pred_ep = pred_coords_512[pred_ep_mask] if pred_ep_mask.any() else np.empty((0, 2))
            pred_jn = pred_coords_512[pred_jn_mask] if pred_jn_mask.any() else np.empty((0, 2))
            gt_ep = gt_coords_512[gt_ep_mask] if gt_ep_mask.any() else np.empty((0, 2))
            gt_jn = gt_coords_512[gt_jn_mask] if gt_jn_mask.any() else np.empty((0, 2))

            pred_edges = []
            pred_edges_gt = []  # Edges predicted using GT node positions
            if do_edges:
                dt_128 = F.interpolate(
                    torch.sigmoid(outputs["skeleton"][b_idx:b_idx+1]),
                    size=(128, 128), mode="bilinear", align_corners=False,
                )
                # Build node tensors in 128-space for edge classifier
                if do_nodes:
                    node_coords_128 = detected.coords
                    node_types_t = detected.types
                    node_scores_t = detected.scores
                else:
                    node_coords_128 = torch.from_numpy(
                        pred_coords_512 / 4.0
                    ).float().to(device)
                    node_types_t = torch.from_numpy(pred_types_np).long().to(device)
                    node_scores_t = pred_scores.to(device)
                # Edge prediction from detected/extracted nodes
                if len(node_coords_128) >= 2:
                    candidates = build_candidate_pairs(node_coords_128, k=args.knn_k)
                    if len(candidates) > 0:
                        edge_logits = model.edge_classifier(
                            outputs["_fpn"][b_idx:b_idx+1], dt_128,
                            node_coords_128, node_types_t, node_scores_t, candidates,
                        )
                        mask = torch.sigmoid(edge_logits) > 0.5
                        pred_edges = [(min(a, b), max(a, b))
                                      for a, b in candidates[mask].cpu().tolist()]
                # Edge prediction from GT nodes (for Gate 2 evaluation)
                gt_c = gt_coords.to(device)
                gt_t = gt_types.to(device)
                if len(gt_c) >= 2:
                    gt_candidates = build_candidate_pairs(gt_c, k=args.knn_k)
                    if len(gt_candidates) > 0:
                        gt_scores = torch.ones(len(gt_c), device=device)
                        gt_edge_logits = model.edge_classifier(
                            outputs["_fpn"][b_idx:b_idx+1], dt_128,
                            gt_c, gt_t, gt_scores, gt_candidates,
                        )
                        gt_mask = torch.sigmoid(gt_edge_logits) > 0.5
                        pred_edges_gt = [(min(a, b), max(a, b))
                                         for a, b in gt_candidates[gt_mask].cpu().tolist()]

            for tol in [5, 10, 15]:
                gm = compute_graph_metrics(
                    pred_ep, pred_jn, pred_edges,
                    gt_ep, gt_jn, gt_edges,
                    tolerance_px=float(tol),
                )
                # Type-pooled node F1: match all nodes regardless of type
                _, _, pooled_f1 = _keypoint_prf(
                    pred_coords_512, gt_coords_512, float(tol),
                )
                val_node_f1s[tol].append(pooled_f1)
                val_ep_f1s[tol].append(gm.endpoint_f1)
                val_jn_f1s[tol].append(gm.junction_f1)
                val_edge_f1s[tol].append(gm.edge_f1)
                # Edge F1 with GT node positions (adjacency-only metric)
                if do_edges:
                    _, _, ef_gt, _ = _edge_prf(
                        gt_coords_512, gt_coords_512,
                        pred_edges_gt, gt_edges, float(tol),
                    )
                    val_edge_gt_f1s[tol].append(ef_gt)

    miou = compute_miou(torch.cat(val_seg_preds), torch.cat(val_seg_targets))
    metrics = {"mIoU_fg": miou["mIoU_fg"]}
    if do_nodes or do_edges:
        for t in [5, 10, 15]:
            metrics[f"node_f1@{t}px"] = float(np.mean(val_node_f1s[t])) if val_node_f1s[t] else 0.0
            metrics[f"ep_f1@{t}px"] = float(np.mean(val_ep_f1s[t])) if val_ep_f1s[t] else 0.0
            metrics[f"jn_f1@{t}px"] = float(np.mean(val_jn_f1s[t])) if val_jn_f1s[t] else 0.0
            metrics[f"edge_f1@{t}px"] = float(np.mean(val_edge_f1s[t])) if val_edge_f1s[t] else 0.0
            metrics[f"edge_gt_f1@{t}px"] = float(np.mean(val_edge_gt_f1s[t])) if val_edge_gt_f1s[t] else 0.0
    if (do_nodes or do_edges) and val_n_pred:
        metrics["avg_pred_nodes"] = float(np.mean(val_n_pred))
        metrics["avg_gt_nodes"] = float(np.mean(val_n_gt))
    if do_nodes and val_pred_scores:
        metrics["pred_score_mean"] = float(np.mean(val_pred_scores))
    if do_nodes and val_gt_hm_vals:
        gt_vals = np.array(val_gt_hm_vals)
        metrics["gt_hm_mean"] = float(gt_vals.mean())
        metrics["gt_hm_median"] = float(np.median(gt_vals))
        metrics["gt_hm_below_0.3"] = float((gt_vals < 0.3).mean())
        metrics["gt_hm_below_0.1"] = float((gt_vals < 0.1).mean())
    return metrics


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    do_nodes = args.phase in ("P3a", "P3b")
    do_edges = args.phase in ("P3b", "P3h")

    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # -- Data --
    all_pairs = discover_all_samples(args.data_root)
    if not all_pairs:
        print("ERROR: No data found.")
        sys.exit(1)
    train_pairs, val_pairs = split_data(all_pairs, args.val_ratio, args.seed)
    if args.overfit > 0:
        train_pairs = train_pairs[:args.overfit]
        val_pairs = train_pairs
        print(f"OVERFIT MODE: {len(train_pairs)} images")
    print(f"Data: {len(all_pairs)} total, {len(train_pairs)} train, {len(val_pairs)} val")

    train_loader = DataLoader(
        DamSegmentP3Dataset(train_pairs, augment=not args.overfit,
                            heatmap_sigma=args.heatmap_sigma),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
        collate_fn=p3_collate_fn,
    )
    val_loader = DataLoader(
        DamSegmentP3Dataset(val_pairs, augment=False, heatmap_sigma=args.heatmap_sigma),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
        collate_fn=p3_collate_fn,
    )

    # -- Model --
    model = MorphoGraphNet(
        backbone="mit_b2", num_classes=NUM_CLASSES,
        fpn_dim=FPN_DIM, graph_heads=(do_nodes or do_edges),
    ).to(device)
    if args.b2_checkpoint:
        loaded, skipped = load_b2_into_p3(str(args.b2_checkpoint), model, device)
        print(f"B2 weights: {len(loaded)} transferred, {len(skipped)} skipped")

    param_counts = model.count_parameters()
    print(f"Parameters: {param_counts['total']:,} total")

    # -- Optimizer + scheduler --
    param_groups = model.get_param_groups(encoder_lr=args.encoder_lr, head_lr=args.head_lr)
    optimizer = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)
    scheduler = make_cosine_schedule(
        optimizer, len(train_loader) * args.epochs, len(train_loader) * args.warmup_epochs,
    )

    # -- Losses --
    seg_loss_fn = WeightedCEDiceLoss(class_weights=DEFAULT_CE_WEIGHTS, ignore_index=255).to(device)
    skel_loss_fn = DTRegressionLoss(loss_type="mse").to(device)
    node_loss_fn = NodeHeatmapLoss(
        beta=args.loss_beta, pos_threshold=args.loss_pos_threshold,
    ).to(device)
    edge_loss_fn = EdgeBCELoss().to(device)
    node_schedule = LossSchedule(weight=args.node_weight, start_epoch=0, ramp_epochs=args.node_ramp_epochs)
    edge_schedule = LossSchedule(weight=args.edge_weight, start_epoch=args.edge_start_epoch, ramp_epochs=args.edge_ramp_epochs)

    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")

    # -- KNN candidate recall audit --
    if do_edges:
        print("Auditing KNN candidate recall on first 50 train samples...")
        audit_ds = DamSegmentP3Dataset(train_pairs[:min(50, len(train_pairs))],
                                       augment=False, heatmap_sigma=args.heatmap_sigma)
        audit_recalls = []
        for a_idx in range(len(audit_ds)):
            sample = audit_ds[a_idx]
            coords, types, edges = sample["gt_node_coords"], sample["gt_node_types"], sample["gt_edges"]
            if len(coords) < 2:
                continue
            cands = build_candidate_pairs(coords, k=args.knn_k)
            p2g = {i: i for i in range(len(coords))}
            info = audit_candidate_recall(coords, p2g, edges, cands)
            audit_recalls.append(info["recall"])
        if audit_recalls:
            mean_recall = np.mean(audit_recalls)
            print(f"  KNN recall (k={args.knn_k}): mean={mean_recall:.3f}, min={min(audit_recalls):.3f}")
            if mean_recall < 0.95:
                print(f"  WARNING: recall < 0.95 — consider increasing --knn-k")

    # -- Training loop --
    print(f"\nTraining {args.phase} for {args.epochs} epochs ({len(train_loader)} batches/epoch)\n")
    best_miou_fg = 0.0
    best_node_f1 = 0.0
    history = defaultdict(list)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        if epoch <= args.freeze_encoder_epochs:
            for p in model.encoder.parameters():
                p.requires_grad = False
        elif epoch == args.freeze_encoder_epochs + 1:
            for p in model.encoder.parameters():
                p.requires_grad = True
            print(f"  Epoch {epoch}: unfreezing encoder")

        avgs = train_one_epoch(
            model, train_loader, optimizer, scheduler, scaler, args,
            seg_loss_fn, skel_loss_fn, node_loss_fn, edge_loss_fn,
            node_schedule, edge_schedule, epoch, do_nodes, do_edges, device,
        )
        for k, v in avgs.items():
            history[f"train_{k}"].append(v)

        val_metrics = validate(model, val_loader, args, do_nodes, do_edges, device)
        history["val_mIoU_fg"].append(val_metrics["mIoU_fg"])
        if do_nodes or do_edges:
            for t in [5, 10, 15]:
                history[f"val_node_f1@{t}px"].append(val_metrics[f"node_f1@{t}px"])
                history[f"val_ep_f1@{t}px"].append(val_metrics[f"ep_f1@{t}px"])
                history[f"val_jn_f1@{t}px"].append(val_metrics[f"jn_f1@{t}px"])
                history[f"val_edge_f1@{t}px"].append(val_metrics[f"edge_f1@{t}px"])
                history[f"val_edge_gt_f1@{t}px"].append(val_metrics[f"edge_gt_f1@{t}px"])
            for diag_key in ["avg_pred_nodes", "avg_gt_nodes", "gt_hm_mean",
                              "gt_hm_median", "gt_hm_below_0.3", "gt_hm_below_0.1"]:
                if diag_key in val_metrics:
                    history[f"val_{diag_key}"].append(val_metrics[diag_key])

        # Checkpoint
        if do_nodes:
            curr_nf1 = val_metrics["node_f1@5px"]
            is_best = val_metrics["mIoU_fg"] > best_miou_fg or (
                val_metrics["mIoU_fg"] >= best_miou_fg - 0.005 and curr_nf1 > best_node_f1
            )
            if is_best:
                best_miou_fg = max(best_miou_fg, val_metrics["mIoU_fg"])
                best_node_f1 = max(best_node_f1, curr_nf1)
        elif do_edges:
            # P3h: checkpoint on edge_gt_f1
            curr_egt = val_metrics.get("edge_gt_f1@10px", 0.0)
            is_best = val_metrics["mIoU_fg"] > best_miou_fg or curr_egt > best_node_f1
            if is_best:
                best_miou_fg = max(best_miou_fg, val_metrics["mIoU_fg"])
                best_node_f1 = max(best_node_f1, curr_egt)
        else:
            is_best = val_metrics["mIoU_fg"] > best_miou_fg
            if is_best:
                best_miou_fg = val_metrics["mIoU_fg"]

        if is_best:
            save_checkpoint(args.output / "best.pt", model, optimizer, epoch, best_miou_fg, args)
        save_checkpoint(args.output / "last.pt", model, optimizer, epoch, best_miou_fg, args)

        elapsed = time.time() - t0
        loss_str = " ".join(f"{k}={v:.4f}" for k, v in avgs.items())
        graph_str = ""
        if do_nodes:
            ap = val_metrics.get('avg_pred_nodes', 0)
            ag = val_metrics.get('avg_gt_nodes', 0)
            gt_below = val_metrics.get('gt_hm_below_0.3', 0)
            graph_str = (f"nF1@5={val_metrics['node_f1@5px']:.3f}"
                         f"(ep={val_metrics['ep_f1@5px']:.3f},jn={val_metrics['jn_f1@5px']:.3f})"
                         f" n={ap:.0f}/{ag:.0f} gt<0.3={gt_below:.0%}")
        elif do_edges:
            ap = val_metrics.get('avg_pred_nodes', 0)
            ag = val_metrics.get('avg_gt_nodes', 0)
            graph_str = (f"nF1@10={val_metrics.get('node_f1@10px', 0):.3f}"
                         f" n={ap:.0f}/{ag:.0f}")
        if do_edges:
            graph_str += (f" eF1@10={val_metrics.get('edge_f1@10px', 0):.3f}"
                          f" eGT@10={val_metrics.get('edge_gt_f1@10px', 0):.3f}")
        print(f"Epoch {epoch:3d}/{args.epochs} | {loss_str} | "
              f"mIoU={val_metrics['mIoU_fg']:.4f} {graph_str} | "
              f"{elapsed:.0f}s{' *' if is_best else ''}")

    # -- Save --
    with open(args.output / "history.json", "w") as f:
        json.dump(dict(history), f, indent=2)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        for k in ["seg", "skel", "node"] + (["edge"] if do_edges else []):
            if f"train_{k}" in history:
                axes[0].plot(history[f"train_{k}"], label=k)
        axes[0].set_title("Train Loss"); axes[0].legend(fontsize=8); axes[0].grid(True, alpha=0.3)
        axes[1].plot(history["val_mIoU_fg"], label="mIoU_fg")
        axes[1].set_title("Val mIoU"); axes[1].legend(); axes[1].grid(True, alpha=0.3)
        if do_nodes or do_edges:
            for t in [5, 10, 15]:
                if f"val_node_f1@{t}px" in history:
                    axes[2].plot(history[f"val_node_f1@{t}px"], label=f"node@{t}px")
                if do_edges and f"val_edge_f1@{t}px" in history:
                    axes[2].plot(history[f"val_edge_f1@{t}px"], label=f"edge@{t}px", linestyle="--")
            axes[2].set_title("Graph Metrics"); axes[2].legend(fontsize=7); axes[2].grid(True, alpha=0.3)
        else:
            axes[2].text(0.5, 0.5, "P3-Base\n(no graph heads)",
                         ha="center", va="center", fontsize=12, transform=axes[2].transAxes)
            axes[2].set_title("Graph Metrics"); axes[2].axis("off")
        plt.tight_layout()
        plt.savefig(args.output / "training_curves.png", dpi=150)
        plt.close()
    except Exception as e:
        print(f"Plot failed: {e}")

    summary = {
        "phase": args.phase, "best_miou_fg": best_miou_fg,
        "best_node_f1_5px": best_node_f1 if do_nodes else None,
        "best_edge_gt_f1_10px": best_node_f1 if (do_edges and not do_nodes) else None,
        "final_val_node_f1_5px": history.get("val_node_f1@5px", [None])[-1] if (do_nodes or do_edges) else None,
        "final_val_ep_f1_5px": history.get("val_ep_f1@5px", [None])[-1] if (do_nodes or do_edges) else None,
        "final_val_jn_f1_5px": history.get("val_jn_f1@5px", [None])[-1] if (do_nodes or do_edges) else None,
        "final_val_edge_f1_10px": history.get("val_edge_f1@10px", [None])[-1] if do_edges else None,
        "final_val_edge_gt_f1_10px": history.get("val_edge_gt_f1@10px", [None])[-1] if do_edges else None,
        "epochs": args.epochs, "total_params": param_counts["total"],
        "overfit": args.overfit, "train_samples": len(train_pairs),
    }
    with open(args.output / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{args.phase} complete. Best mIoU_fg={best_miou_fg:.4f}")
    if do_nodes:
        print(f"Best node F1@5px={best_node_f1:.4f}")
    elif do_edges:
        print(f"Best edge_gt F1@10px={best_node_f1:.4f}")


if __name__ == "__main__":
    main()
