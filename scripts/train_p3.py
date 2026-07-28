"""P3 training: direct node and graph-topology prediction.

Usage:
    # P3a only (node detection):
    python scripts/train_p3.py --phase P3a --data-root data/raw --output runs/P3a

    # P3b (node + edge prediction):
    python scripts/train_p3.py --phase P3b --data-root data/raw --output runs/P3b

    # Overfit test (16 images, 200 epochs):
    python scripts/train_p3.py --phase P3b --data-root data/raw --output runs/P3b_overfit \
        --overfit 16 --epochs 200

    # Load from B2 checkpoint:
    python scripts/train_p3.py --phase P3b --data-root data/raw --output runs/P3b \
        --b2-checkpoint runs/B2_dt_v5/best.pt
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
from PIL import Image
from scipy import ndimage as ndi
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from morphograph.data.schema import decode_rgb_mask, NUM_CLASSES, DEFAULT_CE_WEIGHTS
from morphograph.data.graph_targets import (
    mask_to_dt_target, mask_to_graph,
)
from morphograph.losses.composite import WeightedCEDiceLoss, DTRegressionLoss, LossSchedule
from morphograph.losses.graph_loss import (
    NodeHeatmapLoss, EdgeBCELoss, P3bLossConfig, scheduled_sampling_prob,
)
from morphograph.models.morphograph_net import (
    MorphoGraphNet, load_b2_into_p3, FPN_DIM,
)
from morphograph.models.graph_decoder import (
    extract_nodes, build_candidate_pairs, DetectedNodes, audit_candidate_recall,
)
from morphograph.metrics.graph_metrics import (
    compute_graph_metrics, DEFAULT_KEYPOINT_TOLERANCE_PX,
    RELAXED_TOLERANCE_PX, LENIENT_TOLERANCE_PX,
)
from morphograph.training.utils import (
    set_seed, discover_all_samples, split_data,
    compute_miou, make_cosine_schedule, save_checkpoint,
)


# ---------------------------------------------------------------------------
# Target generation
# ---------------------------------------------------------------------------

def _make_gaussian_heatmap_128(
    coords: np.ndarray,
    shape: tuple[int, int] = (128, 128),
    sigma: float = 1.5,
) -> np.ndarray:
    """Gaussian heatmap at 128x128 with peak=1.0 at each coordinate."""
    heatmap = np.zeros(shape, dtype=np.float32)
    for r, c in coords:
        ri, ci = int(round(r)), int(round(c))
        if 0 <= ri < shape[0] and 0 <= ci < shape[1]:
            heatmap[ri, ci] = 1.0
    if heatmap.any():
        heatmap = ndi.gaussian_filter(heatmap, sigma=sigma)
        heatmap = heatmap / (heatmap.max() + 1e-8)
    return heatmap


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class DamSegmentP3Dataset(Dataset):
    """DamSegment dataset with graph targets for P3."""

    def __init__(
        self,
        pairs: list[tuple[Path, Path]],
        img_size: int = 512,
        augment: bool = False,
        heatmap_sigma: float = 1.5,
    ) -> None:
        self.pairs = pairs
        self.img_size = img_size
        self.augment = augment
        self.heatmap_sigma = heatmap_sigma
        self._transform = None
        if augment:
            self._transform = self._build_augmentation()

    def _build_augmentation(self):
        import albumentations as A
        # Only geometric + photometric augmentations on image+mask.
        # DT and graph targets are re-derived from the augmented mask,
        # NOT transformed as additional targets. This is critical because:
        # - Geometric transforms of float DT values are invalid
        #   (DT depends on distance to mask boundary, not pixel values)
        # - Crop/affine can split components, create new endpoints,
        #   change adjacency — graph must be re-extracted post-augmentation
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
        ])

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict:
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

        # Apply augmentation FIRST (geometry changes topology)
        if self._transform is not None:
            transformed = self._transform(image=img, mask=mask)
            img = transformed["image"]
            mask = transformed["mask"]

        # Re-derive all targets from the (possibly augmented) mask
        crack_binary = (mask == 1).astype(np.uint8)
        dt_target = mask_to_dt_target(crack_binary)

        # Extract graph from augmented crack mask
        # Known limitation: closed-loop components (0 endpoints) are not
        # representable by endpoint/junction heatmaps alone. Per Step 0
        # statistics, if <2% of components are pure loops, this is acceptable.
        graph = mask_to_graph(crack_binary, min_branch_length=10, junction_merge_radius=5)

        # Node coordinates at 128-space (divide by 4)
        scale = self.img_size / 128.0  # 4.0 for 512->128
        gt_endpoints_128 = graph.endpoints.astype(np.float32) / scale
        gt_junctions_128 = graph.junctions.astype(np.float32) / scale

        # Node heatmaps at 128x128
        ep_heatmap = _make_gaussian_heatmap_128(
            gt_endpoints_128, sigma=self.heatmap_sigma,
        )
        jn_heatmap = _make_gaussian_heatmap_128(
            gt_junctions_128, sigma=self.heatmap_sigma,
        )
        # Stack: (2, 128, 128)
        node_heatmap = np.stack([ep_heatmap, jn_heatmap], axis=0)

        # GT node coords + types for edge classifier
        all_coords_128 = np.concatenate(
            [gt_endpoints_128, gt_junctions_128], axis=0
        ) if len(gt_endpoints_128) + len(gt_junctions_128) > 0 else np.empty((0, 2), dtype=np.float32)
        n_ep = len(gt_endpoints_128)
        node_types = np.array(
            [0] * n_ep + [1] * len(gt_junctions_128), dtype=np.int64
        )

        out = {
            "image": torch.from_numpy(img).permute(2, 0, 1).float() / 255.0,
            "mask": torch.from_numpy(mask.copy()).long(),
            "dt_target": torch.from_numpy(dt_target.copy()).float().unsqueeze(0),
            "crack_mask": torch.from_numpy(crack_binary.copy()).float().unsqueeze(0),
            "node_heatmap": torch.from_numpy(node_heatmap),
            "gt_node_coords": torch.from_numpy(all_coords_128),
            "gt_node_types": torch.from_numpy(node_types),
            "gt_edges": graph.edges,
            "gt_num_endpoints": n_ep,
        }
        return out


def p3_collate_fn(batch: list[dict]) -> dict:
    """Custom collate: stack fixed-size, keep variable-size as lists."""
    result = {}
    # Fixed-size tensors
    for key in ["image", "mask", "dt_target", "crack_mask", "node_heatmap"]:
        result[key] = torch.stack([b[key] for b in batch])
    # Variable-size: keep as lists
    for key in ["gt_node_coords", "gt_node_types", "gt_edges", "gt_num_endpoints"]:
        result[key] = [b[key] for b in batch]
    return result


# ---------------------------------------------------------------------------
# Edge label assignment via Hungarian matching
# ---------------------------------------------------------------------------

def build_edge_labels(
    pred_coords: torch.Tensor,
    pred_types: torch.Tensor,
    gt_coords: torch.Tensor,
    gt_types: torch.Tensor,
    gt_edges: list[tuple[int, int]],
    candidate_pairs: torch.Tensor,
    max_match_dist: float = 2.5,
    type_match_strict: bool = True,
    unmatched_far_threshold: float = 5.0,
    unmatched_neg_weight: float = 0.3,
) -> tuple[torch.Tensor, torch.Tensor, dict[int, int]]:
    """Assign edge labels via type-aware Hungarian matching.

    All coordinates are in 128-space. max_match_dist=2.5 corresponds
    to 10px in 512-space (2.5 * 4 = 10), matching the relaxed eval
    tolerance. For stricter training, use 1.25 (= 5px in 512-space).

    Three-state label assignment (Issue #10):
      - Both endpoints matched: label from GT adjacency (pos/neg), weight=1.0
      - Clearly unmatched (min dist to GT > unmatched_far_threshold):
        negative edge, weight=unmatched_neg_weight (penalize false detections)
      - Near matching threshold (ambiguous): IGNORE (weight=0.0)

    Zero-GT-edge handling (Issue #9):
      - < 2 GT nodes: skip (caller handles this)
      - >= 2 GT nodes, 0 GT edges: all candidates are negative (weight=1.0)
      - > 0 GT edges: matched pos/neg + unmatched neg/ignore

    Args:
        type_match_strict: if True, endpoints only match endpoints,
            junctions only match junctions (hard constraint via inf cost).
        unmatched_far_threshold: distance in 128-space above which an
            unmatched node is "clearly false" (default 5.0 = 20px fullres).
        unmatched_neg_weight: loss weight for edges involving clearly
            unmatched (false positive) nodes.

    Returns:
        labels: (E,) binary labels (0/1).
        loss_weight: (E,) per-edge loss weights (0=ignore, 0.3=unmatched neg, 1.0=matched).
        pred_to_gt: dict mapping pred node idx -> GT node idx.
    """
    device = pred_coords.device
    E = len(candidate_pairs)
    labels = torch.zeros(E, device=device)
    loss_weight = torch.zeros(E, device=device)
    pred_to_gt: dict[int, int] = {}

    if len(pred_coords) == 0 or len(gt_coords) == 0:
        return labels, loss_weight, pred_to_gt

    # Build cost matrix
    pc = pred_coords.detach().cpu().numpy()
    gc = gt_coords.detach().cpu().numpy()
    pt = pred_types.detach().cpu().numpy()
    gt = gt_types.detach().cpu().numpy()

    dists = cdist(pc, gc)

    if type_match_strict:
        type_mismatch = (pt[:, None].astype(int) != gt[None, :].astype(int))
        cost = dists.copy()
        cost[type_mismatch] = 1e6
    else:
        type_mismatch = np.abs(pt[:, None].astype(float) - gt[None, :].astype(float))
        cost = dists + 5.0 * type_mismatch

    row_ind, col_ind = linear_sum_assignment(cost)
    for r, c in zip(row_ind, col_ind):
        if dists[r, c] <= max_match_dist:
            pred_to_gt[int(r)] = int(c)

    # Classify unmatched predicted nodes: "clearly far" vs "ambiguous"
    min_dist_to_gt = dists.min(axis=1)  # (N_pred,) min dist to any GT node
    is_clearly_unmatched = {}  # pred_idx -> bool
    for i in range(len(pc)):
        if i not in pred_to_gt:
            is_clearly_unmatched[i] = min_dist_to_gt[i] > unmatched_far_threshold

    # Build GT edge set
    gt_edge_set = set()
    for a, b in gt_edges:
        gt_edge_set.add((min(a, b), max(a, b)))

    # Assign labels with three-state logic
    for e_idx in range(E):
        a, b = candidate_pairs[e_idx].tolist()
        a_matched = a in pred_to_gt
        b_matched = b in pred_to_gt

        if a_matched and b_matched:
            # Both matched: label from GT adjacency
            loss_weight[e_idx] = 1.0
            mapped = (
                min(pred_to_gt[a], pred_to_gt[b]),
                max(pred_to_gt[a], pred_to_gt[b]),
            )
            if mapped in gt_edge_set:
                labels[e_idx] = 1.0
        elif (not a_matched and is_clearly_unmatched.get(a, False)) or \
             (not b_matched and is_clearly_unmatched.get(b, False)):
            # At least one endpoint is clearly a false detection -> negative
            labels[e_idx] = 0.0
            loss_weight[e_idx] = unmatched_neg_weight
        else:
            # Ambiguous: near matching threshold -> IGNORE
            loss_weight[e_idx] = 0.0

    return labels, loss_weight, pred_to_gt


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="P3 training: direct graph prediction")
    parser.add_argument("--phase", choices=["P3-Base", "P3a", "P3b"], default="P3b")
    parser.add_argument("--data-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--b2-checkpoint", type=Path, default=None,
                        help="B2 checkpoint for weight transfer")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--encoder-lr", type=float, default=6e-5)
    parser.add_argument("--head-lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--freeze-encoder-epochs", type=int, default=5,
                        help="Freeze encoder for first N epochs")
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--amp", action="store_true", default=True)
    # Loss weights
    parser.add_argument("--skel-weight", type=float, default=10.0)
    parser.add_argument("--node-weight", type=float, default=1.0)
    parser.add_argument("--edge-weight", type=float, default=1.0)
    parser.add_argument("--node-ramp-epochs", type=int, default=5)
    parser.add_argument("--edge-start-epoch", type=int, default=10)
    parser.add_argument("--edge-ramp-epochs", type=int, default=10)
    # Scheduled sampling
    parser.add_argument("--ss-warmup", type=int, default=10)
    parser.add_argument("--ss-anneal-end", type=int, default=60)
    # Node extraction params
    parser.add_argument("--node-threshold", type=float, default=0.3)
    parser.add_argument("--nms-radius", type=int, default=3)
    parser.add_argument("--max-nodes", type=int, default=50)
    parser.add_argument("--heatmap-sigma", type=float, default=1.5)
    # KNN
    parser.add_argument("--knn-k", type=int, default=8)
    # Overfit test
    parser.add_argument("--overfit", type=int, default=0,
                        help="If >0, use only N images for overfit test")
    args = parser.parse_args()

    set_seed(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
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
        val_pairs = train_pairs  # overfit: val = train
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
        DamSegmentP3Dataset(val_pairs, augment=False,
                            heatmap_sigma=args.heatmap_sigma),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
        collate_fn=p3_collate_fn,
    )

    # -- Model --
    use_graph_heads = args.phase in ("P3a", "P3b")
    print(f"Building MorphoGraphNet (graph_heads={use_graph_heads})...")
    model = MorphoGraphNet(
        backbone="mit_b2",
        num_classes=NUM_CLASSES,
        fpn_dim=FPN_DIM,
        graph_heads=use_graph_heads,
    ).to(device)

    if args.b2_checkpoint:
        print(f"Loading B2 weights from {args.b2_checkpoint}...")
        loaded, skipped = load_b2_into_p3(str(args.b2_checkpoint), model, device)
        print(f"  Transferred: {len(loaded)} keys, Skipped: {len(skipped)} keys")

    param_counts = model.count_parameters()
    print(f"Parameters: {param_counts['total']:,} total")
    for k, v in param_counts.items():
        if k not in ("total", "trainable"):
            print(f"  {k}: {v:,}")

    # -- Optimizer + scheduler --
    param_groups = model.get_param_groups(
        encoder_lr=args.encoder_lr, head_lr=args.head_lr,
    )
    optimizer = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)
    total_steps = len(train_loader) * args.epochs
    warmup_steps = len(train_loader) * args.warmup_epochs
    scheduler = make_cosine_schedule(optimizer, total_steps, warmup_steps)

    # -- Losses --
    seg_loss_fn = WeightedCEDiceLoss(
        class_weights=DEFAULT_CE_WEIGHTS, ignore_index=255,
    ).to(device)
    skel_loss_fn = DTRegressionLoss(loss_type="mse").to(device)
    node_loss_fn = NodeHeatmapLoss().to(device)
    edge_loss_fn = EdgeBCELoss().to(device)

    node_schedule = LossSchedule(
        weight=args.node_weight, start_epoch=0, ramp_epochs=args.node_ramp_epochs,
    )
    edge_schedule = LossSchedule(
        weight=args.edge_weight, start_epoch=args.edge_start_epoch,
        ramp_epochs=args.edge_ramp_epochs,
    )

    do_nodes = args.phase in ("P3a", "P3b")
    do_edges = args.phase == "P3b"

    print(f"\nPhase: {args.phase}")
    loss_desc = f"seg + skel(w={args.skel_weight})"
    if do_nodes:
        loss_desc += f" + node(w={args.node_weight})"
    if do_edges:
        loss_desc += f" + edge(w={args.edge_weight})"
    print(f"Losses: {loss_desc}")
    if do_edges:
        print(f"Scheduled sampling: warmup={args.ss_warmup}, anneal_end={args.ss_anneal_end}")

    # -- AMP --
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")

    # -- Training --
    best_miou_fg = 0.0
    best_node_f1 = 0.0
    history = defaultdict(list)

    # -- Candidate recall audit (Issue #11) --
    if do_edges:
        print("Auditing KNN candidate recall on first 50 train samples...")
        audit_recalls = []
        audit_ds = DamSegmentP3Dataset(train_pairs[:min(50, len(train_pairs))],
                                       augment=False, heatmap_sigma=args.heatmap_sigma)
        for a_idx in range(len(audit_ds)):
            sample = audit_ds[a_idx]
            coords = sample["gt_node_coords"]
            types = sample["gt_node_types"]
            edges = sample["gt_edges"]
            if len(coords) < 2:
                continue
            cands = build_candidate_pairs(coords, k=args.knn_k)
            # All GT nodes match themselves: identity pred_to_gt
            p2g = {i: i for i in range(len(coords))}
            info = audit_candidate_recall(coords, p2g, edges, cands)
            audit_recalls.append(info["recall"])
        if audit_recalls:
            mean_recall = np.mean(audit_recalls)
            print(f"  KNN candidate recall (k={args.knn_k}): "
                  f"mean={mean_recall:.3f}, min={min(audit_recalls):.3f}")
            if mean_recall < 0.95:
                print(f"  WARNING: recall < 0.95 — consider increasing --knn-k")
        print()

    print(f"\nTraining {args.phase} for {args.epochs} epochs...")
    print(f"  Batches/epoch: {len(train_loader)}\n")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        # Encoder freeze schedule
        if epoch <= args.freeze_encoder_epochs:
            for p in model.encoder.parameters():
                p.requires_grad = False
        elif epoch == args.freeze_encoder_epochs + 1:
            for p in model.encoder.parameters():
                p.requires_grad = True
            print(f"  Epoch {epoch}: unfreezing encoder")

        # -- Train --
        model.train()
        epoch_losses = defaultdict(list)

        for batch in train_loader:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            dt_targets = batch["dt_target"].to(device)
            node_heatmaps = batch["node_heatmap"].to(device)

            with torch.amp.autocast("cuda", enabled=args.amp and device.type == "cuda"):
                outputs = model(images)

                # Seg loss
                seg_loss = seg_loss_fn(outputs["seg"], masks)["total"]

                # Skeleton DT loss
                skel_pred = torch.sigmoid(outputs["skeleton"])
                skel_mask = torch.ones_like(dt_targets)
                skel_loss = skel_loss_fn(skel_pred, dt_targets, skel_mask)

                # Node heatmap loss (P3a/P3b only)
                node_loss = torch.tensor(0.0, device=device)
                if do_nodes:
                    node_loss = node_loss_fn(outputs["node_heatmap"], node_heatmaps)

                node_w = node_schedule.effective_weight(epoch) if do_nodes else 0.0
                total_loss = seg_loss + args.skel_weight * skel_loss + node_w * node_loss

                # Edge loss (P3b only)
                edge_loss_val = torch.tensor(0.0, device=device)
                if do_edges and edge_schedule.effective_weight(epoch) > 0:
                    p_gt = scheduled_sampling_prob(
                        epoch, args.ss_warmup, args.ss_anneal_end,
                    )
                    batch_edge_loss = []
                    B = images.shape[0]

                    for b_idx in range(B):
                        gt_coords = batch["gt_node_coords"][b_idx].to(device)
                        gt_types = batch["gt_node_types"][b_idx].to(device)
                        gt_edges = batch["gt_edges"][b_idx]

                        # < 2 GT nodes: skip (no edges possible)
                        if len(gt_coords) < 2:
                            continue
                        # >= 2 GT nodes, 0 GT edges: train all-negative
                        # (do NOT skip — this is valid supervision)

                        # Scheduled sampling: use GT or predicted nodes
                        if random.random() < p_gt:
                            node_coords = gt_coords
                            node_types = gt_types
                            node_scores = torch.ones(len(gt_coords), device=device)
                        else:
                            hm = torch.sigmoid(outputs["node_heatmap"][b_idx:b_idx+1])
                            detected = extract_nodes(
                                hm, threshold=args.node_threshold,
                                nms_radius=args.nms_radius,
                                max_nodes=args.max_nodes,
                            )[0]
                            if len(detected.coords) < 2:
                                continue
                            node_coords = detected.coords
                            node_types = detected.types
                            node_scores = detected.scores

                        # First pass: Hungarian matching to get pred_to_gt
                        # (needed for GT edge guarantee in candidate selection)
                        dummy_candidates = build_candidate_pairs(
                            node_coords, k=args.knn_k,
                        )
                        _, _, pred_to_gt = build_edge_labels(
                            node_coords, node_types,
                            gt_coords, gt_types, gt_edges,
                            dummy_candidates,
                        )

                        # Build candidates with GT guarantee
                        candidates = build_candidate_pairs(
                            node_coords, k=args.knn_k,
                            gt_edges=gt_edges,
                            pred_to_gt=pred_to_gt,
                        )

                        if len(candidates) == 0:
                            continue

                        # Final edge labels (three-state: matched/unmatched-neg/ignore)
                        labels, loss_weight, pred_to_gt = build_edge_labels(
                            node_coords, node_types,
                            gt_coords, gt_types, gt_edges,
                            candidates,
                        )

                        # Normalized DT at 128x128 for corridor path evidence
                        dt_128 = F.interpolate(
                            skel_pred[b_idx:b_idx+1],
                            size=(128, 128), mode="bilinear", align_corners=False,
                        )

                        edge_logits = model.edge_classifier(
                            outputs["_fpn"][b_idx:b_idx+1],
                            dt_128,
                            node_coords, node_types, node_scores,
                            candidates,
                        )

                        b_edge_loss = edge_loss_fn(edge_logits, labels, loss_weight)
                        batch_edge_loss.append(b_edge_loss)

                    if batch_edge_loss:
                        edge_loss_val = torch.stack(batch_edge_loss).mean()
                        edge_w = edge_schedule.effective_weight(epoch)
                        total_loss = total_loss + edge_w * edge_loss_val

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

        avgs = {k: np.mean(v) for k, v in epoch_losses.items()}
        for k, v in avgs.items():
            history[f"train_{k}"].append(v)

        # -- Validate --
        model.eval()
        val_seg_preds = []
        val_seg_targets = []
        val_node_f1s = {t: [] for t in [5, 10, 15]}
        val_edge_f1s = {t: [] for t in [5, 10, 15]}

        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(device)
                masks_val = batch["mask"].to(device)

                with torch.amp.autocast("cuda", enabled=args.amp and device.type == "cuda"):
                    outputs = model(images)

                val_seg_preds.append(outputs["seg"].argmax(dim=1).cpu())
                val_seg_targets.append(masks_val.cpu())

                # Node + edge metrics per image (P3a/P3b only)
                if do_nodes:
                    B = images.shape[0]
                    hm = torch.sigmoid(outputs["node_heatmap"])

                    for b_idx in range(B):
                        gt_coords = batch["gt_node_coords"][b_idx]
                        gt_types = batch["gt_node_types"][b_idx]
                        gt_edges = batch["gt_edges"][b_idx]

                        if len(gt_coords) == 0:
                            continue

                        detected = extract_nodes(
                            hm[b_idx:b_idx+1],
                            threshold=args.node_threshold,
                            nms_radius=args.nms_radius,
                            max_nodes=args.max_nodes,
                        )[0]

                        # Scale to 512-space for metric computation
                        pred_coords_512 = detected.coords.cpu().numpy() * 4.0
                        gt_coords_512 = gt_coords.numpy() * 4.0

                        # Split into endpoints/junctions
                        pred_ep_mask = detected.types.cpu().numpy() == 0
                        pred_jn_mask = detected.types.cpu().numpy() == 1
                        gt_ep_mask = gt_types.numpy() == 0
                        gt_jn_mask = gt_types.numpy() == 1

                        pred_ep = pred_coords_512[pred_ep_mask] if pred_ep_mask.any() else np.empty((0, 2))
                        pred_jn = pred_coords_512[pred_jn_mask] if pred_jn_mask.any() else np.empty((0, 2))
                        gt_ep = gt_coords_512[gt_ep_mask] if gt_ep_mask.any() else np.empty((0, 2))
                        gt_jn = gt_coords_512[gt_jn_mask] if gt_jn_mask.any() else np.empty((0, 2))

                        # Edge prediction (always use predicted nodes at val)
                        pred_edges = []
                        if do_edges and len(detected.coords) >= 2:
                            dt_128 = F.interpolate(
                                torch.sigmoid(outputs["skeleton"][b_idx:b_idx+1]),
                                size=(128, 128), mode="bilinear", align_corners=False,
                            )
                            candidates = build_candidate_pairs(
                                detected.coords, k=args.knn_k,
                            )
                            if len(candidates) > 0:
                                edge_logits = model.edge_classifier(
                                    outputs["_fpn"][b_idx:b_idx+1],
                                    dt_128,
                                    detected.coords,
                                    detected.types,
                                    detected.scores,
                                    candidates,
                                )
                                pred_edge_mask = torch.sigmoid(edge_logits) > 0.5
                                pred_edges = candidates[pred_edge_mask].cpu().tolist()
                                pred_edges = [(min(a, b), max(a, b)) for a, b in pred_edges]

                        for tol_px, tol_key in [(5, 5), (10, 10), (15, 15)]:
                            gm = compute_graph_metrics(
                                pred_ep, pred_jn, pred_edges,
                                gt_ep, gt_jn, gt_edges,
                                tolerance_px=float(tol_px),
                            )
                            node_f1 = (gm.endpoint_f1 + gm.junction_f1) / 2
                            val_node_f1s[tol_key].append(node_f1)
                            val_edge_f1s[tol_key].append(gm.edge_f1)

        miou = compute_miou(torch.cat(val_seg_preds), torch.cat(val_seg_targets))
        history["val_mIoU_fg"].append(miou["mIoU_fg"])

        if do_nodes:
            for t in [5, 10, 15]:
                nf = np.mean(val_node_f1s[t]) if val_node_f1s[t] else 0.0
                ef = np.mean(val_edge_f1s[t]) if val_edge_f1s[t] else 0.0
                history[f"val_node_f1@{t}px"].append(float(nf))
                history[f"val_edge_f1@{t}px"].append(float(ef))

        elapsed = time.time() - t0

        # -- Checkpoint --
        if do_nodes:
            curr_node_f1 = history["val_node_f1@5px"][-1]
            is_best = miou["mIoU_fg"] > best_miou_fg or (
                miou["mIoU_fg"] >= best_miou_fg - 0.005 and curr_node_f1 > best_node_f1
            )
            if is_best:
                best_miou_fg = max(best_miou_fg, miou["mIoU_fg"])
                best_node_f1 = max(best_node_f1, curr_node_f1)
        else:
            # P3-Base: checkpoint purely on mIoU
            is_best = miou["mIoU_fg"] > best_miou_fg
            if is_best:
                best_miou_fg = miou["mIoU_fg"]

        if is_best:
            save_checkpoint(args.output / "best.pt", model, optimizer, epoch, best_miou_fg, args)
        save_checkpoint(args.output / "last.pt", model, optimizer, epoch, best_miou_fg, args)

        # -- Log --
        loss_str = " ".join(f"{k}={v:.4f}" for k, v in avgs.items())
        graph_str = ""
        if do_nodes:
            graph_str = f"nF1@5={history['val_node_f1@5px'][-1]:.3f}"
        if do_edges:
            graph_str += f" eF1@5={history['val_edge_f1@5px'][-1]:.3f}"
        best_marker = " *" if is_best else ""
        print(
            f"Epoch {epoch:3d}/{args.epochs} | {loss_str} | "
            f"mIoU={miou['mIoU_fg']:.4f} {graph_str} | "
            f"{elapsed:.0f}s{best_marker}"
        )

    # -- Save history --
    with open(args.output / "history.json", "w") as f:
        json.dump(dict(history), f, indent=2)

    # -- Plot --
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        for k in ["seg", "skel", "node"] + (["edge"] if do_edges else []):
            axes[0].plot(history[f"train_{k}"], label=k)
        axes[0].set_title("Train Loss")
        axes[0].legend(fontsize=8)
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(history["val_mIoU_fg"], label="mIoU_fg")
        axes[1].set_title("Val mIoU")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        if do_nodes:
            for t in [5, 10, 15]:
                axes[2].plot(history[f"val_node_f1@{t}px"], label=f"node@{t}px")
                if do_edges:
                    axes[2].plot(history[f"val_edge_f1@{t}px"], label=f"edge@{t}px", linestyle="--")
            axes[2].set_title("Graph Metrics")
            axes[2].legend(fontsize=7)
            axes[2].grid(True, alpha=0.3)
        else:
            axes[2].text(0.5, 0.5, "P3-Base\n(no graph heads)",
                         ha="center", va="center", fontsize=12,
                         transform=axes[2].transAxes)
            axes[2].set_title("Graph Metrics")
            axes[2].axis("off")

        plt.tight_layout()
        plt.savefig(args.output / "training_curves.png", dpi=150)
        plt.close()
    except Exception as e:
        print(f"Plot failed: {e}")

    # -- Summary --
    summary = {
        "phase": args.phase,
        "graph_heads": use_graph_heads,
        "best_miou_fg": best_miou_fg,
        "best_node_f1_5px": best_node_f1 if do_nodes else None,
        "final_val_node_f1_5px": history.get("val_node_f1@5px", [None])[-1],
        "final_val_node_f1_10px": history.get("val_node_f1@10px", [None])[-1],
        "final_val_edge_f1_5px": history.get("val_edge_f1@5px", [None])[-1] if do_edges else None,
        "final_val_edge_f1_10px": history.get("val_edge_f1@10px", [None])[-1] if do_edges else None,
        "epochs": args.epochs,
        "total_params": param_counts["total"],
        "b2_checkpoint": str(args.b2_checkpoint) if args.b2_checkpoint else None,
        "overfit": args.overfit,
        "train_samples": len(train_pairs),
        "val_samples": len(val_pairs),
        "seed": args.seed,
    }
    with open(args.output / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{args.phase} training complete.")
    print(f"Best mIoU_fg = {best_miou_fg:.4f}")
    if do_nodes:
        print(f"Best node F1@5px = {best_node_f1:.4f}")
    print(f"Results saved to {args.output}/")


if __name__ == "__main__":
    main()
