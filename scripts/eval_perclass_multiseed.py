"""Per-class evaluation for multi-seed baseline runs.

Loads best.pt from each run directory, runs inference on the same
val split used during training, and computes:
  - iou_crack, iou_spalling (per-class IoU)
  - bf1_crack, bf1_spalling (boundary F1, tolerance=2px)
  - miou_fg (sanity-check; should match best_miou_fg in summary.json)

Results are merged into each run's summary.json under the key
"perclass" and a consolidated table is written to --output.

Usage (run on RunPod where best.pt files exist):
    python scripts/eval_perclass_multiseed.py \\
        --data-root data/raw \\
        --runs-dir runs \\
        --baselines B0 B1a B2_best B3 B5 \\
        --seeds 0 7 42 99 123 \\
        --output runs/perclass_summary.json

    # or target specific dirs:
    python scripts/eval_perclass_multiseed.py \\
        --data-root data/raw \\
        --run-dirs runs/B0_s0 runs/B0_s7 ... \\
        --output runs/perclass_summary.json

    # dry-run to check which dirs will be processed:
    python scripts/eval_perclass_multiseed.py --dry-run \\
        --runs-dir runs --baselines B0 B1a --seeds 0 7 42 99 123
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
from morphograph.metrics.segmentation import compute_iou, compute_boundary_f1
from morphograph.training.utils import DamSegmentDataset, discover_all_samples, split_data


# ---------------------------------------------------------------------------
# Model loading (mirrors eval_cross_domain.py)
# ---------------------------------------------------------------------------

def load_model(checkpoint_path: Path, device: torch.device):
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
        head_flags = {name: any(name in k for k in state)
                      for name in ["seg", "skeleton", "endpoints", "junctions", "width"]}
        model = MorphoAuxNet(
            backbone="mit_b2", num_classes=NUM_CLASSES,
            fpn_dim=FPN_DIM, heads=head_flags,
        )

    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Per-class evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_perclass(model, val_loader, device) -> dict:
    """Compute per-class IoU and BF1 over the val set."""
    iou_accum: dict[int, list] = defaultdict(list)
    bf1_accum: dict[int, list] = defaultdict(list)

    for batch in val_loader:
        images = batch["image"].to(device)
        masks = batch["mask"].numpy()

        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            outputs = model(images)

        preds = outputs["seg"].argmax(dim=1).cpu().numpy()

        for i in range(len(images)):
            gt = masks[i]
            pred = preds[i]

            iou = compute_iou(pred, gt)
            bf1 = compute_boundary_f1(pred, gt, tolerance_px=2)

            for c, v in iou.items():
                iou_accum[c].append(v)
            for c, v in bf1.items():
                bf1_accum[c].append(v)

    def mean(lst):
        return float(np.mean(lst)) if lst else None

    iou_crack    = mean(iou_accum[1])
    iou_spalling = mean(iou_accum[2])
    bf1_crack    = mean(bf1_accum[1])
    bf1_spalling = mean(bf1_accum[2])

    miou_fg = float(np.mean([v for v in [iou_crack, iou_spalling] if v is not None]))

    return {
        "iou_crack":    iou_crack,
        "iou_spalling": iou_spalling,
        "miou_fg":      miou_fg,
        "bf1_crack":    bf1_crack,
        "bf1_spalling": bf1_spalling,
        "n_val":        sum(len(b) for b in iou_accum.values()) // max(len(iou_accum), 1),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def resolve_run_dirs(args) -> list[Path]:
    """Return the list of run directories to process."""
    if args.run_dirs:
        return [Path(d) for d in args.run_dirs]

    # Build from --runs-dir + --baselines + --seeds
    runs_dir = Path(args.runs_dir)
    # Map canonical baseline name → directory prefix
    dir_prefix = {
        "B0":     "B0",
        "B1a":    "B1a",
        "B2_best":"B2_best",
        "B3":     "B3",
        "B5":     "B5",
    }
    dirs = []
    for bl in args.baselines:
        prefix = dir_prefix.get(bl, bl)
        for s in args.seeds:
            d = runs_dir / f"{prefix}_s{s}"
            dirs.append(d)
    return dirs


def main():
    parser = argparse.ArgumentParser(description="Per-class multi-seed eval")
    # Target runs
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--run-dirs", nargs="+", help="Explicit list of run directories")
    group.add_argument("--runs-dir", help="Root runs directory (use with --baselines/--seeds)")
    parser.add_argument("--baselines", nargs="+",
                        default=["B0", "B1a", "B2_best", "B3", "B5"],
                        help="Baseline names (used with --runs-dir)")
    parser.add_argument("--seeds", nargs="+", type=int,
                        default=[0, 7, 42, 99, 123],
                        help="Seeds (used with --runs-dir)")
    # Data
    parser.add_argument("--data-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--val-ratio", type=float, default=0.15)
    # Compute
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    # Output
    parser.add_argument("--output", type=Path, default=Path("runs/perclass_summary.json"))
    parser.add_argument("--dry-run", action="store_true",
                        help="Print which dirs would be processed without running eval")
    args = parser.parse_args()

    run_dirs = resolve_run_dirs(args)

    # ── Dry run ──
    if args.dry_run:
        print(f"Would process {len(run_dirs)} run directories:")
        for d in run_dirs:
            ckpt = d / "best.pt"
            status = "OK" if ckpt.exists() else "MISSING best.pt"
            print(f"  {d}  [{status}]")
        return

    device = torch.device(args.device)
    all_samples = discover_all_samples(args.data_root)
    if not all_samples:
        print(f"ERROR: No data found at {args.data_root}")
        sys.exit(1)

    consolidated: list[dict] = []
    n_ok = n_skip = 0

    for run_dir in run_dirs:
        run_dir = Path(run_dir)
        ckpt_path = run_dir / "best.pt"
        summary_path = run_dir / "summary.json"

        if not ckpt_path.exists():
            print(f"[SKIP] {run_dir}: no best.pt")
            n_skip += 1
            continue

        # Read existing summary to get seed
        seed = 42  # default
        existing = {}
        if summary_path.exists():
            existing = json.loads(summary_path.read_text())
            seed = existing.get("seed", seed)

        print(f"[EVAL] {run_dir.name}  seed={seed} ...", end=" ", flush=True)

        # Val split (must match training)
        _, val_pairs = split_data(all_samples, args.val_ratio, seed)
        val_loader = DataLoader(
            DamSegmentDataset(val_pairs, augment=False),
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=True,
        )

        try:
            model = load_model(ckpt_path, device)
            metrics = evaluate_perclass(model, val_loader, device)
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
        except Exception as e:
            print(f"ERROR: {e}")
            n_skip += 1
            continue

        print(
            f"iou_crack={metrics['iou_crack']:.4f}  "
            f"iou_spalling={metrics['iou_spalling']:.4f}  "
            f"bf1_crack={metrics['bf1_crack']:.4f}  "
            f"miou_fg={metrics['miou_fg']:.4f}  "
            f"(saved best_miou_fg={existing.get('best_miou_fg', '?'):.4f})"
        )

        # Merge into summary.json
        existing["perclass"] = metrics
        summary_path.write_text(json.dumps(existing, indent=2))

        consolidated.append({
            "run_dir": run_dir.name,
            "seed": seed,
            **metrics,
            "best_miou_fg": existing.get("best_miou_fg"),
        })
        n_ok += 1

    # ── Consolidated output ──
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(consolidated, indent=2))

    # ── Print table ──
    print()
    col = [20, 5, 9, 11, 9, 9, 9]
    header = (f"{'Run':^{col[0]}} {'Seed':^{col[1]}} "
              f"{'mIoU_fg':^{col[2]}} {'iou_crack':^{col[3]}} "
              f"{'iou_spall':^{col[4]}} {'bf1_crack':^{col[5]}} {'bf1_spall':^{col[6]}}")
    print(header)
    print("-" * (sum(col) + 6))
    for r in consolidated:
        print(
            f"{r['run_dir']:^{col[0]}} {r['seed']:^{col[1]}} "
            f"{r['miou_fg']:^{col[2]}.4f} {r['iou_crack']:^{col[3]}.4f} "
            f"{r['iou_spalling']:^{col[4]}.4f} {r['bf1_crack']:^{col[5]}.4f} "
            f"{r['bf1_spalling']:^{col[6]}.4f}"
        )

    print()
    print(f"Done: {n_ok} evaluated, {n_skip} skipped.")
    print(f"Consolidated results: {args.output}")


if __name__ == "__main__":
    main()
