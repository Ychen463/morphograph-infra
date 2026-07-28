"""Node/edge count statistics across the training set.

Run on RunPod before P3 implementation to set max_nodes, edge candidate
params, and loop-anchor decision.

Usage:
    python scripts/stats_graph_nodes.py --data-root data/raw --output runs/graph_stats.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from morphograph.data.schema import decode_rgb_mask
from morphograph.data.graph_targets import mask_to_graph
from morphograph.training.utils import discover_all_samples


def get_tier(img_path: Path) -> str:
    """Extract difficulty tier from path."""
    parts = img_path.parts
    for p in parts:
        if p in ("Easy", "Medium", "Hard"):
            return p
    return "Unknown"


def main() -> None:
    parser = argparse.ArgumentParser(description="Graph node/edge statistics")
    parser.add_argument("--data-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--output", type=Path, default=Path("runs/graph_stats.json"))
    parser.add_argument("--img-size", type=int, default=512)
    args = parser.parse_args()

    all_pairs = discover_all_samples(args.data_root)
    if not all_pairs:
        print("ERROR: No data found.")
        sys.exit(1)

    print(f"Processing {len(all_pairs)} images...")

    stats = defaultdict(list)
    tier_stats = defaultdict(lambda: defaultdict(list))
    zero_node_count = 0
    closed_loop_components = 0
    total_components = 0

    for i, (img_path, mask_path) in enumerate(all_pairs):
        mask_raw = np.array(
            Image.open(mask_path).resize(
                (args.img_size, args.img_size), Image.NEAREST
            )
        )
        if mask_raw.ndim == 3:
            mask = decode_rgb_mask(mask_raw)
        else:
            mask = mask_raw.astype(np.uint8)

        crack_binary = (mask == 1).astype(np.uint8)
        if not crack_binary.any():
            stats["num_endpoints"].append(0)
            stats["num_junctions"].append(0)
            stats["num_nodes"].append(0)
            stats["num_edges"].append(0)
            zero_node_count += 1
            continue

        graph = mask_to_graph(crack_binary, min_branch_length=10, junction_merge_radius=5)

        n_ep = len(graph.endpoints)
        n_jn = len(graph.junctions)
        n_nodes = graph.num_nodes
        n_edges = graph.num_edges

        stats["num_endpoints"].append(n_ep)
        stats["num_junctions"].append(n_jn)
        stats["num_nodes"].append(n_nodes)
        stats["num_edges"].append(n_edges)

        tier = get_tier(img_path)
        tier_stats[tier]["num_nodes"].append(n_nodes)
        tier_stats[tier]["num_edges"].append(n_edges)

        # Analyze components for closed loops
        if n_nodes > 0:
            # Build adjacency
            adj = defaultdict(set)
            for a, b in graph.edges:
                adj[a].add(b)
                adj[b].add(a)

            # Find connected components
            visited = set()
            for node_idx in range(n_nodes):
                if node_idx in visited:
                    continue
                component = set()
                queue = [node_idx]
                while queue:
                    n = queue.pop()
                    if n in visited:
                        continue
                    visited.add(n)
                    component.add(n)
                    for nb in adj.get(n, set()):
                        if nb not in visited:
                            queue.append(nb)

                total_components += 1
                # Check if component has no endpoints (closed loop)
                n_ep_count = len(graph.endpoints)
                has_endpoint = any(idx < n_ep_count for idx in component)
                if not has_endpoint and len(component) > 0:
                    closed_loop_components += 1

        if (i + 1) % 100 == 0:
            print(f"  [{i + 1}/{len(all_pairs)}]")

    # Compute percentiles
    def percentiles(vals):
        arr = np.array(vals)
        return {
            "min": int(arr.min()),
            "p50": int(np.percentile(arr, 50)),
            "p75": int(np.percentile(arr, 75)),
            "p90": int(np.percentile(arr, 90)),
            "p95": int(np.percentile(arr, 95)),
            "p99": int(np.percentile(arr, 99)),
            "max": int(arr.max()),
            "mean": float(arr.mean()),
            "std": float(arr.std()),
        }

    result = {
        "total_images": len(all_pairs),
        "zero_node_images": zero_node_count,
        "zero_node_fraction": zero_node_count / len(all_pairs),
        "overall": {k: percentiles(v) for k, v in stats.items()},
        "per_tier": {
            tier: {k: percentiles(v) for k, v in tier_data.items()}
            for tier, tier_data in tier_stats.items()
        },
        "closed_loop_components": closed_loop_components,
        "total_components": total_components,
        "closed_loop_fraction": (
            closed_loop_components / max(total_components, 1)
        ),
        "recommended_max_nodes": int(np.percentile(stats["num_nodes"], 99)),
    }

    # Print summary
    print(f"\n{'=' * 60}")
    print("GRAPH NODE/EDGE STATISTICS")
    print(f"{'=' * 60}")
    print(f"Total images: {result['total_images']}")
    print(f"Zero-node images: {result['zero_node_images']} ({result['zero_node_fraction']:.1%})")
    print(f"\nOverall:")
    for k in ["num_endpoints", "num_junctions", "num_nodes", "num_edges"]:
        p = result["overall"][k]
        print(f"  {k:20s}: p50={p['p50']:3d}  p75={p['p75']:3d}  "
              f"p90={p['p90']:3d}  p95={p['p95']:3d}  p99={p['p99']:3d}  max={p['max']:3d}")
    print(f"\nPer tier:")
    for tier in sorted(result["per_tier"]):
        print(f"  {tier}:")
        for k in ["num_nodes", "num_edges"]:
            p = result["per_tier"][tier][k]
            print(f"    {k:20s}: p50={p['p50']:3d}  p90={p['p90']:3d}  p99={p['p99']:3d}  max={p['max']:3d}")
    print(f"\nClosed-loop components: {closed_loop_components}/{total_components} "
          f"({result['closed_loop_fraction']:.1%})")
    print(f"\nRecommended max_nodes (p99): {result['recommended_max_nodes']}")

    # Save
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
