"""Build graph-level labels from segmentation masks.

Runs the mask -> skeleton -> graph pipeline on all crack masks
and saves derived targets to data/derived/.

Usage:
    python scripts/build_graph_labels.py --manifest data/manifests/all.csv --output data/derived/
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build graph labels from crack masks")
    parser.add_argument("--manifest", type=Path, required=True, help="Dataset manifest CSV")
    parser.add_argument("--output", type=Path, required=True, help="Output directory for derived targets")
    parser.add_argument("--min-branch-length", type=int, default=10, help="Min skeleton branch length")
    args = parser.parse_args()

    # TODO: load manifest, iterate samples, run mask_to_graph pipeline, save results
    raise NotImplementedError("Graph label building not yet implemented")


if __name__ == "__main__":
    main()
