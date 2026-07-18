"""Evaluate a trained model checkpoint.

Usage:
    python scripts/evaluate.py --config configs/experiments/baseline_segformer.example.yaml --checkpoint runs/B0/best.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a model checkpoint")
    parser.add_argument("--config", type=Path, required=True, help="Experiment config YAML")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Model checkpoint path")
    parser.add_argument("--output", type=Path, default=None, help="Output directory for results JSON")
    args = parser.parse_args()

    # TODO: load config + checkpoint, run evaluation, save ExperimentResult
    raise NotImplementedError("Evaluation not yet implemented")


if __name__ == "__main__":
    main()
