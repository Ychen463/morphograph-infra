"""Training entry point.

Usage:
    python scripts/train.py --config configs/experiments/baseline_segformer.example.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a model")
    parser.add_argument("--config", type=Path, required=True, help="Experiment config YAML")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    args = parser.parse_args()

    # TODO: load config, build model/data/optimizer, run training loop
    raise NotImplementedError("Training loop not yet implemented")


if __name__ == "__main__":
    main()
