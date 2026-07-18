"""Run Leave-One-Domain-Out cross-validation.

Orchestrates training + evaluation across all LODO folds.

Usage:
    python scripts/run_lodo.py --config configs/experiments/baseline_segformer.example.yaml --seeds 42 43 44
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run LODO cross-validation")
    parser.add_argument("--config", type=Path, required=True, help="Experiment config YAML")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44], help="Random seeds")
    parser.add_argument("--output", type=Path, default=Path("runs"), help="Output directory")
    args = parser.parse_args()

    # TODO: enumerate folds from domain config, launch train+eval for each fold x seed
    raise NotImplementedError("LODO orchestration not yet implemented")


if __name__ == "__main__":
    main()
