"""Prepare a raw dataset into the canonical manifest format.

Usage:
    python scripts/prepare_dataset.py --domains configs/data/domains.example.yaml --output data/manifests/
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build canonical dataset manifest")
    parser.add_argument("--domains", type=Path, required=True, help="Domain config YAML")
    parser.add_argument("--output", type=Path, required=True, help="Output manifest directory")
    args = parser.parse_args()

    # TODO: load domain config, instantiate adapters, discover samples, write manifest
    raise NotImplementedError("Dataset preparation not yet implemented")


if __name__ == "__main__":
    main()
