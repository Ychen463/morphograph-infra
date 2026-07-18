"""Audit dataset splits for leakage and integrity issues.

Usage:
    python scripts/audit_splits.py --manifest data/manifests/all.csv --protocol configs/protocols/lodo.example.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit split integrity")
    parser.add_argument("--manifest", type=Path, required=True, help="Dataset manifest CSV")
    parser.add_argument("--protocol", type=Path, required=True, help="Protocol config YAML")
    args = parser.parse_args()

    # TODO: load manifest + protocol, run audit_split, print report
    raise NotImplementedError("Split auditing not yet implemented")


if __name__ == "__main__":
    main()
