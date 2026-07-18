"""Canonical sample schema for the MorphoGraph pipeline.

Every sample across all domains is represented as a SampleRecord,
ensuring uniform access patterns regardless of source dataset format.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np

# Canonical class IDs used throughout the pipeline.
CANONICAL_CLASSES = {
    0: "background",
    1: "crack",
    2: "spalling",
    255: "ignore",
}

# Empirical class pixel frequencies from DamSegment (1500 images, 640x640).
DAMSEGMENT_CLASS_FREQ = {
    "background": 0.966,
    "crack": 0.022,
    "spalling": 0.012,
}

# Inverse-frequency CE weights (clipped), derived from class frequencies.
DEFAULT_CE_WEIGHTS = [0.2, 2.0, 3.0]

# Number of semantic classes (excluding ignore).
NUM_CLASSES = 3


def decode_rgb_mask(rgb: np.ndarray) -> np.ndarray:
    """Decode an RGB-encoded annotation mask to canonical class IDs.

    DamSegment convention:
        - Red channel > 127  -> crack (class 1)
        - Blue channel > 127 -> spalling (class 2)
        - When both R and B are active on the same pixel, spalling
          overwrites crack. This is a priority rule in the original
          annotation, not a true spatial overlap. Three-class masks
          are mutually exclusive after decoding.
        - Everything else    -> background (class 0)

    Args:
        rgb: HxWx3 uint8 array.

    Returns:
        HxW uint8 array with canonical class IDs.
    """
    out = np.zeros(rgb.shape[:2], dtype=np.uint8)
    out[rgb[..., 0] > 127] = 1  # crack
    out[rgb[..., 2] > 127] = 2  # spalling overwrites crack
    return out


@dataclass
class SampleRecord:
    """A single sample in the canonical dataset manifest.

    Every adapter must produce SampleRecords with at least
    sample_id, domain_id, and group_id populated.
    """

    sample_id: str
    domain_id: str
    group_id: str
    image_path: Path
    mask_path: Path
    split: Optional[str] = None  # assigned by protocol: train / val / test
    difficulty: Optional[str] = None  # easy / medium / hard (DamSegment tier)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.image_path = Path(self.image_path)
        self.mask_path = Path(self.mask_path)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a flat dict suitable for CSV/JSON output."""
        d = {
            "sample_id": self.sample_id,
            "domain_id": self.domain_id,
            "group_id": self.group_id,
            "image_path": str(self.image_path),
            "mask_path": str(self.mask_path),
            "split": self.split or "",
            "difficulty": self.difficulty or "",
        }
        for k, v in self.metadata.items():
            d[f"meta_{k}"] = v
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> SampleRecord:
        """Deserialize from a flat dict (as read from CSV/JSON)."""
        metadata = {}
        for k, v in d.items():
            if k.startswith("meta_"):
                metadata[k[5:]] = v
        return cls(
            sample_id=d["sample_id"],
            domain_id=d["domain_id"],
            group_id=d["group_id"],
            image_path=Path(d["image_path"]),
            mask_path=Path(d["mask_path"]),
            split=d.get("split") or None,
            difficulty=d.get("difficulty") or None,
            metadata=metadata,
        )


def save_manifest(records: list[SampleRecord], path: Path) -> None:
    """Write a list of SampleRecords to a CSV manifest."""
    if not records:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [r.to_dict() for r in records]
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_manifest(path: Path) -> list[SampleRecord]:
    """Read a CSV manifest into a list of SampleRecords."""
    records = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append(SampleRecord.from_dict(row))
    return records
