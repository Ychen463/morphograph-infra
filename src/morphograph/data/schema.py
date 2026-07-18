"""Canonical sample schema for the MorphoGraph pipeline.

Every sample across all domains is represented as a SampleRecord,
ensuring uniform access patterns regardless of source dataset format.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# Canonical class IDs used throughout the pipeline.
CANONICAL_CLASSES = {
    0: "background",
    1: "crack",
    2: "spalling",
    255: "ignore",
}


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
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.image_path = Path(self.image_path)
        self.mask_path = Path(self.mask_path)
