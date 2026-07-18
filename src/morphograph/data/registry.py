"""Dataset adapter registry.

Each raw dataset gets an adapter that converts its native format
into a list of SampleRecords with canonical class mapping.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar

import numpy as np

from .schema import SampleRecord


class DatasetAdapter(ABC):
    """Base class for dataset-specific adapters.

    Subclass this for each new data source. The adapter is responsible for:
    1. Discovering all samples in the raw dataset directory
    2. Remapping original class IDs to the canonical set
    3. Producing a list of SampleRecords
    """

    # Mapping from original class IDs to canonical IDs.
    # Must be defined by each subclass.
    CLASS_MAP: ClassVar[dict[int, int]]

    @abstractmethod
    def discover_samples(self) -> list[SampleRecord]:
        """Scan the raw dataset and return a list of SampleRecords."""
        raise NotImplementedError

    def to_canonical(self, mask: np.ndarray) -> np.ndarray:
        """Remap a raw annotation mask to canonical class IDs.

        Args:
            mask: HxW array with original class IDs.

        Returns:
            HxW array with canonical class IDs (unmapped classes become 255/ignore).
        """
        out = np.full_like(mask, fill_value=255, dtype=np.uint8)
        for src, dst in self.CLASS_MAP.items():
            out[mask == src] = dst
        return out


# Global adapter registry: domain_id -> adapter class
_REGISTRY: dict[str, type[DatasetAdapter]] = {}


def register_adapter(domain_id: str):
    """Decorator to register a dataset adapter for a given domain."""
    def decorator(cls: type[DatasetAdapter]) -> type[DatasetAdapter]:
        _REGISTRY[domain_id] = cls
        return cls
    return decorator


def get_adapter(domain_id: str) -> type[DatasetAdapter]:
    """Retrieve a registered adapter by domain ID."""
    if domain_id not in _REGISTRY:
        raise KeyError(f"No adapter registered for domain '{domain_id}'. "
                       f"Available: {list(_REGISTRY.keys())}")
    return _REGISTRY[domain_id]
