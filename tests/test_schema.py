"""Tests for SampleRecord schema and manifest I/O."""

from pathlib import Path

import numpy as np

from morphograph.data.schema import (
    SampleRecord,
    decode_rgb_mask,
    load_manifest,
    save_manifest,
)


def _make_records() -> list[SampleRecord]:
    return [
        SampleRecord(
            sample_id="s1",
            domain_id="site_a",
            group_id="A",
            image_path=Path("/data/s1.png"),
            mask_path=Path("/data/s1_mask.png"),
            split="train",
            metadata={"resolution_mm_per_px": 0.5},
        ),
        SampleRecord(
            sample_id="s2",
            domain_id="site_b",
            group_id="B",
            image_path=Path("/data/s2.png"),
            mask_path=Path("/data/s2_mask.png"),
            split="test",
        ),
    ]


def test_to_dict_and_from_dict():
    """Round-trip through dict should preserve fields."""
    record = _make_records()[0]
    d = record.to_dict()
    restored = SampleRecord.from_dict(d)
    assert restored.sample_id == record.sample_id
    assert restored.domain_id == record.domain_id
    assert restored.split == record.split
    assert restored.metadata["resolution_mm_per_px"] == 0.5


def test_manifest_round_trip(tmp_path: Path):
    """Save and load manifest should preserve all records."""
    records = _make_records()
    path = tmp_path / "manifest.csv"
    save_manifest(records, path)
    loaded = load_manifest(path)
    assert len(loaded) == 2
    assert loaded[0].sample_id == "s1"
    assert loaded[1].sample_id == "s2"
    assert loaded[0].split == "train"
    assert loaded[1].split == "test"


def test_empty_split_round_trip():
    """Record with no split should round-trip as None."""
    record = SampleRecord(
        sample_id="s3",
        domain_id="site_c",
        group_id="C",
        image_path=Path("/data/s3.png"),
        mask_path=Path("/data/s3_mask.png"),
    )
    d = record.to_dict()
    restored = SampleRecord.from_dict(d)
    assert restored.split is None


def test_decode_rgb_mask_crack_only():
    """Red channel should map to crack (class 1)."""
    rgb = np.zeros((4, 4, 3), dtype=np.uint8)
    rgb[1, 1, 0] = 200  # red -> crack
    mask = decode_rgb_mask(rgb)
    assert mask[1, 1] == 1
    assert mask[0, 0] == 0


def test_decode_rgb_mask_spalling_overwrites():
    """When both R and B are active, spalling should win."""
    rgb = np.zeros((4, 4, 3), dtype=np.uint8)
    rgb[2, 2, 0] = 200  # red
    rgb[2, 2, 2] = 200  # blue
    mask = decode_rgb_mask(rgb)
    assert mask[2, 2] == 2  # spalling overwrites crack


def test_decode_rgb_mask_mutually_exclusive():
    """After decoding, no pixel should have overlapping classes."""
    rgb = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
    mask = decode_rgb_mask(rgb)
    # Each pixel has exactly one class
    for val in np.unique(mask):
        assert val in {0, 1, 2}
