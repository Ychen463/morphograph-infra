"""Tests for split protocol auditing."""

from pathlib import Path

from morphograph.data.schema import SampleRecord
from morphograph.evaluation.protocol_audit import audit_split, check_split_leakage


def _make_record(sample_id: str, group_id: str, split: str) -> SampleRecord:
    return SampleRecord(
        sample_id=sample_id,
        domain_id=f"domain_{group_id}",
        group_id=group_id,
        image_path=Path(f"/fake/{sample_id}.png"),
        mask_path=Path(f"/fake/{sample_id}_mask.png"),
        split=split,
    )


def test_no_leakage_clean_split():
    """Clean split with no group overlap should pass."""
    records = [
        _make_record("s1", "A", "train"),
        _make_record("s2", "A", "train"),
        _make_record("s3", "B", "test"),
        _make_record("s4", "B", "test"),
    ]
    violations = check_split_leakage(records)
    assert len(violations) == 0


def test_leakage_detected():
    """Same group in train and test should be flagged."""
    records = [
        _make_record("s1", "A", "train"),
        _make_record("s2", "A", "test"),
    ]
    violations = check_split_leakage(records)
    assert len(violations) > 0
    assert "A" in violations[0]


def test_audit_report_passes_on_clean():
    """Full audit should report passed=True on clean split."""
    records = [
        _make_record("s1", "A", "train"),
        _make_record("s2", "B", "test"),
    ]
    report = audit_split(records)
    assert report["passed"] is True
