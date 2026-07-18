"""Split protocol auditing: leakage detection and integrity checks.

Ensures that evaluation protocols are correctly implemented
and no data leaks across train/val/test boundaries.
"""

from __future__ import annotations

from typing import Any

from ..data.schema import SampleRecord


def check_split_leakage(records: list[SampleRecord]) -> list[str]:
    """Verify that no domain/group appears in both train and test splits.

    Args:
        records: list of SampleRecords with split assignments.

    Returns:
        List of violation messages (empty if clean).
    """
    split_groups: dict[str, set[str]] = {}
    for r in records:
        if r.split is None:
            continue
        split_groups.setdefault(r.split, set()).add(r.group_id)

    violations = []
    train_groups = split_groups.get("train", set())
    test_groups = split_groups.get("test", set())
    leaked = train_groups & test_groups
    if leaked:
        violations.append(
            f"Group leakage: groups {leaked} appear in both train and test"
        )
    return violations


def check_val_domain_leakage(records: list[SampleRecord]) -> list[str]:
    """Verify that validation samples come only from training domains.

    Per BENCHMARK_PROTOCOL.md: validation set is drawn only from
    training domains (never from the test domain).

    Args:
        records: list of SampleRecords with split assignments.

    Returns:
        List of violation messages (empty if clean).
    """
    split_groups: dict[str, set[str]] = {}
    for r in records:
        if r.split is None:
            continue
        split_groups.setdefault(r.split, set()).add(r.group_id)

    violations = []
    val_groups = split_groups.get("val", set())
    train_groups = split_groups.get("train", set())
    test_groups = split_groups.get("test", set())

    val_not_in_train = val_groups - train_groups
    if val_not_in_train:
        violations.append(
            f"Validation groups {val_not_in_train} are not in training domains"
        )

    val_in_test = val_groups & test_groups
    if val_in_test:
        violations.append(
            f"Validation groups {val_in_test} overlap with test domain"
        )

    return violations


def check_near_duplicates(
    records: list[SampleRecord],
    phash_threshold: int = 8,
) -> list[tuple[str, str, int]]:
    """Detect near-duplicate images across splits using perceptual hashing.

    Args:
        records: list of SampleRecords with split assignments.
        phash_threshold: maximum Hamming distance to consider as near-duplicate.

    Returns:
        List of (sample_id_a, sample_id_b, distance) tuples for flagged pairs.
    """
    # TODO: implement perceptual hashing and cross-split comparison
    raise NotImplementedError("Near-duplicate detection not yet implemented")


def audit_split(
    records: list[SampleRecord],
    phash_threshold: int = 8,
) -> dict[str, Any]:
    """Run all split integrity checks.

    Args:
        records: list of SampleRecords with split assignments.
        phash_threshold: threshold for near-duplicate detection.

    Returns:
        Audit report dict with keys: leakage_violations,
        val_domain_violations, duplicate_check_status, passed.
    """
    leakage = check_split_leakage(records)
    val_leakage = check_val_domain_leakage(records)

    all_violations = leakage + val_leakage
    return {
        "leakage_violations": leakage,
        "val_domain_violations": val_leakage,
        "duplicate_check_status": "skipped",
        "passed": len(all_violations) == 0,
    }
