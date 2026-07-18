"""Tests for ExperimentResult schema."""

import json
from pathlib import Path

from morphograph.evaluation.result_schema import ExperimentResult


def _make_result(**overrides) -> ExperimentResult:
    defaults = dict(
        experiment_name="B0_test",
        git_commit="abc1234",
        timestamp="2026-01-01T00:00:00",
        config_path="configs/test.yaml",
        config_hash="sha256_config",
        dataset_manifest_hash="sha256_data",
        split_manifest_hash="sha256_split",
        graph_label_version="v1.0",
        metrics={"mIoU": 0.75},
        seed=42,
        fold="site_a",
        checkpoint_selection_metric="val_mIoU",
    )
    defaults.update(overrides)
    return ExperimentResult(**defaults)


def test_save_and_load(tmp_path: Path):
    """Round-trip save and load should preserve all fields."""
    result = _make_result()
    saved = result.save(tmp_path)
    loaded = ExperimentResult.load(saved)
    assert loaded.experiment_name == result.experiment_name
    assert loaded.metrics == result.metrics
    assert loaded.seed == result.seed
    assert loaded.fold == result.fold


def test_validate_passes():
    """A fully populated result should pass validation."""
    result = _make_result()
    assert result.validate() == []


def test_validate_catches_missing_fields():
    """Empty required fields should produce violations."""
    result = _make_result(config_hash="", git_commit="unknown")
    violations = result.validate()
    assert len(violations) >= 2


def test_validate_catches_empty_metrics():
    """Empty metrics dict should be flagged."""
    result = _make_result(metrics={})
    violations = result.validate()
    assert any("metrics" in v for v in violations)


def test_save_json_format(tmp_path: Path):
    """Saved JSON should be valid and contain all traceability fields."""
    result = _make_result()
    path = result.save(tmp_path)
    data = json.loads(path.read_text())
    assert data["dataset_manifest_hash"] == "sha256_data"
    assert data["split_manifest_hash"] == "sha256_split"
    assert data["graph_label_version"] == "v1.0"
    assert data["checkpoint_selection_metric"] == "val_mIoU"
