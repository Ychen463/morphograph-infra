"""Experiment result schema with full traceability.

Every experiment result is stored as a JSON document that includes
git commit hash, manifest hashes, and all governance-required fields,
ensuring full traceability from result to code and data.

See docs/EXPERIMENT_GOVERNANCE.md for the complete list of required fields.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


# Fields that must be non-empty for a result to be considered valid.
_REQUIRED_FIELDS = (
    "experiment_name",
    "git_commit",
    "config_hash",
    "dataset_manifest_hash",
    "split_manifest_hash",
    "graph_label_version",
    "fold",
    "checkpoint_selection_metric",
)


def _get_git_commit() -> str:
    """Return the short SHA of HEAD, or 'unknown' if not in a git repo."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


@dataclass
class ExperimentResult:
    """Structured result from a single experiment run."""

    experiment_name: str
    git_commit: str
    timestamp: str
    config_path: str
    config_hash: str
    dataset_manifest_hash: str
    split_manifest_hash: str
    graph_label_version: str
    metrics: dict[str, Any]
    seed: int
    fold: str
    checkpoint_selection_metric: str
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        experiment_name: str,
        config_path: str,
        config_hash: str,
        dataset_manifest_hash: str,
        split_manifest_hash: str,
        graph_label_version: str,
        metrics: dict[str, Any],
        seed: int,
        fold: str,
        checkpoint_selection_metric: str = "val_mIoU",
        **extra: Any,
    ) -> ExperimentResult:
        """Factory method that auto-fills git commit and timestamp."""
        return cls(
            experiment_name=experiment_name,
            git_commit=_get_git_commit(),
            timestamp=datetime.now().isoformat(),
            config_path=config_path,
            config_hash=config_hash,
            dataset_manifest_hash=dataset_manifest_hash,
            split_manifest_hash=split_manifest_hash,
            graph_label_version=graph_label_version,
            metrics=metrics,
            seed=seed,
            fold=fold,
            checkpoint_selection_metric=checkpoint_selection_metric,
            extra=extra,
        )

    @classmethod
    def load(cls, path: Path) -> ExperimentResult:
        """Load a result from a JSON file."""
        data = json.loads(Path(path).read_text())
        return cls(**data)

    def validate(self) -> list[str]:
        """Check that all governance-required fields are populated.

        Returns:
            List of violation messages (empty if valid).
        """
        violations = []
        for fname in _REQUIRED_FIELDS:
            val = getattr(self, fname)
            if not val or val == "unknown":
                violations.append(f"Field '{fname}' is missing or unknown")
        if not self.metrics:
            violations.append("No metrics recorded")
        return violations

    def save(self, output_dir: Path) -> Path:
        """Save result as JSON to output_dir."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        filename = f"{self.experiment_name}_s{self.seed}_{self.fold}.json"
        path = output_dir / filename
        path.write_text(json.dumps(asdict(self), indent=2))
        return path
