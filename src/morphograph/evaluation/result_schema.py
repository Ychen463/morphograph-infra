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
from typing import Any, Optional


@dataclass
class ExperimentResult:
    """Structured result from a single experiment run.

    Attributes:
        experiment_name: name matching the config file.
        git_commit: short SHA of the code version used.
        timestamp: ISO-format timestamp of when the run completed.
        config_path: path to the config file used.
        config_hash: SHA-256 of the full experiment config.
        dataset_manifest_hash: SHA-256 of the dataset manifest file.
        split_manifest_hash: SHA-256 of the split assignment file.
        graph_label_version: version tag of the graph label pipeline.
        metrics: dict of metric_name -> value (supports nested per-domain).
        seed: random seed used.
        fold: LODO fold identifier (held-out domain).
        checkpoint_selection_metric: metric used to select the best checkpoint.
        extra: any additional metadata.
    """

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
        try:
            commit = subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                text=True,
            ).strip()
        except subprocess.CalledProcessError:
            commit = "unknown"

        return cls(
            experiment_name=experiment_name,
            git_commit=commit,
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

    def save(self, output_dir: Path) -> Path:
        """Save result as JSON to output_dir."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        filename = f"{self.experiment_name}_s{self.seed}_{self.fold}.json"
        path = output_dir / filename
        path.write_text(json.dumps(asdict(self), indent=2))
        return path
