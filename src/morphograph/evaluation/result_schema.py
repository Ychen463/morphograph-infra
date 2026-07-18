"""Experiment result schema with git commit binding.

Every experiment result is stored as a JSON document that includes
the git commit hash, ensuring full traceability from result to code.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class ExperimentResult:
    """Structured result from a single experiment run.

    Attributes:
        experiment_name: name matching the config file.
        git_commit: short SHA of the code version used.
        timestamp: ISO-format timestamp of when the run completed.
        config_path: path to the config file used.
        metrics: dict of metric_name -> value (supports nested per-domain).
        seed: random seed used.
        fold: LODO fold identifier (held-out domain).
        extra: any additional metadata.
    """

    experiment_name: str
    git_commit: str
    timestamp: str
    config_path: str
    metrics: dict[str, Any]
    seed: int
    fold: str
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        experiment_name: str,
        config_path: str,
        metrics: dict[str, Any],
        seed: int,
        fold: str,
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
            metrics=metrics,
            seed=seed,
            fold=fold,
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
