# Experiment Governance

This document defines the rules for experiment execution, protocol freezes, and result traceability. It exists to prevent the test-leakage and post-hoc tuning issues that can invalidate research conclusions.

## Test Domain Evaluation Rules

- Test-domain data must not be used for training, validation, hyperparameter tuning, augmentation policy selection, loss weight tuning, or early stopping.
- Test-domain evaluation is permitted only after all design decisions are finalized and frozen.
- Any change to model architecture, loss function, or training procedure after test-domain evaluation invalidates prior test results. New test evaluation requires a new frozen configuration.

## Validation Domain Selection

- Validation set is drawn exclusively from training domains.
- Validation split is stratified by per-image class frequency.
- Validation fraction and stratification method are fixed in the protocol config and must not change between experiments.

## Foundation Teacher Selection

- If using foundation model features (SAM, DINO, etc.), the teacher model version must be locked before any experiment.
- Teacher selection must not be informed by test-domain performance.
- The selected teacher version is recorded in the experiment config.

## Split Freeze Policy

- Split manifests are frozen before the first model training run.
- Any modification to splits (adding data, changing group assignments, fixing errors) requires:
  - A new manifest version with updated hash
  - Re-running all affected experiments from scratch
  - Documenting the change in RESEARCH_LOG.md

## Failed Experiment Retention

- All experiment results, including failures, are retained and logged.
- Failed experiments are recorded in RESEARCH_LOG.md with failure reason.
- Negative results contribute to understanding and must not be silently discarded.

## Prevention of Test-Driven Method Selection

- The baseline ladder and method list are locked before test evaluation.
- Adding a new method after seeing test results on other methods is not permitted unless:
  - The new method is clearly documented as post-hoc
  - All prior results are reported alongside it
  - The motivation is stated explicitly

## Protocol Freeze Checklist

All items below must be locked before the first test-domain evaluation:

- [ ] Dataset manifest (version + hash)
- [ ] Split manifest (version + hash)
- [ ] Graph label version
- [ ] Canonical class mapping
- [ ] Augmentation policy
- [ ] Loss function and weights
- [ ] Model architecture and head configuration
- [ ] Training hyperparameters (LR, scheduler, epochs, batch size)
- [ ] Checkpoint selection metric
- [ ] Evaluation metrics list

## Required Traceability Fields per Experiment

Every experiment result must record:

| Field | Description |
|-------|-------------|
| `dataset_manifest_hash` | SHA-256 of the dataset manifest file |
| `split_manifest_hash` | SHA-256 of the split assignment file |
| `graph_label_version` | Version tag of the graph label pipeline |
| `config_hash` | SHA-256 of the full experiment config |
| `git_commit` | Git commit hash at training time |
| `seed` | Random seed used |
| `held_out_domain` | Domain held out for testing (LODO fold) |
| `checkpoint_selection_metric` | Metric used to select the best checkpoint |

These fields are enforced by the `ExperimentResult` schema in `src/morphograph/evaluation/result_schema.py`.
