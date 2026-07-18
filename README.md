# MorphoGraph-Infra

Cross-Site Morphology-Graph Learning for Infrastructure Defect Understanding.

## Overview

This project develops a unified morphological graph representation for concrete defects (cracks, spalling) that generalizes across inspection sites and sensors. It combines semantic segmentation with automatically derived graph targets (skeleton, endpoints, junctions, width profiles) to capture structural properties that pixel-level masks alone cannot express.

## Installation

```bash
pip install -e ".[dev]"
```

## Repository Structure

```
configs/          # YAML configurations for domains, experiments, and protocols
data/             # Raw data, manifests, and derived graph targets
docs/             # Project charter, benchmark protocol, data contract, roadmap
scripts/          # Entry points: data prep, training, evaluation, LODO CV
src/morphograph/  # Core library: data, models, evaluation, losses, metrics, training
tests/            # Unit tests
```

## Documentation

- [Project Charter](docs/PROJECT_CHARTER.md) — research question, scope, non-goals
- [Benchmark Protocol](docs/BENCHMARK_PROTOCOL.md) — grouped splits, LODO rules, baseline ladder
- [Data Contract](docs/DATA_CONTRACT.md) — sample schema, canonical classes, ignore policy
- [Experiment Roadmap](docs/EXPERIMENT_ROADMAP.md) — phased plan from infrastructure to analysis

## Quick Start

```bash
# Prepare dataset manifest
python scripts/prepare_dataset.py --domains configs/data/domains.example.yaml --output data/manifests/

# Build graph labels from crack masks
python scripts/build_graph_labels.py --manifest data/manifests/all.csv --output data/derived/

# Run LODO cross-validation
python scripts/run_lodo.py --config configs/experiments/baseline_segformer.example.yaml --seeds 42 43 44
```
