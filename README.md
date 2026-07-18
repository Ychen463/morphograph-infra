# MorphoGraph-Infra

Cross-Site Morphology-Graph Learning for Infrastructure Defect Understanding.

## Overview

This project develops a unified morphological graph representation for concrete defects (cracks, spalling) that generalizes across inspection sites and sensors. It combines semantic segmentation with joint prediction of crack morphological graphs, spalling regions, and their spatial relations. Graph targets (skeleton, endpoints, junctions, width profiles, connectivity) are derived automatically from segmentation masks and capture structural properties that pixel-level masks alone cannot express. The learned morphology-invariant representations are evaluated for cross-domain generalization under a strict Leave-One-Domain-Out (LODO) protocol.

## Installation

```bash
pip install -e ".[dev]"
```

## Repository Structure

```
configs/          # YAML configurations for domains, experiments, and protocols
data/             # Raw data, manifests, and derived graph targets
docs/             # Project charter, benchmark protocol, data contract, roadmap, governance
scripts/          # Entry points: data prep, training, evaluation, LODO CV
src/morphograph/  # Core library: data, models, evaluation, losses, metrics, training
tests/            # Unit tests
```

## Documentation

- [Project Charter](docs/PROJECT_CHARTER.md) — research question, hypotheses, scope, non-goals
- [Benchmark Protocol](docs/BENCHMARK_PROTOCOL.md) — grouped splits, LODO rules, baseline ladder
- [Data Contract](docs/DATA_CONTRACT.md) — sample schema, canonical classes, graph label quality
- [Experiment Roadmap](docs/EXPERIMENT_ROADMAP.md) — phased plan P0-P6 with go/no-go gates
- [Experiment Governance](docs/EXPERIMENT_GOVERNANCE.md) — protocol freeze, traceability, test-domain rules
- [Decisions](docs/DECISIONS.md) — locked research decisions with rationale
- [Research Log](docs/RESEARCH_LOG.md) — experiment phases, results, next steps

## Quick Start

```bash
# Prepare dataset manifest
python scripts/prepare_dataset.py --domains configs/data/domains.example.yaml --output data/manifests/

# Build graph labels from crack masks
python scripts/build_graph_labels.py --manifest data/manifests/all.csv --output data/derived/

# Run LODO cross-validation
python scripts/run_lodo.py --config configs/experiments/baseline_segformer.example.yaml --seeds 42 43 44
```
