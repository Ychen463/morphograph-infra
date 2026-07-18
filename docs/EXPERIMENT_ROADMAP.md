# Experiment Roadmap

## Progressive Mainline (P0-P6)

### P0: Benchmark, Manifests, Split Audit, Experiment Governance

- Set up repository structure, CI, and reproducibility guardrails
- Implement dataset adapters and canonical manifest builder
- Implement grouped split + leakage audit (perceptual hash dedup)
- Establish experiment governance protocol (see EXPERIMENT_GOVERNANCE.md)
- Freeze split manifests before any model training

### P1: Mask-to-Graph/Region Conversion + Graph Gold Subsets

- Build crack mask-to-graph conversion pipeline (skeleton, endpoints, junctions, edges, width)
- Establish Graph-QC Development Set (50-100 images) for tuning conversion parameters
- Establish locked Graph Gold Test Set (100-200 images) for evaluation only
- Implement auto-label quality metrics (endpoint P/R, junction P/R, CC agreement, path preservation, edge coverage, width MAE, false-spur rate)

**P1 Gate**: Auto-generated graph targets must pass quality assessment on gold test set before use in model training. If quality is insufficient, pause model development and improve labels/annotation protocol.

### P2: Representation Baseline Ladder

Run under LODO protocol, 3 seeds each:

| ID | Config | Validates |
|----|--------|-----------|
| B0 | Mask-only SegFormer-B2 (CE+Dice) | Strong pixel baseline |
| B1a | B0 + clDice | Topology loss (primary) |
| B1b | B0 + SRL | Topology loss (supplementary) |
| B2 | B0 + skeleton head | Dense morphology supervision |
| B3 | B0 + endpoint/junction heads | Node supervision |
| B4 | B0 + node + edge supervision | Graph supervision |

Each added head requires a parameter-matched control without graph supervision to separate capacity gain from supervision signal.

**P2 Gate**: If B2-B4 only add parameters but do not improve graph/topology metrics, do not continue stacking heads. Redesign supervision strategy or graph representation.

### P3: Direct Joint Reconstruction

| ID | Config | Validates |
|----|--------|-----------|
| B5 | Joint mask + graph + width | Complete crack graph |
| B6 | B5 + spalling region head | Joint region-graph |

**P3 Gate**: If direct graph reconstruction does not outperform auxiliary supervision, reassess graph decoder architecture before proceeding to DG.

### P4: Graph-Invariant Domain Generalization

Cross-domain ladder:

| ID | Config |
|----|--------|
| D0 | B0, standard ERM |
| D1 | Best region-graph model, standard ERM |
| D2a | CORAL (feature-distribution alignment) |
| D2b | DANN (domain-adversarial alignment) |
| D2c | MixStyle (appearance-statistics perturbation) |
| D3 | Appearance augmentation + graph consistency |
| D4 | Appearance-morphology disentanglement |
| D5 | Graph prototype alignment |
| D6 | Selective foundation-feature transfer |
| D7 | Full Graph-Invariant DG |

Oracle baselines (separate, require known domain ID):

| ID | Config |
|----|--------|
| O0 | Domain-specific BN oracle |
| O1 | Known-domain token oracle |

**P4 Gate**: If morphology representation does not improve worst-domain performance, do not claim graph-invariant generalization. Analyze domain gap sources and label compatibility.

### P5: Crack-Spalling Relations + Engineering Evaluation

| ID | Config |
|----|--------|
| B7 | B6 + relation head (crack-spalling spatial relations) |

B7 belongs to P5 and is not required for the P3 model.

Engineering quantities:
- Always reportable: pixel length, normalized length, pixel width, relative area, component count, longest-path error
- Only with reliable calibration: mm width, physical crack length, mm^2/m^2 area

### P6: Future Work

- Severity grading (requires expert severity labels)
- Temporal defect progression (requires multi-timepoint aligned data)
- Multi-view / 3D reconstruction
- Physically calibrated measurement (requires camera calibration, GSD metadata)

## 12-Step Implementation Sequence

1. Complete all dataset manifests
2. Implement grouped split + leakage audit
3. Implement mask-to-graph conversion
4. Establish human-reviewed graph gold subsets (QC dev + locked test)
5. Implement graph/region/engineering metrics
6. Reproduce mask-only baseline (B0)
7. Run B1a/B1b-B4, verify graph supervision value
8. Implement joint region-graph model (B5-B6)
9. Complete grouped in-domain evaluation
10. Add graph-invariant DG methods (D0-D7)
11. Run LODO, multi-seed, multi-split realizations
12. Add crack-spalling relations (B7) + engineering quantity analysis

## First Paper Scope: MorphoGraph-DG

**Includes**:
- Automatic mask-to-graph/region supervision
- Joint crack-graph and spalling-region prediction
- Morphology-invariant cross-domain learning
- Grouped and leave-one-domain-out benchmark
- Segmentation, graph, topology, and engineering quantity evaluation

**Excludes**:
- Temporal progression
- 3D reconstruction
- Formal severity grading
- Millimetre-level measurement without calibration
