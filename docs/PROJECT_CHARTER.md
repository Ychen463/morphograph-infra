# Project Charter: Cross-Site Morphology-Graph Learning for Infrastructure Defect Understanding

## Research Question

Can we learn a unified morphological graph representation of concrete defects (cracks, spalling) that generalizes across inspection sites, sensors, and environmental conditions — without requiring graph-level annotations at training time — and leverage this representation for cross-domain generalization?

## Scope

- Multi-site concrete dam inspection imagery (UAV + handheld)
- Defect types: cracks (linear), spalling (areal), and their spatial relationships
- Joint prediction of segmentation masks, crack morphological graphs, spalling regions, and crack-spalling spatial relations
- Morphological graph: skeleton-based representation capturing endpoints, junctions, width profiles, and connectivity
- Cross-site generalization via morphology-invariant representations under Leave-One-Domain-Out (LODO) evaluation

## Core Hypotheses

### H1: Graph supervision improves structural quality

At comparable mIoU, a graph-supervised model should significantly improve:
- Endpoint F1, junction F1, edge F1
- Path recall and graph edit distance
- Crack length error

### H2: Explicit graph reconstruction outperforms topology loss alone

Full node/edge supervision should outperform clDice or SRL used in isolation.

### H3: Morphology representation is more domain-stable

Compared to mask-only features, morphology embeddings on held-out domains should exhibit:
- Lower performance drop
- Lower domain variance
- More stable model rankings
- Better worst-domain performance

### H4: Foundation features require selective transfer

Direct LoRA or output KD is not necessarily stable; graph-conditioned selective feature transfer should yield better cross-domain graph metrics.

### H5: Region-graph joint modeling adds value

Adding spalling region and crack-spalling relation supervision should reduce:
- Crack-spalling boundary confusion
- Spalling component error
- Co-occurring defect misses

## Quantitative Success Criteria

### Representation success (vs B0 or best topology-loss baseline)

- Reproducible improvement in endpoint/junction/edge metrics
- Significant improvement in path recall or graph edit distance
- Reduced crack length error
- No significant mIoU degradation

### Cross-domain success (vs D1 or strongest standard DG baseline)

- Improved worst-domain graph/topology performance
- No significant average-domain performance degradation
- Reduced domain variance
- Consistent improvement direction in at least two held-out domains
- Multi-seed results not dependent on a single lucky run

Formulation: Proposed DG is considered successful only if it improves worst-domain graph or engineering performance in at least two held-out domains without reducing average mIoU by more than one percentage point. The threshold is adjustable after baseline variance is established.

## Non-Goals (First Paper)

- Real-time inference optimization (focus is on representation quality)
- Non-concrete infrastructure (bridges, roads) — future work
- 3D reconstruction or depth estimation
- Temporal defect progression tracking (requires multi-timepoint data)
- Formal severity grading (requires expert severity labels)
- Millimetre-level width measurement without camera calibration
- Replacing human inspectors — this is a decision-support tool
