# Research Decisions

Locked decisions with rationale. Once a decision is recorded here, it governs all subsequent implementation and documentation. Changes require explicit discussion, a new entry, and deprecation of the old one.

## D001: Graph is prediction target, not just auxiliary loss

**Decision**: The ultimate goal is joint prediction of mask, crack graph, spalling region, and their spatial relations. Graph supervision is the starting point (P2), but the model must progress to explicit structured prediction (P3).

**Rationale**: Framing graph only as auxiliary loss limits the paper claim to "graph loss improves mIoU by X". The stronger claim — and the one that justifies the full architecture — is that the model produces a usable graph representation for downstream engineering tasks.

## D002: DSBN and domain tokens are oracle baselines

**Decision**: Domain-specific BatchNorm (O0) and known-domain tokens (O1) are classified as oracle baselines, not domain generalization methods. They are reported separately from the D0-D7 DG ladder.

**Rationale**: In LODO, the held-out domain has no training data, so there are no corresponding BN statistics or domain embeddings. Methods requiring known domain identity at inference are domain-aware adaptation, not domain generalization.

## D003: Crack-spalling relations are spatial, not causal

**Decision**: The relation head learns spatial and morphological relations (co-occurrence, adjacency, intersection, containment). The paper must not claim causal relations.

**Rationale**: Single-timepoint 2D images cannot establish that cracks caused spalling. Physical causality requires temporal evidence or 3D structural analysis.

## D004: First paper excludes temporal, 3D, severity, uncalibrated mm measurement

**Decision**: These are listed as future work in PROJECT_CHARTER.md and P6 in EXPERIMENT_ROADMAP.md.

**Rationale**: Each requires data or annotations not currently available. Including them would spread the paper too thin and weaken each contribution.

## D005: clDice and SRL are separate experiments

**Decision**: B1a (clDice) and B1b (SRL) are distinct baseline configurations. They must not be combined into a single ambiguous "clDice/SRL" entry.

**Rationale**: clDice and SRL are different supervision mechanisms with potentially different outcomes. Merging them creates an unreproducible "pick one" configuration.

## D006: DG baselines locked to CORAL, DANN, MixStyle

**Decision**: The first round of standard DG baselines is fixed to these three methods, covering feature-distribution alignment, domain-adversarial alignment, and appearance-statistics perturbation respectively.

**Rationale**: An open-ended baseline list leads to indefinite baseline-chasing. Three methods covering distinct DG families provide sufficient coverage for the first paper.

## D007: Graph gold subset split into QC dev set and locked test set

**Decision**: QC Development Set (50-100 images) for tuning conversion parameters. Graph Gold Test Set (100-200 images) locked for evaluation only.

**Rationale**: Without this separation, graph conversion tuning and model evaluation use the same labels, creating a subtle form of leakage.

## D008: Auto graph labels require quality gate before training

**Decision**: P1 includes a go/no-go gate. Auto-derived graph targets enter training only after quality metrics are reported on the gold test set.

**Rationale**: If auto-skeletonization has systematic errors, the model may learn those errors rather than true crack morphology. The gate ensures label quality is verified before scale-up.

## D009: B7 (relation head) belongs to P5

**Decision**: B7 is part of P5 (Crack-Spalling Relations + Engineering Evaluation) and is not required for the P3 model (Joint Region-Graph Reconstruction).

**Rationale**: Relation labels and relation definitions are less mature than graph labels. Blocking the main model on relation head development would delay the entire project.
