# Experiment Roadmap

## Phase 0: Infrastructure & Data

- Set up repository structure, CI, and reproducibility guardrails
- Implement dataset adapters and canonical manifest builder
- Implement grouped split + leakage audit
- Build crack mask-to-graph conversion pipeline
- Establish gold subset with manually reviewed graph labels

## Phase 1: Baseline Ladder (B0-B5)

- B0: Vanilla SegFormer-B2 (CE+Dice)
- B1: + ImageNet pretrained encoder
- B2: + Standard augmentation
- B3: + Class-weighted CE
- B4: + Boundary-aware loss
- B5: + clDice connectivity loss
- All under LODO protocol, 3 seeds each

## Phase 2: Domain Generalization (B6)

- Domain-specific batch normalization
- Domain tokens / domain-conditional heads
- Evaluate cross-site transfer gap

## Phase 3: MorphoGraph (B7+)

- Add auxiliary graph heads (endpoint, junction, edge, width)
- Multi-task loss balancing
- Evaluate graph quality vs. gold subset
- Ablate each head's contribution

## Phase 4: Analysis & Paper

- Per-domain breakdown and failure analysis
- Graph-based severity grading correlation study
- Compile results for submission
