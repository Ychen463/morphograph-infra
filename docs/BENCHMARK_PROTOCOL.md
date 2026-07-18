# Benchmark Protocol

## Grouped Split Rules

1. **Domain-aware splitting**: Images from the same physical site (domain) must never appear in both train and test sets.
2. **Group ID**: Each domain has a `group_id`. Domains sharing a group are treated as a single unit for splitting.
3. **Validation set**: Drawn only from training domains (never from test domain).
4. **Stratification**: Validation split is stratified by per-image class frequency to maintain class distribution.

## Leave-One-Domain-Out (LODO) Protocol

- For N domain groups, run N folds.
- Each fold holds out one group as the test set.
- Report per-fold metrics and aggregate mean +/- std across folds.
- Minimum 3 seeds per fold for significance.

## Strict Rules

- **No test-domain tuning**: Hyperparameters, augmentation policy, loss weights, and early stopping criteria must be fixed before seeing any test-domain data.
- **No test-domain leakage**: Near-duplicate detection (perceptual hash) must be run to verify no leaked images.
- **Reproducibility**: All splits must be deterministic given the same seed and domain config.

## Baseline Ladder

### Group 1: Representation (B0-B7)

| ID | Config | Validates |
|----|--------|-----------|
| B0 | Mask-only SegFormer-B2 (CE+Dice) | Strong pixel baseline |
| B1a | B0 + clDice | Topology loss (primary) |
| B1b | B0 + SRL | Topology loss (supplementary) |
| B2 | B0 + skeleton head | Dense morphology supervision |
| B3 | B0 + endpoint/junction heads | Node supervision |
| B4 | B0 + node + edge supervision | Graph supervision |
| B5 | Joint mask + graph + width | Complete crack graph |
| B6 | B5 + spalling region head | Joint region-graph |
| B7 | B6 + relation head | Crack-spalling relations (belongs to P5) |

**Capacity control**: Each added head requires a parameter-matched control model without graph supervision. This separates the effect of additional capacity from the supervision signal.

### Group 2: Cross-Domain (D0-D7)

| ID | Config |
|----|--------|
| D0 | B0, standard ERM |
| D1 | Best region-graph model (from Group 1), standard ERM |
| D2a | CORAL (feature-distribution alignment) |
| D2b | DANN (domain-adversarial alignment) |
| D2c | MixStyle (appearance-statistics perturbation) |
| D3 | Appearance augmentation + graph consistency |
| D4 | Appearance-morphology disentanglement |
| D5 | Graph prototype alignment |
| D6 | Selective foundation-feature transfer |
| D7 | Full Graph-Invariant DG |

### Oracle Baselines (Require Known Domain ID)

These are separate from the main DG ladder because they require domain identity at inference time, which is not available in a true LODO zero-shot setting.

| ID | Config |
|----|--------|
| O0 | Domain-specific BN oracle |
| O1 | Known-domain token oracle |
