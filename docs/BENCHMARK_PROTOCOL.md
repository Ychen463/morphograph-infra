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

| ID | Description |
|----|-------------|
| B0 | SegFormer-B2, CE+Dice, no pretraining tricks |
| B1 | B0 + ImageNet-pretrained encoder |
| B2 | B1 + standard augmentation (flip, rotate, color jitter) |
| B3 | B2 + class-weighted CE |
| B4 | B3 + boundary-aware loss (BF1-optimized) |
| B5 | B4 + clDice loss for crack connectivity |
| B6 | B5 + domain-specific BN or domain tokens |
| B7 | B6 + morphological graph auxiliary heads |
