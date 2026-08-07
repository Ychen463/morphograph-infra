# Experimental Log

## Experiment Group 1: Representation Baseline Ladder (P2)

All baselines trained for 100 epochs on DamSegment (1275 train / 225 val), SegFormer-B2.

### Table 1: Baseline Ladder — Segmentation Performance (single seed=42)

| Model | Description | mIoU_fg | IoU_crack | IoU_spalling | BF1_crack | Params |
|-------|-------------|---------|-----------|-------------|-----------|--------|
| B0 | Mask-only (CE+Dice) | 0.673 | 0.531 | — | 0.609 | 27.1M |
| B1a | B0 + clDice | 0.657 | — | — | — | 27.1M |
| B2 (w=1) | B0 + skeleton head (BCE) | 0.664 | — | — | — | 27.2M |
| B2_best (w=10,mse) | B0 + skeleton DT (MSE, unmasked) | 0.683 | 0.544 | 0.165 | 0.618 | 27.2M |
| B3 | B0 + endpoint/junction heads | 0.668 | — | — | — | 27.5M |
| B5 | B0 + full joint (skel+kp+width) | 0.646 | — | — | — | 27.7M |

### Table 1b: Multi-Seed Validation — B0 vs B1a vs B2_best (3 seeds: 42, 123, 7)

| Seed | B0 mIoU_fg | B1a mIoU_fg | B2_best mIoU_fg |
|------|-----------|------------|----------------|
| 7 | 0.6813 | 0.6927 | 0.6866 |
| 42 | 0.6628 | 0.6665 | 0.6718 |
| 123 | 0.6690 | 0.6587 | 0.6703 |
| **Mean** | **0.671 +/- 0.008** | **0.673 +/- 0.015** | **0.676 +/- 0.007** |

### Table 1c: Statistical Tests vs B0

| Method | Mean delta | Paired t p | TOST p (margin=+/-1%) | Conclusion |
|--------|-----------|-----------|----------------------|------------|
| B1a clDice | +0.002 +/- 0.009 | 0.822 (ns) | 0.159 (not eq) | Neither better nor provably equivalent |
| B2_best skelDT | +0.005 +/- 0.003 | 0.145 (ns) | 0.083 (not eq) | Neither better nor provably equivalent |

**Key findings:**
- Three single-seed conclusions overturned by multi-seed validation:
  - B2_best single-seed "+1.0%" (0.683 vs 0.673) → actual +0.5%+/-0.3%, p=0.145 (ns)
  - B1a single-seed "-1.6%" (0.657 vs 0.673) → actual +0.2%+/-0.9%, p=0.822 (ns)
  - (P4) MixStyle single-seed "+52%" OOD → actual ~0%, p=0.878 (ns)
- All auxiliary morphological supervision is statistically indistinguishable from mask-only baseline
- B1a has highest variance (std=0.015 vs B0's 0.008), suggesting clDice makes training less stable
- TOST not significant at n=3 (underpowered), but direction is clear: effects are near zero

### Table 2: Skeleton DT Weight Sweep (B2 variants, seed=42)

| Weight | Loss | mIoU_fg |
|--------|------|---------|
| 1.0 | BCE | 0.664 |
| 5.0 | MSE | 0.673 |
| 8.0 | MSE | 0.678 |
| 9.0 | MSE | 0.680 |
| 10.0 | MSE | 0.683 |
| 11.0 | MSE | 0.681 |
| 12.0 | MSE | 0.682 |
| 13.0 | MSE | 0.680 |
| 15.0 | MSE | 0.674 |
| 20.0 | MSE | 0.669 |

Optimal weight = 10.0, MSE loss, unmasked DT target. However, this advantage does not survive multi-seed validation.

---

## Experiment Group 2: Post-Hoc Graph Extraction (P2 eval)

Three methods evaluated on 219 crack-containing validation images, comparing B0 vs B2_best.

### Method Descriptions
- **Method A (skeleton extraction)**: skeletonize predicted crack mask → extract endpoints/junctions/edges
- **Method B (DT ridge detection)**: find ridges in predicted DT map → extract graph
- **Method C (DT peak detection)**: find local maxima in DT map → extract graph

### Table 3: Graph Extraction Quality (Method A, B2_best, mean over 219 images)

| Metric | Strict @5px | Relaxed @10px | Lenient @15px |
|--------|------------|--------------|--------------|
| Endpoint F1 | 0.290 | 0.433 | 0.509 |
| Junction F1 | 0.392 | 0.405 | 0.408 |
| Edge F1 | 0.092 | 0.220 | 0.277 |
| Path continuity | 0.151 | — | — |
| GED | 0.379 | — | — |
| Width MAE (px) | 2.51 | — | — |

### Table 4: Graph Extraction — B0 vs B2_best (Method A)

| Metric | B0 | B2_best | Wilcoxon p |
|--------|-----|---------|-----------|
| Endpoint F1 @5px | 0.283 | 0.290 | 0.720 (ns) |
| Edge F1 @5px | 0.104 | 0.092 | 0.367 (ns) |
| Edge F1 @10px | 0.219 | 0.220 | 0.632 (ns) |
| Edge F1 @15px | 0.268 | 0.277 | 0.234 (ns) |
| Edge F1 soft | 0.247 | 0.240 | 0.469 (ns) |
| Path continuity | 0.170 | 0.151 | 0.279 (ns) |

**Key finding:** B2 skeleton supervision does NOT significantly improve post-hoc graph extraction (all Wilcoxon p > 0.23). Median edge F1 = 0 for all methods and models. Graph extraction from predicted masks remains fundamentally challenging.

### Method B (DT ridge) and Method C (DT peak) — B2_best only

| Metric | Method B | Method C |
|--------|----------|----------|
| Edge F1 @5px | 0.053 | 0.003 |
| Edge F1 @10px | 0.104 | 0.011 |
| Endpoint F1 @5px | 0.166 | 0.054 |
| GED | 0.663 | 1.284 |

DT-based methods (B, C) perform worse than skeleton extraction (A).

---

## Experiment Group 3: Spalling Instance Evaluation (B6 post-hoc)

Connected-component instance extraction from B2_v5 semantic predictions.

### Table 5: Spalling Instance Metrics (B2_v5, IoU threshold=0.5)

| Metric | Value |
|--------|-------|
| Micro Precision | 0.563 |
| Micro Recall | 0.571 |
| Micro F1 | 0.567 |
| Macro Precision | 0.495 |
| Macro Recall | 0.484 |
| Macro F1 | 0.482 |
| Mean matched IoU | 0.431 |
| GT instances | 70 |
| Pred instances | 71 |

Post-hoc CC provides a reasonable baseline for spalling instance segmentation without any learned instance head.

---

## Experiment Group 4: Direct Graph Prediction (P3)

Neural graph prediction with NodeHeatmapHead (CenterNet-style) + EdgeClassifier.

### Table 6: P3 Overfit Results (16 images, 200 epochs)

| Gate | Metric | P3a (nodes) | P3b (nodes+edges) | Threshold |
|------|--------|------------|-------------------|-----------|
| 1 | node F1 @5px | 0.934 | — | >= 0.90 PASS |
| 2 | edge_gt F1 @10px | — | 0.917 | >= 0.90 PASS |
| 3 | edge F1 @10px (pred nodes) | — | 0.865 | >= 0.70 PASS |

### Table 7: P3 Full Train Results (1275 images, 100 epochs)

| Gate | Metric | P3b | Threshold |
|------|--------|-----|-----------|
| 4 | node F1 @5px | 0.30 | > post-hoc FAIL |
| 5 | edge F1 @10px | 0.195 | > 0.30 FAIL |
| 6 | mIoU_fg | 0.676 | >= 0.673 PASS (barely) |

### Table 8: P3h Hybrid (skeleton nodes + learned edges)

| Metric | P3h | P3b | Delta |
|--------|-----|-----|-------|
| edge_gt F1 @10px | 0.654 | 0.454 | +0.20 |
| edge F1 @10px | 0.183 | 0.195 | -0.01 |
| node F1 @5px | 0.284 | 0.297 | — |
| mIoU_fg | 0.670 | 0.676 | -0.006 |

**Key findings:**
- Overfit gates pass convincingly — architecture is sound
- Full train fails — auto-generated labels from mask_to_graph() create a quality ceiling
- Edge classifier generalizes well (edge_gt F1 0.654 in P3h), node localization is the bottleneck
- Need gold standard graph annotations to resume P3

---

## Experiment Group 5: Cross-Domain Evaluation (P4)

### Table 9: Cross-Domain Baseline (B0 vs B2_v5, no DG training)

| Model | DamSeg mIoU_fg | s2ds mIoU_fg | Domain Gap | s2ds inst_F1 |
|-------|---------------|-------------|-----------|-------------|
| B0 | 0.345 | 0.063 | 0.282 | 0.164 |
| B2 | 0.347 | 0.065 | 0.283 | 0.198 |

Domain gap ~82% (s2ds performance is ~18% of DamSeg).

### Table 10: DG Methods — Multi-Seed Results (3 seeds x 4 methods, FINAL)

Pseudo-domain training: DamSegment difficulty tiers (Easy/Medium/Hard) as proxy domains.

| Method | DG Type | DamSeg mIoU_fg | s2ds mIoU_fg | Domain Gap |
|--------|---------|---------------|-------------|-----------|
| D1 ERM | baseline | 0.675 +/- 0.007 | 0.115 +/- 0.012 | 0.560 +/- 0.015 |
| D2a CORAL | feature align | 0.672 +/- 0.006 | 0.118 +/- 0.010 | 0.554 +/- 0.013 |
| D2b DANN | adversarial | 0.674 +/- 0.009 | 0.120 +/- 0.011 | 0.553 +/- 0.015 |
| D2c MixStyle | style mixing | 0.668 +/- 0.010 | 0.116 +/- 0.006 | 0.552 +/- 0.016 |

### Table 11: DG Per-Class OOD Performance (s2ds, multi-seed mean)

| Method | s2ds crack IoU | s2ds spalling IoU |
|--------|---------------|-------------------|
| D1 ERM | 0.049 +/- 0.006 | 0.180 +/- 0.024 |
| D2a CORAL | 0.059 +/- 0.008 | 0.177 +/- 0.013 |
| D2b DANN | 0.048 +/- 0.004 | 0.192 +/- 0.019 |
| D2c MixStyle | 0.052 +/- 0.009 | 0.180 +/- 0.016 |

### Table 12: Statistical Significance (MixStyle vs ERM)

| Metric | t-stat | p-value | Significant? |
|--------|--------|---------|-------------|
| s2ds mIoU_fg | — | 0.878 | NO |
| Domain gap | — | — | NO |

### Single-Seed Comparison (seed=42 only, for reference)

| Method | DamSeg mIoU_fg | s2ds mIoU_fg | Domain Gap |
|--------|---------------|-------------|-----------|
| D1 ERM | 0.664 | 0.110 | 0.554 |
| D2c MixStyle | 0.658 | 0.167 | 0.490 |
| D2a CORAL | 0.666 | 0.133 | 0.533 |
| D2b DANN | 0.666 | 0.121 | 0.545 |

**Key findings:**
- Single-seed comparison was MISLEADING: MixStyle appeared +52% better (0.167 vs 0.110)
- Multi-seed reveals all 4 methods are statistically equivalent (p=0.878)
- Domain gap ~55% is stable across all methods
- Pseudo-domains (difficulty tiers) do not capture real domain shift
- Standard DG methods fail in single-source pseudo-domain setting
- This is a publishable negative result with proper multi-seed validation

---

## Summary of All Key Results

### Central Negative Finding
No auxiliary morphological supervision (clDice, skeleton DT, keypoint, width) significantly improves crack segmentation over the mask-only baseline when validated with multiple seeds. No standard DG method reduces the cross-site domain gap in a pseudo-domain setting.

### What Works (within scope)
1. Edge classifier architecture is sound (proven by overfit gates and P3h edge_gt results)
2. Post-hoc CC spalling instance extraction achieves F1=0.567 without learned instance head
3. Go/no-go gating prevents wasted computation (P3 full train stopped early)

### What Doesn't Work
1. Skeleton DT (B2): +0.5%+/-0.3% mIoU, p=0.145 — not significant across seeds
2. clDice (B1a): +0.2%+/-0.9% mIoU, p=0.822 — not significant (single-seed "-1.6%" was misleading)
3. Stacking auxiliary heads (B3, B5): degrades performance (-0.5% to -2.7%, single-seed only)
4. Post-hoc graph extraction: median edge F1 = 0 across all methods
5. Direct graph prediction (P3): auto-label quality ceiling blocks generalization
6. DG methods (CORAL, DANN, MixStyle): all equivalent to ERM (p=0.878)
7. Single-seed comparisons: THREE cases of misleading results (B2 "+1%", B1a "-1.6%", MixStyle "+52%")

### Methodological Contributions
1. Progressive baseline ladder with pre-registered go/no-go gates
2. Multi-tier graph evaluation protocol (strict @5px / relaxed @10px / lenient @15px)
3. Demonstration that single-seed comparisons produce unreliable conclusions in both auxiliary supervision and DG settings
4. Systematic analysis of auto-label quality ceiling for graph prediction
5. Pseudo-domain DG failure analysis: difficulty-based tiers do not approximate real domain shift
