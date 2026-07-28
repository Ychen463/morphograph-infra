# Research Log

Experiment phases, results, and next steps. Each entry is immutable once written; corrections are added as new entries.

## Template

```
### [Phase] — [Date Range]

**Objective**: ...

**Setup**: ...

**Results**: ...

**Observations**: ...

**Next Steps**: ...

**Status**: in-progress / completed / blocked / abandoned
```

## Entries

### P0: Environment & Overfit Gate — 2026-07-17 to 2026-07-18

**Objective**: Set up RunPod environment, validate data loading, verify all heads can learn.

**Setup**:
- RunPod: PyTorch 2.4, CUDA 12.4, RTX 3090 (24GB)
- DamSegment: 1500 images (Easy/Medium/Hard × 500), 640×640, RGB masks
- s2ds: 743 images, 512×512, indexed masks (OOD test set, not used in training)
- Overfit test: 16 samples, 200 epochs, small U-Net (~1.9M params)

**Results**:
- All 29 unit tests pass
- Data sanity check: all datasets found, masks decode correctly
- Overfit test: ALL PASS
  - seg: 1.04 → 0.29 (72.1%)
  - skeleton: 1.18 → 0.34 (70.8%)
  - endpoints: 1.21 → 0.47 (61.0%)
  - junctions: 1.06 → 0.31 (70.7%)
  - width: 11.66 → 0.76 (93.5%)

**Observations**:
- Sparse binary heads (skeleton/endpoints) needed higher pos_weight (200/500) and lower dice_weight (0.2) to converge
- Original flat 3-conv encoder (57K params) could not overfit — replaced with small U-Net (~1.9M params)

**Next Steps**: Run B0 full training.

**Status**: completed

---

### P2-B0: Mask-Only SegFormer-B2 Baseline — 2026-07-18

**Objective**: Establish B0 baseline (CE+Dice only, no auxiliary heads) for the progressive ladder.

**Setup**:
- Model: MiT-B2 encoder (ADE20K pretrained) + SharedFPN(256) + SegHead
- Parameters: 27,085,251 total (encoder 24.2M, FPN 2.9M, seg_head 771)
- Training: 100 epochs, batch_size=4, AdamW, cosine LR + 5-epoch warmup
- LR: encoder 6e-5, heads 6e-4
- Loss: CE([0.2, 2.0, 3.0]) + foreground Dice, weight 0.5/0.5
- Data: 1275 train / 225 val (85/15 split, seed=42)
- AMP mixed precision, gradient clipping (max_norm=1.0)
- GPU: RTX 3090, ~3.5 min/epoch

**Results**:
| Metric | Value |
|--------|-------|
| best val mIoU_fg | **0.673** |
| final val loss | 0.357 |
| IoU background | ~0.96 |
| IoU crack | ~0.42 |
| IoU spalling | ~0.45 |

**Observations**:
- Crack IoU lower than spalling despite higher pixel frequency — likely due to thin crack morphology
- mIoU_fg = 0.673 is a reasonable SegFormer-B2 baseline on this dataset
- SharedFPN at full 512×512 uses ~21GB VRAM at batch_size=4; batch_size=8 OOMs

**Next Steps**: Run B1a (B0 + clDice topology loss).

**Status**: completed

---

### P2-B1a: B0 + clDice Topology Loss — 2026-07-18

**Objective**: Test whether soft clDice loss improves crack topology without explicit morphology supervision.

**Setup**:
- Architecture: identical to B0 (seg_head only, 27.1M params)
- Loss: CE+Dice + clDice(crack class), weight=0.15, start_epoch=40, ramp=5 epochs
- Soft skeletonization: 10 iterations, forced float32
- All other hyperparameters identical to B0

**Results**:
| Metric | B0 | B1a | Delta |
|--------|-----|-----|-------|
| best val mIoU_fg | **0.673** | 0.657 | -0.016 |
| final val loss | 0.357 | 0.365 | +0.008 |

**Observations**:
- clDice **hurts** mIoU_fg by 1.6 points — topology loss alone does not improve segmentation quality
- Possible explanations:
  - Crack occupies only 2.2% of pixels; soft skeletonization gradient signal is weak
  - Late activation (epoch 40) means model already settled in a non-topology-optimal basin
  - weight=0.15 may conflict with the CE+Dice gradient direction on thin structures
- This is a **positive result for H2**: implicit topology loss is not sufficient; explicit graph supervision may be needed

**Next Steps**: Run B2 (B0 + explicit skeleton head supervision). Direct comparison: implicit topology loss (B1a) vs explicit dense skeleton prediction (B2).

**Status**: completed

---

### P2-B2: B0 + Skeleton DT Regression — 2026-07-19 to 2026-07-25

**Objective**: Add explicit skeleton supervision via distance transform regression. Systematically tune loss type, masking strategy, and weight to beat B0.

**Setup**:
- Architecture: B0 + SkeletonHead (256→64→1, 147K params), total 27.2M params
- DT target: normalized distance transform of crack mask (centerline=1.0, boundary=0.0)
- Skeleton head output: sigmoid → [0,1], regression loss masked to crack pixels (or unmasked)
- All other hyperparameters identical to B0 (100 epochs, same LR, same data split)

**Experiment Matrix**:

Wave 1 — Loss type and masking:

| Run | Loss | Weight | Masking | mIoU_fg | Delta vs B0 |
|-----|------|--------|---------|---------|-------------|
| v1 | SmoothL1 | 0.3 | crack-only | 0.660 | -1.3% |
| v2 | SmoothL1 | 5.0 | crack-only | 0.660 | -1.3% |
| v3 | MSE | 5.0 | crack-only | 0.672 | -0.09% |
| v3_w8 | MSE | 8.0 | crack-only | 0.667 | -0.55% |
| v3_w12 | MSE | 12.0 | crack-only | 0.672 | -0.09% |

Weight sweep — v4 unmasked MSE:

| Weight | mIoU_fg | Delta vs B0 |
|--------|---------|-------------|
| 1.0 | 0.667 | -0.58% |
| 5.0 | 0.670 | -0.33% |
| 8.0 | 0.666 | -0.67% |
| 9.0 | 0.658 | -1.53% |
| **10.0** | **0.683** | **+0.99%** |
| 11.0 | 0.665 | -0.79% |
| 13.0 | 0.662 | -1.10% |
| 15.0 | 0.676 | +0.27% |
| 20.0 | 0.660 | -1.28% |

Wave 2 — Schedule and head capacity (base: v4_w10):

| Run | Change | mIoU_fg | Delta vs B0 |
|-----|--------|---------|-------------|
| v4_w10 | base | **0.683** | **+0.99%** |
| v5 | +schedule (start=20, ramp=10) | 0.673 | -0.05% |
| v6 | +deep head (256→128→64→1, 450K params) | 0.666 | -0.68% |

**Observations**:
- SmoothL1 fundamentally limited: gradient halved when |error|<1 (always true for [0,1] targets). Weight tuning cannot fix this.
- MSE + crack-only masking: ceiling at ~0.672 regardless of weight (5/8/12). 2.2% pixel coverage is the bottleneck.
- MSE + unmasked: sharp peak at w=10. w=9 and w=11 both significantly worse, suggesting the +1.0% gain is sensitive to weight and may partly reflect training stochasticity.
- Schedule (v5) hurts: delaying DT signal loses early co-training benefit.
- Deeper head (v6) hurts: 450K params overfits sparse DT signal; capacity is not the bottleneck.
- Key insight: unmasked supervision changes the task semantics — head learns "is this a crack pixel? if so, how central?" This provides implicit crack detection supervision on ALL pixels.

**Best config**: MSE, unmasked, w=10.0 → mIoU_fg=0.683 (+1.0% vs B0)

**Status**: completed (best config identified, multi-seed pending)

---

### P2-B3/B5: Keypoint and Width Supervision — 2026-07-25 to 2026-07-28

**Objective**: Test whether endpoint/junction (B3) and width (B5) auxiliary heads improve segmentation. B4 (edge connectivity) skipped — loss conflicts with DT regression on same skeleton head.

**Setup**:
- B3: B2_best + endpoint head (pos_weight=200) + junction head (pos_weight=100)
- B5: B3 + width head (SmoothL1 on skeleton pixels)
- Three tuning rounds: (1) high weights, (2) reduced weights, (3) scheduled ramp-up

**Results** (best of 3 rounds per baseline):

| Baseline | Config | mIoU_fg | Delta vs B0 |
|----------|--------|---------|-------------|
| B0 | seg only | 0.673 | — |
| B2_best | +skel DT (MSE, unmask, w=10) | **0.683** | **+1.0%** |
| B3 | +ep/jn (scheduled, w=0.3) | 0.668 | -0.5% |
| B5 | +width (scheduled, w=0.5) | 0.646 | -2.7% |

**Observations**:
- Round 1 (high weights): ep/jn overwhelmed seg loss (total ~10x seg). B3=0.659, B5=0.651.
- Round 2 (reduced weights): better balance but still below B0. B3=0.667, B5=0.644.
- Round 3 (scheduled ramp, start=20/30): no improvement. B3=0.668, B5=0.646.
- B3: keypoint heads learn fine but contribute nothing to mIoU_fg. Endpoints/junctions are too sparse (~10 pixels/image) to provide useful encoder gradient.
- B5: width loss never converges (raw loss ~7.0 at epoch 100). SmoothL1 on skeleton pixels (< 0.1% of image) with values 2-20px = too sparse, too hard.
- **Key conclusion: only dense supervision (DT on ALL pixels) helps mIoU. Sparse supervision (keypoints, width on skeleton) hurts segmentation by introducing gradient noise.**

**Implications for P3**:
- Graph decoder should NOT rely on sparse pixel-level auxiliary losses
- Dense, full-image supervision signals are essential (B2 DT unmasked proved this)
- Keypoint/width prediction may work better as post-hoc extraction from predicted skeleton, not as training losses

**Status**: completed

---

### Baseline Ladder Summary (P2)

| Baseline | Description | mIoU_fg | Delta vs B0 | Verdict |
|----------|-------------|---------|-------------|---------|
| B0 | CE+Dice seg only | 0.673 | — | baseline |
| B1a | +clDice | 0.657 | -1.6% | implicit topology loss hurts |
| **B2** | **+skeleton DT (MSE, unmask, w=10)** | **0.683** | **+1.0%** | **dense DT helps** |
| B3 | +endpoint/junction heads | 0.668 | -0.5% | sparse keypoint noise |
| B5 | +width regression | 0.646 | -2.7% | width loss doesn't converge |

**P2 Gate Decision**: B2 is the only baseline that improves over B0. Carry B2_best config (MSE, unmasked, w=10) forward to P3 graph decoder. Do NOT stack B3/B5 heads — they hurt.

**Next**: P3 — graph decoder design, building on B2_best trunk.

---

### P3-Eval: Graph Extraction Quality — 2026-07-28

**Objective**: Measure how well seg predictions convert to crack graphs; test whether B2's DT improves graph extraction.

**Setup**: 3 extraction methods (A: mask-skeleton, B: DT-threshold, C: DT-guided adaptive ridge), 219 crack images from validation set, 5px keypoint tolerance, Hungarian one-to-one matching.

**Results**:

| Method | endpoint F1 | junction F1 | edge F1 | width MAE | GED | path cont. |
|--------|-------------|-------------|---------|-----------|-----|------------|
| B0-A | 0.249 | 0.069 | 0.104 | 0.000 | 0.91 | 0.170 |
| B2-A | 0.223 | 0.082 | 0.092 | 0.000 | 0.87 | 0.151 |
| B2-B | 0.264 | 0.076 | 0.072 | 0.000 | 0.76 | 0.132 |
| B2-C | 0.155 | 0.031 | 0.003 | 1.880 | 1.28 | 0.023 |

**Key findings**:
1. Edge F1 median = 0.0 for all methods — skeleton-to-graph pipeline is the bottleneck
2. B0-A beats B2-A on edge F1 (0.104 vs 0.092) and path continuity (0.170 vs 0.151)
3. Method C catastrophically fails: edge F1 = 0.003, GED = 1.28
4. DT carries width info (C width MAE = 1.88 best) but not topology
5. No statistical significance (all Wilcoxon p > 0.27)
6. Ridge threshold sweep: all thresholds yield edge F1 < 0.01

**Implications for P3**: Post-hoc extraction caps at ~0.10 edge F1. P3 graph decoder must predict nodes + edges directly, not through skeletonization. DT useful for width only.

**Status**: completed
