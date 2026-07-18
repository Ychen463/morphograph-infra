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
