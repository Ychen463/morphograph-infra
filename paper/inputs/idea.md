# Do Morphological Auxiliary Losses Actually Improve Crack Segmentation? A Systematic Multi-Seed Study

## Core Idea

Auxiliary morphological supervision — skeleton distance transforms, topology-preserving losses, keypoint heatmaps — is widely assumed to improve crack segmentation quality. Similarly, standard domain generalization methods are expected to reduce cross-site performance gaps. We conduct a rigorous, multi-seed empirical study that challenges both assumptions.

We build a progressive baseline ladder on SegFormer-B2, systematically adding morphological supervision (clDice, skeleton DT, keypoint, width heads) and evaluating with pre-registered go/no-go gates. We then test three standard DG methods (CORAL, DANN, MixStyle) for cross-site generalization. Our central finding is negative: **no auxiliary morphological supervision significantly improves segmentation over the mask-only baseline when validated with multiple random seeds**, and **no standard DG method reduces the cross-site domain gap in a pseudo-domain setting**. We further show that single-seed comparisons produced two cases of misleading positive results that were overturned by multi-seed validation, highlighting the importance of proper experimental methodology in this domain.

## Key Research Questions (and Answers)

1. **Does morphological supervision improve segmentation?** — No. B2 skeleton DT shows +0.5%+/-0.3% (p=0.145, ns), B1a clDice shows +0.2%+/-0.9% (p=0.822, ns). Single-seed claims of "+1%" and "-1.6%" were both overturned.
2. **Can we extract usable crack graphs from predicted masks?** — Poorly. Median edge F1 = 0 across all methods. Skeleton supervision (B2) does not improve graph extraction (Wilcoxon p > 0.23).
3. **Can direct neural graph prediction outperform post-hoc extraction?** — Not yet. Architecture works (overfit gates pass) but auto-generated training labels create a quality ceiling.
4. **Do standard DG methods reduce cross-site performance gap?** — No. CORAL, DANN, MixStyle are all equivalent to ERM (p=0.878). Domain gap ~55% is stable.
5. **Are single-seed comparisons reliable?** — No. Three cases of misleading results overturned: B2 "+1%" (actual +0.5%, ns), B1a "-1.6%" (actual +0.2%, ns), and MixStyle "+52%" (actual ~0%, ns).

## Architecture

- **Backbone**: SegFormer-B2 (MIT encoder + MLP decoder)
- **SharedFPN**: Feature pyramid at 512x512 resolution for auxiliary heads
- **Auxiliary heads**: SegDecoder (semantic), SkeletonHead (DT regression), EndpointHead, JunctionHead, WidthHead
- **P3 extension**: SharedFPN128 at 128x128, NodeHeatmapHead (CenterNet-style), EdgeClassifier (pairwise adjacency)
- **DG methods**: MixStyle (feature statistics mixing), CORAL (covariance alignment), DANN (gradient reversal)

## Datasets

- **DamSegment**: 1500 images (1275 train / 225 val), 640x640 resized to 512x512, 3 classes (background, crack, spalling). Class imbalance: BG 96.6%, crack 2.2%, spalling 1.2%.
- **s2ds**: 743 images, 512x512, out-of-distribution test set from different inspection sites.

## Training Protocol

- Uniform 100 epochs for all baselines (fair ablation)
- Differential LR: encoder 6e-5, heads 6e-4
- CE weights [0.2, 2.0, 3.0] + Dice loss
- GroupNorm (batch=4), mixed precision training
- Per-baseline loss configurations (not monolithic composite)
- Baseline multi-seed: B0 and B2_best validated with 3 seeds (42, 123, 7)
- DG multi-seed: 3 seeds (42, 123, 7) x 4 methods = 12 runs

## Evaluation Protocol

- **Segmentation**: mIoU_fg (foreground), per-class IoU, BF1 (boundary F1)
- **Graph extraction**: endpoint F1, junction F1, edge F1 at 3 tolerance tiers (strict @5px, relaxed @10px, lenient @15px), graph edit distance, path continuity, width MAE
- **Instance**: spalling instance P/R/F1 via connected components + Hungarian matching
- **Cross-domain**: DamSegment val vs s2ds OOD, domain gap = mIoU_fg difference
- **Statistical**: Wilcoxon signed-rank (paired per-image), t-test (multi-seed), significance at p<0.05
