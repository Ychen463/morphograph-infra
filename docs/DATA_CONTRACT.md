# Data Contract

## Sample Schema

Every sample in the dataset must conform to the `SampleRecord` dataclass:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `sample_id` | `str` | Yes | Globally unique identifier |
| `domain_id` | `str` | Yes | Source domain (site/sensor) |
| `group_id` | `str` | Yes | Grouping key for cross-validation |
| `image_path` | `Path` | Yes | Path to input image |
| `mask_path` | `Path` | Yes | Path to segmentation mask |
| `split` | `str` | No | train/val/test (assigned by protocol) |

## Canonical Class Mapping

All datasets must be remapped to the canonical class set before entering the pipeline:

| ID | Class | Description |
|----|-------|-------------|
| 0 | background | Non-defect regions |
| 1 | crack | Linear defects |
| 2 | spalling | Areal surface loss |
| 255 | ignore | Unmapped, ambiguous, or border regions |

## Ignore Class Policy

- Pixels with class 255 are excluded from all loss computations and metric calculations.
- Domain-specific classes that don't map to the canonical set are assigned to ignore.
- A thin ignore border (optional, configurable) may be applied around defect boundaries to reduce annotation noise.

## Derived Targets

Graph targets are derived automatically from canonical masks and stored in `data/derived/`:

- Skeleton maps (binary)
- Endpoint heatmaps (Gaussian-smoothed)
- Junction heatmaps (Gaussian-smoothed)
- Width maps (distance-transform-based)
- Graph adjacency (JSON per sample)

## Graph Label Quality

### Known Noise Sources in Auto-Derived Graph Labels

Automatic skeleton-based graph extraction introduces systematic errors:

- **Short spurs / false branches**: morphological artifacts from mask boundary roughness
- **False junctions**: caused by spurs intersecting the main skeleton
- **Centerline instability in wide cracks**: skeleton oscillates when crack width exceeds ~10 px
- **Crack-spalling boundary errors**: skeleton may extend into adjacent spalling regions
- **Broken mask false endpoints**: mask gaps produce spurious endpoints
- **Adjacent crack merging**: closely spaced parallel cracks may merge during skeletonization

### Graph-QC Development Set (50-100 images)

Purpose: tuning conversion parameters (skeleton pruning threshold, junction merge radius, spur threshold, graph simplification, width extraction).

- May be used during tool and pipeline development
- Selected to cover diverse crack morphologies (straight, branching, network, crack-spalling boundary)

### Graph Gold Test Set (100-200 images)

Purpose: evaluation only. **Locked after creation** — must not be used for tuning graph conversion parameters or model design decisions.

Used for:
- Auto graph label quality assessment
- Final model graph metric evaluation
- Endpoint/junction/edge accuracy reporting
- Width estimation error reporting

Must store per sample:
- Auto-generated graph (before human review)
- Gold graph (after human correction)
- Modification type (spur removal, junction merge, edge correction, endpoint correction, other)
- Annotator ID
- Review status (pending / reviewed / double-reviewed)

Inter-rater agreement: 30-50 images should be independently reviewed by two annotators to report agreement statistics.

### Auto-Label Quality Gate (P1 Go/No-Go)

Auto-generated graph targets may enter model training only after passing quality assessment on the gold test set. Metrics to report:

- Endpoint precision / recall
- Junction precision / recall
- Connected-component agreement
- Path preservation rate
- Edge coverage
- Width MAE
- False-spur rate

Absolute thresholds are defined after QC development set analysis, but the reporting method must be established before any model training begins.

## Engineering Quantities: Scale Boundaries

### Always Reportable (Pixel-Level)

All datasets can report:
- Pixel length and normalized length
- Pixel width
- Relative area (fraction of image or patch)
- Connected component count
- Longest-path error (in pixels)

### Only with Reliable Calibration

The following require confirmed imaging geometry:
- Millimetre width
- Physical crack length
- Square-millimetre or square-metre area

`resolution_mm_per_px` is valid only when the imaging plane, perspective distortion, and scale metadata have been verified. A global mm/px value cannot be assumed to apply uniformly across all pixel locations simply because the field exists in the manifest.
