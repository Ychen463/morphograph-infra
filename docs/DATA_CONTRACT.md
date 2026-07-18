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
