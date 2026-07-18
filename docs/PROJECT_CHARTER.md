# Project Charter: Cross-Site Morphology-Graph Learning for Infrastructure Defect Understanding

## Research Question

Can we learn a unified morphological graph representation of concrete defects (cracks, spalling) that generalizes across inspection sites, sensors, and environmental conditions — without requiring graph-level annotations at training time?

## Scope

- Multi-site concrete dam inspection imagery (UAV + handheld)
- Defect types: cracks (linear), spalling (areal), and their spatial relationships
- Morphological graph: skeleton-based representation capturing endpoints, junctions, width profiles, and connectivity
- Cross-site generalization via Leave-One-Domain-Out (LODO) evaluation

## Core Hypotheses

1. Morphological graph targets derived automatically from segmentation masks provide useful auxiliary supervision that improves both segmentation quality and structural understanding.
2. A shared encoder with task-specific heads (segmentation + graph components) learns more transferable features than mask-only training.
3. Graph-level representations enable downstream tasks (severity grading, progression tracking) that pixel-level masks cannot support.

## Non-Goals

- Real-time inference optimization (focus is on representation quality)
- Non-concrete infrastructure (bridges, roads) — future work
- 3D reconstruction or depth estimation
- Replacing human inspectors — this is a decision-support tool

## Success Criteria

- Demonstrate improved cross-site mIoU and BF1 over mask-only baselines under LODO protocol
- Show that learned graph representations correlate with expert severity assessments
- Produce a reproducible benchmark with grouped splits and protocol audits
