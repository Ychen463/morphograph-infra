"""Training loops and utilities.

Training protocol:
    Optimizer:  AdamW
    LR:         encoder 6e-5 (pretrained), heads 6e-4 (new)
    Scheduler:  CosineAnnealing with 5-epoch warmup
    Batch size: 4 (512x512 on single GPU)
    Epochs:     100 (uniform across all B0-B5)
    Precision:  Mixed fp16 (except clDice soft-skel: force fp32)
    Norm:       GroupNorm(32) in FPN/heads (batch=4 too small for BN)

All baselines must use identical training budget, scheduler,
checkpoint selection, data augmentation, and seeds.
LR values are initial configs; re-confirm on validation groups
after split freeze.

Minimum viability gate before large-scale runs:
    Overfit 8-16 images, verify each head's loss decreases
    and outputs are visually reasonable.
"""
