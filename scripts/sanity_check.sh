#!/usr/bin/env bash
# ============================================================
# 数据 Sanity Check — 在首次训练前跑一次
# ============================================================
# 检查:
#   1. 数据集目录存在且文件数量正确
#   2. 图像分辨率一致
#   3. Mask 解码正确 (RGB -> canonical classes)
#   4. 类别分布统计
#   5. 生成几张可视化预览
#
# 用法:
#   cd /workspace/morphograph-infra
#   bash scripts/sanity_check.sh
# ============================================================
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_DIR"

echo "============================================"
echo " Data Sanity Check"
echo "============================================"
echo ""

python3 << 'PYEOF'
import sys
from pathlib import Path
import numpy as np

# ── Check dataset directories ──
DATA_ROOT = Path("data/raw")
print("[1] Checking dataset directories...")

datasets = {
    "DamSegment Easy":   (DATA_ROOT / "DamSegment/Damage Segmentaion/Easy", 500),
    "DamSegment Medium": (DATA_ROOT / "DamSegment/Damage Segmentaion/Medium", 500),
    "DamSegment Hard":   (DATA_ROOT / "DamSegment/Damage Segmentaion/Hard", 500),
    "S2DS":              (DATA_ROOT / "s2ds", None),
}

all_ok = True
for name, (path, expected) in datasets.items():
    if not path.exists():
        print(f"  MISSING: {name} at {path}")
        all_ok = False
        continue

    img_dir = path / "Images" if "DamSegment" in name else path / "images"
    mask_dir = path / "Labels/Mask" if "DamSegment" in name else path / "masks"

    if not img_dir.exists():
        print(f"  MISSING: {name} images at {img_dir}")
        all_ok = False
        continue

    n_img = len(list(img_dir.iterdir()))
    n_mask = len(list(mask_dir.iterdir())) if mask_dir.exists() else 0

    status = "OK" if (expected is None or n_img == expected) else "MISMATCH"
    print(f"  {name}: {n_img} images, {n_mask} masks [{status}]")

    if n_img != n_mask:
        print(f"    WARNING: image/mask count mismatch!")
        all_ok = False

if not all_ok:
    print("\n  Some datasets are missing or have issues.")
    print("  Place data in data/raw/ and re-run.")
    sys.exit(1)

print("  All datasets found.\n")

# ── Check image properties ──
print("[2] Checking image properties (sampling 5 per dataset)...")
from PIL import Image

for name, (path, _) in datasets.items():
    img_dir = path / "Images" if "DamSegment" in name else path / "images"
    if not img_dir.exists():
        continue

    imgs = sorted(img_dir.iterdir())[:5]
    sizes = set()
    for f in imgs:
        with Image.open(f) as im:
            sizes.add(im.size)

    print(f"  {name}: sizes = {sizes}")

print("")

# ── Check mask decoding ──
print("[3] Checking DamSegment mask decoding (RGB -> canonical)...")
from morphograph.data.schema import decode_rgb_mask

dam_easy = DATA_ROOT / "DamSegment/Damage Segmentaion/Easy"
mask_dir = dam_easy / "Labels/Mask"
if mask_dir.exists():
    masks = sorted(mask_dir.iterdir())[:10]
    class_counts = {0: 0, 1: 0, 2: 0}
    total_px = 0

    for f in masks:
        rgb = np.array(Image.open(f))
        if rgb.ndim != 3 or rgb.shape[2] < 3:
            print(f"  WARNING: {f.name} is not RGB ({rgb.shape})")
            continue

        canonical = decode_rgb_mask(rgb)
        unique = np.unique(canonical)
        for c in unique:
            class_counts[c] = class_counts.get(c, 0) + int((canonical == c).sum())
        total_px += canonical.size

    if total_px > 0:
        for c, count in sorted(class_counts.items()):
            pct = 100 * count / total_px
            label = {0: "background", 1: "crack", 2: "spalling"}.get(c, f"class_{c}")
            print(f"  {label}: {pct:.2f}%")
        print("  Mutually exclusive check: ", end="")
        print("PASS" if set(class_counts.keys()) <= {0, 1, 2} else "FAIL")
else:
    print("  Skipped (no DamSegment data found)")

print("")

# ── Check s2ds masks ──
print("[4] Checking S2DS mask format...")
s2ds_mask_dir = DATA_ROOT / "s2ds/masks"
if s2ds_mask_dir.exists():
    masks = sorted(s2ds_mask_dir.iterdir())[:5]
    for f in masks:
        m = np.array(Image.open(f))
        unique = np.unique(m)
        print(f"  {f.name}: shape={m.shape}, unique={unique}")

    # Check for classes beyond {0, 1, 2}
    all_classes = set()
    for f in sorted(s2ds_mask_dir.iterdir())[:50]:
        m = np.array(Image.open(f))
        all_classes.update(np.unique(m).tolist())

    extra = all_classes - {0, 1, 2, 255}
    if extra:
        print(f"  WARNING: S2DS has classes {extra} not in canonical set!")
        print(f"  These MUST map to ignore(255), not background(0).")
    else:
        print(f"  All classes in {{0,1,2,255}}: OK")
else:
    print("  Skipped (no S2DS data found)")

print("")

# ── Save preview images ──
print("[5] Saving preview images to runs/sanity/...")
import os

preview_dir = Path("runs/sanity")
preview_dir.mkdir(parents=True, exist_ok=True)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dam_easy_imgs = DATA_ROOT / "DamSegment/Damage Segmentaion/Easy/Images"
    dam_easy_masks = DATA_ROOT / "DamSegment/Damage Segmentaion/Easy/Labels/Mask"

    if dam_easy_imgs.exists():
        samples = sorted(dam_easy_imgs.iterdir())[:4]
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        for i, img_path in enumerate(samples):
            img = np.array(Image.open(img_path))
            mask_path = dam_easy_masks / img_path.name.replace(".jpg", ".png")
            if not mask_path.exists():
                # Try same extension
                mask_path = dam_easy_masks / img_path.name

            axes[0, i].imshow(img)
            axes[0, i].set_title(img_path.name[:15])
            axes[0, i].axis("off")

            if mask_path.exists():
                mask_rgb = np.array(Image.open(mask_path))
                canonical = decode_rgb_mask(mask_rgb) if mask_rgb.ndim == 3 else mask_rgb
                axes[1, i].imshow(canonical, cmap="tab10", vmin=0, vmax=3)
            axes[1, i].axis("off")

        plt.tight_layout()
        plt.savefig(preview_dir / "damsegment_preview.png", dpi=100)
        plt.close()
        print(f"  Saved {preview_dir / 'damsegment_preview.png'}")
except Exception as e:
    print(f"  Preview generation failed: {e}")

print("")
print("============================================")
print(" Sanity check complete!")
print("============================================")
PYEOF
