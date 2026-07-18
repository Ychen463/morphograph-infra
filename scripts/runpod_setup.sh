#!/usr/bin/env bash
# ============================================================
# RunPod 一次性环境初始化
# ============================================================
# 用法:
#   1. 在 RunPod 启动一个 PyTorch 容器
#      推荐镜像: runpod/pytorch:2.4.0-py3.11-cuda12.4.1
#   2. ssh 进去后:
#      cd /workspace
#      git clone git@github.com:Ychen463/morphograph-infra.git
#      cd morphograph-infra
#      bash scripts/runpod_setup.sh
#
# 如果你用的是 Network Volume, 数据目录会在 /workspace 下持久化.
# ============================================================
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_DIR"

echo "============================================"
echo " MorphoGraph-Infra RunPod Setup"
echo "============================================"
echo "Repo:   $REPO_DIR"
echo "Python: $(python3 --version)"
echo ""

# ---- Step 1: 安装依赖 (不碰容器的 torch) ----
echo "[1/6] Installing Python dependencies (keeping container's torch)..."
pip install --quiet -r requirements_no_torch.txt
pip install --quiet -e ".[dev]"
echo "  Done."

# ---- Step 2: 验证 torch + CUDA ----
echo ""
echo "[2/6] Verifying PyTorch + CUDA..."
python3 -c "
import torch
print(f'  torch:       {torch.__version__}')
print(f'  CUDA avail:  {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU:         {torch.cuda.get_device_name(0)}')
    print(f'  CUDA ver:    {torch.version.cuda}')
    # Quick GPU sanity
    x = torch.randn(2, 3, device='cuda')
    assert x.device.type == 'cuda'
    print('  GPU compute: OK')
else:
    print('  WARNING: No GPU detected!')
"

# ---- Step 3: 验证关键依赖 ----
echo ""
echo "[3/6] Verifying key imports..."
python3 -c "
import transformers, timm, segmentation_models_pytorch
import albumentations, skimage, scipy, networkx, yaml, rich
print(f'  transformers:  {transformers.__version__}')
print(f'  timm:          {timm.__version__}')
print(f'  smp:           {segmentation_models_pytorch.__version__}')
print(f'  albumentations:{albumentations.__version__}')
print(f'  scikit-image:  {skimage.__version__}')
print('  All imports OK.')
"

# ---- Step 4: 验证 morphograph 包 ----
echo ""
echo "[4/6] Verifying morphograph package..."
python3 -c "
from morphograph.data.schema import SampleRecord, CANONICAL_CLASSES, decode_rgb_mask
from morphograph.data.graph_targets import mask_to_skeleton, detect_keypoints, mask_to_graph
from morphograph.data.registry import DatasetAdapter
from morphograph.evaluation.result_schema import ExperimentResult
from morphograph.evaluation.protocol_audit import audit_split
from morphograph.models.morphograph_net import MorphoAuxNet, BASELINE_HEADS, SharedFPN
from morphograph.losses.composite import WeightedCEDiceLoss, BinaryHeadLoss, B0LossConfig
from morphograph.metrics.segmentation import compute_iou, compute_cldice
from morphograph.metrics.graph_metrics import compute_graph_metrics
print('  All morphograph imports OK.')
"

# ---- Step 5: 跑测试 ----
echo ""
echo "[5/6] Running tests..."
python3 -m pytest tests/ -v --tb=short 2>&1 | tail -20
echo ""

# ---- Step 6: 创建目录结构 ----
echo "[6/6] Creating directory structure..."
mkdir -p data/raw data/derived data/manifests
mkdir -p data/gold/qc_dev data/gold/gold_test
mkdir -p runs/overfit_test
mkdir -p runs/B0 runs/B1a runs/B1b runs/B2 runs/B3 runs/B4
echo "  Directories created."

echo ""
echo "============================================"
echo " Setup complete!"
echo "============================================"
echo ""
echo "Next steps:"
echo "  1. 把 DamSegment 和 s2ds 数据放到 data/raw/"
echo "     data/raw/DamSegment/Damage Segmentaion/{Easy,Medium,Hard}/..."
echo "     data/raw/s2ds/{images,masks}/..."
echo ""
echo "  2. 运行数据 sanity check:"
echo "     bash scripts/sanity_check.sh"
echo ""
echo "  3. 运行 overfit test:"
echo "     python3 scripts/overfit_test.py"
echo ""
