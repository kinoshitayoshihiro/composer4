#!/bin/bash
set -euo pipefail

BASE_DIR="/Volumes/SSD-SCTU3A/ラジオ用/music_21/composer2-3"
cd "$BASE_DIR"

echo "🚀 Drumloops 完全再生成 + Stage2"
echo "=================================="
echo ""

# 仮想環境
source .venv311/bin/activate

# 古いpickleを削除
echo "📦 Cleaning old metadata..."
rm -rf output/drums_metadata
mkdir -p output/drums_metadata

# Stage1: クリーニング + Pickle生成
echo ""
echo "🎵 Stage1: Cleaning + Pickle generation..."
echo "   Expected: ~40 minutes"
echo ""

python -m scripts.clean_midi \
  --in data/loops \
  --out output/drumloops_v3 \
  --quarantine output/drumloops_v3_q \
  --instrument drums \
  --pickle-out output/drums_metadata \
  --shard-size 5000 \
  --resume \
  --emit-meta-json off \
  --jobs 1 \
  2>&1 | tee logs/clean_rerun_$(date +%Y%m%d_%H%M%S).log

echo ""
echo "✅ Stage1 Complete"
echo ""

# 互換性チェック
echo "🔍 Verifying Stage2 compatibility..."
python verify_stage2_compat.py output/drums_metadata || {
    echo "❌ Compatibility check failed"
    exit 1
}

# Stage2実行
echo ""
echo "🎯 Stage2: Metrics calculation..."
echo "   Expected: ~20 minutes"
echo ""

rm -rf output/drumloops_v3_stage2

PYTHONPATH=. python scripts/lamda_stage2_extractor.py \
  --metadata-index output/drums_metadata/drums_index.pkl \
  --metadata-dir output/drums_metadata \
  --input-dir output/drumloops_v3 \
  --output-dir output/drumloops_v3_stage2 \
  --config configs/lamda/drums_stage2.yaml \
  --threshold 70.0 \
  --print-summary \
  2>&1 | tee logs/stage2_rerun_$(date +%Y%m%d_%H%M%S).log

echo ""
echo "✅ All Complete!"
echo ""
echo "📊 Results:"
ls -lh output/drumloops_v3_stage2/
