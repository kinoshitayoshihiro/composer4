#!/bin/bash
set -euo pipefail

BASE_DIR="/Volumes/SSD-SCTU3A/ラジオ用/music_21/composer2-3"
cd "$BASE_DIR"

echo "🎯 Stage2: LAMDA Metrics Calculation"
echo "====================================="
echo ""

# 仮想環境
source .venv311/bin/activate

# 互換性チェック
echo "🔍 Verifying Stage2 compatibility..."
python verify_stage2_compat.py output/drums_metadata || {
    echo "❌ Compatibility check failed"
    exit 1
}
echo "✅ Compatibility OK"
echo ""

# 古いStage2出力を削除
echo "🧹 Cleaning old Stage2 output..."
rm -rf output/drumloops_v3_stage2
mkdir -p logs
echo ""

# Stage2実行
echo "🎯 Running Stage2 extractor..."
echo "   Expected: ~20 minutes"
echo "   Threshold: 70.0"
echo ""

PYTHONPATH=. python scripts/lamda_stage2_extractor.py \
  --metadata-index output/drums_metadata/drums_index.pkl \
  --metadata-dir output/drums_metadata \
  --input-dir output/drumloops_v3 \
  --output-dir output/drumloops_v3_stage2 \
  --config configs/lamda/drums_stage2.yaml \
  --threshold 70.0 \
  --print-summary \
  2>&1 | tee logs/stage2_run_$(date +%Y%m%d_%H%M%S).log

echo ""
echo "✅ Stage2 Complete!"
echo ""
echo "📊 Results Summary:"
echo "-------------------"

# JSONファイルのサマリー表示
if [ -f output/drumloops_v3_stage2/summary.json ]; then
    python3 << 'PYEOF'
import json
with open("output/drumloops_v3_stage2/summary.json") as f:
    data = json.load(f)
    print(f"Total Loops:      {data.get('total_loops', 0):,}")
    print(f"Processed:        {data.get('processed', 0):,}")
    print(f"Passed:           {data.get('passed', 0):,}")
    print(f"Failed:           {data.get('failed', 0):,}")
    
    exc = data.get("exclusions", {})
    if exc:
        print("\n❌ Exclusions:")
        for reason, count in sorted(exc.items(), key=lambda x: -x[1]):
            print(f"   {reason}: {count:,}")
PYEOF
else
    echo "⚠️  summary.json not found"
fi

echo ""
echo "📁 Output files:"
ls -lh output/drumloops_v3_stage2/
