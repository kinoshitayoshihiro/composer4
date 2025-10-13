#!/bin/bash
# Stage3 Drum: 生成→評価→レポート→判定の自動化
# Usage: ./scripts/run_stage3_drum_eval.sh [--tempo 120] [--style pop_straight] [--seed 42]

set -e
REPO=/Volumes/SSD-SCTU3A/ラジオ用/music_21/composer2-3
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUT=$REPO/output/drumgen_eval_$TIMESTAMP

# デフォルト値
TEMPO=120
STYLE="pop_straight"
LENGTH_BARS=64
DENSITY="mid"
SWING=2
N_SAMPLES=10
SEED=42

# 引数パース
while [[ $# -gt 0 ]]; do
  case $1 in
    --tempo) TEMPO="$2"; shift 2 ;;
    --style) STYLE="$2"; shift 2 ;;
    --length-bars) LENGTH_BARS="$2"; shift 2 ;;
    --density) DENSITY="$2"; shift 2 ;;
    --swing) SWING="$2"; shift 2 ;;
    --n-samples) N_SAMPLES="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

mkdir -p "$OUT/generated"

echo "🎵 Stage3 Drum Evaluation Pipeline"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Tempo:       $TEMPO BPM"
echo "Style:       $STYLE"
echo "Length:      $LENGTH_BARS bars"
echo "Density:     $DENSITY"
echo "Swing:       $SWING"
echo "Samples:     $N_SAMPLES"
echo "Seed:        $SEED"
echo "Output:      $OUT"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# 1) 生成 (10サンプル、pop_straight、BPM=120)
echo "⏳ Step 1/4: Generating drum patterns..."
"$REPO/.venv311/bin/python" "$REPO/scripts/generate_drum_samples.py" \
  --n "$N_SAMPLES" \
  --tempo "$TEMPO" \
  --length-bars "$LENGTH_BARS" \
  --style "$STYLE" \
  --density "$DENSITY" \
  --swing "$SWING" \
  --seed "$SEED" \
  --output-dir "$OUT/generated" \
  || { echo "❌ Generation failed"; exit 1; }

echo "✅ Generated $(find "$OUT/generated" -name "*.mid" | wc -l) MIDI files"
echo ""

# 2) Stage2再評価
echo "⏳ Step 2/4: Running Stage2 evaluation..."
if [ ! -f "$REPO/scripts/quick_eval_stage2.py" ]; then
  echo "⚠️  quick_eval_stage2.py not found - skipping Stage2 evaluation"
  echo "   (This is OK for initial testing)"
else
  "$REPO/.venv311/bin/python" "$REPO/scripts/quick_eval_stage2.py" \
    --input-dir "$OUT/generated" \
    --output-dir "$OUT/stage2" \
    || { echo "⚠️  Stage2 evaluation failed (non-fatal)"; }
fi
echo ""

# 3) A/Bレポート生成
echo "⏳ Step 3/4: Generating A/B report..."
if [ ! -f "$REPO/scripts/ab_summarize_v2.py" ]; then
  echo "⚠️  ab_summarize_v2.py not found - skipping A/B report"
  echo "   (Will be implemented in Phase 1.2)"
else
  "$REPO/.venv311/bin/python" "$REPO/scripts/ab_summarize_v2.py" \
    --input "$OUT/stage2" \
    --output "$OUT/stage3_ab_report.md" \
    || { echo "⚠️  A/B report failed (non-fatal)"; }
fi
echo ""

# 4) 受け入れ判定
echo "⏳ Step 4/4: Checking acceptance criteria..."
if [ ! -f "$REPO/scripts/check_acceptance.py" ]; then
  echo "⚠️  check_acceptance.py not found - skipping acceptance check"
  echo "   (Will be implemented in Phase 1.2)"
else
  "$REPO/.venv311/bin/python" "$REPO/scripts/check_acceptance.py" \
    --report "$OUT/stage3_ab_report.md" \
    --bar-violations 0.0 \
    --hat-grid 0.85 \
    --pass-rate 0.65 \
    || { echo "❌ Acceptance criteria not met"; exit 1; }
fi
echo ""

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ Evaluation complete!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Output directory: $OUT"
echo ""
echo "📁 Generated files:"
ls -lh "$OUT/generated" | tail -n +2
echo ""
echo "Next steps:"
echo "  1. Listen to: $OUT/generated/*.mid"
echo "  2. Check report: $OUT/stage3_ab_report.md (if available)"
echo "  3. Run with different styles: ./scripts/run_stage3_drum_eval.sh --style shuffle"
