#!/bin/bash
# Stage3 Drum: 生成→評価→レポート→判定の自動化
# Usage: ./scripts/run_stage3_drum_eval.sh [--tempo 120] [--style pop_straight] [--seed 42]

set -e
# ✔ リポ自動検出（Colab/ローカル共通）
REPO="$(cd "$(dirname "$0")"/.. && pwd)"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUT="$REPO/output/drumgen_eval_$TIMESTAMP"

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

# 1) 生成（DrumAdapter経由）
echo "⏳ Step 1/4: Generating drum patterns..."
"$REPO/.venv311/bin/python" -m adapters.run_drum_adapter \
  --n "$N_SAMPLES" \
  --tempo "$TEMPO" \
  --time-sig "4/4" \
  --length-bars "$LENGTH_BARS" \
  --style "$STYLE" \
  --density "$DENSITY" \
  --swing "$SWING" \
  --seed "$SEED" \
  --out "$OUT/generated" \
  || { echo "❌ Generation failed"; exit 1; }

echo "✅ Generated $(find "$OUT/generated" -name "*.mid" | wc -l) MIDI files"
echo ""

# 2) バッチ評価
echo "⏳ Step 2/4: Running batch evaluation..."
"$REPO/.venv311/bin/python" "$REPO/scripts/eval_drum_batch.py" \
  --input-dir "$OUT/generated" \
  --output-json "$OUT/eval_result.json" \
  --output-csv "$OUT/eval_files.csv" \
  || { echo "⚠️  Evaluation failed (non-fatal)"; }
echo ""

# 3) A/Bレポート生成（ベースラインがある場合）
echo "⏳ Step 3/4: Generating report..."
mkdir -p "$REPO/output/reports"
BASELINE="$REPO/output/reports/baseline.json"

# 現在の評価をreportsに保存
cp "$OUT/eval_result.json" "$REPO/output/reports/current.json" 2>/dev/null || true

if [ -f "$BASELINE" ]; then
  # ベースラインがある場合はA/B比較
  "$REPO/.venv311/bin/python" "$REPO/scripts/ab_report_simple.py" \
    --eval-a "$BASELINE" \
    --eval-b "$REPO/output/reports/current.json" \
    --out-md "$OUT/stage3_ab_report.md" \
    --name-a "Baseline" \
    --name-b "Current ($TIMESTAMP)" \
    || { echo "⚠️  A/B report failed (non-fatal)"; }
else
  echo "⚠️  Baseline not found at $BASELINE"
  echo "   To create baseline: cp $OUT/eval_result.json $BASELINE"
fi
echo ""

# 4) 受け入れ判定
echo "⏳ Step 4/4: Checking acceptance criteria..."
if [ -f "$OUT/eval_result.json" ]; then
  if [ -f "$BASELINE" ]; then
    # ベースラインがあればA/B比較で判定（--strict-exitは既にab_report_simple.pyに含まれている）
    "$REPO/.venv311/bin/python" "$REPO/scripts/ab_report_simple.py" \
      --eval-a "$BASELINE" \
      --eval-b "$REPO/output/reports/current.json" \
      --out-md "$OUT/stage3_ab_report.md" \
      --name-a "Baseline" \
      --name-b "Current ($TIMESTAMP)" \
      --strict-exit \
      && echo "   ✅ PASS: A/B acceptance criteria met" \
      || { echo "   ❌ FAIL: A/B acceptance criteria not met"; exit 1; }
  else
    # ベースラインがない場合は単体閾値判定
    "$REPO/.venv311/bin/python" "$REPO/scripts/check_acceptance.py" \
      --eval-json "$OUT/eval_result.json" \
      --style "$STYLE" \
      --strict-exit \
      && echo "   ✅ PASS: Acceptance criteria met" \
      || { echo "   ❌ FAIL: Acceptance criteria not met"; exit 1; }
  fi
else
  echo "⚠️  No evaluation result to check"
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
if [ -f "$OUT/stage3_ab_report.md" ]; then
  echo "📊 A/B Report:"
  cat "$OUT/stage3_ab_report.md"
  echo ""
fi
echo "Next steps:"
echo "  1. Listen to: $OUT/generated/*.mid"
echo "  2. Check report: $OUT/stage3_ab_report.md (if available)"
echo "  3. Run with different styles: ./scripts/run_stage3_drum_eval.sh --style shuffle"
echo "  4. Set as baseline: cp $OUT/eval_result.json $REPO/output/reports/baseline.json"

