#!/bin/bash
# 本番Stage2バッチ処理: 全56,598曲（LAMDA統合オプション対応）

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
PYTHON="${PROJECT_ROOT}/.venv311/bin/python"

echo "🚀 Stage2 Production Batch Processing"
echo "======================================"
echo "Target: 56,598 MIDI files"
echo "Output: Auto-generated with timestamp"
echo ""

# LAMDA統合オプション処理
LAMDA_OPTION=""
if [[ "$1" == "--with-lamda" ]]; then
    LAMDA_OPTION="--with-lamda"
    echo "🔗 LAMDA Integration: ENABLED"
    echo "  - KILO:       602MB (chords catalog)"
    echo "  - META:       4.1GB (5 shards)"
    echo "  - SIGNATURES: 290MB (timesig rescue)"
    echo "  - TOTALS:     33MB (outlier stats)"
    echo ""
else
    echo "ℹ️  LAMDA Integration: DISABLED (use --with-lamda to enable)"
    echo ""
fi

# バッチ処理実行
"$PYTHON" "${PROJECT_ROOT}/scripts/run_stage2_batch_production.py" \
    --input-dir "${PROJECT_ROOT}/output/stage1" \
    $LAMDA_OPTION

EXIT_CODE=$?

# 出力ディレクトリを特定（最新のstage2_production_*）
LATEST_OUTPUT=$(ls -td "${PROJECT_ROOT}"/output/stage2_production_* 2>/dev/null | head -1)

# 結果確認
echo ""
echo "======================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Production batch COMPLETED"
else
    echo "⚠️  Production batch completed with some failures (exit code: $EXIT_CODE)"
fi

echo ""
echo "📊 Results:"
if [ -n "$LATEST_OUTPUT" ]; then
    echo "  Output dir: $(basename "$LATEST_OUTPUT")"
    echo "  JSON files: $(find "$LATEST_OUTPUT/json" -name "*.json" 2>/dev/null | wc -l | xargs)"
    echo "  CSV: $(ls -lh "$LATEST_OUTPUT/stage2_aggregate.csv" 2>/dev/null | awk '{print $5}')"
else
    echo "  (Output directory not found)"
fi
echo "  Log: $(ls -t "${PROJECT_ROOT}"/logs/stage2_production_*.log 2>/dev/null | head -1 | xargs basename)"

exit $EXIT_CODE
