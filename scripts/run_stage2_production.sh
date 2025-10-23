#!/bin/bash
# 本番Stage2バッチ処理: 全55,640曲

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
PYTHON="${PROJECT_ROOT}/.venv311/bin/python"

echo "🚀 Stage2 Production Batch Processing"
echo "======================================"
echo "Target: 55,640 MIDI files"
echo ""

# バッチ処理実行
"$PYTHON" "${PROJECT_ROOT}/scripts/run_stage2_batch_production.py" \
    --input-dir "${PROJECT_ROOT}/output/stage1" \
    --output-dir "${PROJECT_ROOT}/output/stage2_production" \
    --log-file "${PROJECT_ROOT}/logs/stage2_production.log"

EXIT_CODE=$?

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
echo "  JSON files: $(find "${PROJECT_ROOT}/output/stage2_production/json" -name "*.json" 2>/dev/null | wc -l | xargs)"
echo "  CSV: $(ls -lh "${PROJECT_ROOT}/output/stage2_production/stage2_aggregate.csv" 2>/dev/null | awk '{print $5}')"
echo "  Log: logs/stage2_production.log"

exit $EXIT_CODE
