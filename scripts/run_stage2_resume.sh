#!/bin/bash
# Stage2 Resume処理: 残りの曲を処理

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
PYTHON="${PROJECT_ROOT}/.venv311/bin/python"

echo "🔄 Stage2 Resume Processing"
echo "======================================"

# 処理済みファイルリストを作成
RESUME_FILE="${PROJECT_ROOT}/logs/stage2_processed_files.txt"
find "${PROJECT_ROOT}/output/stage2_production/json" -name "*.stage2.json" 2>/dev/null | \
  xargs -n1 basename | \
  sed 's/.stage2.json$/.mid/' > "$RESUME_FILE"

PROCESSED=$(wc -l < "$RESUME_FILE" | xargs)
echo "Already processed: $PROCESSED files"
echo "Resume file: $RESUME_FILE"
echo ""

# Resume実行（--output-dir は同じパスを指定）
"$PYTHON" "${PROJECT_ROOT}/scripts/run_stage2_batch_production.py" \
    --input-dir "${PROJECT_ROOT}/output/stage1" \
    --output-dir "${PROJECT_ROOT}/output/stage2_production" \
    --log-file "${PROJECT_ROOT}/logs/stage2_resume.log" \
    --resume-from "$RESUME_FILE"

EXIT_CODE=$?

echo ""
echo "======================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Resume batch COMPLETED"
else
    echo "⚠️  Resume batch completed with some failures"
fi

echo ""
echo "📊 Total Results:"
echo "  JSON files: $(find "${PROJECT_ROOT}/output/stage2_production/json" -name "*.json" 2>/dev/null | wc -l | xargs)"
echo "  CSV: $(ls -lh "${PROJECT_ROOT}/output/stage2_production/stage2_aggregate.csv" 2>/dev/null | awk '{print $5}')"
echo "  Log: logs/stage2_resume.log"

exit $EXIT_CODE
