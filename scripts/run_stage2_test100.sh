#!/bin/bash
# スモールテスト: 最初の100曲だけ処理

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
PYTHON="${PROJECT_ROOT}/.venv311/bin/python"

echo "🧪 Stage2 Small Test (100 files)"
echo "================================="

# テスト用の一時ディレクトリ
TEST_INPUT="${PROJECT_ROOT}/output/stage1_test100"
TEST_OUTPUT="${PROJECT_ROOT}/output/stage2_test100"

# クリーンアップ
rm -rf "$TEST_INPUT" "$TEST_OUTPUT"
mkdir -p "$TEST_INPUT"

# Pop909から100曲コピー
echo "📂 Copying 100 files from Pop909..."
find "${PROJECT_ROOT}/output/stage1/pop909/clean" -name "*.mid" | head -100 | while read f; do
    cp "$f" "$TEST_INPUT/"
done

COPIED=$(find "$TEST_INPUT" -name "*.mid" | wc -l | xargs)
echo "✓ Copied $COPIED files"

# バッチ処理実行
echo ""
echo "🚀 Running Stage2 batch processing..."
"$PYTHON" "${PROJECT_ROOT}/scripts/run_stage2_batch_production.py" \
    --input-dir "$TEST_INPUT" \
    --output-dir "$TEST_OUTPUT" \
    --log-file "${PROJECT_ROOT}/logs/stage2_test100.log"

EXIT_CODE=$?

# 結果確認
echo ""
echo "================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Small test PASSED"
    echo ""
    echo "📊 Results:"
    echo "  JSON files: $(find "$TEST_OUTPUT/json" -name "*.json" | wc -l | xargs)"
    echo "  CSV: $(ls -lh "$TEST_OUTPUT/stage2_aggregate.csv" 2>/dev/null | awk '{print $5}')"
    echo "  Log: logs/stage2_test100.log"
else
    echo "❌ Small test FAILED (exit code: $EXIT_CODE)"
    echo "  Check logs/stage2_test100.log for details"
fi

exit $EXIT_CODE
