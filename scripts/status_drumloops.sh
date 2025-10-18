#!/bin/bash
# drumloops クリーニング進捗確認スクリプト
# Stage2互換 Pickle直書き対応版

BASE_DIR="/Volumes/SSD-SCTU3A/ラジオ用/music_21/composer2-3"
EXPECTED_TOTAL=77346

echo "🥁 Drumloops Cleaning Status"
echo "Stage2互換 Pickle直書き方式"
echo "==========================="
echo ""

# プロセス確認
if ps aux | grep "clean_midi.py.*drums" | grep -v grep > /dev/null; then
    echo "✅ Status: RUNNING"
    PID=$(ps aux | grep "clean_midi.py.*drums" | grep -v grep | awk '{print $2}')
    echo "   PID: ${PID}"
else
    echo "⏹️  Status: NOT RUNNING"
fi

echo ""

# ファイル数
CLEANED=$(find "${BASE_DIR}/output/drumloops_v3" -name "*.mid" -o -name "*.midi" 2>/dev/null | wc -l | tr -d ' ')
QUARANTINED=$(find "${BASE_DIR}/output/drumloops_v3_q" -name "*.mid" -o -name "*.midi" 2>/dev/null | wc -l | tr -d ' ')
PICKLE_SHARDS=$(find "${BASE_DIR}/output/drums_metadata" -name "drums_*.pkl" 2>/dev/null | wc -l | tr -d ' ')
TOTAL=$((CLEANED + QUARANTINED))

echo "📊 Progress: ${TOTAL} / ${EXPECTED_TOTAL} files"
echo "   ✅ Cleaned:      ${CLEANED}"
echo "   �️  Quarantined:  ${QUARANTINED}"
echo "   � Pickle Shards: ${PICKLE_SHARDS}"

if [ ${TOTAL} -gt 0 ]; then
    PCT=$(awk "BEGIN {printf \"%.1f\", (${TOTAL}/${EXPECTED_TOTAL})*100}")
    RATE=$(awk "BEGIN {printf \"%.1f\", (${CLEANED}/${TOTAL})*100}")
    echo "   Progress:     ${PCT}%"
    echo "   Success Rate: ${RATE}%"
fi

echo ""

# Pickle互換性チェック
if [ -f "${BASE_DIR}/output/drums_metadata/drums_index.pkl" ]; then
    echo "📦 Pickle Status:"
    echo "   ✅ Index: drums_index.pkl exists"
    echo "   Shards: ${PICKLE_SHARDS} files"
    
    # 簡易検証
    if [ ${PICKLE_SHARDS} -gt 0 ]; then
        EXPECTED_SHARDS=$(awk "BEGIN {printf \"%.0f\", ${CLEANED}/5000}")
        if [ ${PICKLE_SHARDS} -ge ${EXPECTED_SHARDS} ]; then
            echo "   ✅ Shard count looks good"
        else
            echo "   ⚠️  Expected ~${EXPECTED_SHARDS} shards, found ${PICKLE_SHARDS}"
        fi
    fi
else
    echo "📦 Pickle Status:"
    echo "   ⚠️  Index file not found yet"
fi

echo ""

# 最新ログ
LATEST_LOG=$(ls -t "${BASE_DIR}/logs/drumloops_cleaning_"* 2>/dev/null | head -1)
if [ -n "${LATEST_LOG}" ]; then
    echo "📝 Latest Log (last 5 lines):"
    tail -5 "${LATEST_LOG}" | sed 's/^/   /'
fi

echo ""
echo "Commands:"
echo "  Watch:    watch -n 5 ./scripts/status_drumloops.sh"
if [ -n "${LATEST_LOG}" ]; then
    echo "  Log:      tail -f ${LATEST_LOG}"
fi
if ps aux | grep "clean_midi.py.*drums" | grep -v grep > /dev/null; then
    echo "  Stop:     kill ${PID}"
fi
echo "  Verify:   python verify_stage2_compat.py output/drums_metadata"
