#!/bin/bash
# クイック進捗確認スクリプト (LAMDA Pickle対応版)

BASE_DIR="/Volumes/SSD-SCTU3A/ラジオ用/music_21/composer2-3"

echo "🎵 POP909 Cleaning Status (LAMDA-Compatible)"
echo "============================================="
echo ""

# プロセス確認
if ps aux | grep "run_pop909_full.sh" | grep -v grep > /dev/null; then
    echo "✅ Status: RUNNING"
    PID=$(ps aux | grep "run_pop909_full.sh" | grep -v grep | awk '{print $2}')
    echo "   PID: ${PID}"
else
    echo "⏹️  Status: NOT RUNNING"
fi

echo ""

# ファイル数
CLEANED=$(find "${BASE_DIR}/data/cleaned/pop909" -name "*.mid" 2>/dev/null | wc -l | tr -d ' ')
QUARANTINED=$(find "${BASE_DIR}/data/quarantine/pop909" -name "*.mid" 2>/dev/null | wc -l | tr -d ' ')
TOTAL=$((CLEANED + QUARANTINED))

echo "📊 Progress: ${TOTAL} / 2898 files"
echo "   ✅ Cleaned:      ${CLEANED}"
echo "   🗑️  Quarantined:  ${QUARANTINED}"

if [ ${TOTAL} -gt 0 ]; then
    PCT=$(awk "BEGIN {printf \"%.1f\", (${TOTAL}/2898)*100}")
    echo "   Progress:     ${PCT}%"
fi

echo ""

# LAMDA Pickle出力チェック
METADATA_DIR="${BASE_DIR}/data/piano_metadata"
if [ -d "${METADATA_DIR}" ]; then
    SHARD_COUNT=$(find "${METADATA_DIR}" -name "piano_shard_*.pickle" 2>/dev/null | wc -l | tr -d ' ')
    
    echo "🎯 LAMDA Pickle Output:"
    
    if [ -f "${METADATA_DIR}/piano_metadata_index.pickle" ]; then
        echo "   ✅ Status: COMPLETE"
        
        # Pickle統計を取得
        python3 -c "
import pickle
try:
    with open('${METADATA_DIR}/piano_metadata_index.pickle', 'rb') as f:
        data = pickle.load(f)
        print(f'   📦 Shards: {data.get(\"total_shards\", 0)}')
        print(f'   🔢 Loops:  {data.get(\"total_count\", 0)}')
        print(f'   🎹 Instrument: {data.get(\"instrument\", \"unknown\")}')
except Exception as e:
    print(f'   ❌ Error reading pickle: {e}')
" 2>/dev/null
    else
        if [ "${SHARD_COUNT}" -gt 0 ]; then
            echo "   ⏳ Status: GENERATING..."
            echo "   📦 Shards found: ${SHARD_COUNT}"
        else
            echo "   ⏳ Status: PENDING (waiting for MIDI processing to complete)"
        fi
    fi
    echo ""
else
    echo "🎯 LAMDA Pickle Output: Not yet created"
    echo ""
fi

# 最新ログ
LATEST_LOG=$(ls -t "${BASE_DIR}/logs/pop909_cleaning_"*.log 2>/dev/null | head -1)
if [ -n "${LATEST_LOG}" ]; then
    echo "📝 Latest Log:"
    tail -3 "${LATEST_LOG}" | sed 's/^/   /'
fi

echo ""
echo "Commands:"
echo "  Monitor:  ./scripts/monitor_pop909_v2.sh"
echo "  Log:      tail -f ${LATEST_LOG}"
if [ -n "${PID}" ]; then
    echo "  Stop:     kill ${PID}"
fi
echo ""

# Stage 2 準備状況
if [ -f "${METADATA_DIR}/piano_metadata_index.pickle" ]; then
    echo "✅ Ready for Stage 2:"
    echo "   python3 scripts/lamda_stage2_extractor.py \\"
    echo "     --metadata-index ${METADATA_DIR}/piano_metadata_index.pickle \\"
    echo "     --output data/piano_stage2_scored.jsonl"
fi
