#!/bin/bash
# クイック進捗確認スクリプト (Pickle v2対応)

BASE_DIR="/Volumes/SSD-SCTU3A/ラジオ用/music_21/composer2-3"
PICKLE_DIR="${BASE_DIR}/data/piano_metadata"

echo "🎵 POP909 Cleaning Status (Pickle v2)"
echo "====================================="
echo ""

# プロセス確認
if ps aux | grep "run_pop909_full.sh\|run_dataset_full.sh.*POP909" | grep -v grep > /dev/null; then
    echo "✅ Status: RUNNING"
    PID=$(ps aux | grep "run_pop909_full.sh\|run_dataset_full.sh.*POP909" | grep -v grep | awk '{print $2}' | head -1)
    echo "   PID: ${PID}"
else
    echo "⏹️  Status: NOT RUNNING"
    PID=""
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

# Pickle v2 インデックス状態
INDEX_FILE="${PICKLE_DIR}/piano_metadata_v2.pickle"
if [ -f "${INDEX_FILE}" ]; then
    echo "🎯 Pickle Index (v2): ✅ COMPLETE"
    
    # Pickle統計を取得
    python3 - <<'PY' 2>/dev/null || echo "   ⚠️  Could not read pickle"
import pickle
try:
    with open("data/piano_metadata/piano_metadata_v2.pickle", "rb") as f:
        idx = pickle.load(f)
    print(f"   Instrument:   {idx.get('instrument', 'N/A')}")
    print(f"   Total Files:  {idx.get('total_files', 0)}")
    print(f"   Shard Size:   {idx.get('shard_size', 0)}")
    print(f"   Shards:       {len(idx.get('shards', []))}")
except Exception as e:
    print(f"   ❌ Error: {e}")
PY
else
    SHARD_COUNT=$(find "${PICKLE_DIR}" -name "piano_shard_*.pickle" 2>/dev/null | wc -l | tr -d ' ')
    if [ "${SHARD_COUNT}" -gt 0 ]; then
        echo "🎯 Pickle Index (v2): ⏳ GENERATING..."
        echo "   Shards found: ${SHARD_COUNT}"
    else
        echo "🎯 Pickle Index (v2): ⏳ PENDING"
    fi
fi

echo ""

# 最新ログ
LATEST_LOG=$(ls -t "${BASE_DIR}/logs/clean_POP909_"*.log 2>/dev/null | head -1)
if [ -n "${LATEST_LOG}" ]; then
    echo "📝 Latest Log:"
    tail -3 "${LATEST_LOG}" | sed 's/^/   /'
fi

echo ""
echo "Commands:"
echo "  Monitor:  ./scripts/monitor_pop909.sh"
if [ -n "${LATEST_LOG}" ]; then
    echo "  Log:      tail -f ${LATEST_LOG}"
fi
if [ -n "${PID}" ]; then
    echo "  Stop:     kill ${PID}"
fi
echo ""

# Stage 2 準備状況
if [ -f "${INDEX_FILE}" ]; then
    echo "✅ Ready for Stage 2:"
    echo "   python scripts/lamda_stage2_extractor.py \\"
    echo "     --metadata-index ${INDEX_FILE} \\"
    echo "     --output data/piano_stage2_scored.jsonl"
fi
