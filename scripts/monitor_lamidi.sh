#!/bin/bash
# ========================================
# Los-Angeles-MIDI クリーニング進捗モニター (Pickle v2対応)
# ========================================

BASE_DIR="/Volumes/SSD-SCTU3A/ラジオ用/music_21/composer2-3"
# Los-Angeles-MIDIの総ファイル数は実行前に確認が必要
# とりあえず仮の値を設定（実際のファイル数に後で調整）
EXPECTED_TOTAL=10000
PICKLE_DIR="${BASE_DIR}/data/lamidi_metadata"

echo "📊 Los-Angeles-MIDI Cleaning Progress Monitor (Pickle v2)"
echo "================================================"
echo ""

while true; do
    clear
    
    echo "📊 Los-Angeles-MIDI Cleaning Progress Monitor (Pickle v2)"
    echo "================================================"
    echo "Time: $(date '+%Y-%m-%d %H:%M:%S')"
    echo ""
    
    # ファイル数カウント
    CLEANED=$(find "${BASE_DIR}/data/cleaned/lamidi" -name "*.mid" 2>/dev/null | wc -l | tr -d ' ')
    QUARANTINED=$(find "${BASE_DIR}/data/quarantine/lamidi" -name "*.mid" 2>/dev/null | wc -l | tr -d ' ')
    TOTAL=$((CLEANED + QUARANTINED))
    
    # パーセンテージ計算
    if [ ${TOTAL} -gt 0 ]; then
        PCT=$(awk "BEGIN {printf \"%.1f\", (${TOTAL}/${EXPECTED_TOTAL})*100}")
        CLEANED_PCT=$(awk "BEGIN {printf \"%.1f\", (${CLEANED}/${TOTAL})*100}")
        QUARANTINE_PCT=$(awk "BEGIN {printf \"%.1f\", (${QUARANTINED}/${TOTAL})*100}")
    else
        PCT="0.0"
        CLEANED_PCT="0.0"
        QUARANTINE_PCT="0.0"
    fi
    
    # プログレスバー表示
    PROGRESS=$((TOTAL * 50 / EXPECTED_TOTAL))
    if [ ${PROGRESS} -gt 50 ]; then
        PROGRESS=50
    fi
    if [ ${PROGRESS} -gt 0 ]; then
        BAR=$(printf "%-50s" "$(printf '#%.0s' $(seq 1 $PROGRESS))")
    else
        BAR=$(printf "%-50s" "")
    fi
    
    echo "Progress: [${BAR}] ${PCT}%"
    echo ""
    echo "📁 MIDI Files: ${TOTAL} / ${EXPECTED_TOTAL}"
    echo "   ✅ Cleaned:      ${CLEANED} (${CLEANED_PCT}%)"
    echo "   🗑️  Quarantined:  ${QUARANTINED} (${QUARANTINE_PCT}%)"
    echo ""
    
    # Pickle v2 インデックスチェック
    INDEX_FILE="${PICKLE_DIR}/piano_metadata_v2.pickle"
    if [ -f "${INDEX_FILE}" ]; then
        echo "🎯 Pickle Index (v2): ✅ FOUND"
        
        # Python で pickle 情報を取得
        python3 - <<'PY' 2>/dev/null || echo "   ⚠️  Could not read pickle"
import pickle
try:
    with open("data/lamidi_metadata/piano_metadata_v2.pickle", "rb") as f:
        idx = pickle.load(f)
    print(f"   Instrument:   {idx.get('instrument', 'N/A')}")
    print(f"   Total Files:  {idx.get('total_files', 0)}")
    print(f"   Shard Size:   {idx.get('shard_size', 0)}")
    print(f"   Shards:       {len(idx.get('shards', []))}")
except Exception as e:
    print(f"   ❌ Error: {e}")
PY
    else
        # シャードファイル数をカウント
        SHARD_COUNT=$(find "${PICKLE_DIR}" -name "piano_shard_*.pickle" 2>/dev/null | wc -l | tr -d ' ')
        if [ "${SHARD_COUNT}" -gt 0 ]; then
            echo "🎯 Pickle Index (v2): ⏳ GENERATING..."
            echo "   Shards found: ${SHARD_COUNT}"
        else
            echo "🎯 Pickle Index (v2): ⏳ PENDING"
        fi
    fi
    echo ""
    
    # 最新のログファイル
    LATEST_LOG=$(ls -t "${BASE_DIR}/logs/clean_LAMIDI_"*.log 2>/dev/null | head -1)
    if [ -n "${LATEST_LOG}" ]; then
        echo "📝 Latest Log: $(basename ${LATEST_LOG})"
        echo "Last 5 lines:"
        echo "────────────────────────────────────────"
        tail -5 "${LATEST_LOG}" 2>/dev/null | sed 's/^/  /'
        echo "────────────────────────────────────────"
    fi
    
    echo ""
    echo "Press Ctrl+C to exit"
    echo ""
    
    # 完了チェック
    if [ ${TOTAL} -ge ${EXPECTED_TOTAL} ] && [ -f "${INDEX_FILE}" ]; then
        echo "🎉 Processing Complete!"
        echo ""
        echo "✅ Ready for Stage 2:"
        echo "   python scripts/lamda_stage2_extractor.py \\"
        echo "     --metadata-index ${INDEX_FILE} \\"
        echo "     --output data/lamidi_stage2_scored.jsonl"
        break
    fi
    
    # 10秒待機
    sleep 10
done
