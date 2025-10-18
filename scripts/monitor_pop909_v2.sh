#!/bin/bash
# ========================================
# POP909 クリーニング進捗モニター (LAMDA Pickle対応版)
# ========================================

BASE_DIR="/Volumes/SSD-SCTU3A/ラジオ用/music_21/composer2-3"
EXPECTED_TOTAL=2898

echo "📊 POP909 Cleaning Progress Monitor (LAMDA-Compatible)"
echo "========================================================"
echo ""

while true; do
    # クリア画面
    clear
    
    echo "📊 POP909 Cleaning Progress Monitor (LAMDA-Compatible)"
    echo "========================================================"
    echo "Time: $(date '+%Y-%m-%d %H:%M:%S')"
    echo ""
    
    # ファイル数カウント
    CLEANED=$(find "${BASE_DIR}/data/cleaned/pop909" -name "*.mid" 2>/dev/null | wc -l | tr -d ' ')
    QUARANTINED=$(find "${BASE_DIR}/data/quarantine/pop909" -name "*.mid" 2>/dev/null | wc -l | tr -d ' ')
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
    BAR=$(printf "%-50s" "$(printf '#%.0s' $(seq 1 $PROGRESS))")
    
    echo "Progress: [${BAR}] ${PCT}%"
    echo ""
    echo "Files Processed: ${TOTAL} / ${EXPECTED_TOTAL}"
    echo "  ✅ Cleaned:      ${CLEANED} (${CLEANED_PCT}%)"
    echo "  🗑️  Quarantined:  ${QUARANTINED} (${QUARANTINE_PCT}%)"
    echo ""
    
    # メタデータインデックス (個別JSON)
    if [ -f "${BASE_DIR}/data/cleaned/pop909/meta_index.jsonl" ]; then
        INDEX_LINES=$(wc -l < "${BASE_DIR}/data/cleaned/pop909/meta_index.jsonl" 2>/dev/null | tr -d ' ')
        echo "📄 Metadata Index (JSONL): ${INDEX_LINES} entries"
    fi
    
    # LAMDA Pickle出力チェック
    METADATA_DIR="${BASE_DIR}/data/piano_metadata"
    if [ -d "${METADATA_DIR}" ]; then
        SHARD_COUNT=$(find "${METADATA_DIR}" -name "piano_shard_*.pickle" 2>/dev/null | wc -l | tr -d ' ')
        if [ -f "${METADATA_DIR}/piano_metadata_index.pickle" ]; then
            echo "🎯 LAMDA Pickle Output:"
            echo "   ✅ Index:  piano_metadata_index.pickle"
            echo "   📦 Shards: ${SHARD_COUNT} files"
            
            # Pickle内のループ数を取得（Pythonで）
            LOOPS_COUNT=$(python3 -c "
import pickle, sys
try:
    with open('${METADATA_DIR}/piano_metadata_index.pickle', 'rb') as f:
        data = pickle.load(f)
        print(data.get('total_count', 0))
except:
    print(0)
" 2>/dev/null)
            
            if [ "${LOOPS_COUNT}" != "0" ]; then
                echo "   🔢 Loops:  ${LOOPS_COUNT} total"
            fi
        else
            echo "🎯 LAMDA Pickle Output:"
            echo "   ⏳ Generating... (${SHARD_COUNT} shards found)"
        fi
    else
        echo "🎯 LAMDA Pickle Output: Not yet created"
    fi
    echo ""
    
    # 最新のログファイル
    LATEST_LOG=$(ls -t "${BASE_DIR}/logs/pop909_cleaning_"*.log 2>/dev/null | head -1)
    if [ -n "${LATEST_LOG}" ]; then
        echo "📝 Latest Log: $(basename ${LATEST_LOG})"
        echo "Last 5 lines:"
        echo "----------------------------------------"
        tail -5 "${LATEST_LOG}" 2>/dev/null | sed 's/^/  /'
        echo "----------------------------------------"
    fi
    
    echo ""
    echo "Press Ctrl+C to exit monitor"
    echo ""
    
    # 完了チェック
    if [ ${TOTAL} -ge ${EXPECTED_TOTAL} ]; then
        echo "🎉 Processing Complete!"
        
        # 最終統計表示
        if [ -f "${METADATA_DIR}/piano_metadata_index.pickle" ]; then
            echo ""
            echo "✅ LAMDA Pickle Generation: COMPLETE"
            echo "   Ready for Stage 2 processing with lamda_stage2_extractor.py"
        fi
        break
    fi
    
    # 10秒待機
    sleep 10
done
