#!/bin/bash
# ========================================
# LAMDA (Los-Angeles-MIDI) クリーニング進捗モニター
# 楽器別進捗を一覧表示 (Pickle v2対応)
# ========================================

# BASE_DIR自動解決（Git root → スクリプト相対）
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
if [ -z "${BASE_DIR}" ]; then
  if command -v git >/dev/null 2>&1; then
    BASE_DIR="$(git -C "${SCRIPT_DIR}/.." rev-parse --show-toplevel 2>/dev/null || echo "")"
  fi
  : "${BASE_DIR:=${SCRIPT_DIR}/..}"
fi

# EXPECTED_TOTALの自動読込（環境変数 or ファイルから）
: "${EXPECTED_TOTAL:=}"
if [ -z "${EXPECTED_TOTAL}" ] && [ -f "${BASE_DIR}/data/lamda_expected_total.txt" ]; then
  EXPECTED_TOTAL="$(cat "${BASE_DIR}/data/lamda_expected_total.txt" 2>/dev/null || echo "")"
fi
# デフォルト値（ファイルが無い場合）
: "${EXPECTED_TOTAL:=404714}"

echo "📊 LAMDA Multi-Instrument Cleaning Monitor"
echo "================================================"
echo "Expected Total Files: ${EXPECTED_TOTAL}"
echo ""

while true; do
    clear
    
    echo "📊 LAMDA Multi-Instrument Cleaning Monitor (Pickle v2)"
    echo "================================================"
    echo "Time: $(date '+%Y-%m-%d %H:%M:%S')"
    echo ""
    
    # 各楽器の処理状況
    for INST in piano strings guitar bass drums; do
        INST_UPPER=$(echo $INST | tr '[:lower:]' '[:upper:]')
        CLEAN_DIR="${BASE_DIR}/data/cleaned/lamda_${INST}"
        QUAR_DIR="${BASE_DIR}/data/quarantine/lamda_${INST}"
        PICKLE_DIR="${BASE_DIR}/data/lamda_${INST}_metadata"
        
        CLEANED=$(find "${CLEAN_DIR}" -type f \( -name "*.mid" -o -name "*.midi" \) 2>/dev/null | wc -l | tr -d ' ')
        QUARANTINED=$(find "${QUAR_DIR}" -type f \( -name "*.mid" -o -name "*.midi" \) 2>/dev/null | wc -l | tr -d ' ')
        TOTAL=$((CLEANED + QUARANTINED))
        
        if [ ${TOTAL} -gt 0 ]; then
            CLEANED_PCT=$(awk "BEGIN {printf \"%.1f\", (${CLEANED}/${TOTAL})*100}")
            echo "🎹 ${INST_UPPER}:"
            echo "   Total: ${TOTAL}  |  ✅ Cleaned: ${CLEANED} (${CLEANED_PCT}%)  |  🗑️  Quarantined: ${QUARANTINED}"
            
            # Pickle インデックスチェック
            INDEX_FILE="${PICKLE_DIR}/${INST}_metadata_v2.pickle"
            if [ -f "${INDEX_FILE}" ]; then
                echo "   📦 Pickle Index: ✅ READY"
            else
                SHARD_COUNT=$(find "${PICKLE_DIR}" -name "${INST}_shard_*.pickle" 2>/dev/null | wc -l | tr -d ' ')
                if [ "${SHARD_COUNT}" -gt 0 ]; then
                    echo "   📦 Pickle Shards: ${SHARD_COUNT} (Index pending)"
                else
                    echo "   📦 Pickle: ⏳ Pending"
                fi
            fi
        else
            echo "🎹 ${INST_UPPER}: ⏳ Not started or no files processed"
        fi
        echo ""
    done
    
    echo "────────────────────────────────────────"
    
    # 最新のログファイル（パターン拡張: LAMDA_*, clean_*, etc）
    LATEST_LOG=$(ls -t "${BASE_DIR}/logs/"clean_LAMDA_*.log "${BASE_DIR}/logs/"*LAMDA*.log 2>/dev/null | head -1)
    if [ -n "${LATEST_LOG}" ]; then
        echo "📝 Latest Log: $(basename ${LATEST_LOG})"
        echo "Last 3 lines:"
        tail -3 "${LATEST_LOG}" 2>/dev/null | sed 's/^/  /'
    else
        echo "📝 No log files found yet"
    fi
    
    echo ""
    echo "────────────────────────────────────────"
    echo "Press Ctrl+C to exit | Updates every 10s"
    echo ""
    
    # 10秒待機
    sleep 10
done
