#!/bin/bash
# ========================================
# Drumloops 完全クリーニングスクリプト
# Stage2互換 Pickle直書き対応版
# ========================================

set -e

BASE_DIR="/Volumes/SSD-SCTU3A/ラジオ用/music_21/composer2-3"
LOG_FILE="${BASE_DIR}/logs/drumloops_cleaning_$(date +%Y%m%d_%H%M%S).log"

# ログディレクトリ作成
mkdir -p "${BASE_DIR}/logs"

# ログ関数
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "${LOG_FILE}"
}

log "========================================="
log "Drumloops MIDI Cleaning Started"
log "Stage2互換 Pickle直書き方式"
log "========================================="
log "Base Directory: ${BASE_DIR}"
log "Log File: ${LOG_FILE}"
log ""

cd "${BASE_DIR}"

# 仮想環境アクティベート
source .venv311/bin/activate

log "🥁 Starting drumloops cleaning with clean_midi.py..."
log "   Input:         data/loops (77,346 files)"
log "   Output:        output/drumloops_v3"
log "   Quarantine:    output/drumloops_v3_q"
log "   Pickle Output: output/drums_metadata (Stage2互換)"
log "   Instrument:    drums"
log "   Shard Size:    5000"
log "   emit-meta-json: off (pickle直書き)"
log "   Resume:        enabled (SSD事故対応)"
log "   Jobs:          8 (並列処理)"
log ""

# ディレクトリ準備
mkdir -p output/drumloops_v3
mkdir -p output/drumloops_v3_q
mkdir -p output/drums_metadata

# 開始時刻記録
START_TIME=$(date +%s)

# Python スクリプト実行（Stage2互換 Pickle直書き）
PYTHONPATH=. python -m scripts.clean_midi \
    --in "${BASE_DIR}/data/loops" \
    --out "${BASE_DIR}/output/drumloops_v3" \
    --quarantine "${BASE_DIR}/output/drumloops_v3_q" \
    --instrument drums \
    --pickle-out "${BASE_DIR}/output/drums_metadata" \
    --shard-size 5000 \
    --resume \
    --emit-meta-json off \
    --jobs 8 \
    2>&1 | tee -a "${LOG_FILE}"

# 終了時刻記録
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
HOURS=$((ELAPSED / 3600))
MINUTES=$(((ELAPSED % 3600) / 60))
SECONDS=$((ELAPSED % 60))

log ""
log "========================================="
log "✅ Drumloops Cleaning Completed!"
log "========================================="
log "Elapsed Time: ${HOURS}h ${MINUTES}m ${SECONDS}s"
log ""

# 統計表示
CLEANED=$(find output/drumloops_v3 -name "*.mid" -o -name "*.midi" 2>/dev/null | wc -l | tr -d ' ')
QUARANTINED=$(find output/drumloops_v3_q -name "*.mid" -o -name "*.midi" 2>/dev/null | wc -l | tr -d ' ')
PICKLE_COUNT=$(find output/drums_metadata -name "*.pkl" 2>/dev/null | wc -l | tr -d ' ')
TOTAL=$((CLEANED + QUARANTINED))

log "📊 Final Statistics:"
log "   Total Processed:  ${TOTAL} / 77,346"
log "   ✅ Cleaned:       ${CLEANED}"
log "   �️  Quarantined:   ${QUARANTINED}"
log "   � Pickle Files:  ${PICKLE_COUNT}"

if [ ${TOTAL} -gt 0 ]; then
    SUCCESS_RATE=$(awk "BEGIN {printf \"%.2f\", (${CLEANED}/${TOTAL})*100}")
    log "   Success Rate:  ${SUCCESS_RATE}%"
fi

log ""
log "📦 Pickle Output (Stage2互換):"
log "   Directory: output/drums_metadata"
if [ -f output/drums_metadata/drums_index.pkl ]; then
    log "   ✅ Index: drums_index.pkl"
fi
log "   Shards: ${PICKLE_COUNT} files"
log ""
log "Next step: Run Stage2 processing"
log "  Command: ./scripts/run_drumloops_stage2.sh"
log ""
