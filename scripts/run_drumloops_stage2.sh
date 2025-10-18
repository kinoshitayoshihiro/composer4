#!/bin/bash
# ========================================
# Drumloops Stage2 処理スクリプト
# 5軸スコアリング（Timing/Velocity/Groove/Cohesion/Structure）
# Stage2互換 Pickle入力対応版
# ========================================

set -euo pipefail

BASE_DIR="/Volumes/SSD-SCTU3A/ラジオ用/music_21/composer2-3"
LOG_FILE="${BASE_DIR}/logs/drumloops_stage2_$(date +%Y%m%d_%H%M%S).log"

# ログディレクトリ作成
mkdir -p "${BASE_DIR}/logs"

# エラートラップ
trap 'echo "[ERROR] line $LINENO: command \"${BASH_COMMAND}\" failed" | tee -a "${LOG_FILE}"' ERR

# ログ関数
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "${LOG_FILE}"
}

log "========================================="
log "Drumloops Stage2 Processing Started"
log "Stage2互換 Pickle入力"
log "========================================="
log "Base Directory: ${BASE_DIR}"
log "Log File: ${LOG_FILE}"
log ""

cd "${BASE_DIR}"

# 仮想環境アクティベート
source .venv311/bin/activate

# パス設定
META_DIR="output/drums_metadata"
META_INDEX="${META_DIR}/drums_index.pkl"
INPUT_DIR="output/drumloops_v3"
OUT_DIR="output/drumloops_v3_stage2"
CFG="configs/lamda/drums_stage2.yaml"
THRESHOLD="70.0"

# 必須ファイル/ディレクトリの存在確認
[[ -d "${META_DIR}" ]] || { log "❌ Missing metadata dir: ${META_DIR}"; exit 1; }
[[ -f "${META_INDEX}" ]] || { log "❌ Missing index: ${META_INDEX}"; exit 1; }
[[ -f "${CFG}" ]] || { log "❌ Missing config: ${CFG}"; exit 1; }
[[ -d "${INPUT_DIR}" ]] || { log "⚠️  Input dir not found: ${INPUT_DIR}"; }

# Pickle互換性チェック
log "📋 Checking pickle compatibility..."
python verify_stage2_compat.py "${META_DIR}" 2>&1 | tee -a "${LOG_FILE}" || {
    log "⚠️  Pickle compatibility check failed"
    log "   Please ensure clean_midi.py --pickle-out was used"
    exit 1
}

# クリーニング済みファイル数確認（厳密な括弧付き）
CLEANED_COUNT=$(find "${INPUT_DIR}" \( -name "*.mid" -o -name "*.midi" \) -type f 2>/dev/null | wc -l | tr -d ' ')
PICKLE_COUNT=$(find "${META_DIR}" -name "drums_*.pkl" -not -name "*_index.pkl" -type f 2>/dev/null | wc -l | tr -d ' ')

log ""
log "📊 Starting Stage2 processing..."
log "   Cleaned Files:    ${CLEANED_COUNT}"
log "   Pickle Shards:    ${PICKLE_COUNT}"
log "   Metadata Index:   ${META_INDEX}"
log "   Input Dir:        ${INPUT_DIR}"
log "   Output:           ${OUT_DIR}"
log "   Config:           ${CFG}"
log "   Threshold:        ${THRESHOLD}"
log ""

# 出力ディレクトリ準備
mkdir -p "${OUT_DIR}"

# 開始時刻記録
START_TIME=$(date +%s)

# Stage2処理実行（Pickle入力）
log "🎯 Executing Stage2 extractor with pickle input..."

PYTHONPATH=. python scripts/lamda_stage2_extractor.py \
    --metadata-index "${META_INDEX}" \
    --metadata-dir "${META_DIR}" \
    --input-dir "${INPUT_DIR}" \
    --output-dir "${OUT_DIR}" \
    --config "${CFG}" \
    --threshold "${THRESHOLD}" \
    --print-summary \
    2>&1 | tee -a "${LOG_FILE}"

# 終了時刻記録
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
HOURS=$((ELAPSED / 3600))
MINUTES=$(((ELAPSED % 3600) / 60))
SECONDS=$((ELAPSED % 60))

log ""
log "========================================="
log "✅ Stage2 Processing Completed!"
log "========================================="
log "Elapsed Time: ${HOURS}h ${MINUTES}m ${SECONDS}s"
log ""

# 統計表示
if [ -f "${OUT_DIR}/stage2_summary.json" ]; then
    log "📊 Summary generated: ${OUT_DIR}/stage2_summary.json"
fi

if [ -f "${OUT_DIR}/loop_summary.csv" ]; then
    LOOP_COUNT=$(wc -l < "${OUT_DIR}/loop_summary.csv")
    log "📄 Loop summary: ${LOOP_COUNT} entries"
fi

if [ -f "${OUT_DIR}/metrics_score.jsonl" ]; then
    SCORE_COUNT=$(wc -l < "${OUT_DIR}/metrics_score.jsonl")
    log "🎯 Scores generated: ${SCORE_COUNT} loops"
fi

log ""
log "Output files:"
log "  - ${OUT_DIR}/canonical_events.parquet"
log "  - ${OUT_DIR}/loop_summary.csv"
log "  - ${OUT_DIR}/metrics_score.jsonl"
log "  - ${OUT_DIR}/stage2_summary.json"
log ""
