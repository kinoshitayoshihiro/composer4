#!/usr/bin/env bash
################################################################################
# run_stage2_multi.sh
#
# Stage2: 複数データセット一括スコアリング & 選抜（シャード分割対応）
#
# 使用例:
#   bash scripts/run_stage2_multi.sh
#
# 環境変数でカスタマイズ:
#   LIMIT=10000 TH_SOFT=60 TH_HARD=75 bash scripts/run_stage2_multi.sh
################################################################################
set -euo pipefail

# ▼共通デフォルト（必要に応じて環境変数で上書き）
LIMIT="${LIMIT:-5000}"                        # 1バッチあたり件数（省メモリ）
TH_SOFT="${TH_SOFT:-65.0}"                   # soft: 学習母集団確保
TH_HARD="${TH_HARD:-70.0}"                   # hard: 公開/厳選用
ROW_GROUP="${ROW_GROUP:-8192}"               # Parquet 行グループ
MANIFEST_FLUSH="${MANIFEST_FLUSH:-200}"      # Manifest フラッシュ間隔
RESUME_FLAG="${RESUME_FLAG:---resume}"       # 冪等実行
STREAMING_FLAG="${STREAMING_FLAG:---streaming}"   # 逐次書き出し

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║                                                              ║"
echo "║  Stage2: Multi-Dataset Scoring & Selection                  ║"
echo "║                                                              ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "Settings:"
echo "  LIMIT            = ${LIMIT}"
echo "  TH_SOFT          = ${TH_SOFT}"
echo "  TH_HARD          = ${TH_HARD}"
echo "  ROW_GROUP        = ${ROW_GROUP}"
echo "  MANIFEST_FLUSH   = ${MANIFEST_FLUSH}"
echo "  STREAMING_FLAG   = ${STREAMING_FLAG}"
echo "  RESUME_FLAG      = ${RESUME_FLAG}"
echo ""

# ▼データセット定義（TSV形式）
# カラム: name instrument input_dir out_dir meta_dir meta_index_pkl config_yaml
# 注意: Stage2は現在 drums専用です。Guitar/Bass/Strings用のStage2メトリクスは別途実装予定。
DATASETS="$(cat <<'EOF'
SLAKH	drums	output/slakh/clean/drums	output/slakh/stage2/drums	output/slakh_metadata	output/slakh_metadata/drums_index.pkl	configs/lamda/drums_stage2.yaml
LAMDA	drums	output/lamda/clean/drumloops	output/lamda/stage2/drumloops	output/drums_metadata	output/drums_metadata/drums_index.pkl	configs/lamda/drums_stage2.yaml
EOF
)"
# 例：Guitar/Bass/Strings用を将来追加予定
# SLAKH	guitar	output/slakh/clean/guitar	output/slakh/stage2/guitar	output/guitar_metadata	output/guitar_metadata/index.pkl	configs/lamda/guitar_stage2.yaml

# macOSの古いbash対応のため find で総数算出
count_mid() {
  local d="$1"
  if [[ ! -d "${d}" ]]; then
    echo "0"
    return
  fi
  find "$d" -type f \( -iname "*.mid" -o -iname "*.midi" \) 2>/dev/null | wc -l | tr -d ' '
}

run_one() {
  local NAME="$1" INSTR="$2" IN_DIR="$3" OUT_DIR="$4" META_DIR="$5" META_IDX="$6" CFG="$7"

  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "== Stage2: ${NAME}/${INSTR} =="
  echo "   Input:        ${IN_DIR}"
  echo "   Output:       ${OUT_DIR}"
  echo "   Metadata:     ${META_DIR}"
  echo "   Index:        ${META_IDX}"
  echo "   Config:       ${CFG}"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

  # メタデータインデックスが存在しない場合はスキップ
  if [[ ! -f "${META_IDX}" ]]; then
    echo "⚠️  Metadata index not found: ${META_IDX}"
    echo "   Stage1を先に実行してください。"
    echo ""
    return
  fi

  # 入力ディレクトリが存在しない場合はスキップ
  if [[ ! -d "${IN_DIR}" ]]; then
    echo "⚠️  Input directory not found: ${IN_DIR}"
    echo ""
    return
  fi

  mkdir -p "${OUT_DIR}"

  local TOTAL; TOTAL=$(count_mid "${IN_DIR}")
  echo "Total candidates in ${IN_DIR}: ${TOTAL}"

  if [[ ${TOTAL} -eq 0 ]]; then
    echo "⚠️  No MIDI files found. Skipping."
    echo ""
    return
  fi

  local OFFSET=0
  local BATCH_NUM=0
  while [[ ${OFFSET} -lt ${TOTAL} ]]; do
    echo ""
    echo "-- Batch ${BATCH_NUM}: OFFSET=${OFFSET}, LIMIT=${LIMIT}"
    
    python3 scripts/lamda_stage2_extractor.py \
      --metadata-index "${META_IDX}" \
      --metadata-dir "${META_DIR}" \
      --input-dir "${IN_DIR}" \
      --output-dir "${OUT_DIR}/batch_${OFFSET}" \
      --config "${CFG}" \
      --threshold-soft "${TH_SOFT}" \
      --threshold-hard "${TH_HARD}" \
      --limit "${LIMIT}" \
      --offset "${OFFSET}" \
      ${STREAMING_FLAG} \
      ${RESUME_FLAG} \
      --parquet-row-group "${ROW_GROUP}" \
      --manifest-flush-n "${MANIFEST_FLUSH}" \
      --print-summary || {
        echo "⚠️  Batch ${BATCH_NUM} failed, continuing..."
      }

    OFFSET=$((OFFSET + LIMIT))
    BATCH_NUM=$((BATCH_NUM + 1))
  done

  echo ""
  echo "-- Merging results: ${OUT_DIR}"
  
  # metrics_score.jsonl を結合
  if find "${OUT_DIR}" -type f -name 'metrics_score.jsonl' 2>/dev/null | grep -q .; then
    find "${OUT_DIR}" -type f -name 'metrics_score.jsonl' -print0 \
      | xargs -0 cat > "${OUT_DIR}/metrics_score.ALL.jsonl"
    
    local SOFT_COUNT HARD_COUNT TOTAL_COUNT
    TOTAL_COUNT=$(wc -l < "${OUT_DIR}/metrics_score.ALL.jsonl" | tr -d ' ')
    SOFT_COUNT=$(grep -c '"pass_soft":true' "${OUT_DIR}/metrics_score.ALL.jsonl" || echo "0")
    HARD_COUNT=$(grep -c '"pass_hard":true' "${OUT_DIR}/metrics_score.ALL.jsonl" || echo "0")
    
    echo "   Merged metrics: ${TOTAL_COUNT} loops"
    echo "   Passed (soft): ${SOFT_COUNT} loops"
    echo "   Passed (hard): ${HARD_COUNT} loops"
  else
    echo "   ⚠️  No metrics files found to merge"
  fi

  # loop_summary.csv を結合（ヘッダー重複削除）
  if find "${OUT_DIR}" -type f -name 'loop_summary.csv' 2>/dev/null | grep -q .; then
    {
      find "${OUT_DIR}" -type f -name 'loop_summary.csv' -print0 | sort -z | head -z -1 | xargs -0 head -1
      find "${OUT_DIR}" -type f -name 'loop_summary.csv' -print0 | xargs -0 tail -q -n +2
    } > "${OUT_DIR}/loop_summary.ALL.csv"
    echo "   Merged CSV summary"
  fi

  # Parquet は必要に応じて pyarrow で結合（TODO: 将来実装）
  # ここは strings/drums 共通マージの拡張点

  echo ""
  echo "✅ ${NAME}/${INSTR} 完了"
  echo ""
}

# ループ実行
while IFS=$'\t' read -r NAME INSTR IN_DIR OUT_DIR META_DIR META_IDX CFG || [[ -n "${NAME:-}" ]]; do
  # 空行やコメント行はスキップ
  [[ -z "${NAME:-}" || "${NAME:0:1}" == "#" ]] && continue
  run_one "${NAME}" "${INSTR}" "${IN_DIR}" "${OUT_DIR}" "${META_DIR}" "${META_IDX}" "${CFG}"
done <<<"${DATASETS}"

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║                                                              ║"
echo "║  🎉 Stage2 全データセット処理完了! 🎉                       ║"
echo "║                                                              ║"
echo "╚══════════════════════════════════════════════════════════════╝"
