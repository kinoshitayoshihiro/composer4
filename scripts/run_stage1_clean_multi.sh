#!/usr/bin/env bash
################################################################################
# run_stage1_clean_multi.sh
#
# Stage1: 複数データセット一括クリーニング & sharded pickle 生成
#
# 使用例:
#   bash scripts/run_stage1_clean_multi.sh
#
# 環境変数でカスタマイズ:
#   SHARD_SIZE=10000 JOBS=4 RESUME_FLAG="" bash scripts/run_stage1_clean_multi.sh
################################################################################
set -euo pipefail

# ▼共通デフォルト
SHARD_SIZE="${SHARD_SIZE:-5000}"
JOBS="${JOBS:-8}"
EMIT_META_JSON="${EMIT_META_JSON:-off}"   # JSONを残さず pickle 直書き
RESUME_FLAG="${RESUME_FLAG:---resume}"    # 冪等実行

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║                                                              ║"
echo "║  Stage1: Multi-Dataset Cleaning & Sharded Pickle 生成       ║"
echo "║                                                              ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "Settings:"
echo "  SHARD_SIZE       = ${SHARD_SIZE}"
echo "  JOBS             = ${JOBS}"
echo "  EMIT_META_JSON   = ${EMIT_META_JSON}"
echo "  RESUME_FLAG      = ${RESUME_FLAG}"
echo ""

# ▼データセット定義（TSV形式）
# カラム: name instrument raw_in clean_out quarantine_out pickle_out
DATASETS="$(cat <<'EOF'
POP909	melody	data/POP909	output/pop909/clean/melody	output/pop909/quarantine/melody	output/pop909/shards/melody
SLAKH	drums	data/slakh2100_midi/drums	output/slakh/clean/drums	output/slakh/quarantine/drums	output/slakh/shards/drums
SLAKH	guitar	data/slakh2100_midi/guitar	output/slakh/clean/guitar	output/slakh/quarantine/guitar	output/slakh/shards/guitar
SLAKH	bass	data/slakh2100_midi/bass	output/slakh/clean/bass	output/slakh/quarantine/bass	output/slakh/shards/bass
SLAKH	strings	data/slakh2100_midi/strings	output/slakh/clean/strings	output/slakh/quarantine/strings	output/slakh/shards/strings
LAMDA	drums	data/loops	output/lamda/clean/drumloops	output/lamda/quarantine/drumloops	output/lamda/shards/drumloops
EOF
)"

# TSVを1行ずつ処理
while IFS=$'\t' read -r NAME INSTR IN_DIR OUT_DIR Q_DIR PKL_OUT || [[ -n "${NAME:-}" ]]; do
  # 空行やコメント行はスキップ
  [[ -z "${NAME:-}" || "${NAME:0:1}" == "#" ]] && continue

  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "== Stage1: ${NAME}/${INSTR} =="
  echo "   Input:       ${IN_DIR}"
  echo "   Output:      ${OUT_DIR}"
  echo "   Quarantine:  ${Q_DIR}"
  echo "   Pickle:      ${PKL_OUT}"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  
  # ディレクトリ作成
  mkdir -p "${OUT_DIR}" "${Q_DIR}" "${PKL_OUT}"

  # clean_midi.py 実行
  PYTHONPATH=. python3 scripts/clean_midi.py \
    --in "${IN_DIR}" \
    --out "${OUT_DIR}" \
    --quarantine "${Q_DIR}" \
    --instrument "${INSTR}" \
    --pickle-out "${PKL_OUT}" \
    --shard-size "${SHARD_SIZE}" \
    --emit-meta-json "${EMIT_META_JSON}" \
    ${RESUME_FLAG} \
    --jobs "${JOBS}"

  echo ""
  echo "✅ ${NAME}/${INSTR} 完了"
  echo ""
done <<<"${DATASETS}"

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║                                                              ║"
echo "║  🎉 Stage1 全データセット処理完了! 🎉                       ║"
echo "║                                                              ║"
echo "╚══════════════════════════════════════════════════════════════╝"
