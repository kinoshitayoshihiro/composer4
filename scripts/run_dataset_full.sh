#!/usr/bin/env bash
set -Eeuo pipefail

# ===== 共通データセット処理ランナー (Pickle-Direct v2) =====
# 複数データセット（POP909/Slakh/LAMDA等）を統一フローで処理
# SSD停止対策: --resume で既存シャードから再開可能

BASE_DIR="/Volumes/SSD-SCTU3A/ラジオ用/music_21/composer2-3"
PY="${BASE_DIR}/.venv311/bin/python"
CLEANER="${BASE_DIR}/scripts/clean_midi.py"
LOG_DIR="${BASE_DIR}/logs"
TS=$(date +%Y%m%d_%H%M%S)
mkdir -p "${LOG_DIR}"

# ===== データセット定義 =====
# フォーマット: name|instrument|in_dir|clean_dir|quarantine_dir|pickle_dir|shard_size|jobs|seed
# shard_size推奨:
#   - POP909: 5000 (実質1シャード)
#   - Slakh2100: 3000-5000 (中規模)
#   - LAMDA: 4000-5000 (大規模、楽器別で分割処理)
#   - Los-Angeles-MIDI: マルチトラック楽曲データセット（約40万ファイル）
#     → 楽器別に分離してクリーニング (piano/strings/guitar/bass/drums)
DATASETS=$(cat <<'EOF'
POP909|piano|data/POP909|data/cleaned/pop909|data/quarantine/pop909|data/piano_metadata|5000|8|pop909-v2
LAMDA_PIANO|piano|data/Los-Angeles-MIDI/MIDIs|data/cleaned/lamda_piano|data/quarantine/lamda_piano|data/lamda_piano_metadata|4000|8|lamda-piano-v1
LAMDA_STRINGS|strings|data/Los-Angeles-MIDI/MIDIs|data/cleaned/lamda_strings|data/quarantine/lamda_strings|data/lamda_strings_metadata|4000|8|lamda-strings-v1
LAMDA_GUITAR|guitar|data/Los-Angeles-MIDI/MIDIs|data/cleaned/lamda_guitar|data/quarantine/lamda_guitar|data/lamda_guitar_metadata|4000|8|lamda-guitar-v1
LAMDA_BASS|bass|data/Los-Angeles-MIDI/MIDIs|data/cleaned/lamda_bass|data/quarantine/lamda_bass|data/lamda_bass_metadata|4000|8|lamda-bass-v1
LAMDA_DRUMS|drums|data/Los-Angeles-MIDI/MIDIs|data/cleaned/lamda_drums|data/quarantine/lamda_drums|data/lamda_drums_metadata|4000|8|lamda-drums-v1
EOF
)

# ===== 進捗だけ確認したい場合: --dry-run =====
DRY_RUN="${1:-}"

run_one() {
  local name="$1" instrument="$2" in_dir="$3" clean_dir="$4" quar_dir="$5" pkl_dir="$6" shard_size="$7" jobs="$8" seed="$9"

  mkdir -p "${BASE_DIR}/${clean_dir}" "${BASE_DIR}/${quar_dir}" "${BASE_DIR}/${pkl_dir}" "${LOG_DIR}"
  local log="${LOG_DIR}/clean_${name}_${instrument}_${TS}.log"

  echo "================================================================================"
  echo "[START] ${name}/${instrument}"
  echo "================================================================================"
  echo "  Input:      ${in_dir}"
  echo "  Clean:      ${clean_dir}"
  echo "  Quarantine: ${quar_dir}"
  echo "  Pickle:     ${pkl_dir}"
  echo "  Shard Size: ${shard_size}"
  echo "  Jobs:       ${jobs}"
  echo "  Seed:       ${seed}"
  echo "  Log:        ${log}"
  echo "================================================================================"

  if [[ "${DRY_RUN}" == "--dry-run" ]]; then
    echo "[DRY-RUN] Would execute:"
    echo "${PY} ${CLEANER} \\"
    echo "  --in '${in_dir}' \\"
    echo "  --out '${clean_dir}' \\"
    echo "  --quarantine '${quar_dir}' \\"
    echo "  --instrument '${instrument}' \\"
    echo "  --pickle-out '${pkl_dir}' \\"
    echo "  --shard-size '${shard_size}' \\"
    echo "  --resume \\"
    echo "  --emit-meta-json off \\"
    echo "  --jobs '${jobs}' \\"
    echo "  --seed '${seed}'"
    return 0
  fi

  # 実行開始時刻記録
  local start_time end_time elapsed hours minutes seconds
  start_time=$(date +%s)

  (cd "${BASE_DIR}" && \
    ${PY} "${CLEANER}" \
      --in "${in_dir}" \
      --out "${clean_dir}" \
      --quarantine "${quar_dir}" \
      --instrument "${instrument}" \
      --pickle-out "${pkl_dir}" \
      --shard-size "${shard_size}" \
      --resume \
      --emit-meta-json off \
      --jobs "${jobs}" \
      --seed "${seed}" \
  ) 2>&1 | tee -a "${log}"

  # 終了時刻記録
  end_time=$(date +%s)
  elapsed=$((end_time - start_time))
  hours=$((elapsed / 3600))
  minutes=$(((elapsed % 3600) / 60))
  seconds=$((elapsed % 60))

  echo ""
  echo "================================================================================"
  echo "[COMPLETE] ${name}/${instrument}"
  echo "================================================================================"
  echo "  Elapsed Time: ${hours}h ${minutes}m ${seconds}s"

  # ファイル統計
  local cleaned_count quar_count total_count
  cleaned_count=$(find "${BASE_DIR}/${clean_dir}" -name "*.mid" 2>/dev/null | wc -l | tr -d ' ')
  quar_count=$(find "${BASE_DIR}/${quar_dir}" -name "*.mid" 2>/dev/null | wc -l | tr -d ' ')
  total_count=$((cleaned_count + quar_count))

  echo "  📊 File Statistics:"
  echo "     Total Processed:  ${total_count}"
  echo "     ✅ Cleaned:       ${cleaned_count}"
  echo "     🗑️  Quarantined:   ${quar_count}"

  # Pickle v2 インデックスチェック
  local idx
  idx=$(ls "${BASE_DIR}/${pkl_dir}"/*_metadata_v2.pickle 2>/dev/null | head -1 || true)
  if [[ -n "${idx}" ]]; then
    echo ""
    echo "  🎯 Pickle Index (v2):"
    ${PY} - "${idx}" <<'PY'
import pickle, sys, os
idx = sys.argv[1]
try:
    with open(idx, "rb") as f:
        d = pickle.load(f)
    print(f"     Index File:   {os.path.basename(idx)}")
    print(f"     Instrument:   {d.get('instrument', 'N/A')}")
    print(f"     Total Files:  {d.get('total_files', 0)}")
    print(f"     Shard Size:   {d.get('shard_size', 0)}")
    print(f"     Shards:       {len(d.get('shards', []))}")
    print(f"     Base Dir:     {d.get('base_dir', 'N/A')}")
except Exception as e:
    print(f"     ❌ Error reading pickle: {e}")
PY
  else
    echo "  ⚠️  Pickle index not found"
  fi

  echo "================================================================================"
  echo ""
  echo "Log saved: ${log}"
  echo ""
}

# ===== メイン処理 =====
# フィルタ: 引数にデータセット名を渡すと、そのデータセットのみ実行
# 例: ./run_dataset_full.sh POP909
#     ./run_dataset_full.sh --dry-run POP909
FILTERS=()
if [[ "${DRY_RUN}" == "--dry-run" ]]; then
  shift || true
fi
if [[ $# -gt 0 ]]; then
  FILTERS=("$@")
fi

if [[ ${#FILTERS[@]} -gt 0 ]]; then
  echo "[INFO] Processing specific datasets: ${FILTERS[*]}"
else
  echo "[INFO] Processing all datasets"
fi
echo ""

while IFS='|' read -r name instrument in_dir clean_dir quar_dir pkl_dir shard_size jobs seed; do
  [[ -z "${name}" ]] && continue
  
  # フィルタチェック
  if [[ ${#FILTERS[@]} -gt 0 ]]; then
    skip=true
    for f in "${FILTERS[@]}"; do
      if [[ "${name}" == "${f}" ]]; then
        skip=false
        break
      fi
    done
    $skip && continue
  fi
  
  run_one "${name}" "${instrument}" "${in_dir}" "${clean_dir}" "${quar_dir}" "${pkl_dir}" "${shard_size}" "${jobs}" "${seed}"
done <<< "${DATASETS}"

echo ""
echo "================================================================================"
echo "🎉 All datasets processing complete!"
echo "================================================================================"
echo ""
echo "Next steps:"
echo "  - Monitor: ./scripts/monitor_pop909.sh"
echo "  - Validate: ./scripts/validate_and_gate.py"
echo "  - Stage 2: python scripts/lamda_stage2_extractor.py \\"
echo "               --metadata-index data/piano_metadata/piano_metadata_v2.pickle \\"
echo "               --output data/piano_stage2_scored.jsonl"
echo ""
