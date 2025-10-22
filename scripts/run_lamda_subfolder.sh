#!/usr/bin/env bash
# ===== LAMDA サブフォルダ単位処理スクリプト =====
# 40万ファイルを16サブフォルダに分割して処理
#
# 使用方法:
#   ./scripts/run_lamda_subfolder.sh piano 0      # Piano サブフォルダ0を処理
#   ./scripts/run_lamda_subfolder.sh guitar a     # Guitar サブフォルダaを処理
#   ./scripts/run_lamda_subfolder.sh all 0        # 全楽器のサブフォルダ0を処理

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PY="${BASE_DIR}/.venv311/bin/python"
CLEANER="${BASE_DIR}/scripts/clean_midi.py"
LOG_DIR="${BASE_DIR}/logs"
TS=$(date +%Y%m%d_%H%M%S)

mkdir -p "${LOG_DIR}"

# 引数チェック
if [ $# -lt 2 ]; then
    echo "Usage: $0 <instrument|all> <subfolder_id>"
    echo ""
    echo "Examples:"
    echo "  $0 piano 0        # Piano サブフォルダ0"
    echo "  $0 guitar a       # Guitar サブフォルダa"
    echo "  $0 all 0          # 全楽器のサブフォルダ0"
    echo ""
    echo "Subfolder IDs: 0-9, a-f (16 subfolders total)"
    echo "Instruments: piano, guitar, bass, strings, drums, all"
    exit 1
fi

INSTRUMENT="$1"
SUBFOLDER_ID="$2"

# サブフォルダIDの検証
if ! [[ "${SUBFOLDER_ID}" =~ ^[0-9a-f]$ ]]; then
    echo "❌ Error: Invalid subfolder ID '${SUBFOLDER_ID}'"
    echo "   Must be one of: 0-9, a-f"
    exit 1
fi

# 楽器リスト設定
INSTRUMENTS=()
if [ "${INSTRUMENT}" = "all" ]; then
    INSTRUMENTS=("piano" "guitar" "bass" "strings" "drums")
else
    INSTRUMENTS=("${INSTRUMENT}")
fi

# 各楽器を処理
for inst in "${INSTRUMENTS[@]}"; do
    # 大文字変換（macOS互換）
    inst_upper=$(echo "${inst}" | tr '[:lower:]' '[:upper:]')
    
    echo ""
    echo "================================================================================"
    echo "🎵 LAMDA ${inst_upper} - Subfolder ${SUBFOLDER_ID}"
    echo "================================================================================"
    
    # パス設定
    IN_DIR="${BASE_DIR}/data/Los-Angeles-MIDI/MIDIs/${SUBFOLDER_ID}"
    OUT_DIR="${BASE_DIR}/data/cleaned/lamda_${inst}/${SUBFOLDER_ID}"
    QUAR_DIR="${BASE_DIR}/data/quarantine/lamda_${inst}/${SUBFOLDER_ID}"
    PICKLE_DIR="${BASE_DIR}/data/lamda_${inst}_metadata"
    LOG_FILE="${LOG_DIR}/lamda_${inst}_${SUBFOLDER_ID}_${TS}.log"
    
    # ディレクトリ作成
    mkdir -p "${OUT_DIR}" "${QUAR_DIR}" "${PICKLE_DIR}"
    
    echo "  Input:      ${IN_DIR}"
    echo "  Output:     ${OUT_DIR}"
    echo "  Quarantine: ${QUAR_DIR}"
    echo "  Pickle:     ${PICKLE_DIR}/${inst}_shard_${SUBFOLDER_ID}.pickle"
    echo "  Log:        ${LOG_FILE}"
    echo ""
    
    # 実行
    start_time=$(date +%s)
    
    ${PY} "${CLEANER}" \
        --in "${IN_DIR}" \
        --out "${OUT_DIR}" \
        --quarantine "${QUAR_DIR}" \
        --instrument "${inst}" \
        --pickle-out "${PICKLE_DIR}" \
        --subfolder-id "${SUBFOLDER_ID}" \
        --emit-meta-json off \
        --jobs 4 \
        --seed "lamda-${inst}-${SUBFOLDER_ID}" \
        2>&1 | tee "${LOG_FILE}"
    
    end_time=$(date +%s)
    elapsed=$((end_time - start_time))
    hours=$((elapsed / 3600))
    minutes=$(((elapsed % 3600) / 60))
    seconds=$((elapsed % 60))
    
    echo ""
    echo "================================================================================"
    echo "✅ COMPLETE: ${inst_upper} Subfolder ${SUBFOLDER_ID}"
    echo "================================================================================"
    echo "  Elapsed: ${hours}h ${minutes}m ${seconds}s"
    echo "  Log: ${LOG_FILE}"
    echo ""
done

echo ""
echo "🎉 All requested processing complete!"
echo ""
