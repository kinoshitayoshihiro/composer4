#!/usr/bin/env bash
# ========================================
# LAMDA サブフォルダ単位処理ランナー
# ========================================
# Los-Angeles-MIDI/MIDIs は 16サブフォルダ (0-9, a-f) に分かれている
# → 各サブフォルダごとに1 pickleを生成
# → メモリ効率良く、途中停止からの再開が容易
#
# 使用方法:
#   ./scripts/run_lamda_by_subfolder.sh piano        # piano のみ
#   ./scripts/run_lamda_by_subfolder.sh piano 0 1 2  # piano のサブフォルダ 0,1,2 のみ
#   ./scripts/run_lamda_by_subfolder.sh              # 全楽器、全サブフォルダ

set -Eeuo pipefail

BASE_DIR="/Volumes/SSD-SCTU3A/ラジオ用/music_21/composer2-3"
PY="${BASE_DIR}/.venv311/bin/python"
CLEANER="${BASE_DIR}/scripts/clean_midi.py"
LOG_DIR="${BASE_DIR}/logs"
LAMDA_BASE="${BASE_DIR}/data/Los-Angeles-MIDI/MIDIs"

# サブフォルダ一覧 (16個)
SUBFOLDERS=("0" "1" "2" "3" "4" "5" "6" "7" "8" "9" "a" "b" "c" "d" "e" "f")

# 楽器一覧
INSTRUMENTS=("piano" "strings" "guitar" "bass" "drums")

# 引数パース
SELECTED_INSTRUMENT=""
SELECTED_SUBFOLDERS=()

if [[ $# -gt 0 ]]; then
    SELECTED_INSTRUMENT="$1"
    shift
    
    # 残りの引数はサブフォルダ指定
    if [[ $# -gt 0 ]]; then
        SELECTED_SUBFOLDERS=("$@")
    else
        SELECTED_SUBFOLDERS=("${SUBFOLDERS[@]}")
    fi
else
    # 引数なし = 全楽器、全サブフォルダ
    SELECTED_INSTRUMENT="all"
    SELECTED_SUBFOLDERS=("${SUBFOLDERS[@]}")
fi

# 楽器リスト決定
if [[ "${SELECTED_INSTRUMENT}" == "all" ]]; then
    INST_LIST=("${INSTRUMENTS[@]}")
else
    INST_LIST=("${SELECTED_INSTRUMENT}")
fi

echo "================================================================================"
echo "🎵 LAMDA Subfolder-Based Processing"
echo "================================================================================"
echo "Instruments: ${INST_LIST[*]}"
echo "Subfolders:  ${SELECTED_SUBFOLDERS[*]}"
echo "Total tasks: $((${#INST_LIST[@]} * ${#SELECTED_SUBFOLDERS[@]}))"
echo "================================================================================"
echo ""

process_one_subfolder() {
    local instrument="$1"
    local subfolder="$2"
    
    local in_dir="${LAMDA_BASE}/${subfolder}"
    local clean_dir="${BASE_DIR}/data/cleaned/lamda_${instrument}/${subfolder}"
    local quar_dir="${BASE_DIR}/data/quarantine/lamda_${instrument}/${subfolder}"
    local pkl_dir="${BASE_DIR}/data/lamda_${instrument}_metadata"
    local pkl_file="${pkl_dir}/${instrument}_shard_${subfolder}.pickle"
    local TS=$(date +%Y%m%d_%H%M%S)
    local log="${LOG_DIR}/lamda_${instrument}_${subfolder}_${TS}.log"
    
    mkdir -p "${clean_dir}" "${quar_dir}" "${pkl_dir}" "${LOG_DIR}"
    
    echo "================================================================================"
    echo "[START] ${instrument} / subfolder ${subfolder}"
    echo "================================================================================"
    echo "  Input:      ${in_dir}"
    echo "  Clean:      ${clean_dir}"
    echo "  Quarantine: ${quar_dir}"
    echo "  Pickle:     ${pkl_file}"
    echo "  Log:        ${log}"
    echo "================================================================================"
    
    # 既存 pickle があればスキップ
    if [[ -f "${pkl_file}" ]]; then
        echo "✅ SKIP: Pickle already exists: ${pkl_file}"
        echo ""
        return 0
    fi
    
    # ファイル数確認
    local file_count=$(find "${in_dir}" -type f \( -name "*.mid" -o -name "*.midi" \) 2>/dev/null | wc -l | tr -d ' ')
    echo "📊 Files in subfolder: ${file_count}"
    
    if [[ "${file_count}" -eq 0 ]]; then
        echo "⚠️  No MIDI files found, skipping..."
        echo ""
        return 0
    fi
    
    local start_time=$(date +%s)
    
    # 実行
    (cd "${BASE_DIR}" && \
        ${PY} "${CLEANER}" \
            --in "${in_dir}" \
            --out "${clean_dir}" \
            --quarantine "${quar_dir}" \
            --instrument "${instrument}" \
            --pickle-out "${pkl_dir}" \
            --shard-size 100000 \
            --resume \
            --emit-meta-json off \
            --jobs 4 \
            --seed "lamda-${instrument}-${subfolder}" \
            --subfolder-mode "${subfolder}" \
    ) 2>&1 | tee "${log}"
    
    local exit_code=${PIPESTATUS[0]}
    local end_time=$(date +%s)
    local elapsed=$((end_time - start_time))
    local hours=$((elapsed / 3600))
    local minutes=$(((elapsed % 3600) / 60))
    local seconds=$((elapsed % 60))
    
    echo ""
    echo "================================================================================"
    if [[ ${exit_code} -eq 0 ]]; then
        echo "[COMPLETE] ${instrument} / subfolder ${subfolder}"
        echo "✅ Success"
    else
        echo "[FAILED] ${instrument} / subfolder ${subfolder}"
        echo "❌ Exit code: ${exit_code}"
    fi
    echo "================================================================================"
    echo "  Elapsed Time: ${hours}h ${minutes}m ${seconds}s"
    
    # Pickle統計
    if [[ -f "${pkl_file}" ]]; then
        echo "  📦 Pickle size: $(du -h "${pkl_file}" | cut -f1)"
        
        # Python で pickle 内容を表示
        ${PY} - "${pkl_file}" <<'PY' || true
import pickle, sys
try:
    with open(sys.argv[1], "rb") as f:
        data = pickle.load(f)
    if isinstance(data, list):
        print(f"  📊 Entries: {len(data)}")
    elif isinstance(data, dict):
        print(f"  📊 Keys: {list(data.keys())}")
        if "total_files" in data:
            print(f"  📊 Total files: {data['total_files']}")
except Exception as e:
    print(f"  ⚠️  Could not read pickle: {e}")
PY
    fi
    
    echo "================================================================================"
    echo ""
    
    return ${exit_code}
}

# メイン処理ループ
total_tasks=0
success_tasks=0
failed_tasks=0

for instrument in "${INST_LIST[@]}"; do
    for subfolder in "${SELECTED_SUBFOLDERS[@]}"; do
        ((total_tasks++))
        
        if process_one_subfolder "${instrument}" "${subfolder}"; then
            ((success_tasks++))
        else
            ((failed_tasks++))
            echo "❌ Failed: ${instrument}/${subfolder}"
            echo "   Check log: ${LOG_DIR}/lamda_${instrument}_${subfolder}_*.log"
            echo ""
        fi
    done
done

echo ""
echo "================================================================================"
echo "🎉 All LAMDA Subfolder Processing Complete"
echo "================================================================================"
echo "  Total tasks: ${total_tasks}"
echo "  ✅ Success:  ${success_tasks}"
echo "  ❌ Failed:   ${failed_tasks}"
echo "================================================================================"
echo ""
echo "Next steps:"
echo "  1. Verify pickles: ls -lh data/lamda_*_metadata/*.pickle"
echo "  2. Merge index (if needed)"
echo "  3. Start Stage2 training"
echo ""
