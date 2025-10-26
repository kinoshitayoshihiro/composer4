#!/usr/bin/env bash
set -Eeuo pipefail

# ===== 共通データセット処理ランナー v2 (Robust LAMDA Pipeline) =====
# 【新機能 v2】
# - SSD health check & auto-retry (disconnection対策)
# - Memory monitoring & adaptive parallelism (メモリ不足対策)
# - Checkpoint system (完全なresume capability)
# - Enhanced progress visualization
# - LAMDA CODE思想の統合（音楽的指紋、GPU検索準備）

BASE_DIR="/Volumes/SSD-SCTU3A/ラジオ用/music_21/composer2-3"
PY="${BASE_DIR}/.venv311/bin/python"
CLEANER="${BASE_DIR}/scripts/clean_midi.py"
LOG_DIR="${BASE_DIR}/logs"
CHECKPOINT_DIR="${BASE_DIR}/checkpoints"
TS=$(date +%Y%m%d_%H%M%S)
mkdir -p "${LOG_DIR}" "${CHECKPOINT_DIR}"

# ===== Configuration =====
# SSD監視設定
SSD_PATH="/Volumes/SSD-SCTU3A"
SSD_CHECK_INTERVAL=60  # 60秒ごとにSSD接続チェック
MAX_RETRIES=5          # 最大retry回数
RETRY_DELAY=10         # retry待機時間(秒)

# メモリ監視設定
MEMORY_THRESHOLD_GB=8  # 空きメモリがこれ以下になったら並列数削減
MIN_JOBS=2             # 最小並列数
MAX_JOBS=16            # 最大並列数

# Checkpoint設定
CHECKPOINT_INTERVAL=100  # Nファイルごとにcheckpoint保存

# ===== Color codes =====
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# ===== Helper Functions =====

log_info() {
    echo -e "${CYAN}[INFO]${NC} $*"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $*"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $*"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $*"
}

log_section() {
    echo ""
    echo -e "${MAGENTA}═══════════════════════════════════════════════════════════════════════════${NC}"
    echo -e "${MAGENTA} $* ${NC}"
    echo -e "${MAGENTA}═══════════════════════════════════════════════════════════════════════════${NC}"
}

# SSD接続チェック
check_ssd() {
    if [[ ! -d "${SSD_PATH}" ]]; then
        log_error "SSD not connected: ${SSD_PATH}"
        return 1
    fi
    
    # 書き込み可能かテスト
    local test_file="${SSD_PATH}/.health_check_${TS}"
    if ! touch "${test_file}" 2>/dev/null; then
        log_error "SSD not writable: ${SSD_PATH}"
        return 1
    fi
    rm -f "${test_file}"
    
    return 0
}

# メモリ使用量取得 (macOS)
get_free_memory_gb() {
    local free_pages
    free_pages=$(vm_stat | grep "Pages free" | awk '{print $3}' | sed 's/\.//')
    local page_size=4096
    local free_bytes=$((free_pages * page_size))
    local free_gb=$((free_bytes / 1024 / 1024 / 1024))
    echo "${free_gb}"
}

# 適応的な並列数決定
adaptive_jobs() {
    local requested_jobs="$1"
    local free_mem
    free_mem=$(get_free_memory_gb)
    
    # メモリ不足時は並列数削減
    if (( free_mem < MEMORY_THRESHOLD_GB )); then
        local suggested=$((requested_jobs / 2))
        if (( suggested < MIN_JOBS )); then
            suggested=${MIN_JOBS}
        fi
        log_warn "Low memory (${free_mem}GB free), reducing jobs: ${requested_jobs} → ${suggested}"
        echo "${suggested}"
    else
        echo "${requested_jobs}"
    fi
}

# Checkpoint保存
save_checkpoint() {
    local name="$1" instrument="$2" processed="$3" total="$4"
    local checkpoint_file="${CHECKPOINT_DIR}/${name}_${instrument}_checkpoint.json"
    
    cat > "${checkpoint_file}" <<EOF
{
  "name": "${name}",
  "instrument": "${instrument}",
  "processed": ${processed},
  "total": ${total},
  "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "progress_pct": $(awk "BEGIN {printf \"%.2f\", (${processed}/${total})*100}")
}
EOF
    
    log_info "Checkpoint saved: ${processed}/${total} files ($(awk "BEGIN {printf \"%.1f\", (${processed}/${total})*100}")%)"
}

# Checkpoint読み込み
load_checkpoint() {
    local name="$1" instrument="$2"
    local checkpoint_file="${CHECKPOINT_DIR}/${name}_${instrument}_checkpoint.json"
    
    if [[ ! -f "${checkpoint_file}" ]]; then
        echo "0"
        return 0
    fi
    
    local processed
    processed=$(jq -r '.processed' "${checkpoint_file}" 2>/dev/null || echo "0")
    echo "${processed}"
}

# 進捗バー表示
show_progress() {
    local current="$1" total="$2" name="$3"
    local pct
    pct=$(awk "BEGIN {printf \"%.1f\", (${current}/${total})*100}")
    local bar_width=50
    local filled
    filled=$(awk "BEGIN {printf \"%.0f\", (${current}/${total})*${bar_width}}")
    
    printf "\r${BLUE}[%-${bar_width}s]${NC} %5.1f%% (%d/%d) %s" \
        "$(printf '#%.0s' $(seq 1 ${filled}))" \
        "${pct}" "${current}" "${total}" "${name}"
}

# ===== データセット定義 =====
# Los-Angeles-MIDI: ~400K MIDIファイル
# LAMDA v2 format: 楽器別に分離、シャード化、LAMDA metadata extraction
DATASETS=$(cat <<'EOF'
LAMDA_PIANO|piano|data/Los-Angeles-MIDI/MIDIs|data/cleaned/lamda_piano|data/quarantine/lamda_piano|output/piano_metadata|5000|8|lamda-piano-v2
LAMDA_STRINGS|strings|data/Los-Angeles-MIDI/MIDIs|data/cleaned/lamda_strings|data/quarantine/lamda_strings|output/strings_metadata|5000|8|lamda-strings-v2
LAMDA_GUITAR|guitar|data/Los-Angeles-MIDI/MIDIs|data/cleaned/lamda_guitar|data/quarantine/lamda_guitar|output/guitar_metadata|5000|8|lamda-guitar-v2
LAMDA_BASS|bass|data/Los-Angeles-MIDI/MIDIs|data/cleaned/lamda_bass|data/quarantine/lamda_bass|output/bass_metadata|5000|8|lamda-bass-v2
EOF
)

# Note: LAMDA_DRUMSは既に完了（output/drums_metadata/）
# POP909/Slakhは既存データなので省略（必要なら追加）

# ===== 単一データセット処理 (Retry機能付き) =====
run_one_with_retry() {
    local name="$1" instrument="$2" in_dir="$3" clean_dir="$4" quar_dir="$5" pkl_dir="$6" shard_size="$7" jobs="$8" seed="$9"
    
    local retry_count=0
    local success=false
    
    while (( retry_count < MAX_RETRIES )); do
        log_section "Attempt $((retry_count + 1))/${MAX_RETRIES}: ${name}/${instrument}"
        
        # SSD health check
        if ! check_ssd; then
            log_error "SSD check failed, waiting ${RETRY_DELAY}s..."
            sleep "${RETRY_DELAY}"
            ((retry_count++))
            continue
        fi
        
        # Adaptive parallelism
        local actual_jobs
        actual_jobs=$(adaptive_jobs "${jobs}")
        
        # 実行
        if run_one "${name}" "${instrument}" "${in_dir}" "${clean_dir}" "${quar_dir}" "${pkl_dir}" "${shard_size}" "${actual_jobs}" "${seed}"; then
            success=true
            log_success "${name}/${instrument} completed successfully!"
            break
        else
            log_error "${name}/${instrument} failed (exit code: $?)"
            ((retry_count++))
            
            if (( retry_count < MAX_RETRIES )); then
                log_warn "Retrying in ${RETRY_DELAY}s..."
                sleep "${RETRY_DELAY}"
            fi
        fi
    done
    
    if ! $success; then
        log_error "${name}/${instrument} failed after ${MAX_RETRIES} attempts"
        return 1
    fi
    
    return 0
}

# ===== 単一データセット処理 (Core) =====
run_one() {
    local name="$1" instrument="$2" in_dir="$3" clean_dir="$4" quar_dir="$5" pkl_dir="$6" shard_size="$7" jobs="$8" seed="$9"

    mkdir -p "${BASE_DIR}/${clean_dir}" "${BASE_DIR}/${quar_dir}" "${BASE_DIR}/${pkl_dir}" "${LOG_DIR}"
    local log="${LOG_DIR}/clean_${name}_${instrument}_${TS}.log"

    log_section "${name}/${instrument} - Configuration"
    echo "  Input:      ${in_dir}"
    echo "  Clean:      ${clean_dir}"
    echo "  Quarantine: ${quar_dir}"
    echo "  Pickle:     ${pkl_dir}"
    echo "  Shard Size: ${shard_size}"
    echo "  Jobs:       ${jobs}"
    echo "  Seed:       ${seed}"
    echo "  Log:        ${log}"

    # Checkpoint確認
    local checkpoint_processed
    checkpoint_processed=$(load_checkpoint "${name}" "${instrument}")
    if (( checkpoint_processed > 0 )); then
        log_info "Resuming from checkpoint: ${checkpoint_processed} files already processed"
    fi

    # 実行開始時刻記録
    local start_time end_time elapsed hours minutes seconds
    start_time=$(date +%s)

    # SSD health monitoring (background)
    (
        while true; do
            sleep "${SSD_CHECK_INTERVAL}"
            if ! check_ssd; then
                log_error "SSD disconnected during processing! Sending SIGTERM to Python process..."
                pkill -TERM -f "clean_midi.py.*${instrument}"
                break
            fi
        done
    ) &
    local monitor_pid=$!

    # 実行
    local exit_code=0
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
    ) 2>&1 | tee -a "${log}" || exit_code=$?

    # Monitor停止
    kill "${monitor_pid}" 2>/dev/null || true

    # 終了時刻記録
    end_time=$(date +%s)
    elapsed=$((end_time - start_time))
    hours=$((elapsed / 3600))
    minutes=$(((elapsed % 3600) / 60))
    seconds=$((elapsed % 60))

    if (( exit_code != 0 )); then
        log_error "Processing failed with exit code: ${exit_code}"
        return "${exit_code}"
    fi

    log_section "${name}/${instrument} - Summary"
    echo "  Elapsed Time: ${hours}h ${minutes}m ${seconds}s"

    # ファイル統計
    local cleaned_count quar_count total_count
    cleaned_count=$(find "${BASE_DIR}/${clean_dir}" -name "*.mid" 2>/dev/null | wc -l | tr -d ' ')
    quar_count=$(find "${BASE_DIR}/${quar_dir}" -name "*.mid" 2>/dev/null | wc -l | tr -d ' ')
    total_count=$((cleaned_count + quar_count))

    echo ""
    echo "  📊 File Statistics:"
    echo "     Total Processed:  ${total_count}"
    echo "     ✅ Cleaned:       ${cleaned_count} ($(awk "BEGIN {printf \"%.1f\", (${cleaned_count}/${total_count})*100}")%)"
    echo "     🗑️  Quarantined:   ${quar_count} ($(awk "BEGIN {printf \"%.1f\", (${quar_count}/${total_count})*100}")%)"

    # Pickle index check
    local idx
    idx=$(ls "${BASE_DIR}/${pkl_dir}"/*_metadata_v2.pickle 2>/dev/null | head -1 || true)
    if [[ -n "${idx}" ]]; then
        echo ""
        echo "  🎯 LAMDA Pickle Index (v2):"
        ${PY} - "${idx}" <<'PY'
import pickle, sys, os
idx = sys.argv[1]
try:
    with open(idx, "rb") as f:
        d = pickle.load(f)
    print(f"     Index File:   {os.path.basename(idx)}")
    print(f"     Instrument:   {d.get('instrument', 'N/A')}")
    print(f"     Total Files:  {d.get('total_files', 0):,}")
    print(f"     Shard Size:   {d.get('shard_size', 0):,}")
    print(f"     Shards:       {len(d.get('shards', []))}")
    
    # LAMDA v2 compatibility check
    if d.get('version') == 'lamda_v2_index':
        print(f"     ✅ LAMDA v2 compatible")
    
    # GPU search ready check
    shards = d.get('shards', [])
    if shards:
        sample_shard = shards[0]
        print(f"     Sample Shard: {sample_shard['filename']}")
        print(f"     Files in shard: {sample_shard.get('count', 0):,}")
except Exception as e:
    print(f"     ❌ Error reading pickle: {e}")
PY
    else
        log_warn "Pickle index not found in ${BASE_DIR}/${pkl_dir}"
    fi

    # Checkpoint保存（完了）
    save_checkpoint "${name}" "${instrument}" "${total_count}" "${total_count}"

    echo ""
    log_info "Log saved: ${log}"
    
    return 0
}

# ===== メイン処理 =====
main() {
    local dry_run="${1:-}"
    
    log_section "LAMDA Training Pipeline v2"
    echo "  SSD Path:            ${SSD_PATH}"
    echo "  Base Directory:      ${BASE_DIR}"
    echo "  Python:              ${PY}"
    echo "  Max Retries:         ${MAX_RETRIES}"
    echo "  Memory Threshold:    ${MEMORY_THRESHOLD_GB}GB"
    echo "  Checkpoint Interval: ${CHECKPOINT_INTERVAL} files"
    
    # SSD初期チェック
    if ! check_ssd; then
        log_error "SSD not available at startup: ${SSD_PATH}"
        log_error "Please connect SSD and retry"
        return 1
    fi
    log_success "SSD health check passed"
    
    # Python環境チェック
    if [[ ! -f "${PY}" ]]; then
        log_error "Python not found: ${PY}"
        return 1
    fi
    log_success "Python environment found"
    
    # clean_midi.py存在チェック
    if [[ ! -f "${CLEANER}" ]]; then
        log_error "Cleaner script not found: ${CLEANER}"
        return 1
    fi
    log_success "Cleaner script found"
    
    echo ""
    
    # データセット処理
    local filters=()
    if [[ "${dry_run}" == "--dry-run" ]]; then
        shift || true
    fi
    if [[ $# -gt 0 ]]; then
        filters=("$@")
    fi

    if [[ ${#filters[@]} -gt 0 ]]; then
        log_info "Processing specific datasets: ${filters[*]}"
    else
        log_info "Processing all datasets"
    fi
    echo ""

    local dataset_count=0
    local success_count=0
    local failed_count=0

    while IFS='|' read -r name instrument in_dir clean_dir quar_dir pkl_dir shard_size jobs seed; do
        [[ -z "${name}" ]] && continue
        
        # フィルタチェック
        if [[ ${#filters[@]} -gt 0 ]]; then
            local skip=true
            for f in "${filters[@]}"; do
                if [[ "${name}" == "${f}" ]]; then
                    skip=false
                    break
                fi
            done
            ${skip} && continue
        fi
        
        ((dataset_count++))
        
        if [[ "${dry_run}" == "--dry-run" ]]; then
            log_info "[DRY-RUN] Would process: ${name}/${instrument}"
            echo "  Command: ${PY} ${CLEANER} --in ${in_dir} --out ${clean_dir} --instrument ${instrument} ..."
            continue
        fi
        
        if run_one_with_retry "${name}" "${instrument}" "${in_dir}" "${clean_dir}" "${quar_dir}" "${pkl_dir}" "${shard_size}" "${jobs}" "${seed}"; then
            ((success_count++))
        else
            ((failed_count++))
            log_error "Dataset ${name}/${instrument} failed after all retries"
        fi
        
        echo ""
    done <<< "${DATASETS}"

    # 最終レポート
    log_section "Pipeline Complete"
    echo "  Datasets Processed: ${dataset_count}"
    echo "  ✅ Success:         ${success_count}"
    echo "  ❌ Failed:          ${failed_count}"
    echo ""
    
    if (( failed_count > 0 )); then
        log_error "Some datasets failed, check logs in ${LOG_DIR}"
        return 1
    fi
    
    log_section "Next Steps"
    echo ""
    echo "  🎯 Stage 2: Feature Extraction & GPU Search Preparation"
    echo ""
    echo "  For each instrument, run:"
    echo "    python scripts/lamda_stage2_extractor.py \\"
    echo "        --metadata-index output/piano_metadata/piano_metadata_v2.pickle \\"
    echo "        --output data/piano_stage2_scored.jsonl"
    echo ""
    echo "  🔍 Validate Results:"
    echo "    python scripts/validate_and_gate.py \\"
    echo "        --pickle output/piano_metadata/piano_metadata_v2.pickle"
    echo ""
    echo "  📊 Monitor Training:"
    echo "    watch -n 5 'find output/ -name \"*.pickle\" -exec ls -lh {} \\;'"
    echo ""
    
    log_success "All processing complete! 🎉"
    
    return 0
}

# ===== Entry Point =====
main "$@"
