#!/usr/bin/env bash
# run_batch_vioptt_generation.sh - VioPTT WAVバッチ生成スクリプト
#
# Phase 15.2: 複数song_packageの一括WAV出力（並列処理対応）
#
# Usage:
#   bash scripts/run_batch_vioptt_generation.sh \
#     --root song_packages/test_project \
#     --instrument violin \
#     --vst-path /path/to/violin.vst3 \
#     [--jobs 4] \
#     [--no-merge]

set -Eeuo pipefail

# カラー出力
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log() { echo -e "${BLUE}[$(date +'%H:%M:%S')]${NC} $*"; }
err() { echo -e "${RED}[ERROR]${NC} $*" >&2; }
warn() { echo -e "${YELLOW}[WARN]${NC} $*"; }
ok() { echo -e "${GREEN}[OK]${NC} $*"; }

# デフォルト設定
ROOT_DIR=""
INSTRUMENT=""
VST_PATH=""
JOBS=1
NO_MERGE=""

# 引数解析
while [[ $# -gt 0 ]]; do
    case $1 in
        --root)
            ROOT_DIR="$2"
            shift 2
            ;;
        --instrument)
            INSTRUMENT="$2"
            shift 2
            ;;
        --vst-path)
            VST_PATH="$2"
            shift 2
            ;;
        --jobs)
            JOBS="$2"
            shift 2
            ;;
        --no-merge)
            NO_MERGE="--no-merge"
            shift
            ;;
        *)
            err "Unknown option: $1"
            echo "Usage: $0 --root DIR --instrument INST [--vst-path VST] [--jobs N] [--no-merge]"
            exit 1
            ;;
    esac
done

# 引数チェック
if [ -z "$ROOT_DIR" ] || [ -z "$INSTRUMENT" ]; then
    err "Missing required arguments: --root and --instrument"
    echo "Usage: $0 --root DIR --instrument INST [--vst-path VST] [--jobs N] [--no-merge]"
    exit 1
fi

# VST存在チェック
VST_FLAG=""
if [ -n "$VST_PATH" ]; then
    if [ -f "$VST_PATH" ]; then
        VST_FLAG="--vst-path $VST_PATH"
    else
        warn "VST not found: $VST_PATH"
        warn "Generating control MIDI only (no WAV output)"
    fi
fi

log "🎻 Batch VioPTT WAV Generation"
log "============================================================"
log "Root directory: $ROOT_DIR"
log "Instrument:     $INSTRUMENT"
log "VST path:       ${VST_PATH:-N/A (control MIDI only)}"
log "Parallel jobs:  $JOBS"
log "Merge mode:     $([ -n "$NO_MERGE" ] && echo "disabled" || echo "enabled")"
log ""

# song_package一覧取得（null区切りで安全に列挙、空白・日本語パス対応）
# macOS zsh互換版（readarray/mapfile非対応環境）
SONG_PACKAGES=()
while IFS= read -r -d '' song_dir; do
    SONG_PACKAGES+=("$song_dir")
done < <(find "$ROOT_DIR" -name "drums_recommendations.json" -type f -print0 \
  | xargs -0 -I{} sh -c 'printf "%s\0" "$(dirname "{}")"')

if [ ${#SONG_PACKAGES[@]} -eq 0 ]; then
    err "No drums_recommendations.json found in $ROOT_DIR"
    err "Run 'bash scripts/run_batch_song_generation.sh --root $ROOT_DIR' first"
    exit 1
fi

log "📂 Found ${#SONG_PACKAGES[@]} song package(s)"
log ""

# Tempo自動検出関数
detect_tempo() {
    local SONG_DIR="$1"
    local DRUMS_MIDI="$SONG_DIR/drums.mid"
    
    if [ -f "$DRUMS_MIDI" ]; then
        python3 << PYEOF
import pretty_midi
try:
    mid = pretty_midi.PrettyMIDI("$DRUMS_MIDI")
    print(f"{mid.estimate_tempo():.0f}")
except:
    print("120")
PYEOF
    else
        echo "120"
    fi
}

# 並列処理用の関数
process_song() {
    local SONG_DIR="$1"
    local INSTRUMENT="$2"
    local VST_FLAG="$3"
    local NO_MERGE="$4"
    local SONG_NAME=$(basename "$SONG_DIR")
    local PROJECT_NAME=$(basename "$(dirname "$SONG_DIR")")
    local PKG_NAME="${PROJECT_NAME}/${SONG_NAME}"
    
    log "📦 Processing: $PKG_NAME"
    
    # Tempo自動検出
    TEMPO=$(detect_tempo "$SONG_DIR")
    log "   Detected tempo: ${TEMPO} BPM"
    
    # run_vioptt_pipeline.sh 実行
    if bash scripts/run_vioptt_pipeline.sh \
        --song-dir "$SONG_DIR" \
        --instrument "$INSTRUMENT" \
        --tempo-bpm "$TEMPO" \
        $VST_FLAG \
        $NO_MERGE > "/tmp/vioptt_${SONG_NAME}.log" 2>&1; then
        ok "✅ Completed: $PKG_NAME"
        return 0
    else
        err "❌ Failed: $PKG_NAME (see /tmp/vioptt_${SONG_NAME}.log)"
        return 1
    fi
}

export -f process_song detect_tempo log ok err
export INSTRUMENT VST_FLAG NO_MERGE

# 並列処理実行
if command -v parallel &> /dev/null && [ "$JOBS" -gt 1 ]; then
    # GNU parallelが利用可能
    log "🚀 Running with GNU parallel (jobs: $JOBS)..."
    printf '%s\n' "${SONG_PACKAGES[@]}" | parallel -j "$JOBS" process_song {} "$INSTRUMENT" "$VST_FLAG" "$NO_MERGE"
    PARALLEL_STATUS=$?
else
    # 逐次処理
    if [ "$JOBS" -gt 1 ]; then
        warn "GNU parallel not found, falling back to sequential processing"
    fi
    
    SUCCESS_COUNT=0
    FAIL_COUNT=0
    
    for SONG_DIR in "${SONG_PACKAGES[@]}"; do
        if process_song "$SONG_DIR" "$INSTRUMENT" "$VST_FLAG" "$NO_MERGE"; then
            SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
        else
            FAIL_COUNT=$((FAIL_COUNT + 1))
        fi
    done
    
    log ""
    log "============================================================"
    log "🎯 Batch VioPTT Generation Summary"
    log "============================================================"
    log "Total processed: ${#SONG_PACKAGES[@]}"
    log "Success:         $SUCCESS_COUNT"
    log "Failed:          $FAIL_COUNT"
    
    PARALLEL_STATUS=$([ $FAIL_COUNT -eq 0 ] && echo 0 || echo 1)
fi

log ""
if [ $PARALLEL_STATUS -eq 0 ]; then
    ok "✅ Batch VioPTT generation completed successfully!"
else
    err "⚠️  Some songs failed (check /tmp/vioptt_*.log for details)"
fi

exit $PARALLEL_STATUS
