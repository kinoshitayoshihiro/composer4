#!/usr/bin/env bash
# run_batch_song_generation.sh - 複数song_packageのバッチ生成（並列処理対応）
# Usage: bash scripts/run_batch_song_generation.sh [--auto-safe-kit] [--jobs N]

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
AUTO_SAFE_KIT=""
JOBS=1
ROOT_DIR="song_packages"

# 引数解析
while [[ $# -gt 0 ]]; do
    case $1 in
        --auto-safe-kit)
            AUTO_SAFE_KIT="--auto-safe-kit"
            shift
            ;;
        --jobs)
            JOBS="$2"
            shift 2
            ;;
        --root)
            ROOT_DIR="$2"
            shift 2
            ;;
        *)
            err "Unknown option: $1"
            echo "Usage: $0 [--auto-safe-kit] [--jobs N] [--root DIR]"
            exit 1
            ;;
    esac
done

log "🎯 Batch Song Generation Pipeline"
log "============================================================"
log "Root directory: $ROOT_DIR"
log "Parallel jobs:  $JOBS"
log "Auto Safe-Kit:  ${AUTO_SAFE_KIT:-disabled}"
log ""

# song_package一覧取得
SONG_PACKAGES=($(find "$ROOT_DIR" -name "song_package.yaml" -type f | sed 's|/song_package.yaml||'))

if [ ${#SONG_PACKAGES[@]} -eq 0 ]; then
    err "No song_package found in $ROOT_DIR"
    exit 1
fi

log "📂 Found ${#SONG_PACKAGES[@]} song package(s)"
log ""

# 並列処理用の関数
process_song() {
    local SONG_DIR="$1"
    local AUTO_SAFE_KIT_FLAG="$2"
    local SONG_NAME=$(basename "$SONG_DIR")
    local PROJECT_NAME=$(basename "$(dirname "$SONG_DIR")")
    local PKG_NAME="${PROJECT_NAME}/${SONG_NAME}"
    
    log "📦 Processing: $PKG_NAME"
    
    # run_song_generation.sh 実行
    if bash scripts/run_song_generation.sh "$SONG_DIR" $AUTO_SAFE_KIT_FLAG > "/tmp/batch_${SONG_NAME}.log" 2>&1; then
        ok "✅ Completed: $PKG_NAME"
        return 0
    else
        err "❌ Failed: $PKG_NAME (see /tmp/batch_${SONG_NAME}.log)"
        return 1
    fi
}

export -f process_song log ok err

# 並列処理実行
if command -v parallel &> /dev/null && [ "$JOBS" -gt 1 ]; then
    # GNU parallelが利用可能
    log "🚀 Running with GNU parallel (jobs: $JOBS)..."
    printf '%s\n' "${SONG_PACKAGES[@]}" | parallel -j "$JOBS" process_song {} "$AUTO_SAFE_KIT"
    PARALLEL_STATUS=$?
else
    # 逐次処理
    if [ "$JOBS" -gt 1 ]; then
        warn "GNU parallel not found, falling back to sequential processing"
    fi
    
    SUCCESS_COUNT=0
    FAIL_COUNT=0
    
    for SONG_DIR in "${SONG_PACKAGES[@]}"; do
        if process_song "$SONG_DIR" "$AUTO_SAFE_KIT"; then
            SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
        else
            FAIL_COUNT=$((FAIL_COUNT + 1))
        fi
    done
    
    log ""
    log "============================================================"
    log "🎯 Batch Generation Summary"
    log "============================================================"
    log "Total processed: ${#SONG_PACKAGES[@]}"
    log "Success:         $SUCCESS_COUNT"
    log "Failed:          $FAIL_COUNT"
    
    PARALLEL_STATUS=$([ $FAIL_COUNT -eq 0 ] && echo 0 || echo 1)
fi

log ""
log "✅ Batch generation completed!"

exit $PARALLEL_STATUS
