#!/usr/bin/env bash
# run_batch_emotion_extraction.sh - bars.parquetから全曲emotion_profile.json一括生成
#
# Usage:
#   bash scripts/run_batch_emotion_extraction.sh --root song_packages/test_project

set -euo pipefail

# ==============================================================================
# Color & Logging
# ==============================================================================
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log() {
    echo -e "${GREEN}[$(date +'%H:%M:%S')]${NC} $*"
}

warn() {
    echo -e "${YELLOW}[WARN]${NC} $*"
}

error() {
    echo -e "${RED}[ERROR]${NC} $*"
}

ok() {
    echo -e "${GREEN}[OK]${NC} $*"
}

# ==============================================================================
# Argument Parsing
# ==============================================================================
ROOT_DIR=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --root)
            ROOT_DIR="$2"
            shift 2
            ;;
        *)
            error "Unknown option: $1"
            echo "Usage: $0 --root <root_directory>"
            exit 1
            ;;
    esac
done

if [[ -z "$ROOT_DIR" ]]; then
    error "Missing required argument: --root"
    echo "Usage: $0 --root <root_directory>"
    exit 1
fi

if [[ ! -d "$ROOT_DIR" ]]; then
    error "Root directory not found: $ROOT_DIR"
    exit 1
fi

# ==============================================================================
# Find all song packages
# ==============================================================================
log "🎭 Batch Emotion Profile Extraction"
log "Root directory: $ROOT_DIR"

SONG_PACKAGES=()
while IFS= read -r -d '' song_dir; do
    SONG_PACKAGES+=("$song_dir")
done < <(find "$ROOT_DIR" -name "bars.parquet" -type f -print0 \
  | xargs -0 -I{} sh -c 'printf "%s\0" "$(dirname "{}")"')

log "📂 Found ${#SONG_PACKAGES[@]} song package(s)"

# ==============================================================================
# Process each song package
# ==============================================================================
SUCCESS_COUNT=0
FAIL_COUNT=0

for SONG_DIR in "${SONG_PACKAGES[@]}"; do
    SONG_NAME=$(basename "$SONG_DIR")
    PROJECT_NAME=$(basename "$(dirname "$SONG_DIR")")
    
    BARS_FILE="$SONG_DIR/bars.parquet"
    OUTPUT_FILE="$SONG_DIR/emotion_profile.json"
    
    if [[ ! -f "$BARS_FILE" ]]; then
        warn "   bars.parquet not found: $SONG_DIR"
        ((FAIL_COUNT++))
        continue
    fi
    
    log "   Processing: $PROJECT_NAME/$SONG_NAME"
    
    if python3 scripts/extract_emotion_profile.py \
        --bars "$BARS_FILE" \
        --output "$OUTPUT_FILE" >/dev/null 2>&1; then
        
        ok "   ✅ Completed: $PROJECT_NAME/$SONG_NAME"
        ((SUCCESS_COUNT++))
    else
        error "   ❌ Failed: $PROJECT_NAME/$SONG_NAME"
        ((FAIL_COUNT++))
    fi
done

# ==============================================================================
# Summary
# ==============================================================================
log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
log "Total processed: $((SUCCESS_COUNT + FAIL_COUNT))"
log "Success:         $SUCCESS_COUNT"
log "Failed:          $FAIL_COUNT"

if [[ $FAIL_COUNT -eq 0 ]]; then
    ok "✅ Batch emotion extraction completed successfully!"
    exit 0
else
    error "⚠️  Batch emotion extraction completed with $FAIL_COUNT failure(s)"
    exit 1
fi
