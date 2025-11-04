#!/bin/bash
# scripts/check_rhythm_ai_sources.sh
# Rhythm AIソースファイル確認スクリプト

set -euo pipefail

: "${DRY_RUN:=0}"  # 1 to print actions without failing
: "${QUIET:=0}"    # 1 to suppress info logs

log()  { [ "${QUIET}" = "1" ] || printf "%s\n" "$*"; }
warn() { printf "WARN: %s\n" "$*" >&2; }
err()  { printf "ERR: %s\n" "$*" >&2; }
run()  { if [ "${DRY_RUN}" = "1" ]; then printf "DRY: %s\n" "$*"; else eval "$@"; fi; }

log "🔍 Rhythm AI Source Files Check"
log "================================"
log ""

BASE_DIR="${BASE_DIR:-$(pwd)}"

# drumclean_midi
DRUMCLEAN_DIR="${BASE_DIR}/data/Los-Angeles-MIDI/LOCAL_LAMDA/rhythmAI/drumclean_midi"
if [ -d "${DRUMCLEAN_DIR}" ]; then
    MIDI_COUNT=$(find "${DRUMCLEAN_DIR}" \( -name "*.mid" -o -name "*.midi" \) -type f 2>/dev/null | wc -l | tr -d ' ')
    log "📁 drumclean_midi:"
    log "   Location: ${DRUMCLEAN_DIR}"
    log "   MIDI files: ${MIDI_COUNT}"
    
    if [ "${MIDI_COUNT}" -gt 0 ]; then
        log "   Sample files:"
        find "${DRUMCLEAN_DIR}" \( -name "*.mid" -o -name "*.midi" \) -type f 2>/dev/null | head -3 | while read file; do
            log "     - $(basename "$file")"
        done
    fi
else
    warn "drumclean_midi directory not found"
fi

log ""

# groove
GROOVE_DIR="${BASE_DIR}/data/Los-Angeles-MIDI/LOCAL_LAMDA/rhythmAI/groove"
if [ -d "${GROOVE_DIR}" ]; then
    MIDI_COUNT=$(find "${GROOVE_DIR}" \( -name "*.mid" -o -name "*.midi" \) -type f 2>/dev/null | wc -l | tr -d ' ')
    WAV_COUNT=$(find "${GROOVE_DIR}" \( -name "*.wav" -o -name "*.mp3" \) -type f 2>/dev/null | wc -l | tr -d ' ')
    
    log "📁 groove:"
    log "   Location: ${GROOVE_DIR}"
    log "   MIDI files: ${MIDI_COUNT}"
    log "   Audio files: ${WAV_COUNT}"
    
    if [ "${MIDI_COUNT}" -gt 0 ]; then
        log "   Sample MIDI files:"
        find "${GROOVE_DIR}" \( -name "*.mid" -o -name "*.midi" \) -type f 2>/dev/null | head -3 | while read file; do
            log "     - $(basename "$file")"
        done
    fi
else
    warn "groove directory not found"
fi

log ""
log "================================"

# 既存の出力確認
log "📊 Existing Output Check:"
log ""

if [ -d "${BASE_DIR}/output/rhythm_ai" ]; then
    log "output/rhythm_ai/ contents:"
    ls -lh "${BASE_DIR}/output/rhythm_ai/" 2>/dev/null | tail -10
    
    # 既存Parquet確認
    if [ -f "${BASE_DIR}/output/rhythm_ai/rhythm_features_merged.parquet" ]; then
        SIZE=$(du -h "${BASE_DIR}/output/rhythm_ai/rhythm_features_merged.parquet" | cut -f1)
        log ""
        log "✅ Found existing rhythm_features_merged.parquet (${SIZE})"
        
        # レコード数確認
        python3 - << 'PYEOF' || true
import pandas as pd
from pathlib import Path
import os

base_dir = os.environ.get('BASE_DIR', '.')
parquet_path = Path(base_dir) / 'output/rhythm_ai/rhythm_features_merged.parquet'
if parquet_path.exists():
    df = pd.read_parquet(parquet_path)
    print(f'   Records: {len(df):,}')
    print(f'   Columns: {len(df.columns)}')
PYEOF
    fi
else
    warn "output/rhythm_ai/ directory not found (will be created)"
fi

log ""
log "================================"
log "Ready to run: bash scripts/run_rhythm_ai_full.sh"
log ""
log "SUMMARY:"
log "  BASE_DIR = ${BASE_DIR}"
