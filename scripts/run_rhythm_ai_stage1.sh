#!/bin/bash
# scripts/run_rhythm_ai_stage1.sh
# Rhythm AI用ドラムMIDI全体クリーニング

set -e

BASE_DIR="/Volumes/SSD-SCTU3A/ラジオ用/music_21/composer2-3"
cd "${BASE_DIR}"

source .venv311/bin/activate

LOG_FILE="logs/rhythm_ai_stage1_$(date +%Y%m%d_%H%M%S).log"
mkdir -p logs

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "${LOG_FILE}"
}

log "========================================="
log "Rhythm AI Stage1 Processing Started"
log "========================================="
log "Log File: ${LOG_FILE}"
log ""

# ========== 事前チェック ==========
log "📊 Pre-flight Check"
log "-------------------"

DRUMCLEAN_MIDI_COUNT=$(find data/Los-Angeles-MIDI/LOCAL_LAMDA/rhythmAI/drumclean_midi \
  \( -name "*.mid" -o -name "*.midi" \) -type f 2>/dev/null | wc -l | tr -d ' ')

GROOVE_MIDI_COUNT=$(find data/Los-Angeles-MIDI/LOCAL_LAMDA/rhythmAI/groove \
  \( -name "*.mid" -o -name "*.midi" \) -type f 2>/dev/null | wc -l | tr -d ' ')

GROOVE_WAV_COUNT=$(find data/Los-Angeles-MIDI/LOCAL_LAMDA/rhythmAI/groove \
  \( -name "*.wav" -o -name "*.mp3" \) -type f 2>/dev/null | wc -l | tr -d ' ')

log "drumclean_midi: ${DRUMCLEAN_MIDI_COUNT} MIDI files"
log "groove:         ${GROOVE_MIDI_COUNT} MIDI files, ${GROOVE_WAV_COUNT} audio files"
log ""

# 出力ディレクトリ準備
log "📁 Creating output directories..."
mkdir -p output/rhythm_ai/drumclean_midi
mkdir -p output/rhythm_ai/drumclean_metadata
mkdir -p output/rhythm_ai/drumclean_q
mkdir -p output/rhythm_ai/groove_cleaned
mkdir -p output/rhythm_ai/groove_metadata
mkdir -p output/rhythm_ai/groove_q
log "✅ Directories created"
log ""

# ========== 1. drumclean_midi ==========
log "========================================="
log "1. Processing drumclean_midi"
log "========================================="

if [ "${DRUMCLEAN_MIDI_COUNT}" -gt 0 ]; then
    log "Starting clean_midi.py for drumclean_midi..."
    
    python -m scripts.clean_midi \
      --in data/Los-Angeles-MIDI/LOCAL_LAMDA/rhythmAI/drumclean_midi \
      --out output/rhythm_ai/drumclean_midi \
      --quarantine output/rhythm_ai/drumclean_q \
      --instrument drums \
      --pickle-out output/rhythm_ai/drumclean_metadata \
      --shard-size 5000 \
      --resume \
      --emit-meta-json off \
      --jobs 8 \
      2>&1 | tee -a "${LOG_FILE}"
    
    log "✅ drumclean_midi processing completed"
else
    log "⚠️  No MIDI files found in drumclean_midi, skipping..."
fi

log ""

# ========== 2. groove ==========
log "========================================="
log "2. Processing groove (MIDI only)"
log "========================================="

if [ "${GROOVE_MIDI_COUNT}" -gt 0 ]; then
    log "Starting clean_midi.py for groove..."
    log "Note: WAV files (${GROOVE_WAV_COUNT}) will be ignored"
    log ""
    
    python -m scripts.clean_midi \
      --in data/Los-Angeles-MIDI/LOCAL_LAMDA/rhythmAI/groove \
      --out output/rhythm_ai/groove_cleaned \
      --quarantine output/rhythm_ai/groove_q \
      --instrument drums \
      --pickle-out output/rhythm_ai/groove_metadata \
      --shard-size 5000 \
      --resume \
      --emit-meta-json off \
      --jobs 8 \
      2>&1 | tee -a "${LOG_FILE}"
    
    log "✅ groove processing completed"
else
    log "⚠️  No MIDI files found in groove, skipping..."
fi

log ""

# ========== 3. 統合レポート ==========
log "========================================="
log "Processing Summary"
log "========================================="

# drumclean_midi統計
DRUMCLEAN_CLEANED=$(find output/rhythm_ai/drumclean_midi -name "*.mid*" 2>/dev/null | wc -l | tr -d ' ')
DRUMCLEAN_SHARDS=$(find output/rhythm_ai/drumclean_metadata -name "drums_*.pkl" -not -name "*_index.pkl" 2>/dev/null | wc -l | tr -d ' ')
DRUMCLEAN_QUARANTINE=$(find output/rhythm_ai/drumclean_q -name "*.mid*" 2>/dev/null | wc -l | tr -d ' ')

log "drumclean_midi:"
log "  Input:       ${DRUMCLEAN_MIDI_COUNT} files"
log "  Cleaned:     ${DRUMCLEAN_CLEANED} files"
log "  Quarantined: ${DRUMCLEAN_QUARANTINE} files"
log "  Shards:      ${DRUMCLEAN_SHARDS} pickles"

if [ -f "output/rhythm_ai/drumclean_metadata/drums_index.pkl" ]; then
    log "  Index:       ✅ drums_index.pkl created"
else
    log "  Index:       ❌ drums_index.pkl NOT found"
fi

log ""

# groove統計
GROOVE_CLEANED=$(find output/rhythm_ai/groove_cleaned -name "*.mid*" 2>/dev/null | wc -l | tr -d ' ')
GROOVE_SHARDS=$(find output/rhythm_ai/groove_metadata -name "drums_*.pkl" -not -name "*_index.pkl" 2>/dev/null | wc -l | tr -d ' ')
GROOVE_QUARANTINE=$(find output/rhythm_ai/groove_q -name "*.mid*" 2>/dev/null | wc -l | tr -d ' ')

log "groove:"
log "  Input:       ${GROOVE_MIDI_COUNT} files"
log "  Cleaned:     ${GROOVE_CLEANED} files"
log "  Quarantined: ${GROOVE_QUARANTINE} files"
log "  Shards:      ${GROOVE_SHARDS} pickles"

if [ -f "output/rhythm_ai/groove_metadata/drums_index.pkl" ]; then
    log "  Index:       ✅ drums_index.pkl created"
else
    log "  Index:       ❌ drums_index.pkl NOT found"
fi

log ""

# ========== 4. 次ステップ ==========
log "========================================="
log "Next Steps"
log "========================================="
log ""
log "1. Verify pickle compatibility:"
log "   python verify_stage2_compat.py output/rhythm_ai/drumclean_metadata"
log "   python verify_stage2_compat.py output/rhythm_ai/groove_metadata"
log ""
log "2. Run Stage2 extraction (if needed):"
log "   PYTHONPATH=. python scripts/lamda_stage2_extractor.py \\"
log "       --metadata-index output/rhythm_ai/drumclean_metadata/drums_index.pkl \\"
log "       --input-dir output/rhythm_ai/drumclean_midi \\"
log "       --output-dir output/rhythm_ai/drumclean_stage2"
log ""
log "3. Train ML model:"
log "   python scripts/train_rhythm_baseline.py \\"
log "       --input output/rhythm_ai/drumclean_metadata/drums_index.pkl \\"
log "       --output data/patterns/stage2_drums_rhythm_ai.pickle"
log ""
log "========================================="
log "✅ All processing completed!"
log "========================================="
log "Log saved to: ${LOG_FILE}"
