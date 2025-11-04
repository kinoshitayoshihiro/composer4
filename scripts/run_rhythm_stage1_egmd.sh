#!/usr/bin/env bash
set -euo pipefail

#############################################
# Rhythm AI Stage1 Cleaning - E-GMD v1.0.0
# 45,537 MIDI files (.midi extension)
# 2-level subfolder structure:
#   e-gmd-v1.0.0/drummer{1,3-10}/session{1-3,eval_session}/
#############################################

BASE_DIR="/Volumes/SSD-SCTU3A/ラジオ用/music_21/composer2-3"
cd "$BASE_DIR" || exit 1

EGMD_DIR="data/Los-Angeles-MIDI/LOCAL_LAMDA/rhythmAI/e-gmd-v1.0.0"
OUTPUT_DIR="output/rhythm_ai"
LOG_DIR="logs"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/rhythm_stage1_egmd_${TIMESTAMP}.log"

mkdir -p "$LOG_DIR" "$OUTPUT_DIR"

log() {
  echo "[$(date +'%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"
}

log "=========================================="
log "🥁 Rhythm AI Stage1 Cleaning - E-GMD"
log "=========================================="
log "Base Dir: $BASE_DIR"
log "E-GMD Dir: $EGMD_DIR"
log "Output Dir: $OUTPUT_DIR"
log "Log File: $LOG_FILE"
log ""

## ========== 1. Verify E-GMD MIDI files ==========
log "📂 1. Verifying E-GMD MIDI files..."
MIDI_COUNT=$(find "$EGMD_DIR" -type f -name "*.midi" | wc -l | xargs)
log "✅ Found $MIDI_COUNT MIDI files"

if [ "$MIDI_COUNT" -eq 0 ]; then
  log "❌ ERROR: No MIDI files found"
  exit 1
fi

## ========== 2. Stage1 Cleaning - E-GMD ==========
log "🧹 2. Running Stage1 Cleaning for E-GMD..."
log "   - Input: $EGMD_DIR (45,537 files)"
log "   - Output: $OUTPUT_DIR/egmd_cleaned"
log "   - Metadata: $OUTPUT_DIR/egmd_metadata"

python3 scripts/clean_egmd_simple.py \
  --input-dir "$EGMD_DIR" \
  --output-cleaned "$OUTPUT_DIR/egmd_cleaned" \
  --output-metadata "$OUTPUT_DIR/egmd_metadata" \
  --dataset-name egmd \
  --file-extension .midi \
  --max-workers 8 \
  --shard-size 500 \
  --verbose \
  2>&1 | tee -a "${LOG_FILE}"

if [ $? -ne 0 ]; then
  log "❌ ERROR: Stage1 cleaning failed"
  exit 1
fi

log ""
log "✅ Stage1 Cleaning - E-GMD COMPLETE"
log ""

## ========== 3. Verify Output ==========
log "📊 3. Verifying output..."

if [ -f "$OUTPUT_DIR/egmd_metadata/drums_index.pkl" ]; then
  log "✅ Metadata index created: egmd_metadata/drums_index.pkl"
else
  log "⚠️  WARNING: Metadata index not found"
fi

CLEANED_COUNT=$(find "$OUTPUT_DIR/egmd_cleaned" -type f -name "*.mid" 2>/dev/null | wc -l | xargs)
log "✅ Cleaned MIDI files: $CLEANED_COUNT"

SHARD_COUNT=$(find "$OUTPUT_DIR/egmd_metadata" -type f -name "drums_*.pkl" 2>/dev/null | wc -l | xargs)
log "✅ Metadata shards: $SHARD_COUNT"

log ""
log "=========================================="
log "🎉 E-GMD Stage1 Complete!"
log "=========================================="
log "Next steps:"
log "  1. Run Stage2 feature extraction:"
log "     bash scripts/run_rhythm_stage2_egmd.sh"
log "  2. Merge with drumclean + groove datasets"
log "  3. Train ML model on merged dataset (~97,600 records)"
log ""
