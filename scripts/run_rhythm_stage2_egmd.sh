#!/usr/bin/env bash
set -euo pipefail

#############################################
# Rhythm AI Stage2 Feature Extraction - E-GMD
# Expected: ~45,000 cleaned records
#############################################

BASE_DIR="/Volumes/SSD-SCTU3A/ラジオ用/music_21/composer2-3"
cd "$BASE_DIR" || exit 1

OUTPUT_DIR="output/rhythm_ai"
LOG_DIR="logs"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/rhythm_stage2_egmd_${TIMESTAMP}.log"
EXTRACTOR_PATH="scripts/rhythm_stage2_extractor.py"
CONFIG_PATH="configs/rhythm_stage2.yaml"

mkdir -p "$LOG_DIR"

log() {
  echo "[$(date +'%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"
}

log "=========================================="
log "🎵 Rhythm AI Stage2 - E-GMD"
log "=========================================="
log "Base Dir: $BASE_DIR"
log "Output Dir: $OUTPUT_DIR"
log "Extractor: $EXTRACTOR_PATH"
log "Config: $CONFIG_PATH"
log "Log File: $LOG_FILE"
log ""

## ========== 1. Verify E-GMD Stage1 outputs ==========
log "📂 1. Verifying E-GMD Stage1 outputs..."

if [ ! -f "$OUTPUT_DIR/egmd_metadata/drums_index.pkl" ]; then
  log "❌ ERROR: E-GMD metadata index not found"
  log "   Please run: bash scripts/run_rhythm_stage1_egmd.sh"
  exit 1
fi

if [ ! -d "$OUTPUT_DIR/egmd_cleaned" ]; then
  log "❌ ERROR: E-GMD cleaned directory not found"
  exit 1
fi

CLEANED_COUNT=$(find "$OUTPUT_DIR/egmd_cleaned" -type f -name "*.mid" 2>/dev/null | wc -l | xargs)
log "✅ E-GMD cleaned files: $CLEANED_COUNT"

## ========== 2. E-GMD Stage2 Feature Extraction ==========
log "🚀 2. Running E-GMD Stage2 feature extraction..."
log "   - Input: $OUTPUT_DIR/egmd_cleaned ($CLEANED_COUNT files)"
log "   - Metadata: $OUTPUT_DIR/egmd_metadata/drums_index.pkl"
log "   - Output: $OUTPUT_DIR/egmd_stage2"

python3 "$EXTRACTOR_PATH" \
  --lamda-index "$OUTPUT_DIR/egmd_metadata/drums_index.pkl" \
  --input-dir "$OUTPUT_DIR/egmd_cleaned" \
  --output-dir "$OUTPUT_DIR/egmd_stage2" \
  --config "$CONFIG_PATH" \
  --verbose \
  2>&1 | tee -a "$LOG_FILE"

if [ $? -ne 0 ]; then
  log "❌ ERROR: Stage2 feature extraction failed"
  exit 1
fi

log ""
log "✅ E-GMD Stage2 Feature Extraction COMPLETE"
log ""

## ========== 3. Verify Output ==========
log "📊 3. Verifying output..."

PARQUET_PATH="$OUTPUT_DIR/egmd_stage2/rhythm_features.parquet"
if [ -f "$PARQUET_PATH" ]; then
  RECORD_COUNT=$(python3 -c "import pandas as pd; print(len(pd.read_parquet('$PARQUET_PATH')))")
  log "✅ Parquet records: $RECORD_COUNT"
  
  COLS=$(python3 -c "import pandas as pd; print(list(pd.read_parquet('$PARQUET_PATH').columns))")
  log "✅ Columns: $COLS"
else
  log "⚠️  WARNING: Parquet file not found"
fi

log ""
log "=========================================="
log "🎉 E-GMD Stage2 Complete!"
log "=========================================="
log "Next steps:"
log "  1. Merge with drumclean + groove datasets:"
log "     bash scripts/merge_rhythm_datasets.sh"
log "  2. Train ML model on merged dataset (~97,600 records)"
log ""
