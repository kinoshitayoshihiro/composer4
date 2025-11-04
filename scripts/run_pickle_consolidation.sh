#!/bin/bash
# Pickle統合 + Rhythm AI WAV処理実行スクリプト

set -e

BASE_DIR="/Volumes/SSD-SCTU3A/ラジオ用/music_21/composer2-3"
cd "${BASE_DIR}"

source .venv311/bin/activate

LOG_FILE="logs/pickle_consolidation_$(date +%Y%m%d_%H%M%S).log"
mkdir -p logs

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "${LOG_FILE}"
}

log "========================================="
log "Pickle統合 + Rhythm AI WAV処理"
log "========================================="

# ========== Step 1: Rhythm AI WAV処理 ==========
log ""
log "📂 Step 1: Rhythm AI WAV Processing..."

python scripts/clean_wav_rhythm.py \
  --in data/Los-Angeles-MIDI/LOCAL_LAMDA/rhythmAI/groove \
  --out output/rhythm_wav/groove_cleaned \
  --quarantine output/rhythm_wav/groove_q \
  --pickle-out output/rhythm_wav/groove_metadata \
  --jobs 8 \
  --verbose \
  2>&1 | tee -a "${LOG_FILE}"

log "✅ Rhythm AI WAV processing completed"

# ========== Step 2: Pickle統合 ==========
log ""
log "📦 Step 2: Consolidating pickles to LOCAL_LAMDA/pickles/..."

python scripts/consolidate_pickles.py \
  --output-dir output \
  --target-dir data/Los-Angeles-MIDI/LOCAL_LAMDA/pickles \
  --backup \
  --verbose \
  2>&1 | tee -a "${LOG_FILE}"

log "✅ Pickle consolidation completed"

# ========== Step 3: output/ クリーンアップ（オプション） ==========
log ""
log "🧹 Step 3: Cleanup output/ directory (optional)..."

read -p "Delete output/ directory? (y/N): " confirm

if [[ "$confirm" == "y" || "$confirm" == "Y" ]]; then
    log "Deleting output/ directory..."
    rm -rf output
    log "✅ output/ deleted"
else
    log "⚠️  output/ kept (manual cleanup required)"
fi

# ========== サマリー ==========
log ""
log "========================================="
log "Summary"
log "========================================="

python -c "
import json
from pathlib import Path

summary_path = Path('data/Los-Angeles-MIDI/LOCAL_LAMDA/pickles/consolidation_summary.json')

if summary_path.exists():
    with open(summary_path) as f:
        s = json.load(f)
    
    print(f'\nTotal pickles:  {s[\"total_pickles\"]}')
    print(f'Total records:  {s[\"total_records\"]}')
    print(f'\nCategories:')
    
    for cat, stats in s['stats'].items():
        print(f'  {cat}:')
        print(f'    Pickles: {stats[\"pickles_moved\"]}')
        print(f'    Records: {stats[\"total_records\"]}')
else:
    print('⚠️ Summary not found')
"

log "✅ All processing completed!"
log "========================================="
log "Log: ${LOG_FILE}"
