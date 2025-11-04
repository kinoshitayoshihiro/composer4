#!/bin/bash
# scripts/run_rhythm_ai_full.sh
# Rhythm AI完全パイプライン: Stage1クリーニング → Pickle生成 → Stage2特徴量抽出

set -e

BASE_DIR="/Volumes/SSD-SCTU3A/ラジオ用/music_21/composer2-3"
cd "${BASE_DIR}"

# 仮想環境のPythonを使用
PYTHON="${BASE_DIR}/.venv311/bin/python"

LOG_FILE="logs/rhythm_ai_full_$(date +%Y%m%d_%H%M%S).log"
mkdir -p logs output/rhythm_ai

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "${LOG_FILE}"
}

log "========================================="
log "Rhythm AI Full Pipeline"
log "Stage1 → Pickle → Stage2"
log "========================================="
log ""

# ========================================
# Stage1: drumclean_midi（既存クリーン想定）
# ========================================
log "📂 [1/5] Processing drumclean_midi..."

DRUMCLEAN_INPUT="data/Los-Angeles-MIDI/LOCAL_LAMDA/rhythmAI/drumclean_midi"
DRUMCLEAN_OUTPUT="output/rhythm_ai/drumclean_midi"
DRUMCLEAN_PICKLE="output/rhythm_ai/drumclean_metadata"
DRUMCLEAN_Q="output/rhythm_ai/drumclean_q"

# ファイル数確認
DRUMCLEAN_COUNT=$(find "${DRUMCLEAN_INPUT}" \( -name "*.mid" -o -name "*.midi" \) -type f 2>/dev/null | wc -l | tr -d ' ')
log "   Found ${DRUMCLEAN_COUNT} MIDI files"

if [ "${DRUMCLEAN_COUNT}" -eq 0 ]; then
    log "⚠️  No MIDI files found in drumclean_midi, skipping"
else
    ${PYTHON} -m scripts.clean_midi \
      --in "${DRUMCLEAN_INPUT}" \
      --out "${DRUMCLEAN_OUTPUT}" \
      --quarantine "${DRUMCLEAN_Q}" \
      --instrument drums \
      --pickle-out "${DRUMCLEAN_PICKLE}" \
      --shard-size 5000 \
      --resume \
      --emit-meta-json off \
      --jobs 8 \
      2>&1 | tee -a "${LOG_FILE}"
    
    log "✅ drumclean_midi Stage1 completed"
fi

log ""

# ========================================
# Stage1: groove（WAV混在）
# ========================================
log "📂 [2/5] Processing groove (MIDI extraction)..."

GROOVE_INPUT="data/Los-Angeles-MIDI/LOCAL_LAMDA/rhythmAI/groove"
GROOVE_OUTPUT="output/rhythm_ai/groove_cleaned"
GROOVE_PICKLE="output/rhythm_ai/groove_metadata"
GROOVE_Q="output/rhythm_ai/groove_q"

# MIDI抽出確認
GROOVE_MIDI_COUNT=$(find "${GROOVE_INPUT}" \( -name "*.mid" -o -name "*.midi" \) -type f 2>/dev/null | wc -l | tr -d ' ')
GROOVE_WAV_COUNT=$(find "${GROOVE_INPUT}" \( -name "*.wav" -o -name "*.mp3" \) -type f 2>/dev/null | wc -l | tr -d ' ')

log "   Found ${GROOVE_MIDI_COUNT} MIDI files"
log "   Found ${GROOVE_WAV_COUNT} audio files (ignored)"

if [ "${GROOVE_MIDI_COUNT}" -eq 0 ]; then
    log "⚠️  No MIDI files found in groove, skipping"
else
    ${PYTHON} -m scripts.clean_midi \
      --in "${GROOVE_INPUT}" \
      --out "${GROOVE_OUTPUT}" \
      --quarantine "${GROOVE_Q}" \
      --instrument drums \
      --pickle-out "${GROOVE_PICKLE}" \
      --shard-size 5000 \
      --resume \
      --emit-meta-json off \
      --jobs 8 \
      2>&1 | tee -a "${LOG_FILE}"
    
    log "✅ groove Stage1 completed"
fi

log ""

# ========================================
# Stage2: drumclean_midi特徴量抽出
# ========================================
log "📂 [3/5] Stage2 feature extraction (drumclean_midi)..."

if [ -f "${DRUMCLEAN_PICKLE}/drums_index.pkl" ]; then
    ${PYTHON} scripts/rhythm_stage2_extractor.py \
      --lamda-index "${DRUMCLEAN_PICKLE}/drums_index.pkl" \
      --input-dir "${DRUMCLEAN_OUTPUT}" \
      --output-dir "output/rhythm_ai/drumclean_stage2" \
      --config configs/rhythm_stage2.yaml \
      --verbose \
      2>&1 | tee -a "${LOG_FILE}"
    
    log "✅ drumclean_midi Stage2 completed"
else
    log "⚠️  drums_index.pkl not found, skipping Stage2 for drumclean_midi"
fi

log ""

# ========================================
# Stage2: groove特徴量抽出
# ========================================
log "📂 [4/5] Stage2 feature extraction (groove)..."

if [ -f "${GROOVE_PICKLE}/drums_index.pkl" ]; then
    ${PYTHON} scripts/rhythm_stage2_extractor.py \
      --lamda-index "${GROOVE_PICKLE}/drums_index.pkl" \
      --input-dir "${GROOVE_OUTPUT}" \
      --output-dir "output/rhythm_ai/groove_stage2" \
      --config configs/rhythm_stage2.yaml \
      --verbose \
      2>&1 | tee -a "${LOG_FILE}"
    
    log "✅ groove Stage2 completed"
else
    log "⚠️  drums_index.pkl not found, skipping Stage2 for groove"
fi

log ""

# ========================================
# Parquet統合
# ========================================
log "📂 [5/5] Merging Stage2 results..."

${PYTHON} << 'PYEOF'
import pandas as pd
from pathlib import Path

files = [
    'output/rhythm_ai/drumclean_stage2/rhythm_features.parquet',
    'output/rhythm_ai/groove_stage2/rhythm_features.parquet'
]

dfs = []
for f in files:
    p = Path(f)
    if p.exists():
        print(f'Loading {f}...')
        dfs.append(pd.read_parquet(p))

if dfs:
    df_merged = pd.concat(dfs, ignore_index=True)
    
    # 重複削除
    df_merged = df_merged.drop_duplicates(subset=['loop_id'], keep='first')
    
    # 保存
    output_path = Path('output/rhythm_ai/rhythm_features_merged.parquet')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_merged.to_parquet(output_path, compression='snappy', index=False)
    
    print(f'✅ Merged: {len(df_merged)} records')
    print(f'💾 Saved: {output_path}')
else:
    print('⚠️  No parquet files found')
PYEOF

log "✅ Merge completed"

log ""
log "========================================="
log "Summary"
log "========================================="

# 統計表示
${PYTHON} << 'PYEOF'
import pandas as pd
from pathlib import Path

parquet_path = Path('output/rhythm_ai/rhythm_features_merged.parquet')

if parquet_path.exists():
    df = pd.read_parquet(parquet_path)
    
    print(f'Total records: {len(df)}')
    print(f'')
    print('Tempo:')
    print(f'  Mean: {df["tempo_bpm"].mean():.1f} BPM')
    print(f'  Std:  {df["tempo_bpm"].std():.1f} BPM')
    print(f'  Range: {df["tempo_bpm"].min():.1f} - {df["tempo_bpm"].max():.1f}')
    print(f'')
    print('Groove:')
    print(f'  Swing:     {df["swing_pct"].mean():.1f}%')
    print(f'  Backbeat:  {df["backbeat_strength"].mean():.3f}')
    print(f'')
    print('KPIs:')
    print(f'  Kick Downbeat:   {df["kick_downbeat_rate"].mean():.3f}')
    print(f'  Snare Backbeat:  {df["snare_backbeat_rate"].mean():.3f}')
    print(f'  Hat Density:     {df["hat_density"].mean():.1f}')
    print(f'')
    print('Quality Filter:')
    df_filtered = df[
        (df['kick_downbeat_rate'] >= 0.75) &
        (df['snare_backbeat_rate'] >= 0.75) &
        (df['hat_density'] <= 20.0)
    ]
    pass_rate = len(df_filtered) / len(df) * 100
    print(f'  Pass Rate: {pass_rate:.1f}% ({len(df_filtered)}/{len(df)})')
else:
    print('⚠️  Merged parquet not found')
PYEOF

log ""
log "========================================="
log "Next Steps"
log "========================================="
log "1. Verify output:"
log "   ls -lh output/rhythm_ai/rhythm_features_merged.parquet"
log ""
log "2. Train model:"
log "   ${PYTHON} scripts/train_rhythm_baseline.py \\"
log "       --lamda-parquet output/rhythm_ai/rhythm_features_merged.parquet \\"
log "       --output data/patterns/stage2_drums_v1.pickle"
log ""
log "3. Test generation:"
log "   ${PYTHON} scripts/drums_generator_stage2.py \\"
log "       --song-package path/to/song_package.yaml \\"
log "       --out midi_out/drums/"
log ""
log "✅ Rhythm AI Full Pipeline completed!"
log "========================================="
log "Log: ${LOG_FILE}"
