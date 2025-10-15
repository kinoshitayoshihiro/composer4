#!/bin/bash
# ========================================
# 全データセットMIDIクリーニング実行スクリプト
# ========================================

set -e  # エラーで停止

# ベースパス
BASE_DIR="/Volumes/SSD-SCTU3A/ラジオ用/music_21/composer2-3"
DATA_DIR="${BASE_DIR}/data"
SCRIPTS_DIR="${BASE_DIR}/scripts"
OUTPUT_DIR="${BASE_DIR}/data/cleaned"
QUARANTINE_DIR="${BASE_DIR}/data/quarantine"
REPORTS_DIR="${BASE_DIR}/reports"

# 並列数 (マシンスペックに応じて調整)
JOBS=8

# ========================================
# 1. Loops (ドラムループ中心)
# ========================================
echo "🥁 Cleaning Loops dataset..."
python "${SCRIPTS_DIR}/clean_midi.py" \
  --in "${DATA_DIR}/loops" \
  --out "${OUTPUT_DIR}/loops" \
  --instrument drums \
  --quarantine "${QUARANTINE_DIR}/loops" \
  --jobs ${JOBS} \
  --seed "loops-v1"

# ========================================
# 2. Los-Angeles-MIDI (LAMDa - 混合楽器)
# ========================================
echo "🎹 Cleaning LAMDa dataset (Piano)..."
python "${SCRIPTS_DIR}/clean_midi.py" \
  --in "${DATA_DIR}/Los-Angeles-MIDI/MIDIs" \
  --out "${OUTPUT_DIR}/lamda/piano" \
  --instrument piano \
  --quarantine "${QUARANTINE_DIR}/lamda/piano" \
  --jobs ${JOBS} \
  --seed "lamda-piano-v1"

echo "🎸 Cleaning LAMDa dataset (Guitar)..."
python "${SCRIPTS_DIR}/clean_midi.py" \
  --in "${DATA_DIR}/Los-Angeles-MIDI/MIDIs" \
  --out "${OUTPUT_DIR}/lamda/guitar" \
  --instrument guitar \
  --quarantine "${QUARANTINE_DIR}/lamda/guitar" \
  --jobs ${JOBS} \
  --seed "lamda-guitar-v1"

echo "🎻 Cleaning LAMDa dataset (Strings)..."
python "${SCRIPTS_DIR}/clean_midi.py" \
  --in "${DATA_DIR}/Los-Angeles-MIDI/MIDIs" \
  --out "${OUTPUT_DIR}/lamda/strings" \
  --instrument strings \
  --quarantine "${QUARANTINE_DIR}/lamda/strings" \
  --jobs ${JOBS} \
  --seed "lamda-strings-v1"

# ========================================
# 3. XMIDI_Dataset (分析用)
# ========================================
echo "🎵 Cleaning XMIDI dataset (Piano)..."
python "${SCRIPTS_DIR}/clean_midi.py" \
  --in "${DATA_DIR}/XMIDI_Dataset" \
  --out "${OUTPUT_DIR}/xmidi/piano" \
  --instrument piano \
  --quarantine "${QUARANTINE_DIR}/xmidi/piano" \
  --jobs ${JOBS} \
  --seed "xmidi-v1"

# ========================================
# 4. POP909 (ポップス楽曲)
# ========================================
echo "🎤 Cleaning POP909 dataset (Piano)..."
python "${SCRIPTS_DIR}/clean_midi.py" \
  --in "${DATA_DIR}/POP909" \
  --out "${OUTPUT_DIR}/pop909/piano" \
  --instrument piano \
  --quarantine "${QUARANTINE_DIR}/pop909/piano" \
  --jobs ${JOBS} \
  --seed "pop909-v1"

# ========================================
# 5. Slakh2100 (ベーストラック)
# ========================================
echo "🎸 Cleaning Slakh2100 dataset (Bass)..."
python "${SCRIPTS_DIR}/clean_midi.py" \
  --in "${DATA_DIR}/slakh2100_midi" \
  --out "${OUTPUT_DIR}/slakh/bass" \
  --instrument bass \
  --quarantine "${QUARANTINE_DIR}/slakh/bass" \
  --jobs ${JOBS} \
  --seed "slakh-bass-v1"

# ========================================
# 6. 検証フェーズ (Quality Gates)
# ========================================
echo "✅ Running Quality Gates..."

# Loops
python "${SCRIPTS_DIR}/validate_and_gate.py" \
  --in "${OUTPUT_DIR}/loops" \
  --gates "${BASE_DIR}/configs/quality_gates/quality_gates.yaml" \
  --report "${REPORTS_DIR}/loops_validation.json" \
  --summary "${REPORTS_DIR}/loops_summary.jsonl" \
  --fail-on-critical

# LAMDa Piano
python "${SCRIPTS_DIR}/validate_and_gate.py" \
  --in "${OUTPUT_DIR}/lamda/piano" \
  --gates "${BASE_DIR}/configs/quality_gates/quality_gates.yaml" \
  --report "${REPORTS_DIR}/lamda_piano_validation.json" \
  --summary "${REPORTS_DIR}/lamda_piano_summary.jsonl" \
  --fail-on-critical

# POP909
python "${SCRIPTS_DIR}/validate_and_gate.py" \
  --in "${OUTPUT_DIR}/pop909/piano" \
  --gates "${BASE_DIR}/configs/quality_gates/quality_gates.yaml" \
  --report "${REPORTS_DIR}/pop909_validation.json" \
  --summary "${REPORTS_DIR}/pop909_summary.jsonl" \
  --fail-on-critical

# ========================================
# 7. データ分割
# ========================================
echo "📊 Preparing train/val/test splits..."

# LAMDa Piano
python "${SCRIPTS_DIR}/prepare_splits.py" \
  --in "${OUTPUT_DIR}/lamda/piano" \
  --out "${OUTPUT_DIR}/lamda/piano_splits" \
  --seed 42 \
  --min-bucket 5

# POP909 Piano
python "${SCRIPTS_DIR}/prepare_splits.py" \
  --in "${OUTPUT_DIR}/pop909/piano" \
  --out "${OUTPUT_DIR}/pop909/piano_splits" \
  --seed 42 \
  --min-bucket 3

echo "🎉 All cleaning completed!"
echo "📁 Cleaned data: ${OUTPUT_DIR}"
echo "🗑️  Quarantined: ${QUARANTINE_DIR}"
echo "📊 Reports: ${REPORTS_DIR}"
