#!/bin/bash
# ========================================
# 個別データセット実行スクリプト (テスト用)
# ========================================

# 使用例:
#   ./run_single_dataset.sh loops drums 4
#   ./run_single_dataset.sh Los-Angeles-MIDI/MIDIs piano 8

if [ $# -lt 3 ]; then
  echo "Usage: $0 <dataset_path> <instrument> <jobs>"
  echo "Example: $0 loops drums 4"
  exit 1
fi

DATASET_PATH=$1
INSTRUMENT=$2
JOBS=$3

BASE_DIR="/Volumes/SSD-SCTU3A/ラジオ用/music_21/composer2-3"
DATA_DIR="${BASE_DIR}/data"
SCRIPTS_DIR="${BASE_DIR}/scripts"

# データセット名を抽出
DATASET_NAME=$(basename "${DATASET_PATH}")

# 出力パス
OUTPUT_DIR="${BASE_DIR}/data/cleaned/${DATASET_NAME}"
QUARANTINE_DIR="${BASE_DIR}/data/quarantine/${DATASET_NAME}"
REPORT_PATH="${BASE_DIR}/reports/${DATASET_NAME}_${INSTRUMENT}_validation.json"
SUMMARY_PATH="${BASE_DIR}/reports/${DATASET_NAME}_${INSTRUMENT}_summary.jsonl"

# ========================================
# 1. ドライラン (件数確認)
# ========================================
echo "🔍 Dry-run: Checking MIDI files..."
python "${SCRIPTS_DIR}/clean_midi.py" \
  --in "${DATA_DIR}/${DATASET_PATH}" \
  --out "${OUTPUT_DIR}" \
  --instrument "${INSTRUMENT}" \
  --quarantine "${QUARANTINE_DIR}" \
  --dry-run

echo ""
read -p "👆 Continue with cleaning? (y/n): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
  echo "❌ Cancelled."
  exit 0
fi

# ========================================
# 2. クリーニング実行
# ========================================
echo "🧹 Cleaning ${DATASET_NAME} (${INSTRUMENT})..."
python "${SCRIPTS_DIR}/clean_midi.py" \
  --in "${DATA_DIR}/${DATASET_PATH}" \
  --out "${OUTPUT_DIR}" \
  --instrument "${INSTRUMENT}" \
  --quarantine "${QUARANTINE_DIR}" \
  --jobs "${JOBS}" \
  --seed "${DATASET_NAME}-${INSTRUMENT}-v1"

# ========================================
# 3. 検証
# ========================================
echo "✅ Validating cleaned files..."
python "${SCRIPTS_DIR}/validate_and_gate.py" \
  --in "${OUTPUT_DIR}" \
  --gates "${BASE_DIR}/configs/quality_gates/quality_gates.yaml" \
  --report "${REPORT_PATH}" \
  --summary "${SUMMARY_PATH}" \
  --fail-on-critical

# ========================================
# 4. サマリ表示
# ========================================
echo ""
echo "📊 Summary:"
echo "  Cleaned: ${OUTPUT_DIR}"
echo "  Quarantined: ${QUARANTINE_DIR}"
echo "  Report: ${REPORT_PATH}"

# JSONLから統計抽出
if [ -f "${SUMMARY_PATH}" ]; then
  TOTAL=$(wc -l < "${SUMMARY_PATH}" | tr -d ' ')
  PASSED=$(grep '"passed":true' "${SUMMARY_PATH}" | wc -l | tr -d ' ')
  CRITICAL=$(grep '"is_critical":true' "${SUMMARY_PATH}" | wc -l | tr -d ' ')
  
  echo ""
  echo "  Total files: ${TOTAL}"
  echo "  ✅ Passed: ${PASSED}"
  echo "  ❌ Critical: ${CRITICAL}"
fi

echo ""
echo "🎉 Done!"
