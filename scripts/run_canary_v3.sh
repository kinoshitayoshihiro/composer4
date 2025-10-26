#!/bin/bash
# Canary Playlist Test - v3 Guitar ML Production
# 10曲のフル生成でKPI・品質検証

set -e

CANARY_SONGS=(
  "0051b0117de5e669"
  "00148ee0c0cc0030"
  "0052adf9b2340586"
  "00cd2f3138e7ce5a"
  "00a9b5741812429a"
  "00e3a4b0e57648d8"
  "00f1f1b86b026ab5"
  "013639127636e888"
  "0007b9faeb789b22"
  "00f56635928d8196"
)

WORK_DIR="/Volumes/SSD-SCTU3A/ラジオ用/music_21/composer2-3"
OUTPUT_DIR="${WORK_DIR}/midi_out/canary_v3"
LOG_FILE="${WORK_DIR}/logs/canary_v3_$(date +%Y%m%d_%H%M%S).log"

cd "$WORK_DIR" || exit 1

echo "=== Canary Playlist Test - v3 Guitar ML Production ===" | tee -a "$LOG_FILE"
echo "Start: $(date)" | tee -a "$LOG_FILE"
echo "Songs: ${#CANARY_SONGS[@]}" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# 出力ディレクトリ作成
mkdir -p "$OUTPUT_DIR"
mkdir -p logs

# 各曲を生成
SUCCESS_COUNT=0
FAILED_SONGS=()

for song_id in "${CANARY_SONGS[@]}"; do
  echo "----------------------------------------" | tee -a "$LOG_FILE"
  echo "Song: $song_id" | tee -a "$LOG_FILE"
  echo "Start: $(date +%H:%M:%S)" | tee -a "$LOG_FILE"
  
  # chordmap パス構築
  CHORDMAP="data/processed/${song_id}_chordmap.yaml"
  
  if [ ! -f "$CHORDMAP" ]; then
    echo "⚠️  SKIP: chordmap not found" | tee -a "$LOG_FILE"
    FAILED_SONGS+=("$song_id (no chordmap)")
    continue
  fi
  
  # modular_composer.py 実行
  .venv311/bin/python modular_composer.py \
    --main-cfg config/canary_v3_test.yml \
    --chordmap "$CHORDMAP" \
    --output-dir "$OUTPUT_DIR/${song_id}" \
    --song-id "$song_id" \
    2>&1 | tee -a "$LOG_FILE" || {
      echo "❌ FAILED: generation error" | tee -a "$LOG_FILE"
      FAILED_SONGS+=("$song_id (generation error)")
      continue
    }
  
  SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
  echo "✓ SUCCESS ($SUCCESS_COUNT/10)" | tee -a "$LOG_FILE"
  echo "End: $(date +%H:%M:%S)" | tee -a "$LOG_FILE"
done

echo "" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"
echo "Canary Test Complete" | tee -a "$LOG_FILE"
echo "Success: $SUCCESS_COUNT/10" | tee -a "$LOG_FILE"

if [ ${#FAILED_SONGS[@]} -gt 0 ]; then
  echo "" | tee -a "$LOG_FILE"
  echo "Failed songs:" | tee -a "$LOG_FILE"
  for fail in "${FAILED_SONGS[@]}"; do
    echo "  - $fail" | tee -a "$LOG_FILE"
  done
fi

echo "" | tee -a "$LOG_FILE"
echo "Output: $OUTPUT_DIR" | tee -a "$LOG_FILE"
echo "Log: $LOG_FILE" | tee -a "$LOG_FILE"
echo "End: $(date)" | tee -a "$LOG_FILE"

# KPI集計スクリプト呼び出し（存在すれば）
if [ -f "scripts/analyze_canary_kpi.py" ]; then
  echo "" | tee -a "$LOG_FILE"
  echo "=== KPI Analysis ===" | tee -a "$LOG_FILE"
  .venv311/bin/python scripts/analyze_canary_kpi.py \
    --output-dir "$OUTPUT_DIR" \
    2>&1 | tee -a "$LOG_FILE"
fi

echo "" | tee -a "$LOG_FILE"
echo "✓ Canary test completed: $SUCCESS_COUNT/10 songs generated" | tee -a "$LOG_FILE"
