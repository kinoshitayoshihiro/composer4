#!/bin/bash
# Canary Test - v3 Guitar ML Production Quality Check
# 本番設定で10曲のKPI評価（実MIDI生成なし、評価のみ）

set -e

WORK_DIR="/Volumes/SSD-SCTU3A/ラジオ用/music_21/composer2-3"
LOG_FILE="${WORK_DIR}/logs/canary_kpi_$(date +%Y%m%d_%H%M%S).log"

cd "$WORK_DIR" || exit 1

echo "=== Canary KPI Test - v3 Guitar ML Production ===" | tee "$LOG_FILE"
echo "Start: $(date)" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# v3単独評価（本番設定）
echo "Running v3 evaluation with production config..." | tee -a "$LOG_FILE"
echo "  - threshold: 0.0 (ML always-on)" | tee -a "$LOG_FILE"
echo "  - w_proba: 1.00 (rerank disabled)" | tee -a "$LOG_FILE"
echo "  - pickle: stage2_guitar_v3_meta.pickle" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

export PYTHONPATH="$(pwd):$PYTHONPATH"

.venv311/bin/python scripts/ab_test_guitar_v3.py \
  --v3-only \
  --num-songs 10 \
  --v3-pickle data/patterns/stage2_guitar_v3_meta.pickle \
  --output data/canary_kpi_v3_production.csv \
  --conf-thresh 0.00 \
  --w-proba 1.00 \
  --w-accent 0.00 \
  --w-density 0.00 \
  --w-section 0.00 \
  2>&1 | tee -a "$LOG_FILE"

echo "" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"
echo "Canary KPI Test Complete" | tee -a "$LOG_FILE"
echo "Output: data/canary_kpi_v3_production.csv" | tee -a "$LOG_FILE"
echo "Log: $LOG_FILE" | tee -a "$LOG_FILE"
echo "End: $(date)" | tee -a "$LOG_FILE"

# 結果サマリー表示
if [ -f data/canary_kpi_v3_production.csv ]; then
  echo "" | tee -a "$LOG_FILE"
  echo "=== Result Summary ===" | tee -a "$LOG_FILE"
  tail -5 data/canary_kpi_v3_production.csv | tee -a "$LOG_FILE"
fi
