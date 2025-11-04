#!/bin/bash
# KPI監視自動実行スクリプト（cron用）

set -e

WORK_DIR="/Volumes/SSD-SCTU3A/ラジオ用/music_21/composer2-3"
PYTHON_BIN="${WORK_DIR}/.venv311/bin/python"
LOG_DIR="${WORK_DIR}/logs"
MONITORING_DIR="${WORK_DIR}/monitoring"
METRICS_FILE="${MONITORING_DIR}/metrics.prom"
STATS_FILE="${MONITORING_DIR}/kpi_stats.json"
ALERT_LOG="${MONITORING_DIR}/alerts.log"

cd "$WORK_DIR" || exit 1

# ログディレクトリ作成
mkdir -p "$MONITORING_DIR"

# KPI収集実行
echo "=== KPI Collection Started: $(date) ===" | tee -a "$ALERT_LOG"

$PYTHON_BIN monitoring/kpi_collector.py \
  --log-dir "$LOG_DIR" \
  --output-prom "$METRICS_FILE" \
  --output-json "$STATS_FILE" \
  2>&1 | tee -a "$ALERT_LOG"

EXIT_CODE=${PIPESTATUS[0]}

if [ $EXIT_CODE -ne 0 ]; then
  echo "❌ KPI Collection FAILED: $(date)" | tee -a "$ALERT_LOG"
  
  # Slackアラート送信（オプション）
  if [ -n "$SLACK_WEBHOOK_URL" ]; then
    curl -X POST "$SLACK_WEBHOOK_URL" \
      -H 'Content-Type: application/json' \
      -d "{\"text\":\"🚨 Guitar v3 KPI Collection Failed\"}" \
      2>/dev/null || true
  fi
  
  exit 1
fi

echo "✓ KPI Collection SUCCESS: $(date)" | tee -a "$ALERT_LOG"

# KPIゲート判定（JSONから読み取り）
if [ -f "$STATS_FILE" ]; then
  ACCENT_SCORE=$(jq -r '.accent_score.mean' "$STATS_FILE")
  CHORD_FIT=$(jq -r '.chord_fit.mean' "$STATS_FILE")
  ML_USAGE=$(jq -r '.ml_usage.rate' "$STATS_FILE")
  SAFETY_FALLBACK=$(jq -r '.safety_fallback.rate' "$STATS_FILE")
  
  echo "" | tee -a "$ALERT_LOG"
  echo "Current KPIs:" | tee -a "$ALERT_LOG"
  echo "  Accent Score: ${ACCENT_SCORE}" | tee -a "$ALERT_LOG"
  echo "  Chord Fit: ${CHORD_FIT}" | tee -a "$ALERT_LOG"
  echo "  ML Usage: ${ML_USAGE}" | tee -a "$ALERT_LOG"
  echo "  Safety Fallback: ${SAFETY_FALLBACK}" | tee -a "$ALERT_LOG"
  
  # アラート判定（bashでは浮動小数点比較が難しいので、bcを使用）
  if command -v bc &> /dev/null; then
    if [ $(echo "$ACCENT_SCORE < 0.70" | bc -l) -eq 1 ]; then
      echo "⚠️  WARNING: Accent Score below 70%" | tee -a "$ALERT_LOG"
      
      if [ -n "$SLACK_WEBHOOK_URL" ]; then
        curl -X POST "$SLACK_WEBHOOK_URL" \
          -H 'Content-Type: application/json' \
          -d "{\"text\":\"⚠️ Guitar v3 Accent Score: ${ACCENT_SCORE} (warning <70%)\"}" \
          2>/dev/null || true
      fi
    fi
    
    if [ $(echo "$ACCENT_SCORE < 0.65" | bc -l) -eq 1 ]; then
      echo "🚨 CRITICAL: Accent Score below 65%" | tee -a "$ALERT_LOG"
      
      if [ -n "$SLACK_WEBHOOK_URL" ]; then
        curl -X POST "$SLACK_WEBHOOK_URL" \
          -H 'Content-Type: application/json' \
          -d "{\"text\":\"🚨 CRITICAL: Guitar v3 Accent Score: ${ACCENT_SCORE} (<65%)\"}" \
          2>/dev/null || true
      fi
    fi
    
    if [ $(echo "$ML_USAGE < 0.80" | bc -l) -eq 1 ]; then
      echo "⚠️  WARNING: ML Usage below 80%" | tee -a "$ALERT_LOG"
      
      if [ -n "$SLACK_WEBHOOK_URL" ]; then
        curl -X POST "$SLACK_WEBHOOK_URL" \
          -H 'Content-Type: application/json' \
          -d "{\"text\":\"⚠️ Guitar v3 ML Usage: ${ML_USAGE} (warning <80%)\"}" \
          2>/dev/null || true
      fi
    fi
    
    if [ $(echo "$SAFETY_FALLBACK > 0.10" | bc -l) -eq 1 ]; then
      echo "⚠️  WARNING: Safety Fallback above 10%" | tee -a "$ALERT_LOG"
      
      if [ -n "$SLACK_WEBHOOK_URL" ]; then
        curl -X POST "$SLACK_WEBHOOK_URL" \
          -H 'Content-Type: application/json' \
          -d "{\"text\":\"⚠️ Guitar v3 Safety Fallback: ${SAFETY_FALLBACK} (>10%)\"}" \
          2>/dev/null || true
      fi
    fi
  fi
fi

# Prometheusファイル配信（Prometheusのfile_sd_configsディレクトリにコピー）
# PROMETHEUS_TEXTFILE_DIR="/var/lib/prometheus/textfile_collector"
# if [ -d "$PROMETHEUS_TEXTFILE_DIR" ]; then
#   cp "$METRICS_FILE" "$PROMETHEUS_TEXTFILE_DIR/guitar_v3_kpi.prom"
#   echo "✓ Metrics copied to Prometheus textfile collector" | tee -a "$ALERT_LOG"
# fi

echo "=== KPI Collection Completed: $(date) ===" | tee -a "$ALERT_LOG"
echo "" | tee -a "$ALERT_LOG"
