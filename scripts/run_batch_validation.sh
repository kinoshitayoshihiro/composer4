#!/usr/bin/env bash
# run_batch_validation.sh - 複数song_packageのバッチ検証
# Usage: bash scripts/run_batch_validation.sh [output_csv]

set -Eeuo pipefail

# カラー出力
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log() { echo -e "${BLUE}[$(date +'%H:%M:%S')]${NC} $*"; }
err() { echo -e "${RED}[ERROR]${NC} $*" >&2; }
warn() { echo -e "${YELLOW}[WARN]${NC} $*"; }
ok() { echo -e "${GREEN}[OK]${NC} $*"; }

# 出力CSV（デフォルト: output/batch_validation_results.csv）
OUTPUT_CSV="${1:-output/batch_validation_results.csv}"
mkdir -p "$(dirname "$OUTPUT_CSV")"

# CSVヘッダー作成
echo "song_package,total_bars,pass_bars,fail_bars,warn_bars,pass_rate,fail_rate,warn_rate,section_override_count,safe_kit_recommended,timestamp" > "$OUTPUT_CSV"

log "🎯 Batch Validation Pipeline"
log "============================================================"
log "Output CSV: $OUTPUT_CSV"
log ""

# song_package一覧取得
SONG_PACKAGES=($(find song_packages -name "song_package.yaml" -type f | sed 's|/song_package.yaml||'))

if [ ${#SONG_PACKAGES[@]} -eq 0 ]; then
    err "No song_package found"
    exit 1
fi

log "📂 Found ${#SONG_PACKAGES[@]} song package(s):"
for pkg in "${SONG_PACKAGES[@]}"; do
    log "  - $pkg"
done
log ""

# 各song_packageを処理
SUCCESS_COUNT=0
FAIL_COUNT=0

for SONG_DIR in "${SONG_PACKAGES[@]}"; do
    log "📦 Processing: $SONG_DIR"
    log "------------------------------------------------------------"
    
    # song_package名取得
    SONG_NAME=$(basename "$SONG_DIR")
    PROJECT_NAME=$(basename "$(dirname "$SONG_DIR")")
    PKG_NAME="${PROJECT_NAME}/${SONG_NAME}"
    
    # 生成実行
    if bash scripts/run_song_generation.sh "$SONG_DIR" > /dev/null 2>&1; then
        ok "✅ Generation completed: $PKG_NAME"
    else
        err "❌ Generation failed: $PKG_NAME"
        FAIL_COUNT=$((FAIL_COUNT + 1))
        # 失敗時はCSVに記録（ゼロ値）
        echo "$PKG_NAME,0,0,0,0,0.0,0.0,0.0,0,0,$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$OUTPUT_CSV"
        log ""
        continue
    fi
    
    # kpi_gate_report_postgen.json解析
    REPORT="$SONG_DIR/kpi_gate_report_postgen.json"
    if [ ! -f "$REPORT" ]; then
        err "❌ Report not found: $REPORT"
        FAIL_COUNT=$((FAIL_COUNT + 1))
        echo "$PKG_NAME,0,0,0,0,0.0,0.0,0.0,0,0,$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$OUTPUT_CSV"
        log ""
        continue
    fi
    
    # JSON解析（Python使用）
    STATS=$(python3 -c "
import json
import sys

try:
    with open('$REPORT') as f:
        data = json.load(f)
    
    s = data['summary']
    total = s['total_bars']
    pass_bars = s['pass_count']
    fail_bars = s['fail_count']
    warn_bars = s['warning_count']
    
    pass_rate = pass_bars / total if total > 0 else 0.0
    fail_rate = fail_bars / total if total > 0 else 0.0
    warn_rate = warn_bars / total if total > 0 else 0.0
    
    # section_override適用数カウント
    override_count = 0
    safe_kit_count = 0
    for bar_key, bar_data in data.get('results', {}).items():
        for msg in bar_data.get('messages', []):
            if 'section_override' in msg:
                override_count += 1
        if bar_data.get('safe_kit_fallback_recommended', False):
            safe_kit_count += 1
    
    print(f'{total},{pass_bars},{fail_bars},{warn_bars},{pass_rate:.4f},{fail_rate:.4f},{warn_rate:.4f},{override_count},{safe_kit_count}')
except Exception as e:
    print(f'0,0,0,0,0.0,0.0,0.0,0,0', file=sys.stderr)
    sys.exit(1)
")
    
    if [ $? -eq 0 ]; then
        TIMESTAMP=$(date -u +%Y-%m-%dT%H:%M:%SZ)
        echo "$PKG_NAME,$STATS,$TIMESTAMP" >> "$OUTPUT_CSV"
        
        # 統計表示
        IFS=',' read -r TOTAL PASS FAIL WARN PASS_RATE FAIL_RATE WARN_RATE OVERRIDE SAFE <<< "$STATS"
        log "📊 Statistics:"
        log "  Total bars: $TOTAL"
        log "  Pass:       $PASS ($PASS_RATE%)"
        log "  Fail:       $FAIL ($FAIL_RATE%)"
        log "  Warning:    $WARN ($WARN_RATE%)"
        log "  section_override: $OVERRIDE"
        log "  Safe-Kit recommended: $SAFE"
        
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
        ok "✅ Validation completed: $PKG_NAME"
    else
        err "❌ Stats extraction failed: $PKG_NAME"
        FAIL_COUNT=$((FAIL_COUNT + 1))
        echo "$PKG_NAME,0,0,0,0,0.0,0.0,0.0,0,0,$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$OUTPUT_CSV"
    fi
    
    log ""
done

# 総括
log "============================================================"
log "🎯 Batch Validation Summary"
log "============================================================"
log "Total processed: ${#SONG_PACKAGES[@]}"
log "Success:         $SUCCESS_COUNT"
log "Failed:          $FAIL_COUNT"
log ""
log "📄 Results saved to: $OUTPUT_CSV"
log ""

# CSV集計表示
if [ $SUCCESS_COUNT -gt 0 ]; then
    log "📊 Aggregate Statistics:"
    python3 -c "
import csv

with open('$OUTPUT_CSV', 'r') as f:
    reader = csv.DictReader(f)
    rows = [r for r in reader if float(r['total_bars']) > 0]

if not rows:
    print('  No valid data')
else:
    total_bars = sum(int(r['total_bars']) for r in rows)
    total_pass = sum(int(r['pass_bars']) for r in rows)
    total_fail = sum(int(r['fail_bars']) for r in rows)
    total_warn = sum(int(r['warn_bars']) for r in rows)
    total_override = sum(int(r['section_override_count']) for r in rows)
    total_safe = sum(int(r['safe_kit_recommended']) for r in rows)
    
    avg_pass_rate = sum(float(r['pass_rate']) for r in rows) / len(rows)
    avg_fail_rate = sum(float(r['fail_rate']) for r in rows) / len(rows)
    avg_warn_rate = sum(float(r['warn_rate']) for r in rows) / len(rows)
    
    print(f'  Total bars:           {total_bars:,}')
    print(f'  Total Pass:           {total_pass:,} ({total_pass/total_bars*100:.1f}%)')
    print(f'  Total Fail:           {total_fail:,} ({total_fail/total_bars*100:.1f}%)')
    print(f'  Total Warning:        {total_warn:,} ({total_warn/total_bars*100:.1f}%)')
    print(f'  Avg Pass Rate:        {avg_pass_rate*100:.1f}%')
    print(f'  Avg Fail Rate:        {avg_fail_rate*100:.1f}%')
    print(f'  Avg Warning Rate:     {avg_warn_rate*100:.1f}%')
    print(f'  Total section_override: {total_override}')
    print(f'  Total Safe-Kit:       {total_safe} ({total_safe/total_bars*100:.1f}%)')
    print()
    print('  📋 SLO Check (ChatGPT提案基準):')
    print(f'    ✓ Post-gen Pass ≥ 90%: {\"PASS\" if avg_pass_rate >= 0.90 else \"FAIL\"} ({avg_pass_rate*100:.1f}%)')
    print(f'    ✓ Warning 15-30%:      {\"PASS\" if 0.15 <= avg_warn_rate <= 0.30 else \"FAIL\"} ({avg_warn_rate*100:.1f}%)')
    print(f'    ✓ Safe-Kit ≤ 15%:      {\"PASS\" if total_safe/total_bars <= 0.15 else \"FAIL\"} ({total_safe/total_bars*100:.1f}%)')
"
fi

log "✅ Batch validation completed!"
