#!/usr/bin/env bash
# test_piano_acceptance.sh - Piano受け入れテスト自動化スクリプト
#
# VST到着後の単曲E2Eテスト（制御MIDI → WAV → Audio KPI検証）

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

# デフォルト設定
SONG_DIR="song_packages/test_project/test_song"
INSTRUMENT="piano_sfz_salamander"
VST_PATH=""
TEMPO_BPM=120
PROFILE="piano_kpi"
SKIP_RENDER=false

# 引数解析
while [[ $# -gt 0 ]]; do
    case $1 in
        --song-dir)
            SONG_DIR="$2"
            shift 2
            ;;
        --instrument)
            INSTRUMENT="$2"
            shift 2
            ;;
        --vst-path)
            VST_PATH="$2"
            shift 2
            ;;
        --tempo-bpm)
            TEMPO_BPM="$2"
            shift 2
            ;;
        --profile)
            PROFILE="$2"
            shift 2
            ;;
        --skip-render)
            SKIP_RENDER=true
            shift
            ;;
        *)
            err "Unknown option: $1"
            echo "Usage: $0 --vst-path VST [--song-dir DIR] [--instrument INST] [--tempo-bpm BPM] [--profile PROF] [--skip-render]"
            exit 1
            ;;
    esac
done

# VST必須（--skip-renderの場合は不要）
if [ "$SKIP_RENDER" = false ] && [ -z "$VST_PATH" ]; then
    err "Missing required argument: --vst-path"
    echo "Usage: $0 --vst-path VST [--song-dir DIR] [--instrument INST]"
    exit 1
fi

log "🎹 Piano Acceptance Test"
log "============================================================"
log "Song directory: $SONG_DIR"
log "Instrument:     $INSTRUMENT"
log "VST path:       ${VST_PATH:-N/A (skip render)}"
log "Tempo:          $TEMPO_BPM BPM"
log "KPI Profile:    $PROFILE"
log ""

# 環境変数設定
export VIOPTT_KS_ADVANCE_MS=80
export VIOPTT_CC_SLEW_BEATS=0.125

# 出力ファイルパス
CONTROL_MIDI="${SONG_DIR}/${INSTRUMENT}_controls.mid"
MERGED_MIDI="${SONG_DIR}/${INSTRUMENT}_merged.mid"
OUTPUT_WAV="${SONG_DIR}/${INSTRUMENT}_rendered.wav"
KPI_JSON="${SONG_DIR}/audio_kpi_${INSTRUMENT}.json"

# ============================================================
# Test 1: 制御MIDI生成
# ============================================================
log "📊 Test 1/3: Control MIDI Generation"

python3 scripts/vioptt_render_stub.py \
    --hints "${SONG_DIR}/articulation_hints.json" \
    --mapping configs/vioptt_mapping.yaml \
    --instrument "$INSTRUMENT" \
    --output "$CONTROL_MIDI" \
    --tempo-bpm "$TEMPO_BPM"

if [ ! -f "$CONTROL_MIDI" ]; then
    err "Control MIDI not generated!"
    exit 1
fi

ok "✅ Test 1 PASSED: Control MIDI generated"
log ""

# ============================================================
# Test 2: WAVレンダリング（VST指定時のみ）
# ============================================================
if [ "$SKIP_RENDER" = true ]; then
    warn "⏭️  Test 2 SKIPPED: WAV rendering (--skip-render)"
    log ""
else
    log "📊 Test 2/3: WAV Rendering"
    
    bash scripts/run_vioptt_pipeline.sh \
        --song-dir "$SONG_DIR" \
        --instrument "$INSTRUMENT" \
        --vst-path "$VST_PATH" \
        --tempo-bpm "$TEMPO_BPM"
    
    if [ ! -f "$OUTPUT_WAV" ]; then
        err "WAV file not generated!"
        exit 1
    fi
    
    # WAV統計確認
    WAV_SIZE=$(stat -f%z "$OUTPUT_WAV" 2>/dev/null || stat -c%s "$OUTPUT_WAV" 2>/dev/null || echo "0")
    if [ "$WAV_SIZE" -lt 10000 ]; then
        err "WAV file too small (< 10KB)!"
        exit 1
    fi
    
    ok "✅ Test 2 PASSED: WAV rendered (${WAV_SIZE} bytes)"
    log ""
fi

# ============================================================
# Test 3: Audio KPI検証（WAV存在時のみ）
# ============================================================
if [ ! -f "$OUTPUT_WAV" ]; then
    warn "⏭️  Test 3 SKIPPED: Audio KPI (WAV not found)"
    log ""
else
    log "📊 Test 3/3: Audio KPI Validation"
    
    python3 scripts/validate_audio_quality.py \
        --wav "$OUTPUT_WAV" \
        --midi "$MERGED_MIDI" \
        --gate configs/audio_gate_prod.yaml \
        --profile "$PROFILE" \
        --out-json "$KPI_JSON"
    
    if [ ! -f "$KPI_JSON" ]; then
        err "KPI JSON not generated!"
        exit 1
    fi
    
    # KPI結果確認
    OVERALL_STATUS=$(python3 -c "
import json
with open('$KPI_JSON') as f:
    kpi = json.load(f)
print(kpi.get('overall_status', 'UNKNOWN'))
")
    
    if [ "$OVERALL_STATUS" = "FAIL" ]; then
        err "❌ Test 3 FAILED: Audio KPI FAIL"
        
        # FAIL詳細表示
        python3 -c "
import json
with open('$KPI_JSON') as f:
    kpi = json.load(f)
print('\nFailed KPIs:')
for key, value in kpi.items():
    if isinstance(value, dict) and value.get('status') == 'FAIL':
        print(f'  - {key}: {value.get(\"value\")} (SLO: {value.get(\"slo\")})')
"
        exit 1
    elif [ "$OVERALL_STATUS" = "WARNING" ]; then
        warn "⚠️  Test 3 WARNING: Audio KPI WARNING"
        ok "✅ Test 3 PASSED (with warnings)"
    else
        ok "✅ Test 3 PASSED: Audio KPI PASS"
    fi
    
    log ""
fi

# ============================================================
# サマリー
# ============================================================
log "============================================================"
log "🎉 Piano Acceptance Test PASSED!"
log "============================================================"
log ""
log "Generated Files:"
log "  - Control MIDI: $CONTROL_MIDI"
if [ "$SKIP_RENDER" = false ]; then
    log "  - Merged MIDI:  $MERGED_MIDI"
    log "  - WAV Output:   $OUTPUT_WAV"
    log "  - KPI JSON:     $KPI_JSON"
fi
log ""
log "Next Steps:"
log "  1. Listen to WAV: open $OUTPUT_WAV"
log "  2. Review KPI:    cat $KPI_JSON | jq"
log "  3. Run batch:     bash scripts/run_batch_vioptt_generation.sh --instrument $INSTRUMENT --vst-path \"$VST_PATH\""
log ""
