#!/usr/bin/env bash
# run_vioptt_pipeline.sh - VioPTT WAV出力パイプライン統合スクリプト
#
# Phase 15.1: 制御MIDI生成 → MIDI統合 → DAWDreamer WAV出力を1コマンドで実行
#
# Usage:
#   bash scripts/run_vioptt_pipeline.sh \
#     --song-dir song_packages/test_project/test_song \
#     --instrument violin \
#     --vst-path /path/to/violin.vst3 \
#     [--no-merge]  # MIDI統合をスキップ（制御MIDIのみでレンダリング）

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
SONG_DIR=""
INSTRUMENT=""
VST_PATH=""
NO_MERGE=false
TEMPO_BPM=120

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
        --no-merge)
            NO_MERGE=true
            shift
            ;;
        *)
            err "Unknown option: $1"
            echo "Usage: $0 --song-dir DIR --instrument INST --vst-path VST [--tempo-bpm BPM] [--no-merge]"
            exit 1
            ;;
    esac
done

# 引数チェック
if [ -z "$SONG_DIR" ] || [ -z "$INSTRUMENT" ]; then
    err "Missing required arguments: --song-dir and --instrument"
    echo "Usage: $0 --song-dir DIR --instrument INST --vst-path VST [--tempo-bpm BPM] [--no-merge]"
    exit 1
fi

# VST存在チェック（VST_PATH指定時のみ）
# macOS VST3はディレクトリ（.vst3バンドル）なので -d でチェック
if [ -n "$VST_PATH" ] && [ ! -d "$VST_PATH" ] && [ ! -f "$VST_PATH" ]; then
    warn "VST not found: $VST_PATH"
    warn "Skipping DAWDreamer rendering (control MIDI only)"
    VST_PATH=""
fi

# サンプルレート自動検出（SFZパスに48khzを含むなら48000、デフォルト44100）
ENGINE_SR="${ENGINE_SR:-}"
if [[ -z "$ENGINE_SR" && "${VST_PATH:-}" =~ 48[kK][hH][zZ] ]]; then
    ENGINE_SR=48000
    log "🔍 Auto-detected 48kHz from VST path"
fi
ENGINE_SR="${ENGINE_SR:-44100}"

log "🎻 VioPTT WAV Pipeline"
log "============================================================"
log "Song directory: $SONG_DIR"
log "Instrument:     $INSTRUMENT"
log "VST path:       ${VST_PATH:-N/A (control MIDI only)}"
log "Tempo:          $TEMPO_BPM BPM"
log "Merge mode:     $([ "$NO_MERGE" = true ] && echo "disabled" || echo "enabled")"
log ""

# パス設定
DRUMS_RECS="$SONG_DIR/drums_recommendations.json"
ARTICULATION_HINTS="$SONG_DIR/articulation_hints.json"
CONTROL_MIDI="$SONG_DIR/${INSTRUMENT}_controls.mid"
NOTE_MIDI="$SONG_DIR/${INSTRUMENT}.mid"
MERGED_MIDI="$SONG_DIR/${INSTRUMENT}_merged.mid"
OUTPUT_WAV="$SONG_DIR/${INSTRUMENT}_rendered.wav"

# Step 1: articulation_hints.json生成
if [ ! -f "$DRUMS_RECS" ]; then
    err "drums_recommendations.json not found: $DRUMS_RECS"
    err "Run 'bash scripts/run_song_generation.sh $SONG_DIR' first"
    exit 1
fi

log "📊 Step 1/4: Generating articulation_hints.json..."
if [ -f "$ARTICULATION_HINTS" ]; then
    ok "✅ articulation_hints.json already exists (skipping)"
else
    python3 scripts/generate_articulation_hints.py \
        --recommendations "$DRUMS_RECS" \
        --output "$ARTICULATION_HINTS" \
        --tempo-bpm "$TEMPO_BPM"
    ok "✅ articulation_hints.json generated"
fi
log ""

# Step 2: 制御MIDI生成（vioptt_render_stub.py使用）
log "🎹 Step 2/4: Generating control MIDI..."
if [ ! -f "scripts/vioptt_render_stub.py" ]; then
    err "vioptt_render_stub.py not found"
    err "This script should have been created in Phase 14 着手キット"
    exit 1
fi

VIOPTT_MAPPING="configs/vioptt_mapping.yaml"
if [ ! -f "$VIOPTT_MAPPING" ]; then
    err "vioptt_mapping.yaml not found: $VIOPTT_MAPPING"
    err "This file should have been created in Phase 14 着手キット"
    exit 1
fi

# 楽器エイリアスマップ（短縮名 → マッピング実体ID）
# violin/guitar/bass/piano の短縮名をサポート、既に実体IDならそのまま使用
case "$INSTRUMENT" in
    violin)
        INSTRUMENT_ID="violin_solo_synchron"
        ;;
    guitar)
        INSTRUMENT_ID="guitar_steel_ample"
        ;;
    bass)
        INSTRUMENT_ID="bass_electric_trilian"
        ;;
    piano)
        INSTRUMENT_ID="piano_giant_kontakt"
        ;;
    *)
        # すでに実体キー（violin_solo_synchron等）ならそのまま
        INSTRUMENT_ID="$INSTRUMENT"
        ;;
esac

python3 scripts/vioptt_render_stub.py \
    --hints "$ARTICULATION_HINTS" \
    --mapping "$VIOPTT_MAPPING" \
    --instrument "$INSTRUMENT_ID" \
    --output "$CONTROL_MIDI" \
    --tempo-bpm "$TEMPO_BPM"
ok "✅ Control MIDI generated: $CONTROL_MIDI"
log ""

# Step 3: MIDI統合（オプション）
if [ "$NO_MERGE" = false ]; then
    if [ ! -f "$NOTE_MIDI" ]; then
        warn "Note MIDI not found: $NOTE_MIDI"
        warn "Skipping merge (control MIDI only)"
    else
        log "🔗 Step 3/4: Merging control MIDI + note MIDI..."
        python3 scripts/merge_midi_files.py \
            --note-midi "$NOTE_MIDI" \
            --control-midi "$CONTROL_MIDI" \
            --output "$MERGED_MIDI"
        ok "✅ Merged MIDI saved: $MERGED_MIDI"
        log ""
        
        # マージ成功時はmerged MIDIを使用
        CONTROL_MIDI="$MERGED_MIDI"
    fi
else
    log "⏭️  Step 3/4: MIDI merge skipped (--no-merge)"
    log ""
fi

# Step 4: DAWDreamer WAV出力（VST指定時のみ）
if [ -z "$VST_PATH" ]; then
    ok "✅ Pipeline completed (control MIDI only, no VST rendering)"
    log ""
    log "📝 Next steps:"
    log "  1. Install VST plugin"
    log "  2. Run with --vst-path /path/to/vst.vst3"
    exit 0
fi

log "🎧 Step 4/4: Rendering WAV with DAWDreamer..."

# Duration自動算出（drums.midベース、+2.0sマージンで尻切れ防止）
DRUMS_MIDI="$SONG_DIR/drums.mid"
if [ -f "$DRUMS_MIDI" ]; then
    DURATION=$(python3 << PYEOF
import pretty_midi
mid = pretty_midi.PrettyMIDI("$DRUMS_MIDI")
print(f"{mid.get_end_time() + 2.0:.1f}")
PYEOF
    )
    log "   Auto-detected duration: ${DURATION}s (from drums.mid + 2.0s margin)"
else
    DURATION=64.0
    warn "   Using default duration: ${DURATION}s (drums.mid not found)"
fi

# VST排他実行（並列バッチでのVST競合回避）
if command -v flock >/dev/null 2>&1; then
    log "   Using flock for exclusive VST rendering..."
    flock /tmp/daw_render.lock -c "python3 scripts/render_with_dawdreamer.py \
        --note-midi \"$NOTE_MIDI\" \
        --control-midi \"$CONTROL_MIDI\" \
        --vst-path \"$VST_PATH\" \
        --output \"$OUTPUT_WAV\" \
        --duration \"$DURATION\""
else
    warn "   flock not available, rendering without lock (VST may conflict in parallel mode)"
    python3 scripts/render_with_dawdreamer.py \
        --note-midi "$NOTE_MIDI" \
        --control-midi "$CONTROL_MIDI" \
        --vst-path "$VST_PATH" \
        --output "$OUTPUT_WAV" \
        --duration "$DURATION"
fi

ok "✅ WAV rendering completed!"
log ""
log "============================================================"
log "📁 Output files:"
log "  - articulation_hints.json: $ARTICULATION_HINTS"
log "  - Control MIDI:            $CONTROL_MIDI"
log "  - Rendered WAV:            $OUTPUT_WAV"
log "============================================================"
log ""
ok "🎉 VioPTT pipeline completed successfully!"
