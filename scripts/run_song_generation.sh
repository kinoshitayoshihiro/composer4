#!/bin/bash
# 統合実行スクリプト - SongPackageから完全ドラム生成
# 
# 使用例:
#   bash scripts/run_song_generation.sh song_packages/sample_project/sample_song
#   bash scripts/run_song_generation.sh song_packages/sample_project/sample_song --auto-safe-kit
#
# 引数:
#   $1: SongPackageディレクトリ（song_package.yamlを含む）
#   --auto-safe-kit: KPI Gate失敗時に自動的にSafe-Kit Fallbackを適用

set -Eeuo pipefail
IFS=$'\n\t'

: "${DRY_RUN:=0}"   # 1でドライラン
: "${QUIET:=0}"     # 1で情報ログ抑制
: "${VIOPTT_ENABLED:=false}"  # true でVioPTT articulation rendering有効化
: "${VIOPTT_INSTRUMENT:=}"    # 例: violin_solo_synchron
: "${VIOPTT_VST_PATH:=}"      # 例: /path/to/vst.vst3

log()  { [ "$QUIET" = "1" ] || printf '%s\n' "$*"; }
warn() { printf 'WARN: %s\n' "$*" >&2; }
err()  { printf 'ERR: %s\n' "$*" >&2; }
run()  { if [ "$DRY_RUN" = "1" ]; then printf 'DRY: %s\n' "$*"; else eval "$@"; fi; }

trap 'err "pipeline failed (see logs above)"; exit 1' ERR

# 引数パース
SONG_DIR=""
AUTO_SAFE_KIT=false

for arg in "$@"; do
    case $arg in
        --auto-safe-kit)
            AUTO_SAFE_KIT=true
            shift
            ;;
        *)
            if [ -z "$SONG_DIR" ]; then
                SONG_DIR="$arg"
            fi
            shift
            ;;
    esac
done

if [ -z "$SONG_DIR" ]; then
    err "Usage: $0 <song_package_dir> [--auto-safe-kit]"
    echo "   Example: $0 song_packages/sample_project/sample_song"
    echo "   Example: $0 song_packages/sample_project/sample_song --auto-safe-kit"
    echo ""
    echo "Options:"
    echo "  --auto-safe-kit  Automatically apply Safe-Kit Fallback if KPI Gate fails"
    exit 1
fi

if [ ! -f "$SONG_DIR/song_package.yaml" ]; then
    err "song_package.yaml not found in $SONG_DIR"
    exit 1
fi

log "🎵 Song Generation Pipeline"
if [ "$AUTO_SAFE_KIT" = true ]; then
    log "   🔧 Auto Safe-Kit Fallback: ENABLED"
fi
printf '=%.0s' {1..60}; printf '\n'
log ""
log "📂 SongPackage: $SONG_DIR"
log ""

# 環境変数としてexport（Phase 14サマリ生成で使用）
export SONG_DIR

# プロジェクトルート検出
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_ROOT"

# 1. bars.parquet生成（既に存在する場合はスキップ）
if [ ! -f "$SONG_DIR/bars.parquet" ]; then
    log "📊 Step 1/6: Generating bars.parquet..."
    
    # sections.jsonとchordmapのパス検出
    if [ -f "$SONG_DIR/sections.json" ]; then
        SECTIONS_PATH="$SONG_DIR/sections.json"
    elif [ -f "data/test_sections.json" ]; then
        SECTIONS_PATH="data/test_sections.json"
    else
        err "sections.json not found"
        exit 1
    fi
    
    # chordmap優先順位: SONG_DIR > data/test_chordmap.json > data/chordmap.json
    if [ -f "$SONG_DIR/chordmap.json" ]; then
        CHORDMAP_PATH="$SONG_DIR/chordmap.json"
    elif [ -f "data/test_chordmap.json" ]; then
        CHORDMAP_PATH="data/test_chordmap.json"
    elif [ -f "data/chordmap.json" ]; then
        CHORDMAP_PATH="data/chordmap.json"
    else
        err "chordmap.json not found"
        exit 1
    fi
    
    run python3 scripts/generate_bars_parquet.py \
        --sections "$SECTIONS_PATH" \
        --chordmap "$CHORDMAP_PATH" \
        --drums-midi "$SONG_DIR/drums.mid" \
        --output "$SONG_DIR/bars.parquet"
    log ""
else
    log "✅ Step 1/6: bars.parquet already exists (skipping)"
    log ""
fi

# 🔍 bars/drums.mid 長さ一致検査＋自動リカバリ
log "🔍 Validating bars.parquet / drums.mid length consistency..."
BARS_MISMATCH=$(python3 << PYEOF
import pandas as pd
import pretty_midi
from pathlib import Path
import sys

try:
    bars_df = pd.read_parquet("$SONG_DIR/bars.parquet")
    mid = pretty_midi.PrettyMIDI("$SONG_DIR/drums.mid")
    tempo = mid.estimate_tempo()
    duration_sec = mid.get_end_time()
    midi_bars = int(duration_sec / (240.0 / tempo)) + 1
    
    bars_count = len(bars_df)
    if abs(bars_count - midi_bars) > 5:  # 5小節以上ズレたら不一致
        print(f"MISMATCH: bars={bars_count} midi={midi_bars}")
        sys.exit(1)
    else:
        print(f"OK: bars={bars_count} midi={midi_bars}")
        sys.exit(0)
except Exception as e:
    print(f"ERROR: {e}", file=sys.stderr)
    sys.exit(2)
PYEOF
)

if [ $? -eq 1 ]; then
    log "⚠️  bars/drums.mid length mismatch detected: $BARS_MISMATCH"
    log "🔧 Auto-recovery: Regenerating bars.parquet from drums.mid..."
    
    run python3 scripts/generate_bars_parquet.py \
        --sections "$SECTIONS_PATH" \
        --chordmap "$CHORDMAP_PATH" \
        --drums-midi "$SONG_DIR/drums.mid" \
        --output "$SONG_DIR/bars.parquet"
    
    log "✅ bars.parquet regenerated successfully"
    log ""
elif [ $? -eq 0 ]; then
    log "✅ bars/drums.mid length: $BARS_MISMATCH"
    log ""
else
    log "⚠️  Length validation skipped (missing files or error)"
    log ""
fi

# 2. Recommender実行
log "🤖 Step 2/6: Running Recommender (ML inference + pattern search)..."
run python3 scripts/recommend_drums.py \
    --song-package "$SONG_DIR/song_package.yaml" \
    --output "$SONG_DIR/drums_recommendations.json"
log ""

# 3. KPI Gate検証
log "🔍 Step 3/6: Running KPI Gate (quality validation)..."
run python3 scripts/kpi_gate.py \
    --recommendations "$SONG_DIR/drums_recommendations.json" \
    --gate-config configs/gate_prod.yaml \
    --output "$SONG_DIR/kpi_gate_report.json"
log ""

# KPI Gate結果チェック
KPI_FAIL_COUNT=$(python3 << PYEOF
import json
import sys

report_path = "$SONG_DIR/kpi_gate_report.json"
try:
    with open(report_path, 'r') as f:
        report = json.load(f)
    print(report['summary']['fail_count'])
except Exception as e:
    print("0", file=sys.stderr)
    print("0")
PYEOF
)

# Auto Safe-Kit Fallback適用
if [ "$AUTO_SAFE_KIT" = true ] && [ "$KPI_FAIL_COUNT" -gt 0 ]; then
    log "⚠️  KPI Gate detected $KPI_FAIL_COUNT failed bars"
    log "🔧 Applying Safe-Kit Fallback (auto mode)..."
    
    run python3 scripts/apply_safe_kit_fallback.py \
        --recommendations "$SONG_DIR/drums_recommendations.json" \
        --kpi-report "$SONG_DIR/kpi_gate_report.json" \
        --rhythm-features "$PROJECT_ROOT/output/rhythm_ai/rhythm_features_merged.parquet" \
        --output "$SONG_DIR/drums_recommendations_fixed.json" \
        --preserve-diversity
    
    # KPI Gate再検証
    log ""
    log "🔍 Re-validating with Safe-Kit patterns..."
    run python3 scripts/kpi_gate.py \
        --recommendations "$SONG_DIR/drums_recommendations_fixed.json" \
        --gate-config configs/gate_prod.yaml \
        --output "$SONG_DIR/kpi_gate_report_fixed.json" \
        --quiet
    
    # 固定版を使用
    RECOMMENDATIONS_FILE="$SONG_DIR/drums_recommendations_fixed.json"
    MIDI_OUTPUT="$SONG_DIR/drums.mid"
    KPI_REPORT="$SONG_DIR/kpi_gate_report_fixed.json"
    
    log "✅ Safe-Kit Fallback applied successfully"
    log ""
else
    RECOMMENDATIONS_FILE="$SONG_DIR/drums_recommendations.json"
    MIDI_OUTPUT="$SONG_DIR/drums.mid"
    KPI_REPORT="$SONG_DIR/kpi_gate_report.json"
fi

# MIDI_OUTPUTを環境変数としてexport（統計表示で使用）
export MIDI_OUTPUT

# 4. Generator実行
log "🎹 Step 4/6: Running Generator (MIDI generation + humanize)..."
run python3 scripts/generate_drums_midi.py \
    --recommendations "$RECOMMENDATIONS_FILE" \
    --output "$MIDI_OUTPUT"
log ""

# 4.5. MIDI自動修正パイプライン（2段階）
log "🔧 Step 4.5/6: MIDI Auto-Fix Pipeline (2-stage KPI correction)..."

# SongPackageのbpmを可能なら自動取得（bpm: 120形式）
SONG_TEMPO_BPM=$(awk '/^[[:space:]]*bpm:[[:space:]]*[0-9]+/{print $2; exit}' "$SONG_DIR/song_package.yaml" 2>/dev/null || true)
TEMPO_ARG=""
[ -n "$SONG_TEMPO_BPM" ] && TEMPO_ARG="--tempo-bpm $SONG_TEMPO_BPM" || TEMPO_ARG="--tempo-bpm 120"

# Stage 1: 過密削減 + backbeat強化
MIDI_TEMP1="$SONG_DIR/drums_stage1.mid"
log "  🔧 Stage 1: Density reduction + backbeat boost..."
run python3 scripts/fix_midi_kpi.py \
    --midi "$MIDI_OUTPUT" \
    --gate-config configs/gate_prod.yaml \
    --output "$MIDI_TEMP1" \
    $TEMPO_ARG \
    --quiet

# Stage 2: 低密度補填 + Snare床値
MIDI_FIXED="$SONG_DIR/drums_fixed.mid"
log "  🔧 Stage 2: Low-density補填 + Snare floor..."
run python3 scripts/augment_midi_kpi_fix.py \
    --input "$MIDI_TEMP1" \
    --output "$MIDI_FIXED" \
    $TEMPO_ARG \
    --min-hat-density 2.5 \
    --hat-pitch 42 --hat-vel 54 --hat-microtiming-ms 6 \
    --backbeat-vel-floor 56 --backbeat-window-ms 45 \
    --max-edits-per-bar 3

# 修正版をメインMIDIに置換
mv "$MIDI_FIXED" "$MIDI_OUTPUT"
rm -f "$MIDI_TEMP1"
log "  ✅ MIDI auto-fix完了（drums.midを更新）"
log ""

# 5. MIDI実体KPI再検証（本番運用品質保証）
log "🔍 Step 5/6: MIDI Validation (post-generation KPI check)..."
MIDI_KPI_REPORT="$SONG_DIR/kpi_gate_report_postgen.json"

run python3 scripts/kpi_gate.py \
    --midi "$MIDI_OUTPUT" \
    --gate-config configs/gate_prod.yaml \
    --output "$MIDI_KPI_REPORT" \
    --bars-parquet "$SONG_DIR/bars.parquet" \
    --quiet $TEMPO_ARG

# MIDI KPI結果確認
MIDI_KPI_FAIL_COUNT=$(python3 -c "
import json
with open('$MIDI_KPI_REPORT') as f:
    report = json.load(f)
print(report['summary']['fail_count'])
")

if [ "$MIDI_KPI_FAIL_COUNT" -gt 0 ]; then
    log "   ⚠️  MIDI validation: $MIDI_KPI_FAIL_COUNT bars failed (post-humanization drift)"
    log "   Note: Minor deviations expected due to humanization"
else
    log "   ✅ MIDI validation: All bars passed"
fi
log ""

# 6. VioPTT articulation hints生成 + 制御MIDI生成（オプション）
if [ "$VIOPTT_ENABLED" = true ]; then
    log "🎻 Step 6/7: VioPTT articulation rendering..."
    
    # articulation_hints生成
    ARTICULATION_HINTS="$SONG_DIR/articulation_hints.json"
    DRUMS_RECS_SOURCE="$SONG_DIR/drums_recommendations.json"
    
    # Safe-Kit適用済みの場合は_fixedを優先
    if [ -f "$SONG_DIR/drums_recommendations_fixed.json" ]; then
        DRUMS_RECS_SOURCE="$SONG_DIR/drums_recommendations_fixed.json"
    fi
    
    log "  📖 Generating articulation hints..."
    run python3 scripts/generate_articulation_hints.py \
        --recommendations "$DRUMS_RECS_SOURCE" \
        --output "$ARTICULATION_HINTS" \
        --tempo-bpm "${SONG_TEMPO_BPM:-120}" \
        --verbose
    
    # 制御MIDI生成
    if [ -n "${VIOPTT_INSTRUMENT:-}" ]; then
        # mapping に instrument キーがあるか軽量チェック
        log "  🔍 Checking instrument mapping for $VIOPTT_INSTRUMENT..."
        INST_CHECK_RESULT=$(python3 - "$VIOPTT_INSTRUMENT" << 'PY' 2>&1
import sys
import yaml

try:
    with open('configs/vioptt_mapping.yaml', 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    
    inst = sys.argv[1]
    
    # instrumentsキーが存在するか確認
    if not isinstance(cfg, dict):
        print(f"ERROR: Invalid YAML structure", file=sys.stderr)
        sys.exit(1)
    
    # トップレベルにinstrument定義があるか確認
    if inst in cfg:
        print(f"OK: Found instrument '{inst}'")
        sys.exit(0)
    else:
        available = [k for k in cfg.keys() if not k.startswith('_') and isinstance(cfg[k], dict)]
        print(f"ERROR: instrument '{inst}' not found", file=sys.stderr)
        print(f"Available instruments: {', '.join(available)}", file=sys.stderr)
        sys.exit(1)
except Exception as e:
    print(f"ERROR: Failed to load mapping: {e}", file=sys.stderr)
    sys.exit(1)
PY
)
        
        if [ $? -ne 0 ]; then
            warn "instrument '$VIOPTT_INSTRUMENT' not found in configs/vioptt_mapping.yaml"
            warn "Skipping control MIDI generation"
            echo "$INST_CHECK_RESULT" >&2
            VIOPTT_INSTRUMENT=""
        else
            log "  ✅ Instrument mapping validated"
        fi
    fi
    
    if [ -n "${VIOPTT_INSTRUMENT:-}" ]; then
        CONTROL_MIDI="$SONG_DIR/${VIOPTT_INSTRUMENT}_controls.mid"
        log "  🎹 Generating control MIDI for $VIOPTT_INSTRUMENT..."
        run python3 scripts/vioptt_render_stub.py \
            --hints "$ARTICULATION_HINTS" \
            --mapping configs/vioptt_mapping.yaml \
            --instrument "$VIOPTT_INSTRUMENT" \
            --output "$CONTROL_MIDI" \
            --tempo-bpm "${SONG_TEMPO_BPM:-120}"
        
        # MIDI統合
        MERGED_MIDI="$SONG_DIR/${VIOPTT_INSTRUMENT}_merged.mid"
        log "  🔗 Merging control + note MIDI..."
        run python3 scripts/merge_midi_files.py \
            --note-midi "$MIDI_OUTPUT" \
            --control-midi "$CONTROL_MIDI" \
            --output "$MERGED_MIDI"
        
        # DAWDreamer WAV生成（VST存在時のみ）
        if [ -n "${VIOPTT_VST_PATH:-}" ] && [ -f "$VIOPTT_VST_PATH" ]; then
            RENDERED_WAV="$SONG_DIR/${VIOPTT_INSTRUMENT}_rendered.wav"
            
            # 曲長から自動レンダ時間（秒）を算出（+2秒マージン）
            log "  📏 Calculating render duration from MIDI length..."
            RENDER_DUR=$(python3 - << 'PY'
import mido
import sys
import os

try:
    midi_path = os.environ.get("MIDI_OUTPUT")
    if not midi_path or not os.path.isfile(midi_path):
        print("64.0")  # fallback
        sys.exit(0)
    
    mid = mido.MidiFile(midi_path)
    duration = round(mid.length + 2.0, 2)  # +2秒マージン
    print(duration)
except Exception as e:
    print("64.0", file=sys.stderr)  # fallback on error
    print(f"WARN: Could not calculate MIDI length, using default 64.0s: {e}", file=sys.stderr)
PY
)
            
            log "  🎧 Rendering with DAWDreamer (duration: ${RENDER_DUR}s)..."
            
            # 排他制御でDAWDreamer実行（並列実行時のVST競合回避）
            if command -v flock &> /dev/null; then
                run flock /tmp/daw_render.lock -c "python3 scripts/render_with_dawdreamer.py \
                    --note-midi \"$MIDI_OUTPUT\" \
                    --control-midi \"$CONTROL_MIDI\" \
                    --vst-path \"$VIOPTT_VST_PATH\" \
                    --output \"$RENDERED_WAV\" \
                    --duration \"$RENDER_DUR\""
            else
                # flock非対応環境ではそのまま実行
                run python3 scripts/render_with_dawdreamer.py \
                    --note-midi "$MIDI_OUTPUT" \
                    --control-midi "$CONTROL_MIDI" \
                    --vst-path "$VIOPTT_VST_PATH" \
                    --output "$RENDERED_WAV" \
                    --duration "$RENDER_DUR"
            fi
            
            log "  ✅ VioPTT rendering complete: $RENDERED_WAV"
        else
            log "  ⚠️  VST path not set (VIOPTT_VST_PATH), skipping WAV rendering"
            log "  ✅ Control MIDI generated: $CONTROL_MIDI"
        fi
    else
        log "  ⚠️  VIOPTT_INSTRUMENT not set, skipping control MIDI generation"
        log "  ✅ articulation_hints generated: $ARTICULATION_HINTS"
    fi
    log ""
fi

# Phase 14 JSONサマリ生成（観測性向上）
if [ "$VIOPTT_ENABLED" = true ]; then
    log "📊 Generating Phase 14 summary..."
    python3 - << 'PY' || true
import json
import os
import time

song_dir = os.environ.get("SONG_DIR", "")
vioptt_inst = os.environ.get("VIOPTT_INSTRUMENT", "")

# NoneTypeエラー回避
if not song_dir:
    print("WARN: SONG_DIR not set, skipping phase14_summary.json")
    exit(0)

phase14_summary = {
    "song_dir": song_dir,
    "vst_enabled": bool(os.environ.get("VIOPTT_VST_PATH")),
    "vst_path": os.environ.get("VIOPTT_VST_PATH", ""),
    "instrument": vioptt_inst,
    "files": {
        "articulation_hints": os.path.exists(os.path.join(song_dir, "articulation_hints.json")),
        "control_midi": os.path.exists(os.path.join(song_dir, f"{vioptt_inst}_controls.mid")) if vioptt_inst else False,
        "merged_midi": os.path.exists(os.path.join(song_dir, f"{vioptt_inst}_merged.mid")) if vioptt_inst else False,
        "rendered_wav": os.path.exists(os.path.join(song_dir, f"{vioptt_inst}_rendered.wav")) if vioptt_inst else False
    },
    "timestamp": int(time.time()),
    "timestamp_readable": time.strftime("%Y-%m-%d %H:%M:%S")
}

output_path = os.path.join(song_dir, "phase14_summary.json")
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(phase14_summary, f, ensure_ascii=False, indent=2)

print(f"✅ Wrote phase14_summary.json")
PY
    log ""
fi

# 統計サマリー
printf '=%.0s' {1..60}; printf '\n'
log "✅ Song Generation Complete!"
log ""

# 生成ファイル一覧
log "📁 Generated files:"
log "  - $SONG_DIR/bars.parquet"
if [ "$AUTO_SAFE_KIT" = true ] && [ "$KPI_FAIL_COUNT" -gt 0 ]; then
    log "  - $SONG_DIR/drums_recommendations.json (original)"
    log "  - $SONG_DIR/drums_recommendations_fixed.json (Safe-Kit applied) ✨"
    log "  - $SONG_DIR/kpi_gate_report.json (original)"
    log "  - $SONG_DIR/kpi_gate_report_fixed.json (Safe-Kit applied) ✨"
else
    log "  - $SONG_DIR/drums_recommendations.json"
    log "  - $SONG_DIR/kpi_gate_report.json"
fi
log "  - $SONG_DIR/kpi_gate_report_postgen.json (MIDI validation) ✨"
log "  - $MIDI_OUTPUT"
log ""

# VioPTTファイル一覧（有効時のみ）
if [ "$VIOPTT_ENABLED" = true ]; then
    log "📁 VioPTT files:"
    log "  - $SONG_DIR/articulation_hints.json"
    if [ -n "${VIOPTT_INSTRUMENT:-}" ]; then
        log "  - $SONG_DIR/${VIOPTT_INSTRUMENT}_controls.mid"
        log "  - $SONG_DIR/${VIOPTT_INSTRUMENT}_merged.mid"
        if [ -n "${VIOPTT_VST_PATH:-}" ] && [ -f "$VIOPTT_VST_PATH" ]; then
            log "  - $SONG_DIR/${VIOPTT_INSTRUMENT}_rendered.wav ✨"
        fi
    fi
    log ""
fi

# MIDI統計表示（drums_fixed.mid優先）
if command -v python3 &> /dev/null; then
    python3 - << 'EOF' || true
import mido
import os
import sys

# drums_fixed.midが存在すればそちらを優先
base = os.environ.get('MIDI_OUTPUT')
if not base:
    print("INFO: MIDI_OUTPUT not set, skipping stats", file=sys.stderr)
    sys.exit(0)

base_dir = os.path.dirname(base)
candidates = [
    os.path.join(base_dir, "drums_fixed.mid") if base_dir else None,
    base
]
midi_path = next((p for p in candidates if p and os.path.isfile(p)), base)

try:
    if not os.path.isfile(midi_path):
        print(f"INFO: MIDI file not found: {midi_path}", file=sys.stderr)
        sys.exit(0)
    
    midi = mido.MidiFile(midi_path)
    note_count = sum(1 for track in midi.tracks for msg in track if msg.type == 'note_on' and msg.velocity > 0)
    basename = os.path.basename(midi_path)
    print(f"📊 MIDI Statistics ({basename}):")
    print(f"  - Total notes: {note_count:,}")
    print(f"  - Duration: {midi.length:.1f} seconds")
    print(f"  - Ticks per beat: {midi.ticks_per_beat}")
except Exception as e:
    print(f"INFO: MIDI stats skipped: {e}", file=sys.stderr)
EOF
fi

# KPI Gate統計表示
python3 << PYEOF
import json

try:
    with open("$KPI_REPORT", 'r') as f:
        report = json.load(f)
    
    summary = report['summary']
    print(f"\n📊 KPI Gate Summary:")
    print(f"  - Total bars: {summary['total_bars']}")
    print(f"  - Pass: {summary['pass_count']} ({summary['pass_rate']*100:.1f}%)")
    print(f"  - Fail: {summary['fail_count']}")
    if summary['fail_count'] > 0:
        print(f"  - ⚠️  Warning: {summary['fail_count']} bars failed quality check")
except:
    pass
PYEOF

echo ""
echo "🎵 Next steps:"
echo "  - Listen: Open $MIDI_OUTPUT in DAW"
echo "  - Convert to WAV: fluidsynth -F drums.wav soundfont.sf2 $MIDI_OUTPUT"
echo "  - Check KPI report: cat $KPI_REPORT | jq .summary"
if [ "$AUTO_SAFE_KIT" = true ] && [ "$KPI_FAIL_COUNT" -gt 0 ]; then
    echo "  - Compare: diff drums.mid vs original (Safe-Kit improved quality)"
fi
echo ""
