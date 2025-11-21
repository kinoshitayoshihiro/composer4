#!/usr/bin/env bash
# e2e_integrate_midi.sh - Plan統合→MIDI生成 専用（plans-only / no-WAV）
# 用途: Phase A/B で作成済みの各種 *_plan.json を統合し、tempo_map.json を用いて可変テンポMIDIを出力
# 使い方:
#   ./scripts/e2e_integrate_midi.sh data/suno_ai/suno_themesong/song_004 [--ppq 960] [--out-name name.mid] [--dry-run]

set -Eeuo pipefail
IFS=$'\n\t'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="${SCRIPT_DIR%/scripts}"
export PYTHONPATH="${WORKSPACE_ROOT}:${PYTHONPATH:-}"

# venv（任意）
if [[ -f "${WORKSPACE_ROOT}/.venv311/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "${WORKSPACE_ROOT}/.venv311/bin/activate"
fi
PYTHON_BIN="${PYTHON_BIN:-python3}"

usage() {
    cat <<USAGE
Usage: $0 <song-dir> [OPTIONS]
OPTIONS:
    --ppq PPQ        : PPQ resolution (default: 480)
    --out-name NAME  : Output MIDI file name (default: {song_id}_integrated.mid)
    --tempo-bpm BPM  : (fallback) 固定テンポで書き出す場合のみ指定。通常は不要（tempo_map.json優先）
    --split-tracks   : Write multi-track MIDI (one track per instrument)
    --dry-run        : 検証のみ（arrangement_plan.json作成まで）
    -h, --help       : Show this help
USAGE
}

# ---- args ----
[[ $# -lt 1 ]] && usage && exit 1
if [[ "$1" == "-h" || "$1" == "--help" ]]; then usage; exit 0; fi
SONG_DIR="$1"; shift || true
PPQ=480
OUT_NAME=""
TEMPO_BPM_OVERRIDE=""
DRY_RUN=false
SPLIT_TRACKS=false
while [[ $# -gt 0 ]]; do
    case "$1" in
        --ppq) PPQ="${2:-480}"; shift 2;;
        --out-name) OUT_NAME="${2:-}"; shift 2;;
        --tempo-bpm) TEMPO_BPM_OVERRIDE="${2:-}"; shift 2;;
        --split-tracks) SPLIT_TRACKS=true; shift;;
        --dry-run) DRY_RUN=true; shift;;
        -h|--help) usage; exit 0;;
        *) echo "❌ Unknown option: $1"; usage; exit 1;;
    esac
done

echo "🎼 E2E: MIDI Integration (plans-only)"
echo "📂 SONG_DIR  : $SONG_DIR"
echo "🧮 PPQ       : $PPQ"
echo "🧪 DRY_RUN   : $DRY_RUN"
echo

[[ -d "$SONG_DIR" ]] || { echo "❌ Song dir not found: $SONG_DIR"; exit 1; }

PLANS_DIR="$SONG_DIR/plans"
[[ -d "$PLANS_DIR" ]] || { echo "❌ Plans dir not found: $PLANS_DIR"; exit 1; }

# 可変テンポ（推奨）
TEMPO_MAP=""
for c in "$SONG_DIR/analysis/tempo_map.json" "$SONG_DIR/tempo_map.json"; do
    [[ -f "$c" ]] && TEMPO_MAP="$c" && break
done
if [[ -z "$TEMPO_MAP" && -z "$TEMPO_BPM_OVERRIDE" ]]; then
    echo "⚠️  tempo_map.json が無く BPM 指定も無いので 120 BPM を仮採用します。"
    TEMPO_BPM_OVERRIDE="120"
fi
[[ -n "$TEMPO_MAP" ]] && echo "🎵 Using variable tempo: $TEMPO_MAP"
[[ -n "$TEMPO_BPM_OVERRIDE" ]] && echo "🎵 Using fixed tempo  : ${TEMPO_BPM_OVERRIDE} BPM"
echo

# 利用可能な plan を自動検出（存在するものだけ採用）
PLAN_FILES=()
for f in "$PLANS_DIR"/*_plan.json; do
    [[ -e "$f" ]] || continue
    PLAN_FILES+=("$f")
done
if [[ ${#PLAN_FILES[@]} -eq 0 ]]; then
    echo "❌ No *_plan.json under $PLANS_DIR"; exit 1;
fi
echo "📋 Plans detected:"
for f in "${PLAN_FILES[@]}"; do
    EVENTS=$("$PYTHON_BIN" - <<'PY' "$f"
import json,sys
with open(sys.argv[1],'r',encoding='utf-8') as fp:
    d=json.load(fp)

# V2 format: {"metadata": {...}, "events": [...]}
if "metadata" in d and "events" in d:
    print(len(d["events"]))
else:
    # Legacy format: {"tracks": [...]} or {"plan": {"tracks": [...]}}
    tracks = d.get('tracks') or d.get('plan',{}).get('tracks') or []
    print(sum(len(t.get('events',[])) for t in tracks))
PY
)
    printf "   • %s (%s events)\n" "$(basename "$f")" "$EVENTS"
done
echo

# arrangement_orchestrator.py があれば優先利用、無ければフォールバックで単純マージ
ARR_SCRIPT=""
for c in "$WORKSPACE_ROOT/scripts/arrangement_orchestrator.py" "$WORKSPACE_ROOT/arrangement_orchestrator.py"; do
    [[ -f "$c" ]] && ARR_SCRIPT="$c" && break
done

ARR_PLAN="$SONG_DIR/arrangement_plan.json"
echo "🔧 Create arrangement_plan.json"
if [[ -n "$ARR_SCRIPT" ]]; then
    # まずは CLI 形式を試みる（失敗したらフォールバック）
    # Build command as an array to be robust on older bash (macOS)
    cmd=("$PYTHON_BIN" "$ARR_SCRIPT" --out "$ARR_PLAN" --ppq "$PPQ")
    if [[ -n "$TEMPO_MAP" ]]; then cmd+=(--tempo-map "$TEMPO_MAP"); fi
    if [[ -n "$TEMPO_BPM_OVERRIDE" ]]; then cmd+=(--tempo-bpm "$TEMPO_BPM_OVERRIDE"); fi
    for pf in "${PLAN_FILES[@]}"; do cmd+=(--plan "$pf"); done
    set +e
    "${cmd[@]}"
    STATUS=$?
    set -e
    if [[ $STATUS -ne 0 || ! -f "$ARR_PLAN" ]]; then
        echo "⚠️  Orchestrator CLI 利用に失敗。フォールバックで単純マージを実施します。"
        "$PYTHON_BIN" - <<'PY' "$ARR_PLAN" "${PLAN_FILES[@]}"
import json,sys,os
out=sys.argv[1]; plans=sys.argv[2:]
tracks=[]
for p in plans:
    with open(p,'r',encoding='utf-8') as fp:
        d=json.load(fp)
    # Support both track-based format and direct events format
    if 'tracks' in d:
        ts = d['tracks']
    else:
        # Plan files have direct 'events' and 'metadata'
        ts = [d]
    base=os.path.basename(p)
    stem=base.replace('_plan.json','')
    for t in ts:
        # Ensure each track has instrument name
        if 'instrument' not in t:
            t['instrument'] = t.get('name') or t.get('metadata', {}).get('instrument') or stem
        # Copy events to track level if needed
        if 'events' not in t and 'events' in d:
            t['events'] = d['events']
        tracks.append(t)
arr={'tracks': tracks}
with open(out,'w',encoding='utf-8') as fo:
    json.dump(arr, fo, ensure_ascii=False, indent=2)
PY
    fi
else
    echo "ℹ️ arrangement_orchestrator.py が無いのでフォールバック統合を使用します。"
    "$PYTHON_BIN" - <<'PY' "$ARR_PLAN" "${PLAN_FILES[@]}"
import json,sys,os
out=sys.argv[1]; plans=sys.argv[2:]
tracks=[]
for p in plans:
    with open(p,'r',encoding='utf-8') as fp:
        d=json.load(fp)
    # Support both track-based format and direct events format
    if 'tracks' in d:
        ts = d['tracks']
    else:
        # Plan files have direct 'events' and 'metadata'
        ts = [d]
    base=os.path.basename(p)
    stem=base.replace('_plan.json','')
    for t in ts:
        # Ensure each track has instrument name
        if 'instrument' not in t:
            t['instrument'] = t.get('name') or t.get('metadata', {}).get('instrument') or stem
        # Copy events to track level if needed
        if 'events' not in t and 'events' in d:
            t['events'] = d['events']
        tracks.append(t)
arr={'tracks': tracks}
with open(out,'w',encoding='utf-8') as fo:
    json.dump(arr, fo, ensure_ascii=False, indent=2)
PY
fi
[[ -f "$ARR_PLAN" ]] || { echo "❌ arrangement_plan.json not created"; exit 1; }
echo "   ✅ $ARR_PLAN"
echo

if $DRY_RUN; then
    echo "🏁 Dry-run: MIDI 書き出しはスキップしました。"
    exit 0
fi

# json2midi.py のパス解決
JSON2MIDI=""
for c in "$WORKSPACE_ROOT/scripts/json2midi.py" "$WORKSPACE_ROOT/json2midi.py"; do
    [[ -f "$c" ]] && JSON2MIDI="$c" && break
done
[[ -n "$JSON2MIDI" ]] || { echo "❌ json2midi.py not found"; exit 1; }

MIDI_DIR="$SONG_DIR/midi"; mkdir -p "$MIDI_DIR"
SONG_ID="$(basename "$SONG_DIR")"
OUT_NAME="${OUT_NAME:-${SONG_ID}_integrated.mid}"
OUT_MIDI="$MIDI_DIR/$OUT_NAME"

echo "🎹 Write MIDI → $OUT_MIDI"
# json2midi.py に --tempo-map を渡して可変テンポ対応
# --split-tracks で各楽器を独立トラックとして書き出し（オプション指定時）
# --beats-per-bar 4 で 4/4 拍子 (bar/beat → absolute beats 変換に必要)

SPLIT_TRACKS_ARG=()
[[ "$SPLIT_TRACKS" == "true" ]] && SPLIT_TRACKS_ARG+=(--split-tracks)

if [[ -n "$TEMPO_MAP" ]]; then
    "$PYTHON_BIN" "$JSON2MIDI" "$ARR_PLAN" -o "$OUT_MIDI" --tempo-map "$TEMPO_MAP" ${SPLIT_TRACKS_ARG[@]+"${SPLIT_TRACKS_ARG[@]}"} --beats-per-bar 4
elif [[ -n "$TEMPO_BPM_OVERRIDE" ]]; then
    "$PYTHON_BIN" "$JSON2MIDI" "$ARR_PLAN" -o "$OUT_MIDI" -b "$TEMPO_BPM_OVERRIDE" ${SPLIT_TRACKS_ARG[@]+"${SPLIT_TRACKS_ARG[@]}"} --beats-per-bar 4
else
    # arrangement_plan.json の meta.tempo_map_path を自動検出して使用
    "$PYTHON_BIN" "$JSON2MIDI" "$ARR_PLAN" -o "$OUT_MIDI" ${SPLIT_TRACKS_ARG[@]+"${SPLIT_TRACKS_ARG[@]}"} --beats-per-bar 4
fi

[[ -f "$OUT_MIDI" ]] || { echo "❌ MIDI not created: $OUT_MIDI"; exit 1; }
echo "✅ Done: $OUT_MIDI"
