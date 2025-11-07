#!/usr/bin/env bash
# e2e_suno_arrangement.sh - Enhanced Version (Phase E)
# SunoAI完全アレンジメントE2Eワークフロー（エラーハンドリング強化版）
#
# Usage: ./scripts/e2e_suno_arrangement.sh <song-dir> [OPTIONS]

set -Eeuo pipefail
IFS=$'\n\t'

# パッチ4: 再現性Seed環境変数集約
export PYTHONHASHSEED=0
export COMPOSER2_GLOBAL_SEED="${COMPOSER2_GLOBAL_SEED:-42}"
export TF_CPP_MIN_LOG_LEVEL=2
# NumPy古エイリアス対策（pretty_midi等の依存対策）
export COMPOSER2_ENABLE_NUMPY_SHIM=1

# 早期失敗・後始末（トラップ）
trap 'echo "❌ E2E failed at line $LINENO"; exit 1' ERR

# PYTHONPATHを明示的に設定（opsモジュール認識用）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="${SCRIPT_DIR%/scripts}"
export PYTHONPATH="${WORKSPACE_ROOT}:${PYTHONPATH:-}"

# venv環境を自動アクティベート
if [[ -f "${WORKSPACE_ROOT}/.venv311/bin/activate" ]]; then
    source "${WORKSPACE_ROOT}/.venv311/bin/activate"
    echo "✅ Activated venv: ${WORKSPACE_ROOT}/.venv311"
fi

PYTHON_BIN="${PYTHON_BIN:-python3}"
# Magenta専用Python（別venv、依存衝突回避）
MAGENTA_PY="${MAGENTA_PY:-${WORKSPACE_ROOT}/.venv311/bin/python}"

# パッチ7: Magenta venv使用ミス検知
if [[ ! -x "$MAGENTA_PY" ]]; then
    echo "⚠️  MAGENTA_PY not found or not executable: $MAGENTA_PY"
    echo "   Using fallback python3, but Bus error risk remains"
fi
echo "[ENV] which python: $(which python)"
echo "[ENV] MAGENTA_PY: $MAGENTA_PY"
if command -v pip &>/dev/null; then
    echo "[ENV] pip show note-seq: $(pip show note-seq 2>&1 | grep '^Version:' || echo 'NOT INSTALLED')"
fi

# fail if Stage1 bars missing（ダミーbars禁止）
STRICT_STAGE1=${STRICT_STAGE1:-true}
# Skip heavy stem features generation (use existing files)
SKIP_STEM_FEATURES=${SKIP_STEM_FEATURES:-false}

# --- Cleanup trap ---
cleanup() {
    local exit_code=$?
    if [[ $exit_code -ne 0 ]]; then
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "❌ E2E Workflow Failed (exit code: $exit_code)"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        
        # 中間ファイル保存（デバッグ用）
        if [[ -n "${SONG_DIR:-}" ]] && [[ -d "$SONG_DIR" ]]; then
            local debug_dir="$SONG_DIR/.debug_$(date +%Y%m%d_%H%M%S)"
            mkdir -p "$debug_dir"
            
            # エラーログ保存
            if [[ -n "${CURRENT_STEP:-}" ]]; then
                echo "Failed at: $CURRENT_STEP" > "$debug_dir/error.txt"
            fi
            
            # 中間ファイルコピー
            for file in matches_rhythm.json drums_recommendations.json *_plan.json; do
                if [[ -f "$SONG_DIR/$file" ]]; then
                    cp "$SONG_DIR/$file" "$debug_dir/" 2>/dev/null || true
                fi
            done
            
            echo "🔍 Debug files saved to: $debug_dir"
        fi
    fi
}

trap cleanup EXIT

usage() {
  cat <<USAGE
Usage: $0 <song-dir> [OPTIONS]
  <song-dir>      : song_packages/.../song_XXX

OPTIONS:
  --topk N        : Top-K for rhythm search (default: 5)
  --force-match   : Regenerate matches_rhythm.json
  --use-ml        : (deprecated) Use ML for drums
  --drums-mode M  : drums source: rule|ml|real|magenta (default: rule)
  --kpi           : Run KPI Gate validation after generation
  --dry-run       : Generate plans only, skip MIDI writing
  --enable-crepe  : Enable CREPE vocal F0 extraction (Phase C)
  --enable-oaf    : Enable Onsets-and-Frames piano transcription (Phase C)
  --enable-f0-extract : Enable F0 extraction for bass/lead (Phase D)
  --enable-timbre-curves : Enable timbre curves for synth/pad (Phase D)
  --force-regenerate-drums : Force regenerate Magenta drums (delete cached plan)
  --enable-emotion-ai : Enable EmotionAI (section-based emotion profiles)
  --enable-harmony-ai : Enable Harmony AI (adaptive chord progression learning)
  --emotion-profile P : Emotion profile ("auto", "energetic", "calm", "happy", "sad")
  --stems-dir P   : 外部 stem ディレクトリを明示指定（例: data/.../stemswav_001）
  --stem-drums-pattern  GLOB : Drums 検出パターン（例: 'stem_wav_*_(Drums).wav'）
  --stem-vocals-pattern GLOB : Vocals検出パターン（例: 'stem_wav_*_(Vocals).wav'）
  -h, --help      : Show this help

EXAMPLES:
  $0 song_packages/suno_project/song_001
  $0 song_packages/suno_project/song_001 --topk 10 --drums-mode ml --kpi
  $0 song_packages/suno_project/song_001 --drums-mode real --kpi
  $0 song_packages/suno_project/song_001 --drums-mode magenta --kpi
USAGE
}

# --- Default arguments ---
TOPK=5
FORCE_MATCH=true  # 古いmatches再利用を防止（デフォルトで再探索）
USE_ML=false
DRUMS_MODE="magenta"  # ← Magenta Groove デフォルトON
RUN_KPI=false
DRY_RUN=false
# NO-OP安全設計：ファイル不在でもスキップされるため既定ONで運用
ENABLE_CREPE=true
ENABLE_OAF=true
ENABLE_F0_EXTRACT=false
ENABLE_TIMBRE_CURVES=false
FORCE_REGENERATE_DRUMS=false
# EmotionAI/和声AI（デフォルトON）
ENABLE_EMOTION_AI=true
ENABLE_HARMONY_AI=true
EMOTION_PROFILE="auto"  # "auto", "energetic", "calm", "happy", "sad"等

# 外部Stemオプション（配列初期化必須）
declare -a STEMS_ARGS=()
declare -a STEMS_DIR_CANDIDATES
STEMS_DIR_CLI=""
STEM_DRUMS_PATTERN="stem_wav_*_(Drums).wav"
STEM_VOCALS_PATTERN="stem_wav_*_(Vocals).wav"
SKIP_STEM_FEATURES="${SKIP_STEM_FEATURES:-false}"

# --- Groove Polish knobs (can be overridden by env) ---
POLISH_GROOVE="${POLISH_GROOVE:-true}"
POLISH_HH_BOOST_MAX="${POLISH_HH_BOOST_MAX:-10}"
POLISH_HH_OPEN_RATE="${POLISH_HH_OPEN_RATE:-0.15}"
POLISH_TOM_FILL="${POLISH_TOM_FILL:-true}"
POLISH_SNARE_FLAM="${POLISH_SNARE_FLAM:-true}"

# --help処理を最優先（引数なし時より前）
if [[ $# -ge 1 ]] && [[ "$1" == "-h" || "$1" == "--help" ]]; then
    usage
    exit 0
fi

[[ $# -lt 1 ]] && usage && exit 1
SONG_DIR="$1"; shift || true

while [[ $# -gt 0 ]]; do
  case "$1" in
    --topk) TOPK="${2:-5}"; shift 2;;
    --force-match) FORCE_MATCH=true; shift;;
    --use-ml) USE_ML=true; shift;;
    --drums-mode) DRUMS_MODE="${2:-rule}"; shift 2;;
    --kpi) RUN_KPI=true; shift;;
    --dry-run) DRY_RUN=true; shift;;
    --enable-crepe) ENABLE_CREPE=true; shift;;
    --enable-oaf) ENABLE_OAF=true; shift;;
    --enable-f0-extract) ENABLE_F0_EXTRACT=true; shift;;
    --enable-timbre-curves) ENABLE_TIMBRE_CURVES=true; shift;;
    --force-regenerate-drums) FORCE_REGENERATE_DRUMS=true; shift;;
    --enable-emotion-ai) ENABLE_EMOTION_AI=true; shift;;
    --enable-harmony-ai) ENABLE_HARMONY_AI=true; shift;;
    --emotion-profile) EMOTION_PROFILE="${2:-auto}"; shift 2;;
    --stems-dir) STEMS_DIR_CLI="${2:-}"; shift 2;;
    --stem-drums-pattern) STEM_DRUMS_PATTERN="${2:-}"; shift 2;;
    --stem-vocals-pattern) STEM_VOCALS_PATTERN="${2:-}"; shift 2;;
    --skip-stem-features) SKIP_STEM_FEATURES=true; shift;;
    -h|--help) usage; exit 0;;
    *) echo "❌ Unknown option: $1"; usage; exit 1;;
  esac
done

echo "🎵 SunoAI E2E Arrangement Workflow (Hardened)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📂 Song Directory: $SONG_DIR"
echo "   Top-K        : $TOPK"
echo "   Use ML       : $USE_ML"
echo "   Force Match  : $FORCE_MATCH"
echo "   KPI Gate     : $RUN_KPI"
echo "   Dry Run      : $DRY_RUN"
echo "   CREPE        : $ENABLE_CREPE"
echo "   Onsets-Frames: $ENABLE_OAF"
echo "   F0 Extract   : $ENABLE_F0_EXTRACT"
echo "   Timbre Curves: $ENABLE_TIMBRE_CURVES"
echo "   Force Regen Drums: $FORCE_REGENERATE_DRUMS"
echo "   EmotionAI    : $ENABLE_EMOTION_AI"
echo "   Harmony AI   : $ENABLE_HARMONY_AI"
echo "   Emotion Profile: $EMOTION_PROFILE"
echo
echo

# 1. SongPackage読み込み
SONG_PKG="$SONG_DIR/song_package.yaml"
if [[ ! -f "$SONG_PKG" ]]; then
    echo "❌ song_package.yaml not found: $SONG_PKG"
    exit 1
fi

# テンポ抽出（Pythonで堅牢に）
TEMPO_BPM=$(python3 -c "
import yaml, sys
with open('$SONG_PKG', 'r', encoding='utf-8') as f:
    pkg = yaml.safe_load(f)
    meta = pkg.get('meta', {})
    print(meta.get('bpm', meta.get('tempo_bpm', 120.0)))
")

echo "   Tempo: $TEMPO_BPM BPM"
echo

# 共通引数（存在チェックで後続に渡す）
INSTR_ARGS=( --song-package "$SONG_PKG" --bars "$SONG_DIR/bars.parquet" )
if [[ -f "$SONG_DIR/chordmap.json" ]]; then INSTR_ARGS+=( --chordmap "$SONG_DIR/chordmap.json" ); fi
if [[ -f "$SONG_DIR/sections.json" ]]; then INSTR_ARGS+=( --sections "$SONG_DIR/sections.json" ); fi
if [[ -f "$SONG_DIR/lyric_anchors.json" ]]; then INSTR_ARGS+=( --lyric-anchors "$SONG_DIR/lyric_anchors.json" ); fi
if [[ -f "$SONG_DIR/stem_features.parquet" ]]; then INSTR_ARGS+=( --stems-features "$SONG_DIR/stem_features.parquet" ); fi
if [[ -f configs/ghost_hh_rules.yaml ]]; then GHOST_HH_ARG=( --ghost-hh-rules configs/ghost_hh_rules.yaml ); else GHOST_HH_ARG=(); fi

# 出力先定義
DRUMS_PLAN="$SONG_DIR/drums_plan.json"
DRUMS_REC="$SONG_DIR/drums_recommendations.json"

# 1.2. Stage1分析データ自動コピー（analysisディレクトリから）
CURRENT_STEP="Stage1 Data Import"
ANALYSIS_DIR="data/suno_ai/suno_themesong/song_001/analysis"
if [[ -d "$ANALYSIS_DIR" ]]; then
    echo "📋 Step 1.2: Importing Stage1 Analysis Data"
    echo "   Source: $ANALYSIS_DIR"
    
    # chordmap.json（必須）
    if [[ -f "$ANALYSIS_DIR/chordmap.json" ]] && [[ ! -f "$SONG_DIR/chordmap.json" ]]; then
        cp "$ANALYSIS_DIR/chordmap.json" "$SONG_DIR/"
        echo "   ✅ chordmap.json imported"
    fi
    
    # sections.json（必須）
    if [[ -f "$ANALYSIS_DIR/sections.json" ]] && [[ ! -f "$SONG_DIR/sections.json" ]]; then
        cp "$ANALYSIS_DIR/sections.json" "$SONG_DIR/"
        echo "   ✅ sections.json imported"
    fi
    
    # lyric_anchors.json（オプション）
    if [[ -f "$ANALYSIS_DIR/lyric_anchors.json" ]] && [[ ! -f "$SONG_DIR/lyric_anchors.json" ]]; then
        cp "$ANALYSIS_DIR/lyric_anchors.json" "$SONG_DIR/"
        echo "   ✅ lyric_anchors.json imported"
    fi
    
    echo
else
    echo "⚠️  Step 1.2: Analysis directory not found: $ANALYSIS_DIR"
    echo "   Expecting chordmap.json/sections.json in $SONG_DIR"
    echo
fi

# 1.3. bars.parquet は必須（ダミー生成を原則禁止）
CURRENT_STEP="Bars Presence Check"
if [[ ! -f "$SONG_DIR/bars.parquet" ]]; then
    if [[ "$STRICT_STAGE1" == "true" ]]; then
        echo "❌ bars.parquet not found. Abort (STRICT_STAGE1)."
        echo "   → 生成手順: python ops/stems_features.py --extend-bars"
        echo "      で bars_extended.parquet を作成し、bars.parquet に差し替え"
        exit 1
    else
        echo "⚠️  STRICT_STAGE1=false: 最小barsを生成（非推奨）"
        # 最小bars生成（フォールバック）
        if [[ -f "$SONG_DIR/chordmap.json" ]]; then
            "$PYTHON_BIN" -c "
import json
import pandas as pd
from pathlib import Path

song_dir = Path('$SONG_DIR')
chordmap_path = song_dir / 'chordmap.json'

# chordmap読み込み
chordmap_data = json.loads(chordmap_path.read_text(encoding='utf-8'))
events = chordmap_data.get('events', [])

# 最終時刻から小節数推定（仮定: 4/4拍子、4QL=1小節）
max_time = max((ev['time'] for ev in events), default=0)
num_bars = int(max_time / 4.0) + 1

# bars.parquet生成（最小構造）
bars_data = []
for bar_idx in range(num_bars):
    bars_data.append({
        'bar': bar_idx,
        'start_beat': bar_idx * 4.0,
        'end_beat': (bar_idx + 1) * 4.0,
        'time_signature': '4/4',
        'tempo_bpm': $TEMPO_BPM,
        'energy': 0.5,
        'section_label': 'verse'
    })

bars_df = pd.DataFrame(bars_data)
bars_df.to_parquet(song_dir / 'bars.parquet')

print(f'   ✅ bars.parquet generated: {len(bars_df)} bars (FALLBACK)')
"
        fi
    fi
    echo
fi

# 1.5. Stem Features & Bars Extension (Phase A自動化)
CURRENT_STEP="Stem Features Generation"

# SKIP_STEM_FEATURES=true の場合は丸ごとスキップ
if [[ "$SKIP_STEM_FEATURES" == "true" ]]; then
    echo "⏭️  Step 1.5: Skipped (SKIP_STEM_FEATURES=true)"
    echo
else
    # 外部Stem指定を優先、なければ自動探索
    STEMS_DIR_CANDIDATES=()
    if [[ -n "$STEMS_DIR_CLI" ]]; then
        STEMS_DIR_CANDIDATES+=("$STEMS_DIR_CLI")
    fi
    STEMS_DIR_CANDIDATES+=("$SONG_DIR/stemswav" "$SONG_DIR/stemswav_001" "$SONG_DIR/stems")
    
    # data側の自動探索（song_XXX名でマッチング）
    if [[ -z "$STEMS_DIR_CLI" ]]; then
        guess_data_dir="$(dirname "$SONG_DIR")"
        guess_song="$(basename "$SONG_DIR")"
        # data/suno_ai/*/<song_id>/stemswav_001 を探索
        for datapath in data/suno_ai/*/"$guess_song"/stemswav_001; do
            [[ -d "$datapath" ]] && STEMS_DIR_CANDIDATES+=("$datapath")
        done
    fi
    
    STEMS_DIR=""
    for candidate in "${STEMS_DIR_CANDIDATES[@]}"; do
        if [[ -d "$candidate" ]]; then
            STEMS_DIR="$candidate"
            break
        fi
    done
    
    if [[ -n "$STEMS_DIR" ]]; then
        echo "🎼 Step 1.5: Stem Features Generation (Backend-enabled)"
        echo "   Stems dir: $STEMS_DIR"
        
        # STEMS_ARGS配列を安全に構築
        STEMS_ARGS=( --stems "$STEMS_DIR" )
        [[ -n "$STEM_DRUMS_PATTERN" ]]  && STEMS_ARGS+=(--drums-pattern "$STEM_DRUMS_PATTERN")
        [[ -n "$STEM_VOCALS_PATTERN" ]] && STEMS_ARGS+=(--vocals-pattern "$STEM_VOCALS_PATTERN")
        
        if ! "$PYTHON_BIN" ops/stems_features.py \
            "${STEMS_ARGS[@]}" \
            --bars "$SONG_DIR/bars.parquet" \
            --output "$SONG_DIR/stem_features.parquet" \
            --backend-config configs/arranger_weights.yaml \
            --tempo-bpm "$TEMPO_BPM" \
            --inst-activity \
            --extend-bars; then
            echo "⚠️  Stem features generation failed, continuing without stems"
        else
            echo "   ✅ stem_features.parquet generated (with instrument activity)"
            
            # bars_extended.parquet → bars.parquet置換（全工程でdrums_active利用）
            if [[ -f "$SONG_DIR/bars_extended.parquet" ]]; then
                cp "$SONG_DIR/bars.parquet" "$SONG_DIR/bars_original.parquet" 2>/dev/null || true
                cp "$SONG_DIR/bars_extended.parquet" "$SONG_DIR/bars.parquet"
                echo "   ✅ bars.parquet extended (drums_active added)"
            fi
        fi
        echo
    else
        echo "⚠️  Step 1.5: No stems directory found, skipping stem features"
        echo
    fi
fi

# 2. Pattern Matching（既存実行済みの場合はスキップ）
CURRENT_STEP="Pattern Matching"
MATCHES="$SONG_DIR/matches_rhythm.json"
if $FORCE_MATCH || [[ ! -f "$MATCHES" ]]; then
    echo "🔍 Step 1: Pattern Matching (Top-K=$TOPK)"
    if ! "$PYTHON_BIN" scripts/pattern_matcher.py \
        --song-dir "$SONG_DIR" \
        --rhythm-pickle output/rhythm_ai/rhythm_patterns.pickle \
        --topk "$TOPK"; then
        echo "❌ Pattern matching failed"
        exit 1
    fi
    
    # Plan存在チェック
    if [[ ! -f "$MATCHES" ]]; then
        echo "❌ matches_rhythm.json not created"
        exit 1
    fi
    echo
else
    echo "🔁 Step 1: Reusing existing $MATCHES"
    echo
fi

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Phase C: CREPE/Onsets-and-Frames (Optional, NO-OP安全)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
if [[ "$ENABLE_CREPE" == "true" ]]; then
    echo "🎤 Step 1.5: CREPE Vocal F0 Extraction (Phase C)"
    VOCAL_WAV="$SONG_DIR/vocal.wav"
    if [[ -f "$VOCAL_WAV" ]]; then
        "$PYTHON_BIN" ops/crepe_pitch_extract.py \
            --vocal-wav "$VOCAL_WAV" \
            --song-package "$SONG_PKG" \
            --bars "$SONG_DIR/bars.parquet" \
            --out "$SONG_DIR/vocal_f0_crepe.parquet" \
            --anchors "$SONG_DIR/lyric_anchors_crepe.json" || echo "⚠️  CREPE failed (NO-OP)"
        echo "✅ CREPE completed (or NO-OP)"
    else
        echo "⚠️  vocal.wav not found, CREPE skipped (NO-OP)"
    fi
    echo
fi

if [[ "$ENABLE_OAF" == "true" ]]; then
    echo "🎹 Step 1.6: Onsets-and-Frames Piano Transcription (Phase C)"
    PIANO_WAV="$SONG_DIR/piano.wav"
    if [[ -f "$PIANO_WAV" ]]; then
        "$PYTHON_BIN" ops/transcribe_piano_oaf.py \
            --piano-wav "$PIANO_WAV" \
            --song-package "$SONG_PKG" \
            --bars "$SONG_DIR/bars.parquet" \
            --out-midi "$SONG_DIR/piano_onsets_frames.mid" \
            --out-stats "$SONG_DIR/piano_onsets_frames.parquet" || echo "⚠️  OaF failed (NO-OP)"
        echo "✅ Onsets-and-Frames completed (or NO-OP)"
    else
        echo "⚠️  piano.wav not found, OaF skipped (NO-OP)"
    fi
    echo
fi

# Phase D: F0 extraction & Timbre curves (完全NO-OP設計)
if [[ "$ENABLE_F0_EXTRACT" == "true" ]]; then
    echo "🎸 Step 1.7: F0 Extraction for Bass/Lead (Phase D)"
    for stem_role in "Bass" "Lead" "Guitar"; do
        STEM_WAV=$(find "${STEMS_DIR:-$SONG_DIR}" -name "*${stem_role}*.wav" 2>/dev/null | head -1)
        if [[ -f "$STEM_WAV" ]]; then
            OUT_F0="$SONG_DIR/${stem_role,,}_f0.parquet"
            "$PYTHON_BIN" ops/crepe_extract.py \
                --audio "$STEM_WAV" \
                --bars "$SONG_DIR/bars.parquet" \
                --out "$OUT_F0" \
                --hop-ms 10 --smooth-ms 120 || echo "⚠️  F0 extraction failed for $stem_role (NO-OP)"
            [[ -f "$OUT_F0" ]] && echo "✅ F0 extracted: $OUT_F0"
        fi
    done
    echo
fi

if [[ "$ENABLE_TIMBRE_CURVES" == "true" ]]; then
    echo "🎨 Step 1.8: Timbre Curves for Synth/Pad (Phase D)"
    for stem_role in "Synth" "Pad" "SynthPad" "Keys"; do
        STEM_WAV=$(find "${STEMS_DIR:-$SONG_DIR}" -name "*${stem_role}*.wav" 2>/dev/null | head -1)
        if [[ -f "$STEM_WAV" ]]; then
            OUT_TIMBRE="$SONG_DIR/${stem_role,,}_timbre.parquet"
            "$PYTHON_BIN" ops/ddsp_timbre_curves.py \
                --audio "$STEM_WAV" \
                --bars "$SONG_DIR/bars.parquet" \
                --out "$OUT_TIMBRE" \
                --hop-ms 20 --smooth-ms 200 || echo "⚠️  Timbre curves failed for $stem_role (NO-OP)"
            [[ -f "$OUT_TIMBRE" ]] && echo "✅ Timbre curves: $OUT_TIMBRE"
        fi
    done
    echo
fi

# 3. Drums 生成（rule / ml / real）
CURRENT_STEP="Drums Recommendations"
DRUMS_REC="$SONG_DIR/drums_recommendations.json"

# Stems特徴ファイルパス（存在チェック）
STEMS_FEATURES="$SONG_DIR/stem_features.parquet"
STEMS_ARGS=()
if [[ -f "$STEMS_FEATURES" ]]; then
    STEMS_ARGS=("--stems-features" "$STEMS_FEATURES")
    echo "🎯 Stems features detected: $STEMS_FEATURES"
else
    echo "⚠️  No stem_features.parquet found, using bars.parquet only"
fi

if [[ "$DRUMS_MODE" != "real" ]]; then
  if [[ ! -f "$DRUMS_REC" ]]; then
    if [[ "$DRUMS_MODE" == "magenta" ]]; then
        echo "🥁 Step 2: Drums (Magenta GrooVAE humanize)"
        
        # 🧹 Purge old drum artifacts (if --force-regenerate-drums)
        if [[ "$FORCE_REGENERATE_DRUMS" == "true" ]]; then
            echo "🧹 Force regenerate: purging cached drums artifacts"
            rm -f "$SONG_DIR/drums_plan.json" \
                  "$SONG_DIR/drums_plan.log" \
                  "$SONG_DIR/drums_seed.mid" \
                  "$SONG_DIR/drums_grooved.mid" \
                  "$SONG_DIR/drums_plan_seed.json"
            # 過去のMagenta出力も削除（latest linkを破棄）
            rm -rf "${WORKSPACE_ROOT}/data/Magenta_Studio/outputs/$(basename "$SONG_DIR")/latest"
        fi
        
        # Magenta出力ディレクトリ（証跡保存用）
        MAG_OUT_DIR="${WORKSPACE_ROOT}/data/Magenta_Studio/outputs/$(basename "$SONG_DIR")/$(date +%Y%m%d_%H%M%S)"
        mkdir -p "$MAG_OUT_DIR"
        echo "📂 Magenta outputs: $MAG_OUT_DIR"
        
        # パッチ6: Phase E2E Magenta中間物再利用防止ログ強化
        echo "   [Magenta] seed/grooved path: $MAG_OUT_DIR"
        if [[ -d "$MAG_OUT_DIR" ]]; then
            echo "   [Magenta] will write fresh intermediates here (stale reuse prevented)"
        fi
        
        # rule推薦→MIDI化→GrooVAE→plan再化（安全ルート）
        DRUMS_REC_SEED="${SONG_DIR}/drums_recommendations_seed.json"
        
        # Step 2.1: Rule-based seed生成
        if ! "$PYTHON_BIN" scripts/recommend_drums.py \
            --song-package "$SONG_PKG" \
            --output "$DRUMS_REC_SEED" \
            --no-ml \
            --topk "$TOPK" \
            "${STEMS_ARGS[@]}"; then
            echo "⚠️  Magenta seed generation failed, falling back to rule"
            DRUMS_MODE="rule"
            "$PYTHON_BIN" scripts/recommend_drums.py \
                --song-package "$SONG_PKG" \
                --output "$DRUMS_REC" \
                --no-ml \
                --topk "$TOPK" \
                "${STEMS_ARGS[@]}" || exit 1
        else
            # Step 2.2: Seed→plan→MIDI
            "$PYTHON_BIN" scripts/adapt_drums_to_plan.py \
                --recommendations "$DRUMS_REC_SEED" \
                --out "$SONG_DIR/drums_plan_seed.json" \
                --tempo-bpm "$TEMPO_BPM" \
                2>&1 | tee "$MAG_OUT_DIR/magenta_seed_plan.log" || exit 1
            
            "$PYTHON_BIN" scripts/midi_writer.py \
                --plan "$SONG_DIR/drums_plan_seed.json" \
                --out "$SONG_DIR/drums_seed.mid" \
                --bars "$SONG_DIR/bars.parquet" \
                2>&1 | tee "$MAG_OUT_DIR/magenta_seed_gen.log" || exit 1
            
            # Step 2.3: GrooVAE humanize
            echo "🥁 Magenta Groove: seed→grooved"
            if "$MAGENTA_PY" "${WORKSPACE_ROOT}/ops/magenta_groove.py" groove \
                -i "$SONG_DIR/drums_seed.mid" \
                -o "$SONG_DIR/drums_grooved.mid" \
                --temp 0.7 \
                2>&1 | tee "$MAG_OUT_DIR/magenta_groove.log"; then
                
                # 証跡退避
                cp -f "$SONG_DIR/drums_seed.mid" "$MAG_OUT_DIR/"
                cp -f "$SONG_DIR/drums_grooved.mid" "$MAG_OUT_DIR/"
                echo "✅ Magenta証跡保存: $MAG_OUT_DIR"
                
                # パッチ1: Magenta中間物必須化ガード（stale reuse防止）
                if [[ ! -s "$MAG_OUT_DIR/drums_seed.mid" || ! -s "$MAG_OUT_DIR/drums_grooved.mid" ]]; then
                    echo "❌ Magenta intermediates missing (seed/grooved). Abort to avoid stale reuse."
                    exit 1
                fi
                
                # Step 2.4: Grooved MIDI→plan（adapt_drums_to_planで--grooved-mid使用）
                "$PYTHON_BIN" scripts/adapt_drums_to_plan.py \
                    --recommendations "$DRUMS_REC_SEED" \
                    --grooved-mid "$SONG_DIR/drums_grooved.mid" \
                    --out "$SONG_DIR/drums_plan.json" \
                    --tempo-bpm "$TEMPO_BPM" \
                    --bars "$SONG_DIR/bars.parquet" \
                    "${STEMS_ARGS[@]}" || exit 1
                
                # drums_recommendations.jsonへコピー（後続処理用）
                cp "$SONG_DIR/drums_plan.json" "$DRUMS_REC"
                echo "✅ Magenta groove applied successfully"
            else
                echo "⚠️  Magenta groove failed, using seed plan"
                cp "$SONG_DIR/drums_plan_seed.json" "$DRUMS_REC"
            fi
        fi
    elif $USE_ML || [[ "$DRUMS_MODE" == "ml" ]]; then
        echo "🥁 Step 2: Drums Recommendations (ML mode)"
        if ! "$PYTHON_BIN" scripts/recommend_drums.py \
            --song-package "$SONG_PKG" \
            --output "$DRUMS_REC" \
            --topk "$TOPK" \
            "${STEMS_ARGS[@]}"; then
            echo "❌ Drums recommendations failed"
            exit 1
        fi
    else
        echo "🥁 Step 2: Drums Recommendations (Rule-Based)"
        if ! "$PYTHON_BIN" scripts/recommend_drums.py \
            --song-package "$SONG_PKG" \
            --output "$DRUMS_REC" \
            --no-ml \
            --topk "$TOPK" \
            "${STEMS_ARGS[@]}"; then
            echo "❌ Drums recommendations failed"
            exit 1
        fi
    fi
    # Plan存在チェック
    if [[ ! -f "$DRUMS_REC" ]]; then
        echo "❌ drums_recommendations.json not created"
        exit 1
    fi
    echo
  else
    echo "🔁 Step 2: Reusing existing $DRUMS_REC"
    echo
  fi
else
  echo "🥁 Step 2: Drums (real MIDI → plan)"
  if [[ ! -f "$SONG_DIR/drums.mid" ]]; then
    echo "❌ real モードですが $SONG_DIR/drums.mid がありません"
    exit 1
  fi
  # Phase A: bars.parquet渡してゴーストHH自動補完
  BARS_ARG=""
  if [[ -f "$SONG_DIR/bars.parquet" ]]; then
    BARS_ARG="--bars $SONG_DIR/bars.parquet"
  fi
  "$PYTHON_BIN" scripts/drums_midi_to_plan.py \
    --drums-mid "$SONG_DIR/drums.mid" \
    --out "$SONG_DIR/drums_plan.json" \
    --tempo-bpm "$TEMPO_BPM" \
    $BARS_ARG
  echo
fi

# === Activity列存在チェック（楽器別density/velocity調整） ===
echo "🎯 Checking instrument activity columns in stem_features..."
GUITAR_ACTIVITY=""
PIANO_ACTIVITY=""
STRINGS_ACTIVITY=""

if [[ -f "$SONG_DIR/stem_features.parquet" ]]; then
    # Pythonでparquet列名チェック
    ACTIVITY_CHECK=$("$PYTHON_BIN" -c "
import pandas as pd
import sys
try:
    df = pd.read_parquet('$SONG_DIR/stem_features.parquet')
    cols = set(df.columns)
    result = []
    if 'guitar_activity' in cols:
        result.append('guitar')
    if 'piano_activity' in cols:
        result.append('piano')
    if 'strings_activity' in cols:
        result.append('strings')
    print(' '.join(result))
except Exception as e:
    print('', file=sys.stderr)
    sys.exit(0)
" 2>/dev/null)

    # 各楽器のactivity列設定
    for inst in $ACTIVITY_CHECK; do
        case "$inst" in
            guitar)
                GUITAR_ACTIVITY="--activity-col guitar_activity"
                echo "   ✅ guitar_activity detected → density/velocity will be adjusted"
                ;;
            piano)
                PIANO_ACTIVITY="--activity-col piano_activity"
                echo "   ✅ piano_activity detected → density/velocity will be adjusted"
                ;;
            strings)
                STRINGS_ACTIVITY="--activity-col strings_activity"
                echo "   ✅ strings_activity detected → density/velocity will be adjusted"
                ;;
        esac
    done
    
    if [[ -z "$GUITAR_ACTIVITY" && -z "$PIANO_ACTIVITY" && -z "$STRINGS_ACTIVITY" ]]; then
        echo "   ℹ️  No instrument activity columns found (using default behavior)"
    fi
else
    echo "   ⚠️  No stem_features.parquet found, skipping activity detection"
fi
echo

# 4. Bass/Guitar/Piano/Strings Plans（高機能ルート：Stage2 実グルーヴ）
CURRENT_STEP="Instrument Plans (real groove)"
echo "🎸 Step 3 & 🎹 Step 4: Instruments via instrument_midi_to_plan_real.py (Phase 13–32)"

# 3-1) Bass
echo "   ▸ Bass (Stage2 real groove) [STRICT + DEBUG]"
# Phase E: Bass F0オプション追加
BASS_F0_OPT=""
if [[ -f "$SONG_DIR/bass_f0.parquet" ]]; then
    BASS_F0_OPT="--bass-f0 $SONG_DIR/bass_f0.parquet"
    echo "      [Phase E] Bass F0 detected: $SONG_DIR/bass_f0.parquet"
fi
# EmotionAI/和声AIオプション
EMOTION_AI_OPTS=()
if [[ "$ENABLE_EMOTION_AI" == "true" ]]; then
    EMOTION_AI_OPTS+=("--enable-emotion-ai")
    EMOTION_AI_OPTS+=("--emotion-profile" "$EMOTION_PROFILE")
    echo "      [EmotionAI] Enabled with profile: $EMOTION_PROFILE"
fi
if [[ "$ENABLE_HARMONY_AI" == "true" ]]; then
    EMOTION_AI_OPTS+=("--enable-harmony-ai")
    echo "      [Harmony AI] Enabled"
fi
if ! "$PYTHON_BIN" scripts/instrument_midi_to_plan_real.py \
    --role bass \
    "${INSTR_ARGS[@]}" \
    --tension-policy auto \
    --walking-bass \
    --voice-leading \
    --multi-chords \
    --anchors-strict \
    --follow-drum-density \
    $BASS_F0_OPT \
    "${EMOTION_AI_OPTS[@]}" \
    --out "$SONG_DIR/bass_plan.json" \
    2>&1 | tee "$SONG_DIR/bass_plan.log"
then
  echo "❌ Bass plan generation failed"
  cat "$SONG_DIR/bass_plan.log" | tail -20
  exit 1
fi
[[ -f "$SONG_DIR/bass_plan.json" ]] || { echo "❌ bass_plan.json not created"; exit 1; }
# リッチネス即時確認
"$PYTHON_BIN" -c "
import json
j=json.load(open('$SONG_DIR/bass_plan.json'))
tr=j.get('tracks',[{}])[0]; ev=tr.get('events',[])
p=len({e['pitch'] for e in ev}); v=len({e.get('velocity',e.get('vel',0)) for e in ev})
d=len({round(e.get('end_beats',0)-e.get('start_beats',0),3) for e in ev})
print(f'   bass: uniq(p)={p} v={v} d={d} total={len(ev)}')
if p<9 or v<8 or d<6:
    raise SystemExit(f'RICHNESS FAIL: p={p}/9 v={v}/8 d={d}/6')
"

# 3-2) Guitar
echo "   ▸ Guitar (Stage2 real groove) [STRICT + DEBUG]"
if ! "$PYTHON_BIN" scripts/instrument_midi_to_plan_real.py \
    --role guitar \
    "${INSTR_ARGS[@]}" \
    --tension-policy auto \
    --strum \
    --strum-direction auto \
    --strum-width-ms 22 \
    --open-voicing auto \
    --capo 0 \
    --multi-chords \
    --voice-leading \
    $GUITAR_ACTIVITY \
    --anchors-strict \
    --follow-drum-density \
    "${EMOTION_AI_OPTS[@]}" \
    --out "$SONG_DIR/guitar_plan.json" \
    2>&1 | tee "$SONG_DIR/guitar_plan.log"
then
  echo "❌ Guitar plan generation failed"
  cat "$SONG_DIR/guitar_plan.log" | tail -20
  exit 1
fi
[[ -f "$SONG_DIR/guitar_plan.json" ]] || { echo "❌ guitar_plan.json not created"; exit 1; }
"$PYTHON_BIN" -c "
import json
j=json.load(open('$SONG_DIR/guitar_plan.json'))
tr=j.get('tracks',[{}])[0]; ev=tr.get('events',[])
p=len({e['pitch'] for e in ev}); v=len({e.get('velocity',e.get('vel',0)) for e in ev})
d=len({round(e.get('end_beats',0)-e.get('start_beats',0),3) for e in ev})
print(f'   guitar: uniq(p)={p} v={v} d={d} total={len(ev)}')
if p<10 or v<8 or d<6:
    raise SystemExit(f'RICHNESS FAIL: p={p}/10 v={v}/8 d={d}/6')
"

# 4-1) Piano
echo "   ▸ Piano (Stage2 real groove) [STRICT + DEBUG]"
# Phase E: Piano OaFオプション追加
PIANO_OAF_OPT=""
if [[ -f "$SONG_DIR/piano_oaf.json" ]]; then
    PIANO_OAF_OPT="--oaf-piano $SONG_DIR/piano_oaf.json"
    echo "      [Phase E] Piano OaF detected: $SONG_DIR/piano_oaf.json"
fi
if ! "$PYTHON_BIN" scripts/instrument_midi_to_plan_real.py \
    --role piano \
    "${INSTR_ARGS[@]}" \
    --voice-leading \
    --multi-chords \
    $PIANO_ACTIVITY \
    $PIANO_OAF_OPT \
    --anchors-strict \
    --follow-drum-density \
    "${EMOTION_AI_OPTS[@]}" \
    --out "$SONG_DIR/piano_plan.json" \
    2>&1 | tee "$SONG_DIR/piano_plan.log"
then
  echo "❌ Piano plan generation failed"
  cat "$SONG_DIR/piano_plan.log" | tail -20
  exit 1
fi
[[ -f "$SONG_DIR/piano_plan.json" ]] || { echo "❌ piano_plan.json not created"; exit 1; }
"$PYTHON_BIN" -c "
import json
j=json.load(open('$SONG_DIR/piano_plan.json'))
tr=j.get('tracks',[{}])[0]; ev=tr.get('events',[])
p=len({e['pitch'] for e in ev}); v=len({e.get('velocity',e.get('vel',0)) for e in ev})
d=len({round(e.get('end_beats',0)-e.get('start_beats',0),3) for e in ev})
print(f'   piano: uniq(p)={p} v={v} d={d} total={len(ev)}')
if p<10 or v<8 or d<6:
    raise SystemExit(f'RICHNESS FAIL: p={p}/10 v={v}/8 d={d}/6')
"

# 4-2) Strings
echo "   ▸ Strings (Stage2 real groove) [STRICT + DEBUG]"
# Phase E: Timbre Curvesオプション追加（Synth/Pad代表としてstringsに適用）
TIMBRE_CURVES_OPT=""
# Synth/Pad用のtimbre curves検索（複数パターン対応）
for STEM_ROLE in "synthpad" "synth" "pad" "keys"; do
    if [[ -f "$SONG_DIR/${STEM_ROLE}_timbre.parquet" ]]; then
        TIMBRE_CURVES_OPT="--timbral-curves $SONG_DIR/${STEM_ROLE}_timbre.parquet"
        echo "      [Phase E] Timbre curves detected: $SONG_DIR/${STEM_ROLE}_timbre.parquet"
        break
    fi
done
if ! "$PYTHON_BIN" scripts/instrument_midi_to_plan_real.py \
    --role strings \
    "${INSTR_ARGS[@]}" \
    --voice-leading \
    --multi-chords \
    $STRINGS_ACTIVITY \
    $TIMBRE_CURVES_OPT \
    --anchors-strict \
    --follow-drum-density \
    "${EMOTION_AI_OPTS[@]}" \
    --out "$SONG_DIR/strings_plan.json" \
    2>&1 | tee "$SONG_DIR/strings_plan.log"
then
  echo "❌ Strings plan generation failed"
  cat "$SONG_DIR/strings_plan.log" | tail -20
  exit 1
fi
[[ -f "$SONG_DIR/strings_plan.json" ]] || { echo "❌ strings_plan.json not created"; exit 1; }
"$PYTHON_BIN" -c "
import json
j=json.load(open('$SONG_DIR/strings_plan.json'))
tr=j.get('tracks',[{}])[0]; ev=tr.get('events',[])
p=len({e['pitch'] for e in ev}); v=len({e.get('velocity',e.get('vel',0)) for e in ev})
d=len({round(e.get('end_beats',0)-e.get('start_beats',0),3) for e in ev})
print(f'   strings: uniq(p)={p} v={v} d={d} total={len(ev)}')
if p<6 or v<6 or d<4:
    raise SystemExit(f'RICHNESS FAIL: p={p}/6 v={v}/6 d={d}/4')
"

echo

# 6. Drums Plan生成（rule/mlモードのみ）
if [[ "$DRUMS_MODE" != "real" ]]; then
  echo "🥁 Step 5: Drums Plan (hybrid v2: WAV×MIDI fusion)"
  
  # Base arguments (配列形式)
  DRUMS_ARGS=(
    "--out" "$SONG_DIR/drums_plan.json"
    "--tempo-bpm" "$TEMPO_BPM"
    "--recommendations" "$DRUMS_REC"
  )
  
  # Hybrid sources (optional, backward compatible)
  if [[ -f "$SONG_DIR/bars.parquet" ]]; then
    DRUMS_ARGS+=("--bars" "$SONG_DIR/bars.parquet")
  fi
  if [[ -f "$SONG_DIR/drums.mid" ]]; then
    DRUMS_ARGS+=("--stem-midi" "$SONG_DIR/drums.mid")
    echo "   ✅ Using stem MIDI: drums.mid (weak labels)"
  fi
  if [[ -f "$SONG_DIR/stem_features.parquet" ]]; then
    DRUMS_ARGS+=("--stems-features" "$SONG_DIR/stem_features.parquet")
    echo "   ✅ Using stem features: density/hat_density"
  fi
  if [[ -f "$SONG_DIR/lyric_anchors.json" ]]; then
    DRUMS_ARGS+=("--lyric-anchors" "$SONG_DIR/lyric_anchors.json")
    echo "   ✅ Using lyric anchors: vocal ducking"
  fi
  
  # Open-hat policy
  DRUMS_ARGS+=("--oh-open-prob" "0.25" "--oh-close-delay" "0.20" "--oh-avoid-vocal")
  
  # Tom fills (enable for dynamic sections)
  DRUMS_ARGS+=("--enable-fills" "--fill-when" "section,cadence" "--fill-palette" "mid")
  DRUMS_ARGS+=("--fill-max-notes" "8" "--fill-strength" "0.9" "--fill-crash-next")
  
  # KPI強化オプション（Backbeat保障 + ライド→タム2段フィル + 軽フラム）
  DRUMS_ARGS+=("--enforce-backbeat" "--min-backbeat-vel" "86")
  DRUMS_ARGS+=("--light-flam" "--fill-l2")
  
  "$PYTHON_BIN" scripts/adapt_drums_to_plan.py "${DRUMS_ARGS[@]}"
  echo
fi

# 6.5 Drums channel normalization (GM ch10 / index 9)
if [[ -f "$SONG_DIR/drums_plan.json" ]]; then
  echo "🛠  Step 6.5: Normalize drum channel to ch10"
  if ! "$PYTHON_BIN" scripts/fix_plan_drum_channel.py \
      --in "$SONG_DIR/drums_plan.json" \
      --out "$SONG_DIR/drums_plan.json" \
      --channel 9; then
      echo "⚠️  Drum channel normalization failed (continuing)"
  else
      echo "   ✅ Drum channel normalized to 9"
  fi
  echo
fi

# 7. 全パート統合
echo "🎼 Step 6: Full Arrangement (5 tracks)"

# Plan存在チェック
for P in drums bass guitar piano strings; do
    PLAN_FILE="$SONG_DIR/${P}_plan.json"
    if [[ ! -f "$PLAN_FILE" ]]; then
        echo "❌ Missing plan: $PLAN_FILE"
        exit 1
    fi
done

python3 scripts/arrangement_orchestrator.py \
    --drums "$SONG_DIR/drums_plan.json" \
    --bass "$SONG_DIR/bass_plan.json" \
    --guitar "$SONG_DIR/guitar_plan.json" \
    --piano "$SONG_DIR/piano_plan.json" \
    --strings "$SONG_DIR/strings_plan.json" \
    --tempo-bpm "$TEMPO_BPM" \
    --out "$SONG_DIR/full_arrangement.json"
echo

# 6.1. Plan normalization（bar/beat backfill）
echo "🧭 Step 6.1: Plan normalization (bar/beat backfill)"
for P in drums bass guitar piano strings; do
  "$PYTHON_BIN" scripts/backfill_bar_beat.py \
      --plan "$SONG_DIR/${P}_plan.json" \
      --bars "$SONG_DIR/bars.parquet" || true
done
echo

# 6.5. Plan schema normalization（イベントへ channel/velocity を補完）
echo "🧰 Step 6.5: Normalize plan event schema"
"$PYTHON_BIN" ops/plan_normalize_schema.py \
  --in  "$SONG_DIR/full_arrangement.json" \
  --out "$SONG_DIR/full_arrangement.json"
echo

# 6.6. Full arrangement bar/beat backfill
echo "🧭 Step 6.6: Full arrangement bar/beat backfill"
"$PYTHON_BIN" scripts/backfill_bar_beat.py \
  --plan "$SONG_DIR/full_arrangement.json" \
  --bars "$SONG_DIR/bars.parquet"
echo

# 6.7. Plan missing fields fix (start_beats/end_beats/vel/velocity/channel補完)
echo "🔧 Step 6.7: Fix missing fields (start_beats/end_beats/vel/velocity)"
"$PYTHON_BIN" scripts/plan_fix_missing_fields.py \
  --in  "$SONG_DIR/full_arrangement.json" \
  --out "$SONG_DIR/full_arrangement.json" \
  --bars "$SONG_DIR/bars.parquet"
echo

# 7. Plan検証
echo "✅ Step 7: Plan Validation"

# 7-α. Drum channel fixer (force MIDI ch10 before validation)
echo "🔧 Step 7-α: Fix drum channel to MIDI ch10 (index 9)"
python3 scripts/fix_plan_drum_channel.py \
    --in "$SONG_DIR/full_arrangement.json" \
    --required-channel 9 \
    --detect-gm --min-gm-ratio 0.6 || true
echo

"$PYTHON_BIN" scripts/validate_plan.py "$SONG_DIR/full_arrangement.json" --require-drum-channel 9
echo

# 9. MIDI生成
if $DRY_RUN; then
    echo "🔸 Step 8: MIDI Generation (SKIPPED: --dry-run)"
    echo "   Plan validation completed. MIDI generation skipped."
    echo
else
    echo "🎵 Step 8: MIDI Generation"
    "$PYTHON_BIN" scripts/midi_writer.py \
        --plan "$SONG_DIR/full_arrangement.json" \
        --out "$SONG_DIR/full_arrangement.mid" \
        --config configs/plan_humanize.yaml \
        --bars "$SONG_DIR/bars.parquet"
    echo
    
    # 10. 統計表示
    echo "📊 Step 9: MIDI Statistics"
    python3 -c "
from mido import MidiFile
mid = MidiFile('$SONG_DIR/full_arrangement.mid')
print(f'   Tracks: {len(mid.tracks)}')
print(f'   PPQ: {mid.ticks_per_beat}')
total = sum(1 for tr in mid.tracks for msg in tr if msg.type == 'note_on')
print(f'   Total notes: {total}')
print(f'   Duration: {mid.length:.1f}s ({mid.length/60:.1f}min)')
"
    echo
fi

# 11. KPI Gate（任意）
if $RUN_KPI && [[ -f "$SONG_DIR/full_arrangement.mid" ]]; then
    echo "📏 Step 10: KPI Gate Validation (bars.parquet基準)"
    if [[ ! -f "$SONG_DIR/bars.parquet" ]]; then
      echo "❌ bars.parquet が無いためKPI Gate実行不可（必須）"
      exit 1
    fi
    "$PYTHON_BIN" scripts/kpi_gate_enhanced.py \
        --midi "$SONG_DIR/full_arrangement.mid" \
        --bars "$SONG_DIR/bars.parquet" \
        --gate-config configs/gate_prod.yaml \
        --tempo-bpm "$TEMPO_BPM" \
        --skip-quiet-bars \
        --drums-active-col drums_active \
        --drums-active-threshold 0.5 \
        --output "$SONG_DIR/kpi_gate_postgen.json" || true
    echo
fi

# 10.5 KPI Auto-Repair (context aware: cooperative fix - snare+kick+HH)
if $RUN_KPI && [[ -f "$SONG_DIR/kpi_gate_postgen.json" ]]; then
  echo "🩹 Step 10.5: KPI Auto-Repair (cooperative fix - snare+kick+HH)"
  SONG_DIR="$SONG_DIR" "$PYTHON_BIN" - <<'PY'
import json, pathlib, os
import pandas as pd
from math import isfinite

base = pathlib.Path(os.environ["SONG_DIR"])
plan_path = base/"full_arrangement.json"
bars_path = base/"bars.parquet"
kpi_path  = base/"kpi_gate_postgen.json"

data = json.loads(plan_path.read_text())
bars = pd.read_parquet(bars_path)
if "bar_index" in bars.columns and "bar" not in bars.columns:
    bars["bar"] = bars["bar_index"]
bars = bars.set_index("bar", drop=False)

kpi = json.loads(kpi_path.read_text())
results = kpi.get("results", {})

# 失敗バー抽出（backbeat_strength不足のみ）
fails = []
for bar_key, bar_result in results.items():
    if not bar_result.get("kpi_pass", True):
        messages = bar_result.get("messages", [])
        if any("backbeat_strength too low" in msg for msg in messages):
            fails.append({"bar": int(bar_key.split("_")[1]), "reason": " ".join(messages)})

# drums track
drums_tr = next((t for t in data["tracks"] if t.get("role","").lower()=="drums"), None)
if not drums_tr:
    print("ℹ️  no drums track")
    print(json.dumps({"fixed_events":0}))
    raise SystemExit(0)
evs = drums_tr["events"]

def clamp(v,a,b): return a if v<a else b if v>b else v

fixed = 0
for f in fails:
    b = int(f["bar"])
    # Quiet bar? → 補修しない（KPIはパッチAで除外済み）
    if "drums_active" in bars.columns and float(bars.loc[b,"drums_active"]) < 0.5:
        continue

    bar_start = b*4.0
    target_beats = [2.0, 4.0]  # 2拍目・4拍目
    win = 0.12  # ±0.12拍 (約60ms@100BPM) 以内を同拍判定

    # energyベースのVel
    energy = float(bars.loc[b].get("energy_curve", 0.65)) if b in bars.index else 0.65
    
    # 1) スネアを確実に置く / 強化（velocity 上限 127）
    for tb in target_beats:
        hits = [e for e in evs if e.get("bar")==b and e.get("pitch") in (38,40)
                and abs((e.get("start_beats", bar_start + e.get("beat",0.0)) - bar_start) - tb) <= win]
        if not hits:
            # スネアがなければ追加
            snare_vel = max(96, int(100 + 24*energy))
            evs.append({
                "bar": b,
                "beat": round(tb, 3),
                "start_beats": round(bar_start + tb, 3),
                "end_beats": round(bar_start + tb + 0.05, 3),
                "pitch": 38,
                "velocity": snare_vel,
                "vel": snare_vel,
                "channel": 9,
                "role": "drums"
            })
            fixed += 1
        else:
            # 既存スネアを強化
            for e in hits:
                old_vel = e.get("velocity", e.get("vel", 90))
                new_vel = min(127, int(old_vel + 24))
                e["velocity"] = new_vel
                e["vel"] = new_vel

    # 2) 同拍の kick を弱めすぎる場合は軽くダッキング（被り防止）
    for tb in target_beats:
        kicks = [e for e in evs if e.get("bar")==b and e.get("pitch") in (35,36)
                 and abs((e.get("start_beats", bar_start + e.get("beat",0.0)) - bar_start) - tb) <= win]
        for e in kicks:
            old_vel = e.get("velocity", e.get("vel", 80))
            new_vel = max(1, int(old_vel * 0.75))
            e["velocity"] = new_vel
            e["vel"] = new_vel

    # 3) HH を裏拍にアクセント（抜け感で backbeat を聴かせる）
    for tb in target_beats:
        hh_time = tb - 0.5  # 1.5拍/3.5拍あたり
        if hh_time > 0.0:
            hh_vel = min(127, int(70 + 40*energy))
            evs.append({
                "bar": b,
                "beat": round(hh_time, 3),
                "start_beats": round(bar_start + hh_time, 3),
                "end_beats": round(bar_start + hh_time + 0.03, 3),
                "pitch": 42,  # Closed HH
                "velocity": hh_vel,
                "vel": hh_vel,
                "channel": 9,
                "role": "drums"
            })
            fixed += 1

drums_tr["events"] = sorted(evs, key=lambda x:(x.get("bar",0), x.get("start_beats",0.0)))
plan_path.write_text(json.dumps(data, ensure_ascii=False, indent=2))
print(json.dumps({"fixed_events":fixed}))
PY
  # 再MIDI化 → KPI 再評価
  echo "🎵  Regenerating MIDI after auto-repair (if any)"
  "$PYTHON_BIN" scripts/midi_writer.py \
      --plan "$SONG_DIR/full_arrangement.json" \
      --out "$SONG_DIR/full_arrangement.mid" \
      --config configs/plan_humanize.yaml \
      --bars "$SONG_DIR/bars.parquet"
  echo "📏 Re-run KPI Gate after auto-repair"
  "$PYTHON_BIN" scripts/kpi_gate_enhanced.py \
      --midi "$SONG_DIR/full_arrangement.mid" \
      --bars "$SONG_DIR/bars.parquet" \
      --gate-config configs/gate_prod.yaml \
      --tempo-bpm "$TEMPO_BPM" \
      --skip-quiet-bars \
      --drums-active-col drums_active \
      --drums-active-threshold 0.5 \
      --output "$SONG_DIR/kpi_gate_postgen.json" || true
  echo
fi

# 10.6 Groove Polish (chorus HH boost/open, ride→tom fill, snare flams)
if [[ "$POLISH_GROOVE" == "true" ]]; then
  echo "✨ Step 10.6: Groove Polish (chorus HH boost/open, ride→tom fill, snare flams)"
  SONG_DIR="$SONG_DIR" POLISH_HH_BOOST_MAX="$POLISH_HH_BOOST_MAX" POLISH_HH_OPEN_RATE="$POLISH_HH_OPEN_RATE" POLISH_TOM_FILL="$POLISH_TOM_FILL" POLISH_SNARE_FLAM="$POLISH_SNARE_FLAM" "$PYTHON_BIN" - <<'PY'
import json, pathlib, random, os
import pandas as pd

p = pathlib.Path(os.environ["SONG_DIR"])
plan_path = p/"full_arrangement.json"
bars_path = p/"bars.parquet"

# 読み込み
data = json.loads(plan_path.read_text())
bars = pd.read_parquet(bars_path)

# bar列名の吸収
bar_col = "bar"
if "bar" not in bars.columns and "bar_index" in bars.columns:
    bar_col = "bar_index"
bars = bars.set_index(bar_col)

# 参照列の存在チェック（不足時はデフォルト）
def bx(bar, key, default):
    try:
        return bars.loc[bar].get(key, default)
    except Exception:
        return default

def clamp(x, lo, hi): return lo if x<lo else hi if x>hi else x

random.seed(42)

# パラメータ（env受け取り）
HH_BOOST_MAX = int(os.environ.get("POLISH_HH_BOOST_MAX","10"))
HH_OPEN_RATE = float(os.environ.get("POLISH_HH_OPEN_RATE","0.15"))
USE_TOM_FILL = os.environ.get("POLISH_TOM_FILL","true") == "true"
USE_FLAM     = os.environ.get("POLISH_SNARE_FLAM","true") == "true"

# マップ
HH_PITCHES = {42,44}         # Closed/ Pedal
OH_PITCH   = 46              # Open HH
RIDE_PITCH = {51,59,53}      # Ride/ Ride bell/ Ride cym
TOMS_SEQ   = [47,50,45,43]   # Mid,Hi,LowMid,Low（GM）
SNARES     = {38,40}

total_boost=0; opened=0; tomfills=0; flams=0

# 前方参照用に section変化を検知
bars["__next_section"] = bars["section_label"].shift(-1)
bars["__is_transition"] = (bars["section_label"]!=bars["__next_section"]).fillna(False)

for tr in data.get("tracks",[]):
    if tr.get("role","").lower()!="drums": 
        continue
    evs = tr.get("events",[])
    # 1) コーラスでHHを少し持ち上げ＆一部オープン化
    for ev in evs:
        bar = ev.get("bar")
        if bar is None: 
            continue
        sec = bx(bar, "section_label", "verse")
        if sec == "chorus":
            pitch = ev.get("pitch")
            if pitch in HH_PITCHES:
                base = int(ev.get("velocity", ev.get("vel", 64)))
                energy = float(bx(bar, "energy_curve", 0.6))
                boost = int(round(min(HH_BOOST_MAX, 4 + energy*HH_BOOST_MAX/1.6)))
                newv = clamp(base + boost, 1, 127)
                ev["velocity"]=ev["vel"]=newv
                total_boost += 1
                # offbeatのみ少確率でOpen化（拍+0.5近辺）
                sb = ev.get("start_beats", ev.get("bar",0)*4 + ev.get("beat",0.0))
                beat_in_bar = sb - int(sb//4)*4.0
                if abs((beat_in_bar % 1.0) - 0.5) < 0.12 and random.random() < HH_OPEN_RATE:
                    ev["pitch"] = OH_PITCH
                    # 少し長め（0.2〜0.35beat）
                    eb = ev.get("end_beats", sb+0.12)
                    eb = max(eb, sb + 0.25)
                    ev["end_beats"] = eb
                    opened += 1
    # 2) セクション遷移 or fill_likelihood高で Ride→Tom フィル（最終拍の置換）
    if USE_TOM_FILL:
        # バーごとにグループ化
        from collections import defaultdict
        bybar = defaultdict(list)
        for i,e in enumerate(evs): 
            if e.get("bar") is not None: 
                bybar[e["bar"]].append((i,e))
        for bar, items in bybar.items():
            is_tr = bool(bars.loc[bar]["__is_transition"]) if bar in bars.index else False
            fill_lk = float(bx(bar, "fill_likelihood", 0.0))
            if not (is_tr or fill_lk >= 0.55):
                continue
            # 最終拍 [3.0..4.0) のRideを間引き→Tom 16分4連を注入
            start_bar_beat = bar*4.0
            last_beat_start = start_bar_beat + 3.0
            # 対象Rideを削除フラグ
            to_del = []
            for idx,e in items:
                sb = e.get("start_beats", start_bar_beat + e.get("beat",0.0))
                if last_beat_start <= sb < start_bar_beat+4.0 and e.get("pitch") in RIDE_PITCH:
                    to_del.append(idx)
            # 削除（後ろから）
            for idx in sorted(to_del, reverse=True):
                del evs[idx]
            # Tom 4連（0.25刻み）
            basev = items[0][1].get("velocity", items[0][1].get("vel", 76)) if items else 76
            for k,pit in enumerate(TOMS_SEQ):
                sb = last_beat_start + 0.25*k
                if sb >= start_bar_beat + 4.0: 
                    break
                evs.append({
                    "bar": bar,
                    "beat": sb - start_bar_beat,
                    "start_beats": sb,
                    "end_beats": min(sb+0.22, start_bar_beat+4.0),
                    "pitch": pit,
                    "velocity": clamp(int(basev + 4 + k*2), 1, 127),
                    "vel": clamp(int(basev + 4 + k*2), 1, 127),
                    "channel": 9,
                    "role": "drums"
                })
            tomfills += 1
    # 3) スネア・フラム（-0.03beatの軽ドラッグ）
    if USE_FLAM:
        new_notes=[]
        for ev in evs:
            new_notes.append(ev)
            if ev.get("pitch") in SNARES:
                sb = ev.get("start_beats", ev.get("bar",0)*4 + ev.get("beat",0.0))
                beat_in_bar = sb - int(sb//4)*4.0
                if abs(beat_in_bar-2.0)<0.07 or abs(beat_in_bar-4.0)<0.07:
                    pre = max(sb-0.03, int(sb//4)*4.0)  # 負側は同小節内に制限
                    if pre < sb:  # 余地があるときだけ
                        v = clamp(int(0.75*int(ev.get("velocity", ev.get("vel",72)))), 20, 110)
                        new_notes.append({
                            **{k:v for k,v in ev.items() if k in ("bar","channel","role","pitch")},
                            "start_beats": pre,
                            "end_beats": sb,
                            "velocity": v, "vel": v
                        })
                        flams += 1
        tr["events"] = sorted(new_notes, key=lambda x: (x.get("bar",0), x.get("start_beats",0.0)))

print(f"✅ GroovePolish: HH boosted={total_boost}, HH opened={opened}, tomfills={tomfills}, flams={flams}")
plan_path.write_text(json.dumps(data, ensure_ascii=False, indent=2))
PY

  # 再MIDI化 → KPI 再評価
  echo "🎵  Regenerating MIDI after Groove Polish"
  "$PYTHON_BIN" scripts/midi_writer.py \
      --plan "$SONG_DIR/full_arrangement.json" \
      --out "$SONG_DIR/full_arrangement.mid" \
      --config configs/plan_humanize.yaml \
      --bars "$SONG_DIR/bars.parquet"
  if $RUN_KPI; then
    echo "📏 Re-run KPI Gate after Groove Polish"
    "$PYTHON_BIN" scripts/kpi_gate_enhanced.py \
        --midi "$SONG_DIR/full_arrangement.mid" \
        --bars "$SONG_DIR/bars.parquet" \
        --gate-config configs/gate_prod.yaml \
        --tempo-bpm "$TEMPO_BPM" \
        --skip-quiet-bars \
        --drums-active-col drums_active \
        --drums-active-threshold 0.5 \
        --output "$SONG_DIR/kpi_gate_postgen.json" || true
  fi
  echo
fi

# 12. CI回帰防止ガード（最終検証）
if [[ -f "$SONG_DIR/full_arrangement.mid" ]] && [[ ! $DRY_RUN = true ]]; then
    echo "🛡️  Step 11: CI Regression Guard (Final Verification)"
    CURRENT_STEP="CI Verification"
    
    CI_ARGS=(
        "--midi" "$SONG_DIR/full_arrangement.mid"
        "--bars" "$SONG_DIR/bars.parquet"
        "--tempo-bpm" "$TEMPO_BPM"
        "--report" "$SONG_DIR/ci_verify_report.json"
    )
    
    # KPIが有効な場合はCI検証にも含める
    if $RUN_KPI; then
        CI_ARGS+=("--gate-config" "configs/gate_prod.yaml" "--kpi-threshold" "0.90")
    fi
    
    if "$PYTHON_BIN" ops/ci_verify_music_package.py "${CI_ARGS[@]}"; then
        echo "   ✅ CI Verification PASSED"
        echo "      - Tempo meta: Track 0 only"
        echo "      - Downbeats: Match bars.parquet (±1 bar)"
        echo "      - Track durations: Within tolerance"
        echo "      - No overlong notes beyond expected end"
        if $RUN_KPI; then
            echo "      - KPI Gate: Pass rate ≥ 90%"
        fi
    else
        echo "   ❌ CI Verification FAILED"
        echo "      See details: $SONG_DIR/ci_verify_report.json"
        exit 1
    fi
    echo
    
    # 13. 多様性監視（KPI過適合防止）
    echo "🔍 Step 12: Diversity Watch (KPI Overfitting Prevention)"
    CURRENT_STEP="Diversity Watch"
    
    BASELINE_MID="$SONG_DIR/full_arrangement_baseline.mid"
    if [[ -f "$BASELINE_MID" ]]; then
        "$PYTHON_BIN" scripts/diversity_watch.py \
            --current "$SONG_DIR/full_arrangement.mid" \
            --baseline "$BASELINE_MID" \
            --output "$SONG_DIR/diversity_report.json" \
            --threshold 0.20
    else
        # ベースラインがない場合は現在の指標のみ記録
        "$PYTHON_BIN" scripts/diversity_watch.py \
            --current "$SONG_DIR/full_arrangement.mid" \
            --output "$SONG_DIR/diversity_report.json"
        echo "   ℹ️  No baseline found. Recording current features only."
    fi
    echo
fi

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
if $DRY_RUN; then
    echo "✅ E2E Workflow Complete (Dry Run)!"
    echo "   Plans validated: $SONG_DIR/full_arrangement.json"
else
    echo "✅ E2E Workflow Complete!"
    echo "   Output: $SONG_DIR/full_arrangement.mid"
    echo "   CI Report: $SONG_DIR/ci_verify_report.json"
    
    # 14. 再現性タグ焼き込み
    echo
    echo "🔖 Step 13: Stamp Reproducibility Tags"
    CURRENT_STEP="Stamp Reproducibility"
    "$PYTHON_BIN" scripts/stamp_reproducibility_tags.py \
        --song-dir "$SONG_DIR" \
        --arranger-config configs/arranger_weights.yaml \
        --gate-config configs/gate_prod.yaml
fi
