#!/usr/bin/env bash
# ========================================
# Phase A: Audio Analysis (STEP 1-6)
# ========================================
# Output:
#   - tempo_map.json
#   - bars.parquet (with density_target, swing_target, section_label)
#   - sections.json
#   - lyric_anchors.json
#   - chordmap.json (AUTO - reference only, NOT for production)
#   - bars_with_slots.parquet (with fill_slot, riff_slot)
#
# ⚠️  CRITICAL: chordmap.json is AUTO-GENERATED and NOT accurate!
#     → Create manual_chordmap.json before Phase B
# ========================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Default flags
STEM_FEATURES=0  # Skip stem features by default (only for QA/viz)
STRICT_VOCAL=1   # Require vocal WAV
STRICT=1         # Strict mode (exit on error)
DRY_RUN=0

# Optional Local LAMDA bundle build
LOCAL_LAMDA_ENABLE=0
LOCAL_LAMDA_MODE="midi"   # midi | wav
LOCAL_LAMDA_MIDI_ROOT=""
LOCAL_LAMDA_OUT=""
LOCAL_LAMDA_WAV_STEM="accompaniment"
LOCAL_LAMDA_SCRIPT="$REPO_ROOT/scripts/local_lamda/run_build_all.sh"

# Parse arguments
SONG_ROOT=""
STEMS_DIR=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --stems-dir)
      STEMS_DIR="$2"
      shift 2
      ;;
    --stem-features)
      STEM_FEATURES=1
      shift
      ;;
    --no-strict-vocal)
      STRICT_VOCAL=0
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --local-lamda)
      LOCAL_LAMDA_ENABLE=1
      shift
      ;;
    --local-lamda-out)
      LOCAL_LAMDA_OUT="$2"
      shift 2
      ;;
    --local-lamda-midi-root)
      LOCAL_LAMDA_MIDI_ROOT="$2"
      shift 2
      ;;
    --local-lamda-from-wav)
      LOCAL_LAMDA_ENABLE=1
      LOCAL_LAMDA_MODE="wav"
      shift
      ;;
    --local-lamda-wav-stem)
      LOCAL_LAMDA_WAV_STEM="$2"
      shift 2
      ;;
    *)
      SONG_ROOT="$1"
      shift
      ;;
  esac
done

if [[ -z "$SONG_ROOT" ]]; then
  echo "Usage: $0 <song_package_dir> [--stems-dir DIR] [--stem-features] [--no-strict-vocal] [--dry-run] \\
            [--local-lamda] [--local-lamda-midi-root DIR] [--local-lamda-out DIR] [--local-lamda-from-wav] [--local-lamda-wav-stem STEM]"
  echo ""
  echo "Phase A: Audio Analysis (AUTO chordmap generation)"
  echo "  → Creates reference chordmap.json (NOT production-ready)"
  echo "  → ⚠️  Create manual_chordmap.json before running Phase B"
  exit 1
fi

SONG_ROOT="$(cd "$SONG_ROOT" && pwd)"
ANALYSIS_DIR="$SONG_ROOT/analysis"
mkdir -p "$ANALYSIS_DIR"

# Auto-detect stems directory
if [[ -z "$STEMS_DIR" ]]; then
  if [[ -d "$SONG_ROOT/stem_wav" ]]; then
    STEMS_DIR="$SONG_ROOT/stem_wav"
  elif [[ -d "$SONG_ROOT/stems" ]]; then
    STEMS_DIR="$SONG_ROOT/stems"
  else
    echo "❌ Stems directory not found (stem_wav or stems). Use --stems-dir to specify."
    exit 1
  fi
fi

# Default Local LAMDA paths when enabled
if [[ $LOCAL_LAMDA_ENABLE -eq 1 ]]; then
  if [[ -z "$LOCAL_LAMDA_OUT" ]]; then
    LOCAL_LAMDA_OUT="$SONG_ROOT/local_lamda"
  fi

  if [[ $LOCAL_LAMDA_MODE == "midi" && -z "$LOCAL_LAMDA_MIDI_ROOT" ]]; then
    if [[ -d "$SONG_ROOT/midi" ]]; then
      LOCAL_LAMDA_MIDI_ROOT="$SONG_ROOT/midi"
    elif [[ -d "$SONG_ROOT/stem_midi" ]]; then
      LOCAL_LAMDA_MIDI_ROOT="$SONG_ROOT/stem_midi"
    else
      echo "⚠️  Local LAMDA MIDI root not found (expected midi/ or stem_midi/ under song root)."
      [[ $STRICT -eq 1 ]] && exit 1 || LOCAL_LAMDA_ENABLE=0
    fi
  fi
fi

# Python binary
if [[ -f "$REPO_ROOT/.venv311/bin/python" ]]; then
  PYTHON_BIN="$REPO_ROOT/.venv311/bin/python"
elif command -v python3 &>/dev/null; then
  PYTHON_BIN="python3"
else
  echo "❌ Python not found"
  exit 1
fi

echo "========================================="
echo "🎵 Phase A: Audio Analysis"
echo "========================================="
echo "   Song Package: $SONG_ROOT"
echo "   Stems dir: $STEMS_DIR"
echo "   Python: $PYTHON_BIN"
echo ""

# Auto-detect Mix WAV (instrument.wav priority)
echo "🔍 Auto-detecting Mix WAV (instrument.wav優先)..."
MIX_WAV=""
MIX_CANDIDATES=(
  "$STEMS_DIR/instrument.wav"
  "$STEMS_DIR/mix.wav"
  "$STEMS_DIR/Mix.wav"
  "$STEMS_DIR/other.wav"
  "$STEMS_DIR/Other.wav"
  "$STEMS_DIR/stem_wav_001_(Instrumental).wav"
  "$STEMS_DIR/stem_wav_001_(Instrument).wav"
  "$STEMS_DIR/stem_wav_001_(Backing).wav"
  "$STEMS_DIR/stem_wav_001_(Backing Vocals).wav"
  "$STEMS_DIR/stem_wav_001_(Keyboard).wav"
)
for candidate in "${MIX_CANDIDATES[@]}"; do
  if [[ -f "$candidate" ]]; then
    MIX_WAV="$candidate"
    break
  fi
done
if [[ -z "$MIX_WAV" ]]; then
  echo "❌ Mix WAV not found (instrument/mix/other)"
  exit 1
fi
echo "   Mix WAV: $MIX_WAV"

# Auto-detect Vocal WAV
echo "🔍 Auto-detecting Vocal WAV..."
VOCAL_WAV=""
VOCAL_CANDIDATES=(
  "$STEMS_DIR/vocals.wav"
  "$STEMS_DIR/Vocals.wav"
  "$STEMS_DIR/stem_wav_001_(Vocals).wav"
  "$STEMS_DIR/stem_wav_001_(Backing Vocals).wav"
  "$STEMS_DIR/"*ocal*.wav
)
for pattern in "${VOCAL_CANDIDATES[@]}"; do
  for f in $pattern; do
    if [[ -f "$f" ]]; then
      VOCAL_WAV="$f"
      break 2
    fi
  done
done
if [[ -z "$VOCAL_WAV" ]]; then
  echo "   ⚠️  Vocal WAV not found"
  [[ $STRICT_VOCAL -eq 1 ]] && exit 1
else
  echo "   Vocal WAV: $VOCAL_WAV"
fi

# Output paths
STEP1_OUT_BARS="$ANALYSIS_DIR/bars.parquet"
STEP1_OUT_JSON="$ANALYSIS_DIR/tempo_map.json"
STEP2_OUT_JSON="$ANALYSIS_DIR/sections.json"
STEP3_OUT_JSON="$ANALYSIS_DIR/lyric_anchors.json"
STEP4_OUT_JSON="$ANALYSIS_DIR/chordmap.json"
BARS_WITH_SLOTS="$ANALYSIS_DIR/bars_with_slots.parquet"

# ==========================================
# STEP 1: tempo_map.json + bars.parquet
# ==========================================
echo ""
echo "🕐 STEP 1/6: tempo_map.json + bars.parquet"

if [[ -f "$STEP1_OUT_BARS" ]]; then
  echo "   📋 Merge mode: Updating existing bars.parquet..."
  CMD1=("$PYTHON_BIN" "$REPO_ROOT/ops/tempo_map_cli.py" \
        --audio "$MIX_WAV" \
        --bars-in "$STEP1_OUT_BARS" \
        --out-bars "$STEP1_OUT_BARS.tmp" \
        --out-tempo "$STEP1_OUT_JSON" \
        --bpb 4 \
        --prefer-madmom)
  
  if [[ $DRY_RUN -eq 1 ]]; then
    echo "[DRY-RUN] ${CMD1[*]}"
  else
    echo "   Running: ${CMD1[*]}"
    "${CMD1[@]}" && mv "$STEP1_OUT_BARS.tmp" "$STEP1_OUT_BARS" || {
      echo "❌ STEP 1 failed"
      [[ $STRICT -eq 1 ]] && exit 1
    }
  fi
else
  echo "   📋 Fresh mode: Creating new bars.parquet..."
  
  # Get audio duration
  DURATION_SEC=$("$PYTHON_BIN" -c "import librosa; y, sr = librosa.load('$MIX_WAV', sr=None, mono=True); print(len(y) / sr)" 2>/dev/null || echo "")
  
  if [[ -z "$DURATION_SEC" ]]; then
    echo "   ⚠️  Could not detect audio duration, using 180 seconds as default"
    DURATION_SEC=180
  else
    echo "   🎧 Audio duration: ${DURATION_SEC} seconds"
  fi
  
  CMD1=("$PYTHON_BIN" "$REPO_ROOT/ops/tempo_map_cli.py" \
        --audio "$MIX_WAV" \
        --duration-sec "$DURATION_SEC" \
        --out-bars "$STEP1_OUT_BARS" \
        --out-tempo "$STEP1_OUT_JSON" \
        --bpb 4 \
        --prefer-madmom)
  
  if [[ $DRY_RUN -eq 1 ]]; then
    echo "[DRY-RUN] ${CMD1[*]}"
  else
    echo "   Running: ${CMD1[*]}"
    "${CMD1[@]}" || {
      echo "❌ STEP 1 failed"
      [[ $STRICT -eq 1 ]] && exit 1
    }
  fi
fi

# Get median BPM for later steps
MEDIAN_BPM=$("$PYTHON_BIN" -c "import pandas as pd; df=pd.read_parquet('$STEP1_OUT_BARS'); print(df['bpm'].median())" 2>/dev/null || echo "120.0")

# ==========================================
# STEP 2: analysis/sections.json
# ==========================================
echo ""
echo "🕑 STEP 2/6: analysis/sections.json"

CMD2=("$PYTHON_BIN" "$REPO_ROOT/ops/sections_from_audio.py" \
      --stems "$STEMS_DIR" \
      --out "$STEP2_OUT_JSON")

if [[ $DRY_RUN -eq 1 ]]; then
  echo "[DRY-RUN] ${CMD2[*]}"
else
  echo "   Running: ${CMD2[*]}"
  "${CMD2[@]}" || {
    echo "❌ STEP 2 failed"
    [[ $STRICT -eq 1 ]] && exit 1
  }
fi

# STEP 2.5: Update bars.parquet with section metadata
echo ""
echo "🕑.5 STEP 2.5/6: density_target/swing_target/section_label更新"

"$PYTHON_BIN" << EOF
import pandas as pd
import json

bars = pd.read_parquet("$STEP1_OUT_BARS")
sections = json.load(open("$STEP2_OUT_JSON"))

# Default density map
density_map = {
    "intro": 0.50,
    "verse": 0.60,
    "pre_chorus": 0.75,
    "chorus": 0.90,
    "bridge": 0.80,
    "outro": 0.50,
}

for sec in sections.get("sections", []):
    label = sec.get("label", "")
    bar_start = sec.get("bar_start", 0)
    bar_end = sec.get("bar_end", len(bars))
    density = density_map.get(label, 0.70)
    
    mask = (bars.index >= bar_start) & (bars.index < bar_end)
    bars.loc[mask, "density_target"] = density
    bars.loc[mask, "swing_target"] = 0.0
    bars.loc[mask, "section_label"] = label

bars.to_parquet("$STEP1_OUT_BARS")
print(f"✅ Updated density_target/swing_target/section_label for {len(sections.get('sections', []))} sections")
for sec in sections.get("sections", []):
    label = sec.get("label", "")
    bar_start = sec.get("bar_start", 0)
    bar_end = sec.get("bar_end", len(bars))
    density = density_map.get(label, 0.70)
    print(f"   {label:12s} bars {bar_start:3d}-{bar_end:3d} : density={density:.2f}")
EOF

# ==========================================
# STEP 3: lyric_anchors.json
# ==========================================
echo ""
echo "🕒 STEP 3/6: lyric_anchors.json"

if [[ -n "$VOCAL_WAV" && -f "$VOCAL_WAV" ]]; then
  CMD3=("$PYTHON_BIN" "$REPO_ROOT/ops/anchors_from_vocal.py" \
        --vocal "$VOCAL_WAV" \
        --bars "$STEP1_OUT_BARS" \
        --tempo-map "$STEP1_OUT_JSON" \
        --out "$STEP3_OUT_JSON")
  
  if [[ $DRY_RUN -eq 1 ]]; then
    echo "[DRY-RUN] ${CMD3[*]}"
  else
    echo "   Running: ${CMD3[*]}"
    "${CMD3[@]}" || {
      echo "❌ STEP 3 failed"
      [[ $STRICT -eq 1 ]] && exit 1
    }
  fi
else
  echo "   ⚠️  Vocal WAV not found, creating empty lyric_anchors.json"
  echo '[]' > "$STEP3_OUT_JSON"
fi

# ==========================================
# STEP 4: analysis/chordmap.json (AUTO - REFERENCE ONLY)
# ==========================================
echo ""
echo "🕓 STEP 4/6: analysis/chordmap.json (AUTO - REFERENCE ONLY)"
echo "   ⚠️  This is an AUTO-GENERATED reference chordmap"
echo "   ⚠️  Create manual_chordmap.json before Phase B for production use"

# Choose mix audio deterministically for chord recognition
MIX_AUDIO_CANDIDATES=(
  "$STEMS_DIR/instrument.wav"
  "$STEMS_DIR/mix.wav"
  "$STEMS_DIR/Mix.wav"
  "$STEMS_DIR/other.wav"
  "$STEMS_DIR/Other.wav"
  "$STEMS_DIR/stem_wav_001_(Instrumental).wav"
  "$STEMS_DIR/stem_wav_001_(Instrument).wav"
  "$STEMS_DIR/stem_wav_001_(Backing).wav"
  "$STEMS_DIR/stem_wav_001_(Backing Vocals).wav"
  "$STEMS_DIR/stem_wav_001_(Keyboard).wav"
)
CHOSEN_MIX_AUDIO=""
for f in "${MIX_AUDIO_CANDIDATES[@]}"; do
  if [[ -f "$f" ]]; then
    CHOSEN_MIX_AUDIO="$f"
    break
  fi
done
if [[ -z "$CHOSEN_MIX_AUDIO" ]]; then
  echo "❌ STEP 4: mix audio not found in $STEMS_DIR (instrument/mix/other)."
  [[ $STRICT -eq 1 ]] && exit 1
fi
echo "   🎧 Chord recognition audio: ${CHOSEN_MIX_AUDIO}"

CMD4=("$PYTHON_BIN" "$REPO_ROOT/ops/stem_harmony_bar_level.py" \
      --stems "$STEMS_DIR" \
      --audio "$CHOSEN_MIX_AUDIO" \
      --bars "$STEP1_OUT_BARS" \
      --out "$STEP4_OUT_JSON" \
      --use-dp \
      --smoothing 0.05)

if [[ $DRY_RUN -eq 1 ]]; then
  echo "[DRY-RUN] ${CMD4[*]}"
else
  echo "   Running: ${CMD4[*]}"
  "${CMD4[@]}" || {
    echo "❌ STEP 4 failed"
    [[ $STRICT -eq 1 ]] && exit 1
  }
fi

# Validate chord diversity (warn if G# bias detected)
echo "   🔍 Validating chord diversity..."
"$PYTHON_BIN" << EOF
import json
from collections import Counter

chordmap = json.load(open("$STEP4_OUT_JSON"))
events = chordmap.get("events", [])
symbols = [e.get("symbol", "") for e in events]
counter = Counter(symbols)

if symbols:
    most_common = counter.most_common(1)[0]
    ratio = most_common[1] / len(symbols)
    print(f"   Most common chord: {most_common[0]} ({most_common[1]}/{len(symbols)} = {ratio:.1%})")
    
    if ratio > 0.50:
        print(f"   ⚠️  WARNING: {most_common[0]} dominates {ratio:.1%} of all bars!")
        print(f"   ⚠️  AUTO chordmap may be inaccurate - create manual_chordmap.json before Phase B")
    else:
        print(f"   ✅ Chord diversity OK (top chord {ratio:.1%} < 50%)")
else:
    print("   ⚠️  No chord events found")
EOF

# ==========================================
# STEP 5: stems_features.parquet (optional)
# ==========================================
if [[ "$STEM_FEATURES" == "1" ]]; then
  echo ""
  echo "🕔 STEP 5/6: stems_features.parquet (STEM_FEATURES=1)"
  
  STEP5_OUT_FEATURES="$ANALYSIS_DIR/stems_features.parquet"
  ANCHORS_ARG=()
  if [[ -f "$STEP3_OUT_JSON" ]]; then
    ANCHORS_ARG=("--anchors" "$STEP3_OUT_JSON")
  fi

  CMD5=("$PYTHON_BIN" "$REPO_ROOT/ops/stems_features.py" \
        --stems "$STEMS_DIR" \
        --bars "$STEP1_OUT_BARS" \
        "${ANCHORS_ARG[@]}" \
        --output "$STEP5_OUT_FEATURES" \
        --tempo-bpm "$MEDIAN_BPM" \
        --inst-activity)

  if [[ $DRY_RUN -eq 1 ]]; then
    echo "[DRY-RUN] ${CMD5[*]}"
  else
    echo "   Running: ${CMD5[*]}"
    "${CMD5[@]}" || {
      echo "❌ STEP 5 failed"
      [[ $STRICT -eq 1 ]] && exit 1
    }
  fi
fi

# ==========================================
# STEP 6: bars_with_slots.parquet (fill/riff slots)
# ==========================================
echo ""
echo "🎯 STEP 6/6: bars_with_slots.parquet (fill/riff slots)"

if [[ -f "$REPO_ROOT/scripts/add_fill_riff_slots.py" ]]; then
    "$PYTHON_BIN" "$REPO_ROOT/scripts/add_fill_riff_slots.py" \
        --bars "$STEP1_OUT_BARS" \
        --sections "$STEP2_OUT_JSON" \
        --out "$BARS_WITH_SLOTS" \
        --energy-jump-thresh 0.06 \
        --fill-likelihood-thresh 0.15 \
        --boundary-fill always \
        --riff-sections pre_chorus chorus bridge \
        --min-riff-activity 0.2 || {
        echo "⚠️  Fill/riff slot addition failed"
        [[ $STRICT -eq 1 ]] && exit 1
    }
    
    if [[ -f "$BARS_WITH_SLOTS" ]]; then
        cp -f "$STEP1_OUT_BARS" "$ANALYSIS_DIR/bars_original.parquet" 2>/dev/null || true
        cp -f "$BARS_WITH_SLOTS" "$STEP1_OUT_BARS" 2>/dev/null || true
        echo "   ✅ bars.parquet updated with fill_slot/riff_slot"
        echo "   📁 Original backed up to bars_original.parquet"
    fi
else
    echo "⚠️  scripts/add_fill_riff_slots.py not found, skipping"
fi

# ==========================================
# Optional STEP 7: Local LAMDA bundle build
# ==========================================
if [[ $LOCAL_LAMDA_ENABLE -eq 1 ]]; then
  echo "🧱 STEP 7: Local LAMDA bundle"

  if [[ ! -f "$LOCAL_LAMDA_SCRIPT" ]]; then
    echo "⚠️  Local LAMDA builder not found at $LOCAL_LAMDA_SCRIPT"
    [[ $STRICT -eq 1 ]] && exit 1 || LOCAL_LAMDA_ENABLE=0
  fi

  if [[ $LOCAL_LAMDA_MODE == "midi" ]]; then
    if [[ -z "$LOCAL_LAMDA_MIDI_ROOT" || ! -d "$LOCAL_LAMDA_MIDI_ROOT" ]]; then
      echo "⚠️  Local LAMDA MIDI root missing: $LOCAL_LAMDA_MIDI_ROOT"
      [[ $STRICT -eq 1 ]] && exit 1 || LOCAL_LAMDA_ENABLE=0
    fi
    LAMDA_INPUT_ROOT="$LOCAL_LAMDA_MIDI_ROOT"
  else
    LAMDA_INPUT_ROOT="$LOCAL_LAMDA_MIDI_ROOT"
    if [[ -z "$LAMDA_INPUT_ROOT" ]]; then
      LAMDA_INPUT_ROOT="$STEMS_DIR"
    fi
    if [[ -z "$LAMDA_INPUT_ROOT" || ! -d "$LAMDA_INPUT_ROOT" ]]; then
      echo "⚠️  Local LAMDA WAV root missing (set --local-lamda-midi-root or ensure stems dir exists)."
      [[ $STRICT -eq 1 ]] && exit 1 || LOCAL_LAMDA_ENABLE=0
    fi
  fi

  if [[ $LOCAL_LAMDA_ENABLE -eq 1 ]]; then
    mkdir -p "$LOCAL_LAMDA_OUT"
    LAMDA_CMD=("bash" "$LOCAL_LAMDA_SCRIPT" "$LAMDA_INPUT_ROOT" "$LOCAL_LAMDA_OUT")
    if [[ $LOCAL_LAMDA_MODE == "wav" ]]; then
      LAMDA_CMD+=("--from-wav" "--wav-stem" "$LOCAL_LAMDA_WAV_STEM")
    fi

    if [[ $DRY_RUN -eq 1 ]]; then
      echo "[DRY-RUN] ${LAMDA_CMD[*]}"
    else
      echo "   Running: ${LAMDA_CMD[*]}"
      if ! "${LAMDA_CMD[@]}"; then
        echo "❌ Local LAMDA build failed"
        [[ $STRICT -eq 1 ]] && exit 1
      else
        cat > "$ANALYSIS_DIR/local_lamda_bundle.json" <<EOF
{
  "mode": "$LOCAL_LAMDA_MODE",
  "input_root": "${LAMDA_INPUT_ROOT}",
  "output_dir": "${LOCAL_LAMDA_OUT}",
  "generated_at": "$(date -u "+%Y-%m-%dT%H:%M:%SZ")",
  "builder": "${LOCAL_LAMDA_SCRIPT##*/}"
}
EOF
        echo "   ✅ Local LAMDA bundle available at $LOCAL_LAMDA_OUT"
        echo "   ℹ️  Metadata recorded in analysis/local_lamda_bundle.json"
      fi
    fi
  fi
fi

echo ""
echo "========================================="
echo "✅ Phase A Complete!"
echo "========================================="
echo ""
echo "📂 Generated files:"
echo "   - tempo_map.json"
echo "   - bars.parquet (with slots)"
echo "   - sections.json"
echo "   - lyric_anchors.json"
echo "   - chordmap.json (AUTO - REFERENCE ONLY)"
if [[ $LOCAL_LAMDA_ENABLE -eq 1 ]]; then
  echo "   - local_lamda_bundle.json (metadata)"
  echo "   - LOCAL_LAMDA pickles under $LOCAL_LAMDA_OUT"
fi
echo ""
echo "⚠️  NEXT STEP: Create manual_chordmap.json"
echo "   1. Review auto chordmap.json (reference only)"
echo "   2. Create manual_chordmap.json with accurate chords"
echo "   3. Run Phase B: make_song_package_phase_b.sh"
echo ""
