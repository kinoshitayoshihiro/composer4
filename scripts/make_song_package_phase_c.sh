#!/usr/bin/env bash
# ========================================
# Phase C: MIDI Integration
# ========================================
# Prerequisites:
#   - Phase A completed
#   - Phase B completed
#
# Output:
#   - arrangement_plan.json
#   - integrated MIDI file (variable tempo, split tracks)
#
# ========================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

STRICT=1
DRY_RUN=0
SPLIT_TRACKS=1  # Default: enable split tracks

# Parse arguments
SONG_ROOT=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-split-tracks)
      SPLIT_TRACKS=0
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    *)
      SONG_ROOT="$1"
      shift
      ;;
  esac
done

if [[ -z "$SONG_ROOT" ]]; then
  echo "Usage: $0 <song_package_dir> [--no-split-tracks] [--dry-run]"
  echo ""
  echo "Phase C: MIDI Integration"
  echo "  Prerequisites:"
  echo "    - Phase A completed"
  echo "    - Phase B completed"
  exit 1
fi

SONG_ROOT="$(cd "$SONG_ROOT" && pwd)"
ANALYSIS_DIR="$SONG_ROOT/analysis"
PLANS_DIR="$SONG_ROOT/plans"
mkdir -p "$PLANS_DIR"

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
echo "🎹 Phase C: MIDI Integration"
echo "========================================="
echo "   Song Package: $SONG_ROOT"
echo "   Python: $PYTHON_BIN"
echo "   Split tracks: $([[ $SPLIT_TRACKS -eq 1 ]] && echo "YES" || echo "NO")"
echo ""

# Check prerequisites
BARS_PARQUET="$ANALYSIS_DIR/bars.parquet"
TEMPO_MAP="$ANALYSIS_DIR/tempo_map.json"
SECTIONS_JSON="$ANALYSIS_DIR/sections.json"
MANUAL_CHORDMAP="$ANALYSIS_DIR/manual_chordmap.json"

if [[ ! -f "$BARS_PARQUET" ]]; then
  echo "❌ bars.parquet not found. Run Phase A first."
  exit 1
fi

if [[ ! -f "$TEMPO_MAP" ]]; then
  echo "❌ tempo_map.json not found. Run Phase A first."
  exit 1
fi

if [[ ! -f "$MANUAL_CHORDMAP" ]]; then
  echo "❌ manual_chordmap.json not found. Run Phase A first."
  exit 1
fi

# Check plans
REQUIRED_PLANS=("drums" "guitar" "piano" "strings" "bass")
MISSING_PLANS=()

for inst in "${REQUIRED_PLANS[@]}"; do
  if [[ ! -f "$PLANS_DIR/${inst}_plan.json" ]]; then
    MISSING_PLANS+=("$inst")
  fi
done

if [[ ${#MISSING_PLANS[@]} -gt 0 ]]; then
  echo "❌ Missing plans: ${MISSING_PLANS[*]}"
  echo "   Run Phase B first to generate instrument plans."
  exit 1
fi

echo "   ✅ All instrument plans found"

# ==========================================
# Arrangement Plan Generation
# ==========================================
echo ""
echo "🎼 Generating arrangement plan..."

ARRANGEMENT_PLAN="$PLANS_DIR/arrangement_plan.json"

CMD_ARR=("$PYTHON_BIN" "$REPO_ROOT/scripts/arrangement_orchestrator.py" \
         --out "$ARRANGEMENT_PLAN" \
         --tempo-map "$TEMPO_MAP" \
         --bass "$PLANS_DIR/bass_plan.json" \
         --guitar "$PLANS_DIR/guitar_plan.json" \
         --piano "$PLANS_DIR/piano_plan.json" \
         --strings "$PLANS_DIR/strings_plan.json" \
         --drums "$PLANS_DIR/drums_plan.json")

if [[ $DRY_RUN -eq 1 ]]; then
  echo "[DRY-RUN] ${CMD_ARR[*]}"
else
  "${CMD_ARR[@]}" || {
    echo "❌ Arrangement plan generation failed"
    [[ $STRICT -eq 1 ]] && exit 1
  }
  echo "   ✅ arrangement_plan.json"
fi

# ==========================================
# MIDI Integration (variable tempo + split tracks)
# ==========================================
echo ""
echo "🎹 Integrating MIDI (variable tempo + split tracks)..."

INTEGRATED_MIDI="$SONG_ROOT/$(basename "$SONG_ROOT")_integrated.mid"

# Build command
CMD_MIDI=("$PYTHON_BIN" "$REPO_ROOT/json2midi.py" \
          "$ARRANGEMENT_PLAN" \
          -o "$INTEGRATED_MIDI" \
          --tempo-map "$TEMPO_MAP")

# Add split-tracks if enabled
if [[ $SPLIT_TRACKS -eq 1 ]]; then
  CMD_MIDI+=("--split-tracks")
fi

if [[ $DRY_RUN -eq 1 ]]; then
  echo "[DRY-RUN] ${CMD_MIDI[*]}"
else
  "${CMD_MIDI[@]}" || {
    echo "❌ MIDI integration failed"
    [[ $STRICT -eq 1 ]] && exit 1
  }
  
  if [[ -f "$INTEGRATED_MIDI" ]]; then
    FILE_SIZE=$(du -h "$INTEGRATED_MIDI" | cut -f1)
    echo "   ✅ Integrated MIDI: $INTEGRATED_MIDI ($FILE_SIZE)"
  else
    echo "   ❌ MIDI file not created"
    [[ $STRICT -eq 1 ]] && exit 1
  fi
fi

echo ""
echo "========================================="
echo "✅ Phase C Complete!"
echo "========================================="
echo ""
echo "📂 Final outputs:"
echo "   - arrangement_plan.json"
echo "   - $(basename "$INTEGRATED_MIDI") (variable tempo, split tracks)"
echo ""
echo "🎉 Pipeline complete! All phases finished."
echo ""
