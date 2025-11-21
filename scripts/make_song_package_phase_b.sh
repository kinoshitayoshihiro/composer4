#!/usr/bin/env bash
set -euo pipefail

# Phase B runner: executes each V2 instrument generator and forwards any
# ContinueModule overrides to the guitar renderer. Designed to be called after
# Phase A has produced analysis artifacts (bars, sections, chordmap, etc.).

usage() {
  cat <<'USAGE'
Usage: scripts/make_song_package_phase_b.sh <song_root> [Continue options]

Required:
  <song_root>      Song package root produced by Phase A (contains analysis/)

Optional (passed to generate_guitar_plan_v2.py):
  --continue-enable | --continue-disable | --continue-allow-non-slot
  --continue-sections <list>
  --continue-stage3 <path>
  --continue-loop-id <id>
  --continue-groove <path>
  --continue-manifest <path>
  --continue-motif <motif>
  --continue-source-bars <bars>
  --continue-target-bars <bars>
  --continue-seed <int>
USAGE
}

SONG_ROOT=""
declare -a GUITAR_CONTINUE_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --continue-enable|--continue-disable|--continue-allow-non-slot)
      GUITAR_CONTINUE_ARGS+=("$1")
      shift
      ;;
    --continue-sections|--continue-stage3|--continue-loop-id|--continue-groove|\
    --continue-manifest|--continue-motif|--continue-source-bars|\
    --continue-target-bars|--continue-seed)
      if [[ $# -lt 2 ]]; then
        echo "❌ Missing value for $1" >&2
        exit 1
      fi
      GUITAR_CONTINUE_ARGS+=("$1" "$2")
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --*)
      echo "❌ Unknown option: $1" >&2
      exit 1
      ;;
    *)
      if [[ -z "$SONG_ROOT" ]]; then
        SONG_ROOT="$1"
        shift
      else
        echo "❌ Unexpected positional argument: $1" >&2
        exit 1
      fi
      ;;
  esac
done

if [[ -z "$SONG_ROOT" ]]; then
  usage >&2
  exit 1
fi

SONG_ROOT="$(cd "$SONG_ROOT" && pwd)"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYBIN="${PYTHON_BIN:-python3}"

ANALYSIS_DIR="$SONG_ROOT/analysis"
PLANS_DIR="$SONG_ROOT/plans"
POLICY_DIR="$SONG_ROOT/policy"
mkdir -p "$PLANS_DIR"

POLICY_YAML="$POLICY_DIR/$(basename "$SONG_ROOT").yaml"
if [[ ! -f "$POLICY_YAML" ]]; then
  echo "❌ Missing policy file: $POLICY_YAML" >&2
  exit 1
fi

BARS_WITH_SLOTS="$ANALYSIS_DIR/bars_with_slots.parquet"
SECTIONS_JSON="$ANALYSIS_DIR/sections.json"
if [[ -f "$ANALYSIS_DIR/manual_chordmap_enriched.json" ]]; then
  MANUAL_CHORDMAP="$ANALYSIS_DIR/manual_chordmap_enriched.json"
else
  MANUAL_CHORDMAP="$ANALYSIS_DIR/manual_chordmap.json"
fi

for required in "$BARS_WITH_SLOTS" "$SECTIONS_JSON" "$MANUAL_CHORDMAP"; do
  if [[ ! -f "$required" ]]; then
    echo "❌ Missing analysis artifact: $required" >&2
    exit 1
  fi
done

get_engine() {
  local inst="$1"
  "$PYBIN" - "$POLICY_YAML" "$inst" <<'PY'
import sys, yaml
policy, instrument = sys.argv[1:3]
cfg = yaml.safe_load(open(policy, encoding="utf-8")) or {}
print((cfg.get("instruments", {}).get(instrument, {}) or {}).get("engine", "v2"))
PY
}

INSTRUMENTS=(bass guitar piano strings drums)
for inst in "${INSTRUMENTS[@]}"; do
  eng=$(get_engine "$inst")
  echo "[PhaseB] $inst engine=${eng:-v2}"

  CMD=()
  if [[ "$inst" == "drums" ]]; then
    case "$eng" in
      v2|V2|"")
        CMD=("$PYBIN" "$REPO_ROOT/scripts/generate_${inst}_plan_v2.py" \
          --bars "$BARS_WITH_SLOTS" \
          --sections "$SECTIONS_JSON" \
          --policy "$POLICY_YAML" \
          --out "$PLANS_DIR/${inst}_plan.json")
        ;;
      *)
        echo "⚠️ Unknown drums engine '$eng', falling back to v2" >&2
        CMD=("$PYBIN" "$REPO_ROOT/scripts/generate_${inst}_plan_v2.py" \
          --bars "$BARS_WITH_SLOTS" \
          --sections "$SECTIONS_JSON" \
          --policy "$POLICY_YAML" \
          --out "$PLANS_DIR/${inst}_plan.json")
        ;;
    esac
  else
    case "$eng" in
      v2|V2|hybrid)
        CMD=("$PYBIN" "$REPO_ROOT/scripts/generate_${inst}_plan_v2.py" \
          --bars "$BARS_WITH_SLOTS" \
          --chordmap "$MANUAL_CHORDMAP" \
          --sections "$SECTIONS_JSON" \
          --policy "$POLICY_YAML" \
          --out "$PLANS_DIR/${inst}_plan.json")
        ;;
      legacy)
        CMD=("$PYBIN" "$REPO_ROOT/scripts/generate_${inst}_plan.py" \
          --bars "$BARS_WITH_SLOTS" \
          --chordmap "$MANUAL_CHORDMAP" \
          --sections "$SECTIONS_JSON" \
          --out "$PLANS_DIR/${inst}_plan.json")
        ;;
      *)
        echo "⚠️ Unknown engine '$eng' for $inst, defaulting to v2" >&2
        CMD=("$PYBIN" "$REPO_ROOT/scripts/generate_${inst}_plan_v2.py" \
          --bars "$BARS_WITH_SLOTS" \
          --chordmap "$MANUAL_CHORDMAP" \
          --sections "$SECTIONS_JSON" \
          --policy "$POLICY_YAML" \
          --out "$PLANS_DIR/${inst}_plan.json")
        ;;
    esac
  fi

  if [[ "$inst" == "guitar" && ${#GUITAR_CONTINUE_ARGS[@]} -gt 0 ]]; then
    CMD+=("${GUITAR_CONTINUE_ARGS[@]}")
  fi

  echo "   Running: ${CMD[*]}"
  if ! "${CMD[@]}"; then
    echo "❌ $inst generation failed" >&2
    exit 1
  fi

  if [[ "$eng" == "hybrid" && -x "$REPO_ROOT/scripts/adapt_${inst}_to_plan.py" ]]; then
    echo "   Post-adapt: $PYBIN $REPO_ROOT/scripts/adapt_${inst}_to_plan.py"
    "$PYBIN" "$REPO_ROOT/scripts/adapt_${inst}_to_plan.py" \
      --in "$PLANS_DIR/${inst}_plan.json" \
      --out "$PLANS_DIR/${inst}_plan.json" || {
        echo "❌ ${inst} hybrid adaptation failed" >&2
        exit 1
      }
  fi

done

echo "Running quality gate..."
if ! "$PYBIN" "$REPO_ROOT/scripts/quality_gate_fill_riff.py" \
  --plans-dir "$PLANS_DIR" \
  --bars "$BARS_WITH_SLOTS" \
  --sections "$SECTIONS_JSON" \
  --policy "$POLICY_YAML"; then
  echo "❌ Quality gate failed" >&2
  exit 1
fi

echo "✅ Phase B completed"
exit 0
