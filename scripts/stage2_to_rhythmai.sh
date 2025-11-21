#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
cd "$REPO_ROOT"

# Convenience pipeline for automating Stage2 → RhythmAI drum plan generation.
# Rebuilds the groove vocabulary parquet from a Stage2 loop summary and then
# renders both the RhythmAI-enabled and baseline drums plans so CI can diff them.

show_usage() {
  cat <<'USAGE'
Usage: scripts/stage2_to_rhythmai.sh --stage2-dir DIR --bars PATH --sections PATH \
        --policy PATH --rhythmai-out PATH [options]

Required flags:
  --stage2-dir PATH     Stage2 output directory containing loop_summary.csv
  --bars PATH           bars_with_slots.parquet for the song
  --sections PATH       sections.json for the song
  --policy PATH         Policy YAML providing drums config
  --rhythmai-out PATH   Output path for the RhythmAI-enabled drums plan

Optional flags:
  --groove-vocab PATH   Destination parquet for RhythmAI vocab (default: data/groove_vocab.parquet)
  --groove-stats PATH   Destination stats JSON (default: data/groove_vocab_stats.json)
  --baseline-out PATH   Output path for deterministic baseline plan (default: <song-root>/plans/drums_plan_v2_no_ai.json)
  --tempo-bpm NUM       Fallback tempo when bars data lacks tempo (default: 120)
  --python PATH         Python executable to use (default: .venv311/bin/python)
  --skip-baseline       Only emit the RhythmAI plan (useful for quick smoke tests)
  --song-root PATH      Song root directory; used to infer default baseline path

Examples:
  scripts/stage2_to_rhythmai.sh \
    --stage2-dir outputs/stage2_drums_iter8_100PCT \
    --bars data/suno_ai/suno_themesong/song_004/analysis/bars_with_slots.parquet \
    --sections data/suno_ai/suno_themesong/song_004/analysis/sections.json \
    --policy data/suno_ai/suno_themesong/song_004/policy/song_004.yaml \
    --song-root data/suno_ai/suno_themesong/song_004 \
    --rhythmai-out data/suno_ai/suno_themesong/song_004/plans/drums_plan_v2_rhythmai.json
USAGE
}

STAGE2_DIR=""
BARS_PATH=""
SECTIONS_PATH=""
POLICY_PATH=""
SONG_ROOT=""
RHYTHMAI_OUT=""
BASELINE_OUT=""
GROOVE_VOCAB="data/groove_vocab.parquet"
GROOVE_STATS="data/groove_vocab_stats.json"
TEMPO_BPM="120"
PYTHON_BIN=".venv311/bin/python"
SKIP_BASELINE=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --stage2-dir)
      STAGE2_DIR=$2; shift 2;;
    --bars)
      BARS_PATH=$2; shift 2;;
    --sections)
      SECTIONS_PATH=$2; shift 2;;
    --policy)
      POLICY_PATH=$2; shift 2;;
    --song-root)
      SONG_ROOT=$2; shift 2;;
    --rhythmai-out)
      RHYTHMAI_OUT=$2; shift 2;;
    --baseline-out)
      BASELINE_OUT=$2; shift 2;;
    --groove-vocab)
      GROOVE_VOCAB=$2; shift 2;;
    --groove-stats)
      GROOVE_STATS=$2; shift 2;;
    --tempo-bpm)
      TEMPO_BPM=$2; shift 2;;
    --python)
      PYTHON_BIN=$2; shift 2;;
    --skip-baseline)
      SKIP_BASELINE=true; shift 1;;
    -h|--help)
      show_usage; exit 0;;
    *)
      echo "Unknown argument: $1" >&2
      show_usage
      exit 1;;
  esac
done

if [[ -z "$STAGE2_DIR" || -z "$BARS_PATH" || -z "$SECTIONS_PATH" || -z "$POLICY_PATH" || -z "$RHYTHMAI_OUT" ]]; then
  echo "Missing required arguments" >&2
  show_usage
  exit 1
fi

if [[ -z "$BASELINE_OUT" ]]; then
  if [[ -n "$SONG_ROOT" ]]; then
    BASELINE_OUT="$SONG_ROOT/plans/drums_plan_v2_no_ai.json"
  else
    BASELINE_OUT="$(dirname "$RHYTHMAI_OUT")/drums_plan_v2_no_ai.json"
  fi
fi

PYTHON_BIN=$(cd "$(dirname "$PYTHON_BIN")" >/dev/null 2>&1 && pwd)/"$(basename "$PYTHON_BIN")"
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python executable not found: $PYTHON_BIN" >&2
  exit 1
fi

set -x

$PYTHON_BIN scripts/extract_groove_vocab.py \
  --stage2-dir "$STAGE2_DIR" \
  --output-parquet "$GROOVE_VOCAB" \
  --output-stats "$GROOVE_STATS"

COMMON_ARGS=(
  --bars "$BARS_PATH"
  --sections "$SECTIONS_PATH"
  --policy "$POLICY_PATH"
  --groove-vocab "$GROOVE_VOCAB"
  --tempo-bpm "$TEMPO_BPM"
)

mkdir -p "$(dirname "$RHYTHMAI_OUT")"
$PYTHON_BIN scripts/generate_drums_plan_v2.py "${COMMON_ARGS[@]}" --out "$RHYTHMAI_OUT"

if [[ "$SKIP_BASELINE" != true ]]; then
  mkdir -p "$(dirname "$BASELINE_OUT")"
  $PYTHON_BIN scripts/generate_drums_plan_v2.py "${COMMON_ARGS[@]}" --disable-rhythmai --out "$BASELINE_OUT"
fi

set +x

echo "✅ RhythmAI workflow complete"
echo "   Groove vocab : $GROOVE_VOCAB"
echo "   RhythmAI plan: $RHYTHMAI_OUT"
if [[ "$SKIP_BASELINE" != true ]]; then
  echo "   Baseline plan: $BASELINE_OUT"
fi
