#!/usr/bin/env bash
# Batch runner for Composer2-3 E2E with KPI/CI harvesting
# - UTF-8 safe (Japanese paths), NUL-separated traversal
# - Optional GNU parallel
# - Retries, dry-run, per-song timeout
# - Appends CSV summary per song

set -Eeuo pipefail
# UTF-8 locale for JP filenames
export LANG=ja_JP.UTF-8
export LC_ALL=ja_JP.UTF-8
export LANGUAGE=ja_JP.UTF-8
# Whitespace-safe word splitting
IFS=$'\n\t'

SCRIPT_DIR=$(cd -- "$(dirname -- "$0")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)
E2E_BIN="$SCRIPT_DIR/e2e_suno_arrangement.sh"
PY=python3

# Defaults
OUTPUT_CSV="batch_summary.csv"
MAX_WORKERS=1
RETRIES=2          # total attempts = RETRIES+1
CONTINUE_ON_ERROR=false
DRY_RUN=false
TIMEOUT_SECS=0     # 0 = no timeout
eval "E2E_EXTRA=()" # bash array for pass-through
DRUMS_MODE=""      # magenta|rule|real|""
FORCE_REGENERATE=false
KPI_ENABLED=true    # always true for batch, can be disabled via --no-kpi

# Tools: timeout (macOS coreutils -> gtimeout)
TIMEOUT_BIN="timeout"
if ! command -v timeout >/dev/null 2>&1 && command -v gtimeout >/dev/null 2>&1; then
  TIMEOUT_BIN="gtimeout"
fi

# --- help ---
usage() {
  cat <<USAGE
Usage: $(basename "$0") [options] <song_dir> [<song_dir> ...]

Options:
  --output <csv>            Output CSV path (default: $OUTPUT_CSV)
  --max-workers <N>         Parallel workers (default: $MAX_WORKERS)
  --retries <N>             Retries per song on failure (default: $RETRIES)
  --continue-on-error       Continue batch when a song fails
  --dry-run                 Print what would run, don't execute
  --timeout-secs <N>        Per-song wall timeout (0=disabled)
  --drums-mode <mode>       Pass to E2E (magenta|rule|real)
  --force-regenerate-drums  Force re-generate drums artifacts in E2E
  --no-kpi                  Skip E2E's KPI phase (not recommended)
  --e2e-extra "..."         Extra arguments to pass through to E2E

Notes:
  * Japanese / spaced paths are supported (UTF-8, NUL-safe).
  * Requires bash, Python 3.8+, and (optionally) GNU parallel for speed.
USAGE
}

# --- parse args (parent mode) ---
PARENT_MODE=true
WORKER_MODE=false
WORKER_SONG=""

if [[ "${1-}" == "--worker" ]]; then
  # internal: single-song worker mode
  PARENT_MODE=false
  WORKER_MODE=true
  shift
  WORKER_SONG=${1-}
  shift || true
fi

ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help) usage; exit 0;;
    --output) OUTPUT_CSV=$2; shift 2;;
    --max-workers) MAX_WORKERS=${2:?}; shift 2;;
    --retries) RETRIES=${2:?}; shift 2;;
    --continue-on-error) CONTINUE_ON_ERROR=true; shift;;
    --dry-run) DRY_RUN=true; shift;;
    --timeout-secs) TIMEOUT_SECS=${2:?}; shift 2;;
    --drums-mode) DRUMS_MODE=${2:?}; shift 2;;
    --force-regenerate-drums) FORCE_REGENERATE=true; shift;;
    --no-kpi) KPI_ENABLED=false; shift;;
    --e2e-extra) E2E_EXTRA+=("$2"); shift 2;;
    --) shift; while [[ $# -gt 0 ]]; do ARGS+=("$1"); shift; done; break;;
    *) ARGS+=("$1"); shift;;
  esac
done

# --- header for CSV ---
CSV_HEADER="timestamp,song_dir,status,attempts,e2e_exit,kpi_pass_rate,kpi_fail_count,kpi_evaluated_bars,kpi_skipped_bars,ci_pass,ci_fail,ci_warn,total_notes,duration_secs"

# --- helpers ---
ts() { date +"%Y-%m-%d %H:%M:%S"; }

# Safe echo to CSV
csv_append() {
  local line="$1"
  local out="$OUTPUT_CSV"
  if [[ ! -f "$out" ]]; then
    printf '%s\n' "$CSV_HEADER" > "$out"
  fi
  printf '%s\n' "$line" >> "$out"
}

# Read KPI/CI fields via Python (robust against missing keys)
extract_metrics() {
  local song_dir="$1"
  local kpi_json="$song_dir/kpi_gate_postgen.json"
  local ci_json="$song_dir/ci_verify_report.json"
  local mid_path="$song_dir/full_arrangement.mid"
  "$PY" - "$kpi_json" "$ci_json" "$mid_path" <<'PY'
import json, sys, os
kpi = {"summary": {}}
ci = {"summary": {}}
notes = ""
try:
  with open(sys.argv[1], 'r', encoding='utf-8') as f:
    kpi = json.load(f)
except Exception:
  pass
try:
  with open(sys.argv[2], 'r', encoding='utf-8') as f:
    ci = json.load(f)
except Exception:
  pass
# KPI summary
ks = kpi.get("summary", {})
pass_rate = ks.get("pass_rate")
fail_count = ks.get("fail_count")
eval_bars = ks.get("evaluated_bars") or ks.get("evaluated", 0)
# derive skipped if possible
total_bars = ks.get("total_bars") or 0
skipped = total_bars - (eval_bars or 0)
# CI summary
cs = ci.get("summary", {})
ci_pass = cs.get("pass", 0)
ci_fail = cs.get("fail", 0)
ci_warn = cs.get("warn", 0)
# Optional total notes (if present elsewhere in CI)
notes = ci.get("meta", {}).get("total_notes", "")
print(f"PASS_RATE={pass_rate if pass_rate is not None else ''}")
print(f"FAIL_COUNT={fail_count if fail_count is not None else ''}")
print(f"EVAL_BARS={eval_bars if eval_bars is not None else ''}")
print(f"SKIPPED_BARS={skipped if skipped is not None else ''}")
print(f"CI_PASS={ci_pass}")
print(f"CI_FAIL={ci_fail}")
print(f"CI_WARN={ci_warn}")
print(f"TOTAL_NOTES={notes}")
PY
}

# Core single-song runner
process_one() {
  local song_dir="$1"
  local start_ts=$(date +%s)
  local started_at=$(ts)
  local attempt=0
  local e2e_rc=1
  local status="FAIL"

  # Compose E2E arguments
  local args=("$song_dir")
  if [[ "$KPI_ENABLED" == "true" ]]; then args+=("--kpi"); fi
  if [[ -n "$DRUMS_MODE" ]]; then args+=("--drums-mode" "$DRUMS_MODE"); fi
  if [[ "$FORCE_REGENERATE" == "true" ]]; then args+=("--force-regenerate-drums"); fi
  if [[ ${#E2E_EXTRA[@]} -gt 0 ]]; then args+=("${E2E_EXTRA[@]}"); fi

  if [[ "$DRY_RUN" == "true" ]]; then
    echo "DRY-RUN: would process $song_dir :: ${args[*]}"
    status="DRYRUN"
    e2e_rc=0
  else
    while (( attempt <= RETRIES )); do
      ((attempt++))
      echo "[$(ts)] ▶ E2E (attempt $attempt/$((RETRIES+1))): $song_dir"
      if [[ $TIMEOUT_SECS -gt 0 && -x $(command -v "$TIMEOUT_BIN" || true) ]]; then
        set +e
        "$TIMEOUT_BIN" "$TIMEOUT_SECS" bash "$E2E_BIN" "${args[@]}"
        e2e_rc=$?
        set -e
      else
        set +e
        bash "$E2E_BIN" "${args[@]}"
        e2e_rc=$?
        set -e
      fi
      if [[ $e2e_rc -eq 0 ]]; then status="OK"; break; fi
      echo "[$(ts)] ⚠ E2E failed (rc=$e2e_rc) on $song_dir"
      if (( attempt <= RETRIES )); then echo "   ↻ retrying..."; fi
    done
  fi

  # Harvest KPI/CI
  local metrics
  metrics=$(extract_metrics "$song_dir") || true
  # shellcheck disable=SC2046
  eval $(printf '%s\n' "$metrics")

  # Duration
  local end_ts=$(date +%s)
  local dur=$(( end_ts - start_ts ))

  # CSV line (quote fields safely)
  local csv_line
  csv_line=$(printf '"%s","%s","%s",%d,%d,%s,%s,%s,%s,%s,%s,%s,%s,%d' \
    "$started_at" "$song_dir" "$status" "$attempt" "$e2e_rc" \
    "${PASS_RATE:-}" "${FAIL_COUNT:-}" "${EVAL_BARS:-}" "${SKIPPED_BARS:-}" \
    "${CI_PASS:-}" "${CI_FAIL:-}" "${CI_WARN:-}" "${TOTAL_NOTES:-}" "$dur")
  echo "$csv_line"
}

# Worker mode: process single song and print one CSV row
if $WORKER_MODE; then
  if [[ -z "$WORKER_SONG" ]]; then echo "worker requires song_dir" >&2; exit 2; fi
  process_one "$WORKER_SONG"
  exit 0
fi

# Parent mode: build SONG_LIST safely from ARGS (NUL-safe)
SONG_LIST=()
if [[ ${#ARGS[@]} -gt 0 ]]; then
  # Preserve each path as provided
  while IFS= read -r -d '' p; do SONG_LIST+=("$p"); done < <(printf '%s\0' "${ARGS[@]}")
else
  # Auto-discover song_* folders under song_packages (NUL-safe)
  while IFS= read -r -d '' p; do SONG_LIST+=("$p"); done \
    < <(find "$REPO_ROOT/song_packages" -type d -name 'song_*' -print0)
fi

if [[ ${#SONG_LIST[@]} -eq 0 ]]; then
  echo "No song directories given/found" >&2
  exit 1
fi

# Prepare CSV header
if [[ ! -f "$OUTPUT_CSV" ]]; then
  printf '%s\n' "$CSV_HEADER" > "$OUTPUT_CSV"
fi

# Parallel or Serial execution
if (( MAX_WORKERS > 1 )) && command -v parallel >/dev/null 2>&1; then
  echo "Using GNU parallel with $MAX_WORKERS workers"
  # Export context via env vars for child self-call
  export OUTPUT_CSV MAX_WORKERS RETRIES CONTINUE_ON_ERROR DRY_RUN TIMEOUT_SECS \
         E2E_BIN PY DRUMS_MODE FORCE_REGENERATE KPI_ENABLED TIMEOUT_BIN
  # Build a temp CSV to avoid header races
  TMPCSV=$(mktemp)
  printf '%s\n' "$CSV_HEADER" > "$TMPCSV"
  printf '%s\0' "${SONG_LIST[@]}" | \
    parallel -0 -j "$MAX_WORKERS" --linebuffer --halt now,fail=1 \
      bash "$0" --worker {} >> "$TMPCSV"
  # Merge into OUTPUT_CSV
  tail -n +2 "$TMPCSV" >> "$OUTPUT_CSV"
  rm -f "$TMPCSV"
else
  echo "Running serially (parallel not available or MAX_WORKERS=1)"
  for song in "${SONG_LIST[@]}"; do
    # Each call returns one CSV row; append to file
    line=$(bash "$0" --worker "$song")
    csv_append "$line"
    # Respect CONTINUE_ON_ERROR
    status_field=$(printf '%s' "$line" | awk -F, '{gsub(/\"/, "", $3); print $3}')
    if [[ "$status_field" != "\"OK\"" && "$CONTINUE_ON_ERROR" != "true" ]]; then
      echo "Stop on first failure. Use --continue-on-error to proceed." >&2
      exit 1
    fi
  done
fi

echo "✅ Batch complete → $OUTPUT_CSV"
