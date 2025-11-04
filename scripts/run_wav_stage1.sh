#!/usr/bin/env bash
set -euo pipefail

IN="${1:-/path/to/e-gmd/audio/root}"
OUT="${2:-output/wav_stage1}"

mkdir -p "$(dirname "$OUT")"

python scripts/clean_wav_stage1.py \
  --input "$IN" \
  --out-dir "$OUT" \
  --sr 44100 \
  --peak 0.98 \
  --trim-db -50 \
  --min-dur 1.5 \
  --max-dur 120 \
  --write-audio

echo "[OK] Wrote index and cleaned files under $OUT"
