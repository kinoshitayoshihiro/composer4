#!/usr/bin/env bash
# export_midi_from_plan.sh
# -------------------------
# 補助動線: 既存full_arrangement.jsonからMIDI書き出し専用
# E2E本線と同じフラグで再現性を担保
#
# Usage:
#   bash scripts/export_midi_from_plan.sh \
#     data/.../full_arrangement.json \
#     data/.../full_arrangement.mid \
#     data/suno_ai/suno_themesong/song_001

set -euo pipefail

PLAN="${1:?Missing PLAN (full_arrangement.json)}"
OUT="${2:?Missing OUT (*.mid)}"
SONG_DIR="${3:?Missing SONG_DIR}"

# P0: 終端超過対策（必須）
FIX_OVEREND_MS="${FIX_OVEREND_MS:-20}"  # デフォルト20ms

# デバッグ
DEBUG="${DEBUG:-0}"

echo "[export_midi] Plan: $PLAN"
echo "[export_midi] Out:  $OUT"
echo "[export_midi] Dir:  $SONG_DIR"
echo "[export_midi] fix_overend_ms: $FIX_OVEREND_MS"

# midi_writer.py呼び出し（E2E本線と同じフラグ）
python3 scripts/midi_writer.py \
  --plan "$PLAN" \
  --out  "$OUT"  \
  --bars "$SONG_DIR/bars.parquet" \
  --fix-overend-ms "$FIX_OVEREND_MS" \
  --clip-to-bars \
  ${DEBUG:+--debug}

echo "✅ MIDI exported: $OUT"
