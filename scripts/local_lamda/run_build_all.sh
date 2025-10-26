#!/usr/bin/env bash
#
# LOCAL LAMDA-like 4資源を一括生成（MIDI/WAV両対応）
#
# Usage:
#   bash scripts/local_lamda/run_build_all.sh [INPUT_ROOT] [OUTPUT_DIR] [OPTIONS]
#
# Example (MIDI):
#   bash scripts/local_lamda/run_build_all.sh data/suno_ai data/LOCAL_LAMDA/MIDI_version
#
# Example (WAV):
#   bash scripts/local_lamda/run_build_all.sh data/musdb18_wavs data/LOCAL_LAMDA/wav_version/musdb18 --from-wav --wav-stem accompaniment
#

set -euo pipefail

# デフォルト値
ROOT="${1:-data/local_corpus}"
OUT="${2:-data/LOCAL_LAMDA}"
FROM_WAV=0
WAV_STEM="accompaniment"

# オプション解析
shift 2 2>/dev/null || true
while [[ $# -gt 0 ]]; do
  case "$1" in
    --from-wav)
      FROM_WAV=1
      shift
      ;;
    --wav-stem)
      WAV_STEM="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

echo "============================================"
echo "🏗️  LOCAL LAMDA-like Bundle Builder"
echo "============================================"
echo "Input Root: $ROOT"
echo "Output Dir: $OUT"
echo "Mode:       $([ $FROM_WAV -eq 1 ] && echo 'WAV' || echo 'MIDI')"
[ $FROM_WAV -eq 1 ] && echo "WAV Stem:   $WAV_STEM"
echo ""

mkdir -p "$OUT"

if [ $FROM_WAV -eq 1 ]; then
  # WAV経路（MUSDB18/MoisesDB）
  echo "🔊 Building LOCAL LAMDA from WAV (${WAV_STEM})..."
  python3 -m scripts.local_lamda.build_local_from_wav \
    --wav-root "$ROOT" \
    --out-dir "$OUT" \
    --stem "$WAV_STEM"
  
  echo ""
  echo "============================================"
  echo "✅ LOCAL LAMDA (WAV) built!"
  echo "============================================"
  echo "📁 Output directory: $OUT"
  echo "   - LOCAL_KILO_CHORDS_DATA.pickle"
  echo "   - LOCAL_META_DATA_000001.pickle (audio_proxy)"
  echo "   - LOCAL_SIGNATURES_DATA.pickle"
  echo "   - LOCAL_TOTALS.pickle"
  echo "   - LOCAL_ID_MAP.csv"
  
else
  # MIDI経路（既存）
  
  # 1) LOCAL_KILO_CHORDS_DATA.pickle
  echo "📊 Building LOCAL_KILO..."
  python3 -m scripts.local_lamda.build_local_kilo \
    --midi-root "$ROOT" \
    --out-pickle "$OUT/LOCAL_KILO_CHORDS_DATA.pickle" \
    --token-map adapters/lamda_chords_token_map.yaml || true

  # 2) LOCAL_META_DATA_000001.pickle
  echo ""
  echo "📊 Building LOCAL_META..."
  python3 -m scripts.local_lamda.build_local_meta \
    --midi-root "$ROOT" \
    --out-pickle "$OUT/LOCAL_META_DATA_000001.pickle"

  # 3) LOCAL_SIGNATURES_DATA.pickle
  echo ""
  echo "📊 Building LOCAL_SIGNATURES..."
  python3 -m scripts.local_lamda.build_local_signatures \
    --midi-root "$ROOT" \
    --out-pickle "$OUT/LOCAL_SIGNATURES_DATA.pickle" \
    --sig-map-yaml configs/lamda/signature_id_map.yaml || true

  # 4) LOCAL_TOTALS.pickle
  echo ""
  echo "📊 Building LOCAL_TOTALS..."
  python3 -m scripts.local_lamda.build_local_totals \
    --midi-root "$ROOT" \
    --out-pickle "$OUT/LOCAL_TOTALS.pickle"

  # 5) LOCAL_ID_MAP.csv
  echo ""
  echo "📊 Building LOCAL_ID_MAP.csv..."
  python3 -m scripts.local_lamda.build_id_map \
    --midi-root "$ROOT" \
    --out-csv "$OUT/LOCAL_ID_MAP.csv" \
    --id-type kilo

  echo ""
  echo "============================================"
  echo "✅ LOCAL LAMDA (MIDI) built!"
  echo "============================================"
  echo "📁 Output directory: $OUT"
  echo "   - LOCAL_KILO_CHORDS_DATA.pickle"
  echo "   - LOCAL_META_DATA_000001.pickle"
  echo "   - LOCAL_SIGNATURES_DATA.pickle"
  echo "   - LOCAL_TOTALS.pickle"
  echo "   - LOCAL_ID_MAP.csv"
  
fi

echo ""

echo "🔗 Usage:"
echo "   Add these files to Stage2 extraction with:"
echo "   --local-kilo $OUT/LOCAL_KILO_CHORDS_DATA.pickle"
echo "   --local-meta-dir $OUT"
echo "   --local-signatures $OUT/LOCAL_SIGNATURES_DATA.pickle"
echo "   --local-totals $OUT/LOCAL_TOTALS.pickle"
echo "   --prefer-local"

