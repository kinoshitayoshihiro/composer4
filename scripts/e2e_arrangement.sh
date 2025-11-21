#!/usr/bin/env bash
# -*- coding: utf-8 -*-
# scripts/e2e_arrangement.sh
#
# E2E arrangement script with CREPE integration
#
# Usage:
#   bash scripts/e2e_arrangement.sh <song_id>
#
# Example:
#   bash scripts/e2e_arrangement.sh song_004

set -euo pipefail

# UTF-8ロケール設定
export LC_ALL=en_US.UTF-8
export LANG=en_US.UTF-8

SONG_ID="${1:-song_004}"
ROOT="data/suno_ai/suno_themesong/${SONG_ID}"
ANAL="${ROOT}/analysis"
PLANS="${ROOT}/plans"
MIDI="${ROOT}/midi"

# スクリプトディレクトリ（絶対パス）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Python実行環境
if [[ -f "$REPO_ROOT/.venv311/bin/python" ]]; then
  PYTHON_BIN="$REPO_ROOT/.venv311/bin/python"
elif [[ -f "$REPO_ROOT/.venv/bin/python" ]]; then
  PYTHON_BIN="$REPO_ROOT/.venv/bin/python"
else
  PYTHON_BIN="${PYTHON_BIN:-python3}"
fi

echo "🎵 E2E Arrangement: $SONG_ID"
echo "   Python: $PYTHON_BIN"
echo ""

# ==========================================
# 可変テンポ対応MIDI統合（旧: plan_to_midi.py固定BPM版は廃止）
# ==========================================
echo "🎹 Generating variable tempo integrated MIDI..."

TEMPO_MAP="${ANAL}/tempo_map.json"

# tempo_map.json存在確認
if [[ ! -f "$TEMPO_MAP" ]]; then
  echo "❌ Error: tempo_map.json not found: $TEMPO_MAP"
  exit 1
fi

# 統合対象planを収集
TRACK_PLANS=()
[[ -f "${PLANS}/piano_plan_hybrid.json" ]] && TRACK_PLANS+=("${PLANS}/piano_plan_hybrid.json")
[[ -f "${PLANS}/strings_countermelody_plan_vl.json" ]] && TRACK_PLANS+=("${PLANS}/strings_countermelody_plan_vl.json")
[[ -f "${PLANS}/guitar_plan_optimized_micro.json" ]] && TRACK_PLANS+=("${PLANS}/guitar_plan_optimized_micro.json")

if [[ ${#TRACK_PLANS[@]} -eq 0 ]]; then
  echo "⚠️  No CREPE plans found, skipping MIDI generation"
else
  # 各planを個別MIDI化
  for plan in "${TRACK_PLANS[@]}"; do
    plan_name=$(basename "$plan" .json)
    midi_out="${MIDI}/${plan_name}.mid"
    
    # BPM中央値取得（tempo_map.jsonから）
    MEDIAN_BPM=$("$PYTHON_BIN" -c "
import json
from pathlib import Path
import statistics
tempo_map = json.loads(Path('$TEMPO_MAP').read_text())
if isinstance(tempo_map, dict) and 'tempo_points' in tempo_map:
    tempo_points = tempo_map['tempo_points']
    tempos = [p[1] for p in tempo_points if isinstance(p, list) and len(p) >= 2]
elif isinstance(tempo_map, list):
    tempos = [p.get('bpm', 120.0) for p in tempo_map if isinstance(p, dict)]
else:
    tempos = []
print(statistics.median(tempos) if tempos else 120.0)
")
    
    echo "   Converting: $plan_name.mid (BPM=$MEDIAN_BPM)"
    "$PYTHON_BIN" "$REPO_ROOT/scripts/CREPE/plan_to_midi.py" \
      "$plan" \
      "$midi_out" \
      --bpm "$MEDIAN_BPM" || {
      echo "⚠️  MIDI conversion failed: $plan_name"
    }
  done
  
  # 統合MIDI生成
  if [[ -f "$REPO_ROOT/scripts/CREPE/merge_crepe_midis.py" ]]; then
    MERGE_ARGS=("--output" "${MIDI}/${SONG_ID}_hybrid_crepe.mid" "--bpm" "$MEDIAN_BPM")
    [[ -f "${MIDI}/piano_plan_hybrid.mid" ]] && MERGE_ARGS+=("--piano" "${MIDI}/piano_plan_hybrid.mid")
    [[ -f "${MIDI}/strings_countermelody_plan_vl.mid" ]] && MERGE_ARGS+=("--strings" "${MIDI}/strings_countermelody_plan_vl.mid")
    [[ -f "${MIDI}/guitar_plan_optimized_micro.mid" ]] && MERGE_ARGS+=("--guitar" "${MIDI}/guitar_plan_optimized_micro.mid")
    
    "$PYTHON_BIN" "$REPO_ROOT/scripts/CREPE/merge_crepe_midis.py" "${MERGE_ARGS[@]}" || {
      echo "⚠️  MIDI merge failed"
    }
    echo "✅ Integrated MIDI: ${SONG_ID}_hybrid_crepe.mid"
  fi
fi

echo ""
echo "✅ Arrangement complete!"
echo "   Output: ${MIDI}/${SONG_ID}_hybrid_crepe.mid"
