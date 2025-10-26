#!/usr/bin/env python3
"""直接Phase 13をテスト"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import importlib.util

def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

base_path = Path(__file__).parent.parent / "generator/instrument_stage2_base.py"
base_module = load_module("instrument_stage2_base", base_path)

drums_path = Path(__file__).parent.parent / "generator/drums_params_stage2.py"
drums_module = load_module("drums_params_stage2", drums_path)

from music21 import stream, note, instrument as m21instrument

# モックパート
mock_part = stream.Part()
mock_part.insert(0, m21instrument.Percussion())

GM_DRUM_MAP = {'kick': [36], 'snare': [38]}

for bar in range(8):
    offset = bar * 4.0
    kick = note.Note(GM_DRUM_MAP['kick'][0], quarterLength=0.25)
    kick.volume.velocity = 100
    mock_part.insert(offset, kick)

print(f"Original: {len(list(mock_part.flatten().notes))} hits")

# DrumsParamsStage2インスタンス作成
drums = drums_module.DrumsParamsStage2({}, {})

# Phase 13を直接呼び出し
params = {
    "vocabulary": {
        "insert_fills": True,
        "fill_probability": 1.0
    }
}

section_meta = {"label": "Verse", "bar": 0}
mix_context = {}

print("\n直接 _phase_13_vocabulary() を呼び出し...")
try:
    drums._phase_13_vocabulary(mock_part, section_meta, mix_context, params, 42)
    print("✅ Phase 13実行成功")
except Exception as e:
    print(f"❌ Phase 13エラー: {e}")
    import traceback
    traceback.print_exc()

final_count = len(list(mock_part.flatten().notes))
print(f"Final: {final_count} hits (change: {final_count - 8:+d})")
