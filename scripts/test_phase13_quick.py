#!/usr/bin/env python3
"""
Drums Stage2 Quick Phase Test - Phase 13確認
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import importlib.util

def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

base_path = project_root / "generator/instrument_stage2_base.py"
base_module = load_module("instrument_stage2_base", base_path)

drums_path = project_root / "generator/drums_params_stage2.py"
drums_module = load_module("drums_params_stage2", drums_path)

from music21 import stream, note, instrument as m21instrument
import logging

# デバッグログ有効化
logging.basicConfig(level=logging.DEBUG, format='%(levelname)s: %(message)s')

print("Testing Phase 13 (Vocabulary expansion)...")

preset_path = project_root / "data/presets/drums_style_presets.yaml"
drums_stage2 = drums_module.load_drums_presets(style_yaml=preset_path)

GM_DRUM_MAP = {'kick': [36], 'snare': [38], 'hihat_closed': [42]}

# モックパート作成（8小節）
mock_part = stream.Part()
mock_part.insert(0, m21instrument.Percussion())

for bar in range(8):
    offset_base = bar * 4.0
    for beat in [0.0, 2.0]:
        kick = note.Note(GM_DRUM_MAP['kick'][0], quarterLength=0.25)
        kick.volume.velocity = 100
        mock_part.insert(offset_base + beat, kick)

original_count = len(list(mock_part.flatten().notes))
print(f"Original: {original_count} hits")

# Phase 13を確実に発動させるため、fill_probability=1.0に上書き
section_meta = {
    "label": "Verse",
    "bar": 0,
    "emotion": "energetic",
    "drums_style": "simple"
}

mix_context = {
    "chord_changes": [{"offset": 0.0, "chord": "C"}],
    "bass_onsets_ql": [0.0, 2.0, 4.0, 6.0],
}

# fill_probabilityを1.0にオーバーライド
overrides = {
    "vocabulary": {
        "insert_fills": True,
        "fill_probability": 1.0
    }
}

result_part = drums_stage2.apply(
    part=mock_part,
    section_meta=section_meta,
    mix_context=mix_context,
    overrides=overrides,
    seed=42
)

final_count = len(list(result_part.flatten().notes))
change = final_count - original_count

print(f"Final: {final_count} hits (change: {change:+d})")

if change > 0:
    print("✅ Phase 13 (Vocabulary) is working! Fills were added.")
else:
    print("⚠️ Phase 13 did not add fills. Check implementation.")

# MIDI出力
output_dir = project_root / "data/drums_advanced_test"
output_dir.mkdir(parents=True, exist_ok=True)
midi_path = output_dir / "phase13_test.mid"
result_part.write('midi', fp=midi_path)
print(f"MIDI saved: {midi_path}")
