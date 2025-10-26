#!/usr/bin/env python3
"""Drums Stage2 最小テスト（generatorパッケージ回避）"""

import sys
from pathlib import Path

# 直接インポート（generator/__init__.pyを回避）
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 個別モジュール読み込み
import importlib.util

def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# instrument_stage2_baseを読み込み
base_path = project_root / "generator/instrument_stage2_base.py"
base_module = load_module("instrument_stage2_base", base_path)

# drums_params_stage2を読み込み
drums_path = project_root / "generator/drums_params_stage2.py"
drums_module = load_module("drums_params_stage2", drums_path)

# music21インポート
from music21 import stream, note, instrument as m21instrument

print("✅ Modules loaded successfully!")

# プリセット読み込み
preset_path = project_root / "data/presets/drums_style_presets.yaml"
drums_stage2 = drums_module.load_drums_presets(style_yaml=preset_path)

print("✅ Drums Stage2 initialized!")

# モックパート作成
mock_part = stream.Part()
mock_part.insert(0, m21instrument.Percussion())

GM_DRUM_MAP = {'kick': [36], 'snare': [38], 'hihat_closed': [42]}

for bar in range(4):
    offset_base = bar * 4.0
    for beat in [0.0, 2.0]:
        kick = note.Note(GM_DRUM_MAP['kick'][0], quarterLength=0.25)
        kick.volume.velocity = 100
        mock_part.insert(offset_base + beat, kick)
    for beat in [1.0, 3.0]:
        snare = note.Note(GM_DRUM_MAP['snare'][0], quarterLength=0.25)
        snare.volume.velocity = 95
        mock_part.insert(offset_base + beat, snare)
    for eighth in [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]:
        hihat = note.Note(GM_DRUM_MAP['hihat_closed'][0], quarterLength=0.25)
        hihat.volume.velocity = 70
        mock_part.insert(offset_base + eighth, hihat)

original_count = len(list(mock_part.flatten().notes))
print(f"\n📊 Original: {original_count} hits")

# Stage2適用テスト
for style in ["simple", "moderate", "complex", "intense"]:
    test_part = stream.Part()
    test_part.insert(0, m21instrument.Percussion())
    
    # ノートをコピー
    for n in mock_part.flatten().notes:
        test_part.insert(n.offset, n)
    
    section_meta = {
        "label": "Test",
        "bar": 0,
        "emotion": "energetic",
        "drums_style": style
    }
    mix_context = {}
    
    result = drums_stage2.apply(
        part=test_part,
        section_meta=section_meta,
        mix_context=mix_context,
        overrides={},
        seed=42
    )
    
    final_count = len(list(result.flatten().notes))
    print(f"✅ {style:12s}: {final_count} hits")

print("\n🎉 All drums styles tested successfully!")
