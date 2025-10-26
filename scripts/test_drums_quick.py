#!/usr/bin/env python3
"""Drums Stage2 簡易テスト"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from music21 import stream, note, instrument
from generator.drums_params_stage2 import load_drums_presets

# プリセット読み込み
drums_stage2 = load_drums_presets(
    style_yaml=Path("data/presets/drums_style_presets.yaml")
)

# モックパート作成
mock_part = stream.Part()
mock_part.insert(0, instrument.Percussion())

# 4小節のシンプルなドラムパターン
GM_DRUM_MAP = {'kick': [36], 'snare': [38], 'hihat_closed': [42]}

for bar in range(4):
    offset_base = bar * 4.0
    
    # Kick: 1拍目, 3拍目
    for beat in [0.0, 2.0]:
        kick = note.Note(GM_DRUM_MAP['kick'][0], quarterLength=0.25)
        kick.volume.velocity = 100
        mock_part.insert(offset_base + beat, kick)
    
    # Snare: 2拍目, 4拍目
    for beat in [1.0, 3.0]:
        snare = note.Note(GM_DRUM_MAP['snare'][0], quarterLength=0.25)
        snare.volume.velocity = 95
        mock_part.insert(offset_base + beat, snare)
    
    # Hi-Hat: 全8分音符
    for eighth in [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]:
        hihat = note.Note(GM_DRUM_MAP['hihat_closed'][0], quarterLength=0.25)
        hihat.volume.velocity = 70
        mock_part.insert(offset_base + eighth, hihat)

print(f"Original: {len(list(mock_part.flatten().notes))} hits")

# Stage2適用
for style in ["simple", "moderate", "complex", "intense"]:
    test_part = mock_part.makeCopy()
    
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

print("\n✅ All drums styles tested successfully!")
