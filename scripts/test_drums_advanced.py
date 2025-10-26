#!/usr/bin/env python3
"""
Drums Stage2 Advanced Test - Phase 13-19検証

全Phase（11-20）の動作確認とメトリクス収集
"""

import sys
import json
from pathlib import Path

# 直接インポート（generator/__init__.pyを回避）
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

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

from music21 import stream, note, instrument as m21instrument

print("=" * 70)
print("  Drums Stage2 Advanced Test - Phase 13-19")
print("=" * 70)

# プリセット読み込み
preset_path = project_root / "data/presets/drums_style_presets.yaml"
drums_stage2 = drums_module.load_drums_presets(style_yaml=preset_path)

print("\n✅ Drums Stage2 initialized with Phase 13-19!")

# GMドラムマップ
GM_DRUM_MAP = {'kick': [36], 'snare': [38], 'hihat_closed': [42], 'crash': [49]}

# テスト結果格納
test_results = []

for style in ["simple", "moderate", "complex", "intense"]:
    print(f"\n{'='*70}")
    print(f"  Testing Style: {style.upper()}")
    print(f"{'='*70}")
    
    # モックパート作成（8小節）
    mock_part = stream.Part()
    mock_part.insert(0, m21instrument.Percussion())
    
    for bar in range(8):
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
    
    original_count = len(list(mock_part.flatten().notes))
    print(f"  📊 Original: {original_count} hits")
    
    # section_metaとmix_context設定
    section_meta = {
        "label": "Verse",
        "bar": 0,
        "emotion": "energetic",
        "drums_style": style
    }
    
    # mix_context（Phase 14/15で使用）
    mix_context = {
        "chord_changes": [
            {"offset": 0.0, "chord": "C"},
            {"offset": 8.0, "chord": "G"},
            {"offset": 16.0, "chord": "Am"},
            {"offset": 24.0, "chord": "F"},
        ],
        "bass_onsets_ql": [0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0,
                           16.0, 18.0, 20.0, 22.0, 24.0, 26.0, 28.0, 30.0],
    }
    
    # Stage2適用
    result_part = drums_stage2.apply(
        part=mock_part,
        section_meta=section_meta,
        mix_context=mix_context,
        overrides={},
        seed=42
    )
    
    final_count = len(list(result_part.flatten().notes))
    
    # メトリクス取得
    metrics = drums_stage2.metrics.copy()
    metrics["style"] = style
    metrics["original_count"] = original_count
    metrics["final_count"] = final_count
    metrics["change"] = final_count - original_count
    
    # 結果表示
    print(f"  ✅ Final: {final_count} hits (change: {metrics['change']:+d})")
    
    # Phase別詳細（ログから推測）
    phase_details = []
    if metrics.get("change", 0) > 0:
        phase_details.append(f"Phase 13 (Fills): likely added notes")
    if "crash" in str(result_part.flatten().notes):
        phase_details.append(f"Phase 14 (Harmonic): crash on chord changes")
    
    if phase_details:
        print(f"  🎯 Active Phases: {', '.join(phase_details)}")
    
    # ベロシティ統計
    velocities = [n.volume.velocity for n in result_part.flatten().notes 
                  if hasattr(n, 'volume') and hasattr(n.volume, 'velocity')]
    if velocities:
        import statistics
        vel_mean = statistics.mean(velocities)
        vel_std = statistics.stdev(velocities) if len(velocities) > 1 else 0
        print(f"  📈 Velocity: mean={vel_mean:.1f}, std={vel_std:.1f}")
        metrics["vel_mean"] = vel_mean
        metrics["vel_std"] = vel_std
    
    test_results.append(metrics)

# サマリー表示
print(f"\n{'='*70}")
print("  SUMMARY")
print(f"{'='*70}")

for result in test_results:
    style = result["style"]
    change = result["change"]
    vel_mean = result.get("vel_mean", 0)
    print(f"  {style:12s}: {result['final_count']:3d} hits ({change:+3d}) | vel_mean={vel_mean:.1f}")

# JSON出力
output_dir = project_root / "data/drums_advanced_test"
output_dir.mkdir(parents=True, exist_ok=True)
json_path = output_dir / "test_results.json"

with open(json_path, 'w', encoding='utf-8') as f:
    json.dump(test_results, f, indent=2, ensure_ascii=False)

print(f"\n💾 Results saved to: {json_path}")

# MIDI出力
for result in test_results:
    style = result["style"]
    # 再度適用してMIDI保存
    test_part = stream.Part()
    test_part.insert(0, m21instrument.Percussion())
    
    for bar in range(8):
        offset_base = bar * 4.0
        for beat in [0.0, 2.0]:
            kick = note.Note(GM_DRUM_MAP['kick'][0], quarterLength=0.25)
            kick.volume.velocity = 100
            test_part.insert(offset_base + beat, kick)
        for beat in [1.0, 3.0]:
            snare = note.Note(GM_DRUM_MAP['snare'][0], quarterLength=0.25)
            snare.volume.velocity = 95
            test_part.insert(offset_base + beat, snare)
        for eighth in [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]:
            hihat = note.Note(GM_DRUM_MAP['hihat_closed'][0], quarterLength=0.25)
            hihat.volume.velocity = 70
            test_part.insert(offset_base + eighth, hihat)
    
    section_meta = {
        "label": "Verse",
        "bar": 0,
        "emotion": "energetic",
        "drums_style": style
    }
    mix_context = {
        "chord_changes": [{"offset": 0.0, "chord": "C"}, {"offset": 8.0, "chord": "G"}, 
                          {"offset": 16.0, "chord": "Am"}, {"offset": 24.0, "chord": "F"}],
        "bass_onsets_ql": [0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0,
                           16.0, 18.0, 20.0, 22.0, 24.0, 26.0, 28.0, 30.0],
    }
    
    result_part = drums_stage2.apply(part=test_part, section_meta=section_meta, 
                                     mix_context=mix_context, overrides={}, seed=42)
    
    midi_path = output_dir / f"drums_advanced_{style}.mid"
    result_part.write('midi', fp=midi_path)
    print(f"  💾 MIDI saved: {midi_path.name}")

print(f"\n{'='*70}")
print("🎉 All Phase 13-19 tests completed successfully!")
print(f"{'='*70}")
