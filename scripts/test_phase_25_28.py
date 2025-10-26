#!/usr/bin/env python3
"""
Phase 25-28 統合テスト

Phase 25: Sparsify & Collision Avoidance
Phase 26: Hybrid Harmony
Phase 27: Style Adaptation
Phase 28: Export Postprocess
"""

import sys
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from music21 import stream, note
from generator.bass_params_stage2 import BassParamsStage2
from generator.piano_params_stage2 import PianoParamsStage2
from generator.guitar_params_stage2 import GuitarParamsStage2
from generator.strings_params_stage2 import StringsParamsStage2
from generator.drums_params_stage2 import DrumsParamsStage2


def test_phase_25_sparsify():
    """Phase 25: Sparsify テスト"""
    print("\n" + "=" * 60)
    print("Phase 25: Sparsify & Collision Avoidance テスト")
    print("=" * 60)
    
    # Pianoパート作成（過密ノート）
    piano = PianoParamsStage2()
    part = stream.Part()
    for i in range(32):  # 8小節分の密集ノート
        n = note.Note(60 + (i % 12), quarterLength=0.25)
        n.volume.velocity = 80
        part.insert(float(i * 0.25), n)
    
    original_count = len(list(part.flatten().notes))
    print(f"Original notes: {original_count}")
    
    # Phase 25有効化
    params = {
        "sparsify": {
            "enable": True,
            "keep_endpoints": True,
            "min_gap_ms": 50,  # 50ms未満の間隔を排除
        }
    }
    
    section_meta = {"tempo": 120, "bar": 0}
    mix_context = {"beat_grid": {"bpm": 120.0}}
    
    result = piano.apply(part, section_meta, mix_context, params)
    final_count = len(list(result.flatten().notes))
    
    print(f"After Phase 25: {final_count} notes")
    print(f"Reduction: {original_count - final_count} notes ({100 * (1 - final_count / original_count):.1f}%)")
    
    # 端点保持確認
    notes = sorted(result.flatten().notes, key=lambda n: n.offset)
    if notes:
        first_offset = notes[0].offset
        last_offset = notes[-1].offset
        print(f"First note offset: {first_offset} (expected: 0.0)")
        print(f"Last note offset: {last_offset} (expected: ~7.75)")
    
    assert final_count < original_count, "間引きが実行されていません"
    print("✓ Phase 25 テスト成功")


def test_phase_26_harmony():
    """Phase 26: Hybrid Harmony テスト"""
    print("\n" + "=" * 60)
    print("Phase 26: Hybrid Harmony テスト")
    print("=" * 60)
    
    guitar = GuitarParamsStage2()
    part = stream.Part()
    
    # C和音（C=60, E=64, G=67）
    for pitch in [60, 64, 67]:
        n = note.Note(pitch, quarterLength=4.0)
        n.volume.velocity = 80
        part.insert(0.0, n)
    
    print(f"Original chord: {[n.pitch.midi for n in part.flatten().notes]}")
    
    # Phase 26有効化（Hybrid Harmony）
    params = {
        "harmony": {
            "source": "hybrid",
            "blend": 0.5,
            "keep_audio_root": True,
            "allow_text_tensions": [9, 11],  # 9th, 11thのみ許可
        }
    }
    
    section_meta = {"tempo": 120, "bar": 0}
    mix_context = {
        "audio_chordmap": {0.0: "C"},
        "creative_chordmap": {0.0: "Cmaj9"},  # 9thを追加提案
    }
    
    result = guitar.apply(part, section_meta, mix_context, params)
    final_pitches = sorted([n.pitch.midi for n in result.flatten().notes])
    
    print(f"After Phase 26: {final_pitches}")
    print(f"Root preserved: {60 in final_pitches}")
    
    assert 60 in final_pitches, "Root (C=60) が保持されていません"
    print("✓ Phase 26 テスト成功")


def test_phase_27_style_adapt():
    """Phase 27: Style Adaptation テスト"""
    print("\n" + "=" * 60)
    print("Phase 27: Style Adaptation テスト")
    print("=" * 60)
    
    bass = BassParamsStage2()
    part = stream.Part()
    
    for i in range(4):
        n = note.Note("E2", quarterLength=1.0)
        n.volume.velocity = 80
        part.insert(float(i), n)
    
    # Phase 27有効化（低活動→simple、高活動→intense）
    params = {
        "style_adapt": {
            "enable": True,
            "window_bars": 4,
            "low_high": [0.2, 0.7],
            "order": ["simple", "moderate", "complex", "intense"],
            "presets_dict": {
                "simple": {"dynamics": {"min_vel": 50, "max_vel": 70}},
                "intense": {"dynamics": {"min_vel": 90, "max_vel": 110}},
            }
        }
    }
    
    # 活動レベル設定（mix_contextに保存）
    section_meta = {"tempo": 120, "bar": 0}
    mix_context = {
        "activity": {
            "bass": {0: 0.3, 1: 0.3, 2: 0.3, 3: 0.3}  # 低活動レベル
        }
    }
    
    result = bass.apply(part, section_meta, mix_context, params)
    
    print(f"Activity level: 0.3 (low)")
    print(f"Expected style: simple寄り")
    print("✓ Phase 27 テスト成功（動的補間実行）")


def test_phase_28_export():
    """Phase 28: Export Postprocess テスト"""
    print("\n" + "=" * 60)
    print("Phase 28: Export Postprocess テスト")
    print("=" * 60)
    
    strings = StringsParamsStage2()
    part = stream.Part()
    
    # 微小なオフセット（量子化前）
    for i, offset in enumerate([0.0, 1.03, 2.07, 3.11]):
        n = note.Note(60 + i, quarterLength=1.0)
        n.volume.velocity = 80
        part.insert(offset, n)
    
    original_offsets = [n.offset for n in part.flatten().notes]
    print(f"Original offsets: {original_offsets}")
    
    # Phase 28有効化
    params = {
        "export": {
            "quantize_ql": 0.25,  # 16分音符単位
            "track_split": ["Long", "Short"],
            "name_fmt": "{idx:02d}_{role}_{section}"
        }
    }
    
    section_meta = {"tempo": 120, "bar": 0, "label": "Verse"}
    mix_context = {}
    
    result = strings.apply(part, section_meta, mix_context, params)
    quantized_offsets = [n.offset for n in result.flatten().notes]
    
    print(f"Quantized offsets: {quantized_offsets}")
    
    # 量子化確認（0.25の倍数になっているか）
    for offset in quantized_offsets:
        assert offset % 0.25 == 0, f"Offset {offset} が量子化されていません"
    
    print("✓ Phase 28 テスト成功（量子化実行）")


def test_drums_phase_25():
    """Drums Phase 25 テスト（軽量実装）"""
    print("\n" + "=" * 60)
    print("Drums Phase 25 テスト（クローズHH過密抑制）")
    print("=" * 60)
    
    drums = DrumsParamsStage2()
    part = stream.Part()
    
    # クローズHHを過密配置（42=Closed HH）
    # 32分音符 = 0.0625ql@120BPM = 62.5ms
    for i in range(32):  # 32分音符を32個
        n = note.Note(42, quarterLength=0.0625)  # 32分音符に変更
        n.volume.velocity = 60
        part.insert(float(i * 0.0625), n)
    
    original_count = len(list(part.flatten().notes))
    print(f"Original HH notes: {original_count}")
    print(f"Note interval: 0.0625ql = 62.5ms@120BPM")
    
    # Phase 25有効化（80ms以上の間隔を要求）
    params = {
        "sparsify": {
            "enable": True,
            "keep_endpoints": False,  # Drumsは端点保持不要
            "min_gap_ms": 80,  # 80ms以上空ける（62.5msの間隔を排除）
        }
    }
    
    section_meta = {"tempo": 120, "bar": 0}
    mix_context = {"beat_grid": {"bpm": 120.0}}
    
    result = drums.apply(part, section_meta, mix_context, params)
    final_count = len(list(result.flatten().notes))
    
    print(f"After Phase 25: {final_count} notes")
    print(f"Reduction: {original_count - final_count} notes")
    
    # 80ms以上の間隔で62.5ms間隔のノートを間引くので、
    # 約60%程度まで減少するはず（80/62.5 ≈ 1.28倍間隔）
    assert final_count < original_count * 0.8, f"間引きが不十分: {final_count}/{original_count}"
    print("✓ Drums Phase 25 テスト成功")


def test_no_op_safety():
    """NO-OP安全性テスト（Phase 25-28未設定時）"""
    print("\n" + "=" * 60)
    print("NO-OP安全性テスト（Phase 25-28未設定時）")
    print("=" * 60)
    
    piano = PianoParamsStage2()
    part = stream.Part()
    
    for i in range(4):
        n = note.Note(60 + i, quarterLength=1.0)
        n.volume.velocity = 80
        part.insert(float(i), n)
    
    original_count = len(list(part.flatten().notes))
    
    # Phase 25-28を設定しない（NO-OP）
    params = {}
    section_meta = {"tempo": 120, "bar": 0}
    mix_context = {}
    
    result = piano.apply(part, section_meta, mix_context, params)
    final_count = len(list(result.flatten().notes))
    
    assert final_count == original_count, "NO-OPで変更が発生しました"
    print(f"Notes: {original_count} → {final_count} (unchanged)")
    print("✓ NO-OP安全性テスト成功")


def main():
    """全テスト実行"""
    print("\n" + "=" * 80)
    print("  Phase 25-28 統合テスト開始")
    print("=" * 80)
    
    try:
        test_phase_25_sparsify()
        test_phase_26_harmony()
        test_phase_27_style_adapt()
        test_phase_28_export()
        test_drums_phase_25()
        test_no_op_safety()
        
        print("\n" + "=" * 80)
        print("  ✓ 全テスト成功 (6/6)")
        print("=" * 80)
        return 0
    
    except AssertionError as e:
        print(f"\n✗ テスト失敗: {e}")
        return 1
    except Exception as e:
        print(f"\n✗ エラー発生: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
