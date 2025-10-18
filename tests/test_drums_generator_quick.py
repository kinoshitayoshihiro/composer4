#!/usr/bin/env python3
"""
Drums Generator Stage2 - クイックテスト

5つの基本的なテストでドラムジェネレーターの動作を確認します。
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from generator.drums_generator_stage2 import DrumsGeneratorStage2
import music21


def test_1_initialization():
    """Test 1: ジェネレーター初期化"""
    print("\n" + "="*60)
    print("Test 1: Initialization")
    print("="*60)
    
    gen = DrumsGeneratorStage2()
    
    assert gen is not None
    assert gen.patterns == []
    assert gen.default_instrument is not None
    
    print("✅ Initialization successful")
    return True


def test_2_fallback_generation():
    """Test 2: フォールバックドラム生成（4小節）"""
    print("\n" + "="*60)
    print("Test 2: Fallback Generation (4 bars)")
    print("="*60)
    
    gen = DrumsGeneratorStage2()
    
    drum_part = gen.generate(
        bars=4,
        chords=["C", "G", "Am", "F"],
        tempo=120,
        emotion="energetic",
        section="Verse",
        seed=42
    )
    
    all_notes = list(drum_part.flatten().notes)
    
    print(f"   Generated notes: {len(all_notes)}")
    assert len(all_notes) > 0, "No notes generated"
    
    # 4小節 × (2 kick + 2 snare + 8 hihat) = 48音符期待
    expected_notes = 4 * (2 + 2 + 8)
    assert len(all_notes) == expected_notes, f"Expected {expected_notes} notes, got {len(all_notes)}"
    
    print("✅ Fallback generation successful")
    return True


def test_3_tempo_variation():
    """Test 3: 異なるテンポで生成"""
    print("\n" + "="*60)
    print("Test 3: Tempo Variation")
    print("="*60)
    
    gen = DrumsGeneratorStage2()
    
    tempos = [80, 120, 160]
    for tempo in tempos:
        drum_part = gen.generate(
            bars=2,
            chords=["C", "G"],
            tempo=tempo,
            emotion="energetic",
            seed=42
        )
        
        all_notes = list(drum_part.flatten().notes)
        print(f"   Tempo {tempo} BPM: {len(all_notes)} notes")
        
        assert len(all_notes) > 0, f"No notes at tempo {tempo}"
    
    print("✅ Tempo variation successful")
    return True


def test_4_emotion_tags():
    """Test 4: 異なる感情タグで生成"""
    print("\n" + "="*60)
    print("Test 4: Emotion Tags")
    print("="*60)
    
    gen = DrumsGeneratorStage2()
    
    emotions = ["calm_low", "neutral_medium", "happy_high"]
    for emotion in emotions:
        drum_part = gen.generate(
            bars=2,
            chords=["C", "G"],
            tempo=120,
            emotion=emotion,
            seed=42
        )
        
        all_notes = list(drum_part.flatten().notes)
        print(f"   Emotion '{emotion}': {len(all_notes)} notes")
        
        assert len(all_notes) > 0, f"No notes for emotion {emotion}"
    
    print("✅ Emotion tags successful")
    return True


def test_5_midi_export():
    """Test 5: MIDI出力"""
    print("\n" + "="*60)
    print("Test 5: MIDI Export")
    print("="*60)
    
    gen = DrumsGeneratorStage2()
    
    drum_part = gen.generate(
        bars=8,
        chords=["C", "G", "Am", "F", "C", "G", "Am", "F"],
        tempo=120,
        emotion="energetic",
        seed=42
    )
    
    output_path = Path("tests/output/test_drums_stage2.mid")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    drum_part.write('midi', fp=output_path)
    
    assert output_path.exists(), "MIDI file not created"
    
    file_size = output_path.stat().st_size
    print(f"   Output: {output_path}")
    print(f"   Size: {file_size} bytes")
    
    assert file_size > 100, "MIDI file too small"
    
    print("✅ MIDI export successful")
    return True


def main():
    """全テスト実行"""
    print("\n" + "="*60)
    print("  Drums Generator Stage2 - Quick Tests")
    print("="*60)
    
    tests = [
        test_1_initialization,
        test_2_fallback_generation,
        test_3_tempo_variation,
        test_4_emotion_tags,
        test_5_midi_export
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        try:
            result = test_func()
            if result:
                passed += 1
        except Exception as e:
            print(f"❌ {test_func.__name__} FAILED: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "="*60)
    print(f"  Test Results: {passed}/{len(tests)} passed")
    if failed == 0:
        print("  🎉 All tests passed!")
    else:
        print(f"  ⚠️  {failed} test(s) failed")
    print("="*60)
    
    return 0 if failed == 0 else 1


if __name__ == '__main__':
    exit(main())
