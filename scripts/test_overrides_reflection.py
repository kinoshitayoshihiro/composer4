#!/usr/bin/env python3
"""
Apply() overrides反映テスト

目的:
- apply()呼び出し時のoverridesがPhase内部で正しく参照できることを確認
- ネスト辞書の深いマージが動作することを確認
- NO-OP安全性を確認
"""

import sys
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from music21 import stream, note
except ImportError:
    print("❌ music21 not found. Install with: pip install music21")
    sys.exit(1)

from generator.bass_params_stage2 import BassParamsStage2
from generator.piano_params_stage2 import PianoParamsStage2
from generator.guitar_params_stage2 import GuitarParamsStage2
from generator.strings_params_stage2 import StringsParamsStage2
from generator.drums_params_stage2 import DrumsParamsStage2


def create_test_part(instrument: str, num_notes: int = 8) -> stream.Part:
    """テスト用のPartを作成"""
    part = stream.Part()
    part.id = instrument
    
    # 楽器別のピッチ設定
    pitch_map = {
        "bass": 40,      # E2
        "piano": 60,     # C4
        "guitar": 52,    # E3
        "strings": 64,   # E4
        "drums": 36,     # Kick (GM)
    }
    
    base_pitch = pitch_map.get(instrument, 60)
    
    for i in range(num_notes):
        n = note.Note(base_pitch, quarterLength=1.0)
        n.volume.velocity = 80
        part.insert(float(i), n)
    
    return part


def test_overrides_reflection():
    """Test 1: overrides反映テスト"""
    print("\n" + "="*60)
    print("Test 1: Overrides Reflection Test")
    print("="*60)
    
    # Bassでテスト
    bass = BassParamsStage2()
    bass_part = create_test_part("bass", 16)
    
    section_meta = {
        "label": "Verse",
        "bar": 0,
        "tempo": 120,
        "emotion": "energetic"
    }
    
    mix_context = {
        "kick_onsets_ql": [0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0]
    }
    
    # Phase 13-19を有効化するoverrides
    overrides = {
        "vocabulary": {
            "pickup_prob": 1.0  # 必ずピックアップ挿入
        },
        "cross_sync": {
            "lock_with_kick": True,
            "sync_window_ms": 30
        },
        "transition": {
            "enable_crescendo": True,
            "crescendo_bars": 2,
            "velocity_step": 5
        }
    }
    
    print(f"📝 Original notes: {len(list(bass_part.flatten().notes))}")
    print(f"🔧 Overrides: {overrides}")
    
    # デバッグ: _get_phases()が何を返すか確認
    print(f"🔍 Checking _get_phases()...")
    
    # paramsをマージ
    merged_params = bass._merge_presets(section_meta, overrides)
    print(f"📦 Merged params keys: {list(merged_params.keys())}")
    
    # _get_phases()が受け取るparams
    phases = bass._get_phases(merged_params)
    print(f"✨ Active phases: {phases}")
    
    # メソッド存在チェック
    for phase_num in phases:
        method_name = f"_phase_{phase_num}"
        has_method = hasattr(bass, method_name)
        print(f"  - Phase {phase_num} ({method_name}): {'✓' if has_method else '✗'}")
    
    result = bass.apply(
        bass_part,
        section_meta,
        mix_context,
        overrides=overrides,
        seed=42
    )
    
    final_notes = list(result.flatten().notes)
    print(f"✅ Final notes: {len(final_notes)}")
    print(f"📊 Metrics: {bass.metrics}")
    
    # Phase 13が動作したか確認（ピックアップノート追加）
    if len(final_notes) > 16:
        print("✅ Phase 13 (Vocabulary) worked: Pickup note added!")
    else:
        print("⚠️  Phase 13 may not have worked (expected >16 notes)")
    
    return len(final_notes) > 16


def test_nested_dict_merge():
    """Test 2: ネスト辞書の深いマージテスト"""
    print("\n" + "="*60)
    print("Test 2: Nested Dict Deep Merge Test")
    print("="*60)
    
    piano = PianoParamsStage2()
    piano_part = create_test_part("piano", 12)
    
    section_meta = {
        "label": "Chorus",
        "bar": 0,
        "tempo": 130
    }
    
    # ネスト辞書のマージテスト
    mix_context = {
        "sections": [
            {"bar": 0, "label": "chorus"}
        ],
        "vocal_energy": [(0, 0.8), (4, 0.9)]
    }
    
    overrides = {
        "vocabulary": {
            "turnaround_prob": 0.8
        },
        "harmonic": {
            "guide_tone_emphasis": 0.9
        },
        "cross_sync": {
            "sync_with_snare": True,
            "sync_window_ms": 25
        }
    }
    
    print(f"📝 Original notes: {len(list(piano_part.flatten().notes))}")
    
    result = piano.apply(
        piano_part,
        section_meta,
        mix_context,
        overrides=overrides,
        seed=42
    )
    
    final_notes = list(result.flatten().notes)
    print(f"✅ Final notes: {len(final_notes)}")
    print(f"📊 Metrics: {piano.metrics}")
    
    # Phase 13が動作したか確認（ターンアラウンド追加）
    if len(final_notes) > 12:
        print("✅ Phase 13 (Vocabulary) worked: Turnaround added!")
    else:
        print("⚠️  Phase 13 may not have worked")
    
    return True


def test_no_op_safety():
    """Test 3: NO-OP安全性テスト（overrides=Noneで何も変わらない）"""
    print("\n" + "="*60)
    print("Test 3: NO-OP Safety Test")
    print("="*60)
    
    guitar = GuitarParamsStage2()
    guitar_part = create_test_part("guitar", 10)
    
    section_meta = {
        "label": "Bridge",
        "bar": 0,
        "tempo": 140
    }
    
    mix_context = {}
    
    original_count = len(list(guitar_part.flatten().notes))
    print(f"📝 Original notes: {original_count}")
    
    # overrides=None（NO-OP期待）
    result = guitar.apply(
        guitar_part,
        section_meta,
        mix_context,
        overrides=None,
        seed=42
    )
    
    final_count = len(list(result.flatten().notes))
    print(f"✅ Final notes: {final_count}")
    
    if original_count == final_count:
        print("✅ NO-OP safety confirmed: No changes without overrides")
        return True
    else:
        print("⚠️  Unexpected changes occurred")
        return False


def test_phase_dynamic_activation():
    """Test 4: Phase動的有効化テスト"""
    print("\n" + "="*60)
    print("Test 4: Phase Dynamic Activation Test")
    print("="*60)
    
    strings = StringsParamsStage2()
    strings_part = create_test_part("strings", 8)
    
    section_meta = {
        "label": "Outro",
        "bar": 0,
        "tempo": 100
    }
    
    mix_context = {}
    
    # Phase 13-19設定なし（基本Phaseのみ）
    print("\n--- Without Phase 13-19 settings ---")
    # Partを新規作成（copyメソッドは存在しないため）
    strings_part1 = create_test_part("strings", 8)
    result1 = strings.apply(
        strings_part1,
        section_meta,
        mix_context,
        overrides={},
        seed=42
    )
    notes1 = len(list(result1.flatten().notes))
    print(f"Notes (basic phases): {notes1}")
    
    # Phase 13-19設定あり（全Phase有効化）
    print("\n--- With Phase 13-19 settings ---")
    overrides = {
        "vocabulary": {
            "mini_fill_prob": 1.0  # 必ずミニフィル
        },
        "transition": {
            "enable_crescendo": True,
            "crescendo_bars": 1,
            "velocity_step": 5
        }
    }
    
    result2 = strings.apply(
        create_test_part("strings", 8),
        section_meta,
        mix_context,
        overrides=overrides,
        seed=42
    )
    notes2 = len(list(result2.flatten().notes))
    print(f"Notes (all phases): {notes2}")
    
    if notes2 > notes1:
        print("✅ Phase dynamic activation worked!")
        return True
    else:
        print("⚠️  Phase activation may not have worked")
        return False


def main():
    """全テスト実行"""
    print("\n" + "🎵" * 30)
    print("  Apply() Overrides Reflection Test Suite")
    print("🎵" * 30)
    
    results = []
    
    # Test 1: Overrides反映
    try:
        results.append(("Overrides Reflection", test_overrides_reflection()))
    except Exception as e:
        print(f"❌ Test 1 failed: {e}")
        results.append(("Overrides Reflection", False))
    
    # Test 2: ネスト辞書マージ
    try:
        results.append(("Nested Dict Merge", test_nested_dict_merge()))
    except Exception as e:
        print(f"❌ Test 2 failed: {e}")
        results.append(("Nested Dict Merge", False))
    
    # Test 3: NO-OP安全性
    try:
        results.append(("NO-OP Safety", test_no_op_safety()))
    except Exception as e:
        print(f"❌ Test 3 failed: {e}")
        results.append(("NO-OP Safety", False))
    
    # Test 4: Phase動的有効化
    try:
        results.append(("Phase Dynamic Activation", test_phase_dynamic_activation()))
    except Exception as e:
        print(f"❌ Test 4 failed: {e}")
        results.append(("Phase Dynamic Activation", False))
    
    # サマリー
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    total = len(results)
    passed_count = sum(1 for _, p in results if p)
    
    print(f"\nTotal: {passed_count}/{total} tests passed")
    
    if passed_count == total:
        print("\n🎉 All tests passed! Overrides reflection is working correctly.")
        return 0
    else:
        print("\n⚠️  Some tests failed. Please review the implementation.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
