#!/usr/bin/env python3
"""
GuideToneAI v2 単体テスト
Phase 2.0統一context対応版の動作確認
"""

import sys
from pathlib import Path

# Add otobonAI to path
sys.path.insert(0, str(Path(__file__).parent))

from otobonAI.guide_tone_ai_v2 import GuideToneAI, GuideTonePlan


def test_guide_tone_ai_v2():
    """GuideToneAI v2の単体テスト"""

    # 初期化
    guide_hints_path = "data/suno_ai/suno_themesong/song_004/analysis/guide_tone_hints.json"
    rulebook_path = "configs/otobonAI/rulebook.yaml"

    print("🎸 GuideToneAI v2 Unit Test")
    print("=" * 60)

    try:
        guide_ai = GuideToneAI.from_files(guide_hints_path, rulebook_path)
        print(f"✅ GuideToneAI初期化成功")
        print(f"   - Guide hints: {guide_hints_path}")
        print(f"   - Rulebook: {rulebook_path}")
        print()
    except Exception as e:
        print(f"❌ GuideToneAI初期化失敗: {e}")
        import traceback

        traceback.print_exc()
        return False

    # テストケース1: Chorus strings + phrase start
    print("📌 Test Case 1: Chorus Strings + Phrase Start")
    print("-" * 60)
    context1 = {
        "bar_index": 10,
        "section": "chorus",
        "role": "strings",
        "emotion": {"energy": 0.75, "tension": 0.6},
        "lyric": {
            "has_anchor": True,
            "phrase_role": "start",
            "stress_level": 0.8,
            "is_silent": False,
        },
        "chord_symbol": "Cmaj7",
        "key_center": "C",
        "tempo_bpm": 120.0,
        "slots": {"has_fill": False, "has_riff": False},
    }

    try:
        plan1 = guide_ai.get_plan(context1)
        print(f"✅ GuideTonePlan生成成功")
        print(f"   Notes per bar: {plan1.notes_per_bar}")
        print(f"   Preferred degrees: {plan1.preferred_degrees}")
        print(f"   Avoid degrees: {plan1.avoid_degrees}")
        print(f"   Register: {plan1.register}")
        print(f"   Motion: {plan1.motion}")
        print(f"   Phrase Role: {plan1.phrase_role}")
        print(f"   Phrase Shape: {plan1.phrase_shape}")
        print()

        # Validation
        assert (
            plan1.phrase_role == "start"
        ), f"phrase_role should be 'start', got '{plan1.phrase_role}'"
        assert (
            plan1.phrase_shape == "uphill"
        ), f"phrase_shape should be 'uphill' for start, got '{plan1.phrase_shape}'"
        # notes_per_bar should be increased (base 1 + 2 = 3)
        assert (
            plan1.notes_per_bar >= 3
        ), f"notes_per_bar should be increased for start, got {plan1.notes_per_bar}"
        # Note: register depends on rulebook matching, not tested strictly here
        print("✅ Validation passed (Chorus strings + phrase start特性確認)")
        print()

    except Exception as e:
        print(f"❌ Test Case 1失敗: {e}")
        import traceback

        traceback.print_exc()
        return False

    # テストケース2: Verse piano + phrase end
    print("📌 Test Case 2: Verse Piano + Phrase End")
    print("-" * 60)
    context2 = {
        "bar_index": 5,
        "section": "verse",
        "role": "piano",
        "emotion": {"energy": 0.4, "tension": 0.3},
        "lyric": {
            "has_anchor": True,
            "phrase_role": "end",
            "stress_level": 0.5,
            "is_silent": False,
        },
        "chord_symbol": "Am7",
        "key_center": "C",
        "tempo_bpm": 120.0,
        "slots": {"has_fill": False, "has_riff": False},
    }

    try:
        plan2 = guide_ai.get_plan(context2)
        print(f"✅ GuideTonePlan生成成功")
        print(f"   Notes per bar: {plan2.notes_per_bar}")
        print(f"   Preferred degrees: {plan2.preferred_degrees}")
        print(f"   Avoid degrees: {plan2.avoid_degrees}")
        print(f"   Register: {plan2.register}")
        print(f"   Motion: {plan2.motion}")
        print(f"   Phrase Role: {plan2.phrase_role}")
        print(f"   Phrase Shape: {plan2.phrase_shape}")
        print()

        # Validation
        assert plan2.phrase_role == "end", f"phrase_role should be 'end', got '{plan2.phrase_role}'"
        assert (
            plan2.phrase_shape == "downhill"
        ), f"phrase_shape should be 'downhill' for end, got '{plan2.phrase_shape}'"
        # notes_per_bar should be decreased (likely 0 from 1-1, but clamped to min 1)
        assert (
            plan2.notes_per_bar <= 2
        ), f"notes_per_bar should be decreased for end, got {plan2.notes_per_bar}"
        # Note: register depends on rulebook matching, not tested strictly here
        print("✅ Validation passed (Verse piano + phrase end特性確認)")
        print()

    except Exception as e:
        print(f"❌ Test Case 2失敗: {e}")
        import traceback

        traceback.print_exc()
        return False

    # テストケース3: Bridge strings + phrase mid
    print("📌 Test Case 3: Bridge Strings + Phrase Mid")
    print("-" * 60)
    context3 = {
        "bar_index": 25,
        "section": "bridge",
        "role": "strings",
        "emotion": {"energy": 0.55, "tension": 0.5},
        "lyric": {
            "has_anchor": True,
            "phrase_role": "mid",
            "stress_level": 0.6,
            "is_silent": False,
        },
        "chord_symbol": "Fmaj7",
        "key_center": "C",
        "tempo_bpm": 120.0,
        "slots": {"has_fill": False, "has_riff": False},
    }

    try:
        plan3 = guide_ai.get_plan(context3)
        print(f"✅ GuideTonePlan生成成功")
        print(f"   Notes per bar: {plan3.notes_per_bar}")
        print(f"   Preferred degrees: {plan3.preferred_degrees}")
        print(f"   Avoid degrees: {plan3.avoid_degrees}")
        print(f"   Register: {plan3.register}")
        print(f"   Motion: {plan3.motion}")
        print(f"   Phrase Role: {plan3.phrase_role}")
        print(f"   Phrase Shape: {plan3.phrase_shape}")
        print()

        # Validation
        assert plan3.phrase_role == "mid", f"phrase_role should be 'mid', got '{plan3.phrase_role}'"
        assert (
            plan3.phrase_shape is None
        ), f"phrase_shape should be None for mid, got '{plan3.phrase_shape}'"
        print("✅ Validation passed (Bridge strings + phrase mid特性確認)")
        print()

    except Exception as e:
        print(f"❌ Test Case 3失敗: {e}")
        import traceback

        traceback.print_exc()
        return False

    # 比較分析
    print("📊 Comparison Analysis")
    print("=" * 60)
    print(f"Phrase Shape自動調整:")
    print(f"  Start: phrase_shape={plan1.phrase_shape}, notes_per_bar={plan1.notes_per_bar}")
    print(f"  Mid:   phrase_shape={plan3.phrase_shape}, notes_per_bar={plan3.notes_per_bar}")
    print(f"  End:   phrase_shape={plan2.phrase_shape}, notes_per_bar={plan2.notes_per_bar}")
    print()
    print(f"Register対応:")
    print(f"  Chorus strings: {plan1.register}")
    print(f"  Verse piano:    {plan2.register}")
    print(f"  Bridge strings: {plan3.register}")
    print()
    print(f"Preferred degrees:")
    print(f"  Chorus: {plan1.preferred_degrees}")
    print(f"  Verse:  {plan2.preferred_degrees}")
    print(f"  Bridge: {plan3.preferred_degrees}")
    print()

    print("✅ All tests passed!")
    return True


if __name__ == "__main__":
    success = test_guide_tone_ai_v2()
    sys.exit(0 if success else 1)
