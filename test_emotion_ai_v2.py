#!/usr/bin/env python3
"""
EmotionAI v2 単体テスト
Phase 2.0統一context対応版の動作確認
"""

import sys
from pathlib import Path

# Add otobonAI to path
sys.path.insert(0, str(Path(__file__).parent))

from otobonAI.emotion_ai_v2 import EmotionAI, EmotionParams


def test_emotion_ai_v2():
    """EmotionAI v2の単体テスト"""

    # 初期化
    emotion_profile_path = "data/suno_ai/suno_themesong/song_004/analysis/emotion_profile.json"
    rulebook_path = "configs/otobonAI/rulebook.yaml"

    print("🎭 EmotionAI v2 Unit Test")
    print("=" * 60)

    try:
        emotion_ai = EmotionAI.from_files(emotion_profile_path, rulebook_path)
        print(f"✅ EmotionAI初期化成功")
        print(f"   - Emotion profile: {emotion_profile_path}")
        print(f"   - Rulebook: {rulebook_path}")
        print()
    except Exception as e:
        print(f"❌ EmotionAI初期化失敗: {e}")
        return False

    # テストケース1: Chorus高energy + phrase start
    print("📌 Test Case 1: Chorus高energy + Phrase Start")
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
        params1 = emotion_ai.get_params(context1)
        print(f"✅ EmotionParams生成成功")
        print(f"   Energy: {params1.energy:.2f}")
        print(f"   Tension: {params1.tension:.2f}")
        print(f"   Brightness: {params1.brightness:.2f}")
        print(f"   Valence: {params1.valence:.2f}")
        print(f"   Velocity Scale: {params1.velocity_scale:.2f}")
        print(f"   Duration Scale: {params1.duration_scale:.2f}")
        print(f"   Density Scale: {params1.density_scale:.2f}")
        print(f"   Phrase Role: {params1.phrase_role}")
        print(f"   Tags: {params1.tags}")
        print()

        # Validation
        assert (
            params1.phrase_role == "start"
        ), f"phrase_role should be 'start', got '{params1.phrase_role}'"
        assert (
            params1.velocity_scale > 1.0
        ), f"velocity_scale should be > 1.0 for high energy, got {params1.velocity_scale}"
        print("✅ Validation passed (Chorus高energy特性確認)")
        print()

    except Exception as e:
        print(f"❌ Test Case 1失敗: {e}")
        import traceback

        traceback.print_exc()
        return False

    # テストケース2: Verse低energy + phrase end
    print("📌 Test Case 2: Verse低energy + Phrase End")
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
        params2 = emotion_ai.get_params(context2)
        print(f"✅ EmotionParams生成成功")
        print(f"   Energy: {params2.energy:.2f}")
        print(f"   Tension: {params2.tension:.2f}")
        print(f"   Brightness: {params2.brightness:.2f}")
        print(f"   Valence: {params2.valence:.2f}")
        print(f"   Velocity Scale: {params2.velocity_scale:.2f}")
        print(f"   Duration Scale: {params2.duration_scale:.2f}")
        print(f"   Density Scale: {params2.density_scale:.2f}")
        print(f"   Phrase Role: {params2.phrase_role}")
        print(f"   Tags: {params2.tags}")
        print()

        # Validation
        assert (
            params2.phrase_role == "end"
        ), f"phrase_role should be 'end', got '{params2.phrase_role}'"
        assert (
            params2.velocity_scale < 1.0
        ), f"velocity_scale should be < 1.0 for low energy, got {params2.velocity_scale}"
        print("✅ Validation passed (Verse低energy特性確認)")
        print()

    except Exception as e:
        print(f"❌ Test Case 2失敗: {e}")
        import traceback

        traceback.print_exc()
        return False

    # テストケース3: Bridge中energy + phrase mid
    print("📌 Test Case 3: Bridge中energy + Phrase Mid")
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
        params3 = emotion_ai.get_params(context3)
        print(f"✅ EmotionParams生成成功")
        print(f"   Energy: {params3.energy:.2f}")
        print(f"   Tension: {params3.tension:.2f}")
        print(f"   Brightness: {params3.brightness:.2f}")
        print(f"   Valence: {params3.valence:.2f}")
        print(f"   Velocity Scale: {params3.velocity_scale:.2f}")
        print(f"   Duration Scale: {params3.duration_scale:.2f}")
        print(f"   Density Scale: {params3.density_scale:.2f}")
        print(f"   Phrase Role: {params3.phrase_role}")
        print(f"   Tags: {params3.tags}")
        print()

        # Validation
        assert (
            params3.phrase_role == "mid"
        ), f"phrase_role should be 'mid', got '{params3.phrase_role}'"
        assert (
            0.9 <= params3.velocity_scale <= 1.1
        ), f"velocity_scale should be near 1.0 for mid energy, got {params3.velocity_scale}"
        print("✅ Validation passed (Bridge中energy特性確認)")
        print()

    except Exception as e:
        print(f"❌ Test Case 3失敗: {e}")
        import traceback

        traceback.print_exc()
        return False

    # 比較分析
    print("📊 Comparison Analysis")
    print("=" * 60)
    print(f"Chorus高energy vs Verse低energy:")
    print(
        f"  Energy: {params1.energy:.2f} vs {params2.energy:.2f} (diff: {params1.energy - params2.energy:+.2f})"
    )
    print(
        f"  Velocity Scale: {params1.velocity_scale:.2f} vs {params2.velocity_scale:.2f} (diff: {params1.velocity_scale - params2.velocity_scale:+.2f})"
    )
    print(
        f"  Density Scale: {params1.density_scale:.2f} vs {params2.density_scale:.2f} (diff: {params1.density_scale - params2.density_scale:+.2f})"
    )
    print()
    print(f"Phrase Role対応:")
    print(f"  Start: phrase_role={params1.phrase_role}")
    print(f"  Mid:   phrase_role={params3.phrase_role}")
    print(f"  End:   phrase_role={params2.phrase_role}")
    print()

    print("✅ All tests passed!")
    return True


if __name__ == "__main__":
    success = test_emotion_ai_v2()
    sys.exit(0 if success else 1)
