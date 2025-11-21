#!/usr/bin/env python3
"""
Phase 2.0細分化ルールのテスト
ChatGPT提案：section × phrase_role 組み合わせルールの動作確認
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from otobonAI.rulebook_engine import Rulebook


def test_phase2_rulebook():
    """Phase 2.0細分化ルールのテスト"""

    rulebook_path = "configs/otobonAI/rulebook.yaml"

    print("🎼 Phase 2.0 Rulebook細分化ルールテスト")
    print("=" * 70)

    try:
        rulebook = Rulebook.load(rulebook_path)
        print(f"✅ Rulebook読み込み成功: {len(rulebook.rules)} ルール")
        print()
    except Exception as e:
        print(f"❌ Rulebook読み込み失敗: {e}")
        import traceback

        traceback.print_exc()
        return False

    # テストケース1: Chorus strings + phrase_start
    print("📌 Test Case 1: Chorus Strings + Phrase Start")
    print("-" * 70)
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
    }

    try:
        matched_rules = rulebook.list_matched_rules(context1, "guide_tone")
        print(f"✅ マッチルール数: {len(matched_rules)}")
        for i, rule in enumerate(matched_rules[:5], 1):
            print(f"   {i}. {rule.id}: {rule.name}")
        print()

        actions = rulebook.query(context1, "guide_tone")
        print(f"✅ 統合アクション:")
        print(f"   register: {actions.get('register', 'N/A')}")
        print(f"   motion: {actions.get('motion', 'N/A')}")
        print(f"   notes_per_bar: {actions.get('notes_per_bar', 'N/A')}")
        print(f"   phrase_shape: {actions.get('phrase_shape', 'N/A')}")
        print(f"   priority_tones: {actions.get('priority_tones', [])}")
        print()

        # Validation
        assert "GT_STR_CHORUS_PHRASE_START" in [
            r.id for r in matched_rules
        ], "GT_STR_CHORUS_PHRASE_START should match"
        assert (
            actions.get("register") == "high"
        ), f"register should be 'high', got '{actions.get('register')}'"
        assert (
            actions.get("phrase_shape") == "uphill"
        ), f"phrase_shape should be 'uphill', got '{actions.get('phrase_shape')}'"
        print("✅ Validation passed (Chorus phrase_start特性確認)")
        print()

    except Exception as e:
        print(f"❌ Test Case 1失敗: {e}")
        import traceback

        traceback.print_exc()
        return False

    # テストケース2: Chorus strings + phrase_end
    print("📌 Test Case 2: Chorus Strings + Phrase End")
    print("-" * 70)
    context2 = {
        "bar_index": 15,
        "section": "chorus",
        "role": "strings",
        "emotion": {"energy": 0.6, "tension": 0.5},
        "lyric": {
            "has_anchor": True,
            "phrase_role": "end",
            "stress_level": 0.5,
            "is_silent": False,
        },
        "chord_symbol": "Cmaj7",
        "key_center": "C",
        "tempo_bpm": 120.0,
    }

    try:
        matched_rules = rulebook.list_matched_rules(context2, "guide_tone")
        print(f"✅ マッチルール数: {len(matched_rules)}")
        for i, rule in enumerate(matched_rules[:5], 1):
            print(f"   {i}. {rule.id}: {rule.name}")
        print()

        actions = rulebook.query(context2, "guide_tone")
        print(f"✅ 統合アクション:")
        print(f"   register: {actions.get('register', 'N/A')}")
        print(f"   motion: {actions.get('motion', 'N/A')}")
        print(f"   notes_per_bar: {actions.get('notes_per_bar', 'N/A')}")
        print(f"   phrase_shape: {actions.get('phrase_shape', 'N/A')}")
        print(f"   priority_tones: {actions.get('priority_tones', [])}")
        print()

        # Validation
        assert "GT_STR_CHORUS_PHRASE_END" in [
            r.id for r in matched_rules
        ], "GT_STR_CHORUS_PHRASE_END should match"
        assert (
            actions.get("phrase_shape") == "downhill"
        ), f"phrase_shape should be 'downhill', got '{actions.get('phrase_shape')}'"
        print("✅ Validation passed (Chorus phrase_end特性確認)")
        print()

    except Exception as e:
        print(f"❌ Test Case 2失敗: {e}")
        import traceback

        traceback.print_exc()
        return False

    # テストケース3: Verse strings + phrase_start
    print("📌 Test Case 3: Verse Strings + Phrase Start")
    print("-" * 70)
    context3 = {
        "bar_index": 5,
        "section": "verse",
        "role": "strings",
        "emotion": {"energy": 0.4, "tension": 0.3},
        "lyric": {
            "has_anchor": True,
            "phrase_role": "start",
            "stress_level": 0.6,
            "is_silent": False,
        },
        "chord_symbol": "Am7",
        "key_center": "C",
        "tempo_bpm": 120.0,
    }

    try:
        matched_rules = rulebook.list_matched_rules(context3, "guide_tone")
        print(f"✅ マッチルール数: {len(matched_rules)}")
        for i, rule in enumerate(matched_rules[:5], 1):
            print(f"   {i}. {rule.id}: {rule.name}")
        print()

        actions = rulebook.query(context3, "guide_tone")
        print(f"✅ 統合アクション:")
        print(f"   register: {actions.get('register', 'N/A')}")
        print(f"   motion: {actions.get('motion', 'N/A')}")
        print(f"   notes_per_bar: {actions.get('notes_per_bar', 'N/A')}")
        print(f"   priority_tones: {actions.get('priority_tones', [])}")
        print()

        # Validation
        assert "GT_STR_VERSE_PHRASE_START" in [
            r.id for r in matched_rules
        ], "GT_STR_VERSE_PHRASE_START should match"
        assert (
            actions.get("register") == "mid"
        ), f"register should be 'mid', got '{actions.get('register')}'"
        print("✅ Validation passed (Verse phrase_start特性確認)")
        print()

    except Exception as e:
        print(f"❌ Test Case 3失敗: {e}")
        import traceback

        traceback.print_exc()
        return False

    # テストケース4: Emotion domain - Chorus phrase_start
    print("📌 Test Case 4: Emotion - Chorus Strings + Phrase Start")
    print("-" * 70)

    try:
        matched_rules = rulebook.list_matched_rules(context1, "emotion")
        print(f"✅ マッチルール数: {len(matched_rules)}")
        for i, rule in enumerate(matched_rules[:5], 1):
            print(f"   {i}. {rule.id}: {rule.name}")
        print()

        actions = rulebook.query(context1, "emotion")
        print(f"✅ 統合アクション:")
        print(f"   energy_delta: {actions.get('energy_delta', 0.0)}")
        print(f"   tension_delta: {actions.get('tension_delta', 0.0)}")
        print(f"   velocity_scale: {actions.get('velocity_scale', 'N/A')}")
        print(f"   density_scale: {actions.get('density_scale', 'N/A')}")
        print()

        # Validation
        assert "EMO_STR_CHORUS_PHRASE_START" in [
            r.id for r in matched_rules
        ], "EMO_STR_CHORUS_PHRASE_START should match"
        assert (
            actions.get("energy_delta", 0.0) > 0
        ), f"energy_delta should be positive, got {actions.get('energy_delta', 0.0)}"
        print("✅ Validation passed (Emotion boost確認)")
        print()

    except Exception as e:
        print(f"❌ Test Case 4失敗: {e}")
        import traceback

        traceback.print_exc()
        return False

    # テストケース5: Guitar chorus phrase_start
    print("📌 Test Case 5: Chorus Guitar + Phrase Start")
    print("-" * 70)
    context5 = {
        "bar_index": 12,
        "section": "chorus",
        "role": "guitar",
        "emotion": {"energy": 0.65, "tension": 0.55},
        "lyric": {
            "has_anchor": True,
            "phrase_role": "start",
            "stress_level": 0.7,
            "is_silent": False,
        },
        "chord_symbol": "G",
        "key_center": "C",
        "tempo_bpm": 128.0,
    }

    try:
        matched_rules = rulebook.list_matched_rules(context5, "guide_tone")
        print(f"✅ マッチルール数: {len(matched_rules)}")
        for i, rule in enumerate(matched_rules[:5], 1):
            print(f"   {i}. {rule.id}: {rule.name}")
        print()

        actions = rulebook.query(context5, "guide_tone")
        print("✅ 統合アクション:")
        print(f"   register: {actions.get('register', 'N/A')}")
        print(f"   notes_per_bar: {actions.get('notes_per_bar', 'N/A')}")
        print(f"   phrase_shape: {actions.get('phrase_shape', 'N/A')}")
        print()

        assert "GT_GTR_CHORUS_PHRASE_START" in [
            r.id for r in matched_rules
        ], "GT_GTR_CHORUS_PHRASE_START should match"
        assert actions.get("register") == "high", "Guitar chorus start should push high register"
        assert (
            actions.get("notes_per_bar", 0) >= 5
        ), "Guitar chorus start should request busy pattern"
        print("✅ Validation passed (Guitar chorus phrase_start)\n")

    except Exception as e:
        print(f"❌ Test Case 5失敗: {e}")
        import traceback

        traceback.print_exc()
        return False

    # テストケース6: Piano phrase_end landing
    print("📌 Test Case 6: Piano Phrase End Landing")
    print("-" * 70)
    context6 = {
        "bar_index": 18,
        "section": "verse",
        "role": "piano",
        "emotion": {"energy": 0.5, "tension": 0.4},
        "lyric": {
            "has_anchor": True,
            "phrase_role": "end",
            "stress_level": 0.4,
            "is_silent": False,
        },
        "chord_symbol": "Am",
        "key_center": "C",
        "tempo_bpm": 112.0,
    }

    try:
        matched_rules = rulebook.list_matched_rules(context6, "guide_tone")
        print(f"✅ マッチルール数: {len(matched_rules)}")
        for i, rule in enumerate(matched_rules[:5], 1):
            print(f"   {i}. {rule.id}: {rule.name}")
        print()

        actions = rulebook.query(context6, "guide_tone")
        print("✅ 統合アクション:")
        print(f"   register: {actions.get('register', 'N/A')}")
        print(f"   motion: {actions.get('motion', 'N/A')}")
        print(f"   phrase_shape: {actions.get('phrase_shape', 'N/A')}")
        print()

        assert "GT_PNO_PHRASE_END_LANDING" in [
            r.id for r in matched_rules
        ], "GT_PNO_PHRASE_END_LANDING should match"
        assert actions.get("motion") == "hold", "Piano phrase end should favor hold motion"
        assert actions.get("phrase_shape") == "downhill", "Piano phrase end should be downhill"
        print("✅ Validation passed (Piano phrase_end landing)\n")

    except Exception as e:
        print(f"❌ Test Case 6失敗: {e}")
        import traceback

        traceback.print_exc()
        return False

    # テストケース7: Drums phrase_end drop (emotion)
    print("📌 Test Case 7: Drums Phrase End Emotion Drop")
    print("-" * 70)
    context7 = {
        "bar_index": 20,
        "section": "chorus",
        "role": "drums",
        "emotion": {"energy": 0.8, "tension": 0.7},
        "lyric": {
            "has_anchor": True,
            "phrase_role": "end",
            "stress_level": 0.5,
            "is_silent": False,
        },
        "chord_symbol": "F",
        "key_center": "C",
        "tempo_bpm": 128.0,
    }

    try:
        matched_rules = rulebook.list_matched_rules(context7, "emotion")
        print(f"✅ マッチルール数: {len(matched_rules)}")
        for i, rule in enumerate(matched_rules[:5], 1):
            print(f"   {i}. {rule.id}: {rule.name}")
        print()

        actions = rulebook.query(context7, "emotion")
        print("✅ 統合アクション:")
        print(f"   energy_delta: {actions.get('energy_delta', 0.0)}")
        print(f"   density_scale: {actions.get('density_scale', 'N/A')}")
        print(f"   velocity_scale: {actions.get('velocity_scale', 'N/A')}")
        print()

        assert "EMO_DRM_PHRASE_END_DROP" in [
            r.id for r in matched_rules
        ], "EMO_DRM_PHRASE_END_DROP should match"
        assert actions.get("density_scale", 1.0) < 1.0, "Drum phrase end should lower density"
        assert actions.get("velocity_scale", 1.0) < 1.0, "Drum phrase end should reduce velocity"
        print("✅ Validation passed (Drum phrase_end emotion drop)\n")

    except Exception as e:
        print(f"❌ Test Case 7失敗: {e}")
        import traceback

        traceback.print_exc()
        return False

    # まとめ
    print("📊 Summary")
    print("=" * 70)
    print("細分化ルールの動作確認:")
    print("  ✅ Chorus × phrase_start → high register, uphill")
    print("  ✅ Chorus × phrase_end → downhill landing")
    print("  ✅ Verse × phrase_start → mid register, gentle")
    print("  ✅ Emotion boost → chorus phrase_start")
    print("  ✅ Guitar / Piano / Drums 各フレーズロール連携")
    print()
    print("✅ All tests passed!")
    return True


if __name__ == "__main__":
    success = test_phase2_rulebook()
    sys.exit(0 if success else 1)
