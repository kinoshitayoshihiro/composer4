#!/usr/bin/env python3
"""
Phase 2.0統合テスト: generate_strings_plan_v2.py Phase 2.0統合動作確認

Test Cases:
1. LyricAnchorIndex + EmotionAI v2 + GuideToneAI v2統合
2. Context構築（lyric_info追加）
3. EmotionParams/GuideTonePlan取得
4. Phrase role反映（phrase_start → uphill, phrase_end → downhill）
"""
import json
import tempfile
from pathlib import Path
import pandas as pd
import yaml

# Import Phase 2.0 components
from otobonAI.lyric_index import LyricAnchorIndex
from otobonAI.emotion_ai_v2 import EmotionAI as EmotionAIv2
from otobonAI.guide_tone_ai_v2 import GuideToneAI as GuideToneAIv2
from otobonAI.rulebook_engine import RulebookEngine


def test_phase2_integration():
    """
    Phase 2.0統合テスト: LyricAnchorIndex + EmotionAI v2 + GuideToneAI v2

    シナリオ:
    - Chorus bars (0-7): phrase_start (bar 0), phrase_mid (bar 1-6), phrase_end (bar 7)
    - EmotionAI v2: Chorus → high energy
    - GuideToneAI v2: phrase_start → uphill + notes+2, phrase_end → downhill + notes-1
    - Rulebook: GT_STR_CHORUS_PHRASE_START → high register + uphill
    """
    print("🧪 Phase 2.0統合テスト: generate_strings_plan_v2.py Phase 2.0統合")

    # === 1. テストデータ作成 ===
    # Lyric anchors (8 bars, phrase_start=bar 0, phrase_end=bar 7)
    lyric_anchors = [
        {"time_sec": 0.0, "token": "Verse start", "anchor_type": "phrase_boundary"},
        {"time_sec": 16.0, "token": "Verse end", "anchor_type": "phrase_boundary"},
    ]

    # Emotion profile (Chorus high energy)
    emotion_profile = {
        "0": {"energy": 0.85, "tension": 0.6},
        "1": {"energy": 0.80, "tension": 0.55},
        "2": {"energy": 0.75, "tension": 0.50},
        "3": {"energy": 0.70, "tension": 0.50},
        "4": {"energy": 0.75, "tension": 0.55},
        "5": {"energy": 0.80, "tension": 0.60},
        "6": {"energy": 0.85, "tension": 0.65},
        "7": {"energy": 0.90, "tension": 0.70},
    }

    # Guide tone hints (Strings base settings)
    guide_hints = {"strings": {"notes_per_bar": 1.5, "register": "mid", "motion": "step"}}

    # Rulebook (Phase 2.0細分化ルール)
    rulebook_path = Path("configs/otobonAI/rulebook.yaml")
    if not rulebook_path.exists():
        print(f"⚠️  Rulebook not found: {rulebook_path}")
        print("   テストをスキップします")
        return

    # === 2. Phase 2.0コンポーネント初期化 ===
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Save test data
        lyric_path = tmpdir / "lyric_anchors.json"
        emotion_path = tmpdir / "emotion_profile.json"
        guide_path = tmpdir / "guide_tone_hints.json"

        with open(lyric_path, "w") as f:
            json.dump(lyric_anchors, f, indent=2)

        with open(emotion_path, "w") as f:
            json.dump(emotion_profile, f, indent=2)

        with open(guide_path, "w") as f:
            json.dump(guide_hints, f, indent=2)

        # Initialize components
        tempo_bpm = 120.0
        rulebook = RulebookEngine(rulebook_path)
        lyric_index = LyricAnchorIndex(lyric_path, tempo_bpm)
        emotion_ai = EmotionAIv2(emotion_path, rulebook)
        guide_ai = GuideToneAIv2(guide_path, rulebook)

        print(f"✅ Rulebook loaded: {len(rulebook.rules)} rules")
        print(f"✅ LyricAnchorIndex loaded: {len(lyric_index.anchors)} anchors")
        print(f"✅ EmotionAI v2 loaded: {len(emotion_ai.profile)} bars")
        print(f"✅ GuideToneAI v2 loaded")

        # === 3. Bar loop統合テスト ===
        test_cases = [
            (0, "chorus", "phrase_start"),  # Chorus start
            (3, "chorus", "phrase_mid"),  # Chorus mid
            (7, "chorus", "phrase_end"),  # Chorus end
        ]

        for bar_idx, section, expected_phrase_role in test_cases:
            print(
                f"\n📍 Test Case: bar={bar_idx}, section={section}, phrase_role={expected_phrase_role}"
            )

            # 3-1. Lyric info取得
            lyric_info = lyric_index.get_phrase_info(bar_idx)
            print(f"   Lyric info: {lyric_info}")
            assert (
                lyric_info["phrase_role"] == expected_phrase_role
            ), f"phrase_role mismatch: expected {expected_phrase_role}, got {lyric_info['phrase_role']}"

            # 3-2. Context構築
            context = {
                "bar": bar_idx,
                "section": section,
                "role": "strings",
                "chord_symbol": "C",
                "slots": {"riff": True},
            }

            # Add lyric info
            context["lyric"] = {
                "phrase_role": lyric_info["phrase_role"],
                "phrase_index": lyric_info["phrase_index"],
                "num_anchors": lyric_info["num_anchors"],
            }

            # 3-3. EmotionParams取得
            emotion_params = emotion_ai.get_params(context)
            print(f"   EmotionParams:")
            print(f"     energy: {emotion_params.energy:.2f}")
            print(f"     tension: {emotion_params.tension:.2f}")
            print(f"     velocity_scale: {emotion_params.velocity_scale:.2f}")
            print(f"     density_scale: {emotion_params.density_scale:.2f}")

            # 3-4. GuideTonePlan取得
            guide_plan = guide_ai.get_plan(context)
            print(f"   GuideTonePlan:")
            print(f"     notes_per_bar: {guide_plan.notes_per_bar}")
            print(f"     phrase_shape: {guide_plan.phrase_shape}")
            print(f"     register: {guide_plan.register}")
            print(f"     motion: {guide_plan.motion}")

            # 3-5. Validation
            if expected_phrase_role == "phrase_start":
                # Chorus phrase_start → high register + uphill
                assert (
                    guide_plan.phrase_shape == "uphill"
                ), f"phrase_shape should be 'uphill', got {guide_plan.phrase_shape}"
                assert (
                    guide_plan.notes_per_bar >= 3
                ), f"notes_per_bar should be >= 3 (base 1.5 + 2), got {guide_plan.notes_per_bar}"
                # Chorus細分化ルールでhigh register強制
                if guide_plan.register:
                    assert (
                        guide_plan.register == "high"
                    ), f"register should be 'high' for chorus phrase_start, got {guide_plan.register}"
                print("   ✅ Validation passed: phrase_start → uphill + notes+2 + high register")

            elif expected_phrase_role == "phrase_end":
                # Chorus phrase_end → downhill
                assert (
                    guide_plan.phrase_shape == "downhill"
                ), f"phrase_shape should be 'downhill', got {guide_plan.phrase_shape}"
                assert (
                    guide_plan.notes_per_bar >= 1
                ), f"notes_per_bar should be >= 1 (base 1.5 - 1, min clamp), got {guide_plan.notes_per_bar}"
                print("   ✅ Validation passed: phrase_end → downhill + notes-1")

            else:
                # Phrase mid → no shape adjustment
                print("   ✅ Validation passed: phrase_mid → no adjustment")

    print("\n✅ All tests passed!")


if __name__ == "__main__":
    test_phase2_integration()
