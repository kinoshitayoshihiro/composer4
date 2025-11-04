#!/usr/bin/env python3
"""
apply_emotion_to_articulation.py - emotion_profileをarticulation_hintsに反映

emotion_profile.jsonのvalence/arousalをarticulation_hints.jsonに統合し、
表現力を強化したarticulation_hints_with_emotion.jsonを生成します。

Usage:
    python3 scripts/apply_emotion_to_articulation.py \
      --hints song_packages/<project>/<song>/articulation_hints.json \
      --emotion song_packages/<project>/<song>/emotion_profile.json \
      --output song_packages/<project>/<song>/articulation_hints_with_emotion.json
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List


def apply_emotion_modulation(
    hint: Dict,
    section_emotion: Dict,
    global_emotion: Dict,
) -> Dict:
    """
    emotion_profileに基づいてarticulation_hintsを調整

    Arousal高い → accent_score強化、vibrato_depth強化
    Valence高い → legato_ratio強化、energy_curve平滑化
    Arousal低い → staccato_ratio抑制、energy_curve抑制
    Valence低い → accent_score抑制、vibrato_depth抑制
    """
    arousal = section_emotion.get("arousal", global_emotion["arousal"])
    valence = section_emotion.get("valence", global_emotion["valence"])

    # 深いコピー（元のhintを保持）
    modulated = hint.copy()

    # Arousal調整（覚醒度）
    if arousal > 0.7:
        # 高覚醒 → accent強化、vibrato強化
        modulated["accent_score"] = min(1.0, hint.get("accent_score", 0.5) * 1.2)
        modulated["vibrato_depth"] = min(1.0, hint.get("vibrato_depth", 0.5) * 1.15)
        modulated["energy_curve"] = min(1.0, hint.get("energy_curve", 0.5) * 1.1)
    elif arousal < 0.3:
        # 低覚醒 → staccato抑制、energy抑制
        modulated["staccato_ratio"] = hint.get("staccato_ratio", 0.5) * 0.7
        modulated["energy_curve"] = hint.get("energy_curve", 0.5) * 0.8
        modulated["accent_score"] = hint.get("accent_score", 0.5) * 0.8

    # Valence調整（感情価）
    if valence > 0.7:
        # 陽性（明るい）→ legato強化、vibrato抑制
        modulated["legato_ratio"] = min(1.0, hint.get("legato_ratio", 0.5) * 1.15)
        modulated["vibrato_depth"] = hint.get("vibrato_depth", 0.5) * 0.9
    elif valence < 0.3:
        # 陰性（暗い）→ tremolo強化、accent抑制
        modulated["tremolo_ratio"] = min(1.0, hint.get("tremolo_ratio", 0.2) * 1.3)
        modulated["accent_score"] = hint.get("accent_score", 0.5) * 0.85

    # emotion metadataを追加
    modulated["emotion_metadata"] = {
        "arousal": arousal,
        "valence": valence,
        "emotion_label": section_emotion.get("emotion", global_emotion["emotion_label"]),
        "original_accent": hint.get("accent_score", 0.5),
        "original_vibrato": hint.get("vibrato_depth", 0.5),
    }

    return modulated


def apply_emotion_to_articulation(
    hints_path: Path,
    emotion_path: Path,
) -> Dict:
    """articulation_hintsにemotion_profileを適用"""

    # 読み込み
    with open(hints_path) as f:
        hints_data = json.load(f)

    with open(emotion_path) as f:
        emotion_data = json.load(f)

    global_emotion = emotion_data["global_emotion"]
    section_profiles = {
        sec["section_label"]: sec for sec in emotion_data.get("section_profiles", [])
    }
    bar_timeline = {bar["bar_index"]: bar for bar in emotion_data.get("bar_timeline", [])}

    # articulation_hintsの形式確認（list or dict）
    if isinstance(hints_data, list):
        hints_list = hints_data
    elif isinstance(hints_data, dict):
        hints_list = hints_data.get("hints", [])
    else:
        raise ValueError(f"Unexpected hints_data type: {type(hints_data)}")

    # articulation_hintsにemotionを適用
    modulated_hints = []

    for hint in hints_list:
        bar_index = hint.get("bar_index", 0)
        section_label = hint.get("section_label", "unknown")

        # セクション別emotionを取得（なければglobal）
        section_emotion = section_profiles.get(section_label, global_emotion)

        # 小節別emotionを取得（オプション）
        bar_emotion = bar_timeline.get(bar_index, {})
        if bar_emotion:
            # 小節別valence/arousalで上書き
            section_emotion = {
                **section_emotion,
                "arousal": bar_emotion.get("arousal", section_emotion.get("arousal", 0.5)),
                "valence": bar_emotion.get("valence", section_emotion.get("valence", 0.5)),
            }

        # emotion適用
        modulated = apply_emotion_modulation(hint, section_emotion, global_emotion)
        modulated_hints.append(modulated)

    return {
        "hints": modulated_hints,
        "global_emotion": global_emotion,
        "section_profiles": emotion_data.get("section_profiles", []),
        "total_hints": len(modulated_hints),
        "source_hints": str(hints_path.name),
        "source_emotion": str(emotion_path.name),
    }


def main():
    parser = argparse.ArgumentParser(description="Apply emotion profile to articulation hints")
    parser.add_argument("--hints", type=Path, required=True, help="Path to articulation_hints.json")
    parser.add_argument("--emotion", type=Path, required=True, help="Path to emotion_profile.json")
    parser.add_argument(
        "--output", type=Path, required=True, help="Output path for modulated hints"
    )
    args = parser.parse_args()

    if not args.hints.exists():
        raise FileNotFoundError(f"Hints not found: {args.hints}")
    if not args.emotion.exists():
        raise FileNotFoundError(f"Emotion profile not found: {args.emotion}")

    print(f"🎭 Applying emotion profile to articulation hints")
    print(f"   Hints: {args.hints}")
    print(f"   Emotion: {args.emotion}")

    result = apply_emotion_to_articulation(args.hints, args.emotion)

    # 結果保存
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)

    print(f"✅ Modulated hints saved: {args.output}")
    print(f"\n📊 Summary:")
    print(f"   Total hints: {result['total_hints']}")
    print(f"   Global emotion: {result['global_emotion']['emotion_label']}")
    print(f"   Valence: {result['global_emotion']['valence']:.2f}")
    print(f"   Arousal: {result['global_emotion']['arousal']:.2f}")

    if result["section_profiles"]:
        print(f"\n📑 Section Profiles: {len(result['section_profiles'])} sections")
        for sec in result["section_profiles"]:
            print(
                f"   {sec['section_label']}: {sec['emotion']} (V={sec['valence']:.2f}, A={sec['arousal']:.2f})"
            )


if __name__ == "__main__":
    main()
