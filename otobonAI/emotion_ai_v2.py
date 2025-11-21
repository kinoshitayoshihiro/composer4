"""
EmotionAI v2.0 - Phase 2.0 Refactoring
RulebookEngine統合、統一context、dataclass化。
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional
from pathlib import Path
import json

from otobonAI.rulebook_engine import Rulebook


@dataclass
class EmotionParams:
    """
    EmotionAIが返す感情パラメータ（bar単位）。
    """

    energy: float  # 0.0-1.0
    tension: float  # 0.0-1.0
    brightness: float  # 0.0-1.0 (新規)
    valence: float  # 0.0-1.0 (新規)

    velocity_scale: float  # 0.5-1.5（velocity調整倍率）
    duration_scale: float  # 0.7-1.3（duration調整倍率）
    density_scale: float  # 0.5-1.5（density調整倍率）

    phrase_role: str  # "start" | "mid" | "end" | "none"（lyric由来）
    tags: list[str]  # ["vocal_focus", "climax", ...]


class EmotionAI:
    """
    Phase 2.0 EmotionAI:
    - emotion_profile.json（bar単位base値）
    - rulebook.yaml（context依存調整）
    - lyric_anchors（phrase_role）
    を統合して、bar単位のEmotionParamsを返す。
    """

    def __init__(
        self,
        emotion_profile: Dict[int, Dict[str, float]],
        rulebook: Rulebook,
    ):
        """
        Args:
            emotion_profile: {bar_index: {"energy": float, "tension": float, ...}}
            rulebook: Rulebookインスタンス
        """
        self.profile = emotion_profile
        self.rulebook = rulebook

    def get_params(self, context: Dict[str, Any]) -> EmotionParams:
        """
        統一contextからEmotionParamsを生成。

        Args:
            context: {
                "bar_index": int,
                "section": str,
                "role": str,
                "emotion": {"energy": float, "tension": float},  # base値
                "lyric": {
                    "phrase_role": str,
                    "stress_level": float,
                    "is_silent": bool
                },
                "chord_symbol": str,
                ...
            }

        Returns:
            EmotionParams
        """
        bar = context.get("bar_index", 0)

        # Get base emotion from context (if provided) or profile
        if "emotion" in context and isinstance(context["emotion"], dict):
            # Use emotion from context (for testing or override)
            base = context["emotion"]
        else:
            # Get from profile (bar-specific if available)
            base = self.profile.get(
                bar, {"energy": 0.5, "tension": 0.5, "brightness": 0.5, "valence": 0.5}
            )
            # Update context with base emotion
            context = {**context, "emotion": base}

        # Query rulebook for emotion domain
        actions = self.rulebook.query(context, domain="emotion")

        # Apply deltas
        energy = base.get("energy", 0.5) + actions.get("energy_delta", 0.0)
        tension = base.get("tension", 0.5) + actions.get("tension_delta", 0.0)
        brightness = base.get("brightness", 0.5) + actions.get("brightness_delta", 0.0)
        valence = base.get("valence", 0.5) + actions.get("valence_delta", 0.0)

        # Clamp to [0, 1]
        energy = max(0.0, min(1.0, energy))
        tension = max(0.0, min(1.0, tension))
        brightness = max(0.0, min(1.0, brightness))
        valence = max(0.0, min(1.0, valence))

        # Calculate scales from energy/tension
        velocity_scale = 0.8 + (energy * 0.4)  # 0.8-1.2
        duration_scale = 1.0 - (tension * 0.3) + 0.15  # 0.85-1.15（high tension→短く）
        density_scale = 0.5 + energy  # 0.5-1.5

        # Rulebook overrides
        if "velocity_scale" in actions:
            velocity_scale = actions["velocity_scale"]
        if "duration_scale" in actions:
            duration_scale = actions["duration_scale"]
        if "density_scale" in actions:
            density_scale = actions["density_scale"]

        # Phrase role from lyric context
        phrase_role = context.get("lyric", {}).get("phrase_role", "none")

        # Tags from rulebook
        tags = actions.get("tags_add", [])

        return EmotionParams(
            energy=energy,
            tension=tension,
            brightness=brightness,
            valence=valence,
            velocity_scale=velocity_scale,
            duration_scale=duration_scale,
            density_scale=density_scale,
            phrase_role=phrase_role,
            tags=tags,
        )

    @classmethod
    def from_files(
        cls,
        profile_path: Path,
        rulebook_path: Path,
    ) -> "EmotionAI":
        """
        ファイルからEmotionAIインスタンス生成。

        Args:
            profile_path: emotion_profile.json path
            rulebook_path: rulebook.yaml path

        Returns:
            EmotionAI instance
        """
        with open(profile_path, "r", encoding="utf-8") as f:
            profile_data = json.load(f)

        # Convert events list to dict
        profile = {}
        for ev in profile_data.get("events", []):
            bar = ev.get("bar", 0)
            profile[bar] = {
                "energy": ev.get("energy", 0.5),
                "tension": ev.get("tension", 0.5),
                "brightness": ev.get("brightness", 0.5),
                "valence": ev.get("valence", 0.5),
            }

        rulebook = Rulebook.load(rulebook_path)

        return cls(profile, rulebook)


def main():
    """テスト実行"""
    import sys

    if len(sys.argv) < 3:
        print("Usage: python3 emotion_ai_v2.py <emotion_profile.json> <rulebook.yaml>")
        sys.exit(1)

    ai = EmotionAI.from_files(Path(sys.argv[1]), Path(sys.argv[2]))

    # Test context
    context = {
        "bar_index": 0,
        "section": "chorus",
        "role": "strings",
        "lyric": {
            "phrase_role": "start",
            "stress_level": 0.8,
            "is_silent": False,
        },
        "chord_symbol": "Cmaj7",
    }

    params = ai.get_params(context)

    print("🎭 EmotionAI v2.0 Test")
    print(f"Bar 0 (chorus, phrase_start):")
    print(f"  Energy: {params.energy:.2f}")
    print(f"  Tension: {params.tension:.2f}")
    print(f"  Velocity scale: {params.velocity_scale:.2f}")
    print(f"  Duration scale: {params.duration_scale:.2f}")
    print(f"  Density scale: {params.density_scale:.2f}")
    print(f"  Phrase role: {params.phrase_role}")
    print(f"  Tags: {params.tags}")


if __name__ == "__main__":
    main()
