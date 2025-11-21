"""
GuideToneAI v2.0 - Phase 2.0 Refactoring
RulebookEngine統合、統一context、dataclass化、phrase_role統合。
"""

from dataclasses import dataclass
from typing import Dict, Any, List, Optional
from pathlib import Path
import json

from otobonAI.rulebook_engine import Rulebook


@dataclass
class GuideTonePlan:
    """
    GuideToneAIが返すガイドトーンプラン（bar単位）。
    """

    notes_per_bar: int  # 1-8（このbarで生成する音符数）
    preferred_degrees: List[int]  # [1, 3, 5, 7, 9, 11, 13]（優先スケール度数）
    avoid_degrees: List[int]  # [4, 6]など（避けるスケール度数）
    register: str  # "low" | "mid" | "high"
    motion: str  # "step" | "leap_ok" | "hold"
    phrase_role: str  # "start" | "mid" | "end" | "none"（lyric由来）
    phrase_shape: Optional[str]  # "arch" | "uphill" | "downhill" | None


class GuideToneAI:
    """
    Phase 2.0 GuideToneAI:
    - guide_tone_hints.json（bar単位base値）
    - rulebook.yaml（context依存調整）
    - lyric_anchors（phrase_role）
    - chordmap（和声コンテキスト）
    を統合して、bar単位のGuideTonePlanを返す。
    """

    def __init__(
        self,
        guide_tone_hints: Dict[int, Dict[str, Any]],
        rulebook: Rulebook,
    ):
        """
        Args:
            guide_tone_hints: {bar_index: {"notes_per_bar": int, "register": str, ...}}
            rulebook: Rulebookインスタンス
        """
        self.hints = guide_tone_hints
        self.rulebook = rulebook

    def get_plan(self, context: Dict[str, Any]) -> GuideTonePlan:
        """
        統一contextからGuideTonePlanを生成。

        Args:
            context: {
                "bar_index": int,
                "section": str,
                "role": str,
                "emotion": {"energy": float, "tension": float},
                "lyric": {
                    "phrase_role": str,
                    "stress_level": float,
                    "is_silent": bool
                },
                "chord_symbol": str,
                "key_center": str,
                ...
            }

        Returns:
            GuideTonePlan
        """
        bar = context["bar_index"]

        # Base hint from guide_tone_hints.json
        hint = self.hints.get(bar, {})

        # Update context with hint
        context = {**context, "guide_hint": hint}

        # Query rulebook for guide_tone domain
        actions = self.rulebook.query(context, domain="guide_tone")

        # Phrase role from lyric context
        phrase_role = context.get("lyric", {}).get("phrase_role", "none")

        # Notes per bar: base from rulebook or hint (convert to int first)
        notes_per_bar_base = int(actions.get("notes_per_bar", hint.get("notes_per_bar", 4)))

        # Preferred degrees: merge rulebook + hint
        preferred_degrees = list(
            set(actions.get("priority_tones", []) + hint.get("preferred_degrees", [3, 5, 7]))
        )

        # Convert string tones ("3rd", "7th") to scale degrees if needed
        preferred_degrees = self._normalize_degrees(preferred_degrees)

        # Avoid degrees: from rulebook or hint
        avoid_degrees = actions.get("avoid_degrees", hint.get("avoid_degrees", []))
        avoid_degrees = self._normalize_degrees(avoid_degrees)

        # Register: rulebook > hint > default
        register = actions.get("register", hint.get("register", "mid"))

        # Motion: rulebook > hint > default
        motion = actions.get("motion", hint.get("motion", "step"))

        # Phrase shape: rulebook > hint
        phrase_shape = actions.get("phrase_shape", hint.get("phrase_shape"))

        # Phrase role adjustments (apply AFTER getting base values)
        notes_per_bar = notes_per_bar_base
        if phrase_role == "start":
            # Phrase start: uphill shape, more notes
            phrase_shape = phrase_shape or "uphill"
            notes_per_bar = min(8, notes_per_bar + 2)
        elif phrase_role == "end":
            # Phrase end: downhill shape, longer notes
            phrase_shape = phrase_shape or "downhill"
            notes_per_bar = max(1, notes_per_bar - 1)

        # Final clamp
        notes_per_bar = int(max(1, min(notes_per_bar, 8)))

        return GuideTonePlan(
            notes_per_bar=notes_per_bar,
            preferred_degrees=preferred_degrees,
            avoid_degrees=avoid_degrees,
            register=register,
            motion=motion,
            phrase_role=phrase_role,
            phrase_shape=phrase_shape,
        )

    def _normalize_degrees(self, degrees: List[Any]) -> List[int]:
        """
        スケール度数を正規化（文字列→数値変換）。

        "3rd" → 3
        "7th" → 7
        "9th" → 9
        """
        result = []
        degree_map = {
            "root": 1,
            "1st": 1,
            "2nd": 2,
            "9th": 9,
            "3rd": 3,
            "4th": 4,
            "11th": 11,
            "5th": 5,
            "6th": 6,
            "13th": 13,
            "7th": 7,
        }

        for d in degrees:
            if isinstance(d, int):
                result.append(d)
            elif isinstance(d, str):
                result.append(degree_map.get(d.lower(), 1))

        return sorted(set(result))

    @classmethod
    def from_files(
        cls,
        hints_path: Path,
        rulebook_path: Path,
    ) -> "GuideToneAI":
        """
        ファイルからGuideToneAIインスタンス生成。

        Args:
            hints_path: guide_tone_hints.json path
            rulebook_path: rulebook.yaml path

        Returns:
            GuideToneAI instance
        """
        with open(hints_path, "r", encoding="utf-8") as f:
            hints_data = json.load(f)

        # Convert events list to dict
        hints = {}
        for ev in hints_data.get("events", []):
            bar = ev.get("bar", 0)
            hints[bar] = {
                "notes_per_bar": ev.get("notes_per_bar", 4),
                "preferred_degrees": ev.get("preferred_degrees", [3, 5, 7]),
                "avoid_degrees": ev.get("avoid_degrees", []),
                "register": ev.get("register", "mid"),
                "motion": ev.get("motion", "step"),
                "phrase_shape": ev.get("phrase_shape"),
            }

        rulebook = Rulebook.load(rulebook_path)

        return cls(hints, rulebook)


def main():
    """テスト実行"""
    import sys

    if len(sys.argv) < 3:
        print("Usage: python3 guide_tone_ai_v2.py <guide_tone_hints.json> <rulebook.yaml>")
        sys.exit(1)

    ai = GuideToneAI.from_files(Path(sys.argv[1]), Path(sys.argv[2]))

    # Test context
    context = {
        "bar_index": 0,
        "section": "chorus",
        "role": "strings",
        "emotion": {"energy": 0.7, "tension": 0.6},
        "lyric": {
            "phrase_role": "start",
            "stress_level": 0.8,
            "is_silent": False,
        },
        "chord_symbol": "Cmaj7",
        "key_center": "C",
    }

    plan = ai.get_plan(context)

    print("🎵 GuideToneAI v2.0 Test")
    print(f"Bar 0 (chorus, phrase_start):")
    print(f"  Notes per bar: {plan.notes_per_bar}")
    print(f"  Preferred degrees: {plan.preferred_degrees}")
    print(f"  Avoid degrees: {plan.avoid_degrees}")
    print(f"  Register: {plan.register}")
    print(f"  Motion: {plan.motion}")
    print(f"  Phrase role: {plan.phrase_role}")
    print(f"  Phrase shape: {plan.phrase_shape}")


if __name__ == "__main__":
    main()
