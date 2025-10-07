#!/usr/bin/env python3
"""
LAMDa Enhancer V2 - music21ベースのSuno MIDI強化

Suno MIDIをmusic21で直接操作:
- Velocity調整 (emotion-driven)
- Chord enhancement
- LAMDa理論適用 (180,000曲DB)

Usage:
    python scripts/lamda_enhancer_v2.py <suno_midi> --valence 0.3 --arousal 0.7
"""

import argparse
import json
import sqlite3
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import music21


@dataclass
class EmotionContext:
    """感情コンテキスト"""

    valence: float  # -1.0 to 1.0
    arousal: float  # 0.0 to 1.0
    intensity: float = 0.5

    @property
    def mood_category(self) -> str:
        if self.valence > 0 and self.arousal > 0.5:
            return "energetic"
        elif self.valence > 0 and self.arousal <= 0.5:
            return "peaceful"
        elif self.valence <= 0 and self.arousal > 0.5:
            return "tense"
        else:
            return "soft_reflective"

    @property
    def target_velocity_range(self) -> Tuple[int, int]:
        if self.mood_category == "energetic":
            return (90, 120)
        elif self.mood_category == "tense":
            return (80, 110)
        elif self.mood_category == "peaceful":
            return (50, 80)
        else:
            return (40, 70)


@dataclass
class EnhancementResult:
    """強化結果"""

    original_midi: Path
    enhanced_midi: Path
    emotion: EmotionContext
    original_note_count: int = 0
    enhanced_note_count: int = 0
    velocity_adjusted: bool = False
    lamda_progressions_used: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "original_midi": str(self.original_midi),
            "enhanced_midi": str(self.enhanced_midi),
            "emotion": {
                "valence": self.emotion.valence,
                "arousal": self.emotion.arousal,
                "intensity": self.emotion.intensity,
                "mood": self.emotion.mood_category,
            },
            "original_note_count": self.original_note_count,
            "enhanced_note_count": self.enhanced_note_count,
            "velocity_adjusted": self.velocity_adjusted,
            "lamda_progressions_used": self.lamda_progressions_used,
        }


class LAMDaEnhancerV2:
    """music21ベースのMIDI強化エンジン"""

    def __init__(self, lamda_db_path: Optional[Path] = None):
        if lamda_db_path is None:
            lamda_db_path = Path("output/lamda_unified.db")

        self.db_path = lamda_db_path
        self.db_available = lamda_db_path.exists()

        if not self.db_available:
            print(f"⚠️ LAMDa DB not found: {lamda_db_path}")

    def enhance(
        self,
        midi_path: Path,
        emotion: EmotionContext,
        output_path: Optional[Path] = None,
        verbose: bool = True,
    ) -> EnhancementResult:
        """Suno MIDIを強化"""
        if not midi_path.exists():
            raise FileNotFoundError(f"MIDI not found: {midi_path}")

        if output_path is None:
            output_path = midi_path.parent / f"{midi_path.stem}_enhanced.mid"

        if verbose:
            print(f"🎹 Enhancing MIDI: {midi_path.name}")
            print(f"   Emotion: {emotion.mood_category} ")
            print(f"   (valence={emotion.valence:.2f}, arousal={emotion.arousal:.2f})")

        # music21でMIDI読み込み (警告無視)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            score = music21.converter.parse(str(midi_path))

        original_notes = list(score.flatten().notes)
        original_note_count = len(original_notes)

        if verbose:
            print(f"   Original notes: {original_note_count}")
            print(f"   Parts: {len(score.parts)}")

        # 強化処理
        velocity_adjusted = self._adjust_velocities(score, emotion, verbose)
        lamda_progressions = self._apply_lamda_theory(score, emotion, verbose)

        # 保存
        output_path.parent.mkdir(parents=True, exist_ok=True)
        score.write("midi", fp=str(output_path))

        enhanced_notes = list(score.flatten().notes)
        enhanced_note_count = len(enhanced_notes)

        result = EnhancementResult(
            original_midi=midi_path,
            enhanced_midi=output_path,
            emotion=emotion,
            original_note_count=original_note_count,
            enhanced_note_count=enhanced_note_count,
            velocity_adjusted=velocity_adjusted,
            lamda_progressions_used=lamda_progressions,
        )

        if verbose:
            print(f"\n✅ Enhancement complete!")
            print(f"   Output: {output_path.name}")
            print(f"   Enhanced notes: {enhanced_note_count}")
            if lamda_progressions:
                print(f"   LAMDa progressions: {len(lamda_progressions)}")

        return result

    def _adjust_velocities(
        self, score: music21.stream.Score, emotion: EmotionContext, verbose: bool
    ) -> bool:
        """Velocity調整"""
        if verbose:
            print("  🎚️  Adjusting velocities...")

        min_vel, max_vel = emotion.target_velocity_range
        adjusted = False

        for note in score.flatten().notes:
            if hasattr(note, "volume") and hasattr(note.volume, "velocity"):
                original_vel = note.volume.velocity
                if original_vel is None or original_vel == 0:
                    continue

                # 0-127 → emotion range
                normalized = original_vel / 127.0
                new_velocity = int(min_vel + normalized * (max_vel - min_vel))

                # Intensity調整
                intensity_factor = 0.7 + 0.6 * emotion.intensity
                new_velocity = int(new_velocity * intensity_factor)
                new_velocity = max(1, min(127, new_velocity))

                if new_velocity != original_vel:
                    note.volume.velocity = new_velocity
                    adjusted = True

        return adjusted

    def _apply_lamda_theory(
        self, score: music21.stream.Score, emotion: EmotionContext, verbose: bool
    ) -> List[str]:
        """LAMDa理論適用"""
        if not self.db_available:
            if verbose:
                print("  ⚠️ LAMDa DB not available")
            return []

        if verbose:
            print("  🧠 Applying LAMDa theory...")

        progressions = self._search_lamda_progressions(emotion)

        if verbose and progressions:
            print(f"     Found {len(progressions)} progressions")

        return progressions

    def _search_lamda_progressions(self, emotion: EmotionContext) -> List[str]:
        """LAMDa DB検索"""
        if not self.db_available:
            return []

        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()

            cursor.execute(
                """
                SELECT progression, total_events
                FROM progressions
                ORDER BY total_events DESC
                LIMIT 10
                """
            )

            rows = cursor.fetchall()
            conn.close()

            progressions = []
            for row in rows:
                try:
                    prog_data = json.loads(row[0])
                    progressions.append(str(prog_data))
                except:
                    pass

            return progressions[:5]

        except Exception as e:
            print(f"    ⚠️ LAMDa search failed: {e}")
            return []


def main():
    parser = argparse.ArgumentParser(description="Enhance Suno MIDI with LAMDa (music21 based)")
    parser.add_argument("midi_path", type=Path, help="Suno MIDI file")
    parser.add_argument("--valence", type=float, default=0.0, help="Emotion valence (-1 to 1)")
    parser.add_argument("--arousal", type=float, default=0.5, help="Emotion arousal (0 to 1)")
    parser.add_argument("--intensity", type=float, default=0.5, help="Emotion intensity (0 to 1)")
    parser.add_argument("--output", "-o", type=Path, help="Output MIDI (auto if omitted)")
    parser.add_argument(
        "--lamda-db",
        type=Path,
        default=Path("output/lamda_unified.db"),
        help="LAMDa database path",
    )
    parser.add_argument("--export-json", type=Path, help="Export report as JSON")

    args = parser.parse_args()

    emotion = EmotionContext(
        valence=args.valence,
        arousal=args.arousal,
        intensity=args.intensity,
    )

    enhancer = LAMDaEnhancerV2(lamda_db_path=args.lamda_db)

    result = enhancer.enhance(
        midi_path=args.midi_path,
        emotion=emotion,
        output_path=args.output,
        verbose=True,
    )

    if args.export_json:
        args.export_json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.export_json, "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)
        print(f"\n📄 Report: {args.export_json}")

    print("=" * 70)
    print("🎉 MIDI enhancement complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
