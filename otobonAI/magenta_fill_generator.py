#!/usr/bin/env python3
"""
Magenta Fill Generator for Phase 4 — 装飾レイヤ

Generates fills, arpeggios, and embellishments using Magenta models
while preserving the V2 arrangement backbone.

Usage:
    from otobonAI.magenta_fill_generator import MagentaFillGenerator

    gen = MagentaFillGenerator(model_path="models/groovae_2bar_humanize.ckpt")
    fills = gen.generate_fills(
        section="chorus",
        bars=[16, 17],
        chordmap_locked=chords,
        guide_tone_hints=hints,
        policy={"max_events": 32, "temperature": 0.8}
    )
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:
    import note_seq
    from note_seq.protobuf import music_pb2
except ImportError:
    note_seq = None
    music_pb2 = None
    logging.warning("note-seq not installed - Magenta features disabled")


@dataclass
class FillConfig:
    """Configuration for Magenta fill generation."""

    temperature: float = 0.8
    max_events: int = 32
    min_pitch: int = 36
    max_pitch: int = 96
    use_guide_tones: bool = True
    rhythm_density: float = 0.7  # 0.0-1.0


@dataclass
class MagentaFill:
    """A single Magenta-generated fill."""

    bar_start: int
    bar_end: int
    events: list[dict[str, Any]]
    source: str = "magenta"
    confidence: float = 0.5


class MagentaFillGenerator:
    """Generate fills and embellishments using Magenta models."""

    def __init__(
        self,
        model_path: Path | None = None,
        enable_cache: bool = True,
    ):
        """Initialize Magenta fill generator.

        Args:
            model_path: Path to Magenta checkpoint (optional for prototype)
            enable_cache: Cache generated fills for reuse
        """
        self.model_path = model_path
        self.enable_cache = enable_cache
        self.cache: dict[str, MagentaFill] = {}

        if note_seq is None:
            logging.warning("Magenta disabled - install note-seq to enable")
            self.enabled = False
        else:
            self.enabled = True
            logging.info("Magenta fill generator initialized")

    def generate_fills(
        self,
        section: str,
        bars: list[int],
        chordmap_locked: dict[str, Any],
        guide_tone_hints: dict[str, Any] | None = None,
        policy: dict[str, Any] | None = None,
    ) -> list[MagentaFill]:
        """Generate fills for specified bars.

        Args:
            section: Section label (e.g., "chorus", "verse")
            bars: List of bar indices to fill
            chordmap_locked: Locked chord progression
            guide_tone_hints: Optional guide tone suggestions
            policy: Generation policy (temperature, max_events, etc.)

        Returns:
            List of generated fills
        """
        if not self.enabled:
            return []

        config = self._parse_policy(policy)
        fills = []

        for bar in bars:
            cache_key = f"{section}:{bar}"
            if self.enable_cache and cache_key in self.cache:
                fills.append(self.cache[cache_key])
                continue

            fill = self._generate_single_fill(bar, chordmap_locked, guide_tone_hints, config)

            if self.enable_cache:
                self.cache[cache_key] = fill
            fills.append(fill)

        return fills

    def _parse_policy(self, policy: dict[str, Any] | None) -> FillConfig:
        """Parse policy dict into FillConfig."""
        if policy is None:
            return FillConfig()

        return FillConfig(
            temperature=policy.get("temperature", 0.8),
            max_events=policy.get("max_events", 32),
            min_pitch=policy.get("min_pitch", 36),
            max_pitch=policy.get("max_pitch", 96),
            use_guide_tones=policy.get("use_guide_tones", True),
            rhythm_density=policy.get("rhythm_density", 0.7),
        )

    def _generate_single_fill(
        self,
        bar: int,
        chordmap: dict[str, Any],
        guide_hints: dict[str, Any] | None,
        config: FillConfig,
    ) -> MagentaFill:
        """Generate a single fill for one bar.

        Note: This is a prototype implementation.
        Full implementation would use Magenta MusicVAE checkpoint.
        """
        # Extract chord for this bar
        bar_chord = self._get_bar_chord(bar, chordmap)

        # Get guide tones if available
        guide_pitches = []
        if config.use_guide_tones and guide_hints:
            guide_pitches = self._extract_guide_pitches(bar, guide_hints)

        # Generate events (prototype: arpeggio pattern)
        events = self._generate_arpeggio(bar_chord, guide_pitches, config)

        return MagentaFill(
            bar_start=bar,
            bar_end=bar + 1,
            events=events,
            source="magenta_prototype",
            confidence=0.5,
        )

    def _get_bar_chord(self, bar: int, chordmap: dict[str, Any]) -> str:
        """Extract chord symbol for a bar."""
        events = chordmap.get("events", [])
        for event in events:
            if event.get("bar") == bar:
                return event.get("symbol", "C")
        return "C"

    def _extract_guide_pitches(self, bar: int, guide_hints: dict[str, Any]) -> list[int]:
        """Extract guide tone pitches for a bar."""
        pitches = []
        events = guide_hints.get("events", [])
        for event in events:
            if event.get("bar") == bar:
                pitches.extend(event.get("pitches", []))
        return sorted(set(pitches))

    def _generate_arpeggio(
        self,
        chord_symbol: str,
        guide_pitches: list[int],
        config: FillConfig,
    ) -> list[dict[str, Any]]:
        """Generate arpeggio pattern (prototype).

        Full implementation would use Magenta MusicVAE.
        """
        # Simple arpeggio: root, 3rd, 5th, 7th
        root = self._chord_to_root_pitch(chord_symbol, config.min_pitch)
        intervals = [0, 4, 7, 11]  # Major 7th chord

        events = []
        step_duration = 0.25  # 16th notes

        for i, interval in enumerate(intervals):
            if len(events) >= config.max_events:
                break

            pitch = root + interval
            if pitch > config.max_pitch:
                pitch -= 12

            events.append(
                {
                    "pitch": pitch,
                    "start_ql": i * step_duration,
                    "duration_ql": step_duration * 0.9,
                    "velocity": int(80 + np.random.randn() * 10 * config.temperature),
                    "source": "magenta_arpeggio",
                }
            )

        return events

    def _chord_to_root_pitch(self, chord_symbol: str, min_pitch: int) -> int:
        """Convert chord symbol to root pitch."""
        # Simple mapping (extend for full chord vocabulary)
        root_map = {
            "C": 60,
            "D": 62,
            "E": 64,
            "F": 65,
            "G": 67,
            "A": 69,
            "B": 71,
        }
        root_note = chord_symbol[0].upper()
        pitch = root_map.get(root_note, 60)

        # Adjust to min_pitch octave
        while pitch < min_pitch:
            pitch += 12

        return pitch

    def to_json(self, fills: list[MagentaFill], output_path: Path) -> None:
        """Save fills to JSON file."""
        data = {
            "fills": [
                {
                    "bar_start": f.bar_start,
                    "bar_end": f.bar_end,
                    "events": f.events,
                    "source": f.source,
                    "confidence": f.confidence,
                }
                for f in fills
            ],
            "total_fills": len(fills),
            "total_events": sum(len(f.events) for f in fills),
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        logging.info(f"Saved {len(fills)} fills to {output_path}")


def main():
    """Example usage."""
    import sys

    if len(sys.argv) < 2:
        print("Usage: python magenta_fill_generator.py <chordmap.json>")
        sys.exit(1)

    # Load chordmap
    chordmap_path = Path(sys.argv[1])
    with open(chordmap_path) as f:
        chordmap = json.load(f)

    # Generate fills
    gen = MagentaFillGenerator()
    fills = gen.generate_fills(
        section="chorus",
        bars=[16, 17, 18, 19],
        chordmap_locked=chordmap,
        policy={"temperature": 0.8, "max_events": 24},
    )

    # Save output
    output_path = Path("magenta_fills.json")
    gen.to_json(fills, output_path)

    print(f"\n✅ Generated {len(fills)} fills")
    print(f"   Total events: {sum(len(f.events) for f in fills)}")
    print(f"   Output: {output_path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
