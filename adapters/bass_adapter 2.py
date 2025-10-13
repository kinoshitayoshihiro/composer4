#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Bass adapter using BaseInstrumentAdapter.

Minimal pattern generator for bass lines:
- Downbeat-anchored root notes
- Optional passing tones
- E1-G3 range (MIDI 28-55)
"""
from __future__ import annotations
import random
from typing import Any
from typing import Dict

try:
    import pretty_midi
except Exception as e:
    raise RuntimeError("pretty_midi required: pip install pretty_midi") from e

from .base_instrument_adapter import BaseInstrumentAdapter


class BassAdapter(BaseInstrumentAdapter):
    """
    Minimal bass generator using BaseInstrumentAdapter infrastructure.

    Conditions dict keys:
      - tempo: int (default 120)
      - time_sig: str (default "4/4")
      - length_bars: int (default 16)
      - style: str (default "default")
      - density: str ("low"|"mid"|"high", default "mid")

    Generated pattern:
      - Downbeat anchor (C2/E2/G2 root notes)
      - Optional passing tones for mid/high density
      - All notes in E1-G3 range
    """
    part_name = "bass"
    default_time_sig = "4/4"

    # Bass range (E1-G3)
    BASS_RANGE_MIN = 28  # E1
    BASS_RANGE_MAX = 55  # G3

    # Common root notes (in range)
    ROOT_NOTES = [36, 40, 43]  # C2, E2, G2

    def _build_pretty_midi(self, conditions: Dict[str, Any], seed: int) -> "pretty_midi.PrettyMIDI":
        """Build bass MIDI from conditions."""
        rng = random.Random(seed)

        tempo = int(conditions.get("tempo", 120))
        time_sig = conditions.get("time_sig", "4/4")
        bars = int(conditions.get("length_bars", 16))
        density = conditions.get("density", "mid")

        # Parse time signature
        num, den = 4, 4
        try:
            parts = time_sig.split("/")
            num, den = int(parts[0]), int(parts[1])
        except Exception:
            pass

        # Calculate bar length in seconds
        beat_duration = 60.0 / tempo
        bar_length = num * beat_duration * (4.0 / den)

        # Create PrettyMIDI object
        pm = pretty_midi.PrettyMIDI(initial_tempo=tempo)
        bass_instrument = pretty_midi.Instrument(program=33, is_drum=False, name="Bass")

        # Density parameters
        density_map = {
            "low": {"notes_per_bar": 1, "passing_prob": 0.0},
            "mid": {"notes_per_bar": 2, "passing_prob": 0.3},
            "high": {"notes_per_bar": 4, "passing_prob": 0.5},
        }
        params = density_map.get(density, density_map["mid"])
        notes_per_bar = params["notes_per_bar"]
        passing_prob = params["passing_prob"]

        # Generate notes for each bar
        for bar_idx in range(bars):
            bar_start = bar_idx * bar_length

            # Always place note on downbeat (root note)
            root = rng.choice(self.ROOT_NOTES)
            velocity = rng.randint(70, 100)
            duration = beat_duration * 0.8  # Slight gap

            bass_instrument.notes.append(
                pretty_midi.Note(
                    velocity=velocity,
                    pitch=root,
                    start=bar_start,
                    end=bar_start + duration,
                )
            )

            # Additional notes based on density
            if notes_per_bar > 1:
                step = bar_length / notes_per_bar
                for i in range(1, notes_per_bar):
                    # Passing tone or anchor
                    if rng.random() < passing_prob:
                        # Passing tone (nearby pitch)
                        offset = rng.choice([-2, -1, 1, 2])
                        pitch = max(self.BASS_RANGE_MIN, min(self.BASS_RANGE_MAX, root + offset))
                    else:
                        # Another root note
                        pitch = rng.choice(self.ROOT_NOTES)

                    velocity = rng.randint(60, 90)
                    start = bar_start + i * step
                    end = start + beat_duration * 0.6

                    bass_instrument.notes.append(
                        pretty_midi.Note(
                            velocity=velocity,
                            pitch=pitch,
                            start=start,
                            end=end,
                        )
                    )

        pm.instruments.append(bass_instrument)
        return pm


# Convenience function for standalone use
def generate_bass(
    *,
    tempo: int = 120,
    time_sig: str = "4/4",
    length_bars: int = 16,
    style: str = "default",
    density: str = "mid",
    seed: int = 42,
    out_dir: str = "output/bass",
) -> Dict[str, Any]:
    """Standalone bass generation function."""
    adapter = BassAdapter(out_dir=out_dir)
    conditions = {
        "tempo": tempo,
        "time_sig": time_sig,
        "length_bars": length_bars,
        "style": style,
        "density": density,
    }
    return adapter.generate_one(conditions=conditions, seed=seed, apply_humanizer=True, save=True)


if __name__ == "__main__":
    # Example usage
    result = generate_bass(tempo=120, length_bars=8, density="mid", seed=42)
    print(f"Generated: {result['midi_path']}")
    print(f"Token count: {len(result['tokens'])}")
