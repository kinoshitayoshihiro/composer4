#!/usr/bin/env python3
"""
Magenta Fill Generator for Phase 4 — 装飾レイヤ

Generates fills, arpeggios, and embellishments using Magenta models
while preserving the V2 arrangement backbone.

Usage:
    from otobonAI.magenta_fill_generator import MagentaFillGenerator

    import yaml

    policy = yaml.safe_load(open("config/magenta_policy.yaml"))
    gen = MagentaFillGenerator.from_policy(policy)
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
from typing import Any, Sequence

import numpy as np

try:
    import note_seq
    from note_seq.protobuf import music_pb2
except ImportError:
    note_seq = None
    music_pb2 = None
    logging.warning("note-seq not installed - Magenta features disabled")

try:
    from magenta.models.music_vae import configs as music_vae_configs
    from magenta.models.music_vae.trained_model import TrainedModel
except ImportError:  # pragma: no cover - optional dependency
    music_vae_configs = None
    TrainedModel = None


@dataclass
class FillConfig:
    """Configuration for Magenta fill generation."""

    temperature: float = 0.8
    max_events: int = 32
    min_pitch: int = 36
    max_pitch: int = 96
    use_guide_tones: bool = True
    rhythm_density: float = 0.7  # 0.0-1.0
    min_velocity: int = 40
    max_velocity: int = 110
    max_pitch_deviation: int = 12


@dataclass
class MagentaFill:
    """A single Magenta-generated fill."""

    bar_start: int
    bar_end: int
    events: list[dict[str, Any]]
    source: str = "magenta"
    confidence: float = 0.5


@dataclass
class GeneratorSettings:
    """Runtime settings for MagentaFillGenerator."""

    backend: str = "prototype"  # prototype | music_vae
    config_name: str | None = None
    checkpoint: Path | None = None
    steps_per_quarter: int = 4
    bars_per_sample: int = 1

    @staticmethod
    def _safe_int(value: Any, default: int) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    @classmethod
    def from_policy(cls, policy: dict[str, Any]) -> GeneratorSettings:
        generator_cfg = policy.get("generator", {}) or {}
        checkpoints = policy.get("checkpoints", {}) or {}

        backend = str(generator_cfg.get("backend", "prototype")).strip().lower()

        checkpoint_value = generator_cfg.get("checkpoint")
        checkpoint_key = generator_cfg.get("checkpoint_key")
        if not checkpoint_value and checkpoint_key:
            checkpoint_value = checkpoints.get(checkpoint_key)

        checkpoint_path = Path(checkpoint_value).expanduser() if checkpoint_value else None

        config_name = generator_cfg.get("config_name") or generator_cfg.get("music_vae_config")
        if backend == "music_vae" and not config_name:
            config_name = "cat-mel_2bar_small"

        steps_per_quarter = cls._safe_int(generator_cfg.get("steps_per_quarter", 4), 4)
        bars_per_sample = max(1, cls._safe_int(generator_cfg.get("bars_per_sample", 1), 1))

        return cls(
            backend=backend or "prototype",
            config_name=config_name,
            checkpoint=checkpoint_path,
            steps_per_quarter=max(1, steps_per_quarter),
            bars_per_sample=bars_per_sample,
        )


class MagentaFillGenerator:
    """Generate fills and embellishments using Magenta models."""

    def __init__(
        self,
        model_path: Path | str | None = None,
        enable_cache: bool = True,
        settings: GeneratorSettings | None = None,
    ):
        """Initialize Magenta fill generator.

        Args:
            model_path: Path to Magenta checkpoint (optional for prototype)
            enable_cache: Cache generated fills for reuse
        """
        resolved_model_path: Path | None = None
        if isinstance(model_path, str):
            resolved_model_path = Path(model_path).expanduser()
        else:
            resolved_model_path = model_path

        self.settings = settings or GeneratorSettings()
        if resolved_model_path and self.settings.checkpoint is None:
            self.settings.checkpoint = resolved_model_path

        self.enable_cache = enable_cache
        self.cache: dict[str, MagentaFill] = {}
        self.backend = (self.settings.backend or "prototype").lower()
        self._music_vae_model: TrainedModel | None = None

        if self.backend == "music_vae":
            self._init_music_vae_backend()

    @classmethod
    def from_policy(
        cls,
        policy: dict[str, Any],
        enable_cache: bool = True,
    ) -> "MagentaFillGenerator":
        settings = GeneratorSettings.from_policy(policy)
        return cls(model_path=settings.checkpoint, enable_cache=enable_cache, settings=settings)

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
        config = self._parse_policy(policy)
        fills = []

        for bar in bars:
            cache_key = f"{section}:{bar}"
            if self.enable_cache and cache_key in self.cache:
                fills.append(self.cache[cache_key])
                continue

            fill = self._generate_music_vae_fill(bar, chordmap_locked, guide_tone_hints, config)

            if self.enable_cache:
                self.cache[cache_key] = fill
            fills.append(fill)

        return fills

    def _parse_policy(self, policy: dict[str, Any] | None) -> FillConfig:
        """Parse policy dict into FillConfig."""
        if policy is None:
            return FillConfig()

        def _safe_int(value: Any, default: int) -> int:
            try:
                return int(value)
            except (TypeError, ValueError):
                return default

        return FillConfig(
            temperature=policy.get("temperature", 0.8),
            max_events=policy.get("max_events", 32),
            min_pitch=policy.get("min_pitch", 36),
            max_pitch=policy.get("max_pitch", 96),
            use_guide_tones=policy.get("use_guide_tones", True),
            rhythm_density=policy.get("rhythm_density", 0.7),
            min_velocity=_safe_int(policy.get("min_velocity", 40), 40),
            max_velocity=_safe_int(policy.get("max_velocity", 110), 110),
            max_pitch_deviation=_safe_int(policy.get("max_pitch_deviation", 12), 12),
        )

    def _generate_prototype_fill(
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

        events = self._clamp_event_velocities(
            events,
            min_velocity=config.min_velocity,
            max_velocity=config.max_velocity,
        )

        return MagentaFill(
            bar_start=bar,
            bar_end=bar + 1,
            events=events,
            source="magenta_prototype",
            confidence=0.5,
        )

    def _generate_music_vae_fill(
        self,
        bar: int,
        chordmap: dict[str, Any],
        guide_hints: dict[str, Any] | None,
        config: FillConfig,
    ) -> MagentaFill:
        """Generate fills using MusicVAE backend when available."""

        if self.backend != "music_vae" or self._music_vae_model is None or note_seq is None:
            return self._generate_prototype_fill(bar, chordmap, guide_hints, config)

        total_steps = self.settings.steps_per_quarter * 4 * self.settings.bars_per_sample
        try:
            sequences = self._music_vae_model.sample(
                n=1,
                length=total_steps,
                temperature=config.temperature,
            )
        except Exception as exc:  # pragma: no cover - runtime guard
            logging.warning("MusicVAE sampling failed (%s) - falling back to prototype", exc)
            return self._generate_prototype_fill(bar, chordmap, guide_hints, config)

        if not sequences:
            logging.warning("MusicVAE returned no sequences - falling back to prototype")
            return self._generate_prototype_fill(bar, chordmap, guide_hints, config)

        events = self._quantize_sequence_to_events(
            sequence=sequences[0],
            steps_per_quarter=self.settings.steps_per_quarter,
            max_events=config.max_events,
        )

        if not events:
            logging.warning("MusicVAE sequence empty - falling back to prototype")
            return self._generate_prototype_fill(bar, chordmap, guide_hints, config)

        guide_pitches = []
        if config.use_guide_tones and guide_hints:
            guide_pitches = self._extract_guide_pitches(bar, guide_hints)

        events = self._transpose_events_to_target(
            events,
            target_pitch=self._determine_target_pitch(chordmap, bar, guide_pitches, config),
            min_pitch=config.min_pitch,
            max_pitch=config.max_pitch,
            max_deviation=config.max_pitch_deviation,
        )

        events = self._clamp_event_velocities(
            events,
            min_velocity=config.min_velocity,
            max_velocity=config.max_velocity,
        )

        return MagentaFill(
            bar_start=bar,
            bar_end=bar + self.settings.bars_per_sample,
            events=events,
            source="magenta_music_vae",
            confidence=0.8,
        )

    def _clamp_event_velocities(
        self,
        events: Sequence[dict[str, Any]] | None,
        min_velocity: int,
        max_velocity: int,
    ) -> list[dict[str, Any]]:
        if not events:
            return []

        min_vel = max(1, int(min_velocity))
        max_vel = max(min_vel, int(max_velocity))

        clamped: list[dict[str, Any]] = []
        for event in events:
            velocity = int(event.get("velocity", min_vel))
            velocity = max(min_vel, min(max_vel, velocity))
            clamped.append({**event, "velocity": velocity})

        return clamped

    def _init_music_vae_backend(self) -> None:
        """Initialize MusicVAE backend if dependencies exist."""

        if TrainedModel is None or music_vae_configs is None:
            logging.warning("magenta.models.music_vae not available - using prototype backend")
            self.backend = "prototype"
            return

        if note_seq is None:
            logging.warning("note-seq not installed - using prototype backend")
            self.backend = "prototype"
            return

        checkpoint = self.settings.checkpoint
        if not checkpoint or not Path(checkpoint).expanduser().exists():
            logging.warning(
                "Magenta checkpoint missing at %s - using prototype backend", checkpoint
            )
            self.backend = "prototype"
            return

        config_name = self.settings.config_name
        config = music_vae_configs.CONFIG_MAP.get(config_name) if config_name else None
        if config is None:
            logging.warning("Unknown MusicVAE config '%s' - using prototype backend", config_name)
            self.backend = "prototype"
            return

        self._music_vae_model = TrainedModel(
            config,
            batch_size=1,
            checkpoint_dir_or_path=str(checkpoint),
        )
        logging.info("Loaded Magenta MusicVAE backend (%s, %s)", checkpoint, config_name)

    def _quantize_sequence_to_events(
        self,
        sequence: "note_seq.NoteSequence",
        steps_per_quarter: int,
        max_events: int,
    ) -> list[dict[str, Any]]:
        """Convert a NoteSequence to Magenta events list."""

        quantized = note_seq.quantize_note_sequence(sequence, steps_per_quarter=steps_per_quarter)
        events: list[dict[str, Any]] = []

        for note in sorted(quantized.notes, key=lambda n: (n.quantized_start_step, n.pitch)):
            duration_steps = max(1, note.quantized_end_step - note.quantized_start_step)
            events.append(
                {
                    "pitch": int(note.pitch),
                    "start_ql": note.quantized_start_step / steps_per_quarter,
                    "duration_ql": duration_steps / steps_per_quarter,
                    "velocity": int(max(1, min(127, note.velocity))),
                    "source": "magenta_music_vae",
                }
            )

            if len(events) >= max_events:
                break

        return events

    def _transpose_events_to_target(
        self,
        events: Sequence[dict[str, Any]],
        target_pitch: int,
        min_pitch: int,
        max_pitch: int,
        max_deviation: int | None = None,
    ) -> list[dict[str, Any]]:
        if not events:
            return []

        current_median = int(np.median([event.get("pitch", target_pitch) for event in events]))
        shift = target_pitch - current_median
        deviation_limit = None
        if max_deviation is not None:
            try:
                deviation_limit = abs(int(max_deviation))
            except (TypeError, ValueError):  # pragma: no cover - defensive
                deviation_limit = None

        adjusted: list[dict[str, Any]] = []
        for event in events:
            pitch = int(event.get("pitch", target_pitch)) + shift
            while pitch < min_pitch:
                pitch += 12
            while pitch > max_pitch:
                pitch -= 12

            if deviation_limit is not None:
                lower_bound = target_pitch - deviation_limit
                upper_bound = target_pitch + deviation_limit
                if pitch < lower_bound:
                    pitch = lower_bound
                elif pitch > upper_bound:
                    pitch = upper_bound

            adjusted.append({**event, "pitch": pitch})

        return adjusted

    def _determine_target_pitch(
        self,
        chordmap: dict[str, Any],
        bar: int,
        guide_pitches: Sequence[int],
        config: FillConfig,
    ) -> int:
        if guide_pitches:
            return int(np.median(guide_pitches))

        root = self._chord_to_root_pitch(self._get_bar_chord(bar, chordmap), config.min_pitch)
        return root + 12  # favor upper octave for fills

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
