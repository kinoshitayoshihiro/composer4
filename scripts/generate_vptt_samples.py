#!/usr/bin/env python3
"""
VPTT 50-Sample Orthogonal Design Generator

Generates 50 performance technique samples with orthogonal design:
- 2 instruments (piano, violin)
- 3 techniques (staccato, legato, pizzicato)
- 3 tempos (slow=60, medium=120, fast=180 BPM)
- 3 dynamics (soft=pp, medium=mf, loud=ff)

Total combinations: 2 × 3 × 3 × 3 = 54 → Sample 50

Output:
- MIDI files with technique annotations
- Metadata YAML with VPTT labels
- Ready for Stage3 training

Usage:
    python scripts/generate_vptt_samples.py \
        --output-dir data/vptt_samples \
        --seed 42
"""

import argparse
import itertools
import random
from pathlib import Path
from typing import Dict, List, Tuple

import mido
import yaml


class VPTTSampleGenerator:
    """VPTT orthogonal design sample generator."""

    # Orthogonal design parameters
    INSTRUMENTS = ["piano", "violin"]
    TECHNIQUES = {
        "piano": ["staccato", "legato", "sustain"],
        "violin": ["staccato", "legato", "pizzicato"],
    }
    TEMPOS = {
        "slow": 60,
        "medium": 120,
        "fast": 180,
    }
    DYNAMICS = {
        "soft": {"velocity": 45, "marking": "pp"},
        "medium": {"velocity": 80, "marking": "mf"},
        "loud": {"velocity": 110, "marking": "ff"},
    }

    # MIDI program numbers
    PROGRAMS = {
        "piano": 0,  # Acoustic Grand Piano
        "violin": 40,  # Violin
    }

    def __init__(self, seed: int = 42):
        """Initialize generator with random seed."""
        random.seed(seed)
        self.combinations = self._generate_combinations()

    def _generate_combinations(self) -> List[Dict]:
        """Generate all orthogonal combinations."""
        combinations = []
        combo_id = 0

        for instrument in self.INSTRUMENTS:
            techniques = self.TECHNIQUES[instrument]
            for technique, tempo_name, dynamic_name in itertools.product(
                techniques, self.TEMPOS.keys(), self.DYNAMICS.keys()
            ):
                combinations.append(
                    {
                        "id": f"vptt_{combo_id:03d}",
                        "instrument": instrument,
                        "technique": technique,
                        "tempo": tempo_name,
                        "tempo_bpm": self.TEMPOS[tempo_name],
                        "dynamic": dynamic_name,
                        "velocity": self.DYNAMICS[dynamic_name]["velocity"],
                        "marking": self.DYNAMICS[dynamic_name]["marking"],
                    }
                )
                combo_id += 1

        return combinations

    def sample_combinations(self, n: int = 50) -> List[Dict]:
        """Sample N combinations randomly."""
        if n > len(self.combinations):
            raise ValueError(
                f"Requested {n} samples but only {len(self.combinations)} combinations available"
            )
        return random.sample(self.combinations, n)

    def generate_midi(self, spec: Dict) -> mido.MidiFile:
        """
        Generate simple MIDI file for given spec.

        Args:
            spec: Combination dict with instrument, technique, tempo, velocity

        Returns:
            MidiFile with appropriate annotations
        """
        mid = mido.MidiFile(type=1)
        track = mido.MidiTrack()
        mid.tracks.append(track)

        # Set tempo
        tempo = mido.bpm2tempo(spec["tempo_bpm"])
        track.append(mido.MetaMessage("set_tempo", tempo=tempo, time=0))

        # Set program
        program = self.PROGRAMS[spec["instrument"]]
        track.append(mido.Message("program_change", program=program, time=0))

        # Generate simple phrase (4 bars, 4/4)
        ticks_per_beat = mid.ticks_per_beat
        phrase = self._generate_phrase(spec, ticks_per_beat)

        # Add notes
        for note_data in phrase:
            pitch, velocity, duration = note_data

            # Note on
            track.append(
                mido.Message(
                    "note_on",
                    note=pitch,
                    velocity=velocity,
                    time=0,
                )
            )

            # Note off
            track.append(
                mido.Message(
                    "note_off",
                    note=pitch,
                    velocity=0,
                    time=duration,
                )
            )

        # End of track
        track.append(mido.MetaMessage("end_of_track", time=0))

        return mid

    def _generate_phrase(self, spec: Dict, ticks_per_beat: int) -> List[Tuple]:
        """
        Generate simple musical phrase based on technique.

        Returns:
            List of (pitch, velocity, duration_ticks) tuples
        """
        technique = spec["technique"]
        velocity = spec["velocity"]
        instrument = spec["instrument"]

        # Base pitches (C major scale)
        if instrument == "piano":
            pitches = [60, 62, 64, 65, 67, 69, 71, 72]  # C4 to C5
        else:  # violin
            pitches = [55, 57, 59, 60, 62, 64, 66, 67]  # G3 to G4

        phrase = []

        if technique == "staccato":
            # Short detached notes (1/8 note duration, 1/4 note spacing)
            for i in range(8):
                pitch = pitches[i % len(pitches)]
                duration = ticks_per_beat // 2  # 1/8 note
                phrase.append((pitch, velocity, duration))

        elif technique == "legato":
            # Smooth connected notes (1/4 note duration)
            for i in range(8):
                pitch = pitches[i % len(pitches)]
                duration = ticks_per_beat  # 1/4 note
                phrase.append((pitch, velocity, duration))

        elif technique == "pizzicato":
            # Short plucked notes (violin only)
            for i in range(8):
                pitch = pitches[i % len(pitches)]
                duration = ticks_per_beat // 4  # 1/16 note
                # Pizzicato has sharper attack
                vel = min(127, velocity + 10)
                phrase.append((pitch, vel, duration))

        elif technique == "sustain":
            # Long sustained notes (piano only)
            for i in range(4):
                pitch = pitches[i * 2 % len(pitches)]
                duration = ticks_per_beat * 2  # half note
                phrase.append((pitch, velocity, duration))

        return phrase

    def generate_metadata(self, specs: List[Dict]) -> Dict:
        """
        Generate VPTT metadata YAML.

        Args:
            specs: List of combination dicts

        Returns:
            Metadata dict ready for YAML export
        """
        metadata = {
            "dataset": "VPTT-50",
            "description": "50 performance technique samples with orthogonal design",
            "total_samples": len(specs),
            "design": {
                "instruments": self.INSTRUMENTS,
                "techniques": self.TECHNIQUES,
                "tempos": list(self.TEMPOS.keys()),
                "dynamics": list(self.DYNAMICS.keys()),
            },
            "samples": [],
        }

        for spec in specs:
            sample_entry = {
                "id": spec["id"],
                "file": f"{spec['id']}.mid",
                "instrument": spec["instrument"],
                "technique": spec["technique"],
                "tempo": spec["tempo"],
                "tempo_bpm": spec["tempo_bpm"],
                "dynamic": spec["dynamic"],
                "velocity": spec["velocity"],
                "marking": spec["marking"],
            }
            metadata["samples"].append(sample_entry)

        return metadata


def main():
    parser = argparse.ArgumentParser(
        description="Generate VPTT 50 orthogonal design samples"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/vptt_samples"),
        help="Output directory for MIDI files and metadata",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=50,
        help="Number of samples to generate (max 54)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print combinations without generating files",
    )

    args = parser.parse_args()

    # Initialize generator
    print(f"Initializing VPTT generator with seed {args.seed}")
    generator = VPTTSampleGenerator(seed=args.seed)

    # Sample combinations
    print(
        f"Sampling {args.num_samples} combinations from {len(generator.combinations)} total"
    )
    sampled = generator.sample_combinations(args.num_samples)

    if args.dry_run:
        print("\nDry run - sample combinations:")
        for spec in sampled[:5]:
            print(
                f"  {spec['id']}: {spec['instrument']} {spec['technique']} "
                f"@ {spec['tempo_bpm']}bpm {spec['marking']}"
            )
        print(f"  ... and {len(sampled) - 5} more")
        return

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    midi_dir = args.output_dir / "midi"
    midi_dir.mkdir(exist_ok=True)

    # Generate MIDI files
    print(f"\nGenerating MIDI files to {midi_dir}")
    for i, spec in enumerate(sampled):
        midi = generator.generate_midi(spec)
        output_path = midi_dir / f"{spec['id']}.mid"
        midi.save(output_path)

        if (i + 1) % 10 == 0:
            print(f"  Generated {i+1}/{len(sampled)} files...")

    # Generate metadata
    metadata = generator.generate_metadata(sampled)
    metadata_path = args.output_dir / "vptt_metadata.yaml"
    with open(metadata_path, "w", encoding="utf-8") as f:
        yaml.dump(metadata, f, allow_unicode=True, sort_keys=False)

    print(f"\n✅ Generated {len(sampled)} MIDI files")
    print(f"📊 Saved metadata to {metadata_path}")

    # Print statistics
    stats = {
        "instruments": {},
        "techniques": {},
        "tempos": {},
        "dynamics": {},
    }
    for spec in sampled:
        stats["instruments"][spec["instrument"]] = (
            stats["instruments"].get(spec["instrument"], 0) + 1
        )
        stats["techniques"][spec["technique"]] = (
            stats["techniques"].get(spec["technique"], 0) + 1
        )
        stats["tempos"][spec["tempo"]] = stats["tempos"].get(spec["tempo"], 0) + 1
        stats["dynamics"][spec["dynamic"]] = (
            stats["dynamics"].get(spec["dynamic"], 0) + 1
        )

    print("\n📈 Distribution:")
    for category, counts in stats.items():
        print(f"  {category.capitalize()}:")
        for key, count in sorted(counts.items()):
            print(f"    {key}: {count}")


if __name__ == "__main__":
    main()
