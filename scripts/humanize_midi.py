#!/usr/bin/env python3
"""
Lightweight MIDI Humanizer
Adds realistic velocity variation and timing jitter without external dependencies.

Stage3 v1.1 Quality Enhancement - Velocity/Timing improvement
Alternative to GrooVAE (which requires Python <3.11)

Usage:
    python scripts/humanize_midi.py input.mid output.mid --velocity-std 10 --timing-jitter 0.015
"""

import argparse
import random
from pathlib import Path
from typing import Optional

import pretty_midi
import numpy as np


class MIDIHumanizer:
    """Add human-like expression to quantized MIDI"""
    
    def __init__(
        self,
        velocity_std: float = 10.0,
        timing_jitter_seconds: float = 0.015,
        accent_strength: float = 1.3,
        seed: Optional[int] = None
    ):
        """
        Args:
            velocity_std: Standard deviation for velocity variation (0-127 scale)
            timing_jitter_seconds: Max timing deviation in seconds (±)
            accent_strength: Multiplier for strong beat emphasis
            seed: Random seed for reproducibility
        """
        self.velocity_std = velocity_std
        self.timing_jitter = timing_jitter_seconds
        self.accent_strength = accent_strength
        self._seed = seed  # Store for reproducibility
        
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
    
    def humanize(self, midi: pretty_midi.PrettyMIDI) -> pretty_midi.PrettyMIDI:
        """
        Apply humanization to all instruments in MIDI
        
        Args:
            midi: Input MIDI object
            
        Returns:
            Humanized MIDI object (new instance)
        """
        # Reset random seed for reproducibility
        if self.velocity_std > 0 or self.timing_jitter > 0:
            import random
            if hasattr(self, '_seed') and self._seed is not None:
                random.seed(self._seed)
                np.random.seed(self._seed)
        
        humanized = pretty_midi.PrettyMIDI()
        
        # Copy time signature and tempo changes
        humanized.time_signature_changes = midi.time_signature_changes.copy()
        humanized.time_to_tick = midi.time_to_tick
        
        # Process each instrument
        for instrument in midi.instruments:
            new_instrument = pretty_midi.Instrument(
                program=instrument.program,
                is_drum=instrument.is_drum,
                name=instrument.name
            )
            
            for note in instrument.notes:
                # Apply velocity variation
                new_velocity = self._humanize_velocity(
                    note.velocity,
                    note.start,
                    midi
                )
                
                # Apply timing jitter
                new_start = self._humanize_timing(note.start)
                new_end = note.end + (new_start - note.start)  # Preserve duration
                
                new_instrument.notes.append(
                    pretty_midi.Note(
                        velocity=new_velocity,
                        pitch=note.pitch,
                        start=max(0.0, new_start),
                        end=max(new_start + 0.01, new_end)  # Ensure positive duration
                    )
                )
            
            # Copy control changes and pitch bends
            new_instrument.control_changes = instrument.control_changes.copy()
            new_instrument.pitch_bends = instrument.pitch_bends.copy()
            
            humanized.instruments.append(new_instrument)
        
        return humanized
    
    def _humanize_velocity(
        self,
        original_velocity: int,
        note_time: float,
        midi: pretty_midi.PrettyMIDI
    ) -> int:
        """
        Add realistic velocity variation
        
        Strategy:
        1. Add Gaussian noise (std = velocity_std)
        2. Emphasize strong beats (downbeats, beat 1)
        3. Clamp to valid MIDI range [1, 127]
        """
        # Base variation
        noise = np.random.normal(0, self.velocity_std)
        new_velocity = original_velocity + noise
        
        # Accent on strong beats
        if self._is_strong_beat(note_time, midi):
            accent_boost = np.random.uniform(0, self.velocity_std * 0.5)
            new_velocity += accent_boost * self.accent_strength
        
        # Clamp to valid range
        return int(np.clip(new_velocity, 1, 127))
    
    def _humanize_timing(self, note_time: float) -> float:
        """
        Add subtle timing jitter
        
        Uses uniform distribution: ±timing_jitter
        """
        jitter = np.random.uniform(-self.timing_jitter, self.timing_jitter)
        return note_time + jitter
    
    def _is_strong_beat(
        self,
        time: float,
        midi: pretty_midi.PrettyMIDI,
        tolerance: float = 0.05
    ) -> bool:
        """
        Check if time is close to a strong beat (downbeat or beat 1)
        
        Args:
            time: Time in seconds
            midi: MIDI object with time signature info
            tolerance: Time window in seconds to consider "on beat"
        """
        # Get time signature at this point
        ts = midi.time_signature_changes[0] if midi.time_signature_changes else None
        if ts is None:
            # Assume 4/4 if no time signature
            beats_per_bar = 4
        else:
            beats_per_bar = ts.numerator
        
        # Get beat position
        beat = midi.time_to_tick(time) / midi.resolution
        beat_in_bar = beat % beats_per_bar
        
        # Check if close to beat 1 (downbeat)
        return abs(beat_in_bar) < tolerance or abs(beat_in_bar - beats_per_bar) < tolerance


def main():
    parser = argparse.ArgumentParser(description="Humanize MIDI files with velocity and timing variation")
    parser.add_argument("input_midi", type=Path, help="Input MIDI file")
    parser.add_argument("output_midi", type=Path, help="Output humanized MIDI file")
    parser.add_argument("--velocity-std", type=float, default=10.0, help="Velocity variation std (default: 10)")
    parser.add_argument("--timing-jitter", type=float, default=0.015, help="Timing jitter in seconds (default: 0.015)")
    parser.add_argument("--accent-strength", type=float, default=1.3, help="Strong beat accent multiplier (default: 1.3)")
    parser.add_argument("--seed", type=int, help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    # Load MIDI
    print(f"Loading MIDI: {args.input_midi}")
    midi = pretty_midi.PrettyMIDI(str(args.input_midi))
    
    # Humanize
    print(f"Applying humanization (velocity_std={args.velocity_std}, timing_jitter={args.timing_jitter}s)")
    humanizer = MIDIHumanizer(
        velocity_std=args.velocity_std,
        timing_jitter_seconds=args.timing_jitter,
        accent_strength=args.accent_strength,
        seed=args.seed
    )
    humanized_midi = humanizer.humanize(midi)
    
    # Save
    args.output_midi.parent.mkdir(parents=True, exist_ok=True)
    humanized_midi.write(str(args.output_midi))
    print(f"Saved humanized MIDI: {args.output_midi}")
    
    # Report statistics
    original_velocities = [note.velocity for inst in midi.instruments for note in inst.notes]
    humanized_velocities = [note.velocity for inst in humanized_midi.instruments for note in inst.notes]
    
    print("\nVelocity Statistics:")
    print(f"  Original - Mean: {np.mean(original_velocities):.1f}, Std: {np.std(original_velocities):.1f}")
    print(f"  Humanized - Mean: {np.mean(humanized_velocities):.1f}, Std: {np.std(humanized_velocities):.1f}")
    print(f"  Improvement: +{np.std(humanized_velocities) - np.std(original_velocities):.1f} std")


if __name__ == "__main__":
    main()
