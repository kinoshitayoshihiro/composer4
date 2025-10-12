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


def generate_ar1_noise(n: int, phi: float = 0.6, std: float = 1.0, clip: Optional[float] = None) -> np.ndarray:
    """
    Generate AR(1) correlated noise for natural timing/velocity variation.
    
    AR(1) process: x[t] = phi * x[t-1] + epsilon[t]
    where epsilon ~ N(0, std * sqrt(1 - phi^2))
    
    Args:
        n: Number of samples
        phi: Autocorrelation coefficient (0.0 = white noise, 0.9 = high correlation)
        std: Target standard deviation of the stationary distribution
        clip: Optional clipping threshold (±clip)
    
    Returns:
        Array of correlated noise samples
    """
    if n == 0:
        return np.array([])
    
    # Adjust epsilon std to achieve target stationary std
    epsilon_std = std * np.sqrt(1 - phi**2) if phi < 1.0 else std
    
    # Generate AR(1) series
    x = np.zeros(n)
    x[0] = np.random.normal(0, std)  # Initial value from stationary distribution
    
    for t in range(1, n):
        epsilon = np.random.normal(0, epsilon_std)
        x[t] = phi * x[t-1] + epsilon
    
    # Optional clipping
    if clip is not None:
        x = np.clip(x, -clip, clip)
    
    return x


class MIDIHumanizer:
    """Add human-like expression to quantized MIDI"""
    
    def __init__(
        self,
        velocity_std: float = 10.0,
        timing_jitter_seconds: float = 0.015,
        accent_strength: float = 1.3,
        seed: Optional[int] = None,
        # v1.1 enhancements (寸評推奨)
        use_ar1: bool = True,
        ar1_phi: float = 0.6,
        bpm_adaptive: bool = True,
        swing_strength: float = 0.06,
        beat_accent_lut: Optional[list[float]] = None,
    ):
        """
        Args:
            velocity_std: Standard deviation for velocity variation (0-127 scale)
            timing_jitter_seconds: Max timing deviation in seconds (±)
            accent_strength: Multiplier for strong beat emphasis
            seed: Random seed for reproducibility
            use_ar1: Use AR(1) correlated noise instead of independent noise
            ar1_phi: AR(1) autocorrelation coefficient (0.6 recommended)
            bpm_adaptive: Scale timing jitter by BPM (5% of eighth note)
            swing_strength: Backbeat eighth swing amount (0.06 = 6%)
            beat_accent_lut: Per-beat accent multipliers [beat1, beat2, beat3, beat4] (default: [1.15, 0.95, 1.08, 0.98])
        """
        self.velocity_std = velocity_std
        self.timing_jitter = timing_jitter_seconds
        self.accent_strength = accent_strength
        self._seed = seed  # Store for reproducibility
        
        # v1.1 enhancements
        self.use_ar1 = use_ar1
        self.ar1_phi = ar1_phi
        self.bpm_adaptive = bpm_adaptive
        self.swing_strength = swing_strength
        self.beat_accent_lut = beat_accent_lut or [1.15, 0.95, 1.08, 0.98]  # 4/4 default
        
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
    
    def _estimate_bpm(self, midi: pretty_midi.PrettyMIDI) -> float:
        """
        Estimate BPM from MIDI tempo changes.
        
        Returns:
            BPM (default: 120 if no tempo info)
        """
        tempo_changes = midi.get_tempo_changes()
        if len(tempo_changes[1]) > 0:
            return tempo_changes[1][0]  # First tempo
        return 120.0  # Default BPM
    
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
            if hasattr(self, '_seed') and self._seed is not None:
                random.seed(self._seed)
                np.random.seed(self._seed)
        
        # Estimate BPM for adaptive scaling
        bpm = self._estimate_bpm(midi)
        quarter_note_ms = 60000.0 / bpm
        eighth_note_ms = quarter_note_ms / 2
        
        # Calculate adaptive jitter scale
        if self.bpm_adaptive:
            # 5% of eighth note duration (寸評推奨)
            jitter_scale = 0.05 * eighth_note_ms / 1000.0  # Convert to seconds
        else:
            jitter_scale = self.timing_jitter
        
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
            
            # Pre-generate correlated noise for all notes (寸評推奨: AR(1))
            n_notes = len(instrument.notes)
            if self.use_ar1 and n_notes > 0:
                # Timing noise: AR(1) with adaptive scale
                timing_noise = generate_ar1_noise(
                    n=n_notes,
                    phi=self.ar1_phi,
                    std=jitter_scale,
                    clip=0.08 * eighth_note_ms / 1000.0  # Clip at 8% of eighth note
                )
                # Velocity noise: AR(1) with std=4.0 (寸評推奨)
                velocity_noise = generate_ar1_noise(
                    n=n_notes,
                    phi=self.ar1_phi,
                    std=4.0,
                    clip=None
                )
            else:
                timing_noise = np.zeros(n_notes)
                velocity_noise = np.zeros(n_notes)
            
            for i, note in enumerate(instrument.notes):
                # Apply velocity variation with beat accent LUT
                new_velocity = self._humanize_velocity_v11(
                    note.velocity,
                    note.start,
                    midi,
                    velocity_noise[i],
                    bpm
                )
                
                # Apply timing jitter with swing
                new_start = self._humanize_timing_v11(
                    note.start,
                    timing_noise[i],
                    eighth_note_ms
                )
                new_end = note.end + (new_start - note.start)  # Preserve duration
                
                new_instrument.notes.append(
                    pretty_midi.Note(
                        velocity=new_velocity,
                        pitch=note.pitch,
                        start=max(0.0, new_start),
                        end=max(new_start + 1e-4, new_end)  # Ensure positive duration
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
    
    def _humanize_velocity_v11(
        self,
        original_velocity: int,
        note_time: float,
        midi: pretty_midi.PrettyMIDI,
        ar1_noise: float,
        bpm: float
    ) -> int:
        """
        Add realistic velocity variation (v1.1: beat accent LUT + AR(1) noise)
        
        Strategy:
        1. Get beat index (0, 1, 2, 3 for 4/4)
        2. Apply beat-specific accent from LUT
        3. Add AR(1) correlated noise
        4. Add independent Gaussian noise
        5. Clamp to valid MIDI range [1, 127]
        """
        # Get beat index
        beat_idx = self._get_beat_index(note_time, midi, bpm)
        
        # Apply beat accent from LUT
        if beat_idx < len(self.beat_accent_lut):
            beat_accent = self.beat_accent_lut[beat_idx]
        else:
            beat_accent = 1.0
        
        # Base with beat accent
        base_velocity = original_velocity * beat_accent
        
        # Add AR(1) correlated noise
        base_velocity += ar1_noise
        
        # Add independent Gaussian noise
        independent_noise = np.random.normal(0, self.velocity_std)
        new_velocity = base_velocity + independent_noise
        
        # Clamp to valid range
        return int(np.clip(new_velocity, 1, 127))
    
    def _humanize_timing(self, note_time: float) -> float:
        """
        Add subtle timing jitter
        
        Uses uniform distribution: ±timing_jitter
        """
        jitter = np.random.uniform(-self.timing_jitter, self.timing_jitter)
        return note_time + jitter
    
    def _humanize_timing_v11(
        self,
        note_time: float,
        ar1_noise: float,
        eighth_note_ms: float
    ) -> float:
        """
        Add subtle timing jitter (v1.1: AR(1) + backbeat swing)
        
        Strategy:
        1. Apply AR(1) correlated noise (pre-generated)
        2. Add backbeat eighth swing if applicable
        3. Ensure non-negative time
        """
        # Start with AR(1) noise
        shift = ar1_noise
        
        # Add swing to backbeat eighths (寸評推奨: +6% × eighth_ms)
        if self.swing_strength > 0 and self._is_backbeat_eighth(note_time, eighth_note_ms):
            swing_shift = self.swing_strength * 0.5 * eighth_note_ms / 1000.0
            shift += swing_shift
        
        return max(0.0, note_time + shift)
    
    def _get_beat_index(
        self,
        time: float,
        midi: pretty_midi.PrettyMIDI,
        bpm: float
    ) -> int:
        """
        Get beat index (0, 1, 2, 3 for 4/4) for beat accent LUT.
        
        Args:
            time: Time in seconds
            midi: MIDI object with time signature info
            bpm: BPM
        
        Returns:
            Beat index (0-based)
        """
        # Get time signature
        ts = midi.time_signature_changes[0] if midi.time_signature_changes else None
        beats_per_bar = ts.numerator if ts else 4
        
        # Calculate beat position
        quarter_note_seconds = 60.0 / bpm
        beat_position = time / quarter_note_seconds
        beat_in_bar = int(beat_position % beats_per_bar)
        
        return beat_in_bar
    
    def _is_backbeat_eighth(self, time: float, eighth_note_ms: float) -> bool:
        """
        Check if time is on a backbeat eighth (裏拍8分).
        
        Backbeat eighths are at phase ≈ 1.0 in a 2-eighth cycle.
        
        Args:
            time: Time in seconds
            eighth_note_ms: Eighth note duration in milliseconds
        
        Returns:
            True if on backbeat eighth
        """
        time_ms = time * 1000.0
        eighth_phase = (time_ms / eighth_note_ms) % 2.0
        
        # Phase 0.75 < phase < 1.25 is considered backbeat
        return 0.75 < eighth_phase < 1.25
    
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
    parser = argparse.ArgumentParser(description="Humanize MIDI files with velocity and timing variation (v1.1)")
    parser.add_argument("input_midi", type=Path, help="Input MIDI file")
    parser.add_argument("output_midi", type=Path, help="Output humanized MIDI file")
    parser.add_argument("--velocity-std", type=float, default=10.0, help="Velocity variation std (default: 10)")
    parser.add_argument("--timing-jitter", type=float, default=0.015, help="Timing jitter in seconds (default: 0.015)")
    parser.add_argument("--accent-strength", type=float, default=1.3, help="Strong beat accent multiplier (default: 1.3)")
    parser.add_argument("--seed", type=int, help="Random seed for reproducibility")
    
    # v1.1 enhancements (寸評推奨)
    parser.add_argument("--use-ar1", action="store_true", default=True, help="Use AR(1) correlated noise (default: True)")
    parser.add_argument("--ar1-phi", type=float, default=0.6, help="AR(1) autocorrelation (default: 0.6)")
    parser.add_argument("--bpm-adaptive", action="store_true", default=True, help="Scale jitter by BPM (default: True)")
    parser.add_argument("--swing-strength", type=float, default=0.06, help="Backbeat eighth swing (default: 0.06)")
    parser.add_argument("--no-v11", action="store_true", help="Disable v1.1 enhancements (use v1.0 mode)")
    
    args = parser.parse_args()
    
    # Load MIDI
    print(f"Loading MIDI: {args.input_midi}")
    midi = pretty_midi.PrettyMIDI(str(args.input_midi))
    
    # Humanize
    use_v11 = not args.no_v11
    mode_str = "v1.1 (AR1+BPM+Swing)" if use_v11 else "v1.0 (legacy)"
    print(f"Applying humanization {mode_str} (velocity_std={args.velocity_std}, timing_jitter={args.timing_jitter}s)")
    
    humanizer = MIDIHumanizer(
        velocity_std=args.velocity_std,
        timing_jitter_seconds=args.timing_jitter,
        accent_strength=args.accent_strength,
        seed=args.seed,
        use_ar1=args.use_ar1 and use_v11,
        ar1_phi=args.ar1_phi,
        bpm_adaptive=args.bpm_adaptive and use_v11,
        swing_strength=args.swing_strength if use_v11 else 0.0,
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
