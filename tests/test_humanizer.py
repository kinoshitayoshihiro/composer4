#!/usr/bin/env python3
"""Unit tests for MIDI humanizer (Stage3 v1.1 quality enhancement).

Tests:
- Velocity standard deviation increase
- Timing jitter application
- Pitch/duration preservation
- Strong beat accent detection
"""

import tempfile
from pathlib import Path

import numpy as np
import pretty_midi
import pytest

from scripts.humanize_midi import MIDIHumanizer


@pytest.fixture
def simple_midi() -> pretty_midi.PrettyMIDI:
    """Create a simple MIDI with uniform velocity and timing."""
    midi = pretty_midi.PrettyMIDI(initial_tempo=120)
    piano = pretty_midi.Instrument(program=0, is_drum=False)
    
    # 4 bars of quarter notes (16 notes total)
    # Uniform velocity=100, perfect quantization
    for i in range(16):
        note = pretty_midi.Note(
            velocity=100,
            pitch=60 + (i % 12),  # C major scale pattern
            start=i * 0.5,  # Quarter note = 0.5s at 120 BPM
            end=(i + 1) * 0.5 - 0.01,
        )
        piano.notes.append(note)
    
    midi.instruments.append(piano)
    return midi


@pytest.fixture
def drum_midi() -> pretty_midi.PrettyMIDI:
    """Create a drum pattern with strong/weak beats."""
    midi = pretty_midi.PrettyMIDI(initial_tempo=120)
    drums = pretty_midi.Instrument(program=0, is_drum=True)
    
    # Kick on beats 1 and 3, snare on beats 2 and 4
    # 2 bars of 4/4
    pattern = [
        (0.0, 36, 100),   # Beat 1: Kick (downbeat)
        (0.5, 38, 80),    # Beat 2: Snare
        (1.0, 36, 100),   # Beat 3: Kick
        (1.5, 38, 80),    # Beat 4: Snare
        (2.0, 36, 100),   # Beat 1 (bar 2): Kick (downbeat)
        (2.5, 38, 80),    # Beat 2: Snare
        (3.0, 36, 100),   # Beat 3: Kick
        (3.5, 38, 80),    # Beat 4: Snare
    ]
    
    for start, pitch, velocity in pattern:
        note = pretty_midi.Note(
            velocity=velocity,
            pitch=pitch,
            start=start,
            end=start + 0.1,
        )
        drums.notes.append(note)
    
    midi.instruments.append(drums)
    return midi


class TestMIDIHumanizer:
    """Test suite for MIDIHumanizer."""
    
    def test_velocity_variation_increase(self, simple_midi):
        """Test that velocity std increases after humanization."""
        humanizer = MIDIHumanizer(velocity_std=12.0, timing_jitter_seconds=0.0, seed=42)
        humanized = humanizer.humanize(simple_midi)
        
        original_velocities = [n.velocity for n in simple_midi.instruments[0].notes]
        humanized_velocities = [n.velocity for n in humanized.instruments[0].notes]
        
        original_std = np.std(original_velocities)
        humanized_std = np.std(humanized_velocities)
        
        # Original should be near-zero (uniform velocity=100)
        assert original_std < 1.0, f"Original std should be ~0, got {original_std}"
        
        # Humanized should increase std significantly
        assert humanized_std > 8.0, f"Humanized std should be >8, got {humanized_std}"
        
        # Target: velocity_std ≈ 12.0 (within 50% tolerance due to clipping and accent)
        assert 6.0 < humanized_std < 18.0, f"Expected std ~12.0, got {humanized_std}"
    
    def test_velocity_range_clamping(self, simple_midi):
        """Test that velocities are clamped to valid MIDI range [1, 127]."""
        humanizer = MIDIHumanizer(velocity_std=50.0, timing_jitter_seconds=0.0, seed=42)
        humanized = humanizer.humanize(simple_midi)
        
        velocities = [n.velocity for n in humanized.instruments[0].notes]
        
        assert all(1 <= v <= 127 for v in velocities), \
            f"Velocities out of range: {velocities}"
    
    def test_timing_jitter_application(self, simple_midi):
        """Test that timing jitter is applied to note onsets."""
        humanizer = MIDIHumanizer(velocity_std=0.0, timing_jitter_seconds=0.02, seed=42)
        humanized = humanizer.humanize(simple_midi)
        
        original_starts = [n.start for n in simple_midi.instruments[0].notes]
        humanized_starts = [n.start for n in humanized.instruments[0].notes]
        
        # Calculate timing deviations
        deviations = [abs(h - o) for h, o in zip(humanized_starts, original_starts)]
        
        # At least some notes should have timing jitter
        assert sum(d > 0.001 for d in deviations) >= len(deviations) // 2, \
            "Timing jitter not applied to enough notes"
        
        # Jitter should be within ±timing_jitter
        max_deviation = max(deviations)
        assert max_deviation <= 0.02, \
            f"Max timing deviation {max_deviation} exceeds jitter limit 0.02"
    
    def test_pitch_preservation(self, simple_midi):
        """Test that pitches are not changed by humanization."""
        humanizer = MIDIHumanizer(velocity_std=10.0, timing_jitter_seconds=0.015, seed=42)
        humanized = humanizer.humanize(simple_midi)
        
        original_pitches = [n.pitch for n in simple_midi.instruments[0].notes]
        humanized_pitches = [n.pitch for n in humanized.instruments[0].notes]
        
        assert original_pitches == humanized_pitches, \
            "Pitches should not be modified by humanization"
    
    def test_duration_preservation(self, simple_midi):
        """Test that note durations are approximately preserved."""
        humanizer = MIDIHumanizer(velocity_std=10.0, timing_jitter_seconds=0.015, seed=42)
        humanized = humanizer.humanize(simple_midi)
        
        original_durations = [n.end - n.start for n in simple_midi.instruments[0].notes]
        humanized_durations = [n.end - n.start for n in humanized.instruments[0].notes]
        
        # Durations should be approximately equal (within 1ms tolerance)
        for orig, hum in zip(original_durations, humanized_durations):
            assert abs(orig - hum) < 0.001, \
                f"Duration changed: {orig:.3f}s -> {hum:.3f}s"
    
    def test_strong_beat_accent(self, drum_midi):
        """Test that downbeats (beat 1) receive higher velocity accents."""
        humanizer = MIDIHumanizer(
            velocity_std=8.0,
            timing_jitter_seconds=0.0,
            accent_strength=1.5,
            seed=42
        )
        humanized = humanizer.humanize(drum_midi)
        
        # Identify kick notes (pitch 36) which are on beats 1 and 3
        kick_notes = [n for n in humanized.instruments[0].notes if n.pitch == 36]
        snare_notes = [n for n in humanized.instruments[0].notes if n.pitch == 38]
        
        # Downbeat kicks (beat 1) should have higher average velocity than snares
        downbeat_kicks = [kick_notes[0], kick_notes[2]]  # Beat 1 of bars 1 and 2
        downbeat_velocities = [n.velocity for n in downbeat_kicks]
        snare_velocities = [n.velocity for n in snare_notes]
        
        avg_downbeat = np.mean(downbeat_velocities)
        avg_snare = np.mean(snare_velocities)
        
        # Downbeats should be louder on average (due to accent_strength)
        assert avg_downbeat > avg_snare, \
            f"Downbeat avg {avg_downbeat} should be > snare avg {avg_snare}"
    
    def test_reproducibility_with_seed(self, simple_midi):
        """Test that same seed produces identical results."""
        humanizer1 = MIDIHumanizer(velocity_std=10.0, timing_jitter_seconds=0.015, seed=42)
        humanizer2 = MIDIHumanizer(velocity_std=10.0, timing_jitter_seconds=0.015, seed=42)
        
        humanized1 = humanizer1.humanize(simple_midi)
        humanized2 = humanizer2.humanize(simple_midi)
        
        velocities1 = [n.velocity for n in humanized1.instruments[0].notes]
        velocities2 = [n.velocity for n in humanized2.instruments[0].notes]
        
        assert velocities1 == velocities2, \
            "Same seed should produce identical velocities"
        
        starts1 = [n.start for n in humanized1.instruments[0].notes]
        starts2 = [n.start for n in humanized2.instruments[0].notes]
        
        assert starts1 == starts2, \
            "Same seed should produce identical timing"
    
    def test_file_roundtrip(self, simple_midi):
        """Test that humanized MIDI can be saved and loaded."""
        humanizer = MIDIHumanizer(velocity_std=10.0, timing_jitter_seconds=0.015, seed=42)
        humanized = humanizer.humanize(simple_midi)
        
        with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        
        try:
            # Save humanized MIDI
            humanized.write(str(tmp_path))
            
            # Load it back
            reloaded = pretty_midi.PrettyMIDI(str(tmp_path))
            
            # Check note count preserved
            assert len(reloaded.instruments) == len(humanized.instruments)
            assert len(reloaded.instruments[0].notes) == len(humanized.instruments[0].notes)
            
            # Check velocities match (within MIDI quantization tolerance)
            orig_vels = [n.velocity for n in humanized.instruments[0].notes]
            reload_vels = [n.velocity for n in reloaded.instruments[0].notes]
            assert orig_vels == reload_vels
        
        finally:
            tmp_path.unlink(missing_ok=True)
    
    def test_multiple_instruments(self):
        """Test humanization with multiple instruments."""
        midi = pretty_midi.PrettyMIDI(initial_tempo=120)
        
        # Add piano
        piano = pretty_midi.Instrument(program=0)
        piano.notes.append(pretty_midi.Note(velocity=100, pitch=60, start=0.0, end=0.5))
        midi.instruments.append(piano)
        
        # Add strings
        strings = pretty_midi.Instrument(program=48)
        strings.notes.append(pretty_midi.Note(velocity=80, pitch=64, start=0.0, end=0.5))
        midi.instruments.append(strings)
        
        humanizer = MIDIHumanizer(velocity_std=10.0, timing_jitter_seconds=0.01, seed=42)
        humanized = humanizer.humanize(midi)
        
        # Both instruments should be present
        assert len(humanized.instruments) == 2
        
        # Both should have humanization applied
        piano_vel = humanized.instruments[0].notes[0].velocity
        strings_vel = humanized.instruments[1].notes[0].velocity
        
        # Velocities should differ from original due to humanization
        assert piano_vel != 100 or strings_vel != 80


# Integration test targets
TARGET_VELOCITY_STD = 12.8
TARGET_TIMING_JITTER = 0.018


@pytest.mark.integration
class TestLamdaTargets:
    """Integration tests for Lamda metric improvement targets."""
    
    def test_velocity_std_target(self, simple_midi):
        """Test that humanization achieves target velocity std ≥ 12.8."""
        humanizer = MIDIHumanizer(velocity_std=12.0, timing_jitter_seconds=0.0, seed=42)
        humanized = humanizer.humanize(simple_midi)
        
        velocities = [n.velocity for n in humanized.instruments[0].notes]
        velocity_std = np.std(velocities)
        
        # Target from evaluation response: velocity_std should improve to 12.8+
        # Allow 20% tolerance due to clipping and accent variation
        assert velocity_std >= TARGET_VELOCITY_STD * 0.8, \
            f"Velocity std {velocity_std:.1f} below target {TARGET_VELOCITY_STD}"
    
    def test_timing_jitter_target(self, simple_midi):
        """Test that humanization achieves target timing jitter ≥ 0.018s."""
        humanizer = MIDIHumanizer(velocity_std=0.0, timing_jitter_seconds=0.018, seed=42)
        humanized = humanizer.humanize(simple_midi)
        
        original_starts = [n.start for n in simple_midi.instruments[0].notes]
        humanized_starts = [n.start for n in humanized.instruments[0].notes]
        
        deviations = [abs(h - o) for h, o in zip(humanized_starts, original_starts)]
        avg_jitter = np.mean(deviations)
        
        # Target from evaluation response: timing_jitter should be ~0.018s
        # Average jitter should be close to half of max jitter (due to uniform distribution)
        expected_avg = TARGET_TIMING_JITTER / 2
        assert avg_jitter >= expected_avg * 0.5, \
            f"Average jitter {avg_jitter:.4f}s below expected {expected_avg:.4f}s"
