#!/usr/bin/env python3
"""
Integration tests for MIDI humanizer (Stage3 v1.1 Day 2)

Tests end-to-end humanization workflow:
- Quantized MIDI → Humanize → Verify improvement
- Stage3 output → Humanize → Preserve structure
- Batch processing with multiple files
"""

import tempfile
from pathlib import Path

import numpy as np
import pretty_midi
import pytest

from scripts.humanize_midi import MIDIHumanizer


class TestHumanizerIntegration:
    """Integration tests for humanizer pipeline."""
    
    def test_quantized_to_humanized_workflow(self):
        """Test complete workflow: create quantized MIDI → humanize → verify improvement."""
        # Create quantized MIDI
        midi = pretty_midi.PrettyMIDI(initial_tempo=120)
        drums = pretty_midi.Instrument(program=0, is_drum=True)
        
        # 16 notes with uniform velocity=100
        for i in range(16):
            note = pretty_midi.Note(
                velocity=100,
                pitch=36 if i % 2 == 0 else 38,
                start=i * 0.5,
                end=i * 0.5 + 0.1,
            )
            drums.notes.append(note)
        
        midi.instruments.append(drums)
        
        # Humanize
        humanizer = MIDIHumanizer(velocity_std=12.0, timing_jitter_seconds=0.018, seed=42)
        humanized = humanizer.humanize(midi)
        
        # Verify improvement
        original_velocities = [n.velocity for n in midi.instruments[0].notes]
        humanized_velocities = [n.velocity for n in humanized.instruments[0].notes]
        
        original_std = np.std(original_velocities)
        humanized_std = np.std(humanized_velocities)
        
        # Original should be uniform (std ≈ 0)
        assert original_std < 1.0, f"Original should be quantized, got std={original_std}"
        
        # Humanized should have significant variation (target: 12.8)
        assert humanized_std >= 10.0, f"Humanized std {humanized_std} below target 10.0"
        
        # Improvement should be substantial
        improvement = humanized_std - original_std
        assert improvement >= 10.0, f"Improvement {improvement} below target 10.0"
    
    def test_batch_humanization_consistency(self):
        """Test that batch humanization produces consistent results."""
        # Create 3 quantized MIDIs
        midis = []
        for _ in range(3):
            midi = pretty_midi.PrettyMIDI(initial_tempo=120)
            piano = pretty_midi.Instrument(program=0, is_drum=False)
            
            for i in range(8):
                note = pretty_midi.Note(
                    velocity=100,
                    pitch=60 + i,
                    start=i * 0.5,
                    end=i * 0.5 + 0.4,
                )
                piano.notes.append(note)
            
            midi.instruments.append(piano)
            midis.append(midi)
        
        # Humanize all with same seed
        humanizer = MIDIHumanizer(velocity_std=12.0, timing_jitter_seconds=0.018, seed=42)
        humanized_list = [humanizer.humanize(m) for m in midis]
        
        # All should have improved velocity variation
        for humanized in humanized_list:
            velocities = [n.velocity for n in humanized.instruments[0].notes]
            velocity_std = np.std(velocities)
            assert velocity_std >= 8.0, f"Batch item has insufficient std: {velocity_std}"
    
    def test_preserve_note_count(self):
        """Test that humanization preserves number of notes."""
        midi = pretty_midi.PrettyMIDI(initial_tempo=120)
        
        # Add multiple instruments
        for program in [0, 1, 48]:  # Piano, bright piano, strings
            inst = pretty_midi.Instrument(program=program)
            for i in range(10):
                note = pretty_midi.Note(
                    velocity=80 + i,
                    pitch=60 + (i % 12),
                    start=i * 0.25,
                    end=i * 0.25 + 0.2,
                )
                inst.notes.append(note)
            midi.instruments.append(inst)
        
        humanizer = MIDIHumanizer(velocity_std=10.0, timing_jitter_seconds=0.01, seed=42)
        humanized = humanizer.humanize(midi)
        
        # Check instrument and note count
        assert len(humanized.instruments) == len(midi.instruments), \
            "Number of instruments changed"
        
        for orig_inst, hum_inst in zip(midi.instruments, humanized.instruments):
            assert len(hum_inst.notes) == len(orig_inst.notes), \
                f"Note count changed: {len(orig_inst.notes)} → {len(hum_inst.notes)}"
    
    def test_file_io_roundtrip_integration(self):
        """Test complete file I/O workflow with real files."""
        # Create test MIDI
        midi = pretty_midi.PrettyMIDI(initial_tempo=120)
        drums = pretty_midi.Instrument(program=0, is_drum=True)
        
        for i in range(8):
            note = pretty_midi.Note(
                velocity=100,
                pitch=36,
                start=i * 0.5,
                end=i * 0.5 + 0.1,
            )
            drums.notes.append(note)
        
        midi.instruments.append(drums)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            
            # Save original
            original_path = tmpdir_path / "original.mid"
            midi.write(str(original_path))
            
            # Humanize via file I/O
            humanized_path = tmpdir_path / "humanized.mid"
            humanizer = MIDIHumanizer(velocity_std=12.0, timing_jitter_seconds=0.018, seed=42)
            
            loaded_midi = pretty_midi.PrettyMIDI(str(original_path))
            humanized_midi = humanizer.humanize(loaded_midi)
            humanized_midi.write(str(humanized_path))
            
            # Reload and verify
            reloaded = pretty_midi.PrettyMIDI(str(humanized_path))
            
            # Check basic properties
            assert len(reloaded.instruments) == 1
            assert len(reloaded.instruments[0].notes) == 8
            
            # Check velocity variation
            velocities = [n.velocity for n in reloaded.instruments[0].notes]
            velocity_std = np.std(velocities)
            assert velocity_std >= 8.0, f"Reloaded MIDI has insufficient std: {velocity_std}"
    
    def test_timing_jitter_distribution(self):
        """Test that timing jitter follows expected distribution."""
        midi = pretty_midi.PrettyMIDI(initial_tempo=120)
        piano = pretty_midi.Instrument(program=0)
        
        # Create 32 notes at perfect timing
        for i in range(32):
            note = pretty_midi.Note(
                velocity=100,
                pitch=60,
                start=i * 0.25,
                end=i * 0.25 + 0.2,
            )
            piano.notes.append(note)
        
        midi.instruments.append(piano)
        
        # Apply jitter
        humanizer = MIDIHumanizer(
            velocity_std=0.0,
            timing_jitter_seconds=0.020,
            seed=42
        )
        humanized = humanizer.humanize(midi)
        
        # Calculate timing deviations
        original_starts = [n.start for n in midi.instruments[0].notes]
        humanized_starts = [n.start for n in humanized.instruments[0].notes]
        
        deviations = [abs(h - o) for h, o in zip(humanized_starts, original_starts)]
        
        # All deviations should be within ±0.020s
        max_deviation = max(deviations)
        assert max_deviation <= 0.020, \
            f"Max deviation {max_deviation} exceeds jitter limit 0.020"
        
        # Average should be around half of max (for uniform distribution)
        avg_deviation = np.mean(deviations)
        expected_avg = 0.020 / 2
        
        # Allow 50% tolerance due to sampling variability
        assert expected_avg * 0.3 <= avg_deviation <= expected_avg * 1.7, \
            f"Average deviation {avg_deviation} not in expected range"


@pytest.mark.slow
class TestHumanizerPerformance:
    """Performance and stress tests for humanizer."""
    
    def test_large_midi_performance(self):
        """Test humanization on large MIDI (1000+ notes)."""
        midi = pretty_midi.PrettyMIDI(initial_tempo=120)
        piano = pretty_midi.Instrument(program=0)
        
        # Create 1000 notes
        for i in range(1000):
            note = pretty_midi.Note(
                velocity=80 + (i % 40),
                pitch=48 + (i % 24),
                start=i * 0.1,
                end=i * 0.1 + 0.08,
            )
            piano.notes.append(note)
        
        midi.instruments.append(piano)
        
        # Humanize (should complete in reasonable time)
        humanizer = MIDIHumanizer(velocity_std=12.0, timing_jitter_seconds=0.018, seed=42)
        humanized = humanizer.humanize(midi)
        
        # Verify all notes processed
        assert len(humanized.instruments[0].notes) == 1000
        
        # Verify variation was applied
        velocities = [n.velocity for n in humanized.instruments[0].notes]
        velocity_std = np.std(velocities)
        assert velocity_std > 0, "No velocity variation applied"
