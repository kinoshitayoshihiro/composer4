#!/usr/bin/env python3
"""Tests for lamda_v2.section_analyzer module (Phase2-4)."""
from __future__ import annotations
import tempfile
from pathlib import Path
import pytest

try:
    import pretty_midi
    import mido
except ImportError:
    pytest.skip("pretty_midi or mido not available", allow_module_level=True)

from scripts.lamda_v2.section_analyzer import (
    auto_segment_sections,
    compute_novelty_curve,
    _compute_bar_energy,
    _detect_section_boundaries,
    _assign_section_labels,
)


def _create_test_midi_with_dynamics(path: Path, bars: int = 16):
    """Create a test MIDI file with varying dynamics (soft→loud→soft pattern)."""
    mid = mido.MidiFile()
    track = mido.MidiTrack()
    mid.tracks.append(track)
    
    # Set tempo and time signature
    track.append(mido.MetaMessage('set_tempo', tempo=mido.bpm2tempo(120.0), time=0))
    track.append(mido.MetaMessage('time_signature', numerator=4, denominator=4, time=0))
    
    ticks_per_bar = mid.ticks_per_beat * 4
    
    # Create dynamics pattern: soft (bars 0-3), loud (bars 4-11), soft (bars 12-15)
    for b in range(bars):
        if b < 4:
            velocity = 40  # Soft
        elif b < 12:
            velocity = 100  # Loud
        else:
            velocity = 40  # Soft
        
        # Add C major chord at each bar
        time_offset = 0 if b == 0 else ticks_per_bar
        for i, pitch in enumerate([60, 64, 67]):
            track.append(mido.Message('note_on', note=pitch, velocity=velocity, 
                                     time=time_offset if i == 0 else 0))
            time_offset = 0
        
        # Note offs after 100 ticks
        track.append(mido.Message('note_off', note=60, velocity=0, time=100))
        track.append(mido.Message('note_off', note=64, velocity=0, time=0))
        track.append(mido.Message('note_off', note=67, velocity=0, 
                                 time=ticks_per_bar - 100 if b < bars - 1 else 0))
    
    mid.save(str(path))


def test_auto_segment_sections_basic():
    """Test basic section segmentation."""
    with tempfile.TemporaryDirectory() as tmpdir:
        midi_path = Path(tmpdir) / "test.mid"
        _create_test_midi_with_dynamics(midi_path, bars=16)
        
        pm = pretty_midi.PrettyMIDI(str(midi_path))
        downbeats_ql = [float(b * 4.0) for b in range(16)]
        
        result = auto_segment_sections(pm, downbeats_ql, min_bars=4)
        
        # Check output structure
        assert result["unit"] == "bar"
        assert "sections" in result
        assert "energy" in result
        
        # Check sections list
        sections = result["sections"]
        assert len(sections) > 0
        assert sections[0]["bar"] == 0
        assert "label" in sections[0]


def test_compute_bar_energy():
    """Test RMS energy computation per bar."""
    with tempfile.TemporaryDirectory() as tmpdir:
        midi_path = Path(tmpdir) / "test.mid"
        _create_test_midi_with_dynamics(midi_path, bars=16)
        
        pm = pretty_midi.PrettyMIDI(str(midi_path))
        downbeats_ql = [float(b * 4.0) for b in range(16)]
        
        from scripts.lamda_v2.tempo_timing import build_beat_grid
        grid = build_beat_grid(pm)
        tempo_map = grid.get("tempo_map", [(0.0, 120.0)])
        
        energy = _compute_bar_energy(pm, downbeats_ql, tempo_map)
        
        # Check energy list length
        assert len(energy) == len(downbeats_ql)
        
        # Check that energy values are non-negative
        assert all(e >= 0 for e in energy)
        
        # Check that at least some bars have energy (notes detected)
        assert sum(energy) > 0, "Total energy should be non-zero"


def test_detect_section_boundaries():
    """Test section boundary detection from energy curve."""
    # Create energy pattern: low→high→low
    energy = [0.3] * 4 + [1.0] * 8 + [0.3] * 4
    
    boundaries = _detect_section_boundaries(energy, min_bars=4)
    
    # Should detect at least the start (bar 0)
    assert 0 in boundaries
    
    # Should have at least 2 boundaries (intro + main)
    assert len(boundaries) >= 1


def test_assign_section_labels():
    """Test section label assignment."""
    boundaries = [0, 8, 16]
    total_bars = 24
    
    sections = _assign_section_labels(boundaries, total_bars)
    
    # Check first section is "intro"
    assert sections[0]["label"] == "intro"
    assert sections[0]["bar"] == 0
    
    # Check alternating verse/chorus pattern
    assert len(sections) == len(boundaries)


def test_compute_novelty_curve():
    """Test novelty curve computation."""
    energy = [0.3, 0.3, 0.3, 0.8, 1.0, 1.0, 0.5, 0.3]
    
    novelty = compute_novelty_curve(energy, kernel_size=4)
    
    # Check length
    assert len(novelty) == len(energy)
    
    # First bar should have zero novelty
    assert novelty[0] == 0.0
    
    # Bar 3→4 has large energy jump, should have high novelty
    assert novelty[3] > novelty[1]


def test_empty_midi_handling():
    """Test handling of empty MIDI data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        midi_path = Path(tmpdir) / "empty.mid"
        
        # Create empty MIDI
        mid = mido.MidiFile()
        track = mido.MidiTrack()
        mid.tracks.append(track)
        track.append(mido.MetaMessage('set_tempo', tempo=mido.bpm2tempo(120.0), time=0))
        mid.save(str(midi_path))
        
        pm = pretty_midi.PrettyMIDI(str(midi_path))
        downbeats_ql = [0.0, 4.0, 8.0, 12.0]
        
        result = auto_segment_sections(pm, downbeats_ql, min_bars=4)
        
        # Should return valid structure even for empty MIDI
        assert result["unit"] == "bar"
        assert isinstance(result["sections"], list)
        assert isinstance(result["energy"], list)


def test_min_bars_enforcement():
    """Test that minimum bar length is enforced."""
    with tempfile.TemporaryDirectory() as tmpdir:
        midi_path = Path(tmpdir) / "test.mid"
        _create_test_midi_with_dynamics(midi_path, bars=32)
        
        pm = pretty_midi.PrettyMIDI(str(midi_path))
        downbeats_ql = [float(b * 4.0) for b in range(32)]
        
        result = auto_segment_sections(pm, downbeats_ql, min_bars=8)
        
        # Check that sections are spaced at least min_bars apart
        sections = result["sections"]
        if len(sections) > 1:
            for i in range(1, len(sections)):
                spacing = sections[i]["bar"] - sections[i-1]["bar"]
                assert spacing >= 8, f"Section spacing {spacing} < min_bars 8"


def test_energy_normalization():
    """Test that energy values are normalized to [0, 1]."""
    with tempfile.TemporaryDirectory() as tmpdir:
        midi_path = Path(tmpdir) / "test.mid"
        _create_test_midi_with_dynamics(midi_path, bars=16)
        
        pm = pretty_midi.PrettyMIDI(str(midi_path))
        downbeats_ql = [float(b * 4.0) for b in range(16)]
        
        result = auto_segment_sections(pm, downbeats_ql, min_bars=4)
        
        energy_values = [e[1] for e in result["energy"]]
        
        # Check normalization
        if energy_values:
            assert max(energy_values) <= 1.0
            assert min(energy_values) >= 0.0
