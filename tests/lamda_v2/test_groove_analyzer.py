#!/usr/bin/env python3
"""Tests for groove_analyzer module (Phase3)."""

from __future__ import annotations
import pretty_midi as pm
from scripts.lamda_v2.groove_analyzer import analyze_groove


def _mk_midi(bpm: float = 120.0) -> pm.PrettyMIDI:
    """Create test MIDI with straight 8th notes."""
    midi = pm.PrettyMIDI()
    inst = pm.Instrument(program=0)
    
    t = 0.0
    for i in range(8):
        # Straight 8ths at 120 BPM: 0.25s per 8th note
        note = pm.Note(velocity=80, pitch=60, start=t, end=t + 0.1)
        inst.notes.append(note)
        t += 0.25  # 120bpm: 0.5s/beat → 0.25s = 8th
    
    midi.instruments.append(inst)
    return midi


def test_groove_defaults():
    """Test groove analysis returns valid defaults."""
    midi = _mk_midi()
    downbeats = [0.0, 0.5, 1.0]
    
    groove = analyze_groove(midi, downbeats)
    
    # Check structure
    assert "swing_pct" in groove
    assert "backbeat_strength" in groove
    assert "onset_deviation_hist" in groove
    
    # Check ranges
    assert 0.0 <= groove["swing_pct"] <= 100.0
    assert 0.0 <= groove["backbeat_strength"] <= 1.0
    assert isinstance(groove["onset_deviation_hist"], list)


def test_groove_empty_midi():
    """Test groove analysis handles empty MIDI gracefully."""
    midi = pm.PrettyMIDI()
    downbeats = [0.0, 0.5, 1.0]
    
    groove = analyze_groove(midi, downbeats)
    
    # Should return safe defaults
    assert groove["swing_pct"] == 0.0
    assert groove["backbeat_strength"] == 0.5
    assert groove["onset_deviation_hist"] == []


def test_groove_no_downbeats():
    """Test groove analysis with no downbeats."""
    midi = _mk_midi()
    downbeats = []
    
    groove = analyze_groove(midi, downbeats)
    
    # Should return safe defaults
    assert groove["swing_pct"] == 0.0
    assert groove["backbeat_strength"] == 0.5


def test_groove_output_types():
    """Test groove analysis output types are correct."""
    midi = _mk_midi()
    downbeats = [0.0, 0.5, 1.0, 1.5]
    
    groove = analyze_groove(midi, downbeats)
    
    # Type checks
    assert isinstance(groove["swing_pct"], float)
    assert isinstance(groove["backbeat_strength"], float)
    assert isinstance(groove["onset_deviation_hist"], list)
    
    # Histogram values should be integers
    for val in groove["onset_deviation_hist"]:
        assert isinstance(val, int)
