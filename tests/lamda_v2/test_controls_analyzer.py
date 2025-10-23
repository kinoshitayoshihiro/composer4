#!/usr/bin/env python3
"""Tests for controls_analyzer module (Phase3)."""

from __future__ import annotations
import pretty_midi as pm
from scripts.lamda_v2.controls_analyzer import analyze_controls


def test_controls_empty_midi():
    """Test controls analysis handles empty MIDI gracefully."""
    midi = pm.PrettyMIDI()
    
    controls = analyze_controls(midi)
    
    # Check structure
    assert "pb_range" in controls
    assert "cc_summary" in controls
    assert "rpn_seen" in controls
    
    # Empty MIDI should have zero pitch bend
    assert controls["pb_range"] == [0, 0]
    assert controls["rpn_seen"] in (True, False)


def test_controls_with_pitch_bend():
    """Test pitch bend detection."""
    midi = pm.PrettyMIDI()
    inst = pm.Instrument(program=0)
    
    # Add pitch bends
    inst.pitch_bends.append(pm.PitchBend(pitch=-2048, time=0.0))
    inst.pitch_bends.append(pm.PitchBend(pitch=2048, time=1.0))
    
    midi.instruments.append(inst)
    
    controls = analyze_controls(midi)
    
    # Check pitch bend range
    assert controls["pb_range"][0] <= -2048
    assert controls["pb_range"][1] >= 2048


def test_controls_with_cc():
    """Test control change detection."""
    midi = pm.PrettyMIDI()
    inst = pm.Instrument(program=0)
    
    # Add control changes (CC 7 = volume)
    inst.control_changes.append(pm.ControlChange(number=7, value=50, time=0.0))
    inst.control_changes.append(pm.ControlChange(number=7, value=100, time=1.0))
    
    midi.instruments.append(inst)
    
    controls = analyze_controls(midi)
    
    # Check CC summary
    assert "7" in controls["cc_summary"]
    assert controls["cc_summary"]["7"]["min"] == 50
    assert controls["cc_summary"]["7"]["max"] == 100


def test_controls_rpn_detection():
    """Test RPN detection."""
    midi = pm.PrettyMIDI()
    inst = pm.Instrument(program=0)
    
    # Add RPN-related CCs (100, 101 = RPN LSB/MSB)
    inst.control_changes.append(pm.ControlChange(number=101, value=0, time=0.0))
    inst.control_changes.append(pm.ControlChange(number=100, value=0, time=0.0))
    inst.control_changes.append(pm.ControlChange(number=6, value=2, time=0.0))
    
    midi.instruments.append(inst)
    
    controls = analyze_controls(midi)
    
    # RPN should be detected
    assert controls["rpn_seen"] is True


def test_controls_output_types():
    """Test controls analysis output types are correct."""
    midi = pm.PrettyMIDI()
    inst = pm.Instrument(program=0)
    inst.pitch_bends.append(pm.PitchBend(pitch=100, time=0.0))
    midi.instruments.append(inst)
    
    controls = analyze_controls(midi)
    
    # Type checks
    assert isinstance(controls["pb_range"], list)
    assert len(controls["pb_range"]) == 2
    assert isinstance(controls["pb_range"][0], int)
    assert isinstance(controls["pb_range"][1], int)
    assert isinstance(controls["cc_summary"], dict)
    assert isinstance(controls["rpn_seen"], bool)
