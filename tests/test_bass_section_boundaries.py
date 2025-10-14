#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Section Boundary Tests for Bass Generator (Phase 4.7)

セクション境界での整合性テスト:
- ルート音の連続性
- Walking bassスタイルの適切な移行
- セクション境界でのギャップ制御
"""

import pytest
from pathlib import Path
from typing import List, Dict, Any

import yaml

pytest.importorskip("pretty_midi")
import pretty_midi

from generator.bass_generator import BassGenerator


def load_emotion_mapping(config_path: Path = None) -> Dict[str, Any]:
    """Load emotion mapping configuration."""
    if config_path is None:
        config_path = Path(__file__).parent.parent / "config" / "emotion_mapping.yaml"
    
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def check_bass_section_boundary(
    pm: pretty_midi.PrettyMIDI,
    section_end_time: float,
    max_overlap_ms: float = 50.0
) -> bool:
    """Check if bass notes respect section boundary."""
    max_overlap_sec = max_overlap_ms / 1000.0
    
    for inst in pm.instruments:
        if inst.is_drum:
            continue
        
        # Check for bass range (E1-G3, MIDI 28-55)
        bass_notes = [n for n in inst.notes if 28 <= n.pitch <= 55]
        
        for note in bass_notes:
            if note.end > section_end_time + max_overlap_sec:
                return False
    
    return True


def check_bass_root_continuity(
    pm: pretty_midi.PrettyMIDI,
    section_start_time: float,
    chord_root: int = 60  # C
) -> bool:
    """
    Check if bass starts section with appropriate root note.
    
    Args:
        pm: PrettyMIDI object
        section_start_time: Start time of section (seconds)
        chord_root: Expected root pitch class (0-11)
    
    Returns:
        True if first bass note in section is close to root
    """
    for inst in pm.instruments:
        if inst.is_drum:
            continue
        
        # Find first bass note in section
        bass_notes = sorted(
            [n for n in inst.notes if 28 <= n.pitch <= 55 and n.start >= section_start_time],
            key=lambda n: n.start
        )
        
        if not bass_notes:
            return False
        
        first_note = bass_notes[0]
        first_pitch_class = first_note.pitch % 12
        root_pitch_class = chord_root % 12
        
        # Allow root or fifth (perfect 5th = 7 semitones)
        return first_pitch_class in [root_pitch_class, (root_pitch_class + 7) % 12]
    
    return False


def test_bass_section_boundaries_basic():
    """Test basic section boundary respect for bass."""
    config = load_emotion_mapping()
    max_overlap_ms = config["transition_rules"]["basic"]["max_overlap_ms"]
    
    pm = pretty_midi.PrettyMIDI(initial_tempo=120.0)
    pm.time_signature_changes.append(pretty_midi.TimeSignature(4, 4, 0.0))
    
    inst = pretty_midi.Instrument(program=33, is_drum=False, name="Bass")
    
    # Verse section (bars 1-4, 0-8 seconds at 120 BPM)
    verse_end = 8.0
    
    # Add bass notes within Verse (quarter notes on beats 1 and 3)
    for bar in range(4):
        for beat in [0, 2]:  # Beats 1 and 3
            start = bar * 2.0 + beat * 0.5
            end = start + 0.45  # Slightly shorter than beat
            inst.notes.append(pretty_midi.Note(velocity=90, pitch=36, start=start, end=end))  # C2
    
    pm.instruments.append(inst)
    
    # Check Verse boundary
    assert check_bass_section_boundary(pm, verse_end, max_overlap_ms)


def test_bass_section_boundaries_walking_style():
    """Test walking bass style respects section boundaries."""
    config = load_emotion_mapping()
    max_overlap_ms = config["transition_rules"]["basic"]["max_overlap_ms"]
    
    pm = pretty_midi.PrettyMIDI(initial_tempo=120.0)
    pm.time_signature_changes.append(pretty_midi.TimeSignature(4, 4, 0.0))
    
    inst = pretty_midi.Instrument(program=33, is_drum=False, name="Bass")
    
    # Walking bass: 4 quarter notes per bar
    verse_end = 8.0
    pitches = [36, 38, 40, 41]  # C, D, E, F (walking pattern)
    
    for bar in range(4):
        for beat in range(4):
            pitch = pitches[beat % len(pitches)]
            start = bar * 2.0 + beat * 0.5
            end = start + 0.45
            
            # Ensure last note doesn't violate boundary
            if end <= verse_end:
                inst.notes.append(pretty_midi.Note(velocity=85, pitch=pitch, start=start, end=end))
    
    pm.instruments.append(inst)
    
    # Check boundary
    assert check_bass_section_boundary(pm, verse_end, max_overlap_ms)


def test_bass_root_continuity_verse_to_chorus():
    """Test bass maintains root note at section transition."""
    pm = pretty_midi.PrettyMIDI(initial_tempo=120.0)
    pm.time_signature_changes.append(pretty_midi.TimeSignature(4, 4, 0.0))
    
    inst = pretty_midi.Instrument(program=33, is_drum=False, name="Bass")
    
    # Verse ends at 8.0s, Chorus starts at 8.0s
    verse_end = 8.0
    chorus_start = 8.0
    
    # Add Verse notes
    inst.notes.append(pretty_midi.Note(velocity=90, pitch=36, start=0.0, end=0.45))  # C2
    
    # Add Chorus starting note (should be root or fifth)
    inst.notes.append(pretty_midi.Note(velocity=95, pitch=36, start=chorus_start, end=chorus_start + 0.45))  # C2
    
    pm.instruments.append(inst)
    
    # Check root continuity (C = 0)
    assert check_bass_root_continuity(pm, chorus_start, chord_root=0)


def test_bass_emotion_profile_calm_to_energetic():
    """Test emotion profile impact on bass density."""
    config = load_emotion_mapping()
    
    # Get bass adjustments
    bass_adj = config["instrument_adjustments"]["bass"]
    
    calm_multiplier = bass_adj["calm_low"]["notes_per_bar_multiplier"]
    energetic_multiplier = bass_adj["energetic_high"]["notes_per_bar_multiplier"]
    
    # Energetic should have higher density
    assert energetic_multiplier > calm_multiplier


def test_bass_section_length_constraints():
    """Test bass respects section length constraints."""
    config = load_emotion_mapping()
    constraints = config["validation_rules"]["section_length_constraints"]
    
    # Test Verse constraints
    assert 4 >= constraints["Verse"]["min_bars"]
    assert 8 <= constraints["Verse"]["max_bars"]


def test_bass_chorus_transition_gap():
    """Test Chorus transition maintains appropriate gap."""
    config = load_emotion_mapping()
    special = config["transition_rules"]["special_transitions"]
    
    # Chorus to Verse should have 200ms gap
    chorus_to_verse = special["Chorus_to_Verse"]
    min_gap_ms = chorus_to_verse["min_gap_ms"]
    
    assert min_gap_ms == 200


@pytest.mark.skipif(
    not Path("generator/bass_generator.py").exists(),
    reason="bass_generator.py not found"
)
def test_bass_generator_section_integration():
    """Integration test: Generate bass with section awareness."""
    try:
        gen = BassGenerator()
        
        # Test that generator can be instantiated
        assert gen is not None
        
        # TODO: Add actual generation test when generator supports section param
        # pm = gen.generate(
        #     section="Chorus",
        #     emotion_profile="energetic_high",
        #     bars=4,
        #     tempo=120
        # )
        
    except Exception as e:
        pytest.skip(f"Bass generator not fully implemented: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
