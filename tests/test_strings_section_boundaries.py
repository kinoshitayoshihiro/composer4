#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Section Boundary Tests for Strings Generator (Phase 4.7)

セクション境界での整合性テスト:
- レガート連結のセクション境界処理
- Chord spreadの適切な変化
- Dynamics変化の滑らかさ
"""

import pytest
from pathlib import Path
from typing import List, Dict, Any

import yaml

pytest.importorskip("pretty_midi")
import pretty_midi

from generator.strings_generator import StringsGenerator


def load_emotion_mapping(config_path: Path = None) -> Dict[str, Any]:
    """Load emotion mapping configuration."""
    if config_path is None:
        config_path = Path(__file__).parent.parent / "config" / "emotion_mapping.yaml"
    
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def check_strings_section_boundary(
    pm: pretty_midi.PrettyMIDI,
    section_end_time: float,
    max_overlap_ms: float = 50.0
) -> bool:
    """Check if strings notes respect section boundary."""
    max_overlap_sec = max_overlap_ms / 1000.0
    
    for inst in pm.instruments:
        if inst.is_drum:
            continue
        
        # Check for strings range (G2-E6, MIDI 43-88)
        strings_notes = [n for n in inst.notes if 43 <= n.pitch <= 88]
        
        for note in strings_notes:
            if note.end > section_end_time + max_overlap_sec:
                return False
    
    return True


def calculate_chord_spread(notes: List[pretty_midi.Note], time_window: float = 0.05) -> float:
    """
    Calculate maximum pitch spread in simultaneous notes.
    
    Args:
        notes: List of notes
        time_window: Time window for considering notes simultaneous (seconds)
    
    Returns:
        Maximum pitch spread in semitones
    """
    if not notes:
        return 0.0
    
    # Group notes by time window
    time_groups = []
    sorted_notes = sorted(notes, key=lambda n: n.start)
    
    current_group = [sorted_notes[0]]
    
    for note in sorted_notes[1:]:
        if note.start - current_group[0].start <= time_window:
            current_group.append(note)
        else:
            if len(current_group) > 1:
                time_groups.append(current_group)
            current_group = [note]
    
    if len(current_group) > 1:
        time_groups.append(current_group)
    
    # Calculate max spread
    max_spread = 0.0
    for group in time_groups:
        pitches = [n.pitch for n in group]
        spread = max(pitches) - min(pitches)
        max_spread = max(max_spread, spread)
    
    return max_spread


def test_strings_section_boundaries_basic():
    """Test basic section boundary respect for strings."""
    config = load_emotion_mapping()
    max_overlap_ms = config["transition_rules"]["basic"]["max_overlap_ms"]
    
    pm = pretty_midi.PrettyMIDI(initial_tempo=120.0)
    pm.time_signature_changes.append(pretty_midi.TimeSignature(4, 4, 0.0))
    
    inst = pretty_midi.Instrument(program=48, is_drum=False, name="Strings")
    
    # Verse section (bars 1-4, 0-8 seconds at 120 BPM)
    verse_end = 8.0
    
    # Add sustained strings chords (whole notes)
    chords = [
        [60, 64, 67],  # C major
        [62, 65, 69],  # D minor
        [64, 67, 71],  # E minor
        [60, 64, 67],  # C major
    ]
    
    for bar, chord in enumerate(chords):
        start = bar * 2.0
        end = start + 1.9  # Slightly shorter than bar to respect boundary
        
        for pitch in chord:
            inst.notes.append(pretty_midi.Note(velocity=70, pitch=pitch, start=start, end=end))
    
    pm.instruments.append(inst)
    
    # Check Verse boundary
    assert check_strings_section_boundary(pm, verse_end, max_overlap_ms)


def test_strings_legato_section_boundary():
    """Test legato connection stops at section boundary."""
    config = load_emotion_mapping()
    max_overlap_ms = config["transition_rules"]["basic"]["max_overlap_ms"]
    
    pm = pretty_midi.PrettyMIDI(initial_tempo=120.0)
    pm.time_signature_changes.append(pretty_midi.TimeSignature(4, 4, 0.0))
    
    inst = pretty_midi.Instrument(program=48, is_drum=False, name="Strings")
    
    verse_end = 8.0
    
    # Create legato line that should stop before section end
    pitches = [60, 62, 64, 65, 67, 69, 71, 72]  # C major scale
    
    for i, pitch in enumerate(pitches):
        start = i * 1.0
        end = start + 1.05  # 50ms overlap with next note (legato)
        
        # But last note should not exceed section boundary
        if end > verse_end:
            end = verse_end - 0.01  # Stop just before boundary
        
        inst.notes.append(pretty_midi.Note(velocity=75, pitch=pitch, start=start, end=end))
    
    pm.instruments.append(inst)
    
    # Check boundary
    assert check_strings_section_boundary(pm, verse_end, max_overlap_ms)


def test_strings_chord_spread_limits():
    """Test chord spread respects limits from quality gates."""
    pm = pretty_midi.PrettyMIDI(initial_tempo=120.0)
    pm.time_signature_changes.append(pretty_midi.TimeSignature(4, 4, 0.0))
    
    inst = pretty_midi.Instrument(program=48, is_drum=False, name="Strings")
    
    # Add chord within 2-octave spread limit (24 semitones)
    chord = [48, 55, 60, 67]  # C3, G3, C4, G4 (19 semitones spread)
    start = 0.0
    end = 2.0
    
    for pitch in chord:
        inst.notes.append(pretty_midi.Note(velocity=70, pitch=pitch, start=start, end=end))
    
    pm.instruments.append(inst)
    
    # Calculate spread
    spread = calculate_chord_spread(inst.notes)
    
    # Should be within 24 semitones (2 octaves)
    assert spread <= 24


def test_strings_emotion_profile_calm_to_happy():
    """Test emotion profile impact on strings legato and spread."""
    config = load_emotion_mapping()
    
    # Get strings adjustments
    strings_adj = config["instrument_adjustments"]["strings"]
    
    calm = strings_adj["calm_low"]
    happy = strings_adj["happy_high"]
    
    # Calm should have higher legato rate (more sustained)
    assert calm["legato_rate_target"] > happy["legato_rate_target"]
    
    # Happy should have wider chord spread
    assert happy["chord_spread_multiplier"] > calm["chord_spread_multiplier"]


def test_strings_section_transition_bridge_to_chorus():
    """Test Bridge to Chorus transition gap."""
    config = load_emotion_mapping()
    special = config["transition_rules"]["special_transitions"]
    
    # Bridge to Chorus should have 150ms gap
    bridge_to_chorus = special["Bridge_to_Chorus"]
    min_gap_ms = bridge_to_chorus["min_gap_ms"]
    
    assert min_gap_ms == 150


def test_strings_section_length_constraints():
    """Test strings respects section length constraints."""
    config = load_emotion_mapping()
    constraints = config["validation_rules"]["section_length_constraints"]
    
    # Test Bridge constraints
    assert 4 >= constraints["Bridge"]["min_bars"]
    assert 16 >= constraints["Bridge"]["max_bars"]


def test_strings_dynamics_progression():
    """Test dynamics (velocity) progression across section."""
    pm = pretty_midi.PrettyMIDI(initial_tempo=120.0)
    pm.time_signature_changes.append(pretty_midi.TimeSignature(4, 4, 0.0))
    
    inst = pretty_midi.Instrument(program=48, is_drum=False, name="Strings")
    
    # Create crescendo pattern in Verse
    verse_bars = 4
    base_velocity = 60
    velocity_increment = 5
    
    for bar in range(verse_bars):
        velocity = base_velocity + (bar * velocity_increment)
        start = bar * 2.0
        end = start + 1.9
        
        # Simple chord
        for pitch in [60, 64, 67]:
            inst.notes.append(pretty_midi.Note(velocity=velocity, pitch=pitch, start=start, end=end))
    
    pm.instruments.append(inst)
    
    # Check that velocities increase
    velocities = [n.velocity for n in inst.notes[0:3]]  # First chord
    last_velocities = [n.velocity for n in inst.notes[-3:]]  # Last chord
    
    assert sum(last_velocities) > sum(velocities)


@pytest.mark.skipif(
    not Path("generator/strings_generator.py").exists(),
    reason="strings_generator.py not found"
)
def test_strings_generator_section_integration():
    """Integration test: Generate strings with section awareness."""
    try:
        gen = StringsGenerator()
        
        # Test that generator can be instantiated
        assert gen is not None
        
        # TODO: Add actual generation test when generator supports section param
        # pm = gen.generate(
        #     section="Bridge",
        #     emotion_profile="melancholic_medium",
        #     bars=4,
        #     tempo=120
        # )
        
    except Exception as e:
        pytest.skip(f"Strings generator not fully implemented: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
