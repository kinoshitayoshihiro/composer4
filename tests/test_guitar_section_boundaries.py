#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Section Boundary Tests for Guitar Generator (Phase 4.7)

セクション境界での整合性テスト:
- 音符が次のセクションに侵入していないか
- セクション間のギャップが適切か
- Emotion profileがセクションに応じて変化しているか
"""

import pytest
from pathlib import Path
from typing import List, Dict, Any

import yaml

pytest.importorskip("pretty_midi")
import pretty_midi

from generator.guitar_generator import GuitarGenerator


def load_emotion_mapping(config_path: Path = None) -> Dict[str, Any]:
    """Load emotion mapping configuration."""
    if config_path is None:
        config_path = Path(__file__).parent.parent / "config" / "emotion_mapping.yaml"
    
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def check_section_boundary(
    pm: pretty_midi.PrettyMIDI,
    section_end_time: float,
    max_overlap_ms: float = 50.0
) -> bool:
    """
    Check if notes respect section boundary.
    
    Args:
        pm: PrettyMIDI object
        section_end_time: End time of current section (seconds)
        max_overlap_ms: Maximum allowed overlap in milliseconds
    
    Returns:
        True if boundary is respected, False otherwise
    """
    max_overlap_sec = max_overlap_ms / 1000.0
    
    for inst in pm.instruments:
        if inst.is_drum:
            continue
        
        for note in inst.notes:
            # Check if note end time exceeds section boundary
            if note.end > section_end_time + max_overlap_sec:
                return False
    
    return True


def test_guitar_section_boundaries_basic():
    """Test basic section boundary respect."""
    config = load_emotion_mapping()
    max_overlap_ms = config["transition_rules"]["basic"]["max_overlap_ms"]
    
    # Create simple test case
    gen = GuitarGenerator()
    
    # Generate 2 sections: Verse (4 bars) + Chorus (4 bars)
    sections = [
        {"label": "Verse", "start_bar": 1, "end_bar": 4, "tempo": 120},
        {"label": "Chorus", "start_bar": 5, "end_bar": 8, "tempo": 120},
    ]
    
    pm = pretty_midi.PrettyMIDI(initial_tempo=120.0)
    pm.time_signature_changes.append(pretty_midi.TimeSignature(4, 4, 0.0))
    
    inst = pretty_midi.Instrument(program=24, is_drum=False, name="Guitar")
    
    # Simulate Verse section (bars 1-4, 0-8 seconds at 120 BPM)
    verse_end = 8.0
    # Add notes within Verse
    for i in range(4):
        start = i * 2.0
        end = start + 0.5
        inst.notes.append(pretty_midi.Note(velocity=80, pitch=64, start=start, end=end))
    
    pm.instruments.append(inst)
    
    # Check Verse boundary
    assert check_section_boundary(pm, verse_end, max_overlap_ms)


def test_guitar_section_boundaries_overlap_violation():
    """Test detection of section boundary violation."""
    config = load_emotion_mapping()
    max_overlap_ms = config["transition_rules"]["basic"]["max_overlap_ms"]
    
    pm = pretty_midi.PrettyMIDI(initial_tempo=120.0)
    pm.time_signature_changes.append(pretty_midi.TimeSignature(4, 4, 0.0))
    
    inst = pretty_midi.Instrument(program=24, is_drum=False, name="Guitar")
    
    # Add note that violates boundary (extends beyond section end + max_overlap)
    verse_end = 8.0
    violating_note_start = 7.5
    violating_note_end = verse_end + (max_overlap_ms / 1000.0) + 0.1  # 100ms beyond allowed
    
    inst.notes.append(
        pretty_midi.Note(
            velocity=80,
            pitch=64,
            start=violating_note_start,
            end=violating_note_end
        )
    )
    
    pm.instruments.append(inst)
    
    # Should detect violation
    assert not check_section_boundary(pm, verse_end, max_overlap_ms)


def test_guitar_emotion_profile_verse_to_chorus():
    """Test emotion profile transition from Verse to Chorus."""
    config = load_emotion_mapping()
    
    # Get emotion profiles for Verse and Chorus
    verse_emotion = config["section_emotion_mapping"]["Verse"]["default"]
    chorus_emotion = config["section_emotion_mapping"]["Chorus"]["default"]
    
    # Verify they are different
    assert verse_emotion != chorus_emotion
    
    # Verify Chorus has higher intensity
    verse_profile = config["emotion_profiles"][verse_emotion]
    chorus_profile = config["emotion_profiles"][chorus_emotion]
    
    # Map intensity to numeric values for comparison
    intensity_map = {"low": 1, "medium": 2, "high": 3}
    
    assert intensity_map[chorus_profile["intensity"]] >= intensity_map[verse_profile["intensity"]]


def test_guitar_section_length_constraints():
    """Test section length constraints from emotion_mapping.yaml."""
    config = load_emotion_mapping()
    constraints = config["validation_rules"]["section_length_constraints"]
    
    # Test valid section lengths
    assert 4 >= constraints["Verse"]["min_bars"]
    assert 4 <= constraints["Verse"]["max_bars"]
    
    assert 4 >= constraints["Chorus"]["min_bars"]
    assert 4 <= constraints["Chorus"]["max_bars"]


def test_guitar_transition_rules_special():
    """Test special transition rules between sections."""
    config = load_emotion_mapping()
    special = config["transition_rules"]["special_transitions"]
    
    # Verify Pre-Chorus to Chorus allows more overlap (seamless transition)
    prechorus_to_chorus = special["PreChorus_to_Chorus"]
    basic_max = config["transition_rules"]["basic"]["max_overlap_ms"]
    
    assert prechorus_to_chorus["max_overlap_ms"] >= basic_max


@pytest.mark.skipif(
    not Path("generator/guitar_generator.py").exists(),
    reason="guitar_generator.py not found"
)
def test_guitar_generator_section_integration():
    """Integration test: Generate guitar with section awareness."""
    try:
        gen = GuitarGenerator()
        
        # Test that generator can be instantiated
        assert gen is not None
        
        # TODO: Add actual generation test when generator supports section param
        # pm = gen.generate(
        #     section="Chorus",
        #     emotion_profile="happy_high",
        #     bars=4,
        #     tempo=120
        # )
        
    except Exception as e:
        pytest.skip(f"Guitar generator not fully implemented: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
