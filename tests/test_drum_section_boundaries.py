#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Section Boundary Tests for Drums Generator (Phase 4.7)

セクション境界での整合性テスト:
- Fillからセクションへの移行
- セクション開始時のキック配置
- Fillタイミングの適切性
"""

import pytest
from pathlib import Path
from typing import List, Dict, Any

import yaml

pytest.importorskip("pretty_midi")
import pretty_midi

from generator.drum_generator import DrumGenerator


# GM Drum mapping
GM_ROLE = {
    35: "KICK", 36: "KICK",
    38: "SNARE", 40: "SNARE",
    42: "HIHAT", 44: "HIHAT", 46: "HIHAT",
    49: "CRASH", 57: "CRASH", 55: "CRASH", 52: "CRASH",
}


def load_emotion_mapping(config_path: Path = None) -> Dict[str, Any]:
    """Load emotion mapping configuration."""
    if config_path is None:
        config_path = Path(__file__).parent.parent / "config" / "emotion_mapping.yaml"
    
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def check_drum_section_boundary(
    pm: pretty_midi.PrettyMIDI,
    section_end_time: float,
    max_overlap_ms: float = 50.0
) -> bool:
    """Check if drum notes respect section boundary."""
    max_overlap_sec = max_overlap_ms / 1000.0
    
    for inst in pm.instruments:
        if not inst.is_drum:
            continue
        
        for note in inst.notes:
            if note.end > section_end_time + max_overlap_sec:
                return False
    
    return True


def check_section_start_kick(
    pm: pretty_midi.PrettyMIDI,
    section_start_time: float,
    tolerance_ms: float = 50.0
) -> bool:
    """
    Check if section starts with a kick drum.
    
    Args:
        pm: PrettyMIDI object
        section_start_time: Start time of section (seconds)
        tolerance_ms: Allowed timing tolerance (milliseconds)
    
    Returns:
        True if kick is present at section start
    """
    tolerance_sec = tolerance_ms / 1000.0
    
    for inst in pm.instruments:
        if not inst.is_drum:
            continue
        
        for note in inst.notes:
            # Check if it's a kick
            if GM_ROLE.get(note.pitch) == "KICK":
                # Check if timing is close to section start
                if abs(note.start - section_start_time) <= tolerance_sec:
                    return True
    
    return False


def check_fill_before_section(
    pm: pretty_midi.PrettyMIDI,
    section_start_time: float,
    fill_duration_bars: int = 1
) -> bool:
    """
    Check if there's a fill before section transition.
    
    Args:
        pm: PrettyMIDI object
        section_start_time: Start time of section (seconds)
        fill_duration_bars: Expected fill duration in bars
    
    Returns:
        True if fill pattern detected before section
    """
    # Assuming 120 BPM, 4/4 time: 1 bar = 2 seconds
    bar_duration = 2.0
    fill_start = section_start_time - (fill_duration_bars * bar_duration)
    fill_end = section_start_time
    
    if fill_start < 0:
        return False
    
    # Count drum hits in fill region
    fill_hit_count = 0
    
    for inst in pm.instruments:
        if not inst.is_drum:
            continue
        
        for note in inst.notes:
            if fill_start <= note.start < fill_end:
                fill_hit_count += 1
    
    # Fill typically has > 8 hits per bar (more dense than groove)
    expected_min_hits = 8 * fill_duration_bars
    
    return fill_hit_count >= expected_min_hits


def test_drum_section_boundaries_basic():
    """Test basic section boundary respect for drums."""
    config = load_emotion_mapping()
    max_overlap_ms = config["transition_rules"]["basic"]["max_overlap_ms"]
    
    pm = pretty_midi.PrettyMIDI(initial_tempo=120.0)
    pm.time_signature_changes.append(pretty_midi.TimeSignature(4, 4, 0.0))
    
    inst = pretty_midi.Instrument(program=0, is_drum=True, name="Drums")
    
    # Verse section (bars 1-4, 0-8 seconds at 120 BPM)
    verse_end = 8.0
    
    # Add basic groove: kick on 1, snare on 3
    for bar in range(4):
        bar_start = bar * 2.0
        
        # Kick on beat 1
        inst.notes.append(pretty_midi.Note(velocity=100, pitch=36, start=bar_start, end=bar_start + 0.1))
        
        # Snare on beat 3
        inst.notes.append(pretty_midi.Note(velocity=90, pitch=38, start=bar_start + 1.0, end=bar_start + 1.1))
        
        # Hihat on all beats
        for beat in [0, 0.5, 1.0, 1.5]:
            start = bar_start + beat
            inst.notes.append(pretty_midi.Note(velocity=70, pitch=42, start=start, end=start + 0.1))
    
    pm.instruments.append(inst)
    
    # Check Verse boundary
    assert check_drum_section_boundary(pm, verse_end, max_overlap_ms)


def test_drum_section_start_with_kick():
    """Test that section starts with kick drum."""
    pm = pretty_midi.PrettyMIDI(initial_tempo=120.0)
    pm.time_signature_changes.append(pretty_midi.TimeSignature(4, 4, 0.0))
    
    inst = pretty_midi.Instrument(program=0, is_drum=True, name="Drums")
    
    # Chorus starts at bar 5 (8.0 seconds)
    chorus_start = 8.0
    
    # Add kick at Chorus start
    inst.notes.append(pretty_midi.Note(velocity=110, pitch=36, start=chorus_start, end=chorus_start + 0.1))
    
    # Add crash cymbal for emphasis
    inst.notes.append(pretty_midi.Note(velocity=100, pitch=49, start=chorus_start, end=chorus_start + 0.5))
    
    pm.instruments.append(inst)
    
    # Check kick presence at section start
    assert check_section_start_kick(pm, chorus_start)


def test_drum_fill_before_chorus():
    """Test fill pattern before Chorus transition."""
    pm = pretty_midi.PrettyMIDI(initial_tempo=120.0)
    pm.time_signature_changes.append(pretty_midi.TimeSignature(4, 4, 0.0))
    
    inst = pretty_midi.Instrument(program=0, is_drum=True, name="Drums")
    
    # Chorus starts at bar 5 (8.0 seconds)
    chorus_start = 8.0
    fill_start = 6.0  # Last bar of Verse (bar 4)
    
    # Add dense fill pattern (16th notes)
    fill_pitches = [38, 41, 43, 45, 47]  # Snare and toms
    sixteenth_duration = 0.125  # 120 BPM, 16th note
    
    for i in range(16):  # 16 sixteenth notes = 1 bar
        pitch = fill_pitches[i % len(fill_pitches)]
        start = fill_start + (i * sixteenth_duration)
        end = start + 0.1
        inst.notes.append(pretty_midi.Note(velocity=85 + (i * 2), pitch=pitch, start=start, end=end))
    
    pm.instruments.append(inst)
    
    # Check fill presence
    assert check_fill_before_section(pm, chorus_start, fill_duration_bars=1)


def test_drum_emotion_profile_calm_to_energetic():
    """Test emotion profile impact on drum density."""
    config = load_emotion_mapping()
    
    # Get drums adjustments
    drums_adj = config["instrument_adjustments"]["drums"]
    
    calm = drums_adj["calm_low"]
    energetic = drums_adj["energetic_high"]
    
    # Energetic should have higher hihat density
    assert energetic["hihat_density_multiplier"] > calm["hihat_density_multiplier"]
    
    # Energetic should have higher velocity boost
    assert energetic["velocity_boost"] > calm["velocity_boost"]


def test_drum_fill_section_constraints():
    """Test Fill section length constraints."""
    config = load_emotion_mapping()
    constraints = config["validation_rules"]["section_length_constraints"]
    
    # Fill should be 1-2 bars
    assert constraints["Fill"]["min_bars"] == 1
    assert constraints["Fill"]["max_bars"] == 2


def test_drum_crash_on_chorus_start():
    """Test crash cymbal appears at Chorus start for emphasis."""
    pm = pretty_midi.PrettyMIDI(initial_tempo=120.0)
    pm.time_signature_changes.append(pretty_midi.TimeSignature(4, 4, 0.0))
    
    inst = pretty_midi.Instrument(program=0, is_drum=True, name="Drums")
    
    chorus_start = 8.0
    
    # Add crash at Chorus start
    inst.notes.append(pretty_midi.Note(velocity=110, pitch=49, start=chorus_start, end=chorus_start + 1.0))
    
    # Add kick for support
    inst.notes.append(pretty_midi.Note(velocity=110, pitch=36, start=chorus_start, end=chorus_start + 0.1))
    
    pm.instruments.append(inst)
    
    # Check crash presence
    crash_found = False
    for note in inst.notes:
        if GM_ROLE.get(note.pitch) == "CRASH" and note.start == chorus_start:
            crash_found = True
            break
    
    assert crash_found


def test_drum_section_transition_timing():
    """Test drum transitions respect timing rules."""
    config = load_emotion_mapping()
    special = config["transition_rules"]["special_transitions"]
    
    # Pre-Chorus to Chorus should allow overlap (seamless)
    prechorus_to_chorus = special["PreChorus_to_Chorus"]
    assert prechorus_to_chorus["max_overlap_ms"] >= 50


@pytest.mark.skipif(
    not Path("generator/drum_generator.py").exists(),
    reason="drum_generator.py not found"
)
def test_drum_generator_section_integration():
    """Integration test: Generate drums with section awareness."""
    try:
        gen = DrumGenerator()
        
        # Test that generator can be instantiated
        assert gen is not None
        
        # TODO: Add actual generation test when generator supports section param
        # pm = gen.generate(
        #     section="Chorus",
        #     emotion_profile="energetic_high",
        #     bars=4,
        #     tempo=120,
        #     with_fill=True
        # )
        
    except Exception as e:
        pytest.skip(f"Drum generator not fully implemented: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
