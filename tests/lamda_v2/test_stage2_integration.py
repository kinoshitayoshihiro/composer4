#!/usr/bin/env python3
"""Integration tests for Stage2 extractor (LAMDA v2.6+)."""

from __future__ import annotations
import pretty_midi as pm
from scripts.lamda_v2.stage2_extractor import extract_stage2_metadata
from pathlib import Path


def _create_test_midi() -> pm.PrettyMIDI:
    """Create minimal test MIDI."""
    midi = pm.PrettyMIDI()
    inst = pm.Instrument(program=0)
    
    # Simple melody (C major scale)
    for i, pitch in enumerate([60, 62, 64, 65, 67, 69, 71, 72]):
        note = pm.Note(
            velocity=80,
            pitch=pitch,
            start=float(i * 0.5),
            end=float(i * 0.5 + 0.4),
        )
        inst.notes.append(note)
    
    midi.instruments.append(inst)
    return midi


def test_timesig_map_time_field():
    """Test that timesig_map_time field is present in payload."""
    midi = _create_test_midi()
    
    # Save to temp file
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as f:
        midi.write(f.name)
        temp_path = Path(f.name)
    
    try:
        # Extract metadata
        meta = extract_stage2_metadata(temp_path)
        
        # Check schema version
        assert meta["schema_version"].startswith("lamda_v2.6")
        
        # Check timesig_map_time exists
        assert "timesig_map_time" in meta, "timesig_map_time field missing"
        
        # Check format: [(time_sec, "4/4"), ...]
        timesig_map_time = meta["timesig_map_time"]
        assert isinstance(timesig_map_time, list)
        assert len(timesig_map_time) > 0
        
        # First entry should be (0.0, "4/4")
        first = timesig_map_time[0]
        assert isinstance(first, (list, tuple))
        assert len(first) == 2
        assert isinstance(first[0], (int, float))
        assert isinstance(first[1], str)
        
        # Backward compatibility: timesig_map still present
        assert "timesig_map" in meta
        assert isinstance(meta["timesig_map"], list)
        
    finally:
        # Cleanup
        temp_path.unlink(missing_ok=True)


def test_full_payload_structure():
    """Test complete payload structure with all Phase2 fields."""
    midi = _create_test_midi()
    
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as f:
        midi.write(f.name)
        temp_path = Path(f.name)
    
    try:
        meta = extract_stage2_metadata(temp_path)
        
        # Required fields
        required = [
            "schema_version",
            "tempo_map",
            "timesig_map",
            "timesig_map_time",
            "downbeats_sec",
            "downbeats_ql",
            "chordmap",
            "key_hint",
            "modulations",
            "sections_auto",
            "groove",
            "controls",
        ]
        
        for field in required:
            assert field in meta, f"Missing required field: {field}"
        
        # Type checks
        assert isinstance(meta["tempo_map"], list)
        assert isinstance(meta["timesig_map"], list)
        assert isinstance(meta["timesig_map_time"], list)
        assert isinstance(meta["downbeats_sec"], list)
        assert isinstance(meta["downbeats_ql"], list)
        assert isinstance(meta["chordmap"], dict)
        assert isinstance(meta["key_hint"], list)
        assert isinstance(meta["modulations"], list)
        assert isinstance(meta["sections_auto"], dict)
        
    finally:
        temp_path.unlink(missing_ok=True)
