#!/usr/bin/env python3
"""Tests for lamda_v2.stage2_extractor module (Phase2-1)."""
from __future__ import annotations
import json
import tempfile
from pathlib import Path
import pytest

try:
    import pretty_midi
    import mido
except ImportError:
    pytest.skip("pretty_midi or mido not available", allow_module_level=True)

from scripts.lamda_v2.stage2_extractor import (
    extract_stage2_metadata,
    extract_to_json,
    batch_extract,
    SCHEMA_VERSION,
)


def _create_test_midi(path: Path, tempo: float = 120.0, bars: int = 4):
    """Create a simple test MIDI file with notes."""
    mid = mido.MidiFile()
    track = mido.MidiTrack()
    mid.tracks.append(track)
    
    # Set tempo and time signature
    track.append(mido.MetaMessage('set_tempo', tempo=mido.bpm2tempo(tempo), time=0))
    track.append(mido.MetaMessage('time_signature', numerator=4, denominator=4, time=0))
    
    # Add notes (C major chord at each bar)
    ticks_per_bar = mid.ticks_per_beat * 4
    for b in range(bars):
        # C major triad: C, E, G
        for pitch in [60, 64, 67]:
            track.append(mido.Message('note_on', note=pitch, velocity=64, time=0 if pitch == 60 else 0))
        track.append(mido.Message('note_off', note=60, velocity=0, time=100))
        track.append(mido.Message('note_off', note=64, velocity=0, time=0))
        track.append(mido.Message('note_off', note=67, velocity=0, time=ticks_per_bar - 100 if b < bars - 1 else 0))
    
    mid.save(str(path))


def test_extract_stage2_metadata_basic():
    """Test basic metadata extraction from MIDI file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        midi_path = Path(tmpdir) / "test.mid"
        _create_test_midi(midi_path, tempo=120.0, bars=4)
        
        meta = extract_stage2_metadata(midi_path)
        
        # Check schema version
        assert meta["schema_version"] == SCHEMA_VERSION
        
        # Check tempo_map exists
        assert "tempo_map" in meta
        assert isinstance(meta["tempo_map"], list)
        assert len(meta["tempo_map"]) > 0
        
        # Check downbeats
        assert "downbeats_ql" in meta
        assert isinstance(meta["downbeats_ql"], list)
        
        # Check chordmap
        assert "chordmap" in meta
        assert meta["chordmap"]["unit"] == "ql"
        
        # Check key_hint
        assert "key_hint" in meta
        assert isinstance(meta["key_hint"], list)
        
        # Check modulations
        assert "modulations" in meta
        assert isinstance(meta["modulations"], list)


def test_extract_stage2_metadata_error_handling():
    """Test error handling for non-existent file."""
    meta = extract_stage2_metadata(Path("/nonexistent/file.mid"))
    
    assert meta["schema_version"] == f"{SCHEMA_VERSION}_error"
    assert "error" in meta
    assert "not found" in meta["error"].lower()


def test_extract_to_json():
    """Test JSON file output."""
    with tempfile.TemporaryDirectory() as tmpdir:
        midi_path = Path(tmpdir) / "test.mid"
        _create_test_midi(midi_path, tempo=120.0, bars=4)
        
        json_path = extract_to_json(midi_path)
        
        # Check JSON file was created
        assert json_path.exists()
        assert json_path.name == "test.stage2.json"
        
        # Check JSON content
        with json_path.open("r") as f:
            data = json.load(f)
        
        assert data["schema_version"] == SCHEMA_VERSION
        assert "tempo_map" in data
        assert "chordmap" in data


def test_extract_to_json_custom_output():
    """Test JSON file output with custom path."""
    with tempfile.TemporaryDirectory() as tmpdir:
        midi_path = Path(tmpdir) / "test.mid"
        output_path = Path(tmpdir) / "custom_output.json"
        _create_test_midi(midi_path, tempo=120.0, bars=4)
        
        json_path = extract_to_json(midi_path, output_path)
        
        assert json_path == output_path
        assert json_path.exists()


def test_batch_extract():
    """Test batch extraction from directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        input_dir = Path(tmpdir) / "input"
        output_dir = Path(tmpdir) / "output"
        input_dir.mkdir()
        
        # Create multiple test MIDI files
        for i in range(3):
            _create_test_midi(input_dir / f"test_{i}.mid", tempo=120.0, bars=4)
        
        json_paths = batch_extract(input_dir, output_dir)
        
        # Check all files were processed
        assert len(json_paths) == 3
        assert all(p.exists() for p in json_paths)
        
        # Check output directory structure
        assert output_dir.exists()
        assert len(list(output_dir.glob("*.json"))) == 3


def test_payload_structure():
    """Test that payload contains all required fields."""
    with tempfile.TemporaryDirectory() as tmpdir:
        midi_path = Path(tmpdir) / "test.mid"
        _create_test_midi(midi_path, tempo=120.0, bars=4)
        
        meta = extract_stage2_metadata(midi_path)
        
        # Required fields
        required_fields = [
            "schema_version",
            "tempo_map",
            "timesig_map",
            "downbeats_sec",
            "downbeats_ql",
            "chordmap",
            "key_hint",
            "modulations",
            "sections_auto",
            "groove",
            "controls",
        ]
        
        for field in required_fields:
            assert field in meta, f"Missing required field: {field}"


def test_integration_with_chord_and_key_analyzers():
    """Test that chord and key analyzers are properly integrated."""
    with tempfile.TemporaryDirectory() as tmpdir:
        midi_path = Path(tmpdir) / "test.mid"
        _create_test_midi(midi_path, tempo=120.0, bars=8)
        
        meta = extract_stage2_metadata(midi_path)
        
        # Check chordmap integration
        chordmap = meta["chordmap"]
        assert chordmap["unit"] == "ql"
        assert "events" in chordmap
        
        # If events exist, check structure
        if chordmap["events"]:
            event = chordmap["events"][0]
            assert "time" in event
            assert "root" in event
            assert "quality" in event
            assert "confidence" in event
        
        # Check key_hint integration
        key_hint = meta["key_hint"]
        if key_hint:
            # Each entry should be [bar_index, key_name]
            assert isinstance(key_hint[0], list)
            assert len(key_hint[0]) == 2
            assert isinstance(key_hint[0][0], int)  # bar index
            assert isinstance(key_hint[0][1], str)  # key name
