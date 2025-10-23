#!/usr/bin/env python3
"""Tests for lamda_v2.chord_analyzer module (Phase2)."""
from __future__ import annotations
import pytest

try:
    import pretty_midi
except ImportError:
    pytest.skip("pretty_midi not available", allow_module_level=True)

from scripts.lamda_v2.chord_analyzer import (
    extract_bar_chords,
    merge_consecutive_chords,
    enforce_min_dwell,
)


def _make_c_major_triad(start_sec: float, duration: float = 1.0) -> list:
    """Create C major triad notes (C, E, G) = (60, 64, 67)."""
    return [
        pretty_midi.Note(velocity=64, pitch=60, start=start_sec, end=start_sec + duration),
        pretty_midi.Note(velocity=64, pitch=64, start=start_sec, end=start_sec + duration),
        pretty_midi.Note(velocity=64, pitch=67, start=start_sec, end=start_sec + duration),
    ]


def _make_a_minor_triad(start_sec: float, duration: float = 1.0) -> list:
    """Create A minor triad notes (A, C, E) = (69, 60, 64)."""
    return [
        pretty_midi.Note(velocity=64, pitch=69, start=start_sec, end=start_sec + duration),
        pretty_midi.Note(velocity=64, pitch=60, start=start_sec, end=start_sec + duration),
        pretty_midi.Note(velocity=64, pitch=64, start=start_sec, end=start_sec + duration),
    ]


def test_merge_consecutive_chords():
    """Test merge_consecutive_chords() removes duplicates."""
    events = [
        {"time": 0.0, "root": "C", "quality": "maj"},
        {"time": 4.0, "root": "C", "quality": "maj"},
        {"time": 8.0, "root": "Am", "quality": "min"},
        {"time": 12.0, "root": "Am", "quality": "min"},
    ]
    
    merged = merge_consecutive_chords(events)
    
    assert len(merged) == 2
    assert merged[0]["root"] == "C"
    assert merged[1]["root"] == "Am"


def test_enforce_min_dwell():
    """Test enforce_min_dwell() removes short chords."""
    events = [
        {"time": 0.0, "root": "C"},
        {"time": 1.0, "root": "G"},
        {"time": 2.0, "root": "Am"},
        {"time": 6.0, "root": "F"},
    ]
    
    filtered = enforce_min_dwell(events, min_ql=2.0)
    
    roots = [e["root"] for e in filtered]
    assert filtered[-1]["root"] == "F"
