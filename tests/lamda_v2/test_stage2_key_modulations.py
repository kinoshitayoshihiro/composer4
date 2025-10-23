#!/usr/bin/env python3
"""Tests for lamda_v2.key_analyzer module (Phase2)."""
from __future__ import annotations
from scripts.lamda_v2.key_analyzer import estimate_local_key_sequence, to_key_hints_payload


def _cm(root: str, bars: int) -> dict:
    """Helper: synthesize a chordmap with bar-aligned chords of the same root."""
    events = [{"time": float(b * 4.0), "root": root, "quality": "maj"} for b in range(bars)]
    return {"unit": "ql", "events": events}


def test_constant_c_major():
    """Test constant C major key detection (no modulation)."""
    cm = _cm("C", 16)
    seq = estimate_local_key_sequence(cm, win_bars=4, min_hold=4, ks_weight=0.7)
    assert len(seq["keys"]) == 16
    assert all(k == "C" for k in seq["keys"]), "Expected stable C major"
    
    payload = to_key_hints_payload(seq)
    assert payload["key_hint"][0] == [0, "C"]
    assert payload["modulations"] == [], "No modulations expected"


def test_simple_modulation_c_to_g():
    """Test simple C→G modulation detection."""
    # 8 bars of C, then 8 bars of G → modulation around bar 8
    events = []
    for b in range(8):
        events.append({"time": float(b * 4.0), "root": "C", "quality": "maj"})
    for b in range(8, 16):
        events.append({"time": float(b * 4.0), "root": "G", "quality": "maj"})
    cm = {"unit": "ql", "events": events}

    seq = estimate_local_key_sequence(cm, win_bars=4, min_hold=4, ks_weight=0.7)
    keys = seq["keys"]
    assert keys[:6].count("C") >= 5, "Early bars should be C-dominant"
    assert keys[-6:].count("G") >= 5, "Late bars should be G-dominant"
    
    mods = seq["modulations"]
    assert len(mods) >= 1, "Expected at least one modulation"
    # First modulation should occur at or after bar 8 (time=32QL)
    assert mods[0]["time"] >= 32.0, f"Modulation time {mods[0]['time']} too early"


def test_debounce_short_fluctuation():
    """Test debouncing of short key fluctuations (min_hold enforcement)."""
    # C (3 bars) -> G (1 bar spike) -> C (rest)
    # with min_hold=4, the spike should be ignored
    events = []
    for b in range(3):
        events.append({"time": float(b * 4.0), "root": "C", "quality": "maj"})
    events.append({"time": float(3 * 4.0), "root": "G", "quality": "maj"})
    for b in range(4, 12):
        events.append({"time": float(b * 4.0), "root": "C", "quality": "maj"})
    cm = {"unit": "ql", "events": events}

    seq = estimate_local_key_sequence(cm, win_bars=4, min_hold=4, ks_weight=0.7)
    mods = seq["modulations"]
    # No modulation due to debouncing of short spike
    assert len(mods) == 0, f"Expected no modulations, got {mods}"


def test_empty_chordmap():
    """Test handling of empty chordmap."""
    cm = {"unit": "ql", "events": []}
    seq = estimate_local_key_sequence(cm, win_bars=4, min_hold=4, ks_weight=0.7)
    assert seq["keys"] == []
    assert seq["modulations"] == []


def test_payload_format():
    """Test payload formatting for Stage2 integration."""
    cm = _cm("D", 8)
    seq = estimate_local_key_sequence(cm, win_bars=4, min_hold=4, ks_weight=0.7)
    payload = to_key_hints_payload(seq)
    
    # Check key_hint format: [[bar, key], ...]
    assert isinstance(payload["key_hint"], list)
    assert all(isinstance(kh, list) and len(kh) == 2 for kh in payload["key_hint"])
    
    # Check modulations format: [{"time": ql, "to": key}, ...]
    assert isinstance(payload["modulations"], list)
    for mod in payload["modulations"]:
        assert "time" in mod
        assert "to" in mod
        assert isinstance(mod["time"], (int, float))
        assert isinstance(mod["to"], str)
