#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/quick_test_new_features.py

新機能の簡易テスト
"""
from pathlib import Path
import json

def _head_events(obj, n=3):
    """Return first n events from either a dict with "events"/"chords" or a list.
    - If obj is dict and has key "events": use that list.
    - If obj is dict and has key "chords" (7th-chords output): use that list.
    - If obj is a list: slice it directly.
    - Otherwise: return empty list.
    This keeps behavior stable and avoids slicing a dict.
    """
    try:
        if isinstance(obj, dict):
            ev = obj.get("events")
            if ev is None:
                ev = obj.get("chords")  # e.g., chordmap_7th.json format
            if isinstance(ev, list):
                return ev[:n]
            return []
        # list-like
        return obj[:n]
    except Exception:
        return []

print("="*60)
print("NEW FEATURES TEST - Chord Recognition System v3.0")
print("="*60)

# Test 1: --force-key option
print("\n[Test 1] --force-key option")
forced_path = Path("data/test_outputs/chordmap_forced_C.json")
if forced_path.exists():
    data = json.loads(forced_path.read_text())
    head = _head_events(data, 3)
    print(f"  ✓ Generated events with --force-key C")
    print(f"  First 3 events: {head}")
else:
    print(f"  ✗ File not found: {forced_path}")

# Test 2: 7th chords
print("\n[Test 2] 7th chords support")
seventh_path = Path("data/test_outputs/chordmap_7th.json")
if seventh_path.exists():
    data = json.loads(seventh_path.read_text())
    head = _head_events(data, 10)
    print(f"  ✓ Generated 7th chords output")
    print(f"  First 10 events: {head}")
    
    # Try to extract chords
    if isinstance(data, dict) and "events" in data:
        chords = [e.get('chord', 'N') for e in data["events"]]
    elif isinstance(data, list):
        chords = [e.get('chord', 'N') for e in data]
    else:
        chords = []
    
    if chords:
        has_7th = any('7' in c for c in chords)
        print(f"  Contains 7th chords: {has_7th}")
else:
    print(f"  ✗ File not found: {seventh_path}")

# Test 3: Compare forced vs auto
print("\n[Test 3] Key difference analysis")
auto_path = Path("data/suno_ai/suno_themesong/song_001/analysis/chordmap_new.json")
if forced_path.exists() and auto_path.exists():
    forced_data = json.loads(forced_path.read_text())
    auto_data = json.loads(auto_path.read_text())
    
    forced_events = _head_events(forced_data, 999)
    auto_events = _head_events(auto_data, 999)
    
    print(f"  Forced (C key): {len(forced_events)} events")
    print(f"  Auto detect: {len(auto_events)} events")
    
    # Extract roots
    def get_root(chord_str):
        if chord_str == "N":
            return "N"
        if len(chord_str) > 1 and chord_str[1] in ['#', '♯']:
            return chord_str[:2]
        return chord_str[0]
    
    forced_roots = [get_root(e['chord']) for e in forced_data[:5]]
    auto_roots = [get_root(e['chord']) for e in auto_data[:5]]
    
    print(f"  Forced roots (first 5): {forced_roots}")
    print(f"  Auto roots (first 5): {auto_roots}")
else:
    print(f"  ✗ Missing files for comparison")

print("\n" + "="*60)
print("TEST COMPLETE")
print("="*60)
