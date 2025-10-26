#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/test_v4_improvements.py

v4.0改善機能の簡易テスト
"""
from pathlib import Path
import json
import subprocess
import time

print("="*70)
print("CHORD RECOGNITION SYSTEM v4.0 - IMPROVEMENTS TEST")
print("="*70)

stems_dir = "data/suno_ai/suno_themesong/song_001/stemswav_001"
sections = "data/suno_ai/suno_themesong/song_001/analysis/sections.json"
output_dir = Path("results/v4_test")
output_dir.mkdir(parents=True, exist_ok=True)

# Test 1: Cached version (speed test)
print("\n[Test 1] Cached version speed test")
print("-" * 70)

# First run
print("  First run (no cache)...")
start = time.time()
PYTHON = "/Volumes/SSD-SCTU3A/ラジオ用/music_21/composer2-3/.venv311/bin/python"

cmd1 = [
    PYTHON, "ops/stem_harmony_cached.py",
    "--stems", stems_dir,
    "--out", str(output_dir / "chordmap_cached_1st.json"),
    "--sections", sections,
    "--exclude", "Vocals",
    "--force-key", "C",
    "--no-progress"
]
result1 = subprocess.run(cmd1, capture_output=True, text=True, timeout=180)
elapsed1 = time.time() - start

if result1.returncode == 0:
    print(f"  ✓ First run completed: {elapsed1:.1f}s")
    
    # Check cache file
    cache_dir = Path(stems_dir) / ".cache"
    if cache_dir.exists():
        cache_files = list(cache_dir.glob("*.npz"))
        print(f"  ✓ Cache created: {len(cache_files)} files")
    
    # Second run (with cache)
    print("  Second run (with cache)...")
    start = time.time()
    cmd2 = [
        PYTHON, "ops/stem_harmony_cached.py",
        "--stems", stems_dir,
        "--out", str(output_dir / "chordmap_cached_2nd.json"),
        "--sections", sections,
        "--exclude", "Vocals",
        "--force-key", "C",
        "--no-progress"
    ]
    result2 = subprocess.run(cmd2, capture_output=True, text=True, timeout=180)
    elapsed2 = time.time() - start
    
    if result2.returncode == 0:
        print(f"  ✓ Second run completed: {elapsed2:.1f}s")
        speedup = elapsed1 / elapsed2 if elapsed2 > 0 else 0
        print(f"  ✓ Speedup: {speedup:.1f}x faster")
    else:
        print(f"  ✗ Second run failed")
else:
    print(f"  ✗ First run failed")

# Test 2: 7th enhanced version
print("\n[Test 2] 7th chords enhanced version")
print("-" * 70)

cmd_7th = [
    PYTHON, "ops/stem_harmony_7th_v2.py",
    "--stems", stems_dir,
    "--out", str(output_dir / "chordmap_7th_v2.json"),
    "--sections", sections,
    "--exclude", "Vocals",
    "--force-key", "C"
]
result_7th = subprocess.run(cmd_7th, capture_output=True, text=True, timeout=180)

if result_7th.returncode == 0:
    # Check output
    out_path = output_dir / "chordmap_7th_v2.json"
    if out_path.exists():
        data = json.loads(out_path.read_text())
        print(f"  ✓ Generated {len(data)} events")
        
        # Check for 7th chords
        has_7th = any('7' in e['chord'] or 'm7' in e['chord'] for e in data)
        print(f"  ✓ Contains 7th chords: {has_7th}")
        
        # Show first 5 chords
        chords = [e['chord'] for e in data[:5]]
        print(f"  First 5 chords: {chords}")
    else:
        print(f"  ✗ Output file not found")
else:
    print(f"  ✗ 7th enhanced failed: {result_7th.stderr[:100]}")

# Test 3: Extended chords version
print("\n[Test 3] Extended chords version (sus4/add9/6th)")
print("-" * 70)

cmd_ext = [
    PYTHON, "ops/stem_harmony_extended.py",
    "--stems", stems_dir,
    "--out", str(output_dir / "chordmap_extended.json"),
    "--sections", sections,
    "--exclude", "Vocals",
    "--force-key", "C"
]
result_ext = subprocess.run(cmd_ext, capture_output=True, text=True, timeout=180)

if result_ext.returncode == 0:
    out_path = output_dir / "chordmap_extended.json"
    if out_path.exists():
        data = json.loads(out_path.read_text())
        print(f"  ✓ Generated {len(data)} events")
        
        # Check for extended chords
        extended_types = ['sus4', 'sus2', 'add9', '6']
        has_extended = any(any(ext in e['chord'] for ext in extended_types) for e in data)
        print(f"  ✓ Contains extended chords: {has_extended}")
        
        # Show first 5 chords
        chords = [e['chord'] for e in data[:5]]
        print(f"  First 5 chords: {chords}")
    else:
        print(f"  ✗ Output file not found")
else:
    print(f"  ✗ Extended chords failed: {result_ext.stderr[:100]}")

# Test 4: Compare versions
print("\n[Test 4] Version comparison")
print("-" * 70)

versions = {
    "Cached": output_dir / "chordmap_cached_2nd.json",
    "7th Enhanced": output_dir / "chordmap_7th_v2.json",
    "Extended": output_dir / "chordmap_extended.json"
}

for name, path in versions.items():
    if path.exists():
        data = json.loads(path.read_text())
        
        # Handle both formats
        if isinstance(data, dict) and "events" in data:
            events = data["events"]
            chords = [f"{e['root']}{e.get('quality', '')}" for e in events]
        else:
            chords = [e["chord"] for e in data]
        
        print(f"  {name:15s}: {len(chords):3d} events")
    else:
        print(f"  {name:15s}: Not generated")

print("\n" + "="*70)
print("TEST COMPLETE")
print("="*70)
print("\nGenerated files:")
for path in output_dir.glob("*.json"):
    size = path.stat().st_size
    print(f"  {path.name:35s} ({size:,} bytes)")
