#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/final_v4_test.py

v4.0改善機能の最終検証
"""
import time
import subprocess
from pathlib import Path
import json

PYTHON = "/Volumes/SSD-SCTU3A/ラジオ用/music_21/composer2-3/.venv311/bin/python"

print("="*70)
print("CHORD RECOGNITION SYSTEM v4.0 - FINAL VERIFICATION")
print("="*70)

stems_dir = "data/suno_ai/suno_themesong/song_001/stemswav_001"
sections = "data/suno_ai/suno_themesong/song_001/analysis/sections.json"
output_dir = Path("results/v4_final")
output_dir.mkdir(parents=True, exist_ok=True)

# Test 1: Cached version speed test
print("\n[Test 1] Cache Mechanism (stem_harmony_cached.py)")
print("-" * 70)

# Clear cache for clean test
cache_dir = Path(stems_dir) / ".cache"
if cache_dir.exists():
    for f in cache_dir.glob("*.npz"):
        f.unlink()
    print("  Cache cleared for clean test")

# First run
print("  Run 1 (no cache)...")
cmd1 = [
    PYTHON, "ops/stem_harmony_cached.py",
    "--stems", stems_dir,
    "--out", str(output_dir / "cached_1st.json"),
    "--sections", sections,
    "--exclude", "Vocals",
    "--force-key", "C"
]
t0 = time.time()
subprocess.run(cmd1, capture_output=True, text=True, timeout=180, check=True)
elapsed1 = time.time() - t0

# Second run
print("  Run 2 (with cache)...")
cmd2 = [
    PYTHON, "ops/stem_harmony_cached.py",
    "--stems", stems_dir,
    "--out", str(output_dir / "cached_2nd.json"),
    "--sections", sections,
    "--exclude", "Vocals",
    "--force-key", "C"
]
t0 = time.time()
subprocess.run(cmd2, capture_output=True, text=True, timeout=180, check=True)
elapsed2 = time.time() - t0

speedup = elapsed1 / elapsed2 if elapsed2 > 0 else 0
print(f"  ✓ Run 1: {elapsed1:.2f}s")
print(f"  ✓ Run 2: {elapsed2:.2f}s")
print(f"  ✓ Speedup: {speedup:.1f}x faster")

# Test 2: 7th chords with cache
print("\n[Test 2] 7th Chords + Cache (stem_harmony_7th.py)")
print("-" * 70)

# Clear 7th cache
if cache_dir.exists():
    for f in cache_dir.glob("chroma_sync_*.npz"):
        if "7th" in f.name:
            f.unlink()

# First run
print("  Run 1 (no cache)...")
cmd_7th_1 = [
    PYTHON, "ops/stem_harmony_7th.py",
    "--stems", stems_dir,
    "--out", str(output_dir / "7th_1st.json"),
    "--sections", sections,
    "--exclude", "Vocals",
    "--force-key", "C"
]
t0 = time.time()
result = subprocess.run(cmd_7th_1, capture_output=True, text=True, timeout=300)
elapsed_7th_1 = time.time() - t0

if result.returncode == 0:
    # Second run
    print("  Run 2 (with cache)...")
    cmd_7th_2 = [
        PYTHON, "ops/stem_harmony_7th.py",
        "--stems", stems_dir,
        "--out", str(output_dir / "7th_2nd.json"),
        "--sections", sections,
        "--exclude", "Vocals",
        "--force-key", "C"
    ]
    t0 = time.time()
    subprocess.run(cmd_7th_2, capture_output=True, text=True, timeout=180, check=True)
    elapsed_7th_2 = time.time() - t0
    
    speedup_7th = elapsed_7th_1 / elapsed_7th_2 if elapsed_7th_2 > 0 else 0
    print(f"  ✓ Run 1: {elapsed_7th_1:.2f}s")
    print(f"  ✓ Run 2: {elapsed_7th_2:.2f}s")
    print(f"  ✓ Speedup: {speedup_7th:.1f}x faster")
    
    # Check 7th chords
    out_path = output_dir / "7th_2nd.json"
    if out_path.exists():
        data = json.loads(out_path.read_text())
        events = data.get("events", data if isinstance(data, list) else [])
        print(f"  ✓ Generated {len(events)} events")
        
        # Show first 5 chords
        chords = [e.get('chord', 'N') for e in events[:5]]
        print(f"  First 5 chords: {chords}")
else:
    print(f"  ✗ 7th version failed: {result.stderr[:100]}")

# Test 3: Extended chords
print("\n[Test 3] Extended Chords (stem_harmony_extended.py)")
print("-" * 70)

cmd_ext = [
    PYTHON, "ops/stem_harmony_extended.py",
    "--stems", stems_dir,
    "--out", str(output_dir / "extended.json"),
    "--sections", sections,
    "--exclude", "Vocals",
    "--force-key", "C"
]
result_ext = subprocess.run(cmd_ext, capture_output=True, text=True, timeout=180)

if result_ext.returncode == 0:
    out_path = output_dir / "extended.json"
    if out_path.exists():
        data = json.loads(out_path.read_text())
        events = data.get("events", data if isinstance(data, list) else [])
        print(f"  ✓ Generated {len(events)} events")
        
        # Check for extended chords
        chords = [e.get('chord', 'N') for e in events]
        extended_types = ['sus4', 'sus2', 'add9', '6']
        has_extended = any(any(ext in c for ext in extended_types) for c in chords)
        print(f"  ✓ Contains extended chords: {has_extended}")
        
        # Show first 5 chords
        print(f"  First 5 chords: {chords[:5]}")
else:
    print(f"  ✗ Extended version failed: {result_ext.stderr[:100]}")

# Test 4: Compare versions
print("\n[Test 4] Version Comparison")
print("-" * 70)

versions = {
    "Cached": output_dir / "cached_2nd.json",
    "7th Enhanced": output_dir / "7th_2nd.json",
    "Extended": output_dir / "extended.json"
}

for name, path in versions.items():
    if path.exists():
        data = json.loads(path.read_text())
        events = data.get("events", data if isinstance(data, list) else [])
        print(f"  {name:15s}: {len(events):3d} events")

print("\n" + "="*70)
print("VERIFICATION COMPLETE")
print("="*70)

print("\nSummary:")
print(f"  ✓ Cache mechanism: {speedup:.1f}x speedup")
if result.returncode == 0:
    print(f"  ✓ 7th chords cache: {speedup_7th:.1f}x speedup")
if result_ext.returncode == 0:
    print(f"  ✓ Extended chords: working")

print("\nGenerated files:")
for path in sorted(output_dir.glob("*.json")):
    size = path.stat().st_size
    print(f"  {path.name:25s} ({size:,} bytes)")
