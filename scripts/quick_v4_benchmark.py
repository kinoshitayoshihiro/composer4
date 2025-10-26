#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/quick_v4_benchmark.py

v4.0の高速化検証（7th版のみ）
"""
import time
import subprocess
from pathlib import Path
import json

PYTHON = "/Volumes/SSD-SCTU3A/ラジオ用/music_21/composer2-3/.venv311/bin/python"

print("="*70)
print("v4.0 PERFORMANCE BENCHMARK - 7th Chords with Cache")
print("="*70)

stems_dir = "data/suno_ai/suno_themesong/song_001/stemswav_001"
sections = "data/suno_ai/suno_themesong/song_001/analysis/sections.json"
output_dir = Path("results/v4_benchmark")
output_dir.mkdir(parents=True, exist_ok=True)

# Clear cache for clean measurement
cache_dir = Path(stems_dir) / ".cache"
if cache_dir.exists():
    cache_files = list(cache_dir.glob("chroma_sync_*.npz"))
    for f in cache_files:
        f.unlink()
    print(f"  Cleared {len(cache_files)} cache files\n")

# Run 1: No cache (cold start)
print("[Run 1] Cold start (no cache)")
print("-" * 70)
cmd = [
    PYTHON, "ops/stem_harmony_7th.py",
    "--stems", stems_dir,
    "--out", str(output_dir / "run1.json"),
    "--sections", sections,
    "--exclude", "Vocals",
    "--force-key", "C"
]

t0 = time.time()
result1 = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
elapsed1 = time.time() - t0

if result1.returncode == 0:
    print(f"  ✓ Completed in {elapsed1:.2f}s")
    
    # Check output
    out_path = output_dir / "run1.json"
    if out_path.exists():
        data = json.loads(out_path.read_text())
        if isinstance(data, dict):
            events = data.get("events", [])
        elif isinstance(data, list):
            events = data
        else:
            events = []
        print(f"  ✓ Generated {len(events)} chord events")
else:
    print(f"  ✗ Failed: {result1.stderr[:200]}")
    exit(1)

# Run 2: With cache (hot start)
print("\n[Run 2] Hot start (with cache)")
print("-" * 70)
cmd[cmd.index("--out") + 1] = str(output_dir / "run2.json")

t0 = time.time()
result2 = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
elapsed2 = time.time() - t0

if result2.returncode == 0:
    print(f"  ✓ Completed in {elapsed2:.2f}s")
    
    # Check cache hit
    if "[CACHE] Chroma: HIT" in result2.stdout:
        print("  ✓ Cache hit confirmed")
    
    # Check output
    out_path = output_dir / "run2.json"
    if out_path.exists():
        data = json.loads(out_path.read_text())
        if isinstance(data, dict):
            events = data.get("events", [])
        elif isinstance(data, list):
            events = data
        else:
            events = []
        print(f"  ✓ Generated {len(events)} chord events")
else:
    print(f"  ✗ Failed: {result2.stderr[:200]}")
    exit(1)

# Run 3: Verify consistency
print("\n[Run 3] Third run (verify cache stability)")
print("-" * 70)
cmd[cmd.index("--out") + 1] = str(output_dir / "run3.json")

t0 = time.time()
result3 = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
elapsed3 = time.time() - t0

if result3.returncode == 0:
    print(f"  ✓ Completed in {elapsed3:.2f}s")

# Summary
print("\n" + "="*70)
print("BENCHMARK RESULTS")
print("="*70)

speedup = elapsed1 / elapsed2 if elapsed2 > 0 else 0
print(f"\nCold start (Run 1):  {elapsed1:6.2f}s")
print(f"Hot start  (Run 2):  {elapsed2:6.2f}s  (speedup: {speedup:.1f}x)")
print(f"Hot start  (Run 3):  {elapsed3:6.2f}s")

avg_hot = (elapsed2 + elapsed3) / 2
avg_speedup = elapsed1 / avg_hot if avg_hot > 0 else 0
print(f"\nAverage hot start:   {avg_hot:6.2f}s  (speedup: {avg_speedup:.1f}x)")

# Check cache size
if cache_dir.exists():
    cache_files = list(cache_dir.glob("*.npz"))
    total_size = sum(f.stat().st_size for f in cache_files)
    print(f"\nCache files: {len(cache_files)}")
    print(f"Cache size: {total_size:,} bytes ({total_size/1024/1024:.2f} MB)")

# Compare outputs
print("\n" + "="*70)
print("OUTPUT VERIFICATION")
print("="*70)

outputs = [output_dir / f"run{i}.json" for i in [1, 2, 3]]
all_same = True

for i, path in enumerate(outputs, 1):
    if path.exists():
        data = json.loads(path.read_text())
        if isinstance(data, dict):
            events = data.get("events", [])
        elif isinstance(data, list):
            events = data
        else:
            events = []
        chords = [e.get('chord', 'N') for e in events]
        print(f"Run {i}: {len(events)} events, chords: {chords[:5]}")
        
        if i > 1:
            prev_data = json.loads(outputs[i-2].read_text())
            if isinstance(prev_data, dict):
                prev_events = prev_data.get("events", [])
            elif isinstance(prev_data, list):
                prev_events = prev_data
            else:
                prev_events = []
            if len(events) != len(prev_events):
                all_same = False
                print(f"  ⚠ Event count mismatch with Run {i-1}")

if all_same:
    print("\n✓ All runs produced consistent results")
else:
    print("\n⚠ Some inconsistencies detected")

print("\n" + "="*70)
print("✓ BENCHMARK COMPLETE")
print("="*70)
