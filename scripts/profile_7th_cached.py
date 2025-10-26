#!/usr/bin/env python3
"""Profile 7th chord recognition with cache"""
import sys, time
sys.path.insert(0, ".")
from pathlib import Path

# Monkey patch to profile
original_viterbi = None
viterbi_time = 0

def profile_viterbi(*args, **kwargs):
    global viterbi_time
    t0 = time.time()
    result = original_viterbi(*args, **kwargs)
    viterbi_time += time.time() - t0
    return result

print("Profiling 7th chord recognition...")

# Import and patch
from ops import stem_harmony_7th
original_viterbi = stem_harmony_7th.viterbi
stem_harmony_7th.viterbi = profile_viterbi

# Simulate main with profiling
import argparse
sys.argv = [
    "stem_harmony_7th.py",
    "--stems", "data/suno_ai/suno_themesong/song_001/stemswav_001",
    "--out", "results/profile_test.json",
    "--sections", "data/suno_ai/suno_themesong/song_001/analysis/sections.json",
    "--exclude", "Vocals",
    "--force-key", "C"
]

t_total = time.time()
stem_harmony_7th.main()
t_total = time.time() - t_total

print(f"\n[PROFILE]")
print(f"  Total time: {t_total:.2f}s")
print(f"  Viterbi time: {viterbi_time:.2f}s ({100*viterbi_time/t_total:.1f}%)")
print(f"  Other time: {t_total - viterbi_time:.2f}s ({100*(t_total-viterbi_time)/t_total:.1f}%)")
