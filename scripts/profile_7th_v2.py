#!/usr/bin/env python3
"""Profile 7th v2 version"""
import sys
import time
sys.path.insert(0, ".")

import numpy as np
from pathlib import Path

print("Loading modules...")
from ops.stem_harmony_7th_v2 import (
    list_audio_files, mix_harmonic, chroma_sync,
    estimate_local_key_7th, build_loglik_7th_enhanced
)

stems_dir = Path("data/suno_ai/suno_themesong/song_001/stemswav_001")
files = list_audio_files(stems_dir, ["Vocals"])
print(f"Found {len(files)} files")

print("\n1. mix_harmonic...")
t0 = time.time()
y_h, sr = mix_harmonic(files, sr=22050, weights=[])
print(f"   {time.time() - t0:.2f}s")

print("\n2. chroma_sync...")
t0 = time.time()
C_sync, tempo, beat_times = chroma_sync(y_h, sr, bins_per_octave=36, force_key="C")
print(f"   {time.time() - t0:.2f}s")
print(f"   C_sync shape: {C_sync.shape}")

print("\n3. estimate_local_key_7th...")
t0 = time.time()
local_keys = estimate_local_key_7th(C_sync, window=8, agg_fn="gaussian")
print(f"   {time.time() - t0:.2f}s")
print(f"   local_keys shape: {local_keys.shape}")

print("\n4. build_loglik_7th_enhanced...")
t0 = time.time()
local_cfg = {"enable": True, "window": 8, "gamma": 0.3}
n_cfg = {"energy_gamma": 1.0, "conf_gamma": 2.0}
loglik = build_loglik_7th_enhanced(
    C_sync=C_sync,
    gamma_global=0.15,
    local_cfg=local_cfg,
    include_N=False,
    n_cfg=n_cfg,
    section_for_t=None
)
print(f"   {time.time() - t0:.2f}s")
print(f"   loglik shape: {loglik.shape}")

print("\nDONE")
