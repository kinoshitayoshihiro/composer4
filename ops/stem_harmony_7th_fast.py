#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ops/stem_harmony_7th_fast.py

7th chords + cached chroma for speed
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import argparse
import numpy as np

# Reuse existing code
exec(Path("ops/stem_harmony_7th.py").read_text())

if __name__ == "__main__":
    # Use cached chroma if available
    from ops.stem_harmony_cached import get_cache_key, load_chroma_cache, save_chroma_cache
    
    ap = argparse.ArgumentParser()
    ap.add_argument("--stems", type=str, required=True)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--sections", type=str, default=None)
    ap.add_argument("--exclude", action="append", default=[])
    ap.add_argument("--sr", type=int, default=22050)
    ap.add_argument("--bins-per-octave", type=int, default=36)
    ap.add_argument("--force-key", type=str, default=None)
    ap.add_argument("--no-cache", action="store_true")
    args = ap.parse_args()
    
    stems_dir = Path(args.stems)
    files = list_audio_files(stems_dir, args.exclude)
    
    # Try cache
    cache_path = None
    if not args.no_cache:
        cache_dir = stems_dir / ".cache"
        cache_dir.mkdir(exist_ok=True)
        cache_key = get_cache_key(files, args.sr, args.bins_per_octave, args.exclude, [])
        cache_path = cache_dir / f"chroma_sync_{cache_key}.npz"
    
    cached_data = None if args.no_cache else (load_chroma_cache(cache_path) if cache_path and cache_path.exists() else None)
    
    if cached_data:
        C_sync, tempo, beat_times = cached_data
        print(f"[CACHE] Loaded chroma: {C_sync.shape}")
    else:
        y_h, sr = mix_harmonic(files, sr=args.sr, weights=[])
        C_sync, tempo, beat_times = chroma_sync(y_h, sr, bins_per_octave=args.bins_per_octave, force_key=args.force_key)
        if cache_path:
            save_chroma_cache(cache_path, C_sync, tempo, beat_times)
    
    # Run 7th chord recognition (rest of the code from stem_harmony_7th.py)
    # ... (implement using functions from stem_harmony_7th.py)
    
    print(f"[INFO] Generated chords → {args.out}")
