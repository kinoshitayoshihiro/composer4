#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ops/stem_harmony_cached.py  (Cached + Optimized version)

改善点:
- Chroma features キャッシュ（.npy形式で保存・再利用）
- 処理時間短縮（HPSS/CQT結果を再利用）
- 進捗表示追加（tqdm）
- 既存stem_harmony.pyと完全互換

キャッシュファイル:
  <stems_dir>/.cache/chroma_sync_<sr>_<bins>.npz
"""
from __future__ import annotations
import argparse, json, sys, math, hashlib
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import numpy as np
import librosa

try:
    import yaml  # type: ignore
except Exception:
    yaml = None

try:
    from tqdm import tqdm  # type: ignore
    HAS_TQDM = True
except Exception:
    HAS_TQDM = False
    tqdm = None

# Import all functions from stem_harmony.py (reuse existing code)
import sys
sys.path.insert(0, str(Path(__file__).parent))
from stem_harmony import (
    NOTE_NAMES, major_template, minor_template, rotate12,
    cos_sim_columns, list_audio_files, parse_stem_weights,
    load_sections_mapper_and_labeler, key_profile_major, key_profile_minor,
    build_transition, build_loglik, viterbi, path_to_events, save_chordmap,
    load_config, resolve_params_with_config
)

# ---------------- Cache utilities ----------------
def get_cache_key(files: List[Path], sr: int, bins_per_octave: int, excludes: List[str], weights: List[Tuple[str,float]]) -> str:
    """Generate unique cache key for chroma features"""
    # Sort files for deterministic key
    file_names = sorted([f.name for f in files])
    
    # Create hash from parameters
    params_str = f"{file_names}_{sr}_{bins_per_octave}_{excludes}_{weights}"
    return hashlib.md5(params_str.encode()).hexdigest()[:16]

def get_cache_path(stems_dir: Path, cache_key: str) -> Path:
    """Get cache file path"""
    cache_dir = stems_dir / ".cache"
    cache_dir.mkdir(exist_ok=True)
    return cache_dir / f"chroma_sync_{cache_key}.npz"

def load_chroma_cache(cache_path: Path) -> Optional[Tuple[np.ndarray, float, np.ndarray]]:
    """Load cached chroma features"""
    if not cache_path.exists():
        return None
    
    try:
        data = np.load(str(cache_path))
        C_sync = data['C_sync']
        tempo = float(data['tempo'])
        beat_times = data['beat_times']
        print(f"[CACHE] Loaded from {cache_path.name}")
        return C_sync, tempo, beat_times
    except Exception as e:
        print(f"[CACHE] Failed to load: {e}")
        return None

def save_chroma_cache(cache_path: Path, C_sync: np.ndarray, tempo: float, beat_times: np.ndarray):
    """Save chroma features to cache"""
    try:
        np.savez_compressed(
            str(cache_path),
            C_sync=C_sync,
            tempo=tempo,
            beat_times=beat_times
        )
        print(f"[CACHE] Saved to {cache_path.name}")
    except Exception as e:
        print(f"[CACHE] Failed to save: {e}")

# ---------------- Signal processing (with cache) ----------------
def mix_harmonic_cached(files: List[Path], sr: int, weights: List[Tuple[str,float]], use_tqdm: bool = True) -> Tuple[np.ndarray, int]:
    """Mix harmonic components with progress bar"""
    y_sum = None
    
    iterator = tqdm(files, desc="Loading stems") if use_tqdm and HAS_TQDM else files
    
    for fp in iterator:
        y, _sr = librosa.load(str(fp), sr=sr, mono=True)
        y_h, _ = librosa.effects.hpss(y)
        w = 1.0
        name = fp.name.lower()
        for key, val in weights:
            if key in name:
                w = float(val); break
        y_h = y_h * w
        if y_sum is None:
            y_sum = y_h
        else:
            if len(y_h) > len(y_sum):
                y_sum = np.pad(y_sum, (0, len(y_h)-len(y_sum)))
            elif len(y_h) < len(y_sum):
                y_h = np.pad(y_h, (0, len(y_sum)-len(y_h)))
            y_sum = y_sum + y_h
    
    if y_sum is None:
        raise RuntimeError("No usable audio files after excludes.")
    y_sum = y_sum / max(1.0, np.max(np.abs(y_sum)))
    return y_sum.astype(np.float32), sr

def chroma_sync_cached(y_h: np.ndarray, sr: int, bins_per_octave: int = 36, force_key: Optional[str] = None, use_tqdm: bool = True):
    """Chroma sync with progress indication"""
    if use_tqdm and HAS_TQDM:
        print("[Processing] Beat tracking...")
    
    tempo, beats = librosa.beat.beat_track(y=y_h, sr=sr, tightness=100, units='frames')
    if len(beats) == 0:
        onset_env = librosa.onset.onset_strength(y=y_h, sr=sr)
        beats = librosa.beat.onset_detect(onset_envelope=onset_env, sr=sr, units='frames')
        if len(beats) == 0:
            hop_length = 512
            n_frames = 1 + len(y_h)//hop_length
            step = int(max(1, (0.5*sr)//hop_length))
            beats = np.arange(0, n_frames, step, dtype=int)
            tempo = 120.0
    
    tempo = float(tempo[0]) if isinstance(tempo, np.ndarray) and tempo.ndim > 0 else float(tempo)
    beat_times = librosa.frames_to_time(beats, sr=sr)
    
    if force_key is not None:
        tuning = 0.0
        print(f"[INFO] Forcing key to {force_key}, tuning correction disabled")
    else:
        try:
            tuning = librosa.estimate_tuning(y=y_h, sr=sr)
        except Exception:
            tuning = 0.0
    
    if use_tqdm and HAS_TQDM:
        print("[Processing] Computing CQT chroma...")
    
    C = librosa.feature.chroma_cqt(y=y_h, sr=sr, bins_per_octave=bins_per_octave, tuning=tuning)
    C_sync = librosa.util.sync(C, beats, aggregate=np.median)
    
    return C_sync, tempo, beat_times

# ---------------- Main (cached version) ----------------
def main():
    ap = argparse.ArgumentParser(description="Cached chord recognition (faster with .npz cache)")
    ap.add_argument("--stems", required=True, help="Directory containing stem WAVs")
    ap.add_argument("--exclude", action="append", default=[], help="Substring to exclude (e.g., 'Vocals')")
    ap.add_argument("--out", required=True, help="Output chordmap.json path")
    ap.add_argument("--sections", help="sections.json path for QL mapping (optional)")
    ap.add_argument("--config", help="YAML/JSON config for priors/HMM/N-state/weights")
    ap.add_argument("--force-key", help="Force key (e.g., 'C', 'Am') - disables tuning correction")
    ap.add_argument("--sr", type=int, default=22050, help="Resample rate")
    ap.add_argument("--bins-per-octave", type=int, default=36, help="CQT bins per octave")
    ap.add_argument("--stay", type=float, default=0.93, help="HMM stay probability (chord states)")
    ap.add_argument("--near", type=float, default=0.03, help="HMM 4th/5th probability (per edge)")
    ap.add_argument("--include-N", action="store_true", help="Enable No-Chord state (overridden by YAML N_state.enable)")
    ap.add_argument("--n-stay", type=float, default=0.96, help="HMM stay for N state")
    ap.add_argument("--n-out", type=float, default=0.02, help="HMM N->chord probability per chord")
    ap.add_argument("--gamma-global", type=float, default=0.15, help="Global key prior gamma")
    ap.add_argument("--gamma-local", type=float, default=0.30, help="Local key prior gamma (default if YAML omitted)")
    ap.add_argument("--n-energy-gamma", type=float, default=1.0, help="No-Chord energy penalty gamma (default if YAML omitted)")
    ap.add_argument("--n-conf-gamma", type=float, default=2.0, help="No-Chord low-confidence gamma (default if YAML omitted)")
    ap.add_argument("--ql-per-beat", type=float, default=1.0, help="Fallback QL per beat if sections missing")
    ap.add_argument("--stem-weight", action="append", default=[], help="Per-stem weight like 'bass=1.3' (can repeat)")
    ap.add_argument("--no-cache", action="store_true", help="Disable cache (force recompute)")
    ap.add_argument("--no-progress", action="store_true", help="Disable progress bar (for batch processing)")
    args = ap.parse_args()

    cfg = load_config(Path(args.config)) if args.config else {}
    params = resolve_params_with_config(args, cfg)

    stems_dir = Path(args.stems)
    out_path  = Path(args.out)
    sections_path = Path(args.sections) if args.sections else None

    files = list_audio_files(stems_dir, args.exclude)
    if not files:
        print(f"[ERROR] No WAV files found in {stems_dir} (after excludes={args.exclude})", file=sys.stderr)
        sys.exit(2)

    # Cache key
    weights_cli = parse_stem_weights(args.stem_weight)
    weights_cfg = parse_stem_weights(params.get("stem_weight", []))
    all_weights = (weights_cfg or []) + (weights_cli or [])
    
    cache_key = get_cache_key(files, args.sr, args.bins_per_octave, args.exclude, all_weights)
    cache_path = get_cache_path(stems_dir, cache_key)
    
    # Try load from cache
    use_tqdm = not args.no_progress
    cached_data = None if args.no_cache else load_chroma_cache(cache_path)
    
    if cached_data is not None:
        C_sync, tempo, beat_times = cached_data
    else:
        # Compute chroma features
        y_h, sr = mix_harmonic_cached(files, sr=args.sr, weights=all_weights, use_tqdm=use_tqdm)
        C_sync, tempo, beat_times = chroma_sync_cached(y_h, sr, bins_per_octave=args.bins_per_octave, force_key=args.force_key, use_tqdm=use_tqdm)
        
        # Save to cache
        if not args.no_cache:
            save_chroma_cache(cache_path, C_sync, tempo, beat_times)

    # sections.json から QL 換算 & ラベラ
    beat_to_ql, label_at_sec = load_sections_mapper_and_labeler(sections_path, default_ql_per_beat=args.ql_per_beat, beat_times=beat_times)
    def section_for_t(t: int) -> Optional[str]:
        if t < 0 or t >= len(beat_times): return None
        return label_at_sec(float(beat_times[t]))

    # HMM 遷移
    include_N = bool(params["include_N"])
    A = build_transition(
        S = 25 if include_N else 24,
        stay = float(params["hmm"]["stay"]),
        near = float(params["hmm"]["near"]),
        include_N = include_N,
        n_stay = float(params["N_state"]["stay"]),
        n_out  = float(params["N_state"]["out"])
    )

    # log-likelihood
    if use_tqdm and HAS_TQDM:
        print("[Processing] Building log-likelihood...")
    
    loglik = build_loglik(
        C_sync = C_sync,
        gamma_global = float(params["gamma_global"]),
        local_cfg = params["local_key"],
        include_N = include_N,
        n_cfg = params["N_state"],
        section_for_t = section_for_t
    )

    if use_tqdm and HAS_TQDM:
        print("[Processing] Running Viterbi...")
    
    path = viterbi(loglik, A)
    events = path_to_events(path, beat_to_ql)
    save_chordmap(events, out_path)
    print(f"[OK] chordmap events={len(events)} -> {out_path}")

if __name__ == "__main__":
    main()
