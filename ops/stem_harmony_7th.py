#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ops/stem_harmony_7th.py  (7th chords support + caching)

7thコード対応版：
- maj7 (12状態): C, C#, D, ..., B
- min7 (12状態): Cm7, C#m7, Dm7, ..., Bm7
- dom7 (12状態): C7, C#7, D7, ..., B7
- min7b5 (12状態): Cm7b5, C#m7b5, ..., Bm7b5
- Optional N state (1状態)
合計: 48 or 49状態

テンプレート:
- maj7: [1,0,0,0,1,0,0,1,0,0,0,1] (root, maj3, 5th, maj7)
- min7: [1,0,0,1,0,0,0,1,0,0,1,0] (root, min3, 5th, min7)
- dom7: [1,0,0,0,1,0,0,1,0,0,1,0] (root, maj3, 5th, min7)
- min7b5: [1,0,0,1,0,0,1,0,0,0,1,0] (root, min3, dim5, min7)
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Import cache utilities
from ops.cache_utils import (
    hash_params, ensure_cache_dir, compute_and_cache, digest_files
)

import argparse, json, math
from typing import List, Tuple, Dict, Optional

import numpy as np
import librosa

try:
    import yaml  # type: ignore
except Exception:
    yaml = None

NOTE_NAMES = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
CHORD_TYPES = ['maj7', 'min7', 'dom7', 'min7b5']  # 4 types

# ---------------- Templates for 7th chords ----------------
def maj7_template() -> np.ndarray:
    """Major 7th: root(0), maj3(4), 5th(7), maj7(11)"""
    t = np.zeros(12, dtype=float)
    t[[0, 4, 7, 11]] = 1.0
    return t

def min7_template() -> np.ndarray:
    """Minor 7th: root(0), min3(3), 5th(7), min7(10)"""
    t = np.zeros(12, dtype=float)
    t[[0, 3, 7, 10]] = 1.0
    return t

def dom7_template() -> np.ndarray:
    """Dominant 7th: root(0), maj3(4), 5th(7), min7(10)"""
    t = np.zeros(12, dtype=float)
    t[[0, 4, 7, 10]] = 1.0
    return t

def min7b5_template() -> np.ndarray:
    """Half-diminished 7th: root(0), min3(3), dim5(6), min7(10)"""
    t = np.zeros(12, dtype=float)
    t[[0, 3, 6, 10]] = 1.0
    return t

def rotate12(v: np.ndarray, k: int) -> np.ndarray:
    return np.roll(v, int(k) % 12)

# ---------------- Utilities ----------------
def cos_sim_columns(A: np.ndarray, B: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """A: [12,T], B: [12,S] -> [T,S]"""
    A2 = A / (np.linalg.norm(A, axis=0, keepdims=True) + eps)
    B2 = B / (np.linalg.norm(B, axis=0, keepdims=True) + eps)
    return (A2.T @ B2)

def list_audio_files(stems_dir: Path, excludes: List[str]) -> List[Path]:
    files = []
    for p in sorted(stems_dir.glob("*.wav")):
        name = p.name.lower()
        if any(ex.lower() in name for ex in excludes):
            continue
        files.append(p)
    return files

def parse_stem_weights(entries: List[str]) -> List[Tuple[str,float]]:
    weights = []
    for e in entries or []:
        if "=" in e:
            k,v = e.split("=",1)
            try:
                weights.append((k.strip().lower(), float(v)))
            except Exception:
                pass
    return weights

# ---------------- Sections (simplified) ----------------
def _safe_load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}

def load_sections_mapper_and_labeler(sections_path: Optional[Path], default_ql_per_beat: float = 1.0, beat_times: Optional[np.ndarray]=None):
    """Simple QL mapper + labeler (same as stem_harmony.py)"""
    if not sections_path or not sections_path.exists():
        def beat_to_ql(beat_idx: int) -> float:
            return beat_idx * default_ql_per_beat
        def label_at_time_sec(t: float) -> Optional[str]:
            return None
        return beat_to_ql, label_at_time_sec
    
    data = _safe_load_json(sections_path)
    
    # List format (existing)
    if isinstance(data, list):
        sections_list = data
        bar_to_ql = {}
        for sec in sections_list:
            bar = sec.get("bar", 0)
            ql_per_bar = sec.get("ql_per_bar", 4.0)
            bar_to_ql[bar] = ql_per_bar
        
        def beat_to_ql(beat_idx: int) -> float:
            bar = int(beat_idx // 4)
            ql_per_bar = bar_to_ql.get(bar, 4.0)
            ql_per_beat = ql_per_bar / 4.0
            return beat_idx * ql_per_beat
        
        # Build time markers
        markers = []
        if beat_times is not None:
            for sec in sections_list:
                bar = sec.get("bar", 0)
                beat_idx = bar * 4
                if beat_idx < len(beat_times):
                    markers.append((float(beat_times[beat_idx]), sec.get("name", "")))
        
        markers.sort(key=lambda x: x[0])
        
        def label_at_time_sec(t: float) -> Optional[str]:
            lab = None
            for (ts, l) in markers:
                if ts <= t: lab = l
                else: break
            return lab
        
        return beat_to_ql, label_at_time_sec
    
    # Dict format (new)
    else:
        time_sigs = data.get("time_sigs", [])
        bar_to_ql = {}
        for item in time_sigs:
            bar = item.get("bar", 0)
            ts = item.get("time_sig", "4/4")
            num_str, denom_str = ts.split("/")
            num = int(num_str)
            bar_to_ql[bar] = float(num)
        
        def beat_to_ql(beat_idx: int) -> float:
            bar = int(beat_idx // 4)
            ql_per_bar = bar_to_ql.get(bar, 4.0)
            ql_per_beat = ql_per_bar / 4.0
            return beat_idx * ql_per_beat
        
        # Build markers from sections
        markers = []
        sections = data.get("sections", [])
        if beat_times is not None:
            for sec in sections:
                bar = sec.get("bar", 0)
                beat_idx = bar * 4
                if beat_idx < len(beat_times):
                    markers.append((float(beat_times[beat_idx]), sec.get("name", "")))
        
        markers.sort(key=lambda x: x[0])
        
        def label_at_time_sec(t: float) -> Optional[str]:
            lab = None
            for (ts, l) in markers:
                if ts <= t: lab = l
                else: break
            return lab
        
        return beat_to_ql, label_at_time_sec

# ---------------- Signal processing ----------------
def mix_harmonic(files: List[Path], sr: int, weights: List[Tuple[str,float]]) -> Tuple[np.ndarray, int]:
    y_sum = None
    for fp in files:
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

def chroma_sync(y_h: np.ndarray, sr: int, bins_per_octave: int = 36, force_key: Optional[str] = None):
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
    
    C = librosa.feature.chroma_cqt(y=y_h, sr=sr, bins_per_octave=bins_per_octave, tuning=tuning)
    C_sync = librosa.util.sync(C, beats, aggregate=np.median)
    return C_sync, tempo, beat_times

# ---------------- Key profile (simplified) ----------------
def key_profile_major() -> np.ndarray:
    return np.array([6.35,2.23,3.48,2.33,4.38,4.09,2.52,5.19,2.39,3.66,2.29,2.88], dtype=float)

def key_profile_minor() -> np.ndarray:
    return np.array([6.33,2.68,3.52,5.38,2.60,3.53,2.54,4.75,3.98,2.69,3.34,3.17], dtype=float)

# ---------------- HMM for 7th chords (48 or 49 states) ----------------
def build_transition_7th(S: int, stay: float, near: float, include_N: bool, n_stay: float = 0.96, n_out: float = 0.02) -> np.ndarray:
    """
    48 chord states (4 types × 12 roots) + optional N
    State indices:
      0-11: Cmaj7, C#maj7, ..., Bmaj7
      12-23: Cm7, C#m7, ..., Bm7
      24-35: C7, C#7, ..., B7
      36-47: Cm7b5, C#m7b5, ..., Bm7b5
      48: N (if include_N)
    """
    if include_N:
        assert S == 49
    else:
        assert S == 48
    
    A = np.zeros((S, S), dtype=float)
    K = 48  # chord states
    
    # Base probability for distant chords
    base = (1.0 - stay - 2*near) / max(1, K - 3)
    
    # For each chord type (maj7, min7, dom7, min7b5)
    for type_idx in range(4):
        offset = type_idx * 12
        for root in range(12):
            i = offset + root
            A[i, :] = base  # all distant chords
            A[i, i] = stay  # stay on same chord
            
            # Near transitions (4th/5th) within same type
            A[i, offset + (root+7)%12] += near  # 5th up
            A[i, offset + (root+5)%12] += near  # 4th up
    
    # Add N state if enabled
    if include_N:
        N = 48
        A[:K, N] += 1e-3  # small probability to N
        A[N, :] = (1.0 - n_stay - n_out*K) / max(1, S-1)
        A[N, N] = n_stay
        A[N, :K] += n_out
    
    # Normalize
    A = np.maximum(A, 1e-12)
    A = A / A.sum(axis=1, keepdims=True)
    return A

def build_loglik_7th(C_sync: np.ndarray, gamma_global: float, include_N: bool, n_energy_gamma: float = 1.0, n_conf_gamma: float = 2.0) -> np.ndarray:
    """
    Build log-likelihood for 7th chords (48 or 49 states)
    Simplified version without local key prior and section-specific params
    """
    T = C_sync.shape[1]
    S = 49 if include_N else 48
    
    # Build templates for all 48 chords
    templates = []
    for type_idx, template_fn in enumerate([maj7_template, min7_template, dom7_template, min7b5_template]):
        base_tmpl = template_fn()
        for root in range(12):
            templates.append(rotate12(base_tmpl, root))
    
    templates_mat = np.column_stack(templates)  # [12, 48]
    
    # Global key prior (simplified: equal weights for all chords)
    maj_prof = key_profile_major()
    min_prof = key_profile_minor()
    
    # Simple global prior: average of major/minor profiles for each root
    global_prior = np.zeros(48)
    for type_idx in range(4):
        for root in range(12):
            idx = type_idx * 12 + root
            if type_idx in [0, 2]:  # maj7, dom7 -> major-ish
                global_prior[idx] = rotate12(maj_prof, root)[0]
            else:  # min7, min7b5 -> minor-ish
                global_prior[idx] = rotate12(min_prof, root)[0]
    
    global_prior = global_prior / (global_prior.sum() + 1e-9)
    
    # Cosine similarity
    sim = cos_sim_columns(C_sync, templates_mat)  # [T, 48]
    
    # Apply global prior
    loglik = np.log(sim + 1e-12) + gamma_global * np.log(global_prior[None, :] + 1e-12)
    
    # Add N state if enabled
    if include_N:
        energy = np.sum(C_sync**2, axis=0)  # [T]
        energy_norm = energy / (energy.max() + 1e-9)
        confidence = sim.max(axis=1)  # [T]
        
        n_loglik = -n_energy_gamma * energy_norm - n_conf_gamma * confidence
        loglik = np.column_stack([loglik, n_loglik])  # [T, 49]
    
    return loglik  # [T, S]

# ---------------- Viterbi ----------------
def viterbi(loglik: np.ndarray, A: np.ndarray) -> np.ndarray:
    T, S = loglik.shape
    logA = np.log(A + 1e-12)
    delta = np.full((T, S), -np.inf)
    psi = np.zeros((T, S), dtype=int)
    
    delta[0, :] = loglik[0, :]
    
    for t in range(1, T):
        for j in range(S):
            vals = delta[t-1, :] + logA[:, j]
            psi[t, j] = int(np.argmax(vals))
            delta[t, j] = vals[psi[t, j]] + loglik[t, j]
    
    path = np.zeros(T, dtype=int)
    path[-1] = int(np.argmax(delta[-1, :]))
    for t in range(T-2, -1, -1):
        path[t] = psi[t+1, path[t+1]]
    
    return path

# ---------------- Output formatting ----------------
def state_to_chord_7th(state: int, include_N: bool) -> Tuple[str, str]:
    """Convert state index to (root, quality) tuple for unified format"""
    if include_N and state == 48:
        return ("N", "")
    
    type_idx = state // 12
    root_idx = state % 12
    root = NOTE_NAMES[root_idx]
    
    type_names = ['maj7', 'min7', 'dom7', 'min7b5']
    quality = type_names[type_idx]
    
    return (root, quality)

def path_to_events(path: np.ndarray, beat_to_ql, include_N: bool) -> List[dict]:
    """Convert path to events in unified format"""
    events = []
    prev_state = -1
    start_ql = 0.0
    
    for i, state in enumerate(path):
        if state != prev_state:
            if prev_state >= 0:
                root, quality = state_to_chord_7th(prev_state, include_N)
                events.append({
                    "time": start_ql,
                    "root": root,
                    "quality": quality
                })
            start_ql = beat_to_ql(i)
            prev_state = state
    
    if prev_state >= 0:
        root, quality = state_to_chord_7th(prev_state, include_N)
        events.append({
            "time": start_ql,
            "root": root,
            "quality": quality
        })
    
    return events

def save_chordmap(events: List[dict], out_path: Path):
    """Save in unified format"""
    output = {
        "unit": "ql",
        "events": events
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

# ---------------- Main ----------------
def main():
    ap = argparse.ArgumentParser(description="7th chords chord recognition (maj7/min7/dom7/min7b5)")
    ap.add_argument("--stems", required=True, help="Directory containing stem WAVs")
    ap.add_argument("--exclude", action="append", default=[], help="Substring to exclude (e.g., 'Vocals')")
    ap.add_argument("--out", required=True, help="Output chordmap.json path")
    ap.add_argument("--sections", help="sections.json path for QL mapping (optional)")
    ap.add_argument("--force-key", help="Force key (e.g., 'C', 'Am') - disables tuning correction")
    ap.add_argument("--sr", type=int, default=22050, help="Resample rate")
    ap.add_argument("--bins-per-octave", type=int, default=36, help="CQT bins per octave")
    ap.add_argument("--stay", type=float, default=0.93, help="HMM stay probability")
    ap.add_argument("--near", type=float, default=0.03, help="HMM 4th/5th probability")
    ap.add_argument("--include-N", action="store_true", help="Enable No-Chord state")
    ap.add_argument("--n-stay", type=float, default=0.96, help="HMM stay for N state")
    ap.add_argument("--n-out", type=float, default=0.02, help="HMM N->chord probability per chord")
    ap.add_argument("--gamma-global", type=float, default=0.15, help="Global key prior gamma")
    ap.add_argument("--n-energy-gamma", type=float, default=0.5, help="N-state energy penalty")
    ap.add_argument("--n-conf-gamma", type=float, default=1.0, help="N-state confidence penalty")
    ap.add_argument("--ql-per-beat", type=float, default=1.0, help="Fallback QL per beat")
    ap.add_argument("--stem-weight", action="append", default=[], help="Per-stem weight like 'bass=1.3'")
    ap.add_argument("--cache-dir", type=str, default=None, help="Cache directory (default: <stems>/.cache)")
    ap.add_argument("--no-cache", action="store_true", help="Disable cache (force recompute)")
    
    # v4.1: 追加パラメータ
    ap.add_argument("--emit-confidence", action="store_true", help="emit per-event confidence [0..1]")
    ap.add_argument("--min-dwell-ql", type=float, default=0.0, help="global minimum chord dwell in QL (postprocess)")
    
    args = ap.parse_args()
    
    stems_dir = Path(args.stems)
    out_path = Path(args.out)
    sections_path = Path(args.sections) if args.sections else None
    
    files = list_audio_files(stems_dir, args.exclude)
    if not files:
        print(f"[ERROR] No WAV files found in {stems_dir}", file=sys.stderr)
        sys.exit(2)
    
    # Cache setup
    cache_root = ensure_cache_dir(Path(args.cache_dir) if args.cache_dir else (stems_dir / ".cache"))
    files_key = digest_files(files)
    
    # Mix and chroma with caching
    weights = parse_stem_weights(args.stem_weight)
    
    # Cache key for chroma
    chroma_key = hash_params(
        kind="chroma_sync_7th",
        files=files_key,
        sr=args.sr,
        bpo=args.bins_per_octave,
        excludes=sorted(args.exclude),
        force_key=args.force_key or "auto"
    )
    chroma_cache = cache_root / f"chroma_sync_{chroma_key}.npz"
    
    def compute_chroma():
        y_h, sr = mix_harmonic(files, sr=args.sr, weights=weights)
        C_sync, tempo, beat_times = chroma_sync(y_h, sr, bins_per_octave=args.bins_per_octave, force_key=args.force_key)
        return (C_sync, tempo, beat_times)
    
    C_sync, tempo, beat_times = compute_and_cache(
        compute_chroma,
        chroma_cache,
        use_cache=(not args.no_cache),
        keys=("C_sync", "tempo", "beat_times")
    )
    
    if not args.no_cache:
        print(f"[CACHE] Chroma: {'HIT' if chroma_cache.exists() else 'MISS'}")
    
    # QL mapper
    beat_to_ql, _ = load_sections_mapper_and_labeler(sections_path, default_ql_per_beat=args.ql_per_beat, beat_times=beat_times)
    
    # HMM
    S = 49 if args.include_N else 48
    A = build_transition_7th(
        S=S,
        stay=args.stay,
        near=args.near,
        include_N=args.include_N,
        n_stay=args.n_stay,
        n_out=args.n_out
    )
    
    # Log-likelihood
    loglik = build_loglik_7th(
        C_sync=C_sync,
        gamma_global=args.gamma_global,
        include_N=args.include_N,
        n_energy_gamma=args.n_energy_gamma,
        n_conf_gamma=args.n_conf_gamma
    )
    
    # Viterbi
    path = viterbi(loglik, A)
    events = path_to_events(path, beat_to_ql, args.include_N)
    save_chordmap(events, out_path)
    
    print(f"[OK] 7th chords chordmap events={len(events)} -> {out_path}")

if __name__ == "__main__":
    main()
