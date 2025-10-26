#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ops/stem_harmony_7th_v2_cached.py

7th chords recognition with local key prior + CACHING for speed
Combines stem_harmony_cached.py and stem_harmony_7th_v2.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import argparse
import hashlib
import numpy as np
from typing import Optional, Tuple, List

# Import from stem_harmony_cached
from ops.stem_harmony_cached import (
    get_cache_key,
    load_chroma_cache,
    save_chroma_cache
)

# Import from stem_harmony
from ops.stem_harmony import (
    list_audio_files,
    mix_harmonic,
    chroma_sync,
    parse_stem_weights
)

# Import from stem_harmony_7th_v2
from ops.stem_harmony_7th_v2 import (
    rotate12,
    chord_template_maj7,
    chord_template_min7,
    chord_template_dom7,
    chord_template_min7b5,
    key_profile_major,
    key_profile_minor,
    cos_sim_columns,
    estimate_local_key_7th,
    map_key_to_chord_prior_7th,
    build_transition_7th,
    viterbi,
    state_to_chord_7th,
    quantize_ql,
    load_sections_mapper_and_labeler,
    load_config,
    resolve_params_with_config_7th
)

def build_loglik_7th_enhanced(
    C_sync: np.ndarray,
    gamma_global: float,
    local_cfg: dict,
    include_N: bool,
    n_cfg: dict,
    section_for_t = None
) -> np.ndarray:
    """Build log-likelihood with local key prior"""
    T = C_sync.shape[1]
    
    # 48 chord templates
    templates = []
    for root in range(12):
        templates.append(rotate12(chord_template_maj7(), root))
        templates.append(rotate12(chord_template_min7(), root))
        templates.append(rotate12(chord_template_dom7(), root))
        templates.append(rotate12(chord_template_min7b5(), root))
    
    templates_mat = np.column_stack(templates)  # [12, 48]
    
    # Global key prior
    maj_prof = key_profile_major()
    min_prof = key_profile_minor()
    global_prior = np.zeros(48)
    for type_idx in range(4):
        for root in range(12):
            idx = type_idx * 12 + root
            if type_idx in [0, 2]:  # maj7, dom7
                global_prior[idx] = rotate12(maj_prof, root)[0]
            else:  # min7, min7b5
                global_prior[idx] = rotate12(min_prof, root)[0]
    global_prior = global_prior / (global_prior.sum() + 1e-9)
    
    # Cosine similarity
    sim = cos_sim_columns(C_sync, templates_mat)  # [T, 48]
    
    # Apply global prior
    loglik = np.log(sim + 1e-12) + gamma_global * np.log(global_prior[None, :] + 1e-12)
    
    # Local key prior
    if local_cfg.get("enable", True):
        window = local_cfg.get("window", 8)
        agg_fn = local_cfg.get("agg_fn", "gaussian")
        gamma_local = local_cfg.get("gamma", 0.3)
        
        local_keys = estimate_local_key_7th(C_sync, window, agg_fn)
        chord_prior = map_key_to_chord_prior_7th(local_keys)
        loglik += gamma_local * np.log(chord_prior + 1e-12)
    
    # N state
    if include_N:
        energy = np.sum(C_sync**2, axis=0)
        energy_norm = energy / (energy.max() + 1e-9)
        confidence = sim.max(axis=1)
        
        n_energy_gamma = n_cfg.get("energy_gamma", 1.0)
        n_conf_gamma = n_cfg.get("conf_gamma", 2.0)
        
        n_loglik = -n_energy_gamma * energy_norm - n_conf_gamma * confidence
        loglik = np.column_stack([loglik, n_loglik])
    
    return loglik

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stems", type=str, required=True)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--sections", type=str, default=None)
    ap.add_argument("--exclude", action="append", default=[])
    ap.add_argument("--sr", type=int, default=22050)
    ap.add_argument("--bins-per-octave", type=int, default=36)
    ap.add_argument("--force-key", type=str, default=None)
    ap.add_argument("--config", type=str, default=None)
    ap.add_argument("--no-cache", action="store_true", help="Disable chroma cache")
    ap.add_argument("--gamma-global", type=float, default=0.15)
    ap.add_argument("--gamma-local", type=float, default=0.30)
    ap.add_argument("--stem-weight", action="append", default=[])
    args = ap.parse_args()
    
    cfg = load_config(Path(args.config)) if args.config else {}
    params = resolve_params_with_config_7th(args, cfg)
    
    stems_dir = Path(args.stems)
    out_path = Path(args.out)
    sections_path = Path(args.sections) if args.sections else None
    
    files = list_audio_files(stems_dir, args.exclude)
    if not files:
        print(f"[ERROR] No WAV files found", file=sys.stderr)
        sys.exit(2)
    
    # Chroma cache
    cache_path = None
    if not args.no_cache:
        cache_dir = stems_dir / ".cache"
        cache_dir.mkdir(exist_ok=True)
        cache_key = get_cache_key(files, args.sr, args.bins_per_octave, args.exclude, [])
        cache_path = cache_dir / f"chroma_sync_{cache_key}.npz"
    
    # Load or compute chroma
    cached_data = None if args.no_cache else (load_chroma_cache(cache_path) if cache_path and cache_path.exists() else None)
    
    if cached_data is not None:
        C_sync, tempo, beat_times = cached_data
        print(f"[INFO] Loaded chroma from cache: {C_sync.shape}")
    else:
        weights_cli = parse_stem_weights(args.stem_weight)
        weights_cfg = parse_stem_weights(params.get("stem_weight", []))
        y_h, sr = mix_harmonic(files, sr=args.sr, weights=(weights_cfg or []) + (weights_cli or []))
        C_sync, tempo, beat_times = chroma_sync(y_h, sr, bins_per_octave=args.bins_per_octave, force_key=args.force_key)
        
        if cache_path:
            save_chroma_cache(cache_path, C_sync, tempo, beat_times)
            print(f"[INFO] Saved chroma cache: {cache_path}")
    
    # QL mapper
    beat_to_ql, label_at_sec = load_sections_mapper_and_labeler(sections_path, default_ql_per_beat=1.0, beat_times=beat_times)
    def section_for_t(t: int) -> Optional[str]:
        if t < 0 or t >= len(beat_times): return None
        return label_at_sec(float(beat_times[t]))
    
    # HMM
    include_N = bool(params["include_N"])
    S = 49 if include_N else 48
    A = build_transition_7th(
        S=S,
        stay=float(params["hmm"]["stay"]),
        near=float(params["hmm"]["near"]),
        include_N=include_N,
        n_stay=float(params["N_state"]["stay"]),
        n_out=float(params["N_state"]["out"])
    )
    
    # Log-likelihood with local key prior
    loglik = build_loglik_7th_enhanced(
        C_sync=C_sync,
        gamma_global=float(params["gamma_global"]),
        local_cfg=params["local_key"],
        include_N=include_N,
        n_cfg=params["N_state"],
        section_for_t=section_for_t
    )
    
    # Viterbi
    states = viterbi(loglik, A)
    
    # Quantize to QL
    events = []
    prev_chord = None
    for t in range(len(states)):
        s = int(states[t])
        ql = beat_to_ql(t)
        if ql < 0: continue
        chord = state_to_chord_7th(s, include_N)
        
        if chord != prev_chord:
            events.append({"time": float(ql), "chord": chord})
            prev_chord = chord
    
    # Merge consecutive
    if params.get("merge_consecutive", True):
        merged = []
        for e in events:
            if merged and merged[-1]["chord"] == e["chord"]:
                continue
            merged.append(e)
        events = merged
    
    # Quantize
    events = quantize_ql(events, float(params.get("quantize_ql", 0.0)))
    
    # Output
    output = {
        "unit": "ql",
        "events": [{"time": e["time"], "chord": e["chord"]} for e in events]
    }
    
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print(f"[INFO] Generated {len(events)} chord events → {out_path}")
