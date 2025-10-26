#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ops/stem_harmony_extended.py  (Extended chords: sus4/sus2/add9/6th)

対応コード（72状態 + N）:
- maj (12): C, C#, ..., B
- min (12): Cm, C#m, ..., Bm
- sus4 (12): Csus4, C#sus4, ..., Bsus4
- sus2 (12): Csus2, C#sus2, ..., Bsus2
- add9 (12): Cadd9, C#add9, ..., Badd9
- 6th (12): C6, C#6, ..., B6
- N (1, optional): 無和音

合計: 72 or 73状態

テンプレート:
- maj: [1,0,0,0,1,0,0,1,0,0,0,0] (root, maj3, 5th)
- min: [1,0,0,1,0,0,0,1,0,0,0,0] (root, min3, 5th)
- sus4: [1,0,0,0,0,1,0,1,0,0,0,0] (root, 4th, 5th)
- sus2: [1,0,1,0,0,0,0,1,0,0,0,0] (root, maj2, 5th)
- add9: [1,0,1,0,1,0,0,1,0,0,0,0] (root, maj2, maj3, 5th)
- 6th: [1,0,0,0,1,0,0,1,0,1,0,0] (root, maj3, 5th, maj6)

v4.1: キャッシュ移植、confidence付与、最短持続、統一化対応
"""
from __future__ import annotations
import argparse, json, sys, hashlib
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import numpy as np
from scipy import ndimage  # type: ignore

# v4.1: キャッシュユーティリティ
import sys
sys.path.insert(0, str(Path(__file__).parent))
from cache_utils import (
    hash_params, ensure_cache_dir, compute_and_cache, digest_files, save_npz
)

# v4.1: スキーマ統一コンバータ
try:
    from ops.chordmap_unify import unify_chordmap_dict
    _HAS_UNIFY = True
except ImportError:
    _HAS_UNIFY = False

# Import from stem_harmony.py
import sys
sys.path.insert(0, str(Path(__file__).parent))
from stem_harmony import (
    NOTE_NAMES, rotate12, cos_sim_columns, list_audio_files, parse_stem_weights,
    load_sections_mapper_and_labeler, key_profile_major, key_profile_minor,
    mix_harmonic, chroma_sync, load_config
)

# ---------------- Viterbi (for extended states) ----------------
def viterbi(loglik: np.ndarray, A: np.ndarray) -> np.ndarray:
    """Viterbi algorithm for extended chord states"""
    T, S = loglik.shape
    logA = np.log(A + 1e-12)
    dp = np.full((S, T), -np.inf)
    bp = np.zeros((S, T), dtype=int)
    
    dp[:, 0] = loglik[0, :]
    for t in range(1, T):
        for s in range(S):
            M = dp[:, t-1] + logA[:, s]
            bp[s, t] = int(np.argmax(M))
            dp[s, t] = M[bp[s, t]] + loglik[t, s]
    
    path = np.zeros(T, dtype=int)
    path[T-1] = int(np.argmax(dp[:, T-1]))
    for t in range(T-2, -1, -1):
        path[t] = bp[path[t+1], t+1]
    
    return path

# ---------------- Extended chord templates ----------------
def maj_template() -> np.ndarray:
    t = np.zeros(12, dtype=float)
    t[[0, 4, 7]] = 1.0
    return t

def min_template() -> np.ndarray:
    t = np.zeros(12, dtype=float)
    t[[0, 3, 7]] = 1.0
    return t

def sus4_template() -> np.ndarray:
    t = np.zeros(12, dtype=float)
    t[[0, 5, 7]] = 1.0
    return t

def sus2_template() -> np.ndarray:
    t = np.zeros(12, dtype=float)
    t[[0, 2, 7]] = 1.0
    return t

def add9_template() -> np.ndarray:
    t = np.zeros(12, dtype=float)
    t[[0, 2, 4, 7]] = 1.0
    return t

def sixth_template() -> np.ndarray:
    t = np.zeros(12, dtype=float)
    t[[0, 4, 7, 9]] = 1.0
    return t

# ---------------- HMM for extended chords ----------------
def build_transition_extended(S: int, stay: float, near: float, include_N: bool, n_stay: float = 0.96, n_out: float = 0.02) -> np.ndarray:
    """
    72 chord states (6 types × 12 roots) + optional N
    State indices:
      0-11: C, C#, ..., B (major)
      12-23: Cm, C#m, ..., Bm (minor)
      24-35: Csus4, C#sus4, ..., Bsus4
      36-47: Csus2, C#sus2, ..., Bsus2
      48-59: Cadd9, C#add9, ..., Badd9
      60-71: C6, C#6, ..., B6
      72: N (if include_N)
    """
    if include_N:
        assert S == 73
    else:
        assert S == 72
    
    A = np.zeros((S, S), dtype=float)
    K = 72
    base = (1.0 - stay - 2*near) / max(1, K - 3)
    
    # For each chord type
    for type_idx in range(6):
        offset = type_idx * 12
        for root in range(12):
            i = offset + root
            A[i, :] = base
            A[i, i] = stay
            A[i, offset + (root+7)%12] += near  # 5th up
            A[i, offset + (root+5)%12] += near  # 4th up
    
    # Add N state
    if include_N:
        N = 72
        A[:K, N] += 1e-3
        A[N, :] = (1.0 - n_stay - n_out*K) / max(1, S-1)
        A[N, N] = n_stay
        A[N, :K] += n_out
    
    A = np.maximum(A, 1e-12)
    A = A / A.sum(axis=1, keepdims=True)
    return A

# ---------------- Log-likelihood (extended) ----------------
def build_loglik_extended(C_sync: np.ndarray, gamma_global: float, gamma_local: float, include_N: bool, n_energy_gamma: float, n_conf_gamma: float) -> np.ndarray:
    """Build log-likelihood for extended chords (72 or 73 states)"""
    T = C_sync.shape[1]
    S = 73 if include_N else 72
    
    # Build 72 templates
    templates = []
    for template_fn in [maj_template, min_template, sus4_template, sus2_template, add9_template, sixth_template]:
        base_tmpl = template_fn()
        for root in range(12):
            templates.append(rotate12(base_tmpl, root))
    
    templates_mat = np.column_stack(templates)  # [12, 72]
    
    # Global prior (simplified: equal weights)
    maj_prof = key_profile_major()
    min_prof = key_profile_minor()
    global_prior = np.zeros(72)
    
    for type_idx in range(6):
        for root in range(12):
            idx = type_idx * 12 + root
            if type_idx in [0, 4, 5]:  # maj, add9, 6th -> major-ish
                global_prior[idx] = rotate12(maj_prof, root)[0]
            elif type_idx == 1:  # min
                global_prior[idx] = rotate12(min_prof, root)[0]
            else:  # sus4, sus2 -> neutral
                global_prior[idx] = (rotate12(maj_prof, root)[0] + rotate12(min_prof, root)[0]) / 2
    
    global_prior = global_prior / (global_prior.sum() + 1e-9)
    
    # Cosine similarity
    sim = cos_sim_columns(C_sync, templates_mat)  # [T, 72]
    
    # Apply global prior
    loglik = np.log(sim + 1e-12) + gamma_global * np.log(global_prior[None, :] + 1e-12)
    
    # Local key prior (simple Gaussian smoothing)
    if gamma_local > 0:
        sim_smooth = ndimage.gaussian_filter1d(sim, sigma=2.0, axis=0, mode='nearest')
        local_prior = sim_smooth / (sim_smooth.sum(axis=1, keepdims=True) + 1e-9)
        loglik += gamma_local * np.log(local_prior + 1e-12)
    
    # Add N state
    if include_N:
        energy = np.sum(C_sync**2, axis=0)
        energy_norm = energy / (energy.max() + 1e-9)
        confidence = sim.max(axis=1)
        
        n_loglik = -n_energy_gamma * energy_norm - n_conf_gamma * confidence
        loglik = np.column_stack([loglik, n_loglik])
    
    return loglik

# ---------------- Output (unified format) ----------------
def state_to_chord_extended(state: int, include_N: bool) -> Tuple[str, str]:
    """Convert state to (root, quality)"""
    if include_N and state == 72:
        return ("N", "")
    
    type_idx = state // 12
    root_idx = state % 12
    root = NOTE_NAMES[root_idx]
    
    type_names = ['maj', 'min', 'sus4', 'sus2', 'add9', '6']
    quality = type_names[type_idx]
    
    return (root, quality)

def path_to_events(path: np.ndarray, beat_to_ql, include_N: bool) -> List[dict]:
    events = []
    prev_state = -1
    start_ql = 0.0
    
    for i, state in enumerate(path):
        if state != prev_state:
            if prev_state >= 0:
                root, quality = state_to_chord_extended(prev_state, include_N)
                events.append({
                    "time": start_ql,
                    "root": root,
                    "quality": quality
                })
            start_ql = beat_to_ql(i)
            prev_state = state
    
    if prev_state >= 0:
        root, quality = state_to_chord_extended(prev_state, include_N)
        events.append({
            "time": start_ql,
            "root": root,
            "quality": quality
        })
    
    return events

# ---------------- Main ----------------
def main():
    ap = argparse.ArgumentParser(description="Extended chord recognition (sus4/sus2/add9/6th)")
    ap.add_argument("--stems", required=True, help="Directory containing stem WAVs")
    ap.add_argument("--exclude", action="append", default=[], help="Substring to exclude")
    ap.add_argument("--out", required=True, help="Output chordmap.json path")
    ap.add_argument("--sections", help="sections.json path (optional)")
    ap.add_argument("--force-key", help="Force key")
    ap.add_argument("--sr", type=int, default=22050, help="Resample rate")
    ap.add_argument("--bins-per-octave", type=int, default=36, help="CQT bins")
    ap.add_argument("--stay", type=float, default=0.93, help="HMM stay")
    ap.add_argument("--near", type=float, default=0.03, help="HMM near")
    ap.add_argument("--include-N", action="store_true", help="Enable N state")
    ap.add_argument("--n-stay", type=float, default=0.96, help="N stay")
    ap.add_argument("--n-out", type=float, default=0.02, help="N->chord")
    ap.add_argument("--gamma-global", type=float, default=0.15, help="Global prior")
    ap.add_argument("--gamma-local", type=float, default=0.20, help="Local prior")
    ap.add_argument("--n-energy-gamma", type=float, default=0.5, help="N energy")
    ap.add_argument("--n-conf-gamma", type=float, default=1.0, help="N conf")
    ap.add_argument("--ql-per-beat", type=float, default=1.0, help="QL per beat")
    ap.add_argument("--stem-weight", action="append", default=[], help="Stem weight")
    
    # v4.1: キャッシュオプション
    ap.add_argument("--cache-dir", type=str, default=None, help="Cache directory (default: <stems>/.cache)")
    ap.add_argument("--no-cache", action="store_true", help="Disable cache (force recompute)")
    
    # v4.1: 追加パラメータ
    ap.add_argument("--emit-confidence", action="store_true", help="emit per-event confidence [0..1]")
    ap.add_argument("--min-dwell-ql", type=float, default=4.0, help="global minimum chord dwell in QL (postprocess, default: 4.0=1bar)")
    ap.add_argument("--mark-modulation", action="store_true", help="mark key modulation events")
    ap.add_argument("--snap-to-bar", type=float, default=0.0, help="snap chord changes to bar boundaries (4.0=1bar, 0=off)")
    args = ap.parse_args()
    
    stems_dir = Path(args.stems)
    out_path = Path(args.out)
    sections_path = Path(args.sections) if args.sections else None
    
    files = list_audio_files(stems_dir, args.exclude)
    if not files:
        print(f"[ERROR] No WAV files found", file=sys.stderr)
        sys.exit(2)
    
    # v4.1: キャッシュ設定
    cache_dir_path = Path(args.cache_dir) if args.cache_dir else (stems_dir / ".cache")
    use_cache = not args.no_cache
    if use_cache:
        ensure_cache_dir(cache_dir_path)
    
    # v4.1: キャッシュキー生成
    weights = parse_stem_weights(args.stem_weight)
    cache_key = hash_params(
        files=[str(f) for f in files],
        file_digests=digest_files(files) if use_cache else "",
        exclude=sorted(args.exclude),
        sr=args.sr,
        bins_per_octave=args.bins_per_octave,
        force_key=args.force_key or "",
        weights=weights,
    )
    
    # v4.1: キャッシュされたchroma取得 or 計算
    def _compute_chroma():
        y_h, sr_mix = mix_harmonic(files, sr=args.sr, weights=weights)
        C_sync, tempo, beat_times = chroma_sync(y_h, sr_mix, bins_per_octave=args.bins_per_octave, force_key=args.force_key)
        return (C_sync, np.array([tempo]), beat_times, np.array([sr_mix]))
    
    cache_path = cache_dir_path / f"chroma_ext_{cache_key}.npz"
    if use_cache and cache_path.exists():
        C_sync, tempo_arr, beat_times, sr_arr = compute_and_cache(
            _compute_chroma,
            cache_path,
            use_cache=True,
            keys=("C_sync", "tempo", "beat_times", "sr")
        )
        tempo = float(tempo_arr[0])
        sr = int(sr_arr[0])
        print(f"[CACHE] HIT: chroma_ext_{cache_key[:8]}.npz")
    else:
        C_sync, tempo_arr, beat_times, sr_arr = _compute_chroma()
        tempo = float(tempo_arr[0])
        sr = int(sr_arr[0])
        if use_cache:
            save_npz(cache_path, C_sync=C_sync, tempo=tempo_arr, beat_times=beat_times, sr=sr_arr)
            print(f"[CACHE] SAVE: chroma_ext_{cache_key[:8]}.npz")
    
    # QL mapper
    beat_to_ql, _ = load_sections_mapper_and_labeler(sections_path, default_ql_per_beat=args.ql_per_beat, beat_times=beat_times)
    
    # HMM
    S = 73 if args.include_N else 72
    A = build_transition_extended(
        S=S,
        stay=args.stay,
        near=args.near,
        include_N=args.include_N,
        n_stay=args.n_stay,
        n_out=args.n_out
    )
    
    # Log-likelihood
    loglik = build_loglik_extended(
        C_sync=C_sync,
        gamma_global=args.gamma_global,
        gamma_local=args.gamma_local,
        include_N=args.include_N,
        n_energy_gamma=args.n_energy_gamma,
        n_conf_gamma=args.n_conf_gamma
    )
    
    # Viterbi
    path = viterbi(loglik, A)
    events = path_to_events(path, beat_to_ql, args.include_N)
    
    # v4.1: 小節スナップ
    if args.snap_to_bar > 0:
        events = _snap_to_bar(events, args.snap_to_bar)
        print(f"[INFO] Snapped to bar boundaries ({args.snap_to_bar}QL)")
    
    # v4.1: confidence付与
    if args.emit_confidence:
        # 各イベントの最大類似度をconfidenceとして付与
        for i, ev in enumerate(events):
            if i < len(loglik):
                state_idx = path[i]
                if state_idx < len(loglik[i]):
                    conf = float(np.exp(loglik[i, state_idx]))
                    ev["confidence"] = min(1.0, max(0.0, conf))
    
    # v4.1: 最短持続フィルタ
    if args.min_dwell_ql > 0:
        events = _apply_min_dwell(events, args.min_dwell_ql)
    
    # v4.1: スキーマ統一化
    output = {"unit": "ql", "events": events}
    if _HAS_UNIFY:
        output = unify_chordmap_dict(output)
        print(f"[INFO] Unified chordmap schema (events: {len(output.get('events', []))})")
    
    # 保存
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print(f"[OK] Extended chords events={len(events)} -> {out_path}")


def _apply_min_dwell(events: List[dict], min_ql: float) -> List[dict]:
    """v4.1: 最短持続フィルタ（短すぎるコードを前のコードに統合）"""
    if not events or min_ql <= 0:
        return events
    
    filtered = [events[0]]
    for ev in events[1:]:
        prev = filtered[-1]
        dur = ev["time"] - prev["time"]
        if dur < min_ql:
            # 短すぎる場合は前のコードを延長
            continue
        filtered.append(ev)
    
    return filtered


def _snap_to_bar(events: List[dict], bar_ql: float) -> List[dict]:
    """v4.1: 小節境界スナップ（コード変化を小節頭に吸着）
    
    Args:
        events: イベントリスト
        bar_ql: 小節のQL長（4.0 = 1小節）
    
    Returns:
        スナップされたイベントリスト
    """
    if not events or bar_ql <= 0:
        return events
    
    snapped = []
    for ev in events:
        # 小節頭に丸める
        original_time = ev["time"]
        snapped_time = round(original_time / bar_ql) * bar_ql
        
        # 同じ時刻にすでにイベントがある場合は後のものを優先
        if snapped and abs(snapped[-1]["time"] - snapped_time) < 0.1:
            # 既存イベントを上書き
            snapped[-1] = {**ev, "time": snapped_time}
        else:
            snapped.append({**ev, "time": snapped_time})
    
    return snapped


if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()
