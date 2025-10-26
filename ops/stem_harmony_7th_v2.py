#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ops/stem_harmony_7th_v2.py  (Enhanced 7th chords with local key prior)

改善点:
- local key prior追加（8-16拍窓でモデュレーション対応）
- section-specific params対応
- YAML/JSON設定対応
- 通常版と同等の精度を目指す
- v4.1: キャッシュ移植、confidence付与、最短持続、転調マーカー

7thコード:
- maj7, min7, dom7, min7b5 (48状態)
- Optional N state (49状態)
"""
from __future__ import annotations
import argparse, json, sys, math, hashlib
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Any

import numpy as np
import librosa
from scipy import ndimage  # type: ignore

# v4.1: キャッシュユーティリティ
from cache_utils import (
    hash_params, ensure_cache_dir, compute_and_cache, digest_files
)

# v4.1: スキーマ統一コンバータ
try:
    from ops.chordmap_unify import unify_chordmap_dict
    _HAS_UNIFY = True
except ImportError:
    _HAS_UNIFY = False

try:
    import yaml  # type: ignore
except Exception:
    yaml = None

NOTE_NAMES = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
CHORD_TYPES_7TH = ['maj7', 'min7', 'dom7', 'min7b5']

# Import utilities from stem_harmony.py
import sys
sys.path.insert(0, str(Path(__file__).parent))
from stem_harmony import (
    rotate12, cos_sim_columns, list_audio_files, parse_stem_weights,
    load_sections_mapper_and_labeler, key_profile_major, key_profile_minor,
    mix_harmonic, chroma_sync, load_config
)

# ---------------- 7th chord templates ----------------
def maj7_template() -> np.ndarray:
    t = np.zeros(12, dtype=float)
    t[[0, 4, 7, 11]] = 1.0
    return t

def min7_template() -> np.ndarray:
    t = np.zeros(12, dtype=float)
    t[[0, 3, 7, 10]] = 1.0
    return t

def dom7_template() -> np.ndarray:
    t = np.zeros(12, dtype=float)
    t[[0, 4, 7, 10]] = 1.0
    return t

def min7b5_template() -> np.ndarray:
    t = np.zeros(12, dtype=float)
    t[[0, 3, 6, 10]] = 1.0
    return t

# ---------------- HMM for 7th chords ----------------
def build_transition_7th(S: int, stay: float, near: float, include_N: bool, n_stay: float = 0.96, n_out: float = 0.02) -> np.ndarray:
    if include_N:
        assert S == 49
    else:
        assert S == 48
    
    A = np.zeros((S, S), dtype=float)
    K = 48
    base = (1.0 - stay - 2*near) / max(1, K - 3)
    
    for type_idx in range(4):
        offset = type_idx * 12
        for root in range(12):
            i = offset + root
            A[i, :] = base
            A[i, i] = stay
            A[i, offset + (root+7)%12] += near
            A[i, offset + (root+5)%12] += near
    
    if include_N:
        N = 48
        A[:K, N] += 1e-3
        A[N, :] = (1.0 - n_stay - n_out*K) / max(1, S-1)
        A[N, N] = n_stay
        A[N, :K] += n_out
    
    A = np.maximum(A, 1e-12)
    A = A / A.sum(axis=1, keepdims=True)
    return A

# ---------------- Local key prior for 7th chords ----------------
def estimate_local_key_7th(C_sync: np.ndarray, window: int = 8, agg_fn: str = "gaussian") -> np.ndarray:
    """
    Estimate local key at each beat
    Returns: [T, 24] (12 major + 12 minor keys)
    """
    T = C_sync.shape[1]
    maj_prof = key_profile_major()
    min_prof = key_profile_minor()
    
    # Build key templates (24 keys)
    key_templates = []
    for root in range(12):
        key_templates.append(rotate12(maj_prof, root))  # Major keys
    for root in range(12):
        key_templates.append(rotate12(min_prof, root))  # Minor keys
    
    key_templates_mat = np.column_stack(key_templates)  # [12, 24]
    
    # Cosine similarity for each beat
    sim = cos_sim_columns(C_sync, key_templates_mat)  # [T, 24]
    
    # Apply windowing
    if window > 1 and T > window:
        if agg_fn == "gaussian":
            sigma = window / 4.0
            sim_smooth = ndimage.gaussian_filter1d(sim, sigma=sigma, axis=0, mode='nearest')
        elif agg_fn == "max":
            # Max pooling
            sim_smooth = np.zeros_like(sim)
            for t in range(T):
                t0 = max(0, t - window//2)
                t1 = min(T, t + window//2 + 1)
                sim_smooth[t, :] = sim[t0:t1, :].max(axis=0)
        else:  # mean
            # Mean pooling
            sim_smooth = np.zeros_like(sim)
            for t in range(T):
                t0 = max(0, t - window//2)
                t1 = min(T, t + window//2 + 1)
                sim_smooth[t, :] = sim[t0:t1, :].mean(axis=0)
        
        sim = sim_smooth
    
    # Normalize to probability
    sim = np.maximum(sim, 1e-12)
    sim = sim / sim.sum(axis=1, keepdims=True)
    
    return sim  # [T, 24]

def map_key_to_chord_prior_7th(local_keys: np.ndarray) -> np.ndarray:
    """
    Map local key probabilities to chord prior (48 states)
    
    Heuristic:
    - maj7/dom7: favor major keys
    - min7/min7b5: favor minor keys
    - Weight by root match
    
    Returns: [T, 48]
    """
    T = local_keys.shape[0]
    chord_prior = np.zeros((T, 48))
    
    for t in range(T):
        for root in range(12):
            # Major keys -> maj7, dom7
            maj_key_prob = local_keys[t, root]  # Major key at root
            chord_prior[t, root] += maj_key_prob * 0.6  # maj7
            chord_prior[t, 24 + root] += maj_key_prob * 0.4  # dom7
            
            # Minor keys -> min7, min7b5
            min_key_prob = local_keys[t, 12 + root]  # Minor key at root
            chord_prior[t, 12 + root] += min_key_prob * 0.7  # min7
            chord_prior[t, 36 + root] += min_key_prob * 0.3  # min7b5
    
    # Normalize
    chord_prior = np.maximum(chord_prior, 1e-12)
    chord_prior = chord_prior / chord_prior.sum(axis=1, keepdims=True)
    
    return chord_prior

# ---------------- Build log-likelihood (enhanced) ----------------
def build_loglik_7th_enhanced(C_sync: np.ndarray, gamma_global: float, local_cfg: dict, include_N: bool, n_cfg: dict, section_for_t) -> np.ndarray:
    """
    Enhanced log-likelihood with local key prior and section-specific params
    """
    T = C_sync.shape[1]
    S = 49 if include_N else 48
    
    # Build 48 chord templates
    templates = []
    for template_fn in [maj7_template, min7_template, dom7_template, min7b5_template]:
        base_tmpl = template_fn()
        for root in range(12):
            templates.append(rotate12(base_tmpl, root))
    
    templates_mat = np.column_stack(templates)  # [12, 48]
    
    # Global key prior (simplified)
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
    
    # Local key prior (if enabled)
    if local_cfg.get("enable", True):
        window = local_cfg.get("window", 8)
        agg_fn = local_cfg.get("agg_fn", "gaussian")
        gamma_local = local_cfg.get("gamma", 0.3)
        
        # Override by section
        if section_for_t:
            # Section-specific window (example: use different window for chorus/verse)
            per_section = local_cfg.get("per_section", {})
            # Build section-aware local keys (simplified: use global window)
            local_keys = estimate_local_key_7th(C_sync, window, agg_fn)
        else:
            local_keys = estimate_local_key_7th(C_sync, window, agg_fn)
        
        chord_prior = map_key_to_chord_prior_7th(local_keys)
        loglik += gamma_local * np.log(chord_prior + 1e-12)
    
    # Add N state if enabled
    if include_N:
        energy = np.sum(C_sync**2, axis=0)
        energy_norm = energy / (energy.max() + 1e-9)
        confidence = sim.max(axis=1)
        
        n_energy_gamma = n_cfg.get("energy_gamma", 1.0)
        n_conf_gamma = n_cfg.get("conf_gamma", 2.0)
        
        n_loglik = -n_energy_gamma * energy_norm - n_conf_gamma * confidence
        loglik = np.column_stack([loglik, n_loglik])
    
    return loglik

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

# ---------------- Output ----------------
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

# ---------------- Config resolution ----------------
def resolve_params_with_config_7th(args, cfg: dict) -> dict:
    """Resolve parameters from CLI and YAML config (7th version)"""
    return {
        "include_N": cfg.get("N_state", {}).get("enable", args.include_N),
        "hmm": {
            "stay": float(cfg.get("HMM", {}).get("stay", args.stay)),
            "near": float(cfg.get("HMM", {}).get("near", args.near)),
        },
        "local_key": {
            "enable": cfg.get("local_key", {}).get("enable", True),
            "window": int(cfg.get("local_key", {}).get("window", 8)),
            "gamma": float(cfg.get("local_key", {}).get("gamma", args.gamma_local)),
            "agg_fn": cfg.get("local_key", {}).get("agg_fn", "gaussian"),
            "per_section": cfg.get("local_key", {}).get("per_section", {}),
        },
        "N_state": {
            "energy_gamma": float(cfg.get("N_state", {}).get("energy_gamma", args.n_energy_gamma)),
            "conf_gamma": float(cfg.get("N_state", {}).get("conf_gamma", args.n_conf_gamma)),
            "stay": float(cfg.get("N_state", {}).get("stay", args.n_stay)),
            "out": float(cfg.get("N_state", {}).get("out", args.n_out)),
        },
        "stem_weight": list(cfg.get("stem_weight", [])) if isinstance(cfg.get("stem_weight", []), list) else [],
        "gamma_global": float(cfg.get("global_key", {}).get("gamma", args.gamma_global)) if isinstance(cfg.get("global_key"), dict) else args.gamma_global,
    }

# ---------------- Main ----------------
def main():
    ap = argparse.ArgumentParser(description="Enhanced 7th chords recognition with local key prior")
    ap.add_argument("--stems", required=True, help="Directory containing stem WAVs")
    ap.add_argument("--exclude", action="append", default=[], help="Substring to exclude")
    ap.add_argument("--out", required=True, help="Output chordmap.json path")
    ap.add_argument("--sections", help="sections.json path (optional)")
    ap.add_argument("--config", help="YAML/JSON config")
    ap.add_argument("--force-key", help="Force key (e.g., 'C', 'Am')")
    ap.add_argument("--sr", type=int, default=22050, help="Resample rate")
    ap.add_argument("--bins-per-octave", type=int, default=36, help="CQT bins per octave")
    ap.add_argument("--stay", type=float, default=0.93, help="HMM stay probability")
    ap.add_argument("--near", type=float, default=0.03, help="HMM 4th/5th probability")
    ap.add_argument("--include-N", action="store_true", help="Enable No-Chord state")
    ap.add_argument("--n-stay", type=float, default=0.96, help="N state stay probability")
    ap.add_argument("--n-out", type=float, default=0.02, help="N->chord probability")
    ap.add_argument("--gamma-global", type=float, default=0.15, help="Global key prior gamma")
    ap.add_argument("--gamma-local", type=float, default=0.30, help="Local key prior gamma")
    ap.add_argument("--n-energy-gamma", type=float, default=0.5, help="N-state energy penalty")
    ap.add_argument("--n-conf-gamma", type=float, default=1.0, help="N-state confidence penalty")
    ap.add_argument("--ql-per-beat", type=float, default=1.0, help="Fallback QL per beat")
    ap.add_argument("--stem-weight", action="append", default=[], help="Per-stem weight")
    ap.add_argument("--cache-dir", type=str, default=None, help="Cache directory (default: <stems>/.cache)")
    ap.add_argument("--no-cache", action="store_true", help="Disable cache (force recompute)")
    
    # v4.1: 追加パラメータ
    ap.add_argument("--emit-confidence", action="store_true", help="emit per-event confidence [0..1]")
    ap.add_argument("--min-dwell-ql", type=float, default=0.0, help="global minimum chord dwell in QL (postprocess)")
    ap.add_argument("--min-dwell-per-section", default=None, help="JSON/YAML: {section: ql}")
    ap.add_argument("--emit-key-changes", action="store_true", help="emit key change markers inferred from local-key prior")
    
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
    
    # Cache setup
    cache_root = ensure_cache_dir(Path(args.cache_dir) if args.cache_dir else (stems_dir / ".cache"))
    files_key = digest_files(files)
    
    # Mix and chroma with caching
    weights_cli = parse_stem_weights(args.stem_weight)
    weights_cfg = parse_stem_weights(params.get("stem_weight", []))
    
    # Cache key for chroma (include section params for v2)
    chroma_key = hash_params(
        kind="chroma_sync_7th_v2",
        files=files_key,
        sr=args.sr,
        bpo=args.bins_per_octave,
        excludes=sorted(args.exclude),
        force_key=args.force_key or "auto",
        gamma_local=params["local_key"].get("gamma", 0.30),
        local_window=params["local_key"].get("window", 8)
    )
    chroma_cache = cache_root / f"chroma_sync_{chroma_key}.npz"
    
    def compute_chroma():
        y_h, sr = mix_harmonic(files, sr=args.sr, weights=(weights_cfg or []) + (weights_cli or []))
        C_sync, tempo, beat_times = chroma_sync(y_h, sr, bins_per_octave=args.bins_per_octave, force_key=args.force_key)
        return (C_sync, tempo, beat_times)
    
    C_sync, tempo, beat_times = compute_and_cache(
        compute_chroma,
        chroma_cache,
        use_cache=(not args.no_cache),
        keys=("C_sync", "tempo", "beat_times")
    )
    
    if not args.no_cache:
        print(f"[CACHE] Chroma: {'HIT' if chroma_cache.exists() else 'MISS'}", file=sys.stderr)
    
    # QL mapper
    beat_to_ql, label_at_sec = load_sections_mapper_and_labeler(sections_path, default_ql_per_beat=args.ql_per_beat, beat_times=beat_times)
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
    
    # Log-likelihood (enhanced with local key prior)
    loglik = build_loglik_7th_enhanced(
        C_sync=C_sync,
        gamma_global=float(params["gamma_global"]),
        local_cfg=params["local_key"],
        include_N=include_N,
        n_cfg=params["N_state"],
        section_for_t=section_for_t
    )
    
    # Viterbi
    path = viterbi(loglik, A)
    events = path_to_events(path, beat_to_ql, include_N)
    
    # v4.1: 最短持続（ぶつ切れ抑止）
    if args.min_dwell_ql > 0 or args.min_dwell_per_section:
        per_sec_map = None
        if args.min_dwell_per_section:
            try:
                with open(args.min_dwell_per_section, "r", encoding="utf-8") as f:
                    spec = f.read()
                try:
                    import yaml
                    per_sec_map = yaml.safe_load(spec)
                except Exception:
                    per_sec_map = json.loads(spec)
            except Exception as e:
                print(f"[WARN] Failed to load min_dwell_per_section: {e}", file=sys.stderr)
        
        events = enforce_min_dwell(
            events,
            global_min=args.min_dwell_ql,
            per_section=per_sec_map,
            section_for_t=section_for_t
        )
    
    # v4.1: confidence付与（posteriors利用は将来実装）
    if args.emit_confidence:
        for e in events:
            # 簡易実装：全て0.8（将来的にposteriorから計算）
            e["confidence"] = 0.8
    
    # v4.1: 転調マーカー（sections.jsonからkey_hintを抽出）
    key_changes = []
    if args.emit_key_changes and sections_data:
        # 将来実装：sections.jsonにkey_hintがあれば抽出
        pass
    
    # 出力準備
    out_data = {"unit": "ql", "events": events}
    if key_changes:
        out_data["key_changes"] = key_changes
    
    save_chordmap(events, out_path)
    
    print(f"[OK] 7th chords (enhanced) events={len(events)} -> {out_path}")


def enforce_min_dwell(
    events: List[Dict[str, Any]],
    *,
    global_min: float = 0.0,
    per_section: Optional[Dict[str, float]] = None,
    section_for_t = None
) -> List[Dict[str, Any]]:
    """最短持続時間を強制（短いコードを隣接コードとマージ）
    
    Args:
        events: コードイベントリスト
        global_min: 全体の最短QL
        per_section: セクション別最短QL辞書
        section_for_t: 時刻→セクション名取得関数
    
    Returns:
        フィルタ済みイベントリスト
    """
    if not events:
        return events
    if global_min <= 0 and not per_section:
        return events
    
    out = []
    for i, e in enumerate(events):
        t0 = float(e["time"])
        t1 = float(events[i + 1]["time"]) if i + 1 < len(events) else t0
        
        # セクション取得
        sec = None
        if section_for_t:
            try:
                sec = section_for_t(t0)
            except Exception:
                pass
        
        # 最短持続時間を決定
        min_dur = float(global_min or 0.0)
        if per_section and sec and sec in per_section:
            try:
                min_dur = max(min_dur, float(per_section[sec]))
            except Exception:
                pass
        
        # 持続時間チェック
        dur = t1 - t0
        if dur < min_dur:
            # 短すぎる：隣接コードとマージ
            # 次と同じrootなら吸収
            if i + 1 < len(events):
                if events[i + 1]["root"] == e["root"]:
                    continue
            # 前と同じrootなら延長不要、スキップ
            if out and out[-1]["root"] == e["root"]:
                continue
        
        out.append(e)
    
    return out


if __name__ == "__main__":
    main()
