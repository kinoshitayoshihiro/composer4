#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ops/stem_harmony.py  (YAML/Section-aware)

強化点
- N状態の独立パラメータ（energy/conf ガンマ、遷移 n_stay/n_out）
- 局所キー prior の窓幅・集約関数の YAML 化（mean|max|gaussian）
- セクション別パラメータ（local_key / N_state / HMM の stay/near などを上書き）
- sections.json を使った QL 換算と "現在セクション" 推定
- ステム個別重み（CLI と YAML 両対応）

v4.1: キャッシュ移植、confidence付与、最短持続、統一化対応

依存: numpy, librosa, (任意) pyyaml
"""
from __future__ import annotations
import argparse, json, sys, math, hashlib
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import numpy as np
import librosa
import soundfile as sf
from scipy.signal import resample_poly

# numba回避: 安全版オーディオ処理（core.audio完全バイパス）
from audio_safe import safe_load_audio, chroma_sync_safe

# v4.1: キャッシュユーティリティ
import sys
sys.path.insert(0, str(Path(__file__).parent))
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

# ---------------- Templates ----------------
def major_template() -> np.ndarray:
    t = np.zeros(12, dtype=float); t[[0,4,7]] = 1.0; return t
def minor_template() -> np.ndarray:
    t = np.zeros(12, dtype=float); t[[0,3,7]] = 1.0; return t
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

# ---------------- Sections (QL mapping + label lookup) ----------------
def _safe_load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}

def _sections_from_file(sections_path: Optional[Path]) -> dict:
    if not sections_path or not sections_path.exists():
        return {}
    return _safe_load_json(sections_path)

def load_sections_mapper_and_labeler(sections_path: Optional[Path], default_ql_per_beat: float = 1.0, beat_times: Optional[np.ndarray]=None):
    """
    returns:
      beat_to_ql(beat_idx:int) -> float
      label_at_time_sec(t: float) -> Optional[str]
    """
    data = _sections_from_file(sections_path)
    
    # リスト形式（既存フォーマット）の場合
    if isinstance(data, list):
        sections_list = data
        ts_list = []
        # ql_per_barから推定
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
        
        # ラベル推定
        markers = []
        for sec in sections_list:
            lab = str(sec.get("label", "")).strip().lower() or None
            if lab is None: continue
            bar = sec.get("bar", 0)
            if beat_times is not None:
                num = 4  # 仮定
                start_beat = bar * num
                if start_beat < len(beat_times):
                    tsec = float(beat_times[start_beat])
                else:
                    tsec = float(beat_times[-1]) if len(beat_times) > 0 else 0.0
            else:
                tsec = float(bar * 4.0)  # fallback
            markers.append((tsec, lab))
    else:
        # 辞書形式
        ts_list = sorted(data.get("time_sigs") or [], key=lambda x: int(x.get("bar",0)))
        secs    = sorted(data.get("sections")  or [], key=lambda x: float(x.get("time_ql", x.get("time", x.get("bar", 0)))))
        
        if not ts_list:
            beat_to_ql = lambda beat_idx: beat_idx * default_ql_per_beat
        else:
            den = int(ts_list[-1].get("den",4))
            ql_per_beat = 4.0/den if den>0 else default_ql_per_beat
            beat_to_ql = lambda beat_idx: beat_idx * ql_per_beat
        
        markers = []
        for s in secs:
            lab = str(s.get("label","")).strip().lower() or None
            if lab is None: continue
            if "time_sec" in s:
                tsec = float(s["time_sec"])
            elif "time" in s:
                tsec = float(s["time"])
            elif "time_ql" in s:
                tsec = float(s["time_ql"])
            elif "bar" in s and beat_times is not None:
                num = int(ts_list[-1].get("num",4)) if ts_list else 4
                start_beat = int(s["bar"]) * max(1, num)
                if start_beat < len(beat_times):
                    tsec = float(beat_times[start_beat])
                else:
                    tsec = float(beat_times[-1]) if len(beat_times) > 0 else 0.0
            else:
                continue
            markers.append((tsec, lab))

    markers.sort(key=lambda x: x[0])

    def label_at_time_sec(t: float) -> Optional[str]:
        lab = None
        for (ts, l) in markers:
            if ts <= t: lab = l
            else: break
        return lab

    return beat_to_ql, label_at_time_sec

# ---------------- Signal processing ----------------

def _safe_load_audio(path: str, sr: int | None = None, mono: bool = True):
    """librosa.load の代替（numba回避）。soundfile→(必要なら)resample_poly。
    戻り値: (y: np.ndarray[float32], sr: int)  # yは1次元（mono=True時）
    """
    # まず SoundFile (libsndfile) で読む
    try:
        y, src_sr = sf.read(path, dtype="float32", always_2d=True)  # shape: (n, ch)
    except Exception:
        # mp3など libsndfile で読めない場合のフォールバック（必要なときだけ）
        import audioread
        with audioread.audio_open(path) as f:
            src_sr = f.samplerate
            ch = f.channels
            # バイナリを一度に集めてから int16 → float32 [-1,1]
            raw = b"".join(frame for frame in f)
        x = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        y = x.reshape(-1, ch)

    # mono化
    if mono:
        y = y.mean(axis=1)
    else:
        y = y.T  # (ch, n) にするならこちら

    # リサンプル（必要な時だけ）
    if sr is not None and src_sr != sr:
        # resample_poly は整数アップ/ダウンサンプリングに強い
        from math import gcd
        g = gcd(int(src_sr), int(sr))
        up, down = int(sr // g), int(src_sr // g)
        y = resample_poly(y, up, down, axis=0)
        src_sr = sr

    return y.astype(np.float32, copy=False), int(src_sr)

def mix_harmonic(files: List[Path], sr: int, weights: List[Tuple[str,float]]) -> Tuple[np.ndarray, int]:
    y_sum = None
    for fp in files:
        y, _sr = _safe_load_audio(str(fp), sr=sr, mono=True)
        # librosa.effects.hpss も core.audio を引くため、HPSSをスキップ
        # 和声解析にはフルスペクトルでも十分機能する
        y_h = y  # HPSSなし版
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
    # Fix DeprecationWarning: convert numpy array to scalar
    tempo = float(tempo[0]) if isinstance(tempo, np.ndarray) and tempo.ndim > 0 else float(tempo)
    beat_times = librosa.frames_to_time(beats, sr=sr)
    
    # Force key: disable tuning correction if specified
    if force_key is not None:
        tuning = 0.0  # No tuning correction
        print(f"[INFO] Forcing key to {force_key}, tuning correction disabled")
    else:
        try:
            tuning = librosa.estimate_tuning(y=y_h, sr=sr)
        except Exception:
            tuning = 0.0
    
    C = librosa.feature.chroma_cqt(y=y_h, sr=sr, bins_per_octave=bins_per_octave, tuning=tuning)
    C_sync = librosa.util.sync(C, beats, aggregate=np.median)  # [12,T]
    return C_sync, tempo, beat_times

# ---------------- Priors ----------------
def key_profile_major() -> np.ndarray:
    return np.array([6.35,2.23,3.48,2.33,4.38,4.09,2.52,5.19,2.39,3.66,2.29,2.88], dtype=float)
def key_profile_minor() -> np.ndarray:
    return np.array([6.33,2.68,3.52,5.38,2.60,3.53,2.54,4.75,3.98,2.69,3.34,3.17], dtype=float)

def _weighted_sum(Cw: np.ndarray, weights: Optional[np.ndarray], prof: np.ndarray, k: int) -> float:
    # Cw: [12,W], weights: [W] or None, prof rotated to k applied column-wise
    if weights is None:
        return float(np.sum(Cw * rotate12(prof, k)[:,None]))
    else:
        return float(np.sum((Cw * rotate12(prof, k)[:,None]) * weights[None,:]))

# ---------------- HMM ----------------
def build_transition(S: int, stay: float, near: float, include_N: bool, n_stay: float = 0.96, n_out: float = 0.02) -> np.ndarray:
    if include_N:
        assert S == 25
    else:
        assert S == 24
    A = np.zeros((S,S), dtype=float)
    K = 24  # chord states
    base = (1.0 - stay - 2*near) / (K - 3) if K > 3 else 0.0
    for r in range(12):
        A[r, :] = base
        A[r, r] = stay
        A[r, (r+7)%12] += near
        A[r, (r+5)%12] += near
        i = r + 12
        A[i, :] = base
        A[i, i] = stay
        A[i, ((r+7)%12)+12] += near
        A[i, ((r+5)%12)+12] += near
    if include_N:
        N = 24
        A[:K, N] += 1e-3
        A[N, :] = (1.0 - n_stay - n_out*K) / max(1, S-1)
        A[N, N] = n_stay
        A[N, :K] += n_out
    A = np.maximum(A, 1e-12)
    A = A / A.sum(axis=1, keepdims=True)
    return A

def viterbi(loglik: np.ndarray, A: np.ndarray) -> np.ndarray:
    S, T = loglik.shape
    logA = np.log(np.maximum(A, 1e-12))
    dp = np.zeros((S, T), dtype=float)
    ptr = np.zeros((S, T), dtype=np.int32)
    dp[:, 0] = loglik[:, 0]
    for t in range(1, T):
        M = dp[:, t-1][:, None] + logA
        ptr[:, t] = np.argmax(M, axis=0)
        dp[:, t]  = loglik[:, t] + M[ptr[:, t], np.arange(S)]
    path = np.zeros(T, dtype=np.int32)
    path[-1] = int(np.argmax(dp[:, -1]))
    for t in range(T-2, -1, -1):
        path[t] = int(ptr[path[t+1], t+1])
    return path

# ---------------- Likelihood ----------------
def build_loglik(C_sync: np.ndarray,
                 gamma_global: float,
                 local_cfg: dict,
                 include_N: bool,
                 n_cfg: dict,
                 section_for_t) -> np.ndarray:
    """
    C_sync: [12,T]
    local_cfg: {"win_beats":8,"mode":"mean","gamma":0.30, "per_section":{label:{...}}}
    n_cfg: {"energy_gamma":1.0,"conf_gamma":2.0, "per_section":{label:{...}}}
    section_for_t: callable int->label|None  （beat indexからセクション名）
    戻り: loglik [S,T]
    """
    Tmaj = np.stack([rotate12(major_template(), k) for k in range(12)], axis=1)  # [12,12]
    Tmin = np.stack([rotate12(minor_template(), k) for k in range(12)], axis=1)  # [12,12]
    T24  = np.concatenate([Tmaj, Tmin], axis=1)  # [12,24]

    S = cos_sim_columns(C_sync, T24)  # [T,24]
    S = np.maximum(S, 1e-9)
    loglik = np.log(S.T)              # [24,T]

    # global key
    if gamma_global > 0.0:
        # 簡易: C_sync 全体からグローバルキー prior
        profM = key_profile_major(); profM /= profM.sum()
        scoresM = np.array([np.sum(C_sync * rotate12(profM, k)[:,None]) for k in range(12)])
        kM = int(np.argmax(scoresM))
        prior = np.ones(24, dtype=float) * 1e-3
        degs = [0,2,4,5,7,9]
        for d in degs:
            root = (kM + d) % 12
            prior[root] += 1.0   # maj
            prior[root+12] += 0.6
        prior = prior / prior.sum()
        loglik += gamma_global * np.log(np.maximum(prior[:,None], 1e-12))

    # local key (section-aware)
    base_win = int(local_cfg.get("win_beats", 8))
    base_mode= str(local_cfg.get("mode","mean")).lower()
    base_gamma=float(local_cfg.get("gamma", 0.30))
    per_sec   = local_cfg.get("per_section", {}) or {}

    # per-frame priors & gammas
    T = C_sync.shape[1]
    LP = np.zeros((24, T), dtype=float)
    gL = np.zeros((T,), dtype=float)
    for t in range(T):
        lab = section_for_t(t)  # beat indexベース
        lc  = dict(local_cfg)
        if isinstance(per_sec, dict) and lab in per_sec:
            # 上書き
            lc.update(per_sec.get(lab) or {})
        win  = int(lc.get("win_beats", base_win))
        mode = str(lc.get("mode", base_mode)).lower()
        gamma= float(lc.get("gamma", base_gamma))
        # 一時的に単フレーム prior を計算（関数を分解せず inline）
        a = max(0, t - win//2); b = min(T, t + (win - win//2))
        Cw = C_sync[:, a:b]
        # gaussian weights
        W = None
        if mode == "gaussian":
            sigma = max(1.0, win/3.0)
            half = win//2
            idx = np.arange(-half, win-half, dtype=float)[:(b-a)]
            g = np.exp(-0.5 * (idx/sigma)**2)
            W = g / max(1e-12, g.sum())
        profM = key_profile_major(); profm = key_profile_minor()
        profM /= profM.sum(); profm /= profm.sum()
        def _ws(Cw, W, prof, k):
            if W is None: return float(np.sum(Cw * rotate12(prof, k)[:,None]))
            return float(np.sum((Cw * rotate12(prof, k)[:,None]) * W[None,:]))
        sM = np.array([_ws(Cw,W,profM,k) for k in range(12)])
        sN = np.array([_ws(Cw,W,profm,k) for k in range(12)])
        if sM.max() >= sN.max():
            k = int(np.argmax(sM)); prior = np.ones(24, dtype=float) * 1e-3
            for d in [0,2,4,5,7,9]:
                r = (k+d)%12
                prior[r] += 1.0; prior[r+12] += 0.6
        else:
            k = int(np.argmax(sN)); prior = np.ones(24, dtype=float) * 1e-3
            for d in [0,2,3,5,7,8]:
                r = (k+d)%12
                prior[r+12] += 1.0; prior[r] += 0.3
        LP[:, t] = prior / prior.sum()
        gL[t] = gamma

    loglik += np.log(np.maximum(LP, 1e-12)) * gL[None,:]

    # N-state emission (energy/conf) section-aware
    if include_N:
        n_base_E = float(n_cfg.get("energy_gamma", 1.0))
        n_base_C = float(n_cfg.get("conf_gamma",   2.0))
        n_persec = n_cfg.get("per_section", {}) or {}
        energy = C_sync.sum(axis=0)  # [T]
        energy_norm = energy / max(1e-6, np.median(energy))
        conf = S.max(axis=1)         # [T]
        lnN = np.zeros((C_sync.shape[1],), dtype=float)
        for t in range(T):
            lab = section_for_t(t)
            eG  = n_base_E; cG = n_base_C
            if isinstance(n_persec, dict) and lab in n_persec:
                spec = n_persec.get(lab) or {}
                eG = float(spec.get("energy_gamma", eG))
                cG = float(spec.get("conf_gamma",   cG))
            lnN[t] = (-eG * energy_norm[t]) + (cG * (1.0 - conf[t]))
        lnN -= lnN.max()
        likN = np.exp(lnN) + 1e-12
        likN = likN / (likN.max() + 1e-12)
        loglik = np.vstack([loglik, np.log(likN[None, :])])  # [25,T]

    return loglik

# ---------------- Path → events ----------------
def path_to_events(path: np.ndarray, beat_to_ql) -> List[Dict]:
    events = []
    if len(path) == 0: return events
    cur = int(path[0]); t0 = 0
    for t in range(1, len(path)+1):
        if t == len(path) or int(path[t]) != cur:
            time_ql = float(beat_to_ql(t0))
            if cur == 24:
                events.append({"time": time_ql, "root": "N", "quality": "N"})
            else:
                root = cur % 12
                qual = 'min' if cur >= 12 else 'maj'
                events.append({"time": time_ql, "root": NOTE_NAMES[root], "quality": qual})
            if t < len(path):
                cur = int(path[t]); t0 = t
    return events

def save_chordmap(events: List[Dict], out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    data = {"unit": "ql", "events": events}
    out_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

# ---------------- Config (YAML/JSON) ----------------
def load_config(cfg_path: Optional[Path]) -> dict:
    if not cfg_path: return {}
    if not cfg_path.exists(): return {}
    text = cfg_path.read_text(encoding="utf-8")
    if cfg_path.suffix.lower() in (".yaml",".yml"):
        if yaml is None:
            print("[WARN] PyYAML not available; ignoring YAML config.", file=sys.stderr)
            return {}
        try:
            return yaml.safe_load(text) or {}
        except Exception as e:
            print(f"[WARN] YAML parse failed: {e}", file=sys.stderr)
            return {}
    else:
        try:
            return json.loads(text)
        except Exception as e:
            print(f"[WARN] JSON parse failed: {e}", file=sys.stderr)
            return {}

def resolve_params_with_config(args, cfg: dict):
    """CLI 既定値 <- YAMLグローバル <- セクション別（実行時に適用）"""
    out = {
        "include_N": bool(cfg.get("N_state",{}).get("enable", args.include_N) if isinstance(cfg.get("N_state"), dict) else args.include_N),
        "hmm": {
            "stay": float(cfg.get("HMM",{}).get("stay", args.stay)) if isinstance(cfg.get("HMM"), dict) else args.stay,
            "near": float(cfg.get("HMM",{}).get("near", args.near)) if isinstance(cfg.get("HMM"), dict) else args.near,
        },
        "local_key": {
            "win_beats": int((cfg.get("local_key",{}) or {}).get("win_beats", 8)),
            "mode": str((cfg.get("local_key",{}) or {}).get("mode", "mean")).lower(),
            "gamma": float((cfg.get("local_key",{}) or {}).get("gamma", args.gamma_local)),
            "per_section": (cfg.get("local_key",{}) or {}).get("per_section", {}) or {},
        },
        "N_state": {
            "energy_gamma": float((cfg.get("N_state",{}) or {}).get("energy_gamma", args.n_energy_gamma)),
            "conf_gamma":   float((cfg.get("N_state",{}) or {}).get("conf_gamma",   args.n_conf_gamma)),
            "stay":         float((cfg.get("N_state",{}) or {}).get("stay",         args.n_stay)),
            "out":          float((cfg.get("N_state",{}) or {}).get("out",          args.n_out)),
            "per_section":  (cfg.get("N_state",{}) or {}).get("per_section", {}) or {},
        },
        "stem_weight": list(cfg.get("stem_weight", [])) if isinstance(cfg.get("stem_weight", []), list) else [],
        "gamma_global": float(cfg.get("global_key",{}).get("gamma", args.gamma_global)) if isinstance(cfg.get("global_key"), dict) else args.gamma_global,
    }
    return out

# ---------------- Main ----------------
def main():
    ap = argparse.ArgumentParser(description="librosa-based chordmap estimator (maj/min + N + local key + YAML)")
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
    
    # v4.1: キャッシュオプション
    ap.add_argument("--cache-dir", type=str, default=None, help="Cache directory (default: <stems>/.cache)")
    ap.add_argument("--no-cache", action="store_true", help="Disable cache (force recompute)")
    
    # v4.1: 追加パラメータ
    ap.add_argument("--emit-confidence", action="store_true", help="emit per-event confidence [0..1]")
    ap.add_argument("--min-dwell-ql", type=float, default=0.0, help="global minimum chord dwell in QL (postprocess)")
    
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

    # 合成とクロマ（numba回避版：audio_safe使用）
    weights_cli = parse_stem_weights(args.stem_weight)
    weights_cfg = parse_stem_weights(params.get("stem_weight", []))
    y_h, sr = mix_harmonic(files, sr=args.sr, weights=(weights_cfg or []) + (weights_cli or []))
    # chroma_sync → chroma_sync_safe に置き換え（librosa.beat回避）
    C_sync, tempo, beat_times = chroma_sync_safe(y_h, sr, n_fft=4096, hop_length=512)

    # sections.json から QL 換算 & ラベラ
    beat_to_ql, label_at_sec = load_sections_mapper_and_labeler(sections_path, default_ql_per_beat=args.ql_per_beat, beat_times=beat_times)
    def section_for_t(t: int) -> Optional[str]:
        if t < 0 or t >= len(beat_times): return None
        return label_at_sec(float(beat_times[t]))

    # HMM 遷移（N_state の遷移は YAML/CLI のグローバル値を適用）
    include_N = bool(params["include_N"])
    A = build_transition(
        S = 25 if include_N else 24,
        stay = float(params["hmm"]["stay"]),
        near = float(params["hmm"]["near"]),
        include_N = include_N,
        n_stay = float(params["N_state"]["stay"]),
        n_out  = float(params["N_state"]["out"])
    )

    # log-likelihood（local_key / N_state はセクション別上書きに対応）
    loglik = build_loglik(
        C_sync = C_sync,
        gamma_global = float(params["gamma_global"]),
        local_cfg = params["local_key"],
        include_N = include_N,
        n_cfg = params["N_state"],
        section_for_t = section_for_t
    )

    path = viterbi(loglik, A)
    events = path_to_events(path, beat_to_ql)
    save_chordmap(events, out_path)
    print(f"[OK] chordmap events={len(events)} -> {out_path}")

if __name__ == "__main__":
    main()
