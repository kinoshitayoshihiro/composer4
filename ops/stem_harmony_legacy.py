#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ops/stem_harmony.py  (enhanced)

librosa + テンプレート + HMM/Viterbi によるコード推定。
拡張:
- 局所キー（モデュレーション）: 8〜16拍窓の局所キー事前重み
- No-Chord (N) 状態: 低エネルギー/低確信度で無和音を明示
- sections.json の拍子/テンポを参照して QL 換算を厳密化（フォールバックあり）
- ステム個別重み: Bass/Keys を強め、FX を弱め等

出力: chordmap.json 形式
  {"unit": "ql", "events": [{"time": <QL(float)>, "root": "C#", "quality": "maj|min|N"}]}

依存: numpy, librosa
    pip install numpy librosa
"""
from __future__ import annotations
import argparse, json, sys, math
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import numpy as np
import librosa

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

# ---------------- Sections (QL mapping) ----------------
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
def mix_harmonic(files: List[Path], sr: int, weights: List[Tuple[str,float]]) -> Tuple[np.ndarray, int]:
    """ステムを読み込み、ハーモニック成分のみ重み付き合成。"""
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

def chroma_sync(y_h: np.ndarray, sr: int, bins_per_octave: int = 36):
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
    beat_times = librosa.frames_to_time(beats, sr=sr)
    try:
        tuning = librosa.estimate_tuning(y=y_h, sr=sr)
    except Exception:
        tuning = 0.0
    C = librosa.feature.chroma_cqt(y=y_h, sr=sr, bins_per_octave=bins_per_octave, tuning=tuning)
    C_sync = librosa.util.sync(C, beats, aggregate=np.median)  # [12,T]
    return C_sync, float(tempo), beat_times

# ---------------- Priors: global & local key ----------------
def key_profile_major() -> np.ndarray:
    return np.array([6.35,2.23,3.48,2.33,4.38,4.09,2.52,5.19,2.39,3.66,2.29,2.88], dtype=float)
def key_profile_minor() -> np.ndarray:
    return np.array([6.33,2.68,3.52,5.38,2.60,3.53,2.54,4.75,3.98,2.69,3.34,3.17], dtype=float)

def global_key_prior(C_sync: np.ndarray) -> np.ndarray:
    """24次元の一様 + グローバルキー偏り（弱め）"""
    profM = key_profile_major(); profM /= profM.sum()
    scoresM = np.array([np.sum(C_sync * rotate12(profM, k)[:,None]) for k in range(12)])
    kM = int(np.argmax(scoresM))
    prior = np.ones(24, dtype=float) * 1e-3
    # I,ii,iii,IV,V,vi あたりを優遇（簡易）
    degs = [0,2,4,5,7,9]
    for d in degs:
        root = (kM + d) % 12
        prior[root] += 1.0   # maj
        prior[root+12] += 0.6
    return prior / prior.sum()

def local_key_prior(C_sync: np.ndarray, win: int = 8) -> np.ndarray:
    """
    8〜16拍窓で局所キーを推定し、各フレーム毎の 24次元 prior を返す [24,T]。
    簡易: メジャー/マイナーのいずれかで最大スコアのキーを採用し、そのダイアトニックに重み。
    """
    T = C_sync.shape[1]
    profM = key_profile_major(); profm = key_profile_minor()
    profM /= profM.sum(); profm /= profm.sum()
    priors = np.zeros((24, T), dtype=float)
    degs_M = [0,2,4,5,7,9]  # I,ii,iii,IV,V,vi
    degs_m = [0,2,3,5,7,8]  # i, ii°, III, iv, v, VI くらいの簡易（厳密ではない）
    for t in range(T):
        a = max(0, t - win//2); b = min(T, t + (win - win//2))
        Cw = C_sync[:, a:b]
        sM = np.array([np.sum(Cw * rotate12(profM, k)[:,None]) for k in range(12)])
        sN = np.array([np.sum(Cw * rotate12(profm, k)[:,None]) for k in range(12)])
        if sM.max() >= sN.max():
            k = int(np.argmax(sM)); prior = np.ones(24, dtype=float) * 1e-3
            for d in degs_M:
                r = (k+d)%12
                prior[r] += 1.0; prior[r+12] += 0.6
        else:
            k = int(np.argmax(sN)); prior = np.ones(24, dtype=float) * 1e-3
            for d in degs_m:
                r = (k+d)%12
                prior[r+12] += 1.0; prior[r] += 0.3
        priors[:, t] = prior / prior.sum()
    return priors  # [24,T]

# ---------------- HMM ----------------
def build_transition(S: int, stay: float, near: float, include_N: bool, n_stay: float = 0.96, n_out: float = 0.02) -> np.ndarray:
    """
    S = 24 or 25 (含N)。Row: from, Col: to
    - Chord内: self=stay, 4th/5th=near、その他微小一様
    - N状態: 強い self (n_stay)、他へ n_out、残りは微小一様
    """
    if include_N:
        assert S == 25
    else:
        assert S == 24
    A = np.zeros((S,S), dtype=float)
    K = 24  # chord states
    base = (1.0 - stay - 2*near) / (K - 3) if K > 3 else 0.0
    for r in range(12):
        # Maj block
        A[r, :] = base
        A[r, r] = stay
        A[r, (r+7)%12] += near
        A[r, (r+5)%12] += near
        # Min block
        i = r + 12
        A[i, :] = base
        A[i, i] = stay
        A[i, ((r+7)%12)+12] += near
        A[i, ((r+5)%12)+12] += near
    if include_N:
        N = 24
        A[:K, N] += 1e-3  # chord->N 微小
        A[N, :] = (1.0 - n_stay - n_out*K) / max(1, S-1)
        A[N, N] = n_stay
        A[N, :K] += n_out
    # 正規化
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
                 gamma_local: float,
                 include_N: bool,
                 n_energy_gamma: float,
                 n_conf_gamma: float) -> np.ndarray:
    """
    戻り: loglik [S,T] (S=24 or 25)
    - 24テンプレ + cosine 類似度
    - グローバル/ローカルキー prior を log で加算
    - N 状態は「低エネルギー」「低確信度」で上げる
    """
    Tmaj = np.stack([rotate12(major_template(), k) for k in range(12)], axis=1)  # [12,12]
    Tmin = np.stack([rotate12(minor_template(), k) for k in range(12)], axis=1)  # [12,12]
    T24  = np.concatenate([Tmaj, Tmin], axis=1)  # [12,24]

    S = cos_sim_columns(C_sync, T24)  # [T,24]
    S = np.maximum(S, 1e-9)
    loglik = np.log(S.T)              # [24,T]

    # global key
    if gamma_global > 0.0:
        gp = global_key_prior(C_sync)[:, None]  # [24,1]
        loglik += gamma_global * np.log(np.maximum(gp, 1e-12))

    # local key per frame
    if gamma_local > 0.0:
        lp = local_key_prior(C_sync, win=8)  # [24,T]
        loglik += gamma_local * np.log(np.maximum(lp, 1e-12))

    if include_N:
        # N の尤度: 低エネルギー + 低確信度のとき高く
        energy = C_sync.sum(axis=0)  # [T]
        energy_norm = energy / max(1e-6, np.median(energy))
        conf = S.max(axis=1)         # [T] 最大類似度
        # 単純合成: exp( -gammaE * energy_norm ) * exp( gammaC * (1-conf) )
        lnN = (-n_energy_gamma * energy_norm) + (n_conf_gamma * (1.0 - conf))
        lnN -= lnN.max()  # 数値安定
        likN = np.exp(lnN) + 1e-12   # [T]
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

    # 合成とクロマ
    weights_cli = parse_stem_weights(args.stem_weight)
    weights_cfg = parse_stem_weights(params.get("stem_weight", []))
    y_h, sr = mix_harmonic(files, sr=args.sr, weights=(weights_cfg or []) + (weights_cli or []))
    C_sync, tempo, beat_times = chroma_sync(y_h, sr, bins_per_octave=args.bins_per_octave)

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

    stems_dir = Path(args.stems)
    out_path  = Path(args.out)
    sections_path = Path(args.sections) if args.sections else None

    files = list_audio_files(stems_dir, args.exclude)
    if not files:
        print(f"[ERROR] No WAV files found in {stems_dir} (after excludes={args.exclude})", file=sys.stderr)
        sys.exit(2)
    weights = parse_stem_weights(args.stem_weight)

    y_h, sr = mix_harmonic(files, sr=args.sr, weights=weights)
    C_sync, tempo, beat_times = chroma_sync(y_h, sr, bins_per_octave=args.bins_per_octave)

    loglik = build_loglik(
        C_sync,
        gamma_global=args.gamma_global,
        gamma_local=args.gamma_local,
        include_N=args.include_N,
        n_energy_gamma=args.n_energy_gamma,
        n_conf_gamma=args.n_conf_gamma,
    )
    S = 25 if args.include_N else 24
    A = build_transition(S=S, stay=args.stay, near=args.near, include_N=args.include_N, n_stay=args.n_stay, n_out=args.n_out)
    path = viterbi(loglik, A)

    beat_to_ql = load_sections_mapper(sections_path, default_ql_per_beat=args.ql_per_beat)
    events = path_to_events(path, beat_to_ql)
    save_chordmap(events, out_path)
    print(f"[OK] chordmap events={len(events)} -> {out_path}")

if __name__ == "__main__":
    main()
