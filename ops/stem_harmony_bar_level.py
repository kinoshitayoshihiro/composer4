#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ops/stem_harmony_bar_level.py - Bar-level chord recognition

従来の問題：
- HMMのstay確率が高く、コード変化が少ない（1セクション1コード）
- ビート同期が粗く、小節粒度にならない

新アプローチ：
- 小節ごとにクロマを集約
- 各小節で最適なコードを独立に決定
- HMMは小節間の遷移のみに使用（stay確率を大幅に下げる）
"""
from __future__ import annotations

# ⚠️ 環境対策: numba JIT完全無効化（librosa 0.10.2.post1 + numba 0.58.1でも必要）
import os

os.environ["NUMBA_DISABLE_JIT"] = "1"  # numbaのJITを完全オフ（遅くなるがエラー回避）
os.environ["NUMBA_THREADING_LAYER"] = "workqueue"  # 競合回避
os.environ["NUMBA_CACHE_DIR"] = "/tmp/numba-cache"  # 壊れたキャッシュ回避

import argparse, json, sys, hashlib
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import numpy as np
import librosa
from scipy import ndimage

# Import from stem_harmony.py
sys.path.insert(0, str(Path(__file__).parent))
from stem_harmony import (
    NOTE_NAMES,
    rotate12,
    cos_sim_columns,
    list_audio_files,
    parse_stem_weights,
    load_sections_mapper_and_labeler,
    key_profile_major,
    key_profile_minor,
    mix_harmonic,
)

# numba回避版を使用
from audio_safe import chroma_sync_safe

# v4.1: Cache
from cache_utils import hash_params, ensure_cache_dir, compute_and_cache, digest_files, save_npz


# Extended chord templates
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


# --- Decoration v3: Triad-only → 装飾付与（add9/6/sus2/sus4） ---
import json
from collections import deque
from typing import Callable

_NAME_TO_PC = {
    "C": 0,
    "C#": 1,
    "Db": 1,
    "D": 2,
    "D#": 3,
    "Eb": 3,
    "E": 4,
    "F": 5,
    "F#": 6,
    "Gb": 6,
    "G": 7,
    "G#": 8,
    "Ab": 8,
    "A": 9,
    "A#": 10,
    "Bb": 10,
    "B": 11,
}


def _parse_priority(spec: str):
    """優先順位パース: "add9>6>sus" → [("add9",["add9"]), ("6",["6"]), ("sus",["sus4","sus2"])]"""
    tokens = [t.strip().lower() for t in spec.split(">") if t.strip()]
    out = []
    for t in tokens:
        if t == "sus":
            out.append(("sus", ["sus4", "sus2"]))
        elif t in ("add9", "6", "sus4", "sus2"):
            out.append((t, [t]))
    if not out:
        out = [("add9", ["add9"]), ("6", ["6"]), ("sus", ["sus4", "sus2"])]
    return out


def _load_decoration_config(path: Optional[str]) -> Dict[str, dict]:
    """セクション別しきい値のJSON/YAML読み込み"""
    if not path or not os.path.exists(path):
        return {}
    if path.endswith(".json"):
        data = json.load(open(path, "r", encoding="utf-8"))
    else:
        try:
            import yaml

            data = yaml.safe_load(open(path, "r", encoding="utf-8"))
        except Exception:
            data = {}
    out = {}
    if isinstance(data, dict):
        for k, v in data.items():
            if isinstance(v, dict):
                out[k] = v
    return out


def _scale_set(tonic_pc: int, mode: str):
    """キーのダイアトニックスケール（7音）"""
    if mode == "maj":
        return {(tonic_pc + i) % 12 for i in (0, 2, 4, 5, 7, 9, 11)}
    return {(tonic_pc + i) % 12 for i in (0, 2, 3, 5, 7, 8, 10)}  # natural minor


def decorate_events_with_bar_chroma_v3(
    events: list,
    C_bars: np.ndarray,  # (12, B)
    label_at_ql: Optional[Callable[[float], Optional[str]]] = None,
    ql_per_bar: float = 4.0,
    # しきい値（ON/OFF分離）
    tau_on: float = 0.30,  # add9/6 付与
    tau_off: Optional[float] = None,  # 未指定なら tau_on-0.05
    sus_tau_on: float = 0.35,  # sus 付与
    sus_tau_off: Optional[float] = None,  # 未指定なら sus_tau_on-0.05
    # 安定化
    switch_margin: float = 0.08,  # 装飾→別装飾へ切替の要求差
    min_dwell_bars: int = 1,  # 最短持続
    triad_guard: float = 0.35,  # 3rdがこれ以上強いとsus禁止
    # 密度リミッタ（スライド窓）
    density_window_bars: int = 8,
    max_decorations_per_window: int = 6,
    # 優先順位
    priority: str = "add9>6>sus",
    allow_sus: bool = True,
    allow_m6: bool = False,  # 最小パッチではデフォルトOFF
    # キー・バイアス（非ダイアトニックを弱める）
    key_bias: Optional[Tuple[int, str]] = None,  # (tonic_pc, "maj"|"min")
    key_bias_penalty: float = 0.05,
    # セクション別上書き
    section_cfg: Optional[Dict[str, dict]] = None,
    # ChatGPT推奨: triad vs 装飾のバランス調整
    triad_gain: float = 0.75,  # triad強度を控えめに（0.65-0.85推奨）
    triad_w3: float = 0.60,  # 3rdの重み
    triad_w5: float = 0.40,  # 5thの重み
    sus_third_coeff: float = 0.80,  # sus計算時の3rd差し引き係数（0.75-0.85推奨）
    decor_margin: float = -0.02,  # 装飾vsトライアドの勝利マージン（-0.04～0.00推奨）
):
    """
    Triad-only イベントを入力し、装飾を 1 つだけ付与（add9 / 6 / sus2/sus4）。
    - 二重付与禁止: 1イベントにつき装飾は高々1種
    - フリップフロップ防止: ヒステリシス + 最短持続
    - セクション毎の閾値: section_cfg により上書き可能
    - 優先順位: priority で切替（例: "sus>add9>6"）
    """
    if C_bars.size == 0:
        return events

    tau_off = tau_on - 0.05 if tau_off is None else tau_off
    sus_tau_off = sus_tau_on - 0.05 if sus_tau_off is None else sus_tau_off

    B = C_bars.shape[1]
    Cn = C_bars / (C_bars.sum(axis=0, keepdims=True) + 1e-9)

    # セクション設定
    cfg_all = section_cfg or {}
    base = dict(
        tau_on=tau_on,
        tau_off=tau_off,
        sus_tau_on=sus_tau_on,
        sus_tau_off=sus_tau_off,
        switch_margin=switch_margin,
        min_dwell_bars=min_dwell_bars,
        triad_guard=triad_guard,
        allow_sus=allow_sus,
    )

    def cfg_for_section(name: Optional[str]):
        c = dict(base)
        if "default" in cfg_all:
            c.update(cfg_all["default"])
        if name in cfg_all:
            c.update(cfg_all[name])
        return c

    prio = _parse_priority(priority)

    # キー集合
    scale = _scale_set(*key_bias) if key_bias else None

    def deco_score_map(r: int, q0: str, c: np.ndarray):
        """装飾スコア計算（ChatGPT推奨: triad控えめ & sus柔軟化）"""
        third_pc = (r + (3 if q0 == "m" else 4)) % 12
        fifth_pc = (r + 7) % 12
        deg2, deg4, deg6 = (r + 2) % 12, (r + 5) % 12, (r + 9) % 12

        # 追加音スコア（susは3rdを弱めに差し引く）
        s_add9 = c[deg2]
        s_6 = c[deg6] if (q0 == "" or allow_m6) else -1.0
        s_sus2 = c[deg2] - sus_third_coeff * c[third_pc]  # 0.80 * 3rdを引く
        s_sus4 = c[deg4] - sus_third_coeff * c[third_pc]  # 0.80 * 3rdを引く

        # triad強度は控えめ & 重み付き（3rd優先）
        triad_strength = triad_gain * (triad_w3 * c[third_pc] + triad_w5 * c[fifth_pc])

        # キー・バイアス（追加音がスケール外なら減点）
        if scale is not None:
            if deg2 not in scale:
                s_add9 -= key_bias_penalty
                s_sus2 -= key_bias_penalty
            if deg4 not in scale:
                s_sus4 -= key_bias_penalty
            if ((r + 9) % 12) not in scale:
                s_6 -= key_bias_penalty

        return {
            "add9": s_add9,
            "6": s_6,
            "sus2": s_sus2,
            "sus4": s_sus4,
            "triad": triad_strength,
            "third": c[third_pc],
        }

    # 密度制御
    win = deque(maxlen=max(1, int(density_window_bars)))
    win_count = 0

    out = []
    last_change_bar = -(10**9)
    last_deco = None  # "", "m", "add9","6","sus2","sus4"

    for ev in events:
        q0 = ev.get("quality", "")  # triad-only想定
        if q0 not in ("", "m"):
            # triad以外は触らない
            out.append(ev)
            last_deco = q0
            # 装飾扱いにカウント
            win.append(1)
            win_count = sum(win)
            continue

        b = int(np.clip(round(ev["time"] / ql_per_bar), 0, B - 1))
        c = Cn[:, b]
        sec_name = label_at_ql(ev["time"]) if label_at_ql else None
        cfg = cfg_for_section(sec_name)

        sc = deco_score_map(_NAME_TO_PC.get(ev["root"], 0), q0, c)

        # triad-guard: 3rdが強いならsusを封じる
        if sc["third"] >= float(cfg["triad_guard"]):
            sc["sus2"], sc["sus4"] = -1.0, -1.0

        # 候補（ONしきい値）
        cand = {}
        if sc["add9"] >= float(cfg["tau_on"]):
            cand["add9"] = sc["add9"]
        if sc["6"] >= float(cfg["tau_on"]):
            cand["6"] = sc["6"]
        if cfg["allow_sus"]:
            if sc["sus4"] >= float(cfg["sus_tau_on"]):
                cand["sus4"] = sc["sus4"]
            if sc["sus2"] >= float(cfg["sus_tau_on"]):
                cand["sus2"] = sc["sus2"]

        # まず、前装飾の"保持"可否を評価（OFFしきい値）
        keep_last = False
        prev_score = -1.0
        if last_deco in ("add9", "6", "sus2", "sus4"):
            prev_score = sc[last_deco]
            off_th = float(cfg["sus_tau_off"] if "sus" in last_deco else cfg["tau_off"])
            if prev_score >= off_th:
                keep_last = True

        # 優先順位で新候補を選ぶ
        new_q, chosen_score = q0, -1.0
        for key, variants in prio:
            best = None
            best_val = -1.0
            for v in variants:
                if v in cand and cand[v] > best_val:
                    best, best_val = v, cand[v]
            if best is not None:
                new_q, chosen_score = best, best_val
                break

        # 最短持続
        if last_deco in ("add9", "6", "sus2", "sus4") and (b - last_change_bar) < int(
            cfg["min_dwell_bars"]
        ):
            new_q, chosen_score = last_deco, prev_score

        # "保持 vs 切替"の最終判定（スイッチ・マージン）
        if keep_last and new_q != last_deco:
            if chosen_score < (prev_score + float(cfg["switch_margin"])):
                new_q, chosen_score = last_deco, prev_score

        # triad優先（勝ち切れない/密度超過）
        triad_strength = sc["triad"]
        # 密度リミッタ
        if new_q in ("add9", "6", "sus2", "sus4"):
            if len(win) == win.maxlen and sum(win) >= int(max_decorations_per_window):
                new_q = q0  # triadへ戻す

        # triadに勝ち切れているか（ChatGPT推奨: マージンを緩めて装飾を採用しやすく）
        if new_q in ("add9", "6", "sus2", "sus4"):
            # decor_margin=-0.02: 装飾がtriadと同程度〜わずかに弱くても採用
            if chosen_score < (triad_strength + float(decor_margin)):
                new_q = q0

        # 出力・状態更新
        if new_q != q0:
            ev = {**ev, "quality": new_q}
            if new_q != last_deco:
                last_change_bar = b
            last_deco = new_q
            win.append(1)
        else:
            last_deco = q0
            win.append(0)

        win_count = sum(win)
        out.append(ev)

    return out


def estimate_key_ks(C_sync: np.ndarray) -> Tuple[int, str]:
    """
    Krumhansl-Schmuckler 簡易版で曲全体のキーを推定 (ChatGPT推奨)

    Args:
        C_sync: Chroma features, either [12, T] (multi-frame) or [12] (single vector)

    Returns: (tonic_pc[0..11], "maj"|"min")
    """
    maj_prof = key_profile_major()
    min_prof = key_profile_minor()

    # 全体平均クロマ
    C = np.maximum(C_sync, 0.0)

    # Handle both 1D and 2D input
    if C.ndim == 2:
        C = C / (C.sum(axis=0, keepdims=True) + 1e-12)
        mean_c = C.mean(axis=1)  # (12,)
    else:
        # Already 1D
        mean_c = C / (C.sum() + 1e-12)

    best = (-1e9, 0, "maj")
    for mode, prof in (("maj", maj_prof), ("min", min_prof)):
        for tonic in range(12):
            score = float((mean_c * rotate12(prof, tonic)).sum())
            if score > best[0]:
                best = (score, tonic, mode)
    _, tonic_pc, mode = best
    return tonic_pc, mode


def build_chord_templates(triad_only: bool = False) -> Tuple[np.ndarray, List[Tuple[str, str]]]:
    """Build chord templates

    Args:
        triad_only: If True, only build major/minor triads (24 states)
                   If False, build extended templates (72 states)

    Returns:
        templates: [12, N] array (12 chroma bins, N chord types)
        labels: [(root, quality), ...] list of N labels
    """
    if triad_only:
        # Triad-only: major ("") and minor ("m") - 24 states total
        types = [
            ("", maj_template()),  # major
            ("m", min_template()),  # minor
        ]
    else:
        # Extended: major, minor, sus4, sus2, add9, 6th - 72 states
        types = [
            ("", maj_template()),  # major
            ("m", min_template()),  # minor
            ("sus4", sus4_template()),  # sus4
            ("sus2", sus2_template()),  # sus2
            ("add9", add9_template()),  # add9
            ("6", sixth_template()),  # 6th
        ]

    templates = []
    labels = []

    for root_idx in range(12):
        for quality, tmpl in types:
            rotated = rotate12(tmpl, root_idx)
            templates.append(rotated)
            labels.append((NOTE_NAMES[root_idx], quality))

    return np.array(templates).T, labels  # [12, N]


def aggregate_chroma_by_bars(
    C_sync: np.ndarray, beat_times: np.ndarray, ql_per_bar: float = 4.0
) -> Tuple[np.ndarray, np.ndarray]:
    """Aggregate chroma by bars (DEPRECATED: use aggregate_chroma_by_bar_times)

    Args:
        C_sync: [12, T] chroma synchronized to beats
        beat_times: [T] beat times in QL
        ql_per_bar: QL per bar (default: 4.0)

    Returns:
        C_bars: [12, B] chroma aggregated by bars
        bar_times: [B] bar start times in QL
    """
    if len(beat_times) == 0:
        return C_sync, np.array([0.0])

    max_ql = beat_times[-1]
    num_bars = int(np.ceil(max_ql / ql_per_bar))

    C_bars = []
    bar_times_list = []

    for bar_idx in range(num_bars):
        bar_start = bar_idx * ql_per_bar
        bar_end = (bar_idx + 1) * ql_per_bar

        # Find beats in this bar
        mask = (beat_times >= bar_start) & (beat_times < bar_end)
        if not np.any(mask):
            # No beats in this bar, use previous bar's chroma
            if C_bars:
                C_bars.append(C_bars[-1])
            else:
                C_bars.append(np.zeros(12))
        else:
            # Average chroma over beats in this bar
            bar_chroma = np.mean(C_sync[:, mask], axis=1)
            C_bars.append(bar_chroma)

        bar_times_list.append(bar_start)

    return np.array(C_bars).T, np.array(bar_times_list)


def aggregate_chroma_by_bar_times(
    y: np.ndarray,
    sr: int,
    bar_starts_sec: np.ndarray,
    bar_ends_sec: np.ndarray,
    n_fft: int = 4096,
    hop_length: int = 512,
) -> np.ndarray:
    """Aggregate chroma by bar time intervals from bars.parquet (numba-free)

    Args:
        y: Audio signal
        sr: Sample rate
        bar_starts_sec: [B] bar start times in seconds (from bars.parquet)
        bar_ends_sec: [B] bar end times in seconds (from bars.parquet)
        n_fft: FFT size for chroma
        hop_length: Hop length for chroma

    Returns:
        C_bars: [12, B] chroma aggregated by bars (guaranteed to match len(bar_starts_sec))
    """
    from scipy import signal

    # STFT computation (scipy - no numba)
    f, t, Zxx = signal.stft(y, fs=sr, nperseg=n_fft, noverlap=n_fft - hop_length)

    # Power spectrogram
    S = np.abs(Zxx) ** 2

    # Manual chroma filterbank (avoiding librosa.filters.chroma which uses numba)
    freqs = f  # Frequency bins from STFT
    n_bins = len(freqs)

    # Reference: A440 tuning, C0 = 16.35 Hz
    # Chroma mapping: C=0, C#=1, D=2, ..., B=11
    chroma_fb = np.zeros((12, n_bins))

    # Map each frequency bin to chroma class
    for i, freq in enumerate(freqs):
        if freq < 20.0:  # Skip very low frequencies
            continue

        # Convert frequency to semitones above C0 (16.35 Hz)
        C0 = 16.351597831287414  # C0 in Hz
        semitones = 12.0 * np.log2(freq / C0)

        # Chroma class (0-11, C=0)
        chroma_class = int(np.round(semitones)) % 12

        # Weighted contribution (Gaussian-like weighting around the nearest semitone)
        deviation = semitones - np.round(semitones)
        weight = np.exp(-0.5 * (deviation / 0.25) ** 2)  # Narrow Gaussian

        chroma_fb[chroma_class, i] += weight

    # Normalize filterbank columns
    col_sums = chroma_fb.sum(axis=0, keepdims=True)
    col_sums[col_sums == 0] = 1.0
    chroma_fb /= col_sums

    # Apply filterbank to get chroma features
    C = np.dot(chroma_fb, S)

    # Normalize chroma vectors (L2 norm per time frame)
    col_norms = np.sqrt(np.sum(C**2, axis=0, keepdims=True))
    col_norms[col_norms < 1e-10] = 1.0
    C /= col_norms

    # Time axis from STFT
    frame_times = t

    # Aggregate by bars
    num_bars = len(bar_starts_sec)
    C_bars = np.zeros((12, num_bars))

    for bar_idx in range(num_bars):
        start_sec = bar_starts_sec[bar_idx]
        end_sec = bar_ends_sec[bar_idx]

        # Find frames within this bar's time range
        mask = (frame_times >= start_sec) & (frame_times < end_sec)

        if np.any(mask):
            # Average chroma over frames in this bar
            C_bars[:, bar_idx] = np.mean(C[:, mask], axis=1)
        else:
            # No frames in this bar - use previous bar or zeros
            if bar_idx > 0:
                C_bars[:, bar_idx] = C_bars[:, bar_idx - 1]
            else:
                C_bars[:, bar_idx] = 0.0

    return C_bars


def recognize_chords_per_bar(
    C_bars: np.ndarray, templates: np.ndarray, labels: List[Tuple[str, str]], smoothing: float = 0.1
) -> Tuple[List[int], np.ndarray]:
    """Recognize chords for each bar independently (FIXED: no undefined reference)

    Args:
        C_bars: [12, B] bar-aggregated chroma
        templates: [12, S] chord templates
        labels: [(root, quality), ...] chord labels
        smoothing: Temporal smoothing strength (0-1)

    Returns:
        chord_indices: [B] chord index for each bar
        confidences: [B] confidence scores
    """
    # Compute similarity
    sim = cos_sim_columns(C_bars, templates)  # [B, S]
    B, S = sim.shape

    chord_idx = np.empty(B, dtype=int)
    conf = np.empty(B, dtype=float)

    # 1小節目
    chord_idx[0] = int(np.argmax(sim[0]))
    conf[0] = float(sim[0, chord_idx[0]])

    # 2小節目以降：前回ラベルのみを微ブースト
    for b in range(1, B):
        if smoothing > 0:
            sim[b, chord_idx[b - 1]] += float(smoothing)
        chord_idx[b] = int(np.argmax(sim[b]))
        conf[b] = float(sim[b, chord_idx[b]])

    return chord_idx.tolist(), conf


def recognize_chords_per_bar_dp(
    C_bars: np.ndarray,
    templates: np.ndarray,
    labels: List[Tuple[str, str]],
    change_penalty: float = 0.15,  # ← ChatGPT推奨: 0.35→0.15 (軽めに)
    key_bias: Optional[Tuple[int, str]] = None,
    epsilon: float = 1e-12,
) -> Tuple[List[int], np.ndarray]:
    """
    遷移ペナルティ付きDP（Pottsモデル）- ChatGPT推奨版 v2
    - 各小節でテンプレートとの類似度から負対数尤度を作り、
      「前小節と同じならコスト0／変わるなら change_penalty」を足して最短路を解きます。
    - これで「コード変化の密度」を change_penalty だけで安定制御できます（0.12〜0.20が目安）。

    Args:
        C_bars: (12, B) 小節集約クロマ
        templates: (12, K) コードテンプレート
        labels: [(root, quality)] * K
        change_penalty: コードが変わる時だけ課すコスト (0.12-0.20推奨)
        key_bias: (tonic_pc, "maj"|"min") を与えると、非ダイアトニックに微小ペナルティ
        epsilon: 数値安定用

    Returns:
        path: List[int]  最良コードindex（各小節）
        conf: np.ndarray (B,) 信頼度 [0..1]
    """
    K = templates.shape[1]
    B = C_bars.shape[1]

    # cos類似度 → 負対数尤度
    Tn = templates / (np.linalg.norm(templates, axis=0, keepdims=True) + epsilon)  # (12,K)
    Cn = C_bars / (np.linalg.norm(C_bars, axis=0, keepdims=True) + epsilon)  # (12,B)
    sim = np.clip((Tn.T @ Cn), 0.0, 1.0)  # (K,B)
    nll = -np.log(sim + epsilon)  # (K,B)

    # ❶ ChatGPT推奨: 拡張コード(6/add9/sus)は骨格では控えめに - 微ペナルティ
    ext_pen = np.zeros(K, dtype=float)
    for k, (_root, q) in enumerate(labels):
        if q in ("6", "add9", "sus2", "sus4"):
            ext_pen[k] = 0.06  # 0.05–0.10で調整。triad (maj/min) を骨格優先に
    nll += ext_pen[:, None]  # (K,) → (K, 1) broadcasting to (K, B)

    # ダイアトニック・バイアス（任意）
    if key_bias is not None:
        tonic_pc, mode = key_bias
        name_to_pc = {
            "C": 0,
            "C#": 1,
            "Db": 1,
            "D": 2,
            "D#": 3,
            "Eb": 3,
            "E": 4,
            "F": 5,
            "F#": 6,
            "Gb": 6,
            "G": 7,
            "G#": 8,
            "Ab": 8,
            "A": 9,
            "A#": 10,
            "Bb": 10,
            "B": 11,
        }

        def base_kind(q):
            q = q or ""
            if "min" in q:
                return "min"
            if "maj" in q or "sus" in q or "add" in q or q == "6":
                return "maj"
            return "maj"

        dia_maj = {0: "maj", 2: "min", 4: "min", 5: "maj", 7: "maj", 9: "min", 11: "dim"}
        dia_min = {0: "min", 2: "dim", 3: "maj", 5: "min", 7: "min", 8: "maj", 10: "maj"}
        dia = dia_maj if mode == "maj" else dia_min

        penalty = np.zeros_like(nll)
        for k, (root, qual) in enumerate(labels):
            pc = name_to_pc.get(root, 0)
            deg = (pc - tonic_pc) % 12
            need = dia.get(deg, None)
            kind = base_kind(qual)
            in_key = (need == "dim" and kind == "min") or (need in ("maj", "min") and kind == need)
            if not in_key:
                penalty[k, :] = 0.08  # ごく小さいペナルティ（暴走防止用）
        nll += penalty

    # DP
    dp = np.zeros_like(nll)
    bp = np.zeros((K, B), dtype=np.int32)

    dp[:, 0] = nll[:, 0]
    for b in range(1, B):
        prev = dp[:, b - 1]  # (K,)
        stay_cost = prev  # 同コード継続
        change_cost = np.min(prev) + change_penalty  # 別コードへ
        best_prev_idx = int(np.argmin(prev))
        for s in range(K):
            if stay_cost[s] <= change_cost:
                dp[s, b] = nll[s, b] + stay_cost[s]
                bp[s, b] = s
            else:
                dp[s, b] = nll[s, b] + change_cost
                bp[s, b] = best_prev_idx

    # 後退
    path = [int(np.argmin(dp[:, B - 1]))]
    for b in range(B - 1, 0, -1):
        path.append(int(bp[path[-1], b]))
    path = list(reversed(path))

    # 信頼度＝softmaxベースの事後確率（ChatGPT推奨: 常に0..1に収まる）
    def _softmax(v: np.ndarray, temp: float = 8.0) -> np.ndarray:
        """数値安定化したsoftmax"""
        x = v - np.max(v)
        ex = np.exp(x * temp)
        return ex / (np.sum(ex) + 1e-12)

    conf = np.empty(B, dtype=float)
    for b in range(B):
        post = _softmax(sim[:, b], temp=8.0)  # simは類似度（高いほど良い）
        s = path[b]
        conf[b] = float(post[s])

    conf = np.clip(conf, 0.0, 1.0)  # 念のためclamp

    # 健全性チェック
    assert np.all(np.isfinite(conf)), "NaN/Inf in confidence"
    assert np.all(
        (conf >= 0.0) & (conf <= 1.0)
    ), f"confidence not in [0,1]: min={conf.min():.3f}, max={conf.max():.3f}"

    return path, conf


def main():
    ap = argparse.ArgumentParser(description="Bar-level chord recognition (1 bar = 1 chord)")
    ap.add_argument("--stems", required=True, help="Stems directory")
    ap.add_argument("--out", required=True, help="Output chordmap.json")
    ap.add_argument(
        "--audio",
        type=str,
        default=None,
        help="Explicit mix audio path (e.g. stem_wav/instrument.wav). If given, prefer this over auto-detection.",
    )
    ap.add_argument("--sections", help="sections.json (optional)")
    ap.add_argument("--bars", help="bars.parquet (for bar count/tempo alignment)")
    ap.add_argument("--exclude", action="append", default=[], help="Exclude stems")
    ap.add_argument("--force-key", help="Force key (e.g., 'C', 'Am')")
    ap.add_argument("--sr", type=int, default=22050, help="Sample rate")
    ap.add_argument("--bins-per-octave", type=int, default=36, help="CQT bins")
    ap.add_argument("--ql-per-bar", type=float, default=4.0, help="QL per bar")
    ap.add_argument("--smoothing", type=float, default=0.1, help="Temporal smoothing (0-1)")
    ap.add_argument("--stem-weight", action="append", default=[], help="Stem weight")

    # ChatGPT推奨: 遷移ペナルティ制御 (0.12-0.20推奨)
    ap.add_argument(
        "--change-penalty",
        type=float,
        default=0.15,
        help="DP change penalty (0.12-0.20 recommended)",
    )
    ap.add_argument("--use-dp", action="store_true", help="Use DP-based recognition (recommended)")
    ap.add_argument(
        "--triad-only",
        action="store_true",
        help="Use only major/minor triads (24 states), then decorate as needed",
    )

    # v4.1 options
    ap.add_argument("--cache-dir", type=str, default=None, help="Cache directory")
    ap.add_argument("--no-cache", action="store_true", help="Disable cache")
    ap.add_argument("--emit-confidence", action="store_true", help="Emit confidence")
    ap.add_argument("--min-dwell-bars", type=int, default=1, help="Min chord duration in bars")

    # 装飾フェーズv3オプション (ChatGPT推奨: triad-only → 必要箇所のみ装飾)
    ap.add_argument(
        "--decorate",
        action="store_true",
        help="After triad decoding, add add9/6/sus only where bar-chroma strongly supports it",
    )
    ap.add_argument(
        "--decorate-priority",
        type=str,
        default="add9>6>sus",
        help='Decoration priority. Examples: "sus>add9>6", "add9>6>sus", "6>add9>sus"',
    )
    ap.add_argument("--dec-tau-on", type=float, default=0.30, help="add9/6 ON threshold")
    ap.add_argument(
        "--dec-tau-off",
        type=float,
        default=None,
        help="add9/6 OFF threshold (default: tau_on-0.05)",
    )
    ap.add_argument("--dec-sus-tau-on", type=float, default=0.35, help="sus ON threshold")
    ap.add_argument(
        "--dec-sus-tau-off",
        type=float,
        default=None,
        help="sus OFF threshold (default: sus_tau_on-0.05)",
    )
    ap.add_argument(
        "--dec-switch-margin",
        type=float,
        default=0.08,
        help="Margin required for decoration switch",
    )
    ap.add_argument(
        "--dec-triad-guard",
        type=float,
        default=0.35,
        help="3rd threshold to block sus (higher → more triads)",
    )
    ap.add_argument(
        "--dec-density-window-bars",
        type=int,
        default=8,
        help="Sliding window size for density limiter",
    )
    ap.add_argument("--dec-density-max", type=int, default=6, help="Max decorations per window")
    ap.add_argument("--dec-no-sus", action="store_true", help="Disable sus globally")
    ap.add_argument("--dec-allow-m6", action="store_true", help="Enable m6 chord type")
    ap.add_argument(
        "--dec-key-bias", type=str, default=None, help="Key bias (e.g., 'D:maj' or 'B:min')"
    )
    ap.add_argument(
        "--dec-key-bias-penalty",
        type=float,
        default=0.05,
        help="Penalty for non-diatonic decoration",
    )
    ap.add_argument(
        "--dec-config",
        type=str,
        default=None,
        help="Section-specific thresholds JSON/YAML (default/Verse/Chorus...)",
    )

    # ChatGPT推奨: triad vs 装飾のバランス調整
    ap.add_argument(
        "--dec-triad-gain",
        type=float,
        default=0.75,
        help="Triad strength multiplier (0.65-0.85, lower → more decorations)",
    )
    ap.add_argument("--dec-triad-w3", type=float, default=0.60, help="3rd weight in triad strength")
    ap.add_argument("--dec-triad-w5", type=float, default=0.40, help="5th weight in triad strength")
    ap.add_argument(
        "--dec-sus-third-coeff",
        type=float,
        default=0.80,
        help="sus 3rd subtraction coefficient (0.75-0.85)",
    )
    ap.add_argument(
        "--dec-decor-margin",
        type=float,
        default=-0.02,
        help="Decoration vs triad margin (-0.04 to 0.00)",
    )

    args = ap.parse_args()

    stems_dir = Path(args.stems)
    out_path = Path(args.out)
    sections_path = Path(args.sections) if args.sections else None

    # --- Choose audio source deterministically (instrument.wav priority) ---
    chosen_audio = None
    if args.audio:
        ap = Path(args.audio)
        if ap.is_file():
            chosen_audio = str(ap)
            print(f"[INFO] Using audio for chord recognition: {chosen_audio}")
        else:
            print(f"[WARN] --audio given but not found: {ap}", file=sys.stderr)

    if chosen_audio is None:
        # Prefer instrument.wav > mix.wav > other.wav (existence check)
        candidates = ["instrument.wav", "mix.wav", "Mix.wav", "other.wav", "Other.wav"]
        for name in candidates:
            candidate_path = stems_dir / name
            if candidate_path.is_file():
                chosen_audio = str(candidate_path)
                print(f"[INFO] Using audio for chord recognition: {chosen_audio} (auto-detected)")
                break

    if chosen_audio is None:
        print("[ERROR] No suitable mix audio found (instrument/mix/other).", file=sys.stderr)
        sys.exit(2)

    files = list_audio_files(stems_dir, args.exclude)
    if not files:
        print("[ERROR] No WAV files found", file=sys.stderr)
        sys.exit(2)

    # Cache setup
    cache_dir_path = Path(args.cache_dir) if args.cache_dir else (stems_dir / ".cache")
    use_cache = not args.no_cache
    if use_cache:
        ensure_cache_dir(cache_dir_path)

    # Cache key
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

    # Compute chroma (numba回避版使用)
    # Use chosen_audio as primary source
    def _compute_chroma():
        # Load chosen_audio directly for chord recognition using safe_load_audio (numba回避)
        from ops.audio_safe import safe_load_audio

        y_audio, sr_audio = safe_load_audio(chosen_audio, sr=args.sr, mono=True)
        return (y_audio, np.array([sr_audio]))

    cache_path = cache_dir_path / f"audio_{cache_key}.npz"
    if use_cache and cache_path.exists():
        y_audio, sr_arr = compute_and_cache(
            _compute_chroma,
            cache_path,
            use_cache=True,
            keys=("y_audio", "sr"),
        )
        sr_audio = int(sr_arr[0])
        print(f"[CACHE] HIT: audio_{cache_key[:8]}.npz")
    else:
        y_audio, sr_arr = _compute_chroma()
        sr_audio = int(sr_arr[0])
        if use_cache:
            save_npz(cache_path, y_audio=y_audio, sr=sr_arr)
            print(f"[CACHE] SAVE: audio_{cache_key[:8]}.npz")

    # bars.parquetから小節時刻を読み込み（必須）
    if not args.bars or not Path(args.bars).exists():
        print("[ERROR] --bars is required for bar-level chord recognition", file=sys.stderr)
        sys.exit(2)

    import pandas as pd

    bars_df = pd.read_parquet(args.bars)
    bars_df = bars_df.sort_values("bar_index").reset_index(drop=True)

    if "start_sec" not in bars_df.columns or "end_sec" not in bars_df.columns:
        print("[ERROR] bars.parquet must contain start_sec and end_sec columns", file=sys.stderr)
        sys.exit(2)

    bar_starts_sec = bars_df["start_sec"].values
    bar_ends_sec = bars_df["end_sec"].values
    num_bars = len(bars_df)

    print(
        f"[INFO] bars.parquet specifies {num_bars} bars (range: {bar_starts_sec[0]:.2f}s - {bar_ends_sec[-1]:.2f}s)"
    )

    # Aggregate chroma by bar time intervals (bars.parquetベース)
    C_bars = aggregate_chroma_by_bar_times(
        y_audio,
        sr_audio,
        bar_starts_sec,
        bar_ends_sec,
        n_fft=4096,
        hop_length=512,
    )

    print(f"[INFO] Chroma aggregated: {C_bars.shape[1]} bars (matches bars.parquet: {num_bars})")

    # bar_times for compatibility (use start_sec)
    bar_times = bar_starts_sec

    print(f"[INFO] Aggregated to {C_bars.shape[1]} bars")

    # Build templates (triad-only if --triad-only, otherwise extended)
    templates, labels = build_chord_templates(triad_only=args.triad_only)
    n_states = len(labels)
    print(
        f"[INFO] Using {'triad-only' if args.triad_only else 'extended'} templates: {n_states} states"
    )

    # キー推定 (ChatGPT推奨: 自動推定 or --force-key)
    kb = None
    if args.force_key:
        # "C" or "Am" をパース
        _pc = {
            "C": 0,
            "C#": 1,
            "Db": 1,
            "D": 2,
            "D#": 3,
            "Eb": 3,
            "E": 4,
            "F": 5,
            "F#": 6,
            "Gb": 6,
            "G": 7,
            "G#": 8,
            "Ab": 8,
            "A": 9,
            "A#": 10,
            "Bb": 10,
            "B": 11,
        }
        name = args.force_key.strip()
        kb = (_pc[name[:-1]], "min") if name.endswith("m") else (_pc[name], "maj")
        print(f"[INFO] Forced key: {name} → {kb}")
    else:
        # C_barsから自動推定（全小節の平均クロマから）
        avg_chroma = np.mean(C_bars, axis=1)
        tonic_pc, mode = estimate_key_ks(avg_chroma)
        kb = (tonic_pc, mode)
        key_name = NOTE_NAMES[tonic_pc] + ("m" if mode == "min" else "")
        print(f"[INFO] Estimated key: {key_name} → {kb}")

    # Recognize chords per bar (DP版 or 簡易版)
    if args.use_dp:
        chord_indices, confidences = recognize_chords_per_bar_dp(
            C_bars,
            templates,
            labels,
            change_penalty=args.change_penalty,
            key_bias=kb,  # ← ChatGPT推奨: 弱いバイアスとして使用
        )
        print(f"[INFO] DP-based recognition (change_penalty={args.change_penalty}, key_bias={kb})")
    else:
        chord_indices, confidences = recognize_chords_per_bar(
            C_bars, templates, labels, smoothing=args.smoothing
        )
        print(f"[INFO] Greedy recognition (smoothing={args.smoothing})")

    # Build events
    events = []
    for bar_idx, (chord_idx, conf) in enumerate(zip(chord_indices, confidences)):
        root, quality = labels[chord_idx]
        event = {"time": float(bar_times[bar_idx]), "root": root, "quality": quality}
        if args.emit_confidence:
            event["confidence"] = float(conf)
        events.append(event)

    # Min dwell filter
    if args.min_dwell_bars > 1:
        filtered = [events[0]]
        for ev in events[1:]:
            if ev["root"] == filtered[-1]["root"] and ev["quality"] == filtered[-1]["quality"]:
                # Same chord, skip
                continue
            # Count bars since last change
            bars_since_change = (ev["time"] - filtered[-1]["time"]) / args.ql_per_bar
            if bars_since_change < args.min_dwell_bars:
                # Too short, extend previous chord
                continue
            filtered.append(ev)
        events = filtered

    # 装飾フェーズv3 (ChatGPT推奨: triad-only → 必要箇所のみ装飾)
    if args.decorate:
        # セクション設定のロード
        section_cfg = _load_decoration_config(args.dec_config)

        # キー・バイアス（--dec-key-bias or 推定キーを使用）
        key_bias = None
        if args.dec_key_bias:
            name, mode = args.dec_key_bias.split(":")
            name = name.strip()
            mode = mode.strip().lower()
            tonic = _NAME_TO_PC.get(name.upper(), 0)
            key_bias = (tonic, "maj" if mode.startswith("maj") else "min")
        elif kb:
            key_bias = kb  # 推定キーを使用

        # QL → section名 の関数（セクション情報なしの場合はNone返す）
        def get_section(ql):
            return None  # セクション情報は使わない（bars.parquetベースのため）

        events = decorate_events_with_bar_chroma_v3(
            events,
            C_bars,
            label_at_ql=get_section,
            ql_per_bar=args.ql_per_bar,
            tau_on=args.dec_tau_on,
            tau_off=args.dec_tau_off,
            sus_tau_on=args.dec_sus_tau_on,
            sus_tau_off=args.dec_sus_tau_off,
            switch_margin=args.dec_switch_margin,
            min_dwell_bars=args.min_dwell_bars,
            triad_guard=args.dec_triad_guard,
            density_window_bars=args.dec_density_window_bars,
            max_decorations_per_window=args.dec_density_max,
            priority=args.decorate_priority,
            allow_sus=(not args.dec_no_sus),
            allow_m6=args.dec_allow_m6,
            key_bias=key_bias,
            key_bias_penalty=args.dec_key_bias_penalty,
            section_cfg=section_cfg,
            # ChatGPT推奨: triad vs 装飾バランス
            triad_gain=args.dec_triad_gain,
            triad_w3=args.dec_triad_w3,
            triad_w5=args.dec_triad_w5,
            sus_third_coeff=args.dec_sus_third_coeff,
            decor_margin=args.dec_decor_margin,
        )

        # 装飾後の統計
        triads = sum(1 for e in events if e.get("quality") in ("", "m"))
        extended = sum(1 for e in events if e.get("quality") in ("6", "add9", "sus4", "sus2"))
        print(
            f"[INFO] Decoration applied: {triads} triads ({triads/len(events):.1%}), {extended} extended ({extended/len(events):.1%})"
        )

    # Output
    output = {"unit": "ql", "events": events}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"[OK] Bar-level chordmap: {len(events)} events -> {out_path}")
    print(
        f"[INFO] Density: {len(events)} events / {len(bar_times)} bars = {len(events)/len(bar_times):.2f} chords/bar"
    )


if __name__ == "__main__":
    main()
