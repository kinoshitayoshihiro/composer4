# ops/audio_safe.py
"""
librosa.core.audio完全回避版：NumPy/SciPyのみでオーディオ処理
numba JIT問題を根本解決するための安全実装
"""
from __future__ import annotations
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly, stft, find_peaks
from math import gcd

def safe_load_audio(path: str, sr: int | None = None, mono: bool = True):
    """librosa.load の完全代替（numba回避）。戻り: (y: float32 1D, sr:int)"""
    y, src_sr = sf.read(path, dtype="float32", always_2d=True)  # (n, ch)
    if mono:
        y = y.mean(axis=1)
    else:
        y = y.T  # (ch, n) を使いたい場合に変更
    if sr is not None and src_sr != sr:
        g = gcd(int(src_sr), int(sr))
        y = resample_poly(y, int(sr//g), int(src_sr//g), axis=0)
        src_sr = sr
    return y.astype(np.float32, copy=False), int(src_sr)

def stft_mag(y: np.ndarray, sr: int, n_fft: int = 4096, hop_length: int = 512, win: str = "hann"):
    """|STFT|（振幅スペクトル）。戻り: (mag: (F,T), frame_rate: sr/hop)"""
    f, t, Z = stft(y, fs=sr, nperseg=n_fft, noverlap=n_fft-hop_length,
                   window=win, boundary=None, padded=False)
    mag = np.abs(Z).astype(np.float32)
    return f, t, mag

def onset_envelope(mag: np.ndarray, mean_norm: bool = True, smooth: int = 8):
    """半波整流スペクトルフラックス + 移動平均スムージング"""
    # 差分（時間方向）
    D = np.diff(mag, axis=1)
    D[D < 0] = 0.0
    env = D.sum(axis=0)
    if mean_norm:
        m = env.mean() + 1e-8
        env = env / m
    if smooth > 1:
        k = np.ones(smooth, dtype=np.float32) / float(smooth)
        env = np.convolve(env, k, mode="same")
    return env.astype(np.float32)

def estimate_tempo_bpm(env: np.ndarray, frame_rate: float, bpm_min=60.0, bpm_max=200.0):
    """自己相関からグローバルBPMの最大ピークを取る簡易法"""
    # 正規化 & DC除去
    e = env - env.mean()
    e = e / (np.std(e) + 1e-8)
    ac = np.correlate(e, e, mode="full")[len(e)-1:]  # 非負ラグ
    # ラグ→BPM
    lmin = int(round(frame_rate * 60.0 / bpm_max))
    lmax = int(round(frame_rate * 60.0 / bpm_min))
    lmin = max(lmin, 1)
    if lmax <= lmin + 1:
        lmax = lmin + 2
    lag = lmin + int(np.argmax(ac[lmin:lmax]))
    bpm = 60.0 * frame_rate / float(lag)
    return float(bpm), int(lag)

def place_beats(env: np.ndarray, lag: int, prominence=0.1, search_radius: int = 2):
    """等間隔ラフグリッド→各近傍で最大ピークにスナップ"""
    T = len(env)
    # 先頭ピーク
    peaks, _ = find_peaks(env, prominence=prominence)
    if len(peaks) == 0:
        # フラットなケース：単純な等間隔
        return np.arange(0, T, lag, dtype=int)
    start = peaks[0]
    grid = list(range(start, T, lag))
    beats = []
    for g in grid:
        a = max(0, g - search_radius)
        b = min(T-1, g + search_radius)
        local = a + int(np.argmax(env[a:b+1]))
        beats.append(local)
    # 単調性の確保（重複除去）
    beats = np.unique(np.array(beats, dtype=int))
    return beats

def chroma_from_stft(freqs: np.ndarray, mag: np.ndarray, ref_A4: float = 440.0):
    """簡易クロマ（HPCPライク）：FFT周波数→MIDI→12ピッチクラスに集約"""
    eps = 1e-12
    # 有効帯域（~ 50Hz〜5kHz 程度）に絞るとノイズが減る
    band = (freqs >= 50.0) & (freqs <= 5000.0)
    f = freqs[band]
    M = mag[band, :]  # (F', T)

    # 周波数→連続MIDI
    midi = 69.0 + 12.0 * np.log2(np.maximum(f, eps) / ref_A4)
    # 最近傍の半音に丸めてPCへ
    pc = (np.round(midi).astype(int) % 12)

    C = np.zeros((12, M.shape[1]), dtype=np.float32)
    for k in range(len(f)):
        C[pc[k]] += M[k]
    # 列正規化
    C /= (C.sum(axis=0, keepdims=True) + 1e-9)
    return C  # (12, T)

def chroma_sync_safe(y: np.ndarray, sr: int, n_fft=4096, hop_length=512):
    """安全版：STFT→onset→テンポ→ビート→クロマのビート同期"""
    freqs, times, mag = stft_mag(y, sr, n_fft=n_fft, hop_length=hop_length)
    env = onset_envelope(mag, smooth=8)
    frame_rate = sr / hop_length
    bpm, lag = estimate_tempo_bpm(env, frame_rate)
    beat_frames = place_beats(env, lag=lag, search_radius=2)

    # クロマ
    C = chroma_from_stft(freqs, mag)
    # ビート位置でサンプル（中央値近似）
    if len(beat_frames) == 0:
        # 何も取れないケースはフレーム全体を返す
        return C, bpm, np.arange(C.shape[1], dtype=int)
    Cb = C[:, beat_frames]
    return Cb, bpm, beat_frames.astype(int)
