#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ops/stems_features.py - Stem WAV → Bar-level Features Extraction

Stem WAVファイルから小節別の音響特徴を抽出してParquet保存

Features:
- Drums: hat_density, kick_peak_db, snare_backbeat, fill_likelihood
- Mix: loudness_db, energy_curve
- Anchors同期: vocal_stress_bars

Backend切替（arranger_weights.yaml features_backend）:
- Phase A: madmom（beats/downbeats） + librosa_enhanced（hat_density） + pyloudnorm（LUFS）
- Phase B: YAMNet（hat_density）
- Phase C: Chordino/Essentia（chords/key）

Usage:
    python ops/stems_features.py \
        --stems data/suno_ai/suno_themesong/song_001/stemswav_001 \
        --bars data/suno_ai/suno_themesong/song_001/bars.parquet \
        --anchors data/suno_ai/suno_themesong/song_001/analysis/lyric_anchors.json \
        --output data/suno_ai/suno_themesong/song_001/stem_features.parquet \
        --backend-config configs/arranger_weights.yaml \
        --tempo-bpm 74.68
"""
from __future__ import annotations
import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Sequence

import numpy as np
import pandas as pd
import yaml
from scipy.signal import find_peaks

# audio_safe.py の安全版ローダを使用
try:
    from ops.audio_safe import safe_load_audio, stft_mag, onset_envelope

    HAS_AUDIO_SAFE = True
except ImportError:
    HAS_AUDIO_SAFE = False
    import soundfile as sf
    import librosa

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# FeaturesBackend（バックエンド切替）
try:
    from ops.features_backends import FeaturesBackend

    HAS_BACKENDS = True
except ImportError:
    HAS_BACKENDS = False
    logger.warning("features_backends.py not found, using librosa-only mode")


# ============================================================================
# Audio Loading (audio_safe優先、フォールバック対応)
# ============================================================================


def load_audio(path: Path, sr: int = 22050) -> Tuple[np.ndarray, int]:
    """安全なオーディオ読み込み（audio_safe優先）"""
    if HAS_AUDIO_SAFE:
        return safe_load_audio(str(path), sr=sr, mono=True)
    else:
        # librosaフォールバック
        y, loaded_sr = librosa.load(path, sr=sr, mono=True)
        return y, loaded_sr


# ============================================================================
# Drums Features Extraction
# ============================================================================


def extract_drums_features(
    drums_path: Path,
    bars_df: pd.DataFrame,
    backend=None,  # FeaturesBackend instance (optional)
    sr: int = 22050,
) -> pd.DataFrame:
    """
    Drums Stem → 小節別特徴抽出

    Features:
    - hat_density: ハイハット密度（notes/beat）
    - kick_peak_db: キックピーク強度（dB）
    - snare_backbeat: スネアバックビートスコア（0-1）
    - fill_likelihood: Fill確率（0-1、セクション境界で高）

    Args:
        drums_path: Drums Stem WAVファイルパス
        bars_df: bars.parquet DataFrame
        backend: FeaturesBackend instance (None時はlibrosa-only)
        sr: サンプリングレート
    """
    logger.info(f"Extracting drums features from: {drums_path}")

    y, _ = load_audio(drums_path, sr=sr)

    features = []

    # hat_density: backendがあればbar単位で抽出
    # （YAMLノブ適用は後段のintegrate_stem_features()で実施）
    for idx, bar in bars_df.iterrows():
        start_sec = bar["start_sec"]
        end_sec = bar["end_sec"]
        start = int(start_sec * sr)
        end = int(end_sec * sr)
        seg = y[start:end]

        # Feature extraction with backend support
        if backend and hasattr(backend, "extract_hat_density"):
            # Backend使用（librosa_enhanced / yamnet / panns）
            hat_density = backend.extract_hat_density(drums_path, y, sr, start_sec, end_sec)
        else:
            # Fallback: 既存librosa実装
            hat_density = _hat_density(seg, sr, bar.get("beats", 4))

        kick_peak_db = _kick_peak_db(seg, sr)
        snare_backbeat = _snare_backbeat_score(seg, sr, bar.get("beats", 4))
        fill_likelihood = _fill_likelihood(
            seg,
            sr,
            is_section_boundary=bar.get("is_section_boundary", False),
            bar_in_section=bar.get("bar_in_section", 0),
        )

        features.append(
            {
                "bar": idx,
                "hat_density": float(hat_density),
                "kick_peak_db": float(kick_peak_db),
                "snare_backbeat": float(snare_backbeat),
                "fill_likelihood": float(fill_likelihood),
            }
        )

    logger.info(f"Extracted drums features for {len(features)} bars")
    return pd.DataFrame(features)


def _hat_density(y: np.ndarray, sr: int, beats: int) -> float:
    """ハイハット密度（6-12kHz帯域のOnset数/beat）"""
    if len(y) == 0:
        return 0.0

    # High-pass filter (簡易版: pre-emphasis)
    y_hp = np.diff(y, prepend=y[0])

    if HAS_AUDIO_SAFE:
        _, _, mag = stft_mag(y_hp, sr, n_fft=2048, hop_length=512)
        env = onset_envelope(mag, smooth=8)
    else:
        env = librosa.onset.onset_strength(y=y_hp, sr=sr, hop_length=512)

    # Onset検出
    peaks, _ = find_peaks(env, prominence=np.median(env) * 0.5)

    return len(peaks) / max(beats, 1)


def _kick_peak_db(y: np.ndarray, sr: int) -> float:
    """キックピーク強度（30-120Hz帯域）"""
    if len(y) == 0:
        return -80.0

    # Low-pass (簡易版: 移動平均)
    window = int(sr / 100)  # ~10ms
    y_lp = np.convolve(y, np.ones(window) / window, mode="same")

    peak = np.max(np.abs(y_lp))
    return 20 * np.log10(peak + 1e-9)


def _snare_backbeat_score(y: np.ndarray, sr: int, beats: int) -> float:
    """スネアバックビートスコア（2拍目・4拍目の相対エネルギー）"""
    if len(y) == 0 or beats < 2:
        return 0.0

    # 拍分割
    hop = len(y) // beats
    beat_energies = []

    for i in range(beats):
        seg = y[i * hop : (i + 1) * hop]
        energy = np.sqrt(np.mean(seg**2))
        beat_energies.append(energy)

    # 2拍目・4拍目のエネルギー
    backbeat_energy = 0.0
    if beats >= 2:
        backbeat_energy += beat_energies[1]
    if beats >= 4:
        backbeat_energy += beat_energies[3]

    avg_energy = np.mean(beat_energies) + 1e-9

    return backbeat_energy / (avg_energy * min(beats // 2, 2))


def _fill_likelihood(
    y: np.ndarray, sr: int, is_section_boundary: bool, bar_in_section: int
) -> float:
    """Fill確率（0-1、セクション境界・エネルギー勾配ベース）"""
    if len(y) == 0:
        return 0.0

    # ベースライン: セクション境界
    score = 0.8 if is_section_boundary else 0.2

    # セクション末尾ブースト
    if bar_in_section >= 7 and not is_section_boundary:
        score = max(score, 0.6)

    # エネルギー勾配チェック
    rms = np.sqrt(np.mean(y**2))
    score = min(1.0, score + rms * 0.3)

    return score


# ============================================================================
# Mix Features Extraction
# ============================================================================


def extract_mix_features(
    mix_path: Path,
    bars_df: pd.DataFrame,
    backend=None,  # FeaturesBackend instance (optional)
    sr: int = 22050,
) -> pd.DataFrame:
    """
    Mix/Other Stem → 小節別特徴抽出

    Features:
    - loudness_db: ラウドネス（dB）
    - energy_curve: エネルギーカーブ（0-1正規化）

    Args:
        mix_path: Mix/Other Stem WAVファイルパス
        bars_df: bars.parquet DataFrame
        backend: FeaturesBackend instance (None時はRMS-only)
        sr: サンプリングレート
    """
    logger.info(f"Extracting mix features from: {mix_path}")

    y, _ = load_audio(mix_path, sr=sr)

    features = []

    for idx, bar in bars_df.iterrows():
        start_sec = bar["start_sec"]
        end_sec = bar["end_sec"]
        start = int(start_sec * sr)
        end = int(end_sec * sr)
        seg = y[start:end]

        # Loudness with backend support
        if backend and hasattr(backend, "extract_loudness"):
            # Backend使用（pyloudnorm LUFS / essentia）
            loudness_db = backend.extract_loudness(y, sr, start_sec, end_sec)
        else:
            # Fallback: RMS Loudness
            rms = np.sqrt(np.mean(seg**2))
            loudness_db = 20 * np.log10(rms + 1e-9)

        features.append(
            {
                "bar": idx,
                "loudness_db": float(loudness_db),
            }
        )

    df = pd.DataFrame(features)

    # Energy Curve正規化（0-1）
    min_db = df["loudness_db"].min()
    max_db = df["loudness_db"].max()

    if max_db > min_db:
        df["energy_curve"] = (df["loudness_db"] - min_db) / (max_db - min_db)
    else:
        df["energy_curve"] = 0.5

    logger.info(f"Extracted mix features for {len(features)} bars")
    return df


# ============================================================================
# Instrument Activity Helpers (Band-RMS from Other stem)
# ============================================================================


def compute_band_rms_per_bar(
    y: np.ndarray, sr: int, bars_df: pd.DataFrame, band_low_hz: float, band_high_hz: float
) -> np.ndarray:
    """
    帯域フィルタ適用後のRMSを小節ごとに算出

    Args:
        y: Audio signal (mono)
        sr: Sample rate
        bars_df: bars DataFrame with start_sec/end_sec
        band_low_hz: Band pass filter low cutoff (Hz)
        band_high_hz: Band pass filter high cutoff (Hz)

    Returns:
        Array of RMS values per bar
    """
    import librosa
    from scipy import signal

    # Butterworth band-pass filter (4th order)
    sos = signal.iirfilter(
        4, [band_low_hz, band_high_hz], rs=24, btype="band", ftype="butter", fs=sr, output="sos"
    )

    # Apply filter
    y_filtered = signal.sosfilt(sos, y)

    # Compute RMS per bar
    rms_values = []
    for idx, bar in bars_df.iterrows():
        start_sample = int(bar["start_sec"] * sr)
        end_sample = int(bar["end_sec"] * sr)

        segment = y_filtered[start_sample:end_sample]
        if len(segment) > 0:
            rms = float(np.sqrt(np.mean(segment**2)))
        else:
            rms = 0.0

        rms_values.append(rms)

    return np.array(rms_values, dtype=np.float32)


def normalize_01(values: np.ndarray, threshold: float = 0.0) -> np.ndarray:
    """
    0-1正規化（MinMax後に閾値適用）

    Args:
        values: Input array
        threshold: Threshold to apply AFTER normalization (0..1)

    Returns:
        Normalized array [0, 1]
    """
    v = np.array(values, dtype=np.float32)

    vmin = float(np.min(v))
    vmax = float(np.max(v))

    if vmax > vmin:
        normalized = (v - vmin) / (vmax - vmin)
    else:
        normalized = np.zeros_like(v)

    # 正規化後に閾値適用
    normalized[normalized < threshold] = 0.0

    return np.clip(normalized, 0.0, 1.0)


# ============================================================================
# Vocal Anchors Integration
# ============================================================================


def extract_vocal_stress_bars(anchors_path: Optional[Path], bars_df: pd.DataFrame) -> pd.DataFrame:
    """
    Vocal Anchors → Stress発生バー検出

    Features:
    - vocal_stress: Stress anchor存在フラグ（0/1）
    """
    if not anchors_path or not anchors_path.exists():
        logger.warning("No anchors file, skipping vocal stress detection")
        return pd.DataFrame({"bar": bars_df.index, "vocal_stress": 0})

    logger.info(f"Extracting vocal stress from: {anchors_path}")

    with open(anchors_path, "r", encoding="utf-8") as f:
        anchors_data = json.load(f)

    anchors = anchors_data.get("anchors", [])

    # Stress anchorの時刻抽出
    stress_times = [a["time"] for a in anchors if a.get("class") == "stress"]

    # バーごとに判定
    vocal_stress = []

    for idx, bar in bars_df.iterrows():
        start = bar["start_sec"]
        end = bar["end_sec"]

        # この区間にstressがあるか
        has_stress = any(start <= t < end for t in stress_times)

        vocal_stress.append({"bar": idx, "vocal_stress": int(has_stress)})

    logger.info(f"Detected {sum(v['vocal_stress'] for v in vocal_stress)} stress bars")
    return pd.DataFrame(vocal_stress)


# ============================================================================
# Main Integration
# ============================================================================


def integrate_stem_features(
    stems_dir: Path,
    bars_df: pd.DataFrame,
    anchors_path: Optional[Path] = None,
    backend=None,  # FeaturesBackend instance (optional)
    drums_pattern: str = "",
    vocals_pattern: str = "",
    yaml_config: dict = None,  # YAMLノブ設定追加
    inst_activity: bool = False,  # 楽器別activity計算フラグ
) -> pd.DataFrame:
    """
    全Stem特徴を統合

    Args:
        stems_dir: Stem WAVファイルディレクトリ
        bars_df: bars.parquet DataFrame
        anchors_path: lyric_anchors.json パス（optional）
        backend: FeaturesBackend instance (optional)
        drums_pattern: Drumsファイル検出パターン（glob、例: 'stem_wav_*_(Drums).wav'）
        vocals_pattern: Vocalsファイル検出パターン（glob、例: 'stem_wav_*_(Vocals).wav'）
        yaml_config: YAMLノブ設定（optional）
        inst_activity: 楽器別activity計算フラグ（guitar/piano/strings_activity列追加）

    Returns:
        DataFrame with columns:
        - bar, hat_density, kick_peak_db, snare_backbeat, fill_likelihood,
          loudness_db, energy_curve, vocal_stress
        - (optional) guitar_activity, piano_activity, strings_activity
    """
    # YAMLノブ取得
    fb_cfg = yaml_config or {}
    norm_cfg = fb_cfg.get("normalization", {})
    da_cfg = fb_cfg.get("drums_active", {})

    # ヘルパー関数（YAMLノブ適用用）
    def _ma(series: pd.Series, k: int) -> pd.Series:
        """移動平均スムージング"""
        if k is None or k <= 1:
            return series
        return series.rolling(window=k, min_periods=1, center=True).mean()

    def _percentile_scale(x: pd.Series, p_low=5, p_high=95, target=(0.0, 1.0)) -> pd.Series:
        """パーセンタイル正規化"""
        lo, hi = np.percentile(x.values, [p_low, p_high]) if len(x) else (0.0, 1.0)
        if hi <= lo:
            return pd.Series(np.zeros_like(x), index=x.index)
        t0, t1 = target
        y = (x - lo) / (hi - lo)
        y = np.clip(y, 0.0, 1.0)
        return pd.Series(t0 + y * (t1 - t0), index=x.index)

    def _minmax(x: pd.Series, clip=(0.0, 1.0), target=(0.0, 1.0)) -> pd.Series:
        """MinMax正規化"""
        lo, hi = (np.min(x.values) if len(x) else 0.0, np.max(x.values) if len(x) else 1.0)
        if hi <= lo:
            return pd.Series(np.zeros_like(x), index=x.index)
        y = (x - lo) / (hi - lo)
        y = np.clip(y, 0.0, 1.0)
        t0, t1 = target
        return pd.Series(t0 + y * (t1 - t0), index=x.index)

    def _apply_norm(x: pd.Series, spec: dict, default_target=(0.0, 1.0)) -> pd.Series:
        """統一正規化API"""
        method = (spec or {}).get("method", "minmax")
        if method == "percentile":
            p_low = spec.get("p_low", 5)
            p_high = spec.get("p_high", 95)
            target = tuple(spec.get("target_range", list(default_target)))
            y = _percentile_scale(x, p_low, p_high, target)
        elif method == "zscore":
            mu = float(np.mean(x)) if len(x) else 0.0
            sigma = float(np.std(x)) + 1e-8
            z = (x - mu) / sigma
            # z→0..1へ（±2σに収めて）
            y = (np.clip(z, -2, 2) + 2) / 4.0
            y = pd.Series(y, index=x.index)
        else:
            target = tuple(spec.get("target_range", list(default_target)))
            y = _minmax(x, clip=tuple(spec.get("clip_to", [0.0, 1.0])), target=target)
        k = int((spec or {}).get("bar_smooth_bars", 1))
        return _ma(pd.Series(y, index=x.index), k)

    def _hysteresis(values: pd.Series, low_high=(0.12, 0.18), min_len=2) -> pd.Series:
        """ヒステリシス判定（チラつき防止）"""
        low, high = low_high
        state = 0
        runlen = 0
        out = []
        for v in values.values:
            if state == 0 and v >= high:
                state = 1
                runlen = 1
            elif state == 1 and v < low:
                state = 0
                runlen = 0
            else:
                if state == 1:
                    runlen += 1
            out.append(state)
            # active化直後に最短継続
            if state == 1 and runlen < max(1, min_len):
                out[-1] = 1
        return pd.Series(out, index=values.index, dtype=float)

    # Drums特徴（パターン優先、なければ自動検出）
    drums_path = None
    if drums_pattern:
        candidates = sorted(stems_dir.glob(drums_pattern))
        if candidates:
            drums_path = candidates[0]
            logger.info(f"Drums detected by pattern: {drums_path.name}")

    if not drums_path:
        for candidate in stems_dir.glob("*[Dd]rums*.wav"):
            drums_path = candidate
            break

    if drums_path:
        drums_df = extract_drums_features(drums_path, bars_df, backend)  # backend渡し
    else:
        logger.warning("No drums stem found, using zeros")
        drums_df = pd.DataFrame(
            {
                "bar": bars_df.index,
                "hat_density": 0.0,
                "kick_peak_db": -80.0,
                "snare_backbeat": 0.0,
                "fill_likelihood": 0.0,
            }
        )

    # Mix特徴（Other.wav、なければ自動合成）
    mix_path = None
    for candidate in stems_dir.glob("*[Oo]ther*.wav"):
        mix_path = candidate
        break

    if not mix_path:
        # Drums/Vocals以外のStemから自動合成
        logger.warning("No 'Other' stem, auto-mixing non-drum/vocal stems")
        import glob

        import soundfile as sf

        all_wavs = sorted(glob.glob(str(stems_dir / "*.wav")))

        # Vocals検出（パターン優先）
        vocals_names = set()
        if vocals_pattern:
            vocals_cands = sorted(stems_dir.glob(vocals_pattern))
            vocals_names.update(c.name for c in vocals_cands)
        for c in stems_dir.glob("*[Vv]ocal*.wav"):
            vocals_names.add(c.name)

        # 除外リスト
        exclude = set()
        if drums_path:
            exclude.add(drums_path.name)
        exclude.update(vocals_names)

        # 合成対象
        cands = [w for w in all_wavs if Path(w).name not in exclude]
        if cands:
            # 1本目基準で軽量ミックス（モノラル化）
            y0, sr = sf.read(cands[0], always_2d=False)
            if y0.ndim == 2:
                y0 = y0.mean(axis=1)
            mix = np.zeros_like(y0, dtype=np.float32)

            for p in cands:
                y, sr2 = sf.read(p, always_2d=False)
                if y.ndim == 2:
                    y = y.mean(axis=1)
                y = y.astype(np.float32)
                if len(y) < len(mix):
                    y = np.pad(y, (0, len(mix) - len(y)))
                elif len(y) > len(mix):
                    mix = np.pad(mix, (0, len(y) - len(mix)))
                mix += y

            # 正規化
            peak = float(np.max(np.abs(mix))) or 1.0
            mix = (mix / (peak * 1.05)).astype(np.float32)
            tmp_other = stems_dir / "_auto_Other.wav"
            sf.write(str(tmp_other), mix, sr)
            mix_path = tmp_other
            logger.info(f"Built Other from {len(cands)} stems -> {tmp_other.name}")
        else:
            logger.warning("No non-drum/vocal stems found for auto-mix")

    if mix_path:
        mix_df = extract_mix_features(mix_path, bars_df, backend)  # backend渡し
        # Keep mix_path for instrument activity calculation (below)
    else:
        logger.warning("No mix stem found, using zeros")
        mix_df = pd.DataFrame({"bar": bars_df.index, "loudness_db": -80.0, "energy_curve": 0.5})
        mix_path = None  # No Other stem available

    # Vocal Stress
    vocal_df = extract_vocal_stress_bars(anchors_path, bars_df)

    # 統合
    merged = drums_df.merge(mix_df, on="bar").merge(vocal_df, on="bar")

    # ==== 楽器別Activity計算（--inst-activity） ====
    if inst_activity and mix_path:
        logger.info("Computing instrument activity (guitar/piano/strings) from Other stem")

        import soundfile as sf

        # Other stem読み込み（モノラル化）
        y_other, sr = sf.read(str(mix_path), always_2d=False)
        if y_other.ndim == 2:
            y_other = y_other.mean(axis=1)
        y_other = y_other.astype(np.float32)

        # YAMLノブから帯域設定取得（デフォルト値あり）
        inst_cfg = fb_cfg.get("instrument_activity", {})
        guitar_cfg = inst_cfg.get("guitar", {})
        piano_cfg = inst_cfg.get("piano", {})
        strings_cfg = inst_cfg.get("strings", {})

        # 帯域設定
        guitar_band = (
            float(guitar_cfg.get("band_low", 2000)),
            float(guitar_cfg.get("band_high", 5000)),
        )
        piano_band = (
            float(piano_cfg.get("band_low", 300)),
            float(piano_cfg.get("band_high", 4000)),
        )
        strings_band = (
            float(strings_cfg.get("band_low", 500)),
            float(strings_cfg.get("band_high", 7000)),
        )

        # 閾値
        guitar_thresh = float(guitar_cfg.get("threshold", 0.4))
        piano_thresh = float(piano_cfg.get("threshold", 0.45))
        strings_thresh = float(strings_cfg.get("threshold", 0.5))

        # 帯域RMS計算
        guitar_rms = compute_band_rms_per_bar(y_other, sr, bars_df, *guitar_band)
        piano_rms = compute_band_rms_per_bar(y_other, sr, bars_df, *piano_band)
        strings_rms = compute_band_rms_per_bar(y_other, sr, bars_df, *strings_band)

        # 0-1正規化（閾値適用）
        merged["guitar_activity"] = normalize_01(guitar_rms, threshold=guitar_thresh)
        merged["piano_activity"] = normalize_01(piano_rms, threshold=piano_thresh)
        merged["strings_activity"] = normalize_01(strings_rms, threshold=strings_thresh)

        # 3bar移動平均で滑らかに（点滅抑制）
        smooth_bars = int(inst_cfg.get("smooth_bars", 3))
        if smooth_bars > 1:
            for col in ("guitar_activity", "piano_activity", "strings_activity"):
                vals = merged[col].values
                smoothed = (
                    pd.Series(vals)
                    .rolling(window=smooth_bars, center=True, min_periods=1)
                    .mean()
                    .values
                )
                merged[col] = smoothed
            logger.info(f"   Applied {smooth_bars}-bar smoothing to activity columns")

        logger.info(
            f"   guitar_activity: mean={merged['guitar_activity'].mean():.3f}, "
            f"active_bars={int((merged['guitar_activity'] > 0.1).sum())}"
        )
        logger.info(
            f"   piano_activity: mean={merged['piano_activity'].mean():.3f}, "
            f"active_bars={int((merged['piano_activity'] > 0.1).sum())}"
        )
        logger.info(
            f"   strings_activity: mean={merged['strings_activity'].mean():.3f}, "
            f"active_bars={int((merged['strings_activity'] > 0.1).sum())}"
        )
    elif inst_activity:
        logger.warning("--inst-activity requested but no Other stem found, skipping")

    # ==== YAMLノブ適用 ====

    # (1) energy_curve 正規化（Lively: パーセンタイル [0.3, 1.0]）
    if norm_cfg.get("energy_curve"):
        energy_norm = _apply_norm(
            pd.Series(merged["energy_curve"].values),
            norm_cfg.get("energy_curve"),
            default_target=(0.0, 1.0),
        )
        merged["energy_curve"] = energy_norm.values

    # (2) hat_density 正規化（オプション、内部用）
    if norm_cfg.get("hat_density"):
        hat_norm = _apply_norm(
            pd.Series(merged["hat_density"].values),
            norm_cfg.get("hat_density"),
            default_target=(0.0, 1.0),
        )
        merged["hat_density_norm"] = hat_norm.values
    else:
        merged["hat_density_norm"] = merged["hat_density"]

    # (3) drums_active 判定（生値ベース + ヒステリシス）
    # 基本判定: hat_density >= 0.15 または kick_peak_db >= -60
    active_raw = ((merged["hat_density"] >= 0.15) | (merged["kick_peak_db"] >= -60.0)).astype(float)

    # ヒステリシス適用（チラつき防止）
    if da_cfg:
        hysteresis_range = tuple(da_cfg.get("hysteresis", [0.10, 0.16]))
        min_len = int(da_cfg.get("min_active_len_bars", 2))
        drums_active = _hysteresis(
            pd.Series(active_raw.values), low_high=hysteresis_range, min_len=min_len
        )
        merged["drums_active"] = drums_active.values
    else:
        merged["drums_active"] = active_raw.values

    # hat_density_norm列は内部用なので削除
    if "hat_density_norm" in merged.columns:
        merged = merged.drop(columns=["hat_density_norm"])

    logger.info(
        f"   drums_active: {merged['drums_active'].sum()} active bars, {(~merged['drums_active'].astype(bool)).sum()} break bars"
    )

    return merged


def main():
    parser = argparse.ArgumentParser(description="Extract bar-level features from Stem WAV files")
    parser.add_argument("--stems", type=Path, required=True, help="Stem WAV directory")
    parser.add_argument("--bars", type=Path, required=True, help="bars.parquet file")
    parser.add_argument(
        "--anchors", type=Path, default=None, help="lyric_anchors.json file (optional)"
    )
    parser.add_argument("--output", type=Path, required=True, help="Output stem_features.parquet")
    parser.add_argument("--sr", type=int, default=22050, help="Sample rate (default: 22050)")
    parser.add_argument(
        "--tempo-bpm",
        type=float,
        default=120.0,
        help="Tempo in BPM (for start_sec calculation, default: 120)",
    )
    parser.add_argument(
        "--backend-config",
        type=Path,
        default=None,
        help="arranger_weights.yaml (features_backend config, optional)",
    )
    parser.add_argument(
        "--extend-bars",
        action="store_true",
        help="Extend bars.parquet with start_sec/end_sec/drums_active (saves to bars_extended.parquet)",
    )
    parser.add_argument(
        "--inst-activity",
        action="store_true",
        help="Add guitar_activity/piano_activity/strings_activity columns (band-RMS from Other stem)",
    )
    # Suno命名対応
    parser.add_argument(
        "--drums-pattern", type=str, default="", help="例: 'stem_wav_*_(Drums).wav'"
    )
    parser.add_argument(
        "--vocals-pattern", type=str, default="", help="例: 'stem_wav_*_(Vocals).wav'"
    )

    args = parser.parse_args()

    # Load backend config + YAMLノブ読み込み
    backend = None
    fb_cfg = {}

    if args.backend_config and args.backend_config.exists() and HAS_BACKENDS:
        import yaml

        with open(args.backend_config, encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}

        features_backend_config = config.get("features_backend", {})
        backend = FeaturesBackend(features_backend_config)
        logger.info(f"Backend config loaded from: {args.backend_config}")

        # YAMLノブ取得（integrate_stem_features に渡す）
        fb_cfg = features_backend_config
    else:
        if args.backend_config and not args.backend_config.exists():
            logger.warning(f"Backend config not found: {args.backend_config}, using librosa-only")
        elif not HAS_BACKENDS:
            logger.warning("features_backends.py not available, using librosa-only")

    # Load bars
    logger.info(f"Loading bars from: {args.bars}")
    bars_df = pd.read_parquet(args.bars)
    logger.info(f"Loaded {len(bars_df)} bars")

    # start_sec/end_secが無い場合は計算（4/4前提、BPM推定）
    if "start_sec" not in bars_df.columns or "end_sec" not in bars_df.columns:
        logger.warning("start_sec/end_sec not in bars.parquet, calculating from bar_index")

        # BPM推定（song_package.yamlから取得するのが理想だが、ここでは固定値またはユーザー指定）
        # デフォルト: 120 BPM、4/4拍子 → 1小節 = 2.0秒
        tempo_bpm = args.tempo_bpm
        sec_per_beat = 60.0 / tempo_bpm
        sec_per_bar = sec_per_beat * 4.0

        # 列名正規化（bar or bar_index）
        bar_col = "bar_index" if "bar_index" in bars_df.columns else "bar"
        bars_df["start_sec"] = bars_df[bar_col] * sec_per_bar
        bars_df["end_sec"] = (bars_df[bar_col] + 1) * sec_per_bar

        logger.info(
            f"   Calculated start_sec/end_sec (tempo={tempo_bpm} BPM, bar_duration={sec_per_bar:.3f}s)"
        )

    # Extract features with backend support + YAMLノブ渡し
    features_df = integrate_stem_features(
        stems_dir=args.stems,
        bars_df=bars_df,
        anchors_path=args.anchors,
        backend=backend,  # backend渡し（Phase A統合完了）
        drums_pattern=args.drums_pattern,
        vocals_pattern=args.vocals_pattern,
        yaml_config=fb_cfg,  # YAMLノブ設定渡し
        inst_activity=args.inst_activity,  # 楽器別activityフラグ渡し
    )

    # Save stem_features.parquet
    args.output.parent.mkdir(parents=True, exist_ok=True)
    features_df.to_parquet(args.output, index=False)

    logger.info(f"✅ Saved stem features to: {args.output}")
    logger.info(f"   Bars: {len(features_df)}")
    logger.info(f"   Columns: {list(features_df.columns)}")

    # bars.parquet拡張（--extend-barsフラグ時）
    if args.extend_bars:
        # bars_dfにstart_sec/end_sec/drums_activeをマージ
        bars_extended = bars_df.copy()

        # start_beat/end_beat追加（未存在時は計算）
        if "start_beat" not in bars_extended.columns:
            bars_extended["start_beat"] = bars_extended.index * 4.0
        if "end_beat" not in bars_extended.columns:
            bars_extended["end_beat"] = (bars_extended.index + 1) * 4.0

        # start_sec/end_sec追加（未存在時は既に計算済み）
        if "start_sec" not in bars_extended.columns:
            bars_extended["start_sec"] = features_df["bar"].map(
                lambda x: x * (60.0 / args.tempo_bpm * 4.0)
            )
            bars_extended["end_sec"] = features_df["bar"].map(
                lambda x: (x + 1) * (60.0 / args.tempo_bpm * 4.0)
            )

        # drums_active追加
        drums_active_map = features_df.set_index("bar")["drums_active"].to_dict()
        bars_extended["drums_active"] = bars_extended.index.map(
            lambda x: drums_active_map.get(x, 1)
        )

        # 保存（bars_extended.parquet）
        bars_extended_path = args.bars.parent / "bars_extended.parquet"
        bars_extended.to_parquet(bars_extended_path, index=False)

        logger.info(f"✅ Saved extended bars to: {bars_extended_path}")
        logger.info("   Added columns: start_beat, end_beat, start_sec, end_sec, drums_active")
        logger.info(f"   drums_active: {bars_extended['drums_active'].sum()} active bars")

    # Stats
    print("\n" + "=" * 60)
    print("Stem Features Statistics")
    print("=" * 60)
    print(features_df.describe())
    print("=" * 60)


if __name__ == "__main__":
    main()
