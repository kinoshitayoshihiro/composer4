#!/usr/bin/env python3
"""
features_backends.py
--------------------
Stem特徴抽出バックエンド切替（段階導入）

Phase A: madmom（beats/downbeats） + librosa_enhanced（hat_density）
Phase B: YAMNet（hat_density） + pyloudnorm（LUFS）
Phase C: Chordino/Essentia（chords/key）

バックエンド選択はconfigs/arranger_weights.yaml の features_backend セクションで制御。
各バックエンドは個別にインポート可能（欠落時は librosa フォールバック）。
"""

import numpy as np
import librosa
from pathlib import Path
from typing import Optional, Tuple, List, Dict
import logging

logger = logging.getLogger(__name__)


# ========== Beats/Downbeats Backends ==========

def extract_beats_librosa(
    audio: np.ndarray,
    sr: int,
    **kwargs
) -> np.ndarray:
    """
    librosa.beat.beat_track によるビート抽出
    
    Returns:
        beat_times: (N,) ビート時刻（秒）
    """
    _, beat_frames = librosa.beat.beat_track(y=audio, sr=sr, **kwargs)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)
    return beat_times


def extract_beats_madmom(
    audio_path: Path,
    fps: int = 100,
    **kwargs
) -> np.ndarray:
    """
    madmom RNN + DBN によるビート抽出
    
    Args:
        audio_path: 音声ファイルパス（madmomはファイル読み込み）
        fps: 時間分解能（Hz）
    
    Returns:
        beat_times: (N,) ビート時刻（秒）
    """
    try:
        from madmom.features.beats import RNNBeatProcessor, DBNBeatTrackingProcessor
    except ImportError:
        logger.warning("madmom not installed, falling back to librosa")
        audio, sr = librosa.load(str(audio_path), sr=None, mono=True)
        return extract_beats_librosa(audio, sr)
    
    act = RNNBeatProcessor()(str(audio_path))
    beat_times = DBNBeatTrackingProcessor(fps=fps)(act)
    
    logger.info(f"   madmom beats: {len(beat_times)} beats extracted")
    return beat_times


def extract_downbeats_madmom(
    audio_path: Path,
    beats_per_bar: List[int] = [3, 4],
    fps: int = 100,
    **kwargs
) -> Tuple[np.ndarray, np.ndarray]:
    """
    madmom RNN + DBN によるダウンビート抽出
    
    Args:
        audio_path: 音声ファイルパス
        beats_per_bar: 想定拍子（[3, 4]で3/4, 4/4対応）
        fps: 時間分解能（Hz）
    
    Returns:
        downbeat_times: (M,) ダウンビート時刻（秒）
        beat_positions: (M,) ビート位置（1=ダウンビート、2,3,4=拍内位置）
    """
    try:
        from madmom.features.downbeats import RNNDownBeatProcessor, DBNDownBeatTrackingProcessor
    except ImportError:
        logger.warning("madmom not installed, using librosa beats as fallback")
        audio, sr = librosa.load(str(audio_path), sr=None, mono=True)
        beat_times = extract_beats_librosa(audio, sr)
        # フォールバック: 4拍ごとにダウンビート
        downbeat_times = beat_times[::4]
        beat_positions = np.ones(len(downbeat_times), dtype=int)
        return downbeat_times, beat_positions
    
    act = RNNDownBeatProcessor()(str(audio_path))
    result = DBNDownBeatTrackingProcessor(beats_per_bar=beats_per_bar, fps=fps)(act)
    
    # result: (N, 2) [(time, beat_position), ...]
    downbeat_times = result[:, 0]
    beat_positions = result[:, 1].astype(int)
    
    # ダウンビート（position=1）のみ抽出
    downbeat_mask = beat_positions == 1
    downbeat_times = downbeat_times[downbeat_mask]
    
    logger.info(f"   madmom downbeats: {len(downbeat_times)} downbeats extracted")
    return downbeat_times, beat_positions


# ========== Hat Density Backends ==========

def extract_hat_density_librosa(
    audio: np.ndarray,
    sr: int,
    bar_start_sec: float,
    bar_end_sec: float,
    **kwargs
) -> float:
    """
    librosa スペクトルフラックスによるハット密度推定（既存実装）
    
    Returns:
        density: ハット密度（0.0～）
    """
    start_sample = int(bar_start_sec * sr)
    end_sample = int(bar_end_sec * sr)
    bar_audio = audio[start_sample:end_sample]
    
    if len(bar_audio) < sr * 0.1:
        return 0.0
    
    # スペクトルフラックス
    S = np.abs(librosa.stft(bar_audio))
    onset_env = librosa.onset.onset_strength(S=S, sr=sr)
    onset_frames = librosa.onset.onset_detect(
        onset_envelope=onset_env,
        sr=sr,
        backtrack=False
    )
    
    density = len(onset_frames)
    return float(density)


def extract_hat_density_librosa_enhanced(
    audio: np.ndarray,
    sr: int,
    bar_start_sec: float,
    bar_end_sec: float,
    bandpass_low: float = 5000.0,
    bandpass_high: float = 12000.0,
    onset_threshold: float = 0.6,
    aggregate_window: float = 0.1,
    **kwargs
) -> float:
    """
    librosa 帯域限定 + ロバスト閾値によるハット密度推定（Phase A改善版）
    
    Args:
        bandpass_low: ハイパスフィルタ周波数（Hz、5kHz推奨）
        bandpass_high: ローパスフィルタ周波数（Hz、12kHz推奨）
        onset_threshold: オンセット検出閾値（0～1、0.6推奨）
        aggregate_window: フレーム集計窓サイズ（秒、0.1推奨）
    
    Returns:
        density: ハット密度（0.0～）
    """
    start_sample = int(bar_start_sec * sr)
    end_sample = int(bar_end_sec * sr)
    bar_audio = audio[start_sample:end_sample]
    
    if len(bar_audio) < sr * 0.1:
        return 0.0
    
    # 帯域限定（5-12kHz）
    from scipy.signal import butter, sosfilt
    
    # ハイパスフィルタ（5kHz）
    sos_high = butter(4, bandpass_low, btype='high', fs=sr, output='sos')
    filtered = sosfilt(sos_high, bar_audio)
    
    # ローパスフィルタ（12kHz）
    sos_low = butter(4, bandpass_high, btype='low', fs=sr, output='sos')
    filtered = sosfilt(sos_low, filtered)
    
    # スペクトルフラックス（帯域限定後）
    S = np.abs(librosa.stft(filtered, n_fft=2048, hop_length=512))
    onset_env = librosa.onset.onset_strength(S=S, sr=sr, hop_length=512)
    
    # ロバスト閾値でオンセット検出
    onset_frames = librosa.onset.onset_detect(
        onset_envelope=onset_env,
        sr=sr,
        hop_length=512,
        backtrack=False,
        delta=onset_threshold * onset_env.max() if onset_env.max() > 0 else 0.1
    )
    
    # 集計窓内でフレームをカウント
    window_samples = int(aggregate_window * sr)
    window_frames = window_samples // 512
    
    # フレーム密度（集計窓あたりのオンセット数）
    if window_frames > 0:
        density = len(onset_frames) / max(1, len(onset_env) // window_frames)
    else:
        density = len(onset_frames)
    
    return float(density)


def extract_hat_density_yamnet(
    audio_path: Path,
    bar_start_sec: float,
    bar_end_sec: float,
    threshold: float = 0.3,
    target_classes: List[str] = ["Hi-hat", "Cymbal"],
    **kwargs
) -> float:
    """
    YAMNet（AudioSet分類器）によるハット密度推定（Phase B）
    
    Args:
        audio_path: 音声ファイルパス
        bar_start_sec: 小節開始時刻（秒）
        bar_end_sec: 小節終了時刻（秒）
        threshold: Hi-hat確率閾値（0～1）
        target_classes: AudioSetクラス名リスト
    
    Returns:
        density: ハット密度（0.0～）
    """
    try:
        import tensorflow as tf
        import tensorflow_hub as hub
    except ImportError:
        logger.warning("TensorFlow/YAMNet not installed, falling back to librosa_enhanced")
        audio, sr = librosa.load(str(audio_path), sr=None, mono=True)
        return extract_hat_density_librosa_enhanced(audio, sr, bar_start_sec, bar_end_sec)
    
    # YAMNetモデル読み込み（キャッシュ）
    if not hasattr(extract_hat_density_yamnet, '_yamnet_model'):
        model = hub.load('https://tfhub.dev/google/yamnet/1')
        extract_hat_density_yamnet._yamnet_model = model
        # クラス名CSVをダウンロード（TF-Hub 2.x互換）
        import csv
        import io
        try:
            import urllib.request
            csv_url = 'https://raw.githubusercontent.com/tensorflow/models/master/research/audioset/yamnet/yamnet_class_map.csv'
            csv_text = urllib.request.urlopen(csv_url, timeout=10).read().decode('utf-8')
            reader = csv.reader(io.StringIO(csv_text))
            next(reader)  # ヘッダスキップ
            class_names = [row[2] for row in reader]  # display_name列
            extract_hat_density_yamnet._class_names = class_names
        except Exception as e:
            logger.warning(f"   YAMNet class names download failed: {e}, using fallback")
            extract_hat_density_yamnet._class_names = []
        logger.info("   YAMNet model loaded")
    
    model = extract_hat_density_yamnet._yamnet_model
    class_names = extract_hat_density_yamnet._class_names
    
    # 音声読み込み（小節範囲）
    audio, sr = librosa.load(str(audio_path), sr=16000, mono=True)  # YAMNetは16kHz固定
    start_sample = int(bar_start_sec * sr)
    end_sample = int(bar_end_sec * sr)
    bar_audio = audio[start_sample:end_sample]
    
    if len(bar_audio) < sr * 0.1:
        return 0.0
    
    # YAMNet推論
    try:
        scores, embeddings, spectrogram = model(bar_audio)
    except Exception as e:
        logger.warning(f"   YAMNet inference failed: {e}, using librosa fallback")
        return extract_hat_density_librosa_enhanced(audio, sr, bar_start_sec, bar_end_sec)
    
    # Hi-hat/Cymbalクラスのインデックス
    target_indices = []
    for cls in target_classes:
        if cls in class_names:
            target_indices.append(class_names.index(cls))
    
    if len(target_indices) == 0:
        logger.warning(f"   YAMNet: target classes {target_classes} not found")
        return 0.0
    
    # 確率集計（フレームごとに閾値超えをカウント）
    # TensorFlowテンソルをnumpyに変換してからインデックス処理
    scores_np = scores.numpy()  # (frames, 521)
    target_scores = scores_np[:, target_indices]  # (frames, num_classes)
    max_scores = target_scores.max(axis=1)  # (frames,)
    
    density = (max_scores > threshold).sum()
    
    return float(density)


# ========== Loudness Backends ==========

def extract_loudness_rms(
    audio: np.ndarray,
    sr: int,
    bar_start_sec: float,
    bar_end_sec: float,
    **kwargs
) -> float:
    """
    RMSラウドネス（既存実装）
    
    Returns:
        loudness_db: ラウドネス（dB）
    """
    start_sample = int(bar_start_sec * sr)
    end_sample = int(bar_end_sec * sr)
    bar_audio = audio[start_sample:end_sample]
    
    if len(bar_audio) == 0:
        return -80.0
    
    rms = np.sqrt(np.mean(bar_audio ** 2))
    loudness_db = 20 * np.log10(rms + 1e-8)
    
    return float(loudness_db)


def extract_loudness_pyloudnorm(
    audio: np.ndarray,
    sr: int,
    bar_start_sec: float,
    bar_end_sec: float,
    block_size: float = 0.4,
    **kwargs
) -> float:
    """
    pyloudnorm（EBU R128 LUFS）によるラウドネス推定（Phase A/B）
    
    Args:
        audio: 音声信号（mono or stereo）
        sr: サンプリングレート
        bar_start_sec: 小節開始時刻（秒）
        bar_end_sec: 小節終了時刻（秒）
        block_size: EBU R128ブロックサイズ（秒、0.4推奨）
    
    Returns:
        lufs: ラウドネス（LUFS、dB相当）
    """
    try:
        import pyloudnorm as pyln
    except ImportError:
        logger.warning("pyloudnorm not installed, falling back to RMS")
        return extract_loudness_rms(audio, sr, bar_start_sec, bar_end_sec)
    
    start_sample = int(bar_start_sec * sr)
    end_sample = int(bar_end_sec * sr)
    bar_audio = audio[start_sample:end_sample]
    
    if len(bar_audio) < sr * 0.1:
        return -80.0
    
    # EBU R128 Meter
    meter = pyln.Meter(sr, block_size=block_size)
    
    # ステレオ変換（pyloudnormは2ch推奨）
    if bar_audio.ndim == 1:
        bar_audio_stereo = np.stack([bar_audio, bar_audio], axis=-1)
    else:
        bar_audio_stereo = bar_audio
    
    try:
        lufs = meter.integrated_loudness(bar_audio_stereo)
    except ValueError:
        # 短すぎる音声の場合
        return -80.0
    
    return float(lufs)


# ========== Backend Dispatcher ==========

class FeaturesBackend:
    """
    バックエンド切替ディスパッチャー
    
    arranger_weights.yaml の features_backend セクションに基づき、
    適切なバックエンド関数を選択して呼び出す。
    """
    
    def __init__(self, config: Dict):
        """
        Args:
            config: arranger_weights.yaml['features_backend']
        """
        self.config = config
        self.beats_backend = config.get('beats', 'librosa')
        self.downbeats_backend = config.get('downbeats', 'none')
        self.hat_density_backend = config.get('hat_density', 'librosa')
        self.loudness_backend = config.get('loudness', 'rms')
        
        logger.info(f"FeaturesBackend initialized:")
        logger.info(f"  beats: {self.beats_backend}")
        logger.info(f"  downbeats: {self.downbeats_backend}")
        logger.info(f"  hat_density: {self.hat_density_backend}")
        logger.info(f"  loudness: {self.loudness_backend}")
    
    def extract_beats(self, audio_path: Path, audio: np.ndarray, sr: int) -> np.ndarray:
        """ビート抽出"""
        if self.beats_backend == 'madmom':
            return extract_beats_madmom(
                audio_path,
                fps=self.config.get('madmom', {}).get('fps', 100)
            )
        else:  # librosa
            return extract_beats_librosa(audio, sr)
    
    def extract_downbeats(
        self, 
        audio_path: Path, 
        audio: np.ndarray, 
        sr: int
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        ダウンビート抽出
        
        Returns:
            downbeat_times: (M,) ダウンビート時刻（秒）
            beat_positions: (M,) ビート位置（madmomのみ、librosaはNone）
        """
        if self.downbeats_backend == 'madmom':
            return extract_downbeats_madmom(
                audio_path,
                beats_per_bar=self.config.get('madmom', {}).get('beats_per_bar', [3, 4]),
                fps=self.config.get('madmom', {}).get('fps', 100)
            )
        elif self.downbeats_backend == 'librosa':
            beat_times = self.extract_beats(audio_path, audio, sr)
            # 4拍ごとにダウンビート
            downbeat_times = beat_times[::4]
            return downbeat_times, None
        else:  # none
            return np.array([]), None
    
    def extract_hat_density(
        self,
        audio_path: Path,
        audio: np.ndarray,
        sr: int,
        bar_start_sec: float,
        bar_end_sec: float
    ) -> float:
        """ハット密度抽出"""
        if self.hat_density_backend == 'yamnet':
            return extract_hat_density_yamnet(
                audio_path,
                bar_start_sec,
                bar_end_sec,
                threshold=self.config.get('yamnet', {}).get('threshold', 0.3),
                target_classes=self.config.get('yamnet', {}).get('target_classes', ["Hi-hat", "Cymbal"])
            )
        elif self.hat_density_backend == 'librosa_enhanced':
            enhanced_config = self.config.get('librosa_enhanced', {})
            return extract_hat_density_librosa_enhanced(
                audio,
                sr,
                bar_start_sec,
                bar_end_sec,
                bandpass_low=enhanced_config.get('bandpass_low', 5000.0),
                bandpass_high=enhanced_config.get('bandpass_high', 12000.0),
                onset_threshold=enhanced_config.get('onset_threshold', 0.6),
                aggregate_window=enhanced_config.get('aggregate_window', 0.1)
            )
        else:  # librosa
            return extract_hat_density_librosa(audio, sr, bar_start_sec, bar_end_sec)
    
    def extract_loudness(
        self,
        audio: np.ndarray,
        sr: int,
        bar_start_sec: float,
        bar_end_sec: float
    ) -> float:
        """ラウドネス抽出"""
        if self.loudness_backend == 'pyloudnorm':
            return extract_loudness_pyloudnorm(
                audio,
                sr,
                bar_start_sec,
                bar_end_sec,
                block_size=self.config.get('pyloudnorm', {}).get('block_size', 0.4)
            )
        else:  # rms
            return extract_loudness_rms(audio, sr, bar_start_sec, bar_end_sec)
