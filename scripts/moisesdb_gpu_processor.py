#!/usr/bin/env python3
"""
MoisesDB GPU Processor

GPU加速によるWAV処理（torchaudio + CUDA）

Features:
- GPU対応リサンプリング（torch.nn.functional.interpolate）
- バッチ処理による効率化
- 自動CPU/GPUフォールバック
- メモリ効率的なストリーミング処理

Requirements:
    pip install torch torchaudio

Usage:
    from scripts.moisesdb_gpu_processor import GPUWAVProcessor
    
    processor = GPUWAVProcessor(device='cuda', batch_size=16)
    resampled = processor.resample_batch(wav_tensors, target_sr=22050)
"""

import logging
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import numpy as np

try:
    import torch
    import torchaudio
    import torchaudio.transforms as T
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    torchaudio = None
    T = None

logger = logging.getLogger(__name__)


class GPUWAVProcessor:
    """
    GPU加速WAV処理クラス
    
    torchaudioを使用してGPU上でオーディオ処理を実行。
    CUDA未使用時は自動的にCPUにフォールバック。
    
    Attributes:
        device (str): 'cuda', 'mps', 'cpu'
        batch_size (int): バッチサイズ（GPU使用時）
        dtype (torch.dtype): データ型（デフォルト: float32）
    """
    
    def __init__(
        self,
        device: Optional[str] = None,
        batch_size: int = 16,
        dtype: torch.dtype = torch.float32
    ):
        """
        初期化
        
        Args:
            device: 'cuda', 'mps', 'cpu', または None（自動検出）
            batch_size: バッチサイズ（GPU使用時）
            dtype: Tensorデータ型
        """
        if not TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch/torchaudio not installed. "
                "Install with: pip install torch torchaudio"
            )
        
        # デバイス自動検出
        if device is None:
            if torch.cuda.is_available():
                device = 'cuda'
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = 'mps'
            else:
                device = 'cpu'
        
        self.device = torch.device(device)
        self.batch_size = batch_size
        self.dtype = dtype
        
        logger.info(f"GPUWAVProcessor initialized: device={self.device}, batch_size={batch_size}")
        
        # CUDA情報表示
        if self.device.type == 'cuda':
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            logger.info(f"GPU: {gpu_name} ({gpu_memory:.1f} GB)")
    
    def is_gpu_available(self) -> bool:
        """GPU利用可能か確認"""
        return self.device.type in ['cuda', 'mps']
    
    def get_device_info(self) -> Dict[str, Any]:
        """デバイス情報取得"""
        info = {
            'device': str(self.device),
            'type': self.device.type,
            'available': True
        }
        
        if self.device.type == 'cuda':
            info.update({
                'name': torch.cuda.get_device_name(0),
                'memory_total_gb': torch.cuda.get_device_properties(0).total_memory / 1e9,
                'memory_allocated_gb': torch.cuda.memory_allocated(0) / 1e9,
                'cuda_version': torch.version.cuda
            })
        
        return info
    
    def load_audio(
        self,
        file_path: Path,
        target_sr: Optional[int] = None,
        mono: bool = True
    ) -> Tuple[torch.Tensor, int]:
        """
        WAVファイル読み込み（GPU対応）
        
        Args:
            file_path: WAVファイルパス
            target_sr: リサンプリング先サンプルレート（Noneなら元のまま）
            mono: モノラル変換するか
        
        Returns:
            (waveform, sample_rate): Tensor (C, T), サンプルレート
        """
        # CPU上でロード
        waveform, sr = torchaudio.load(str(file_path))
        
        # モノラル変換
        if mono and waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        # リサンプリング
        if target_sr is not None and sr != target_sr:
            resampler = T.Resample(sr, target_sr).to(self.device)
            waveform = waveform.to(self.device)
            waveform = resampler(waveform)
            sr = target_sr
        else:
            waveform = waveform.to(self.device)
        
        return waveform, sr
    
    def save_audio(
        self,
        file_path: Path,
        waveform: torch.Tensor,
        sample_rate: int
    ):
        """
        WAVファイル保存
        
        Args:
            file_path: 保存先パス
            waveform: Tensor (C, T)
            sample_rate: サンプルレート
        """
        # CPUに移動して保存
        waveform_cpu = waveform.cpu()
        file_path.parent.mkdir(parents=True, exist_ok=True)
        torchaudio.save(str(file_path), waveform_cpu, sample_rate)
    
    def resample(
        self,
        waveform: torch.Tensor,
        orig_sr: int,
        target_sr: int
    ) -> torch.Tensor:
        """
        リサンプリング（GPU加速）
        
        Args:
            waveform: Tensor (C, T)
            orig_sr: 元のサンプルレート
            target_sr: 目標サンプルレート
        
        Returns:
            Resampled Tensor (C, T')
        """
        if orig_sr == target_sr:
            return waveform
        
        resampler = T.Resample(orig_sr, target_sr).to(self.device)
        waveform = waveform.to(self.device)
        return resampler(waveform)
    
    def resample_batch(
        self,
        waveforms: List[torch.Tensor],
        orig_sr: int,
        target_sr: int
    ) -> List[torch.Tensor]:
        """
        バッチリサンプリング（GPU加速）
        
        Args:
            waveforms: List of Tensors [(C, T), ...]
            orig_sr: 元のサンプルレート
            target_sr: 目標サンプルレート
        
        Returns:
            List of resampled Tensors
        """
        if orig_sr == target_sr:
            return waveforms
        
        resampler = T.Resample(orig_sr, target_sr).to(self.device)
        resampled = []
        
        for waveform in waveforms:
            waveform = waveform.to(self.device)
            resampled.append(resampler(waveform))
        
        return resampled
    
    def concatenate_segments(
        self,
        segments: List[torch.Tensor],
        sample_rate: int
    ) -> torch.Tensor:
        """
        セグメント結合（GPU上で実行）
        
        Args:
            segments: List of Tensors [(C, T), ...]
            sample_rate: サンプルレート
        
        Returns:
            結合されたTensor (C, T_total)
        """
        # すべてのセグメントをGPUに転送
        segments_gpu = [seg.to(self.device) for seg in segments]
        
        # 結合（GPU上で実行）
        concatenated = torch.cat(segments_gpu, dim=1)
        
        return concatenated
    
    def compute_spectrogram(
        self,
        waveform: torch.Tensor,
        sample_rate: int,
        n_fft: int = 2048,
        hop_length: int = 512,
        n_mels: Optional[int] = None
    ) -> torch.Tensor:
        """
        スペクトログラム計算（GPU加速）
        
        Args:
            waveform: Tensor (C, T)
            sample_rate: サンプルレート
            n_fft: FFTサイズ
            hop_length: ホップ長
            n_mels: Mel bins（Noneなら通常のSTFT）
        
        Returns:
            Spectrogram Tensor (C, F, T) or MelSpectrogram (C, n_mels, T)
        """
        waveform = waveform.to(self.device)
        
        if n_mels is not None:
            # Mel Spectrogram
            transform = T.MelSpectrogram(
                sample_rate=sample_rate,
                n_fft=n_fft,
                hop_length=hop_length,
                n_mels=n_mels
            ).to(self.device)
        else:
            # Regular Spectrogram
            transform = T.Spectrogram(
                n_fft=n_fft,
                hop_length=hop_length,
                power=2.0
            ).to(self.device)
        
        return transform(waveform)
    
    def compute_chromagram(
        self,
        waveform: torch.Tensor,
        sample_rate: int,
        n_fft: int = 2048,
        hop_length: int = 512,
        n_chroma: int = 12
    ) -> torch.Tensor:
        """
        クロマグラム計算（GPU加速）
        
        Args:
            waveform: Tensor (C, T)
            sample_rate: サンプルレート
            n_fft: FFTサイズ
            hop_length: ホップ長
            n_chroma: クロマビン数（通常12）
        
        Returns:
            Chromagram Tensor (C, n_chroma, T)
        """
        # まずMel Spectrogram計算
        mel_spec = self.compute_spectrogram(
            waveform, sample_rate, n_fft, hop_length, n_mels=128
        )
        
        # クロマグラムに変換（簡易実装: Mel → Chroma マッピング）
        # 注: librosa.feature.chromaに相当する完全実装は複雑
        # ここでは周波数ビンをクロマに折りたたむ
        chroma = self._mel_to_chroma(mel_spec, n_chroma)
        
        return chroma
    
    def _mel_to_chroma(
        self,
        mel_spec: torch.Tensor,
        n_chroma: int = 12
    ) -> torch.Tensor:
        """
        Mel Spectrogram → Chromagram 変換（簡易版）
        
        Args:
            mel_spec: Tensor (C, n_mels, T)
            n_chroma: クロマビン数
        
        Returns:
            Chroma Tensor (C, n_chroma, T)
        """
        # Mel binsをクロマビンに折りたたむ
        # 簡易実装: 128 mels → 12 chromaにグループ化
        n_mels = mel_spec.shape[1]
        bins_per_chroma = n_mels // n_chroma
        
        chroma = []
        for i in range(n_chroma):
            start = i * bins_per_chroma
            end = start + bins_per_chroma
            chroma.append(mel_spec[:, start:end, :].sum(dim=1, keepdim=True))
        
        chroma_tensor = torch.cat(chroma, dim=1)
        
        # 正規化
        chroma_tensor = chroma_tensor / (chroma_tensor.sum(dim=1, keepdim=True) + 1e-8)
        
        return chroma_tensor
    
    def extract_harmonic_percussive(
        self,
        waveform: torch.Tensor,
        sample_rate: int,
        n_fft: int = 2048,
        hop_length: int = 512,
        margin: float = 2.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Harmonic-Percussive Source Separation（GPU加速）
        
        Args:
            waveform: Tensor (C, T)
            sample_rate: サンプルレート
            n_fft: FFTサイズ
            hop_length: ホップ長
            margin: マージンパラメータ
        
        Returns:
            (harmonic, percussive): 両方ともTensor (C, T)
        """
        # STFT計算
        waveform = waveform.to(self.device)
        stft_transform = T.Spectrogram(
            n_fft=n_fft,
            hop_length=hop_length,
            power=None  # 複素数STFT
        ).to(self.device)
        
        stft = stft_transform(waveform)  # (C, F, T) complex
        
        # パワースペクトログラム
        power_spec = torch.abs(stft) ** 2
        
        # Median filtering（簡易版）
        # 横方向（時間）: harmonic
        # 縦方向（周波数）: percussive
        harmonic_mask = self._median_filter_2d(power_spec, kernel_size=(1, 17))
        percussive_mask = self._median_filter_2d(power_spec, kernel_size=(17, 1))
        
        # Mask適用
        harmonic_stft = stft * (harmonic_mask / (harmonic_mask + percussive_mask + 1e-8))
        percussive_stft = stft * (percussive_mask / (harmonic_mask + percussive_mask + 1e-8))
        
        # ISTFT
        istft_transform = T.InverseSpectrogram(
            n_fft=n_fft,
            hop_length=hop_length
        ).to(self.device)
        
        harmonic = istft_transform(harmonic_stft)
        percussive = istft_transform(percussive_stft)
        
        return harmonic, percussive
    
    def _median_filter_2d(
        self,
        tensor: torch.Tensor,
        kernel_size: Tuple[int, int]
    ) -> torch.Tensor:
        """
        2D Median Filter（簡易実装）
        
        Args:
            tensor: Tensor (C, F, T)
            kernel_size: (freq_kernel, time_kernel)
        
        Returns:
            Filtered Tensor (C, F, T)
        """
        # PyTorchにMedianFilterがないため、MaxPoolで代用（近似）
        # 本格実装はscipy.ndimage.median_filterと同等の処理が必要
        
        # パディング
        pad_f = kernel_size[0] // 2
        pad_t = kernel_size[1] // 2
        
        padded = torch.nn.functional.pad(
            tensor,
            (pad_t, pad_t, pad_f, pad_f),
            mode='reflect'
        )
        
        # MaxPoolで近似（中央値の代わり）
        filtered = torch.nn.functional.max_pool2d(
            padded,
            kernel_size=kernel_size,
            stride=1
        )
        
        return filtered
    
    def process_segment_batch(
        self,
        segment_paths: List[Path],
        target_sr: int = 22050
    ) -> torch.Tensor:
        """
        セグメントバッチ処理（GPU加速）
        
        Args:
            segment_paths: セグメントファイルパスリスト
            target_sr: 目標サンプルレート
        
        Returns:
            結合されたTensor (1, T)
        """
        segments = []
        
        for path in segment_paths:
            waveform, sr = self.load_audio(path, target_sr=target_sr, mono=True)
            segments.append(waveform)
        
        # GPU上で結合
        concatenated = self.concatenate_segments(segments, target_sr)
        
        return concatenated
    
    def estimate_stem_quality(
        self,
        waveform: torch.Tensor,
        sample_rate: int
    ) -> Dict[str, float]:
        """
        ステム品質推定（GPU加速）
        
        Args:
            waveform: Tensor (C, T)
            sample_rate: サンプルレート
        
        Returns:
            {
                'high_freq_ratio': float,
                'harmonic_persistence': float,
                'percussive_ratio': float
            }
        """
        # Mel Spectrogram
        mel_spec = self.compute_spectrogram(
            waveform, sample_rate, n_mels=128
        )  # (C, n_mels, T)
        
        # 高周波成分比率（上位50% vs 下位50%）
        mid = mel_spec.shape[1] // 2
        high_energy = mel_spec[:, mid:, :].sum()
        low_energy = mel_spec[:, :mid, :].sum()
        high_freq_ratio = (high_energy / (high_energy + low_energy + 1e-8)).item()
        
        # Harmonic-Percussive分離
        harmonic, percussive = self.extract_harmonic_percussive(waveform, sample_rate)
        
        harmonic_energy = (harmonic ** 2).sum()
        percussive_energy = (percussive ** 2).sum()
        total_energy = harmonic_energy + percussive_energy + 1e-8
        
        harmonic_persistence = (harmonic_energy / total_energy).item()
        percussive_ratio = (percussive_energy / total_energy).item()
        
        return {
            'high_freq_ratio': high_freq_ratio,
            'harmonic_persistence': harmonic_persistence,
            'percussive_ratio': percussive_ratio
        }
    
    def clear_cache(self):
        """GPUキャッシュクリア"""
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
            logger.info("CUDA cache cleared")


def check_gpu_availability() -> Dict[str, Any]:
    """
    GPU利用可能性チェック
    
    Returns:
        {
            'cuda_available': bool,
            'mps_available': bool,
            'cuda_device_count': int,
            'recommended_device': str
        }
    """
    if not TORCH_AVAILABLE:
        return {
            'cuda_available': False,
            'mps_available': False,
            'cuda_device_count': 0,
            'recommended_device': 'cpu',
            'error': 'PyTorch not installed'
        }
    
    info = {
        'cuda_available': torch.cuda.is_available(),
        'mps_available': hasattr(torch.backends, 'mps') and torch.backends.mps.is_available(),
        'cuda_device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0
    }
    
    # 推奨デバイス
    if info['cuda_available']:
        info['recommended_device'] = 'cuda'
        info['cuda_devices'] = [
            torch.cuda.get_device_name(i)
            for i in range(info['cuda_device_count'])
        ]
    elif info['mps_available']:
        info['recommended_device'] = 'mps'
    else:
        info['recommended_device'] = 'cpu'
    
    return info


if __name__ == '__main__':
    # GPU可用性チェック
    print("=== GPU Availability Check ===")
    gpu_info = check_gpu_availability()
    for key, value in gpu_info.items():
        print(f"{key}: {value}")
    
    if gpu_info['recommended_device'] == 'cpu':
        print("\n⚠️  GPU not available. CPU mode only.")
        exit(0)
    
    # テスト実行
    print(f"\n=== Testing GPUWAVProcessor (device={gpu_info['recommended_device']}) ===")
    processor = GPUWAVProcessor(device=gpu_info['recommended_device'])
    
    device_info = processor.get_device_info()
    print("\nDevice Info:")
    for key, value in device_info.items():
        print(f"  {key}: {value}")
    
    print("\n✅ GPUWAVProcessor initialized successfully!")
