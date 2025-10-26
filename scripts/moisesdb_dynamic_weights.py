#!/usr/bin/env python3
"""
MoisesDB Dynamic Weight Adjustment

ハーモニック系ステムの品質を分析し、重みを動的に調整。

Features:
- スペクトル分析によるステム品質評価
- 重み動的調整（0.0-1.0）
- audio_chordmap.yaml生成時の最適化
- 低品質ステムの自動ダウンウェイト

Quality Metrics:
- harmonic_persistence: 和音成分の持続性（高い = piano/guitar）
- high_freq_ratio: 高周波成分比率（高い = crisp）
- percussive_ratio: パーカッシブ成分（低い = harmonic）

Usage:
    from scripts.moisesdb_dynamic_weights import DynamicWeightAdjuster
    
    adjuster = DynamicWeightAdjuster()
    
    # ステム品質分析
    qualities = adjuster.analyze_stems_quality(stem_paths)
    
    # 重み調整
    adjusted_weights = adjuster.adjust_weights(
        stem_types=['piano', 'guitar', 'bass'],
        qualities=qualities
    )
"""

import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False

logger = logging.getLogger(__name__)


# デフォルト重み（静的）
DEFAULT_STEM_WEIGHTS = {
    'piano': 0.40,
    'keys': 0.40,
    'guitar': 0.35,
    'bass': 0.20,
    'strings': 0.30,
    'synth': 0.25,
    'brass': 0.25,
    'pad': 0.20,
    'other': 0.10,
    'drums': 0.0,
    'percussion': 0.0,
    'vocals': 0.0
}

# 品質スコア重み（動的調整用）
QUALITY_SCORE_WEIGHTS = {
    'harmonic_persistence': 0.50,  # 最重要
    'high_freq_ratio': 0.30,
    'low_percussive': 0.20
}


class DynamicWeightAdjuster:
    """
    動的重み調整クラス
    
    スペクトル分析でステム品質を評価し、重みを動的に調整。
    低品質ステムの重みを下げ、高品質ステムの重みを上げる。
    """
    
    def __init__(
        self,
        sr: int = 22050,
        use_gpu: bool = False,
        quality_threshold: float = 0.4
    ):
        """
        初期化
        
        Args:
            sr: サンプルレート
            use_gpu: GPU使用フラグ
            quality_threshold: 品質閾値（これ未満は重みを大幅減少）
        """
        self.sr = sr
        self.use_gpu = use_gpu
        self.quality_threshold = quality_threshold
        
        # GPU対応
        if use_gpu:
            try:
                from scripts.moisesdb_gpu_processor import GPUWAVProcessor
                self.gpu_processor = GPUWAVProcessor(device=None)
                logger.info("GPU processor initialized for dynamic weights")
            except ImportError:
                logger.warning("GPU requested but not available, using CPU")
                self.use_gpu = False
                self.gpu_processor = None
        else:
            self.gpu_processor = None
    
    def analyze_stem_quality(
        self,
        wav_path: Path
    ) -> Dict[str, float]:
        """
        単一ステムの品質分析
        
        Args:
            wav_path: WAVファイルパス
        
        Returns:
            {
                'harmonic_persistence': float (0-1),
                'high_freq_ratio': float (0-1),
                'percussive_ratio': float (0-1),
                'quality_score': float (0-1)
            }
        """
        if self.use_gpu and self.gpu_processor:
            return self._analyze_quality_gpu(wav_path)
        else:
            return self._analyze_quality_cpu(wav_path)
    
    def _analyze_quality_cpu(
        self,
        wav_path: Path
    ) -> Dict[str, float]:
        """CPU版品質分析"""
        if not LIBROSA_AVAILABLE:
            logger.warning("librosa not available, returning default quality")
            return {
                'harmonic_persistence': 0.5,
                'high_freq_ratio': 0.5,
                'percussive_ratio': 0.5,
                'quality_score': 0.5
            }
        
        # WAV読み込み
        y, sr = librosa.load(str(wav_path), sr=self.sr, mono=True)
        
        # 1. Harmonic-Percussive分離
        y_harmonic, y_percussive = librosa.effects.hpss(y)
        
        harmonic_energy = np.sum(y_harmonic ** 2)
        percussive_energy = np.sum(y_percussive ** 2)
        total_energy = harmonic_energy + percussive_energy + 1e-8
        
        harmonic_persistence = harmonic_energy / total_energy
        percussive_ratio = percussive_energy / total_energy
        
        # 2. 周波数スペクトル分析
        mel_spec = librosa.feature.melspectrogram(
            y=y, sr=sr, n_mels=128, hop_length=512
        )
        
        # 高周波成分（上位50% vs 下位50%）
        mid_freq = mel_spec.shape[0] // 2
        high_energy = np.sum(mel_spec[mid_freq:, :])
        low_energy = np.sum(mel_spec[:mid_freq, :])
        
        high_freq_ratio = high_energy / (high_energy + low_energy + 1e-8)
        
        # 3. 品質スコア計算
        quality_score = (
            harmonic_persistence * QUALITY_SCORE_WEIGHTS['harmonic_persistence'] +
            high_freq_ratio * QUALITY_SCORE_WEIGHTS['high_freq_ratio'] +
            (1.0 - percussive_ratio) * QUALITY_SCORE_WEIGHTS['low_percussive']
        )
        
        return {
            'harmonic_persistence': float(harmonic_persistence),
            'high_freq_ratio': float(high_freq_ratio),
            'percussive_ratio': float(percussive_ratio),
            'quality_score': float(quality_score)
        }
    
    def _analyze_quality_gpu(
        self,
        wav_path: Path
    ) -> Dict[str, float]:
        """GPU版品質分析"""
        # GPU processorのestimate_stem_quality使用
        waveform, sr = self.gpu_processor.load_audio(
            wav_path,
            target_sr=self.sr,
            mono=True
        )
        
        quality_metrics = self.gpu_processor.estimate_stem_quality(waveform, sr)
        
        # 品質スコア計算
        quality_score = (
            quality_metrics['harmonic_persistence'] * 
            QUALITY_SCORE_WEIGHTS['harmonic_persistence'] +
            quality_metrics['high_freq_ratio'] * 
            QUALITY_SCORE_WEIGHTS['high_freq_ratio'] +
            (1.0 - quality_metrics['percussive_ratio']) * 
            QUALITY_SCORE_WEIGHTS['low_percussive']
        )
        
        return {
            'harmonic_persistence': quality_metrics['harmonic_persistence'],
            'high_freq_ratio': quality_metrics['high_freq_ratio'],
            'percussive_ratio': quality_metrics['percussive_ratio'],
            'quality_score': quality_score
        }
    
    def analyze_stems_quality(
        self,
        stem_paths: Dict[str, Path]
    ) -> Dict[str, Dict[str, float]]:
        """
        複数ステムの品質分析
        
        Args:
            stem_paths: {'piano': Path(...), 'guitar': Path(...), ...}
        
        Returns:
            {
                'piano': {'quality_score': 0.85, ...},
                'guitar': {'quality_score': 0.72, ...},
                ...
            }
        """
        qualities = {}
        
        for stem_type, wav_path in stem_paths.items():
            logger.info(f"Analyzing quality: {stem_type}")
            qualities[stem_type] = self.analyze_stem_quality(wav_path)
        
        return qualities
    
    def adjust_weights(
        self,
        stem_types: List[str],
        qualities: Dict[str, Dict[str, float]],
        base_weights: Optional[Dict[str, float]] = None
    ) -> Dict[str, float]:
        """
        品質スコアに基づき重みを動的調整
        
        Args:
            stem_types: ['piano', 'guitar', 'bass']
            qualities: analyze_stems_quality()の出力
            base_weights: ベース重み（Noneならデフォルト使用）
        
        Returns:
            {'piano': 0.45, 'guitar': 0.35, 'bass': 0.20} (合計1.0に正規化)
        """
        if base_weights is None:
            base_weights = DEFAULT_STEM_WEIGHTS
        
        adjusted = {}
        
        for stem_type in stem_types:
            base_weight = base_weights.get(stem_type, 0.1)
            
            if stem_type in qualities:
                quality_score = qualities[stem_type]['quality_score']
                
                # 品質スコアによる調整係数
                if quality_score < self.quality_threshold:
                    # 低品質 → 重みを大幅減少
                    adjustment = quality_score / self.quality_threshold * 0.5
                elif quality_score > 0.7:
                    # 高品質 → 重みを増加
                    adjustment = 1.0 + (quality_score - 0.7) * 0.5
                else:
                    # 中品質 → そのまま
                    adjustment = 1.0
                
                adjusted[stem_type] = base_weight * adjustment
            else:
                # 品質情報なし → ベース重みそのまま
                adjusted[stem_type] = base_weight
        
        # 正規化（合計1.0）
        total = sum(adjusted.values())
        if total > 0:
            adjusted = {k: v / total for k, v in adjusted.items()}
        
        return adjusted
    
    def generate_weighted_chordmap(
        self,
        stem_paths: Dict[str, Path],
        output_yaml: Path,
        stem_roles: Optional[Dict[str, str]] = None
    ) -> Dict[str, float]:
        """
        動的重み付きaudio_chordmap.yaml生成
        
        Args:
            stem_paths: {'piano': Path('merged_piano.wav'), ...}
            output_yaml: 出力YAMLパス
            stem_roles: ステムロール（Noneなら自動判定）
        
        Returns:
            調整後の重み {'piano': 0.45, ...}
        """
        # 1. 品質分析
        qualities = self.analyze_stems_quality(stem_paths)
        
        # 2. 重み調整
        stem_types = list(stem_paths.keys())
        adjusted_weights = self.adjust_weights(stem_types, qualities)
        
        # 3. ロール判定（未指定の場合）
        if stem_roles is None:
            stem_roles = self._auto_detect_roles(stem_types, qualities)
        
        # 4. YAML生成
        self._write_yaml(
            output_yaml,
            stem_paths,
            adjusted_weights,
            stem_roles,
            qualities
        )
        
        logger.info(f"Generated weighted chordmap: {output_yaml}")
        logger.info(f"Adjusted weights: {adjusted_weights}")
        
        return adjusted_weights
    
    def _auto_detect_roles(
        self,
        stem_types: List[str],
        qualities: Dict[str, Dict[str, float]]
    ) -> Dict[str, str]:
        """
        ステムロール自動判定
        
        Args:
            stem_types: ['piano', 'guitar', 'bass']
            qualities: 品質分析結果
        
        Returns:
            {'piano': 'harmonic', 'guitar': 'harmonic', 'bass': 'bass'}
        """
        roles = {}
        
        for stem_type in stem_types:
            if stem_type == 'bass':
                roles[stem_type] = 'bass'
            elif stem_type == 'drums':
                roles[stem_type] = 'drums'
            elif stem_type in ['piano', 'keys', 'guitar', 'strings', 'synth']:
                roles[stem_type] = 'harmonic'
            else:
                # 品質スコアで判定
                if stem_type in qualities:
                    quality = qualities[stem_type]
                    if quality['percussive_ratio'] > 0.5:
                        roles[stem_type] = 'drums'
                    elif quality['harmonic_persistence'] > 0.5:
                        roles[stem_type] = 'harmonic'
                    else:
                        roles[stem_type] = 'other'
                else:
                    roles[stem_type] = 'other'
        
        return roles
    
    def _write_yaml(
        self,
        output_yaml: Path,
        stem_paths: Dict[str, Path],
        weights: Dict[str, float],
        roles: Dict[str, str],
        qualities: Dict[str, Dict[str, float]]
    ):
        """
        audio_chordmap.yaml書き込み
        
        Args:
            output_yaml: 出力パス
            stem_paths: ステムファイルパス
            weights: 調整後重み
            roles: ステムロール
            qualities: 品質情報
        """
        output_yaml.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_yaml, 'w') as f:
            f.write("# Auto-generated audio_chordmap.yaml with dynamic weights\n")
            f.write("# Generated by: moisesdb_dynamic_weights.py\n\n")
            
            f.write("stems:\n")
            
            for stem_type in sorted(stem_paths.keys()):
                wav_path = stem_paths[stem_type]
                weight = weights.get(stem_type, 0.0)
                role = roles.get(stem_type, 'other')
                
                f.write(f"  - name: {stem_type}\n")
                f.write(f"    file: {wav_path.name}\n")
                f.write(f"    weight: {weight:.4f}\n")
                f.write(f"    role: {role}\n")
                
                # 品質情報（コメント）
                if stem_type in qualities:
                    q = qualities[stem_type]
                    f.write(f"    # quality_score: {q['quality_score']:.3f}\n")
                    f.write(f"    # harmonic: {q['harmonic_persistence']:.3f}\n")
                    f.write(f"    # high_freq: {q['high_freq_ratio']:.3f}\n")
                
                f.write("\n")
            
            # 投票設定
            f.write("voting:\n")
            f.write("  method: weighted  # 重み付き投票\n")
            f.write("  normalize: true   # 重み正規化済み\n")
            f.write("  min_agreement: 0.5  # 最小合意率\n")


def adjust_weights_simple(
    stem_types: List[str],
    stem_paths: Dict[str, Path],
    sr: int = 22050,
    use_gpu: bool = False
) -> Dict[str, float]:
    """
    簡易版動的重み調整（関数インターフェース）
    
    Args:
        stem_types: ['piano', 'guitar', 'bass']
        stem_paths: {'piano': Path(...), 'guitar': Path(...)}
        sr: サンプルレート
        use_gpu: GPU使用フラグ
    
    Returns:
        {'piano': 0.45, 'guitar': 0.35, 'bass': 0.20}
    """
    adjuster = DynamicWeightAdjuster(sr=sr, use_gpu=use_gpu)
    
    # 品質分析
    qualities = adjuster.analyze_stems_quality(stem_paths)
    
    # 重み調整
    adjusted_weights = adjuster.adjust_weights(stem_types, qualities)
    
    return adjusted_weights


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description="Dynamic Weight Adjustment")
    parser.add_argument(
        '--stem-dir',
        type=Path,
        required=True,
        help='Directory containing stem WAV files'
    )
    parser.add_argument(
        '--output-yaml',
        type=Path,
        default=Path('audio_chordmap.yaml'),
        help='Output YAML path'
    )
    parser.add_argument(
        '--use-gpu',
        action='store_true',
        help='Use GPU acceleration'
    )
    parser.add_argument(
        '--sr',
        type=int,
        default=22050,
        help='Sample rate'
    )
    
    args = parser.parse_args()
    
    # ステムファイル収集
    stem_paths = {}
    for wav_file in args.stem_dir.glob('*.wav'):
        stem_name = wav_file.stem.split('_')[-1]  # song_001_piano.wav → piano
        stem_paths[stem_name] = wav_file
    
    print(f"Found {len(stem_paths)} stems: {list(stem_paths.keys())}")
    
    # 動的重み調整
    adjuster = DynamicWeightAdjuster(sr=args.sr, use_gpu=args.use_gpu)
    adjusted_weights = adjuster.generate_weighted_chordmap(
        stem_paths=stem_paths,
        output_yaml=args.output_yaml
    )
    
    print("\n✅ Adjusted weights:")
    for stem, weight in sorted(adjusted_weights.items(), key=lambda x: -x[1]):
        print(f"  {stem:12s}: {weight:.4f}")
