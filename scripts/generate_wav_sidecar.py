#!/usr/bin/env python3
"""
WAV Sidecar JSON Generator - メタデータ拡張

既存のWAVデータセットにsidecar JSONを追加します。
RhythmAI / 和声AI / EmotionAI 共用のメタデータを生成します。

Features:
- Pre/Post LUFS計算
- Crest Factor計算
- Pre/Post MD5ハッシュ
- トリムオフセット記録
- ツールバージョン記録

Usage:
    python scripts/generate_wav_sidecar.py \
        --input output/wav_cleaned/moisesdb \
        --config configs/wav_stage1.yaml \
        --jobs 8
"""

import argparse
import hashlib
import json
import logging
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import yaml
from tqdm import tqdm

try:
    import librosa
    import soundfile as sf
    import pyloudnorm as pyln
except ImportError:
    print("❌ Error: Required packages not installed.")
    print("   Install: pip install librosa soundfile pyloudnorm")
    sys.exit(1)


def get_git_sha() -> str:
    """Git SHA取得"""
    try:
        result = subprocess.run(
            ['git', 'rev-parse', '--short', 'HEAD'],
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout.strip()
    except:
        return "unknown"


def compute_md5_pcm(audio: np.ndarray, sr: int) -> str:
    """PCMデータのMD5ハッシュ計算"""
    # モノラル化 + 22.05kHzダウンサンプル（ID安定性向上）
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    
    if sr != 22050:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=22050)
    
    # int16変換
    audio_int16 = (audio * 32767).astype(np.int16)
    
    return hashlib.md5(audio_int16.tobytes()).hexdigest()


def compute_lufs(audio: np.ndarray, sr: int) -> float:
    """LUFS（Integrated Loudness）計算"""
    try:
        meter = pyln.Meter(sr)
        
        # ステレオ化（pyloudnormはステレオ必須）
        if audio.ndim == 1:
            audio = np.stack([audio, audio], axis=-1)
        
        loudness = meter.integrated_loudness(audio)
        
        return loudness
    except:
        return -70.0  # エラー時のデフォルト


def compute_crest_factor(audio: np.ndarray) -> float:
    """Crest Factor計算（dB）"""
    peak = np.max(np.abs(audio))
    rms = np.sqrt(np.mean(audio ** 2))
    
    if rms < 1e-9:
        return 0.0
    
    crest_linear = peak / rms
    crest_db = 20 * np.log10(crest_linear)
    
    return crest_db


def generate_sidecar(song_dir: Path, config: Dict[str, Any]) -> Dict[str, Any]:
    """Sidecar JSON生成"""
    song_id = song_dir.name
    
    # audio_chordmap.yaml読み込み
    chordmap_path = song_dir / "audio_chordmap.yaml"
    
    if not chordmap_path.exists():
        return {
            'song_id': song_id,
            'status': 'skip',
            'reason': 'no_chordmap'
        }
    
    with open(chordmap_path, 'r', encoding='utf-8') as f:
        chordmap_data = yaml.safe_load(f)
    
    # WAVファイル探索（想定: song_id.wav）
    wav_candidates = list(song_dir.glob("*.wav")) + list(song_dir.glob("*.mp3"))
    
    if not wav_candidates:
        return {
            'song_id': song_id,
            'status': 'skip',
            'reason': 'no_audio_file'
        }
    
    audio_path = wav_candidates[0]
    
    # オーディオ読み込み
    try:
        audio, sr = librosa.load(audio_path, sr=None, mono=False)
    except Exception as e:
        return {
            'song_id': song_id,
            'status': 'error',
            'reason': f'load_error: {e}'
        }
    
    # ステレオ→(samples, channels)へ変換
    if audio.ndim == 1:
        audio_analysis = audio
        channels = 1
    else:
        audio_analysis = audio.T  # (channels, samples) → (samples, channels)
        channels = audio.shape[0]
    
    # Pre-clean メトリクス
    md5_pre = compute_md5_pcm(audio_analysis if audio_analysis.ndim == 1 else audio_analysis[:, 0], sr)
    lufs_pre = compute_lufs(audio_analysis, sr)
    peak_pre = float(np.max(np.abs(audio_analysis)))
    peak_pre_dbfs = 20 * np.log10(peak_pre) if peak_pre > 0 else -np.inf
    crest_pre = compute_crest_factor(audio_analysis if audio_analysis.ndim == 1 else audio_analysis[:, 0])
    
    # Post-clean（正規化シミュレーション）
    peak_target = config.get('peak_target', 0.98)
    
    if peak_pre > 0:
        gain_linear = peak_target / peak_pre
        gain_db = 20 * np.log10(gain_linear)
        audio_normalized = audio_analysis * gain_linear
    else:
        gain_db = 0.0
        audio_normalized = audio_analysis
    
    md5_post = compute_md5_pcm(audio_normalized if audio_normalized.ndim == 1 else audio_normalized[:, 0], sr)
    lufs_post = compute_lufs(audio_normalized, sr)
    peak_post = float(np.max(np.abs(audio_normalized)))
    peak_post_dbfs = 20 * np.log10(peak_post) if peak_post > 0 else -np.inf
    
    # Sidecar JSON構築
    sidecar = {
        'version': config.get('metadata', {}).get('version', '1.0.0'),
        'pipeline': 'wav_stage1',
        'git_sha': get_git_sha(),
        'song_id': song_id,
        'source_audio': str(audio_path),
        
        # Audio Properties
        'audio': {
            'sr_in': int(sr),
            'sr_out': int(sr),
            'bitdepth_in': 16,  # 推定
            'bitdepth_out': 16,
            'channels_in': channels,
            'channels_out': channels,
            'duration_s': float(len(audio_analysis) / sr)
        },
        
        # IDs
        'ids': {
            'audio_src_id': md5_pre[:16],
            'audio_clean_id': md5_post[:16],
            'md5_pcm_pre': md5_pre,
            'md5_pcm_post': md5_post
        },
        
        # Normalization
        'normalization': {
            'pre_norm_peak_dbfs': float(peak_pre_dbfs),
            'post_norm_peak_dbfs': float(peak_post_dbfs),
            'gain_applied_db': float(gain_db),
            'peak_target': float(peak_target)
        },
        
        # Loudness
        'loudness': {
            'lufs_integrated_pre': float(lufs_pre),
            'lufs_integrated_post': float(lufs_post),
            'crest_factor_db_pre': float(crest_pre)
        },
        
        # Trimming（現状は未実装）
        'trimming': {
            'trim_head_ms': 0.0,
            'trim_tail_ms': 0.0,
            'silence_trim_enabled': config.get('trim_silence', False),
            'silence_margin_ms': config.get('silence_margin_ms', 50)
        },
        
        # Tool Versions
        'tool_versions': {
            'librosa': librosa.__version__,
            'soundfile': sf.__version__,
            'pyloudnorm': pyln.__version__,
            'python': sys.version.split()[0]
        }
    }
    
    # 保存
    sidecar_path = song_dir / "audio_metadata.json"
    
    with open(sidecar_path, 'w', encoding='utf-8') as f:
        json.dump(sidecar, f, indent=2, ensure_ascii=False)
    
    return {
        'song_id': song_id,
        'status': 'success',
        'sidecar_path': str(sidecar_path)
    }


def main():
    parser = argparse.ArgumentParser(
        description="WAV Sidecar JSON Generator"
    )
    parser.add_argument(
        '--input',
        type=Path,
        required=True,
        help='Input directory (e.g., output/wav_cleaned/moisesdb)'
    )
    parser.add_argument(
        '--config',
        type=Path,
        default=Path('configs/wav_stage1.yaml'),
        help='WAV Stage1 config file'
    )
    parser.add_argument(
        '--jobs',
        type=int,
        default=8,
        help='Parallel workers'
    )
    
    args = parser.parse_args()
    
    # Config読み込み
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 楽曲ディレクトリ収集
    song_dirs = sorted([d for d in args.input.iterdir() if d.is_dir()])
    
    print(f"\n{'='*70}")
    print(f"WAV Sidecar JSON Generation")
    print(f"{'='*70}")
    print(f"Input: {args.input}")
    print(f"Total songs: {len(song_dirs)}")
    print(f"{'='*70}\n")
    
    # 並列処理
    results = []
    
    with ProcessPoolExecutor(max_workers=args.jobs) as executor:
        futures = {
            executor.submit(generate_sidecar, song_dir, config): song_dir
            for song_dir in song_dirs
        }
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Generating"):
            result = future.result()
            results.append(result)
    
    # サマリー
    success = [r for r in results if r['status'] == 'success']
    skip = [r for r in results if r['status'] == 'skip']
    error = [r for r in results if r['status'] == 'error']
    
    print(f"\n{'='*70}")
    print(f"Summary")
    print(f"{'='*70}")
    print(f"Total:   {len(results)}")
    print(f"Success: {len(success)}")
    print(f"Skip:    {len(skip)}")
    print(f"Error:   {len(error)}")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
