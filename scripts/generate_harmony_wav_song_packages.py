#!/usr/bin/env python3
"""
Generate WAV Harmony Song Packages

MoisesDB/MUSDB18のWAVデータからsong_package.yamlを生成します。

Usage:
    python scripts/generate_harmony_wav_song_packages.py \
        --input output/wav_cleaned/moisesdb \
        --output output/wav_harmony_packages/moisesdb \
        --dataset moisesdb
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import yaml
from tqdm import tqdm


def setup_logging():
    """ロギング設定"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def load_audio_chordmap(chordmap_path: Path) -> Dict[str, Any]:
    """audio_chordmap.yaml読み込み"""
    with open(chordmap_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def load_beat_grid(beat_grid_path: Path) -> Dict[str, Any]:
    """beat_grid.json読み込み"""
    with open(beat_grid_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def generate_bars_from_beat_times(beat_times: List[float], tempo_bpm: float = 120.0) -> pd.DataFrame:
    """
    beat_timesから小節データ生成
    
    beat_times構造: [0.0, 0.5, 1.0, 1.5, 2.0, ...]（秒単位のビート時刻配列）
    """
    if not beat_times or len(beat_times) == 0:
        # デフォルト: 8小節 @ 120 BPM
        beat_duration = 60.0 / tempo_bpm
        bar_duration = beat_duration * 4  # 4拍子
        return pd.DataFrame({
            'bar_number': list(range(1, 9)),
            'start_sec': [i * bar_duration for i in range(8)],
            'end_sec': [(i + 1) * bar_duration for i in range(8)],
            'duration_sec': [bar_duration] * 8,
            'time_signature': ['4/4'] * 8
        })
    
    # ビート時刻から小節を4拍単位で分割
    bars_data = []
    beats_per_bar = 4
    num_bars = len(beat_times) // beats_per_bar
    
    for bar_idx in range(num_bars):
        beat_start_idx = bar_idx * beats_per_bar
        beat_end_idx = beat_start_idx + beats_per_bar
        
        start_sec = beat_times[beat_start_idx]
        end_sec = beat_times[min(beat_end_idx, len(beat_times) - 1)]
        
        bars_data.append({
            'bar_number': bar_idx + 1,
            'start_sec': start_sec,
            'end_sec': end_sec,
            'duration_sec': end_sec - start_sec,
            'time_signature': '4/4'
        })
    
    # 最後の不完全な小節を処理
    remaining_beats = len(beat_times) % beats_per_bar
    if remaining_beats > 0:
        beat_start_idx = num_bars * beats_per_bar
        start_sec = beat_times[beat_start_idx]
        
        # 最後のビート間隔から終了時刻を推定
        if len(beat_times) > 1:
            avg_beat_duration = (beat_times[-1] - beat_times[0]) / max(1, len(beat_times) - 1)
            end_sec = beat_times[-1] + avg_beat_duration
        else:
            end_sec = start_sec + 2.0  # デフォルト2秒
        
        bars_data.append({
            'bar_number': num_bars + 1,
            'start_sec': start_sec,
            'end_sec': end_sec,
            'duration_sec': end_sec - start_sec,
            'time_signature': f"{remaining_beats}/4"
        })
    
    return pd.DataFrame(bars_data)


def create_song_package(song_dir: Path, output_dir: Path, dataset: str) -> bool:
    """
    楽曲Song Package生成
    
    Returns:
        成功時True、失敗時False
    """
    song_id = song_dir.name
    output_song_dir = output_dir / song_id
    output_song_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # audio_chordmap.yaml読み込み
        chordmap_path = song_dir / "audio_chordmap.yaml"
        if not chordmap_path.exists():
            logging.warning(f"[{song_id}] audio_chordmap.yaml not found")
            return False
        
        chordmap_data = load_audio_chordmap(chordmap_path)
        
        # テンポフォールバックチェーン: chordmap → bars → 120.0
        tempo_bpm = chordmap_data.get('tempo_bpm')
        if tempo_bpm is None or tempo_bpm <= 0:
            tempo_bpm = 120.0
        
        # beat_grid.json読み込み
        beat_grid_path = song_dir / "beat_grid.json"
        beat_times = []
        
        if beat_grid_path.exists():
            beat_grid_data = load_beat_grid(beat_grid_path)
            beat_times = beat_grid_data.get('beat_times', [])
        else:
            logging.warning(f"[{song_id}] beat_grid.json not found, using default")
        
        # bars.parquet生成
        bars_df = generate_bars_from_beat_times(beat_times, tempo_bpm)
        
        # 実オーディオ長を取得してクリップ（尻切れ防止）
        audio_duration = chordmap_data.get('audio_duration_sec')
        if audio_duration and len(bars_df) > 0:
            last_idx = bars_df.index[-1]
            if bars_df.loc[last_idx, 'end_sec'] > audio_duration:
                bars_df.loc[last_idx, 'end_sec'] = audio_duration
                bars_df.loc[last_idx, 'duration_sec'] = audio_duration - bars_df.loc[last_idx, 'start_sec']
        
        # bars.parquetフォールバック: テンポ未設定時はbars平均から推定
        if tempo_bpm == 120.0 and len(bars_df) > 0 and 'duration_sec' in bars_df.columns:
            avg_bar_duration = bars_df['duration_sec'].mean()
            if avg_bar_duration > 0:
                inferred_tempo = 240.0 / avg_bar_duration  # 4/4拍子前提
                tempo_bpm = inferred_tempo
        
        bars_parquet_path = output_song_dir / "bars.parquet"
        bars_df.to_parquet(bars_parquet_path, index=False)
        
        # song_package.yaml生成
        package_data = {
            'song_id': song_id,
            'dataset': dataset,
            'source': 'wav',
            'tempo_bpm': float(tempo_bpm),
            'time_signature': '4/4',
            'total_bars': int(len(bars_df)),
            'duration_sec': float(bars_df['end_sec'].max() if len(bars_df) > 0 else 16.0),
            'paths': {
                'bars': str(bars_parquet_path.relative_to(output_song_dir)),
                'audio_chordmap': str(chordmap_path),
                'beat_grid': str(beat_grid_path) if beat_grid_path.exists() else None
            },
            'harmony': {
                'roles': [role['role'] for role in chordmap_data.get('chordmap', [])],
                'weights': {
                    role['role']: float(role.get('weight', 0.0))
                    for role in chordmap_data.get('chordmap', [])
                }
            },
            'metadata': {
                'policy_profile': chordmap_data.get('policy_metadata', {}).get('profile', dataset),
                'policy_version': int(chordmap_data.get('policy_metadata', {}).get('version', 2))
            }
        }
        
        package_yaml_path = output_song_dir / "song_package.yaml"
        with open(package_yaml_path, 'w', encoding='utf-8') as f:
            yaml.safe_dump(package_data, f, default_flow_style=False, allow_unicode=True)
        
        return True
    
    except Exception as e:
        logging.error(f"[{song_id}] Failed to create package: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Generate WAV Harmony Song Packages")
    parser.add_argument('--input', type=str, required=True, help='Input directory (e.g., output/wav_cleaned/moisesdb)')
    parser.add_argument('--output', type=str, required=True, help='Output directory (e.g., output/wav_harmony_packages/moisesdb)')
    parser.add_argument('--dataset', type=str, required=True, choices=['moisesdb', 'musdb18'], help='Dataset name')
    
    args = parser.parse_args()
    
    setup_logging()
    
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 楽曲ディレクトリ取得
    song_dirs = [d for d in input_dir.iterdir() if d.is_dir()]
    
    print("=" * 100)
    print(f"WAV Harmony Song Package Generation")
    print("=" * 100)
    print(f"Dataset: {args.dataset}")
    print(f"Input: {input_dir}")
    print(f"Total songs: {len(song_dirs)}")
    print("=" * 100)
    print()
    
    # Song Package生成
    success_count = 0
    with tqdm(song_dirs, desc="Generating packages") as pbar:
        for song_dir in pbar:
            if create_song_package(song_dir, output_dir, args.dataset):
                success_count += 1
            pbar.set_postfix({'success': success_count})
    
    # インデックス生成
    index_data = []
    for pkg_dir in output_dir.iterdir():
        if pkg_dir.is_dir():
            pkg_yaml = pkg_dir / "song_package.yaml"
            if pkg_yaml.exists():
                try:
                    with open(pkg_yaml, 'r') as f:
                        pkg = yaml.safe_load(f)
                    
                    if pkg is None:
                        continue
                    
                    index_data.append({
                        'song_id': pkg.get('song_id', ''),
                        'dataset': pkg.get('dataset', args.dataset),
                        'source': pkg.get('source', 'wav'),
                        'package_path': str(pkg_dir),
                        'total_bars': pkg.get('total_bars', 0),
                        'duration_sec': pkg.get('duration_sec', 0.0),
                        'roles': ','.join(pkg.get('harmony', {}).get('roles', []))
                    })
                except Exception as e:
                    logging.warning(f"Failed to load {pkg_yaml}: {e}")
                    continue
    
    index_df = pd.DataFrame(index_data)
    index_csv_path = output_dir.parent / f"{args.dataset}_song_packages_index.csv"
    index_df.to_csv(index_csv_path, index=False)
    
    print()
    print("=" * 100)
    print("Summary")
    print("=" * 100)
    print(f"Total:    {len(song_dirs)}")
    print(f"Success:  {success_count} ({success_count/len(song_dirs)*100:.1f}%)")
    print(f"Failed:   {len(song_dirs) - success_count}")
    print(f"Index:    {index_csv_path} ({len(index_df)} records)")
    print("=" * 100)


if __name__ == '__main__':
    main()
