#!/usr/bin/env python3
"""
Harmony AI WAV Pickle Builder

WAV Song Package → Pickle統合（和声パターン辞書）
  - lite/fat モード（既定: lite = chordmapは外部YAML参照）
  - role/weight情報の抽出
  - IDマップ/重複/欠損チェック
  - song_package.yaml 再帰探索

Usage:
    python scripts/build_harmony_wav_pickle.py \
        --song-packages-moisesdb output/wav_harmony_packages/moisesdb \
        --song-packages-musdb18 output/wav_harmony_packages/musdb18 \
        --output output/harmony_wav/harmony_patterns.pickle \
        --metadata-out output/harmony_wav/harmony_metadata.json \
        --mode lite \
        --id-column song_id
"""

import argparse
import json
import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime
import hashlib

import pandas as pd
import yaml
from tqdm import tqdm


def sha256_file(p: Path) -> Optional[str]:
    """ファイルのSHA256計算"""
    try:
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(1024*1024), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return None


def load_song_packages(package_dirs: List[Path]) -> List[Dict[str, Any]]:
    """
    Song Package再帰探索・読み込み
    
    Returns:
        List of {song_id, dataset, source, tempo_bpm, total_bars, duration_sec,
                 roles, weights, audio_chordmap_path, bars_path, package_dir}
    """
    packages = []
    
    for pkg_dir in package_dirs:
        yaml_files = sorted(pkg_dir.rglob("song_package.yaml"))
        
        for yaml_path in tqdm(yaml_files, desc=f"Loading packages from {pkg_dir.name}"):
            try:
                with open(yaml_path, 'r', encoding='utf-8') as f:
                    data = yaml.safe_load(f)
                
                if data is None:
                    continue
                
                # Harmony情報抽出
                harmony = data.get('harmony', {})
                roles = harmony.get('roles', [])
                weights = harmony.get('weights', {})
                
                # Pathsからaudio_chordmap取得
                paths = data.get('paths', {})
                audio_chordmap_path = paths.get('audio_chordmap', '')
                bars_path = paths.get('bars', '')
                
                packages.append({
                    'song_id': data.get('song_id', ''),
                    'dataset': data.get('dataset', ''),
                    'source': data.get('source', 'wav'),
                    'tempo_bpm': float(data.get('tempo_bpm', 120.0)),
                    'total_bars': int(data.get('total_bars', 0)),
                    'duration_sec': float(data.get('duration_sec', 0.0)),
                    'roles': roles,
                    'weights': weights,
                    'audio_chordmap_path': audio_chordmap_path,
                    'bars_path': str(yaml_path.parent / bars_path) if bars_path else '',
                    'package_dir': str(yaml_path.parent),
                    'time_signature': data.get('time_signature', '4/4'),
                    'policy_profile': data.get('metadata', {}).get('policy_profile', ''),
                    'policy_version': int(data.get('metadata', {}).get('policy_version', 2))
                })
            
            except Exception as e:
                logging.warning(f"Failed to load {yaml_path}: {e}")
                continue
    
    return packages


def build_pickle(
    packages: List[Dict[str, Any]],
    mode: str = "lite",
    id_column: str = "song_id"
) -> Dict[str, Any]:
    """
    Pickle構築
    
    Args:
        packages: Song Package情報リスト
        mode: "lite" (外部参照) or "fat" (全データ埋め込み)
        id_column: ID列名
    
    Returns:
        Pickle dictionary
    """
    # DataFrame化
    df = pd.DataFrame(packages)
    
    # ID重複/欠損チェック
    id_nulls = df[id_column].isna().sum()
    id_duplicates = df[id_column].duplicated().sum()
    
    # ID → インデックスマッピング
    id_index = {row[id_column]: idx for idx, row in df.iterrows()}
    
    # Dataset分布
    dataset_counts = df['dataset'].value_counts().to_dict() if 'dataset' in df.columns else {}
    
    # Role統計
    all_roles = []
    for roles in df['roles']:
        all_roles.extend(roles)
    role_counts = pd.Series(all_roles).value_counts().to_dict()
    
    # Lite/Fat mode処理
    if mode == "lite":
        # 外部参照モード（audio_chordmap_pathのみ保持）
        chordmap_data = None
        bars_data = None
    else:
        # Fat mode: すべてのchordmap/bars読み込み（重い）
        chordmap_data = []
        bars_data = []
        
        for _, row in df.iterrows():
            # audio_chordmap読み込み
            chordmap_path = Path(row['audio_chordmap_path'])
            if chordmap_path.exists():
                with open(chordmap_path, 'r') as f:
                    chordmap_data.append(yaml.safe_load(f))
            else:
                chordmap_data.append(None)
            
            # bars.parquet読み込み
            bars_path = Path(row['bars_path'])
            if bars_path.exists():
                bars_data.append(pd.read_parquet(bars_path))
            else:
                bars_data.append(None)
    
    # Preview rows（3件、numpy対応）
    preview_df = df.head(3).copy()
    preview_rows = []
    for _, row in preview_df.iterrows():
        row_dict = {}
        for col, val in row.items():
            # List/dict型を先にチェック
            if isinstance(val, (list, dict)):
                row_dict[col] = val
            elif hasattr(val, 'tolist'):  # numpy array
                row_dict[col] = val.tolist()
            elif hasattr(val, 'item'):  # numpy scalar
                row_dict[col] = val.item()
            elif isinstance(val, (int, float, str, bool)):
                row_dict[col] = val
            elif val is None or (isinstance(val, float) and pd.isna(val)):
                row_dict[col] = None
            else:
                row_dict[col] = str(val)
        preview_rows.append(row_dict)
    
    # Pickle dictionary構築
    result = {
        'metadata': {
            'version': '1.0.0',
            'created_at': datetime.now().isoformat(),
            'mode': mode,
            'id_column': id_column,
            'total_songs': len(df),
            'datasets': dataset_counts,
            'role_distribution': role_counts,
            'columns': list(df.columns),
            'id_nulls': int(id_nulls),
            'id_duplicates': int(id_duplicates)
        },
        'song_info': df.to_dict(orient='records'),
        'id_index': id_index,
        'chordmap_data': chordmap_data,  # None (lite) or List[Dict] (fat)
        'bars_data': bars_data,  # None (lite) or List[DataFrame] (fat)
        'preview_rows': preview_rows
    }
    
    return result


def main():
    parser = argparse.ArgumentParser(description="Build Harmony AI WAV Pickle")
    parser.add_argument('--song-packages-moisesdb', type=str, required=True,
                        help='MoisesDB song packages directory')
    parser.add_argument('--song-packages-musdb18', type=str, required=True,
                        help='MUSDB18 song packages directory')
    parser.add_argument('--output', type=str, required=True,
                        help='Output pickle file path')
    parser.add_argument('--metadata-out', type=str, required=True,
                        help='Output metadata JSON path')
    parser.add_argument('--mode', type=str, default='lite', choices=['lite', 'fat'],
                        help='Pickle mode: lite (external refs) or fat (embedded data)')
    parser.add_argument('--id-column', type=str, default='song_id',
                        help='ID column name')
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
    
    # Song Package読み込み
    package_dirs = [
        Path(args.song_packages_moisesdb),
        Path(args.song_packages_musdb18)
    ]
    
    print("📦 Loading Song Packages...")
    packages = load_song_packages(package_dirs)
    print(f"  Total packages: {len(packages)}")
    print()
    
    # Pickle構築
    print("🔧 Building integrated pickle...")
    pickle_data = build_pickle(packages, mode=args.mode, id_column=args.id_column)
    print()
    
    # Pickle保存
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'wb') as f:
        pickle.dump(pickle_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    pickle_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"💾 Saving pickle: {output_path}")
    print(f"  Size: {pickle_size_mb:.2f} MB")
    print()
    
    # Metadata JSON保存
    metadata_path = Path(args.metadata_out)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Pickle SHA256計算
    pickle_sha256 = sha256_file(output_path)
    
    metadata = {
        'pickle_file': str(output_path),
        'pickle_size_mb': round(pickle_size_mb, 2),
        'pickle_sha256': pickle_sha256,
        'created_at': pickle_data['metadata']['created_at'],
        'mode': args.mode,
        'id_column': args.id_column,
        'total_songs': pickle_data['metadata']['total_songs'],
        'datasets': pickle_data['metadata']['datasets'],
        'role_distribution': pickle_data['metadata']['role_distribution'],
        'columns': pickle_data['metadata']['columns'],
        'id_nulls': pickle_data['metadata']['id_nulls'],
        'id_duplicates': pickle_data['metadata']['id_duplicates'],
        'preview_rows': pickle_data['preview_rows']
    }
    
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    print(f"📝 Saving metadata: {metadata_path}")
    print()
    
    # Summary
    print("=" * 100)
    print("Summary")
    print("=" * 100)
    print(f"Total Songs:   {pickle_data['metadata']['total_songs']}")
    print(f"Datasets:      {pickle_data['metadata']['datasets']}")
    print(f"Roles:         {pickle_data['metadata']['role_distribution']}")
    print(f"Mode:          {args.mode}")
    print(f"ID column:     {args.id_column}")
    print(f"ID nulls:      {pickle_data['metadata']['id_nulls']}")
    print(f"ID duplicates: {pickle_data['metadata']['id_duplicates']}")
    print(f"Pickle size:   {pickle_size_mb:.2f} MB")
    print(f"Pickle SHA256: {pickle_sha256}")
    print("=" * 100)


if __name__ == '__main__':
    main()
