#!/usr/bin/env python3
"""
generate_benchmark_json.py - ベンチマーク曲集JSON自動生成スクリプト

configs/benchmarks/*.yamlから全ベンチマーク曲を読み込み、
multi_song_benchmark.jsonを生成します。

Usage:
    python scripts/generate_benchmark_json.py [--output OUTPUT]
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import yaml


def load_benchmark_yaml(yaml_path: Path) -> Dict[str, Any]:
    """ベンチマークYAMLファイルを読み込み"""
    with open(yaml_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    return data


def extract_benchmark_metadata(yaml_data: Dict[str, Any], yaml_path: Path) -> Dict[str, Any]:
    """YAMLからベンチマークメタデータを抽出"""
    meta = yaml_data.get('meta', {})
    
    benchmark_entry = {
        'id': yaml_path.stem,  # ファイル名(拡張子なし)
        'file': str(yaml_path.relative_to(yaml_path.parents[2])),  # プロジェクトルートからの相対パス
        'metadata': {
            'title': meta.get('title', 'Unknown'),
            'artist': meta.get('artist', 'Benchmark Suite'),
            'genre': meta.get('genre', 'Unknown'),
            'style': meta.get('style', 'Unknown'),
            'difficulty': meta.get('difficulty', 'medium'),
            'seed': meta.get('seed', 0),
        },
        'expected_metrics': meta.get('expected_metrics', {}),
        'quality_thresholds': yaml_data.get('quality_thresholds', {}),
        'global_config': yaml_data.get('global', {}),
        'section_count': len(yaml_data.get('sections', [])),
    }
    
    return benchmark_entry


def generate_benchmark_json(benchmarks_dir: Path, output_path: Path) -> None:
    """ベンチマークJSON生成"""
    
    # 全YAMLファイルを取得
    yaml_files = sorted(benchmarks_dir.glob('*.yaml'))
    
    if not yaml_files:
        print(f"⚠️  No YAML files found in {benchmarks_dir}", file=sys.stderr)
        sys.exit(1)
    
    print(f"📂 Found {len(yaml_files)} benchmark YAML files")
    
    # 各YAMLを処理
    benchmark_songs: List[Dict[str, Any]] = []
    
    for yaml_path in yaml_files:
        print(f"   Processing: {yaml_path.name}")
        
        try:
            yaml_data = load_benchmark_yaml(yaml_path)
            metadata = extract_benchmark_metadata(yaml_data, yaml_path)
            benchmark_songs.append(metadata)
            
        except Exception as e:
            print(f"❌ Error processing {yaml_path.name}: {e}", file=sys.stderr)
            continue
    
    # ジャンル別にグループ化
    genres: Dict[str, List[Dict[str, Any]]] = {}
    for song in benchmark_songs:
        genre = song['metadata']['genre']
        if genre not in genres:
            genres[genre] = []
        genres[genre].append(song)
    
    # 最終JSON生成
    benchmark_json = {
        'version': '1.0',
        'generated': datetime.now().isoformat(),
        'description': 'Benchmark song suite for regression testing and quality validation',
        'total_songs': len(benchmark_songs),
        'genres': {
            genre: len(songs) for genre, songs in genres.items()
        },
        'songs': benchmark_songs,
    }
    
    # ファイル出力
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(benchmark_json, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Generated benchmark JSON: {output_path}")
    print(f"   Total songs: {len(benchmark_songs)}")
    print(f"   Genres: {', '.join(genres.keys())}")
    
    # 統計サマリー
    print("\n📊 Benchmark Suite Summary:")
    for genre, songs in sorted(genres.items()):
        print(f"   {genre}: {len(songs)} songs")
        for song in songs:
            difficulty = song['metadata']['difficulty']
            title = song['metadata']['title']
            print(f"      - {title} ({difficulty})")


def main():
    parser = argparse.ArgumentParser(
        description='Generate benchmark suite JSON from YAML files'
    )
    parser.add_argument(
        '--benchmarks-dir',
        type=str,
        default='configs/benchmarks',
        help='Directory containing benchmark YAML files (default: configs/benchmarks)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='multi_song_benchmark.json',
        help='Output JSON file path (default: multi_song_benchmark.json)'
    )
    
    args = parser.parse_args()
    
    # パス解決
    project_root = Path(__file__).parent.parent
    benchmarks_dir = project_root / args.benchmarks_dir
    output_path = project_root / args.output
    
    if not benchmarks_dir.exists():
        print(f"❌ Benchmarks directory not found: {benchmarks_dir}", file=sys.stderr)
        sys.exit(1)
    
    # JSON生成実行
    generate_benchmark_json(benchmarks_dir, output_path)


if __name__ == '__main__':
    main()
