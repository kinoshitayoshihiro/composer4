#!/usr/bin/env python3
"""
Voicings Guide Generator from Chordmap

chordmap.jsonからボイシング・ガイド（許容テンション/avoid/解決）を生成

Usage:
    python make_voicings_guide.py \
        --chordmap analysis/chordmap.json \
        --out-csv analysis/voicings_guide.csv
"""

import argparse
import json
import pandas as pd
from pathlib import Path


def get_tensions_and_avoids(root: str, quality: str) -> dict:
    """コード種別ごとの許容テンション/avoid音"""
    
    # 基本ボイシングルール
    rules = {
        'major': {
            'tensions': ['9', '13'],
            'avoid': ['11'],  # avoid #11 unless altered
            'resolution': 'stable',
        },
        'minor': {
            'tensions': ['9', '11'],
            'avoid': ['13'],  # avoid b13 unless modal
            'resolution': 'stable',
        },
        'dominant': {
            'tensions': ['9', '13', 'b9', '#9', 'b13'],
            'avoid': [],  # dominant は柔軟
            'resolution': 'to_tonic',
        },
        'diminished': {
            'tensions': ['b9', 'b13'],
            'avoid': ['9', '11'],
            'resolution': 'to_tonic',
        },
        'augmented': {
            'tensions': ['#9', '#11'],
            'avoid': ['9', '13'],
            'resolution': 'unstable',
        },
        'sus4': {
            'tensions': ['9'],
            'avoid': ['3'],  # sus4はtritoneを避ける
            'resolution': 'to_major',
        },
    }
    
    # quality正規化
    quality_normalized = quality.lower()
    if '7' in quality_normalized or 'dom' in quality_normalized:
        key = 'dominant'
    elif 'dim' in quality_normalized or '°' in quality_normalized:
        key = 'diminished'
    elif 'aug' in quality_normalized or '+' in quality_normalized:
        key = 'augmented'
    elif 'sus' in quality_normalized:
        key = 'sus4'
    elif 'min' in quality_normalized or 'm' == quality_normalized:
        key = 'minor'
    else:
        key = 'major'
    
    return rules.get(key, rules['major'])


def generate_voicings_guide(chordmap_path: Path) -> pd.DataFrame:
    """ボイシング・ガイド生成"""
    
    with open(chordmap_path) as f:
        chordmap = json.load(f)
    
    rows = []
    
    for event in chordmap:
        bar = event.get('bar', 0)
        root = event.get('root', 'C')
        quality = event.get('quality', 'major')
        
        # テンション/avoid取得
        voicing_info = get_tensions_and_avoids(root, quality)
        
        rows.append({
            'bar': bar,
            'root': root,
            'quality': quality,
            'allowed_tensions': ','.join(voicing_info['tensions']),
            'avoid_notes': ','.join(voicing_info['avoid']),
            'resolution_type': voicing_info['resolution'],
        })
    
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description='Voicings guide generator')
    parser.add_argument('--chordmap', type=Path, required=True)
    parser.add_argument('--out-csv', type=Path, required=True)
    
    args = parser.parse_args()
    
    # ボイシングガイド生成
    voicings_df = generate_voicings_guide(args.chordmap)
    
    # CSV出力
    voicings_df.to_csv(args.out_csv, index=False)
    
    print(f"✅ Voicings guide generated: {len(voicings_df)} chords")
    print(f"   Output: {args.out_csv}")
    
    # 統計
    resolution_counts = voicings_df['resolution_type'].value_counts()
    print(f"   Resolution types: {resolution_counts.to_dict()}")


if __name__ == '__main__':
    main()
