#!/usr/bin/env python3
"""
A/B比較メトリクス集計スクリプト
旧処理 vs 新処理のaudio_chordmap.yamlを比較して品質指標を算出

Usage:
    python scripts/ab_compare_policy_metrics.py \
        --old-dir data/old_wav_guide \
        --new-dir data/new_wav_guide \
        --output metrics_comparison.csv
"""

import argparse
import yaml
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np
from collections import Counter
import warnings

warnings.filterwarnings('ignore')


def load_chordmap(yaml_path: Path) -> Optional[Dict[str, Any]]:
    """audio_chordmap.yamlを読み込み"""
    try:
        with open(yaml_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"⚠️  Failed to load {yaml_path}: {e}")
        return None


def extract_chords_from_chordmap(chordmap: Dict[str, Any]) -> List[str]:
    """
    chordmapから採用コード列を抽出
    
    フォーマット対応:
    - chordmap: [{role, weight, chord_candidates}] 形式
    - bars: [{chord, confidence, votes}] 形式（推奨）
    """
    chords = []
    
    # 形式1: bars配列（推奨形式）
    if 'bars' in chordmap:
        for bar in chordmap['bars']:
            if isinstance(bar, dict) and 'chord' in bar:
                chords.append(bar['chord'])
    
    # 形式2: segments/measures配列（フォールバック）
    elif 'segments' in chordmap:
        for seg in chordmap['segments']:
            if isinstance(seg, dict) and 'chord' in seg:
                chords.append(seg['chord'])
    elif 'measures' in chordmap:
        for measure in chordmap['measures']:
            if isinstance(measure, dict) and 'chord' in measure:
                chords.append(measure['chord'])
    
    # 形式3: chordmap配列のみ（候補のみで採用コードなし）
    # この場合は最初の候補を使用
    elif 'chordmap' in chordmap:
        for entry in chordmap['chordmap']:
            if isinstance(entry, dict) and 'chord_candidates' in entry:
                cands = entry['chord_candidates']
                if cands and len(cands) > 0:
                    chords.append(cands[0])
    
    return chords


def calculate_chord_entropy(chords: List[str]) -> float:
    """
    コード分布の正規化エントロピー（0..1）
    
    多様性が高い（不安定）→1.0に近い
    単調（安定）→0.0に近い
    """
    if not chords:
        return 0.0
    
    # コード出現頻度
    counter = Counter(chords)
    total = len(chords)
    
    # エントロピー計算
    entropy = 0.0
    for count in counter.values():
        p = count / total
        if p > 0:
            entropy -= p * np.log2(p)
    
    # 正規化（最大エントロピーで割る）
    max_entropy = np.log2(len(counter)) if len(counter) > 1 else 1.0
    normalized = entropy / max_entropy if max_entropy > 0 else 0.0
    
    return normalized


def calculate_segment_stability(chords: List[str]) -> float:
    """
    セグメント安定度: 1 - 変化回数/(バー数-1)
    
    1.0 = 完全に安定（変化なし）
    0.0 = 毎小節変化
    """
    if len(chords) <= 1:
        return 1.0
    
    # 連続するコードの変化回数をカウント
    changes = sum(1 for i in range(1, len(chords)) if chords[i] != chords[i-1])
    max_changes = len(chords) - 1
    
    stability = 1.0 - (changes / max_changes) if max_changes > 0 else 1.0
    return stability


def extract_confidence_from_chordmap(chordmap: Dict[str, Any]) -> List[float]:
    """bars配列からconfidenceを抽出"""
    confidences = []
    
    if 'bars' in chordmap:
        for bar in chordmap['bars']:
            if isinstance(bar, dict) and 'confidence' in bar:
                conf = bar['confidence']
                if conf is not None:
                    confidences.append(float(conf))
    elif 'segments' in chordmap:
        for seg in chordmap['segments']:
            if isinstance(seg, dict) and 'confidence' in seg:
                conf = seg['confidence']
                if conf is not None:
                    confidences.append(float(conf))
    
    return confidences


def extract_bass_root_agreement(chordmap: Dict[str, Any]) -> Optional[float]:
    """
    bassの推定rootと採用コードのrootの一致率
    
    bars: [{chord: "C", votes: {bass: {top: {root: "C"}}}}] 形式を想定
    """
    if 'bars' not in chordmap:
        return None
    
    agreements = []
    
    for bar in chordmap['bars']:
        if not isinstance(bar, dict):
            continue
        
        chord = bar.get('chord', '')
        votes = bar.get('votes', {})
        
        # bassのtop root取得
        bass_votes = votes.get('bass', {})
        bass_top = bass_votes.get('top', {})
        bass_root = bass_top.get('root', '')
        
        # コードのroot取得（簡易版: コード名の最初の文字）
        chord_root = chord[0] if chord else ''
        
        if bass_root and chord_root:
            agreements.append(1.0 if bass_root == chord_root else 0.0)
    
    if not agreements:
        return None
    
    return sum(agreements) / len(agreements)


def compute_metrics_for_song(chordmap: Dict[str, Any]) -> Dict[str, Any]:
    """1曲分のメトリクスを計算"""
    chords = extract_chords_from_chordmap(chordmap)
    confidences = extract_confidence_from_chordmap(chordmap)
    
    metrics = {
        'num_bars': len(chords),
        'chord_entropy': calculate_chord_entropy(chords) if chords else None,
        'segment_stability': calculate_segment_stability(chords) if chords else None,
        'conf_mean': np.mean(confidences) if confidences else None,
        'conf_std': np.std(confidences) if confidences else None,
        'bass_root_agreement': extract_bass_root_agreement(chordmap),
        'unique_chords': len(set(chords)) if chords else 0,
        'most_common_chord': Counter(chords).most_common(1)[0][0] if chords else None
    }
    
    # policy_metadataも取得（新処理の場合）
    if 'policy_metadata' in chordmap:
        pm = chordmap['policy_metadata']
        metrics['policy_profile'] = pm.get('profile', 'unknown')
        metrics['policy_version'] = pm.get('version', 1)
        metrics['weights_digest'] = pm.get('weights_digest', 'n/a')
    else:
        metrics['policy_profile'] = 'legacy'
        metrics['policy_version'] = 1
        metrics['weights_digest'] = 'legacy'
    
    return metrics


def compare_directories(old_dir: Path, new_dir: Path) -> pd.DataFrame:
    """2つのディレクトリを比較してメトリクスCSVを生成"""
    results = []
    
    # 新ディレクトリの全曲をベースに比較
    new_songs = list(new_dir.glob('*/audio_chordmap.yaml'))
    
    print(f"Found {len(new_songs)} songs in new directory")
    
    for new_yaml in new_songs:
        song_id = new_yaml.parent.name
        old_yaml = old_dir / song_id / 'audio_chordmap.yaml'
        
        # 新処理のメトリクス
        new_chordmap = load_chordmap(new_yaml)
        if not new_chordmap:
            continue
        
        new_metrics = compute_metrics_for_song(new_chordmap)
        
        # 旧処理のメトリクス（存在する場合）
        old_metrics = None
        if old_yaml.exists():
            old_chordmap = load_chordmap(old_yaml)
            if old_chordmap:
                old_metrics = compute_metrics_for_song(old_chordmap)
        
        # 結果を記録
        row = {
            'song_id': song_id,
            
            # 新処理
            'new_chord_entropy': new_metrics['chord_entropy'],
            'new_segment_stability': new_metrics['segment_stability'],
            'new_conf_mean': new_metrics['conf_mean'],
            'new_bass_root_agreement': new_metrics['bass_root_agreement'],
            'new_unique_chords': new_metrics['unique_chords'],
            'new_num_bars': new_metrics['num_bars'],
            'new_policy_profile': new_metrics['policy_profile'],
            'new_weights_digest': new_metrics['weights_digest'],
        }
        
        # 旧処理（存在する場合）
        if old_metrics:
            row.update({
                'old_chord_entropy': old_metrics['chord_entropy'],
                'old_segment_stability': old_metrics['segment_stability'],
                'old_conf_mean': old_metrics['conf_mean'],
                'old_bass_root_agreement': old_metrics['bass_root_agreement'],
                'old_unique_chords': old_metrics['unique_chords'],
                'old_num_bars': old_metrics['num_bars'],
            })
            
            # 差分計算
            if new_metrics['chord_entropy'] is not None and old_metrics['chord_entropy'] is not None:
                row['delta_entropy'] = new_metrics['chord_entropy'] - old_metrics['chord_entropy']
            if new_metrics['segment_stability'] is not None and old_metrics['segment_stability'] is not None:
                row['delta_stability'] = new_metrics['segment_stability'] - old_metrics['segment_stability']
            if new_metrics['conf_mean'] is not None and old_metrics['conf_mean'] is not None:
                row['delta_conf_mean'] = new_metrics['conf_mean'] - old_metrics['conf_mean']
        else:
            row.update({
                'old_chord_entropy': None,
                'old_segment_stability': None,
                'old_conf_mean': None,
                'old_bass_root_agreement': None,
                'old_unique_chords': None,
                'old_num_bars': None,
                'delta_entropy': None,
                'delta_stability': None,
                'delta_conf_mean': None,
            })
        
        results.append(row)
    
    df = pd.DataFrame(results)
    return df


def print_summary(df: pd.DataFrame):
    """サマリー統計を出力"""
    print("\n" + "="*80)
    print("A/B比較メトリクス サマリー")
    print("="*80)
    
    # 全体統計
    total = len(df)
    with_old = df['old_chord_entropy'].notna().sum()
    
    print(f"\n全体:")
    print(f"  新処理曲数: {total}")
    print(f"  旧処理曲数: {with_old}")
    print(f"  比較可能曲数: {with_old}")
    
    # 新処理の平均
    print(f"\n新処理の平均メトリクス:")
    print(f"  chord_entropy: {df['new_chord_entropy'].mean():.4f}")
    print(f"  segment_stability: {df['new_segment_stability'].mean():.4f}")
    print(f"  conf_mean: {df['new_conf_mean'].mean():.4f}" if df['new_conf_mean'].notna().sum() > 0 else "  conf_mean: N/A")
    print(f"  unique_chords: {df['new_unique_chords'].mean():.2f}")
    
    # 旧処理の平均（存在する場合）
    if with_old > 0:
        print(f"\n旧処理の平均メトリクス:")
        print(f"  chord_entropy: {df['old_chord_entropy'].mean():.4f}")
        print(f"  segment_stability: {df['old_segment_stability'].mean():.4f}")
        print(f"  conf_mean: {df['old_conf_mean'].mean():.4f}" if df['old_conf_mean'].notna().sum() > 0 else "  conf_mean: N/A")
        print(f"  unique_chords: {df['old_unique_chords'].mean():.2f}")
        
        # 差分の平均
        print(f"\n改善度（新 - 旧）:")
        print(f"  Δ entropy: {df['delta_entropy'].mean():.4f} (負=改善)")
        print(f"  Δ stability: {df['delta_stability'].mean():.4f} (正=改善)")
        print(f"  Δ conf_mean: {df['delta_conf_mean'].mean():.4f}" if df['delta_conf_mean'].notna().sum() > 0 else "  Δ conf_mean: N/A")
    
    # プロファイル分布
    print(f"\n新処理のプロファイル分布:")
    profile_counts = df['new_policy_profile'].value_counts()
    for profile, count in profile_counts.items():
        print(f"  {profile}: {count}曲")
    
    print("\n" + "="*80)


def main():
    parser = argparse.ArgumentParser(
        description="A/B比較メトリクス集計（旧処理 vs 新処理）"
    )
    parser.add_argument(
        '--old-dir',
        type=Path,
        required=True,
        help='旧処理の出力ディレクトリ（audio_chordmap.yamlがあるディレクトリの親）'
    )
    parser.add_argument(
        '--new-dir',
        type=Path,
        required=True,
        help='新処理の出力ディレクトリ（audio_chordmap.yamlがあるディレクトリの親）'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('metrics_comparison.csv'),
        help='出力CSVファイル名'
    )
    
    args = parser.parse_args()
    
    if not args.old_dir.exists():
        print(f"❌ Old directory not found: {args.old_dir}")
        return
    
    if not args.new_dir.exists():
        print(f"❌ New directory not found: {args.new_dir}")
        return
    
    print(f"Comparing:")
    print(f"  Old: {args.old_dir}")
    print(f"  New: {args.new_dir}")
    
    # メトリクス計算
    df = compare_directories(args.old_dir, args.new_dir)
    
    # CSV出力
    df.to_csv(args.output, index=False)
    print(f"\n✅ Saved metrics to: {args.output}")
    
    # サマリー表示
    print_summary(df)


if __name__ == '__main__':
    main()
