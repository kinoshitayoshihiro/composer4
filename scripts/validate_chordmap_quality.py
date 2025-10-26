#!/usr/bin/env python3
"""
ChordMap品質検証スクリプト
- 和声ラベル精度評価（サンプリング検証）
- music21推定の妥当性確認（特にテンション）
- コード遷移の自然性チェック（V→I等）
- 空MIDIファイル対応（0 chord eventsの曲が存在）
"""

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import yaml
from tqdm import tqdm


# ============================================================
# コード遷移ルール（V→I等の妥当性チェック）
# ============================================================

# 主要なコード進行パターン（簡易版）
VALID_PROGRESSIONS = {
    # V→I系
    ('G', 'C'): 'V-I',
    ('D', 'G'): 'V-I',
    ('A', 'D'): 'V-I',
    ('E', 'A'): 'V-I',
    ('B', 'E'): 'V-I',
    ('F#', 'B'): 'V-I',
    
    # ii→V系
    ('D', 'G'): 'ii-V',
    ('A', 'D'): 'ii-V',
    ('E', 'A'): 'ii-V',
    
    # IV→I系
    ('F', 'C'): 'IV-I',
    ('C', 'G'): 'IV-I',
    ('G', 'D'): 'IV-I',
    
    # I→V系
    ('C', 'G'): 'I-V',
    ('G', 'D'): 'I-V',
    ('D', 'A'): 'I-V',
}


def analyze_chord_progression(root1: str, root2: str) -> str:
    """コード進行の妥当性を判定"""
    key = (root1, root2)
    return VALID_PROGRESSIONS.get(key, 'other')


# ============================================================
# ChordMap検証
# ============================================================

def validate_chordmap(chordmap_path: Path, song_id: str) -> Dict:
    """
    1つのchordmap.jsonを検証
    
    Returns:
        validation_result dict
    """
    result = {
        'song_id': song_id,
        'status': 'ok',
        'issues': [],
        'stats': {}
    }
    
    try:
        with open(chordmap_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        events = data.get('events', [])
        key_changes = data.get('key_changes', [])
        
        # 基本統計
        result['stats']['num_events'] = len(events)
        result['stats']['num_key_changes'] = len(key_changes)
        
        # 空チェック
        if len(events) == 0:
            result['issues'].append({
                'type': 'empty_chordmap',
                'severity': 'warning',
                'message': 'No chord events found'
            })
            result['status'] = 'warning'
            return result
        
        # 時間順序チェック
        times = [e['time'] for e in events]
        if times != sorted(times):
            result['issues'].append({
                'type': 'time_order',
                'severity': 'error',
                'message': 'Events are not in chronological order'
            })
            result['status'] = 'error'
        
        # 最短持続チェック（2.0 QL想定）
        min_hold_violations = []
        for i in range(len(events) - 1):
            duration = events[i + 1]['time'] - events[i]['time']
            if duration < 2.0:
                min_hold_violations.append({
                    'index': i,
                    'time': events[i]['time'],
                    'duration': duration
                })
        
        if min_hold_violations:
            result['issues'].append({
                'type': 'min_hold_violation',
                'severity': 'warning',
                'message': f'{len(min_hold_violations)} events violate min_hold_ql=2.0',
                'examples': min_hold_violations[:5]
            })
            if result['status'] == 'ok':
                result['status'] = 'warning'
        
        # コード品質チェック
        root_counter = Counter()
        quality_counter = Counter()
        tension_counter = Counter()
        confidence_values = []
        
        for event in events:
            root = event.get('root', 'unknown')
            quality = event.get('quality', '')
            tensions = event.get('tensions', [])
            confidence = event.get('confidence', 0.0)
            
            root_counter[root] += 1
            quality_counter[quality] += 1
            for t in tensions:
                tension_counter[t] += 1
            confidence_values.append(confidence)
        
        result['stats']['root_distribution'] = dict(root_counter.most_common(5))
        result['stats']['quality_distribution'] = dict(quality_counter)
        result['stats']['tension_distribution'] = dict(tension_counter)
        result['stats']['avg_confidence'] = sum(confidence_values) / len(confidence_values) if confidence_values else 0.0
        result['stats']['min_confidence'] = min(confidence_values) if confidence_values else 0.0
        
        # 無効コードチェック（quality空 + confidence低）
        invalid_chords = []
        low_confidence_chords = []
        empty_quality_chords = []
        
        for i, event in enumerate(events):
            root = event.get('root', 'unknown')
            quality = event.get('quality', '')
            confidence = event.get('confidence', 0.0)
            
            # quality空文字列のケース
            if quality == '':
                empty_quality_chords.append({
                    'index': i,
                    'time': event.get('time'),
                    'confidence': confidence,
                    'root': root
                })
            
            # confidence低（<0.3）のケース
            if confidence < 0.3:
                low_confidence_chords.append({
                    'index': i,
                    'time': event.get('time'),
                    'confidence': confidence,
                    'chord': f"{root}{quality}"
                })
            
            # 真の無効（rootがunknown）
            if root == 'unknown':
                invalid_chords.append({
                    'index': i,
                    'time': event.get('time'),
                    'event': event
                })
        
        # Empty qualityは警告レベル
        if empty_quality_chords:
            result['issues'].append({
                'type': 'empty_quality',
                'severity': 'warning',
                'message': f'{len(empty_quality_chords)} events have empty quality (should default to "maj")',
                'examples': empty_quality_chords[:5]
            })
            if result['status'] == 'ok':
                result['status'] = 'warning'
        
        # Low confidenceも警告レベル
        if low_confidence_chords:
            result['issues'].append({
                'type': 'low_confidence',
                'severity': 'warning',
                'message': f'{len(low_confidence_chords)} events have confidence < 0.3',
                'examples': low_confidence_chords[:5]
            })
            if result['status'] == 'ok':
                result['status'] = 'warning'
        
        # 真の無効のみエラー
        if invalid_chords:
            result['issues'].append({
                'type': 'invalid_chords',
                'severity': 'error',
                'message': f'{len(invalid_chords)} truly invalid chord events (root=unknown)',
                'examples': invalid_chords[:3]
            })
            result['status'] = 'error'
        
        # コード進行自然性チェック
        progression_types = Counter()
        for i in range(len(events) - 1):
            root1 = events[i].get('root')
            root2 = events[i + 1].get('root')
            if root1 and root2:
                prog_type = analyze_chord_progression(root1, root2)
                progression_types[prog_type] += 1
        
        result['stats']['progression_distribution'] = dict(progression_types)
        
        # V→I率（妥当性の指標）
        total_progressions = sum(progression_types.values())
        vi_count = progression_types.get('V-I', 0)
        result['stats']['vi_ratio'] = vi_count / total_progressions if total_progressions > 0 else 0.0
        
    except Exception as e:
        result['status'] = 'error'
        result['issues'].append({
            'type': 'exception',
            'severity': 'error',
            'message': str(e)
        })
    
    return result


# ============================================================
# データセット全体の検証
# ============================================================

def validate_dataset(input_root: Path, sample_size: int = None) -> pd.DataFrame:
    """
    データセット全体を検証
    
    Args:
        input_root: midi_guide root
        sample_size: サンプリング数（Noneで全曲）
    
    Returns:
        validation_results DataFrame
    """
    # 曲フォルダ収集
    song_dirs = [d for d in input_root.iterdir() if d.is_dir()]
    
    if sample_size and sample_size < len(song_dirs):
        song_dirs = random.sample(song_dirs, sample_size)
    
    print(f"Validating {len(song_dirs)} songs...")
    
    results = []
    
    for song_dir in tqdm(song_dirs, desc="Validating"):
        song_id = song_dir.name
        chordmap_path = song_dir / 'chordmap.json'
        
        if not chordmap_path.exists():
            results.append({
                'song_id': song_id,
                'status': 'error',
                'issues': [{'type': 'missing_file', 'severity': 'error', 'message': 'chordmap.json not found'}],
                'stats': {}
            })
            continue
        
        result = validate_chordmap(chordmap_path, song_id)
        results.append(result)
    
    return pd.DataFrame(results)


# ============================================================
# サマリーレポート生成
# ============================================================

def generate_summary_report(df: pd.DataFrame, output_path: Path):
    """サマリーレポートを生成"""
    
    report = []
    report.append("=" * 80)
    report.append("ChordMap Quality Validation Report")
    report.append("=" * 80)
    report.append("")
    
    # 全体統計
    report.append(f"Total songs: {len(df)}")
    report.append(f"Status distribution:")
    status_counts = df['status'].value_counts()
    for status, count in status_counts.items():
        report.append(f"  {status}: {count} ({count/len(df)*100:.1f}%)")
    report.append("")
    
    # 空ChordMap
    empty_songs = df[df['stats'].apply(lambda x: x.get('num_events', 0) == 0)]
    report.append(f"Empty chordmaps (0 events): {len(empty_songs)}")
    if len(empty_songs) > 0:
        report.append("  Examples:")
        for idx, row in empty_songs.head(5).iterrows():
            report.append(f"    - {row['song_id']}")
    report.append("")
    
    # 平均統計
    valid_songs = df[df['stats'].apply(lambda x: x.get('num_events', 0) > 0)]
    if len(valid_songs) > 0:
        avg_events = valid_songs['stats'].apply(lambda x: x.get('num_events', 0)).mean()
        avg_confidence = valid_songs['stats'].apply(lambda x: x.get('avg_confidence', 0)).mean()
        avg_vi_ratio = valid_songs['stats'].apply(lambda x: x.get('vi_ratio', 0)).mean()
        
        report.append("Average statistics (non-empty songs):")
        report.append(f"  Chord events per song: {avg_events:.1f}")
        report.append(f"  Average confidence: {avg_confidence:.3f}")
        report.append(f"  V→I progression ratio: {avg_vi_ratio:.3f}")
        report.append("")
    
    # 品質分布
    report.append("Quality distribution (all songs):")
    all_qualities = Counter()
    for stats in df['stats']:
        qual_dist = stats.get('quality_distribution', {})
        for q, count in qual_dist.items():
            all_qualities[q] += count
    
    for quality, count in all_qualities.most_common(10):
        report.append(f"  {quality}: {count}")
    report.append("")
    
    # テンション分布
    report.append("Tension distribution (all songs):")
    all_tensions = Counter()
    for stats in df['stats']:
        tension_dist = stats.get('tension_distribution', {})
        for t, count in tension_dist.items():
            all_tensions[t] += count
    
    if all_tensions:
        for tension, count in all_tensions.most_common(5):
            report.append(f"  {tension}: {count}")
    else:
        report.append("  (No tensions found)")
    report.append("")
    
    # イシュー集計
    report.append("Issue summary:")
    issue_types = Counter()
    for issues in df['issues']:
        for issue in issues:
            issue_types[issue['type']] += 1
    
    for issue_type, count in issue_types.most_common():
        report.append(f"  {issue_type}: {count}")
    report.append("")
    
    report.append("=" * 80)
    
    # ファイル書き込み
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))
    
    print(f"Summary report saved to: {output_path}")
    
    # コンソール出力
    print('\n'.join(report))


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Validate ChordMap quality for LAMDA MIDI dataset"
    )
    
    parser.add_argument(
        '--input-root',
        type=Path,
        required=True,
        help='Input root directory (midi_guide)'
    )
    parser.add_argument(
        '--sample',
        type=int,
        default=None,
        help='Sample size for validation (default: all songs)'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('output/validation'),
        help='Output directory for reports (default: output/validation)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for sampling (default: 42)'
    )
    
    args = parser.parse_args()
    
    # ランダムシード設定
    random.seed(args.seed)
    
    # 出力ディレクトリ作成
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # 検証実行
    df = validate_dataset(args.input_root, args.sample)
    
    # 結果保存
    results_path = args.output_dir / 'validation_results.csv'
    df.to_csv(results_path, index=False)
    print(f"Detailed results saved to: {results_path}")
    
    # サマリーレポート生成
    summary_path = args.output_dir / 'validation_summary.txt'
    generate_summary_report(df, summary_path)
    
    # エラー曲リスト
    error_songs = df[df['status'] == 'error']
    if len(error_songs) > 0:
        error_path = args.output_dir / 'error_songs.txt'
        with open(error_path, 'w', encoding='utf-8') as f:
            for song_id in error_songs['song_id']:
                f.write(f"{song_id}\n")
        print(f"Error songs list saved to: {error_path}")
    
    # 終了コード
    if len(error_songs) > 0:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == '__main__':
    main()
