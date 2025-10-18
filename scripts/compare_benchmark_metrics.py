#!/usr/bin/env python3
"""
compare_benchmark_metrics.py - ベンチマーク品質メトリクス比較スクリプト

Before/AfterのMIDIファイルを比較し、メトリクス差分を計算します。

Usage:
    python scripts/compare_benchmark_metrics.py --before before.mid --after after.mid
    python scripts/compare_benchmark_metrics.py --suite multi_song_benchmark.json
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import mido
import numpy as np


def load_midi(midi_path: Path) -> mido.MidiFile:
    """MIDIファイル読み込み"""
    return mido.MidiFile(midi_path)


def extract_notes_from_track(track: mido.MidiTrack) -> List[Dict[str, Any]]:
    """トラックからノート情報を抽出"""
    notes = []
    current_time = 0
    
    for msg in track:
        current_time += msg.time
        
        if msg.type == 'note_on' and msg.velocity > 0:
            notes.append({
                'time': current_time,
                'pitch': msg.note,
                'velocity': msg.velocity,
                'channel': msg.channel,
            })
    
    return notes


def calculate_basic_metrics(notes: List[Dict[str, Any]]) -> Dict[str, float]:
    """基本メトリクス計算"""
    if not notes:
        return {
            'note_count': 0,
            'pitch_mean': 0.0,
            'pitch_std': 0.0,
            'pitch_range': 0.0,
            'velocity_mean': 0.0,
            'velocity_std': 0.0,
            'note_density': 0.0,
        }
    
    pitches = [n['pitch'] for n in notes]
    velocities = [n['velocity'] for n in notes]
    times = [n['time'] for n in notes]
    
    total_duration = max(times) - min(times) if len(times) > 1 else 1
    
    metrics = {
        'note_count': len(notes),
        'pitch_mean': float(np.mean(pitches)),
        'pitch_std': float(np.std(pitches)),
        'pitch_range': float(max(pitches) - min(pitches)),
        'velocity_mean': float(np.mean(velocities)),
        'velocity_std': float(np.std(velocities)),
        'note_density': len(notes) / (total_duration / 480.0),  # notes per beat (480 ticks/beat)
    }
    
    return metrics


def compare_metrics(before_metrics: Dict[str, float], after_metrics: Dict[str, float]) -> Dict[str, Any]:
    """メトリクス比較と差分計算"""
    comparison = {}
    
    for key in before_metrics.keys():
        before_val = before_metrics[key]
        after_val = after_metrics[key]
        
        # 差分計算
        absolute_diff = after_val - before_val
        
        # パーセンテージ変化
        if before_val != 0:
            percent_change = (absolute_diff / before_val) * 100.0
        else:
            percent_change = 0.0 if after_val == 0 else float('inf')
        
        comparison[key] = {
            'before': before_val,
            'after': after_val,
            'absolute_diff': absolute_diff,
            'percent_change': percent_change,
        }
    
    return comparison


def analyze_midi_file(midi_path: Path) -> Dict[str, Any]:
    """MIDIファイル全体を分析"""
    midi = load_midi(midi_path)
    
    # 全トラックからノート抽出
    all_notes = []
    track_metrics = []
    
    for i, track in enumerate(midi.tracks):
        notes = extract_notes_from_track(track)
        
        if notes:
            all_notes.extend(notes)
            metrics = calculate_basic_metrics(notes)
            track_metrics.append({
                'track_index': i,
                'track_name': track.name if hasattr(track, 'name') else f'Track {i}',
                'metrics': metrics,
            })
    
    # 全体メトリクス
    overall_metrics = calculate_basic_metrics(all_notes)
    
    return {
        'file': str(midi_path.name),
        'track_count': len(midi.tracks),
        'active_tracks': len(track_metrics),
        'overall_metrics': overall_metrics,
        'track_metrics': track_metrics,
    }


def compare_two_files(before_path: Path, after_path: Path, output_path: Optional[Path] = None) -> Dict[str, Any]:
    """2つのMIDIファイルを比較"""
    
    print(f"📊 Comparing MIDI files...")
    print(f"   Before: {before_path.name}")
    print(f"   After:  {after_path.name}")
    
    # 各ファイルを分析
    before_analysis = analyze_midi_file(before_path)
    after_analysis = analyze_midi_file(after_path)
    
    # 全体メトリクス比較
    overall_comparison = compare_metrics(
        before_analysis['overall_metrics'],
        after_analysis['overall_metrics']
    )
    
    # 結果サマリー
    comparison_result = {
        'before': before_analysis,
        'after': after_analysis,
        'comparison': overall_comparison,
    }
    
    # 出力
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(comparison_result, f, indent=2, ensure_ascii=False)
        
        print(f"\n✅ Comparison saved to: {output_path}")
    
    # コンソール表示
    print("\n📈 Metric Changes:")
    for metric_name, values in overall_comparison.items():
        before_val = values['before']
        after_val = values['after']
        percent = values['percent_change']
        
        symbol = '🔼' if percent > 0 else '🔽' if percent < 0 else '➖'
        print(f"   {symbol} {metric_name}: {before_val:.2f} → {after_val:.2f} ({percent:+.1f}%)")
    
    return comparison_result


def main():
    parser = argparse.ArgumentParser(
        description='Compare benchmark MIDI metrics (before/after)'
    )
    parser.add_argument(
        '--before',
        type=str,
        help='Before MIDI file path'
    )
    parser.add_argument(
        '--after',
        type=str,
        help='After MIDI file path'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='benchmark_comparison.json',
        help='Output JSON file path (default: benchmark_comparison.json)'
    )
    
    args = parser.parse_args()
    
    # パス解決
    project_root = Path(__file__).parent.parent
    
    if not args.before or not args.after:
        print("❌ Error: Both --before and --after arguments are required", file=sys.stderr)
        parser.print_help()
        sys.exit(1)
    
    before_path = Path(args.before)
    after_path = Path(args.after)
    output_path = project_root / args.output
    
    if not before_path.exists():
        print(f"❌ Before file not found: {before_path}", file=sys.stderr)
        sys.exit(1)
    
    if not after_path.exists():
        print(f"❌ After file not found: {after_path}", file=sys.stderr)
        sys.exit(1)
    
    # 比較実行
    compare_two_files(before_path, after_path, output_path)


if __name__ == '__main__':
    main()
