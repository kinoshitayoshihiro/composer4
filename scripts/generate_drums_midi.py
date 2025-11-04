#!/usr/bin/env python3
"""
Drums Generator - MIDI生成+ヒューマナイズ

drums_recommendations.jsonから推奨パターンを読み込み、MIDI生成:
1. drums_recommendations.json読み込み
2. 各小節のpattern_idからMIDI検索（rhythm_features_merged.parquet経由）
3. ヒューマナイズ適用（micro_timing, velocity_variance）
4. MIDI連結+書き出し → drums.mid出力

使用例:
    python3 scripts/generate_drums_midi.py \
        --recommendations song_packages/sample_project/sample_song/drums_recommendations.json \
        --output song_packages/sample_project/sample_song/drums.mid
"""

import argparse
import json
import pandas as pd
import numpy as np
import mido
from pathlib import Path
from typing import Dict, List, Optional


def load_recommendations(json_path: Path) -> dict:
    """drums_recommendations.json読み込み"""
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_rhythm_features(parquet_path: Path) -> pd.DataFrame:
    """rhythm_features_merged.parquet読み込み（未使用だが互換性のため残す）"""
    return pd.read_parquet(parquet_path)


def find_midi_path(loop_id: str, project_root: Path) -> Optional[Path]:
    """pattern_idからMIDIパス検索
    
    Args:
        loop_id: パターンID（e.g., "12_latin-brazilian-sambareggae_96_beat_4-4_1"）
        project_root: プロジェクトルート（composer2-3）
    
    Returns:
        MIDIファイルパス or None
    """
    # データセット種別推定（loop_idプレフィックスから）
    if loop_id.startswith('egmd_'):
        # E-GMD: output/rhythm_ai/egmd_cleaned/
        base_dir = project_root / 'output/rhythm_ai/egmd_cleaned'
        # egmd_000019.mid のような形式
        midi_filename = f"{loop_id}.mid"
        
        # サブディレクトリ検索（0/, 1/, ...）
        for subdir in base_dir.glob('*'):
            if subdir.is_dir():
                midi_path = subdir / midi_filename
                if midi_path.exists():
                    return midi_path
        
    elif '_' in loop_id:
        # groove or drumclean: output/rhythm_ai/groove_cleaned/ or output/rhythm_ai/drumclean_midi/
        # loop_id: "12_latin-brazilian-sambareggae_96_beat_4-4_1"
        # MIDIファイル名: "12_latin-brazilian-sambareggae_96_beat_4-4.mid"（サフィックス_1除去）
        
        # サフィックス除去（最後の_数字）
        parts = loop_id.rsplit('_', 1)
        if len(parts) == 2 and parts[1].isdigit():
            base_name = parts[0]
        else:
            base_name = loop_id
        
        # groove検索
        groove_dir = project_root / 'output/rhythm_ai/groove_cleaned'
        if groove_dir.exists():
            for midi_file in groove_dir.rglob('*.mid'):
                if base_name in midi_file.stem:
                    return midi_file
        
        # drumclean検索
        drumclean_dir = project_root / 'output/rhythm_ai/drumclean_midi'
        if drumclean_dir.exists():
            for midi_file in drumclean_dir.rglob('*.mid'):
                if base_name in midi_file.stem:
                    return midi_file
    
    return None


def load_midi_pattern(midi_path: Path) -> Optional[mido.MidiFile]:
    """MIDIパターン読み込み
    
    Args:
        midi_path: MIDIファイルパス
    
    Returns:
        MIDIファイルオブジェクト or None
    """
    try:
        return mido.MidiFile(midi_path)
    except Exception as e:
        print(f"   ⚠️  Failed to load MIDI: {midi_path} ({e})")
        return None


def apply_humanize(
    notes: List[Dict],
    micro_timing_ms: float = 10.0,
    velocity_variance: int = 5,
    seed: Optional[int] = None
) -> List[Dict]:
    """ヒューマナイズ適用
    
    Args:
        notes: ノートリスト（[{'time': tick, 'note': int, 'velocity': int, 'duration': tick}]）
        micro_timing_ms: マイクロタイミング範囲（±ms）
        velocity_variance: ベロシティ分散（±）
        seed: 乱数シード
    
    Returns:
        ヒューマナイズ後のノートリスト
    """
    if seed is not None:
        np.random.seed(seed)
    
    humanized = []
    
    for note in notes:
        # マイクロタイミング（±10ms = ±24 ticks at 480 tpb, 120 bpm）
        time_offset = int(np.random.uniform(-micro_timing_ms, micro_timing_ms) * 0.48)  # 仮定: 480 tpb
        
        # ベロシティ分散
        velocity_offset = int(np.random.uniform(-velocity_variance, velocity_variance))
        
        humanized.append({
            'time': max(0, note['time'] + time_offset),
            'note': note['note'],
            'velocity': np.clip(note['velocity'] + velocity_offset, 1, 127),
            'duration': note['duration'],
        })
    
    return humanized


def extract_notes_from_midi(midi_file: mido.MidiFile) -> List[Dict]:
    """MIDIファイルからノート抽出
    
    Args:
        midi_file: MIDIファイルオブジェクト
    
    Returns:
        ノートリスト（[{'time': tick, 'note': int, 'velocity': int, 'duration': tick}]）
    """
    notes = []
    current_time = 0
    note_on_events = {}
    
    # ドラムトラック検索（channel 9 or track名に"drum"）
    for track in midi_file.tracks:
        current_time = 0
        
        for msg in track:
            current_time += msg.time
            
            if msg.type == 'note_on' and msg.velocity > 0:
                note_on_events[msg.note] = {
                    'time': current_time,
                    'velocity': msg.velocity,
                }
            
            elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                if msg.note in note_on_events:
                    note_on = note_on_events.pop(msg.note)
                    notes.append({
                        'time': note_on['time'],
                        'note': msg.note,
                        'velocity': note_on['velocity'],
                        'duration': current_time - note_on['time'],
                    })
    
    return notes


def create_midi_from_notes(
    notes: List[Dict],
    ticks_per_beat: int = 480,
    tempo: int = 500000  # 120 bpm
) -> mido.MidiFile:
    """ノートリストからMIDI生成
    
    Args:
        notes: ノートリスト
        ticks_per_beat: ティック/ビート
        tempo: テンポ（マイクロ秒/ビート）
    
    Returns:
        MIDIファイルオブジェクト
    """
    midi = mido.MidiFile(ticks_per_beat=ticks_per_beat)
    track = mido.MidiTrack()
    midi.tracks.append(track)
    
    # テンポ設定
    track.append(mido.MetaMessage('set_tempo', tempo=tempo, time=0))
    
    # トラック名
    track.append(mido.MetaMessage('track_name', name='Drums', time=0))
    
    # ノートをtime順にソート
    sorted_notes = sorted(notes, key=lambda x: x['time'])
    
    # イベント化
    events = []
    for note in sorted_notes:
        events.append({
            'time': note['time'],
            'type': 'note_on',
            'note': note['note'],
            'velocity': note['velocity'],
        })
        events.append({
            'time': note['time'] + note['duration'],
            'type': 'note_off',
            'note': note['note'],
            'velocity': 0,
        })
    
    # timeでソート
    events.sort(key=lambda x: x['time'])
    
    # デルタタイム計算
    current_time = 0
    for event in events:
        delta = event['time'] - current_time
        
        if event['type'] == 'note_on':
            track.append(mido.Message('note_on', 
                                     note=event['note'], 
                                     velocity=event['velocity'], 
                                     time=delta, 
                                     channel=9))  # channel 9 = drums
        else:
            track.append(mido.Message('note_off', 
                                     note=event['note'], 
                                     velocity=0, 
                                     time=delta, 
                                     channel=9))
        
        current_time = event['time']
    
    # End of track
    track.append(mido.MetaMessage('end_of_track', time=0))
    
    return midi


def generate_drums_midi(
    recommendations_path: Path,
    output_path: Path,
    micro_timing_ms: float = 10.0,
    velocity_variance: int = 5,
    verbose: bool = True
):
    """ドラムMIDI生成メイン処理
    
    Args:
        recommendations_path: drums_recommendations.jsonパス
        output_path: drums.mid出力パス
        micro_timing_ms: マイクロタイミング範囲（±ms）
        velocity_variance: ベロシティ分散（±）
        verbose: 詳細出力
    """
    # recommendations読み込み
    if verbose:
        print(f"📖 Loading recommendations: {recommendations_path}")
    
    recommendations = load_recommendations(recommendations_path)
    
    if verbose:
        print(f"   Total bars: {len(recommendations)}")
    
    # プロジェクトルート検出
    current = recommendations_path.parent
    while current.name != 'composer2-3' and current.parent != current:
        current = current.parent
    project_root = current
    
    if verbose:
        print(f"   Project root: {project_root}")
    
    # MIDI生成
    all_notes = []
    ticks_per_bar = 1920  # 4/4, 480 tpb → 1920 ticks/bar
    
    # 新形式対応（recommendations配列 or 旧形式bar_*ディクショナリ）
    if 'recommendations' in recommendations:
        # 新形式: {"metadata": {...}, "recommendations": [...]}
        bars_list = recommendations['recommendations']
    elif any(k.startswith('bar_') for k in recommendations.keys()):
        # 旧形式: {"bar_0": {...}, "bar_1": {...}}
        bars_dict = {k: v for k, v in recommendations.items() if k.startswith('bar_')}
        bar_keys = sorted(bars_dict.keys(), key=lambda x: int(x.split('_')[1]))
        bars_list = [bars_dict[k] for k in bar_keys]
    else:
        raise ValueError("Invalid recommendations format")
    
    for bar_data in bars_list:
        bar_idx = bar_data.get('bar', bar_data.get('bar_index', 0))
        pattern_id = bar_data.get('pattern_id', bar_data.get('pattern', {}).get('pattern_id'))
        
        # MIDIパス検索
        midi_path = find_midi_path(pattern_id, project_root)
        
        if midi_path is None:
            if verbose:
                print(f"   ⚠️  Bar {bar_idx}: MIDI not found for {pattern_id}")
            continue
        
        # 絶対パス化
        if not midi_path.is_absolute():
            midi_path = project_root / midi_path
        
        # MIDI読み込み
        midi_file = load_midi_pattern(midi_path)
        
        if midi_file is None:
            continue
        
        # ノート抽出
        notes = extract_notes_from_midi(midi_file)
        
        # ヒューマナイズ
        notes = apply_humanize(notes, micro_timing_ms, velocity_variance, seed=bar_idx)
        
        # 時間オフセット（小節位置）
        bar_offset = bar_idx * ticks_per_bar
        for note in notes:
            note['time'] += bar_offset
        
        all_notes.extend(notes)
    
    if verbose:
        print(f"\n📊 Generation Statistics:")
        print(f"   Total notes: {len(all_notes)}")
        print(f"   Time range: 0 .. {max([n['time'] for n in all_notes]) if all_notes else 0} ticks")
    
    # MIDI書き出し
    midi = create_midi_from_notes(all_notes, ticks_per_beat=480, tempo=789473)  # 76.01 bpm
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    midi.save(output_path)
    
    if verbose:
        print(f"\n✅ Saved MIDI: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Generate drums MIDI from recommendations'
    )
    parser.add_argument(
        '--recommendations',
        type=Path,
        required=True,
        help='Path to drums_recommendations.json'
    )
    parser.add_argument(
        '--output',
        type=Path,
        required=True,
        help='Path to output drums.mid'
    )
    parser.add_argument(
        '--micro-timing',
        type=float,
        default=10.0,
        help='Micro timing range (±ms, default: 10.0)'
    )
    parser.add_argument(
        '--velocity-variance',
        type=int,
        default=5,
        help='Velocity variance (±, default: 5)'
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress verbose output'
    )
    
    args = parser.parse_args()
    
    generate_drums_midi(
        args.recommendations,
        args.output,
        micro_timing_ms=args.micro_timing,
        velocity_variance=args.velocity_variance,
        verbose=not args.quiet
    )


if __name__ == '__main__':
    main()
