#!/usr/bin/env python3
"""
MIDI統計分析ツール

詳細なMIDI統計を生成:
1. トラック別note density
2. Velocity分布
3. Timing variance（humanize効果検証）
4. セクション別分析（bars.parquet統合）

使用例:
    python3 scripts/analyze_midi_stats.py \
        --midi song_packages/suno_project/song_001/full_arrangement_6tracks_real.mid \
        --bars-parquet song_packages/suno_project/song_001/bars.parquet \
        --output song_packages/suno_project/song_001/midi_analysis.json
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
from collections import defaultdict

try:
    from mido import MidiFile
    MIDO_AVAILABLE = True
except ImportError:
    MIDO_AVAILABLE = False

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

try:
    import pretty_midi as pm
    PRETTYMIDI_AVAILABLE = True
except ImportError:
    PRETTYMIDI_AVAILABLE = False


# ========== jSymbolic系MIDI統計特徴（軽量版） ==========
def extract_jsymbolic_like_features(midi_path: Path) -> Dict:
    """jSymbolic2参照の体系的MIDI特徴抽出（学習＆監視向け）
    
    jSymbolic2の200+項目から代表的な特徴を軽量実装。
    KPI合否には未使用（参考指標のみ）。
    
    特徴カテゴリ（jSymbolic2準拠）:
        - P (Pitch): 音高範囲、分散、最頻音など
        - R (Rhythm): 密度、IOI（Inter-Onset Interval）統計
        - D (Dynamics): Velocity分散、ダイナミックレンジ
        - H (Harmony): 垂直音程エントロピー（簡易版）
    
    研究背景:
        jSymbolic2 (McKay et al., 2018) - 体系的なMIDI特徴設計の参考標準
        ISMIR/MIREX評価用特徴セットの実務簡易版
    
    Args:
        midi_path: MIDIファイルパス
    
    Returns:
        {
            "duration_sec": float,
            "note_count": int,
            "pitch_range": int,           # P-1相当
            "pitch_var": float,            # P-3相当
            "velocity_var": float,         # D-2相当
            "dynamic_range": float,        # D-1相当
            "density_notes_per_sec": float,# R-1相当
            "ioi_mean": float,             # R-5相当
            "ioi_var": float,              # R-6相当
        }
    """
    if not PRETTYMIDI_AVAILABLE:
        return {
            "error": "pretty_midi not available",
            "duration_sec": 0.0
        }
    
    try:
        m = pm.PrettyMIDI(str(midi_path))
        
        # 全楽器のノート統合
        notes = [n for inst in m.instruments for n in inst.notes]
        
        if not notes:
            return {"duration_sec": 0.0, "note_count": 0}
        
        dur = m.get_end_time()
        pitches = [n.pitch for n in notes]
        vels = [n.velocity for n in notes]
        onsets = sorted([n.start for n in notes])
        
        # Inter-Onset Interval（連続ノート間隔）
        ioi = np.diff(onsets) if len(onsets) > 1 else np.array([0.0])
        
        # Pitch統計（jSymbolic P系）
        pitch_range = int(max(pitches) - min(pitches)) if pitches else 0
        pitch_var = float(np.var(pitches)) if pitches else 0.0
        pitch_mean = float(np.mean(pitches)) if pitches else 0.0
        
        # Dynamics統計（jSymbolic D系）
        velocity_var = float(np.var(vels)) if vels else 0.0
        dynamic_range = float(max(vels) - min(vels)) if vels else 0.0
        velocity_mean = float(np.mean(vels)) if vels else 0.0
        
        # Rhythm統計（jSymbolic R系）
        density_notes_per_sec = len(notes) / dur if dur > 0 else 0.0
        ioi_mean = float(np.mean(ioi)) if len(ioi) > 0 else 0.0
        ioi_var = float(np.var(ioi)) if len(ioi) > 0 else 0.0
        ioi_std = float(np.std(ioi)) if len(ioi) > 0 else 0.0
        
        # 簡易Harmony統計（垂直音程の複雑度指標、R系の補助）
        # 同時刻ノート数の最大値（ポリフォニー度）
        polyphony_max = 0
        if notes:
            time_bins = np.arange(0, dur, 0.1)  # 100msビン
            for t in time_bins:
                active = sum(1 for n in notes if n.start <= t < n.end)
                polyphony_max = max(polyphony_max, active)
        
        return {
            # Metadata
            "duration_sec": float(dur),
            "note_count": len(notes),
            
            # Pitch（jSymbolic P系）
            "pitch_range": pitch_range,           # P-1: Pitch Range
            "pitch_var": pitch_var,                # P-3: Pitch Variance
            "pitch_mean": pitch_mean,              # P-2: Mean Pitch
            
            # Dynamics（jSymbolic D系）
            "velocity_var": velocity_var,          # D-2: Velocity Variance
            "velocity_mean": velocity_mean,        # D-1: Mean Velocity
            "dynamic_range": dynamic_range,        # D-3: Dynamic Range
            
            # Rhythm（jSymbolic R系）
            "density_notes_per_sec": density_notes_per_sec,  # R-1: Note Density
            "ioi_mean": ioi_mean,                  # R-5: Mean IOI
            "ioi_var": ioi_var,                    # R-6: IOI Variance
            "ioi_std": ioi_std,                    # R-7: IOI Std Dev
            
            # Harmony（簡易版）
            "polyphony_max": int(polyphony_max),   # 最大同時発音数
            
            # Method
            "method": "jsymbolic_like_lightweight",
            "reference": "jSymbolic2 (McKay et al., 2018)"
        }
    
    except Exception as e:
        return {
            "error": str(e),
            "duration_sec": 0.0
        }


def extract_track_stats(track, track_name: str, ppq: int) -> Dict:
    """トラック統計抽出"""
    notes = []
    note_ons = {}
    
    current_tick = 0
    for msg in track:
        current_tick += msg.time
        
        if msg.type == 'note_on' and msg.velocity > 0:
            note_ons[msg.note] = {'tick': current_tick, 'vel': msg.velocity}
        elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
            if msg.note in note_ons:
                on_event = note_ons.pop(msg.note)
                duration = current_tick - on_event['tick']
                notes.append({
                    'tick': on_event['tick'],
                    'pitch': msg.note,
                    'vel': on_event['vel'],
                    'dur': duration
                })
    
    if not notes:
        return {
            'track_name': track_name,
            'note_count': 0,
            'velocity_mean': 0,
            'velocity_std': 0,
            'velocity_min': 0,
            'velocity_max': 0,
            'duration_mean_ms': 0,
            'duration_std_ms': 0,
            'timing_variance_ms': 0,
            'density_notes_per_bar': 0
        }
    
    velocities = [n['vel'] for n in notes]
    durations_ms = [n['dur'] / ppq * 500 for n in notes]  # 仮定：120 BPM
    
    # Timing variance（連続ノート間隔の標準偏差）
    if len(notes) > 1:
        intervals = [notes[i+1]['tick'] - notes[i]['tick'] for i in range(len(notes)-1)]
        intervals_ms = [iv / ppq * 500 for iv in intervals]
        timing_variance = np.std(intervals_ms)
    else:
        timing_variance = 0.0
    
    # Density（仮定：150小節、4/4）
    total_bars = 150
    density = len(notes) / total_bars
    
    return {
        'track_name': track_name,
        'note_count': len(notes),
        'velocity_mean': float(np.mean(velocities)),
        'velocity_std': float(np.std(velocities)),
        'velocity_min': int(np.min(velocities)),
        'velocity_max': int(np.max(velocities)),
        'duration_mean_ms': float(np.mean(durations_ms)),
        'duration_std_ms': float(np.std(durations_ms)),
        'timing_variance_ms': float(timing_variance),
        'density_notes_per_bar': float(density)
    }


def extract_section_stats(
    midi_path: Path,
    bars_parquet_path: Optional[Path],
    ppq: int
) -> Dict:
    """セクション別統計（bars.parquet統合）"""
    if bars_parquet_path is None or not PANDAS_AVAILABLE:
        return {}
    
    try:
        bars_df = pd.read_parquet(bars_parquet_path)
    except Exception:
        return {}
    
    # セクション別小節数（section_label列使用）
    section_col = 'section_label' if 'section_label' in bars_df.columns else 'section'
    energy_col = 'energy_curve' if 'energy_curve' in bars_df.columns else 'energy'
    
    section_counts = bars_df[section_col].value_counts().to_dict()
    
    # セクション別平均energy
    section_energy = bars_df.groupby(section_col)[energy_col].mean().to_dict()
    
    return {
        'section_counts': section_counts,
        'section_energy': section_energy
    }


def analyze_midi(
    midi_path: Path,
    bars_parquet_path: Optional[Path] = None,
    verbose: bool = True
) -> Dict:
    """MIDI統計分析"""
    if not MIDO_AVAILABLE:
        raise ImportError("mido is required. Install with: pip install mido")
    
    if verbose:
        print(f"📖 Loading MIDI: {midi_path}")
    
    mid = MidiFile(midi_path)
    ppq = mid.ticks_per_beat
    
    if verbose:
        print(f"   Tracks: {len(mid.tracks)}, PPQ: {ppq}")
    
    # トラック別統計
    track_names = ['Tempo', 'Bass', 'Guitar', 'Piano', 'Strings', 'Drums']
    track_stats = []
    
    for i, track in enumerate(mid.tracks):
        name = track_names[i] if i < len(track_names) else f'Track_{i}'
        stats = extract_track_stats(track, name, ppq)
        track_stats.append(stats)
        
        if verbose and stats['note_count'] > 0:
            print(f"   {name:10s}: {stats['note_count']:4d} notes, "
                  f"vel={stats['velocity_mean']:.1f}±{stats['velocity_std']:.1f}, "
                  f"timing_var={stats['timing_variance_ms']:.1f}ms")
    
    # 全体統計
    total_notes = sum(s['note_count'] for s in track_stats)
    duration_sec = mid.length
    
    # セクション別統計
    section_stats = extract_section_stats(midi_path, bars_parquet_path, ppq)
    
    return {
        'metadata': {
            'midi_path': str(midi_path),
            'tracks': len(mid.tracks),
            'ppq': ppq,
            'duration_sec': duration_sec,
            'duration_min': duration_sec / 60,
            'total_notes': total_notes
        },
        'track_stats': track_stats,
        'section_stats': section_stats
    }


def main():
    parser = argparse.ArgumentParser(description='Analyze MIDI statistics')
    parser.add_argument('--midi', type=Path, required=True, help='Path to MIDI file')
    parser.add_argument(
        '--bars-parquet',
        type=Path,
        default=None,
        help='Optional bars.parquet for section-aware analysis'
    )
    parser.add_argument('--output', type=Path, required=True, help='Path to output JSON')
    parser.add_argument('--quiet', action='store_true', help='Suppress verbose output')
    parser.add_argument(
        '--jsymbolic',
        action='store_true',
        help='Extract jSymbolic-like features (lightweight reference indicators)'
    )
    
    args = parser.parse_args()
    
    # jSymbolic特徴抽出（オプション）
    jsymbolic_features = None
    if args.jsymbolic:
        if not args.quiet:
            print(f"📊 Extracting jSymbolic-like features...")
        jsymbolic_features = extract_jsymbolic_like_features(args.midi)
        if not args.quiet and 'error' not in jsymbolic_features:
            print(f"   Note count: {jsymbolic_features['note_count']}")
            print(f"   Pitch range: {jsymbolic_features['pitch_range']}")
            print(f"   Density: {jsymbolic_features['density_notes_per_sec']:.2f} notes/sec")
            print(f"   IOI mean: {jsymbolic_features['ioi_mean']*1000:.1f}ms")
    
    stats = analyze_midi(args.midi, args.bars_parquet, verbose=not args.quiet)
    
    # jSymbolic特徴をstatsに統合
    if jsymbolic_features is not None:
        stats['jsymbolic_features'] = jsymbolic_features
    
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Saved analysis: {args.output}")
    if jsymbolic_features and 'error' not in jsymbolic_features:
        print(f"   jSymbolic features: {len([k for k in jsymbolic_features.keys() if not k.startswith('_')])} metrics")


if __name__ == '__main__':
    main()
