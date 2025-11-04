#!/usr/bin/env python3
"""
generate_manifest.py
--------------------
成果物マニフェスト生成

全Plan/MIDIファイルのSHA256、作成時刻、メタデータを記録し、
再現性・トレーサビリティ向上を図る。

Usage:
    python3 scripts/generate_manifest.py \
        --song-dir song_packages/suno_project/song_001 \
        --output song_packages/suno_project/song_001/full_arrangement.manifest.json
"""

import argparse
import hashlib
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List

try:
    from mido import MidiFile
    MIDO_AVAILABLE = True
except ImportError:
    MIDO_AVAILABLE = False


def compute_sha256(file_path: Path) -> str:
    """ファイルのSHA256ハッシュ計算"""
    sha256 = hashlib.sha256()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b''):
            sha256.update(chunk)
    return sha256.hexdigest()


def get_file_metadata(file_path: Path, base_dir: Path) -> Dict[str, Any]:
    """ファイルメタデータ取得"""
    stat = file_path.stat()
    
    # 相対パス計算（失敗時は絶対パス）
    try:
        rel_path = str(file_path.relative_to(base_dir.resolve()))
    except ValueError:
        rel_path = str(file_path)
    
    return {
        'path': rel_path,
        'size_bytes': stat.st_size,
        'created': datetime.fromtimestamp(stat.st_ctime).isoformat(),
        'modified': datetime.fromtimestamp(stat.st_mtime).isoformat(),
        'sha256': compute_sha256(file_path)
    }


def analyze_plan(plan_path: Path) -> Dict[str, Any]:
    """Plan JSON分析"""
    try:
        data = json.loads(plan_path.read_text(encoding='utf-8'))
        tracks = data.get('tracks', [])
        total_events = sum(len(tr.get('events', [])) for tr in tracks)
        
        return {
            'ppq': data.get('ppq'),
            'tempo_bpm': data.get('tempo_bpm'),
            'tracks': len(tracks),
            'total_events': total_events,
            'track_names': [tr.get('name', 'Unknown') for tr in tracks]
        }
    except Exception as e:
        return {'error': str(e)}


def analyze_midi(midi_path: Path) -> Dict[str, Any]:
    """MIDI分析"""
    if not MIDO_AVAILABLE:
        return {'error': 'mido not available'}
    
    try:
        mid = MidiFile(midi_path)
        total_notes = sum(
            1 for tr in mid.tracks 
            for msg in tr 
            if msg.type == 'note_on' and msg.velocity > 0
        )
        
        return {
            'ppq': mid.ticks_per_beat,
            'tracks': len(mid.tracks),
            'total_notes': total_notes,
            'duration_sec': round(mid.length, 1),
            'duration_min': round(mid.length / 60, 2)
        }
    except Exception as e:
        return {'error': str(e)}


def generate_manifest(
    song_dir: Path,
    output_path: Path,
    verbose: bool = True
) -> Dict[str, Any]:
    """マニフェスト生成"""
    if verbose:
        print(f"📋 Generating manifest for: {song_dir}")
    
    # 絶対パス化
    song_dir = song_dir.resolve()
    cwd = Path.cwd().resolve()
    
    try:
        rel_song_dir = str(song_dir.relative_to(cwd))
    except ValueError:
        rel_song_dir = str(song_dir)
    
    manifest = {
        'generated_at': datetime.now().isoformat(),
        'song_dir': rel_song_dir,
        'files': {}
    }
    
    # Plan JSONファイル検索
    plan_files = [
        'bass_plan.json',
        'guitar_plan.json',
        'piano_plan.json',
        'strings_plan.json',
        'drums_plan.json',
        'drums_plan_real.json',
        'full_arrangement.json'
    ]
    
    for plan_name in plan_files:
        plan_path = song_dir / plan_name
        if plan_path.exists():
            if verbose:
                print(f"   Analyzing: {plan_name}")
            
            file_meta = get_file_metadata(plan_path, cwd)
            plan_analysis = analyze_plan(plan_path)
            
            manifest['files'][plan_name] = {
                **file_meta,
                'type': 'plan',
                'analysis': plan_analysis
            }
    
    # MIDIファイル検索
    midi_files = [
        'full_arrangement.mid',
        'full_arrangement_6tracks_real.mid',
        'drums.mid'
    ]
    
    for midi_name in midi_files:
        midi_path = song_dir / midi_name
        if midi_path.exists():
            if verbose:
                print(f"   Analyzing: {midi_name}")
            
            file_meta = get_file_metadata(midi_path, cwd)
            midi_analysis = analyze_midi(midi_path)
            
            manifest['files'][midi_name] = {
                **file_meta,
                'type': 'midi',
                'analysis': midi_analysis
            }
    
    # その他重要ファイル
    other_files = [
        'song_package.yaml',
        'bars.parquet',
        'chordmap.json',
        'drums_recommendations.json',
        'kpi_gate_report.json',
        'midi_analysis.json'
    ]
    
    for other_name in other_files:
        other_path = song_dir / other_name
        if other_path.exists():
            if verbose:
                print(f"   Adding: {other_name}")
            
            file_meta = get_file_metadata(other_path, cwd)
            manifest['files'][other_name] = {
                **file_meta,
                'type': 'metadata'
            }
    
    # マニフェスト保存
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    
    if verbose:
        print(f"\n✅ Manifest saved: {output_path}")
        print(f"   Total files: {len(manifest['files'])}")
    
    return manifest


def main():
    parser = argparse.ArgumentParser(description='Generate manifest for arrangement artifacts')
    parser.add_argument(
        '--song-dir',
        type=Path,
        required=True,
        help='Song directory (e.g., song_packages/suno_project/song_001)'
    )
    parser.add_argument(
        '--output',
        type=Path,
        required=True,
        help='Output manifest JSON path'
    )
    parser.add_argument('--quiet', action='store_true', help='Suppress verbose output')
    
    args = parser.parse_args()
    
    generate_manifest(args.song_dir, args.output, verbose=not args.quiet)


if __name__ == '__main__':
    main()
