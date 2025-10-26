#!/usr/bin/env python3
"""
LOCAL LAMDA WAV版 Pickle生成システム（Content-based file_id）

SQLiteデータベースから5軸pickle生成:
- KILO: コード進行データ
- META: BPM/genre/total_notes
- SIGNATURES: time signature
- TOTALS: 統計情報
- ID_MAP: song_id→file_id マッピング

Usage:
    # SQLite → Pickle変換
    python scripts/local_lamda_wav_pickle_builder.py \\
        --input-db data/moisesdb_wav_unified.db \\
        --output-dir data/local_lamda/moisesdb_wav \\
        --source-name moisesdb \\
        --verbose
"""

import argparse
import csv
import json
import pickle
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


class WAVPickleBuilder:
    """SQLiteからLAMDA互換5軸pickle生成"""
    
    def __init__(
        self,
        db_path: Path,
        output_dir: Path,
        source_name: str = "local_wav",
        prefix: str = "LOCAL_WAV"
    ):
        self.db_path = db_path
        self.output_dir = output_dir
        self.source_name = source_name
        self.prefix = prefix
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # データ格納
        self.kilo_data = {}      # {song_id: [chord_events]}
        self.meta_data = {}      # {song_id: {bpm, genre, ...}}
        self.signatures_data = {} # {song_id: time_sig}
        self.totals = {}         # 統計
        self.id_map = []         # [(song_id, file_id, ...)]
    
    def build_from_sqlite(self, verbose: bool = True):
        """SQLiteデータベースから全データ読み込み"""
        if verbose:
            print(f"\n{'='*70}")
            print(f"Building pickles from: {self.db_path}")
            print(f"{'='*70}")
        
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # 1. WAV特徴量読み込み
        cursor.execute("""
            SELECT song_id, file_id, duration, tempo, num_beats, num_onsets,
                   chord_candidates, activity_mean, features_json, manifest
            FROM wav_features
            ORDER BY song_id
        """)
        
        wav_rows = cursor.fetchall()
        
        if verbose:
            print(f"📊 Loaded {len(wav_rows)} WAV features")
        
        for row in wav_rows:
            song_id = row['song_id']
            file_id = row['file_id']
            
            # KILO: コード候補
            chord_candidates = json.loads(row['chord_candidates'] or '[]')
            if chord_candidates:
                # LAMDA形式: [{"chord": "C", "time": 0.0}, ...]
                kilo_events = [
                    {"chord": chord, "time": i * 4.0}  # 仮の時間（4小節ごと）
                    for i, chord in enumerate(chord_candidates)
                ]
                self.kilo_data[song_id] = kilo_events
            
            # META: メタデータ
            self.meta_data[song_id] = {
                'bpm': float(row['tempo']) if row['tempo'] else 120.0,
                'genre': self.source_name,
                'total_notes': int(row['num_onsets']) if row['num_onsets'] else 0,
                'duration': float(row['duration']) if row['duration'] else 0.0,
                'num_beats': int(row['num_beats']) if row['num_beats'] else 0,
                'activity_mean': float(row['activity_mean']) if row['activity_mean'] else 0.0,
            }
            
            # SIGNATURES: time signature（デフォルト4/4）
            self.signatures_data[song_id] = [4, 4]
            
            # ID_MAP
            manifest = json.loads(row['manifest']) if row['manifest'] else {}
            self.id_map.append({
                'song_id': song_id,
                'file_id': file_id,
                'role': manifest.get('role', 'mix'),
                'sr': manifest.get('sr', 22050),
                'channels': manifest.get('channels', 2),
            })
        
        # 2. MIDI特徴量読み込み（あれば）
        try:
            cursor.execute("""
                SELECT song_id, chords, stage2_json
                FROM midi_features
                WHERE chords IS NOT NULL
            """)
            
            midi_rows = cursor.fetchall()
            
            if verbose and midi_rows:
                print(f"🎹 Loaded {len(midi_rows)} MIDI features")
            
            for row in midi_rows:
                song_id = row['song_id']
                chords = json.loads(row['chords'])
                
                # MIDIコード情報でKILOを上書き/補完
                if 'events' in chords:
                    self.kilo_data[song_id] = chords['events']
        except sqlite3.OperationalError:
            if verbose:
                print("⚠️ midi_features table not found (WAV-only mode)")
        
        # 3. TOTALS: 統計計算
        self.totals = {
            'total_songs': len(self.meta_data),
            'total_chords': sum(len(v) for v in self.kilo_data.values()),
            'avg_bpm': np.mean([m['bpm'] for m in self.meta_data.values()]),
            'avg_duration': np.mean([m['duration'] for m in self.meta_data.values()]),
            'source_name': self.source_name,
        }
        
        conn.close()
        
        if verbose:
            print(f"\n✅ Data loaded:")
            print(f"   KILO entries: {len(self.kilo_data)}")
            print(f"   META entries: {len(self.meta_data)}")
            print(f"   SIGNATURES entries: {len(self.signatures_data)}")
            print(f"   ID_MAP entries: {len(self.id_map)}")
    
    def save_pickles(self, verbose: bool = True):
        """5軸pickleファイル保存"""
        if verbose:
            print(f"\n{'='*70}")
            print(f"Saving pickles to: {self.output_dir}")
            print(f"{'='*70}")
        
        # 1. KILO: コード進行
        kilo_path = self.output_dir / f"{self.prefix}_KILO_CHORDS_DATA.pickle"
        with open(kilo_path, 'wb') as f:
            pickle.dump(self.kilo_data, f)
        if verbose:
            print(f"✅ KILO: {kilo_path.name} ({len(self.kilo_data)} entries)")
        
        # 2. META: メタデータ（シャード形式）
        meta_dir = self.output_dir / f"{self.prefix}_META_DATA"
        meta_dir.mkdir(exist_ok=True)
        
        meta_shard_path = meta_dir / f"{self.prefix}_META_DATA_000000.pickle"
        with open(meta_shard_path, 'wb') as f:
            pickle.dump(self.meta_data, f)
        if verbose:
            print(f"✅ META: {meta_shard_path.name} ({len(self.meta_data)} entries)")
        
        # 3. SIGNATURES: time signature
        sig_path = self.output_dir / f"{self.prefix}_SIGNATURES_DATA.pickle"
        with open(sig_path, 'wb') as f:
            pickle.dump(self.signatures_data, f)
        if verbose:
            print(f"✅ SIGNATURES: {sig_path.name} ({len(self.signatures_data)} entries)")
        
        # 4. TOTALS: 統計情報
        totals_path = self.output_dir / f"{self.prefix}_TOTALS.pickle"
        with open(totals_path, 'wb') as f:
            pickle.dump(self.totals, f)
        if verbose:
            print(f"✅ TOTALS: {totals_path.name}")
        
        # 5. ID_MAP: CSV形式
        id_map_path = self.output_dir / f"local_wav_id_map.csv"
        with open(id_map_path, 'w', newline='', encoding='utf-8') as f:
            if self.id_map:
                writer = csv.DictWriter(f, fieldnames=self.id_map[0].keys())
                writer.writeheader()
                writer.writerows(self.id_map)
        if verbose:
            print(f"✅ ID_MAP: {id_map_path.name} ({len(self.id_map)} entries)")
        
        if verbose:
            print(f"\n{'='*70}")
            print(f"Pickle generation complete!")
            print(f"{'='*70}")


def main():
    parser = argparse.ArgumentParser(
        description="LOCAL LAMDA WAV版 Pickle生成"
    )
    parser.add_argument(
        '--input-db',
        type=Path,
        required=True,
        help='入力SQLiteデータベース'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        required=True,
        help='出力pickleディレクトリ'
    )
    parser.add_argument(
        '--source-name',
        type=str,
        default='local_wav',
        help='データソース名'
    )
    parser.add_argument(
        '--prefix',
        type=str,
        default='LOCAL_WAV',
        help='Pickleファイル名プレフィックス'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='詳細ログ'
    )
    
    args = parser.parse_args()
    
    # Pickle生成
    builder = WAVPickleBuilder(
        db_path=args.input_db,
        output_dir=args.output_dir,
        source_name=args.source_name,
        prefix=args.prefix
    )
    
    builder.build_from_sqlite(verbose=args.verbose)
    builder.save_pickles(verbose=args.verbose)
    
    print(f"\n✅ All pickles saved to: {args.output_dir}")


if __name__ == '__main__':
    main()
