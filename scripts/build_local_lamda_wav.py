#!/usr/bin/env python3
"""
WAV版 LOCAL LAMDA Pickle Builder

Suno AI循環方式データ（WAV → MIDI → Stage2）用の5軸pickle作成

Features:
- WAV → MIDI変換済みデータからLAMDA互換pickle作成
- KILO_CHORDS_DATA（コード進行カタログ）
- META_DATA（パッチ分布/統計情報）
- SIGNATURES_DATA（拍子シグネチャ）
- TOTALS_MATRIX（外れ値スコア）
- ID_MAP（ファイルマッピング）

Usage:
    # MoisesDB + MUSDB18統合DBから5軸pickle作成
    python scripts/build_local_lamda_wav.py \\
        --input-db data/wav_unified.db \\
        --output-dir data/local_lamda/wav \\
        --shard-size 5000

    # Stage2 JSONから直接作成
    python scripts/build_local_lamda_wav.py \\
        --input-json-dir output/stage2/json \\
        --output-dir data/local_lamda/wav \\
        --shard-size 5000
"""

import argparse
import csv
import hashlib
import json
import pickle
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class WAVLocalLAMDABuilder:
    """WAV版LOCAL LAMDA 5軸Pickle構築"""
    
    def __init__(
        self,
        output_dir: Path,
        shard_size: int = 5000,
        verbose: bool = True
    ):
        self.output_dir = output_dir
        self.shard_size = shard_size
        self.verbose = verbose
        
        # 出力ディレクトリ作成
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 5軸データストレージ
        self.kilo_data: Dict[str, List] = {}  # file_id → chords
        self.meta_data: Dict[str, Dict] = {}  # file_id → metadata
        self.signatures_data: Dict[str, List] = {}  # file_id → time_signatures
        self.totals_data: Dict[str, Dict] = {}  # file_id → pitch/dur/vel stats
        self.id_map: Dict[str, str] = {}  # src_id → target_id
    
    def build_from_sqlite(self, db_path: Path):
        """
        SQLiteデータベースから5軸pickle作成
        
        Args:
            db_path: wav_dataset_integration.pyが作成したSQLite DB
        """
        if self.verbose:
            print(f"📊 Reading from SQLite: {db_path}")
        
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # wav_dataset_meta テーブルから読み込み
        cursor.execute("""
            SELECT 
                song_id, hash_id, dataset_type, duration, 
                selected_stem, available_stems, midi_path
            FROM wav_dataset_meta
        """)
        
        rows = cursor.fetchall()
        
        if self.verbose:
            print(f"✅ Found {len(rows)} songs in database")
        
        for row in rows:
            song_id, hash_id, dataset_type, duration, selected_stem, available_stems_json, midi_path = row
            
            # ID_MAP登録
            self.id_map[song_id] = hash_id
            
            # MIDI_PATHからStage2 JSON読み込み（存在すれば）
            if midi_path:
                midi_path_obj = Path(midi_path)
                stage2_json = midi_path_obj.parent / f"{midi_path_obj.stem}.stage2.json"
                
                if stage2_json.exists():
                    self._process_stage2_json(song_id, stage2_json)
        
        # progressions テーブルからコード進行読み込み
        cursor.execute("""
            SELECT hash_id, progression, total_events, chord_events
            FROM progressions
        """)
        
        prog_rows = cursor.fetchall()
        
        if self.verbose:
            print(f"✅ Found {len(prog_rows)} chord progressions")
        
        for hash_id, progression_json, total_events, chord_events in prog_rows:
            if progression_json:
                chords = json.loads(progression_json)
                self.kilo_data[hash_id] = chords
        
        conn.close()
    
    def build_from_json_dir(self, json_dir: Path):
        """
        Stage2 JSONディレクトリから5軸pickle作成
        
        Args:
            json_dir: Stage2 JSONファイルが格納されたディレクトリ
        """
        if self.verbose:
            print(f"📂 Reading from JSON directory: {json_dir}")
        
        json_files = sorted(json_dir.glob('*.json'))
        
        if self.verbose:
            print(f"✅ Found {len(json_files)} JSON files")
        
        for json_file in json_files:
            file_id = json_file.stem
            self._process_stage2_json(file_id, json_file)
    
    def _process_stage2_json(self, file_id: str, json_path: Path):
        """
        Stage2 JSONから5軸データ抽出
        
        Args:
            file_id: ファイルID
            json_path: Stage2 JSONファイルパス
        """
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
            
            # 1. KILO_CHORDS_DATA（コード進行）
            if 'chords' in data or 'chordmap_external' in data:
                chords = data.get('chords', [])
                if not chords and 'chordmap_external' in data:
                    # chordmap_externalから抽出
                    chordmap = data['chordmap_external']
                    if 'chords' in chordmap:
                        chords = chordmap['chords']
                
                if chords:
                    # LAMDA KILO形式: [(root, quality, time_ql), ...]
                    kilo_chords = []
                    for chord in chords:
                        root = chord.get('root', 'C')
                        quality = chord.get('quality', 'maj')
                        time_ql = chord.get('time_ql', 0.0)
                        kilo_chords.append((root, quality, time_ql))
                    
                    self.kilo_data[file_id] = kilo_chords
            
            # 2. META_DATA（パッチ分布/統計）
            meta = {}
            
            # パッチ分布
            if 'patch_summary' in data:
                meta['patches'] = data['patch_summary']
            
            # 統計情報
            if 'note_stats_meta' in data:
                stats = data['note_stats_meta']
                meta['total_notes'] = stats.get('total_notes', 0)
                meta['avg_velocity'] = stats.get('avg_velocity', 64.0)
                meta['pitch_range'] = stats.get('pitch_range', [36, 84])
            
            # BPM
            if 'bpm' in data:
                meta['bpm'] = data['bpm']
            
            # Genre（データセット名から推定）
            if 'dataset_type' in data:
                meta['genre'] = data['dataset_type']
            
            if meta:
                self.meta_data[file_id] = meta
            
            # 3. SIGNATURES_DATA（拍子）
            if 'signatures' in data:
                self.signatures_data[file_id] = data['signatures']
            elif 'time_signature' in data:
                # 単一拍子の場合
                self.signatures_data[file_id] = [data['time_signature']]
            
            # 4. TOTALS_MATRIX（外れ値スコア）
            if 'outliers' in data:
                self.totals_data[file_id] = data['outliers']
        
        except Exception as e:
            if self.verbose:
                print(f"⚠️  Error processing {json_path}: {e}")
    
    def save_pickles(self):
        """5軸pickleファイル保存"""
        if self.verbose:
            print(f"\n💾 Saving 5-axis pickles to {self.output_dir}")
        
        # 1. KILO_CHORDS_DATA（dict形式そのまま - lamda_sources.pyと互換）
        kilo_path = self.output_dir / "LOCAL_WAV_KILO_CHORDS_DATA.pickle"
        with open(kilo_path, 'wb') as f:
            # LAMDA互換フォーマット: {file_id: [(root, quality, time_ql), ...]}
            pickle.dump(self.kilo_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        kilo_size = kilo_path.stat().st_size / (1024 * 1024)
        if self.verbose:
            print(f"✅ KILO_CHORDS_DATA: {kilo_path.name} ({kilo_size:.2f} MB, {len(self.kilo_data)} entries)")
        
        # 2. META_DATA（シャード分割、LAMDA互換リスト形式）
        meta_dir = self.output_dir / "LOCAL_WAV_META_DATA"
        meta_dir.mkdir(exist_ok=True)
        
        meta_shards = self._shard_dict_to_list(self.meta_data, self.shard_size)
        total_meta_size = 0
        
        for shard_idx, shard_data in enumerate(meta_shards):
            shard_path = meta_dir / f"LOCAL_WAV_META_DATA_{shard_idx:06d}.pickle"
            with open(shard_path, 'wb') as f:
                # LAMDA互換フォーマット: [(file_id, meta_dict), ...]
                pickle.dump(shard_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            shard_size = shard_path.stat().st_size / (1024 * 1024)
            total_meta_size += shard_size
            
            if self.verbose:
                print(f"✅ META_DATA shard {shard_idx}: {shard_path.name} ({shard_size:.2f} MB, {len(shard_data)} entries)")
        
        if self.verbose:
            print(f"   Total META_DATA: {total_meta_size:.2f} MB")
        
        # 3. SIGNATURES_DATA（dict形式そのまま）
        sig_path = self.output_dir / "LOCAL_WAV_SIGNATURES_DATA.pickle"
        with open(sig_path, 'wb') as f:
            # LAMDA互換フォーマット: {file_id: ["4/4", "3/4", ...]}
            pickle.dump(self.signatures_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        sig_size = sig_path.stat().st_size / (1024 * 1024)
        if self.verbose:
            print(f"✅ SIGNATURES_DATA: {sig_path.name} ({sig_size:.2f} MB, {len(self.signatures_data)} entries)")
        
        # 4. TOTALS_MATRIX
        totals_path = self.output_dir / "LOCAL_WAV_TOTALS.pickle"
        with open(totals_path, 'wb') as f:
            pickle.dump(self.totals_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        totals_size = totals_path.stat().st_size / (1024 * 1024)
        if self.verbose:
            print(f"✅ TOTALS_MATRIX: {totals_path.name} ({totals_size:.2f} MB, {len(self.totals_data)} entries)")
        
        # 5. ID_MAP（CSV形式）
        id_map_path = self.output_dir / "local_wav_id_map.csv"
        with open(id_map_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['src_id', 'target_id'])
            for src_id, target_id in self.id_map.items():
                writer.writerow([src_id, target_id])
        
        id_map_size = id_map_path.stat().st_size / (1024 * 1024)
        if self.verbose:
            print(f"✅ ID_MAP: {id_map_path.name} ({id_map_size:.2f} MB, {len(self.id_map)} entries)")
        
        # サマリー
        total_size = kilo_size + total_meta_size + sig_size + totals_size + id_map_size
        if self.verbose:
            print(f"\n📊 Total pickle size: {total_size:.2f} MB")
            print(f"   KILO: {len(self.kilo_data)} entries")
            print(f"   META: {len(self.meta_data)} entries")
            print(f"   SIGNATURES: {len(self.signatures_data)} entries")
            print(f"   TOTALS: {len(self.totals_data)} entries")
            print(f"   ID_MAP: {len(self.id_map)} entries")
    
    def _shard_dict(self, data: Dict, shard_size: int) -> List[Dict]:
        """
        辞書をシャード分割（dict形式）
        
        Args:
            data: 分割対象の辞書
            shard_size: シャードサイズ
        
        Returns:
            シャードリスト
        """
        shards = []
        current_shard = {}
        
        for key, value in data.items():
            current_shard[key] = value
            
            if len(current_shard) >= shard_size:
                shards.append(current_shard)
                current_shard = {}
        
        # 残りを追加
        if current_shard:
            shards.append(current_shard)
        
        return shards
    
    def _shard_dict_to_list(self, data: Dict, shard_size: int) -> List[List[Tuple]]:
        """
        辞書をLAMDA互換リスト形式でシャード分割
        
        Args:
            data: 分割対象の辞書
            shard_size: シャードサイズ
        
        Returns:
            シャードリスト（LAMDA互換: [[(file_id, meta_dict), ...], ...]）
        """
        shards = []
        current_shard = []
        
        for file_id, meta_dict in data.items():
            current_shard.append((file_id, meta_dict))
            
            if len(current_shard) >= shard_size:
                shards.append(current_shard)
                current_shard = []
        
        # 残りを追加
        if current_shard:
            shards.append(current_shard)
        
        return shards


def main():
    parser = argparse.ArgumentParser(
        description="WAV版 LOCAL LAMDA 5軸Pickle Builder"
    )
    
    # 入力ソース（排他）
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        '--input-db',
        type=Path,
        help='SQLite database from wav_dataset_integration.py'
    )
    input_group.add_argument(
        '--input-json-dir',
        type=Path,
        help='Stage2 JSON directory'
    )
    
    # 出力設定
    parser.add_argument(
        '--output-dir',
        type=Path,
        required=True,
        help='Output directory for 5-axis pickles'
    )
    parser.add_argument(
        '--shard-size',
        type=int,
        default=5000,
        help='Shard size for META_DATA (default: 5000)'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Verbose output'
    )
    
    args = parser.parse_args()
    
    # ビルダー初期化
    builder = WAVLocalLAMDABuilder(
        output_dir=args.output_dir,
        shard_size=args.shard_size,
        verbose=args.verbose
    )
    
    # データ読み込み
    if args.input_db:
        builder.build_from_sqlite(args.input_db)
    elif args.input_json_dir:
        builder.build_from_json_dir(args.input_json_dir)
    
    # Pickle保存
    builder.save_pickles()
    
    print(f"\n✅ WAV版 LOCAL LAMDA pickles created successfully!")
    print(f"📂 Output directory: {args.output_dir}")


if __name__ == '__main__':
    main()
