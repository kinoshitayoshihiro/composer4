#!/usr/bin/env python3
"""
WAV Dataset Integration (MoisesDB + MUSDB18)

WAV版 LOCAL LAMDA統合システム - 複数データセット対応

Supported Datasets:
- MoisesDB (segmented WAVs)
- MUSDB18 (stem-separated WAVs)

Features:
- 自動データセット検出
- ハーモニック系ステム自動選択
- WAV → MIDI → Stage2メタデータ
- LAMDA互換SQLite統合

Usage:
    # MoisesDB
    python scripts/wav_dataset_integration.py \\
        --input-dir /path/to/MoisesDB \\
        --dataset-type moisesdb \\
        --output-db data/wav_unified.db

    # MUSDB18
    python scripts/wav_dataset_integration.py \\
        --input-dir /path/to/musdb18 \\
        --dataset-type musdb18 \\
        --output-db data/wav_unified.db

    # 自動検出
    python scripts/wav_dataset_integration.py \\
        --input-dir /path/to/dataset \\
        --output-db data/wav_unified.db
"""

import argparse
import hashlib
import json
import re
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import librosa
import numpy as np
import soundfile as sf

# WAV → MIDI変換
try:
    from scripts.suno_wav_to_midi import convert_basic, post_process_midi
    WAV_TO_MIDI_AVAILABLE = True
except ImportError:
    WAV_TO_MIDI_AVAILABLE = False
    print("⚠️ suno_wav_to_midi not available, MIDI conversion disabled")

# LAMDA Stage2統合
try:
    from scripts.lamda_v2.stage2_extractor import extract_stage2_metadata
    STAGE2_AVAILABLE = True
except ImportError:
    STAGE2_AVAILABLE = False
    print("⚠️ lamda_v2.stage2_extractor not available")


# ========== Config ==========

# ハーモニック系ステム優先度
HARMONIC_STEM_PRIORITY = [
    'piano',
    'keys',
    'guitar',
    'bass',
    'strings',
    'synth',
    'brass',
    'pad',
    'other',
]

EXCLUDED_STEMS = [
    'vocals',
    'drums',
    'percussion',
]

# MUSDB18ステムマッピング
MUSDB18_STEM_MAPPING = {
    'vocals': 'vocals',
    'drums': 'drums',
    'bass': 'bass',
    'other': 'other'  # guitar/piano/etc
}


class DatasetDetector:
    """データセット自動検出"""
    
    @staticmethod
    def detect_dataset_type(input_dir: Path) -> str:
        """
        ディレクトリ構造からデータセットタイプを検出
        
        Returns:
            'moisesdb', 'musdb18', or 'unknown'
        """
        # サンプリング: 最初の3ディレクトリをチェック
        subdirs = [d for d in input_dir.iterdir() if d.is_dir()]
        
        if not subdirs:
            return 'unknown'
        
        sample_dirs = subdirs[:3]
        
        for subdir in sample_dirs:
            files = list(subdir.glob('*.wav'))
            
            if not files:
                continue
            
            # MoisesDB: segment_XXXX_stem.wav
            if any('segment_' in f.name for f in files):
                return 'moisesdb'
            
            # MUSDB18: vocals.wav, drums.wav, bass.wav, other.wav
            stem_names = {f.stem for f in files}
            if {'vocals', 'drums', 'bass', 'other'}.issubset(stem_names):
                return 'musdb18'
        
        return 'unknown'


class MUSDB18Processor:
    """MUSDB18データセット処理"""
    
    def __init__(self, sr: int = 22050):
        self.sr = sr
    
    def collect_stems(self, song_dir: Path) -> Dict[str, Path]:
        """
        MUSDB18ステム収集
        
        Returns:
            {'vocals': Path('vocals.wav'), 'drums': ..., 'bass': ..., 'other': ...}
        """
        stems = {}
        
        for stem_name in ['vocals', 'drums', 'bass', 'other']:
            stem_path = song_dir / f'{stem_name}.wav'
            if stem_path.exists():
                stems[stem_name] = stem_path
        
        return stems
    
    def select_harmonic_stem(
        self,
        stems: Dict[str, Path]
    ) -> Optional[str]:
        """
        MUSDB18からハーモニック系ステム選択
        
        MUSDB18は'other'に全てのハーモニック楽器が含まれる
        """
        # 優先順位: other (harmonic) > bass > 除外(vocals, drums)
        if 'other' in stems:
            return 'other'
        elif 'bass' in stems:
            return 'bass'
        else:
            return None


class WAVDatasetIntegrator:
    """WAVデータセット統合（MoisesDB + MUSDB18）"""
    
    def __init__(
        self,
        db_path: Path,
        midi_output_dir: Path,
        sr: int = 22050,
        dataset_type: Optional[str] = None,
        use_gpu: bool = False,
        dynamic_weights: bool = False
    ):
        self.db_path = db_path
        self.midi_output_dir = midi_output_dir
        self.sr = sr
        self.dataset_type = dataset_type
        self.use_gpu = use_gpu
        self.dynamic_weights = dynamic_weights
        
        # GPU対応
        if use_gpu:
            try:
                from scripts.moisesdb_gpu_processor import GPUWAVProcessor
                self.gpu_processor = GPUWAVProcessor(device=None)
                print(f"✅ GPU acceleration enabled: {self.gpu_processor.device}")
            except ImportError:
                print("⚠️  PyTorch not installed, falling back to CPU")
                self.use_gpu = False
                self.gpu_processor = None
        else:
            self.gpu_processor = None
        
        # 動的重み調整
        if dynamic_weights:
            try:
                from scripts.moisesdb_dynamic_weights import DynamicWeightAdjuster
                self.weight_adjuster = DynamicWeightAdjuster(sr=sr, use_gpu=use_gpu)
                print(f"✅ Dynamic weight adjustment enabled")
            except ImportError:
                print("⚠️  Dynamic weights module not available")
                self.dynamic_weights = False
                self.weight_adjuster = None
        else:
            self.weight_adjuster = None
        
        # データセット固有プロセッサ
        self.musdb18_processor = MUSDB18Processor(sr=sr)
        
        # MoisesDB用（既存）
        try:
            from scripts.moisesdb_integration import (
                SegmentMerger,
                HarmonicStemSelector
            )
            self.segment_merger = SegmentMerger(sr=sr)
            self.stem_selector = HarmonicStemSelector()
        except ImportError:
            print("⚠️  MoisesDB modules not available")
            self.segment_merger = None
            self.stem_selector = None
        
        self._init_database()
    
    def _init_database(self):
        """データベース初期化（LAMDA互換）"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # progressions テーブル（LAMDA互換）
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS progressions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                hash_id TEXT NOT NULL,
                progression TEXT NOT NULL,
                total_events INTEGER,
                chord_events INTEGER,
                source_file TEXT
            )
        """)
        
        # wav_dataset_meta テーブル（統合メタデータ）
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS wav_dataset_meta (
                song_id TEXT PRIMARY KEY,
                hash_id TEXT NOT NULL,
                dataset_type TEXT,
                duration REAL,
                num_segments INTEGER,
                selected_stem TEXT,
                available_stems TEXT,
                midi_path TEXT
            )
        """)
        
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_hash_id ON progressions(hash_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_dataset ON wav_dataset_meta(dataset_type)")
        
        conn.commit()
        conn.close()
    
    def process_dataset(
        self,
        input_dir: Path,
        max_songs: int = -1,
        verbose: bool = True
    ):
        """
        データセット処理（自動検出または指定）
        
        Args:
            input_dir: データセットルートディレクトリ
            max_songs: 処理する最大曲数（-1で全件）
            verbose: 詳細ログ表示
        """
        # データセットタイプ検出
        if self.dataset_type is None:
            detected_type = DatasetDetector.detect_dataset_type(input_dir)
            print(f"🔍 Detected dataset type: {detected_type}")
            self.dataset_type = detected_type
        
        if self.dataset_type == 'unknown':
            print("❌ Unknown dataset type. Please specify --dataset-type manually.")
            return
        
        # 曲リスト取得
        song_dirs = sorted([d for d in input_dir.iterdir() if d.is_dir()])
        
        if max_songs > 0:
            song_dirs = song_dirs[:max_songs]
        
        print(f"📊 Processing {len(song_dirs)} songs from {self.dataset_type}...")
        
        # データセット別処理
        for i, song_dir in enumerate(song_dirs, 1):
            if verbose:
                print(f"\n[{i}/{len(song_dirs)}] Processing: {song_dir.name}")
            
            try:
                if self.dataset_type == 'moisesdb':
                    result = self._process_moisesdb_song(song_dir, verbose)
                elif self.dataset_type == 'musdb18':
                    result = self._process_musdb18_song(song_dir, verbose)
                else:
                    print(f"⚠️  Unsupported dataset type: {self.dataset_type}")
                    continue
                
                if verbose and result['status'] == 'success':
                    print(f"✅ {result['song_id']}: {result['duration']:.2f}s")
            
            except Exception as e:
                print(f"❌ Error processing {song_dir.name}: {e}")
                continue
    
    def _process_moisesdb_song(
        self,
        song_dir: Path,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """MoisesDB曲処理（既存ロジック）"""
        # scripts/moisesdb_integration.py の process_song_directory を使用
        from scripts.moisesdb_integration import MoisesDBIntegrator
        
        integrator = MoisesDBIntegrator(
            db_path=self.db_path,
            midi_output_dir=self.midi_output_dir,
            sr=self.sr,
            use_gpu=self.use_gpu,
            dynamic_weights=self.dynamic_weights
        )
        
        return integrator.process_song_directory(song_dir, verbose)
    
    def _process_musdb18_song(
        self,
        song_dir: Path,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """MUSDB18曲処理"""
        song_id = song_dir.name
        
        # 1. ステム収集
        stems = self.musdb18_processor.collect_stems(song_dir)
        
        if not stems:
            print(f"⚠️  No stems found in {song_dir}")
            return {'status': 'skipped', 'reason': 'no_stems'}
        
        # 2. ハーモニック系ステム選択
        selected_stem = self.musdb18_processor.select_harmonic_stem(stems)
        
        if not selected_stem:
            print(f"⚠️  No harmonic stem found in {list(stems.keys())}")
            return {'status': 'skipped', 'reason': 'no_harmonic_stem'}
        
        if verbose:
            print(f"✅ Selected stem: {selected_stem}")
            print(f"   Available: {list(stems.keys())}")
        
        # 3. WAVロード（リサンプリング）
        stem_path = stems[selected_stem]
        y, sr = librosa.load(str(stem_path), sr=self.sr, mono=True)
        duration = len(y) / sr
        
        # 4. WAV保存（統一フォーマット）
        output_wav_path = self.midi_output_dir / f"{song_id}_{selected_stem}.wav"
        output_wav_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(output_wav_path), y, sr)
        
        if verbose:
            print(f"✅ Resampled to {self.sr}Hz")
            print(f"   Duration: {duration:.2f}s")
        
        # 5. WAV → MIDI変換
        midi_path = None
        if WAV_TO_MIDI_AVAILABLE:
            midi_path = self._convert_to_midi(
                output_wav_path,
                song_id,
                verbose
            )
        
        # 6. LAMDA Stage2メタデータ抽出
        stage2_meta = None
        if midi_path and STAGE2_AVAILABLE:
            stage2_meta = self._extract_stage2_features(midi_path, verbose)
        
        # 7. データベース登録
        hash_id = self._calc_hash(song_id)
        self._save_to_database(
            song_id=song_id,
            hash_id=hash_id,
            dataset_type='musdb18',
            duration=duration,
            num_segments=1,  # MUSDB18はセグメントなし
            selected_stem=selected_stem,
            available_stems=list(stems.keys()),
            midi_path=midi_path,
            stage2_meta=stage2_meta
        )
        
        return {
            'status': 'success',
            'song_id': song_id,
            'hash_id': hash_id,
            'selected_stem': selected_stem,
            'duration': duration,
            'midi_path': str(midi_path) if midi_path else None
        }
    
    def _convert_to_midi(
        self,
        wav_path: Path,
        song_id: str,
        verbose: bool = True
    ) -> Optional[Path]:
        """WAV → MIDI変換"""
        if not WAV_TO_MIDI_AVAILABLE:
            return None
        
        midi_path = self.midi_output_dir / f"{song_id}.mid"
        
        try:
            if verbose:
                print(f"🎹 Converting to MIDI: {wav_path.name}")
            
            # basic-pitch変換
            midi_data = convert_basic(str(wav_path))
            
            # 後処理
            midi_data = post_process_midi(midi_data)
            
            # 保存
            midi_data.write(str(midi_path))
            
            if verbose:
                print(f"✅ MIDI saved: {midi_path.name}")
            
            return midi_path
        
        except Exception as e:
            print(f"⚠️  MIDI conversion failed: {e}")
            return None
    
    def _extract_stage2_features(
        self,
        midi_path: Path,
        verbose: bool = True
    ) -> Optional[Dict[str, Any]]:
        """LAMDA Stage2メタデータ抽出"""
        if not STAGE2_AVAILABLE:
            return None
        
        try:
            if verbose:
                print(f"📊 Extracting Stage2 metadata...")
            
            meta = extract_stage2_metadata(midi_path)
            
            if verbose:
                if 'chords' in meta:
                    print(f"✅ Extracted {len(meta['chords'])} chord events")
            
            return meta
        
        except Exception as e:
            print(f"⚠️  Stage2 extraction failed: {e}")
            return None
    
    def _calc_hash(self, song_id: str) -> str:
        """hash_id計算"""
        return hashlib.sha256(song_id.encode()).hexdigest()[:16]
    
    def _save_to_database(
        self,
        song_id: str,
        hash_id: str,
        dataset_type: str,
        duration: float,
        num_segments: int,
        selected_stem: str,
        available_stems: List[str],
        midi_path: Optional[Path],
        stage2_meta: Optional[Dict[str, Any]]
    ):
        """データベース保存"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # wav_dataset_meta テーブル
        cursor.execute("""
            INSERT OR REPLACE INTO wav_dataset_meta
            (song_id, hash_id, dataset_type, duration, num_segments, 
             selected_stem, available_stems, midi_path)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            song_id,
            hash_id,
            dataset_type,
            duration,
            num_segments,
            selected_stem,
            json.dumps(available_stems),
            str(midi_path) if midi_path else None
        ))
        
        # progressions テーブル（Stage2メタデータがある場合）
        if stage2_meta and 'chords' in stage2_meta:
            cursor.execute("""
                INSERT OR REPLACE INTO progressions
                (hash_id, progression, total_events, chord_events, source_file)
                VALUES (?, ?, ?, ?, ?)
            """, (
                hash_id,
                json.dumps(stage2_meta['chords']),
                len(stage2_meta.get('events', [])),
                len(stage2_meta['chords']),
                f"{dataset_type}/{song_id}"
            ))
        
        conn.commit()
        conn.close()


def main():
    parser = argparse.ArgumentParser(
        description="WAV Dataset Integration (MoisesDB + MUSDB18)"
    )
    
    parser.add_argument(
        '--input-dir',
        type=Path,
        required=True,
        help='Dataset root directory'
    )
    parser.add_argument(
        '--output-db',
        type=Path,
        required=True,
        help='Output SQLite database path'
    )
    parser.add_argument(
        '--dataset-type',
        type=str,
        choices=['moisesdb', 'musdb18', 'auto'],
        default='auto',
        help='Dataset type (auto-detect by default)'
    )
    parser.add_argument(
        '--midi-output-dir',
        type=Path,
        default=None,
        help='MIDI output directory (default: auto from db_path)'
    )
    parser.add_argument(
        '--max-songs',
        type=int,
        default=-1,
        help='Maximum songs to process (-1 for all)'
    )
    parser.add_argument(
        '--sr',
        type=int,
        default=22050,
        help='Target sample rate'
    )
    parser.add_argument(
        '--use-gpu',
        action='store_true',
        help='Use GPU acceleration'
    )
    parser.add_argument(
        '--dynamic-weights',
        action='store_true',
        help='Enable dynamic weight adjustment'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Verbose output'
    )
    
    args = parser.parse_args()
    
    # MIDI出力ディレクトリ自動設定
    if args.midi_output_dir is None:
        args.midi_output_dir = args.output_db.parent / f"{args.output_db.stem}_midi"
    
    # データセットタイプ
    dataset_type = None if args.dataset_type == 'auto' else args.dataset_type
    
    # 統合実行
    integrator = WAVDatasetIntegrator(
        db_path=args.output_db,
        midi_output_dir=args.midi_output_dir,
        sr=args.sr,
        dataset_type=dataset_type,
        use_gpu=args.use_gpu,
        dynamic_weights=args.dynamic_weights
    )
    
    integrator.process_dataset(
        input_dir=args.input_dir,
        max_songs=args.max_songs,
        verbose=args.verbose
    )
    
    print(f"\n✅ Processing complete!")
    print(f"📊 Database: {args.output_db}")
    print(f"🎹 MIDI files: {args.midi_output_dir}")


if __name__ == '__main__':
    main()
