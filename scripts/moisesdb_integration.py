#!/usr/bin/env python3
"""
MoisesDB Integration (WAV版 LOCAL LAMDA)

Features:
- 複数WAVセグメント統合（1曲 = N segments）
- ハーモニック系ステム自動選択（guitar/piano/keys/strings優先）
- LAMDA Stage2メタデータ抽出（WAV → MIDI → 特徴量）
- SQLite統合（lamda_unified.dbと同一スキーマ）

Input:
    MoisesDB/
    ├── song_001/
    │   ├── segment_0000_vocals.wav
    │   ├── segment_0000_drums.wav
    │   ├── segment_0000_guitar.wav
    │   ├── segment_0001_vocals.wav
    │   └── ...
    └── song_002/
        └── ...

Output:
    - data/moisesdb_unified.db (SQLite)
    - data/moisesdb_midi/ (変換済みMIDI)
    - data/moisesdb_meta.jsonl (メタデータ)

Usage:
    python scripts/moisesdb_integration.py \\
        --input-dir /path/to/MoisesDB \\
        --output-db data/moisesdb_unified.db \\
        --max-songs 100
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

# WAV → MIDI変換（scripts/suno_wav_to_midi.pyを参照）
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

# ハーモニック系ステム優先度（高→低）
HARMONIC_STEM_PRIORITY = [
    'piano',
    'keys',
    'guitar',
    'bass',
    'strings',
    'synth',
    'brass',
    'pad',
    'other',  # fallback
]

# 除外ステム（非ハーモニック）
EXCLUDED_STEMS = [
    'vocals',
    'drums',
    'percussion',
]

# ステム重み設定（chordmap投票用）
STEM_WEIGHTS = {
    'piano': 0.40,
    'keys': 0.40,
    'guitar': 0.35,
    'strings': 0.10,
    'pad': 0.15,
    'synth': 0.20,
    'bass': 0.20,
    'brass': 0.15,
    'other': 0.05,
    # 除外ステムは重み0
    'vocals': 0.0,
    'drums': 0.0,
    'percussion': 0.0,
}

# セグメント名パターン（例: "segment_0000_guitar.wav"）
SEGMENT_PATTERN = re.compile(r'segment_(\d+)_([a-z_]+)\.wav')


# ========== Segment Merger ==========

class SegmentMerger:
    """複数WAVセグメントを1つに統合"""
    
    def __init__(self, sr: int = 22050):
        self.sr = sr
    
    def merge_segments(
        self,
        segment_paths: List[Path],
        output_path: Path
    ) -> Dict[str, Any]:
        """
        セグメントを時系列で結合
        
        Args:
            segment_paths: セグメントファイル（ソート済み想定）
            output_path: 出力WAVパス
        
        Returns:
            {
                'duration': float,
                'num_segments': int,
                'sample_rate': int
            }
        """
        if not segment_paths:
            raise ValueError("No segments provided")
        
        # セグメント番号でソート
        sorted_paths = sorted(
            segment_paths,
            key=lambda p: self._extract_segment_number(p)
        )
        
        merged_audio = []
        
        for seg_path in sorted_paths:
            audio, sr = librosa.load(str(seg_path), sr=self.sr, mono=True)
            merged_audio.append(audio)
        
        # 結合
        full_audio = np.concatenate(merged_audio)
        
        # 保存
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(output_path), full_audio, self.sr)
        
        return {
            'duration': len(full_audio) / self.sr,
            'num_segments': len(sorted_paths),
            'sample_rate': self.sr
        }
    
    def _extract_segment_number(self, path: Path) -> int:
        """セグメント番号を抽出"""
        match = SEGMENT_PATTERN.match(path.name)
        if match:
            return int(match.group(1))
        return 0


# ========== Stem Selector ==========

class HarmonicStemSelector:
    """ハーモニック系ステム自動選択"""
    
    def __init__(self, priority: List[str] = HARMONIC_STEM_PRIORITY):
        self.priority = priority
    
    def select_best_stem(
        self,
        available_stems: List[str]
    ) -> Optional[str]:
        """
        優先度に基づいて最適なステムを選択
        
        Args:
            available_stems: 利用可能なステム名リスト
        
        Returns:
            選択されたステム名（Noneの場合は該当なし）
        """
        # 除外ステムをフィルタ
        filtered = [
            s for s in available_stems
            if s not in EXCLUDED_STEMS
        ]
        
        # 優先度順に検索
        for stem_type in self.priority:
            if stem_type in filtered:
                return stem_type
        
        # フォールバック: 最初の非除外ステム
        return filtered[0] if filtered else None
    
    def select_harmonic_stems_with_weights(
        self,
        available_stems: List[str]
    ) -> Tuple[List[str], Dict[str, float]]:
        """
        複数ハーモニック系ステムを重み付きで選択
        
        Args:
            available_stems: 利用可能なステム名リスト
        
        Returns:
            (harmonic_stems, weights)
            - harmonic_stems: 選択されたステムリスト
            - weights: {stem_name: weight}
        
        Example:
            >>> selector.select_harmonic_stems_with_weights(['guitar', 'piano', 'drums'])
            (['guitar', 'piano'], {'guitar': 0.35, 'piano': 0.40})
        """
        # 除外ステムをフィルタ
        harmonic_stems = [
            s for s in available_stems
            if s not in EXCLUDED_STEMS
        ]
        
        # 重み割り当て
        weights = {}
        for stem in harmonic_stems:
            weights[stem] = STEM_WEIGHTS.get(stem, 0.05)  # デフォルト: 0.05
        
        # 正規化（合計が1.0になるように）
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v / total_weight for k, v in weights.items()}
        
        return harmonic_stems, weights
    
    def classify_stem(self, stem_name: str) -> str:
        """ステム名からカテゴリを推定"""
        stem_lower = stem_name.lower()
        
        for category in self.priority + EXCLUDED_STEMS:
            if category in stem_lower:
                return category
        
        return 'other'
    
    def analyze_stem_spectral_features(
        self,
        wav_path: Path,
        sr: int = 22050
    ) -> Dict[str, float]:
        """
        RMSスペクトル特徴からステムロールを自動判定（オプション）
        
        Args:
            wav_path: WAVファイルパス
            sr: サンプリングレート
        
        Returns:
            {
                'high_freq_ratio': float,  # 高域比率（0-1）
                'harmonic_persistence': float,  # 和声持続性（0-1）
                'percussive_ratio': float,  # 打楽器比率（0-1）
                'predicted_role': str  # 推定ロール
            }
        """
        try:
            # オーディオ読み込み
            y, _ = librosa.load(str(wav_path), sr=sr, duration=30.0)  # 最初の30秒
            
            # スペクトログラム
            S = np.abs(librosa.stft(y))
            
            # 高域比率（8kHz以上）
            nyquist = sr / 2
            high_freq_idx = int(S.shape[0] * (8000 / nyquist))
            high_freq_ratio = np.mean(S[high_freq_idx:]) / (np.mean(S) + 1e-10)
            
            # Harmonic/Percussive分離
            S_harmonic, S_percussive = librosa.decompose.hpss(S)
            harmonic_ratio = np.sum(S_harmonic) / (np.sum(S) + 1e-10)
            percussive_ratio = np.sum(S_percussive) / (np.sum(S) + 1e-10)
            
            # 和声持続性（長時間フレーム相関）
            chroma = librosa.feature.chroma_stft(S=S, sr=sr)
            harmonic_persistence = np.mean([
                np.corrcoef(chroma[:, i], chroma[:, i+10])[0, 1]
                for i in range(chroma.shape[1] - 10)
                if not np.isnan(np.corrcoef(chroma[:, i], chroma[:, i+10])[0, 1])
            ])
            
            # ロール推定
            predicted_role = self._predict_role_from_features(
                high_freq_ratio,
                harmonic_persistence,
                percussive_ratio
            )
            
            return {
                'high_freq_ratio': float(high_freq_ratio),
                'harmonic_persistence': float(harmonic_persistence),
                'percussive_ratio': float(percussive_ratio),
                'predicted_role': predicted_role
            }
        
        except Exception as e:
            print(f"⚠️ Spectral analysis failed for {wav_path}: {e}")
            return {
                'high_freq_ratio': 0.0,
                'harmonic_persistence': 0.0,
                'percussive_ratio': 0.0,
                'predicted_role': 'other'
            }
    
    def _predict_role_from_features(
        self,
        high_freq_ratio: float,
        harmonic_persistence: float,
        percussive_ratio: float
    ) -> str:
        """特徴量からロールを推定"""
        # 打楽器判定
        if percussive_ratio > 0.7:
            return 'drums'
        
        # ピアノ判定（高域＋和声持続）
        if high_freq_ratio > 0.4 and harmonic_persistence > 0.6:
            return 'piano'
        
        # ギター判定（短周期減衰＋中域）
        if 0.2 < high_freq_ratio < 0.5 and harmonic_persistence < 0.5:
            return 'guitar'
        
        # ストリングス判定（高和声持続）
        if harmonic_persistence > 0.7:
            return 'strings'
        
        # フォールバック
        return 'other'


# ========== MoisesDB Integrator ==========

class MoisesDBIntegrator:
    """MoisesDB → LAMDA統合DB構築"""
    
    def __init__(
        self,
        db_path: Path,
        midi_output_dir: Path,
        sr: int = 22050,
        use_gpu: bool = False,
        dynamic_weights: bool = False
    ):
        self.db_path = db_path
        self.midi_output_dir = midi_output_dir
        self.sr = sr
        self.use_gpu = use_gpu
        self.dynamic_weights = dynamic_weights
        
        # GPU対応プロセッサ初期化
        if use_gpu:
            try:
                from scripts.moisesdb_gpu_processor import GPUWAVProcessor
                self.gpu_processor = GPUWAVProcessor(device=None)  # 自動検出
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
                self.weight_adjuster = DynamicWeightAdjuster(
                    sr=sr,
                    use_gpu=use_gpu
                )
                print(f"✅ Dynamic weight adjustment enabled")
            except ImportError:
                print("⚠️  Dynamic weights module not available")
                self.dynamic_weights = False
                self.weight_adjuster = None
        else:
            self.weight_adjuster = None
        
        self.merger = SegmentMerger(sr=sr)
        self.selector = HarmonicStemSelector()
        
        self._init_database()
    
    def _init_database(self):
        """データベース初期化（LAMDA互換スキーマ）"""
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
        
        # moisesdb_meta テーブル（MoisesDB固有）
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS moisesdb_meta (
                song_id TEXT PRIMARY KEY,
                hash_id TEXT NOT NULL,
                duration REAL,
                num_segments INTEGER,
                selected_stem TEXT,
                available_stems TEXT,
                midi_path TEXT
            )
        """)
        
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_hash_id ON progressions(hash_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_source ON progressions(source_file)")
        
        conn.commit()
        conn.close()
    
    def process_song(
        self,
        song_dir: Path,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        1曲分のディレクトリを処理（process_song_directoryのエイリアス）
        
        Args:
            song_dir: song_XXX/ ディレクトリ
            verbose: 詳細ログ表示
        
        Returns:
            処理結果メタデータ
        """
        return self.process_song_directory(song_dir, verbose)
    
    def process_song_directory(
        self,
        song_dir: Path,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        1曲分のディレクトリを処理
        
        Args:
            song_dir: song_XXX/ ディレクトリ
        
        Returns:
            処理結果メタデータ
        """
        song_id = song_dir.name
        
        if verbose:
            print(f"\n{'='*70}")
            print(f"Processing: {song_id}")
            print(f"{'='*70}")
        
        # 1. セグメント収集
        segments_by_stem = self._collect_segments(song_dir)
        
        if not segments_by_stem:
            print(f"⚠️ No segments found in {song_dir}")
            return {'status': 'skipped', 'reason': 'no_segments'}
        
        # 2. 最適ステム選択
        available_stems = list(segments_by_stem.keys())
        selected_stem = self.selector.select_best_stem(available_stems)
        
        if not selected_stem:
            print(f"⚠️ No harmonic stem found in {available_stems}")
            return {'status': 'skipped', 'reason': 'no_harmonic_stem'}
        
        if verbose:
            print(f"✅ Selected stem: {selected_stem}")
            print(f"   Available: {available_stems}")
        
        # 3. セグメント統合（GPU対応）
        merged_wav_path = self.midi_output_dir / f"{song_id}_{selected_stem}.wav"
        
        if self.use_gpu and self.gpu_processor:
            # GPU加速版
            merge_info = self._merge_segments_gpu(
                segments_by_stem[selected_stem],
                merged_wav_path
            )
        else:
            # CPU版（従来通り）
            merge_info = self.merger.merge_segments(
                segments_by_stem[selected_stem],
                merged_wav_path
            )
        
        if verbose:
            print(f"✅ Merged {merge_info['num_segments']} segments")
            print(f"   Duration: {merge_info['duration']:.2f}s")
        
        # 4. WAV → MIDI変換
        midi_path = None
        if WAV_TO_MIDI_AVAILABLE:
            midi_path = self._convert_to_midi(
                merged_wav_path,
                song_id,
                verbose
            )
        
        # 5. LAMDA Stage2メタデータ抽出
        stage2_meta = None
        if midi_path and STAGE2_AVAILABLE:
            stage2_meta = self._extract_stage2_features(
                midi_path,
                verbose
            )
        
        # 6. データベース登録
        hash_id = self._calc_hash(song_id)
        self._save_to_database(
            song_id=song_id,
            hash_id=hash_id,
            merge_info=merge_info,
            selected_stem=selected_stem,
            available_stems=available_stems,
            midi_path=midi_path,
            stage2_meta=stage2_meta
        )
        
        return {
            'status': 'success',
            'song_id': song_id,
            'hash_id': hash_id,
            'selected_stem': selected_stem,
            'duration': merge_info['duration'],
            'midi_path': str(midi_path) if midi_path else None
        }
    
    def _collect_segments(
        self,
        song_dir: Path
    ) -> Dict[str, List[Path]]:
        """
        セグメントファイルをステム別に収集
        
        Returns:
            {
                'guitar': [segment_0000_guitar.wav, segment_0001_guitar.wav],
                'drums': [segment_0000_drums.wav, ...],
                ...
            }
        """
        segments_by_stem = defaultdict(list)
        
        for wav_file in song_dir.glob('*.wav'):
            match = SEGMENT_PATTERN.match(wav_file.name)
            if match:
                stem_name = match.group(2)
                stem_category = self.selector.classify_stem(stem_name)
                segments_by_stem[stem_category].append(wav_file)
        
        return dict(segments_by_stem)
    
    def _merge_segments_gpu(
        self,
        segment_paths: List[Path],
        output_path: Path
    ) -> Dict[str, Any]:
        """
        GPU加速版セグメント統合
        
        Args:
            segment_paths: セグメントファイルパスリスト
            output_path: 出力WAVパス
        
        Returns:
            {
                'num_segments': int,
                'duration': float,
                'sample_rate': int
            }
        """
        # GPU上でセグメント処理
        concatenated = self.gpu_processor.process_segment_batch(
            segment_paths,
            target_sr=self.sr
        )
        
        # 保存
        self.gpu_processor.save_audio(
            output_path,
            concatenated,
            self.sr
        )
        
        # メタデータ
        duration = concatenated.shape[1] / self.sr
        
        return {
            'num_segments': len(segment_paths),
            'duration': duration,
            'sample_rate': self.sr
        }
    
    def _convert_to_midi(
        self,
        wav_path: Path,
        song_id: str,
        verbose: bool
    ) -> Optional[Path]:
        """WAV → MIDI変換"""
        midi_path = self.midi_output_dir / f"{song_id}.mid"
        
        try:
            if verbose:
                print(f"🎹 Converting to MIDI...")
            
            # basic-pitch変換
            midi = convert_basic(wav_path)
            
            # 後処理（quantize, normalize）
            midi = post_process_midi(
                midi,
                quantize=True,
                quantize_resolution=16,
                normalize_velocity=True,
                velocity_range=(40, 100)
            )
            
            midi.write(str(midi_path))
            
            if verbose:
                print(f"✅ MIDI saved: {midi_path.name}")
            
            return midi_path
        
        except Exception as e:
            print(f"❌ MIDI conversion failed: {e}")
            return None
    
    def _extract_stage2_features(
        self,
        midi_path: Path,
        verbose: bool
    ) -> Optional[Dict[str, Any]]:
        """LAMDA Stage2メタデータ抽出"""
        try:
            if verbose:
                print(f"📊 Extracting Stage2 features...")
            
            meta = extract_stage2_metadata(midi_path)
            
            if verbose:
                print(f"✅ Stage2 extracted:")
                print(f"   Tempo: {meta.get('tempo', {}).get('bpm', 'N/A')}")
                print(f"   Chords: {len(meta.get('chords', {}).get('events', []))}")
            
            return meta
        
        except Exception as e:
            print(f"⚠️ Stage2 extraction failed: {e}")
            return None
    
    def _calc_hash(self, song_id: str) -> str:
        """ハッシュID生成（LAMDA互換）"""
        return hashlib.md5(song_id.encode()).hexdigest()
    
    def _save_to_database(
        self,
        song_id: str,
        hash_id: str,
        merge_info: Dict,
        selected_stem: str,
        available_stems: List[str],
        midi_path: Optional[Path],
        stage2_meta: Optional[Dict]
    ):
        """データベース保存"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # moisesdb_meta テーブル
        cursor.execute("""
            INSERT OR REPLACE INTO moisesdb_meta
            (song_id, hash_id, duration, num_segments, selected_stem, available_stems, midi_path)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            song_id,
            hash_id,
            merge_info['duration'],
            merge_info['num_segments'],
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
                len(stage2_meta['chords'].get('events', [])),
                song_id
            ))
        
        conn.commit()
        conn.close()
    
    def generate_audio_chordmap_yaml(
        self,
        song_dir: Path,
        output_yaml_path: Path,
        use_spectral_analysis: bool = False,
        merged_wav_paths: Optional[Dict[str, Path]] = None
    ) -> Dict[str, Any]:
        """
        audio_chordmap.yaml 生成（重み付き統合用）
        
        Args:
            song_dir: 曲ディレクトリ（セグメント含む）
            output_yaml_path: 出力YAMLパス
            use_spectral_analysis: スペクトル解析を使用するか
            merged_wav_paths: 結合済みWAVパス（動的重み調整用）
        
        Returns:
            生成されたYAML内容（dict）
        
        Example YAML:
            stems:
              guitar:
                weight: 0.35
                role: harmonic
              piano:
                weight: 0.40
                role: harmonic
              drums:
                weight: 0.0
                role: excluded
            aggregate_method: weighted_average
        """
        # セグメント収集
        segments_by_stem = self._collect_segments(song_dir)
        
        # ハーモニック系ステム選択（重み付き）
        available_stems = list(segments_by_stem.keys())
        harmonic_stems, weights = self.selector.select_harmonic_stems_with_weights(
            available_stems
        )
        
        # 動的重み調整（有効な場合）
        if self.dynamic_weights and self.weight_adjuster and merged_wav_paths:
            print(f"🎯 Applying dynamic weight adjustment...")
            
            # ハーモニック系ステムのみ調整
            harmonic_wav_paths = {
                stem: merged_wav_paths[stem]
                for stem in harmonic_stems
                if stem in merged_wav_paths
            }
            
            # 動的重み計算
            adjusted_weights = self.weight_adjuster.generate_weighted_chordmap(
                stem_paths=harmonic_wav_paths,
                output_yaml=output_yaml_path
            )
            
            # 重み更新
            weights.update(adjusted_weights)
            
            print(f"✅ Dynamic weights applied: {adjusted_weights}")
        
        # YAML構造
        yaml_data = {
            'song_id': song_dir.name,
            'stems': {},
            'aggregate_method': 'weighted_average',
            'dynamic_weights_enabled': self.dynamic_weights,
            'metadata': {
                'total_stems': len(available_stems),
                'harmonic_stems': len(harmonic_stems),
                'excluded_stems': [s for s in available_stems if s in EXCLUDED_STEMS]
            }
        }
        
        # ステム情報構築
        for stem in available_stems:
            stem_info = {
                'weight': weights.get(stem, 0.0),
                'role': 'harmonic' if stem in harmonic_stems else 'excluded'
            }
            
            # スペクトル解析（オプション）
            if use_spectral_analysis and segments_by_stem[stem]:
                first_segment = segments_by_stem[stem][0]
                features = self.selector.analyze_stem_spectral_features(
                    first_segment,
                    sr=self.sr
                )
                stem_info['spectral_features'] = features
                stem_info['predicted_role'] = features['predicted_role']
            
            yaml_data['stems'][stem] = stem_info
        
        # YAML保存（動的重み調整が有効でない場合のみ）
        if not (self.dynamic_weights and merged_wav_paths):
            output_yaml_path.parent.mkdir(parents=True, exist_ok=True)
            
            import yaml
            with open(output_yaml_path, 'w', encoding='utf-8') as f:
                yaml.dump(yaml_data, f, default_flow_style=False, allow_unicode=True)
        
        return yaml_data
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # moisesdb_meta テーブル
        cursor.execute("""
            INSERT OR REPLACE INTO moisesdb_meta
            (song_id, hash_id, duration, num_segments, selected_stem, available_stems, midi_path)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            song_id,
            hash_id,
            merge_info['duration'],
            merge_info['num_segments'],
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
                len(stage2_meta['chords'].get('events', [])),
                song_id
            ))
        
        conn.commit()
        conn.close()
    
    def process_dataset(
        self,
        input_dir: Path,
        max_songs: int = -1,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """データセット全体を処理"""
        song_dirs = sorted([d for d in input_dir.iterdir() if d.is_dir()])
        
        if max_songs > 0:
            song_dirs = song_dirs[:max_songs]
        
        print(f"\n{'='*70}")
        print(f"MoisesDB Integration")
        print(f"{'='*70}")
        print(f"Total songs: {len(song_dirs)}")
        print(f"Output DB: {self.db_path}")
        print(f"MIDI dir: {self.midi_output_dir}")
        print(f"{'='*70}")
        
        results = {
            'success': 0,
            'skipped': 0,
            'failed': 0,
            'processed_songs': []
        }
        
        for song_dir in song_dirs:
            try:
                result = self.process_song_directory(song_dir, verbose)
                
                if result['status'] == 'success':
                    results['success'] += 1
                    results['processed_songs'].append(result)
                else:
                    results['skipped'] += 1
            
            except Exception as e:
                print(f"❌ Failed to process {song_dir.name}: {e}")
                results['failed'] += 1
        
        return results
    
    # ========== Query Interface (LAMDA互換) ==========
    
    def query_by_hash(self, hash_id: str) -> Optional[Dict[str, Any]]:
        """hash_idで検索（LAMDA互換）"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # progressions テーブルから検索
        cursor.execute("""
            SELECT progression, total_events, chord_events, source_file
            FROM progressions
            WHERE hash_id = ?
        """, (hash_id,))
        
        row = cursor.fetchone()
        
        if not row:
            conn.close()
            return None
        
        # moisesdb_meta も取得
        cursor.execute("""
            SELECT song_id, duration, num_segments, selected_stem, available_stems, midi_path
            FROM moisesdb_meta
            WHERE hash_id = ?
        """, (hash_id,))
        
        meta_row = cursor.fetchone()
        conn.close()
        
        result = {
            'hash_id': hash_id,
            'progression': json.loads(row[0]),
            'total_events': row[1],
            'chord_events': row[2],
            'source_file': row[3]
        }
        
        if meta_row:
            result.update({
                'song_id': meta_row[0],
                'duration': meta_row[1],
                'num_segments': meta_row[2],
                'selected_stem': meta_row[3],
                'available_stems': json.loads(meta_row[4]),
                'midi_path': meta_row[5]
            })
        
        return result
    
    def query_by_stem(self, stem_type: str, limit: int = 10) -> List[Dict[str, Any]]:
        """ステムタイプで検索"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT song_id, hash_id, duration, selected_stem, midi_path
            FROM moisesdb_meta
            WHERE selected_stem = ?
            LIMIT ?
        """, (stem_type, limit))
        
        rows = cursor.fetchall()
        conn.close()
        
        return [
            {
                'song_id': row[0],
                'hash_id': row[1],
                'duration': row[2],
                'selected_stem': row[3],
                'midi_path': row[4]
            }
            for row in rows
        ]
    
    def query_by_duration(
        self,
        min_duration: float,
        max_duration: float,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """曲長で検索"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT song_id, hash_id, duration, selected_stem
            FROM moisesdb_meta
            WHERE duration BETWEEN ? AND ?
            ORDER BY duration
            LIMIT ?
        """, (min_duration, max_duration, limit))
        
        rows = cursor.fetchall()
        conn.close()
        
        return [
            {
                'song_id': row[0],
                'hash_id': row[1],
                'duration': row[2],
                'selected_stem': row[3]
            }
            for row in rows
        ]
    
    def get_statistics(self) -> Dict[str, Any]:
        """データベース統計（LAMDA互換）"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 全体統計
        cursor.execute("SELECT COUNT(*) FROM progressions")
        total_progressions = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM moisesdb_meta")
        total_songs = cursor.fetchone()[0]
        
        # ステム別カウント
        cursor.execute("""
            SELECT selected_stem, COUNT(*)
            FROM moisesdb_meta
            GROUP BY selected_stem
        """)
        stem_counts = dict(cursor.fetchall())
        
        # 平均曲長
        cursor.execute("SELECT AVG(duration) FROM moisesdb_meta")
        avg_duration = cursor.fetchone()[0] or 0.0
        
        conn.close()
        
        return {
            'total_progressions': total_progressions,
            'total_songs': total_songs,
            'stem_counts': stem_counts,
            'avg_duration': avg_duration
        }
    
    def export_to_lamda_format(self, output_path: Path):
        """LAMDAフォーマットでエクスポート"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT p.hash_id, p.progression, m.song_id, m.selected_stem
            FROM progressions p
            JOIN moisesdb_meta m ON p.hash_id = m.hash_id
        """)
        
        rows = cursor.fetchall()
        conn.close()
        
        # JSONL形式で出力
        with open(output_path, 'w', encoding='utf-8') as f:
            for row in rows:
                entry = {
                    'hash_id': row[0],
                    'progression': json.loads(row[1]),
                    'song_id': row[2],
                    'stem': row[3]
                }
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
        
        print(f"✅ Exported {len(rows)} entries to {output_path}")


# ========== CLI ==========

def main():
    parser = argparse.ArgumentParser(
        description="MoisesDB WAV → LAMDA統合DB構築"
    )
    parser.add_argument(
        '--input-dir',
        type=Path,
        help='MoisesDBディレクトリ（song_XXX/ を含む）'
    )
    parser.add_argument(
        '--output-db',
        type=Path,
        default=Path('data/moisesdb_unified.db'),
        help='出力SQLiteデータベース'
    )
    parser.add_argument(
        '--midi-output-dir',
        type=Path,
        default=Path('data/moisesdb_midi'),
        help='MIDI出力ディレクトリ'
    )
    parser.add_argument(
        '--max-songs',
        type=int,
        default=-1,
        help='処理する最大曲数（-1=全曲）'
    )
    parser.add_argument(
        '--sr',
        type=int,
        default=22050,
        help='リサンプリングレート'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='詳細ログ'
    )
    
    # クエリモード
    parser.add_argument(
        '--query-mode',
        choices=['hash', 'stem', 'duration', 'stats'],
        help='クエリモード（処理済みDB検索）'
    )
    parser.add_argument('--hash-id', type=str, help='検索するhash_id')
    parser.add_argument('--stem', type=str, help='検索するステムタイプ')
    parser.add_argument('--min-duration', type=float, help='最小曲長（秒）')
    parser.add_argument('--max-duration', type=float, help='最大曲長（秒）')
    parser.add_argument('--limit', type=int, default=10, help='検索結果上限')
    
    # audio_chordmap.yaml生成モード
    parser.add_argument(
        '--generate-chordmap',
        action='store_true',
        help='audio_chordmap.yaml生成モード'
    )
    parser.add_argument(
        '--song-dir',
        type=Path,
        help='曲ディレクトリ（chordmap生成用）'
    )
    parser.add_argument(
        '--chordmap-output',
        type=Path,
        help='出力YAMLパス（デフォルト: {song_dir}/audio_chordmap.yaml）'
    )
    parser.add_argument(
        '--use-spectral-analysis',
        action='store_true',
        help='スペクトル解析による自動ロール判定を使用'
    )
    
    args = parser.parse_args()
    
    # Integratorインスタンス
    integrator = MoisesDBIntegrator(
        db_path=args.output_db,
        midi_output_dir=args.midi_output_dir,
        sr=args.sr
    )
    
    # audio_chordmap.yaml生成モード
    if args.generate_chordmap:
        if not args.song_dir or not args.song_dir.exists():
            print("❌ --song-dir required and must exist for chordmap generation")
            return
        
        output_yaml = args.chordmap_output or (args.song_dir / 'audio_chordmap.yaml')
        
        print(f"\n{'='*70}")
        print("Generating audio_chordmap.yaml")
        print(f"{'='*70}")
        print(f"Song dir: {args.song_dir}")
        print(f"Output: {output_yaml}")
        print(f"Spectral analysis: {args.use_spectral_analysis}")
        print(f"{'='*70}")
        
        yaml_data = integrator.generate_audio_chordmap_yaml(
            song_dir=args.song_dir,
            output_yaml_path=output_yaml,
            use_spectral_analysis=args.use_spectral_analysis
        )
        
        print(f"\n✅ Generated audio_chordmap.yaml:")
        print(f"   Total stems: {yaml_data['metadata']['total_stems']}")
        print(f"   Harmonic stems: {yaml_data['metadata']['harmonic_stems']}")
        print(f"   Weights:")
        for stem, info in yaml_data['stems'].items():
            weight = info['weight']
            role = info['role']
            emoji = '🎹' if weight > 0 else '❌'
            print(f"     {emoji} {stem}: {weight:.2f} ({role})")
        
        return
    
    # クエリモード
    if args.query_mode:
        if args.query_mode == 'hash':
            if not args.hash_id:
                print("❌ --hash-id required for hash query")
                return
            result = integrator.query_by_hash(args.hash_id)
            print(json.dumps(result, indent=2, ensure_ascii=False))
        
        elif args.query_mode == 'stem':
            if not args.stem:
                print("❌ --stem required for stem query")
                return
            results = integrator.query_by_stem(args.stem, args.limit)
            print(json.dumps(results, indent=2, ensure_ascii=False))
        
        elif args.query_mode == 'duration':
            if args.min_duration is None or args.max_duration is None:
                print("❌ --min-duration and --max-duration required")
                return
            results = integrator.query_by_duration(
                args.min_duration,
                args.max_duration,
                args.limit
            )
            print(json.dumps(results, indent=2, ensure_ascii=False))
        
        elif args.query_mode == 'stats':
            stats = integrator.get_statistics()
            print(json.dumps(stats, indent=2, ensure_ascii=False))
        
        return
    
    # 処理モード
    if not args.input_dir:
        print("❌ --input-dir required for processing mode")
        return
    
    results = integrator.process_dataset(
        input_dir=args.input_dir,
        max_songs=args.max_songs,
        verbose=args.verbose
    )
    
    # サマリー出力
    print(f"\n{'='*70}")
    print("Processing Summary")
    print(f"{'='*70}")
    print(f"✅ Success: {results['success']}")
    print(f"⚠️  Skipped: {results['skipped']}")
    print(f"❌ Failed: {results['failed']}")
    print(f"{'='*70}")
    
    # メタデータJSONL出力
    meta_output = args.output_db.with_suffix('.jsonl')
    with open(meta_output, 'w', encoding='utf-8') as f:
        for song in results['processed_songs']:
            f.write(json.dumps(song, ensure_ascii=False) + '\n')
    
    print(f"📄 Metadata saved: {meta_output}")
    
    # 統計表示
    stats = integrator.get_statistics()
    print(f"\n📊 Database Statistics:")
    print(f"  Total songs: {stats['total_songs']:,}")
    print(f"  Total progressions: {stats['total_progressions']:,}")
    print(f"  Avg duration: {stats['avg_duration']:.2f}s")
    print(f"  Stem distribution:")
    for stem, count in sorted(stats['stem_counts'].items()):
        print(f"    - {stem}: {count:,}")


if __name__ == '__main__':
    main()
