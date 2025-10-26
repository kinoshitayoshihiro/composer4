#!/usr/bin/env python3
"""
LOCAL LAMDA WAV版統合システム (Content-based file_id + song_id対応)

楽曲ディレクトリ単位で処理し、SQLiteデータベースに保存。
後続のpickle_builderで5軸pickle生成。

Usage:
    # MUSDB18テスト（2曲）
    python scripts/local_lamda_wav_integration.py \\
        --input-dir data/Los-Angeles-MIDI/LOCAL_LAMDA/wav_version/musdb18_decoded \\
        --output-db data/musdb18_wav_test.db \\
        --source-name musdb18 \\
        --max-songs 2 \\
        --verbose
"""

import argparse
import hashlib
import json
import os
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

import librosa
import numpy as np
import soundfile as sf


# ========== Content-based file_id ==========

def sha256_file(path: str, blocksize: int = 4 * 1024 * 1024) -> str:
    """ファイルのSHA-256ハッシュを計算"""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(blocksize), b""):
            h.update(chunk)
    return h.hexdigest()


def build_canonical_manifest(
    song_id: str,
    wav_path: Path,
    sr: int,
    channels: int,
    role: str = "mix"
) -> Dict[str, Any]:
    """Content-based file_id用のマニフェスト生成"""
    size = os.path.getsize(wav_path)
    sha = sha256_file(str(wav_path))
    
    return {
        "version": "ok-audio-1.0",
        "song_id": song_id,
        "role": role,
        "sr": int(sr),
        "channels": int(channels),
        "segments": [{
            "relpath": wav_path.name,
            "size": size,
            "sha256": sha,
            "start_sec": 0.0
        }]
    }


def compute_file_id(canonical_manifest: Dict[str, Any]) -> str:
    """file_id生成（内容ベース）"""
    payload = json.dumps(
        canonical_manifest,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


# ========== WAV Feature Extractor ==========

class WAVFeatureExtractor:
    """WAV音声特徴量抽出"""
    
    def __init__(self, sr: int = 22050):
        self.sr = sr
    
    def extract_features(self, wav_path: Path, verbose: bool = False) -> Dict[str, Any]:
        """WAV特徴量抽出"""
        if verbose:
            print(f"🎵 Extracting features from: {wav_path.name}")
        
        y, sr = librosa.load(str(wav_path), sr=self.sr, mono=True)
        duration = len(y) / sr
        
        features = {
            'duration': duration,
            'sample_rate': sr,
        }
        
        # 1. テンポ＆ビート
        try:
            tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
            features['tempo'] = float(tempo)
            features['num_beats'] = len(beats)
            if verbose:
                print(f"   Tempo: {tempo:.1f} BPM, Beats: {len(beats)}")
        except Exception as e:
            if verbose:
                print(f"   ⚠️ Beat detection failed: {e}")
            features['tempo'] = None
            features['num_beats'] = 0
        
        # 2. オンセット
        try:
            onset_env = librosa.onset.onset_strength(y=y, sr=sr)
            onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)
            features['num_onsets'] = len(onsets)
            if verbose:
                print(f"   Onsets: {len(onsets)}")
        except Exception as e:
            if verbose:
                print(f"   ⚠️ Onset detection failed: {e}")
            features['num_onsets'] = 0
        
        # 3. コード候補
        try:
            chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
            chroma_mean = np.mean(chroma, axis=1)
            pitch_classes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
            top_pitches_idx = np.argsort(chroma_mean)[-3:][::-1]
            features['chord_candidates'] = [pitch_classes[i] for i in top_pitches_idx]
            if verbose:
                print(f"   Chord candidates: {features['chord_candidates']}")
        except Exception as e:
            if verbose:
                print(f"   ⚠️ Chroma failed: {e}")
            features['chord_candidates'] = []
        
        # 4. Activity
        try:
            rms = librosa.feature.rms(y=y)[0]
            features['activity_mean'] = float(np.mean(rms))
            features['activity_std'] = float(np.std(rms))
        except:
            features['activity_mean'] = 0.0
            features['activity_std'] = 0.0
        
        # 5. Spectral
        try:
            spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
            features['spectral_centroid_mean'] = float(np.mean(spectral_centroid))
            features['spectral_rolloff_mean'] = float(np.mean(spectral_rolloff))
        except:
            features['spectral_centroid_mean'] = 0.0
            features['spectral_rolloff_mean'] = 0.0
        
        return features


# ========== Integrator ==========

class LocalLAMDAIntegrator:
    """LOCAL LAMDA WAV版統合システム"""
    
    def __init__(
        self,
        db_path: Path,
        wav_features_dir: Path,
        source_name: str = "local_wav",
        sr: int = 22050
    ):
        self.db_path = db_path
        self.wav_features_dir = wav_features_dir
        self.source_name = source_name
        self.sr = sr
        
        self.extractor = WAVFeatureExtractor(sr=sr)
        self._init_database()
    
    def _init_database(self):
        """データベース初期化（song_id + file_id + manifest追加）"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS wav_features (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                song_id TEXT NOT NULL,
                file_id TEXT UNIQUE NOT NULL,
                file_path TEXT NOT NULL,
                duration REAL,
                tempo REAL,
                num_beats INTEGER,
                num_onsets INTEGER,
                chord_candidates TEXT,
                activity_mean REAL,
                activity_std REAL,
                spectral_centroid_mean REAL,
                spectral_rolloff_mean REAL,
                manifest TEXT,
                features_json TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_song_id ON wav_features(song_id)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_file_id ON wav_features(file_id)
        """)
        
        conn.commit()
        conn.close()
    
    def process_song_directory(
        self,
        song_dir: Path,
        verbose: bool = True
    ) -> Optional[Dict[str, Any]]:
        """楽曲ディレクトリ処理（GTステム: other/bass/drumsを処理、mix/vocals除外）"""
        # MUSDB18 GTステム優先リスト（mix/vocals除外）
        stem_priority = ['other', 'bass', 'drums', 'percussion', 'guitar', 'piano', 'keys', 'strings']
        
        song_id = song_dir.name
        
        if verbose:
            print(f"\n{'='*70}")
            print(f"Processing: {song_id}")
            print(f"{'='*70}")
        
        processed_stems = []
        
        # ステムファイル探索＆処理
        for stem_role in stem_priority:
            wav_path = song_dir / f"{stem_role}.wav"
            if not wav_path.exists():
                continue
            
            if verbose:
                print(f"🎵 Processing stem: {stem_role}.wav")
            
            # WAV特徴量抽出
            wav_features = self.extractor.extract_features(wav_path, verbose=False)
            
            # サンプルレート/チャンネル取得
            try:
                info = sf.info(str(wav_path))
                sr = info.samplerate
                channels = info.channels
            except:
                sr = self.sr
                channels = 1
            
            # Content-based file_id生成（ステム役割を含む）
            canonical_manifest = build_canonical_manifest(
                song_id=song_id,
                wav_path=wav_path,
                sr=sr,
                channels=channels,
                role=stem_role
            )
            file_id = compute_file_id(canonical_manifest)
            
            if verbose:
                print(f"   → Role: {stem_role}, File ID: {file_id}")
            
            # データベース保存（song_id + role の複合キーで識別）
            self._save_to_database(
                song_id=song_id,
                file_id=file_id,
                wav_path=wav_path,
                wav_features=wav_features,
                canonical_manifest=canonical_manifest,
                role=stem_role
            )
            
            # WAV特徴量JSON保存
            self._save_features_json(song_id, file_id, wav_features, stem_role)
            
            processed_stems.append({
                'role': stem_role,
                'file_id': file_id,
                'duration': wav_features['duration']
            })
        
        if not processed_stems:
            if verbose:
                print(f"⚠️ No GT stems found in {song_id}")
            return None
        
        if verbose:
            print(f"✅ Processed {len(processed_stems)} stems: {[s['role'] for s in processed_stems]}")
        
        return {
            'status': 'success',
            'song_id': song_id,
            'stems': processed_stems
        }
    
    def _save_to_database(
        self,
        song_id: str,
        file_id: str,
        wav_path: Path,
        wav_features: Dict,
        canonical_manifest: Dict,
        role: str = "mix"
    ):
        """データベース保存（role情報を含む）"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO wav_features
            (song_id, file_id, file_path, duration, tempo, num_beats, num_onsets,
             chord_candidates, activity_mean, activity_std, spectral_centroid_mean,
             spectral_rolloff_mean, manifest, features_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            f"{song_id}#{role}",  # song_id + role で一意に識別
            file_id,
            str(wav_path),
            wav_features['duration'],
            wav_features.get('tempo'),
            wav_features.get('num_beats'),
            wav_features.get('num_onsets'),
            json.dumps(wav_features.get('chord_candidates', [])),
            wav_features.get('activity_mean'),
            wav_features.get('activity_std'),
            wav_features.get('spectral_centroid_mean'),
            wav_features.get('spectral_rolloff_mean'),
            json.dumps(canonical_manifest),
            json.dumps(wav_features)
        ))
        
        conn.commit()
        conn.close()
    
    def _save_features_json(self, song_id: str, file_id: str, features: Dict, role: str = "mix"):
        """WAV特徴量JSON保存"""
        json_dir = self.wav_features_dir / self.source_name
        json_dir.mkdir(parents=True, exist_ok=True)
        
        json_path = json_dir / f"{song_id}.{role}.{file_id}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(features, f, ensure_ascii=False, indent=2)
    
    def process_dataset(
        self,
        input_dir: Path,
        max_songs: int = -1,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """データセット処理"""
        # 楽曲ディレクトリ一覧
        song_dirs = [d for d in input_dir.iterdir() if d.is_dir()]
        
        if max_songs > 0:
            song_dirs = song_dirs[:max_songs]
        
        print(f"\n{'='*70}")
        print(f"LOCAL LAMDA WAV Integration - {self.source_name}")
        print(f"{'='*70}")
        print(f"Input dir: {input_dir}")
        print(f"Total song directories: {len(song_dirs)}")
        print(f"Output DB: {self.db_path}")
        print(f"{'='*70}")
        
        results = {
            'source': self.source_name,
            'success': 0,
            'failed': 0,
            'total_stems': 0,
            'processed_songs': []
        }
        
        for i, song_dir in enumerate(song_dirs, 1):
            if verbose:
                print(f"\n[{i}/{len(song_dirs)}]")
            
            try:
                result = self.process_song_directory(song_dir, verbose)
                
                if result and result['status'] == 'success':
                    results['success'] += 1
                    results['total_stems'] += len(result['stems'])
                    results['processed_songs'].append(result)
                else:
                    results['failed'] += 1
            
            except Exception as e:
                print(f"❌ Failed to process {song_dir.name}: {e}")
                results['failed'] += 1
        
        return results


# ========== CLI ==========

def main():
    parser = argparse.ArgumentParser(
        description="LOCAL LAMDA WAV版統合システム"
    )
    parser.add_argument(
        '--input-dir',
        type=Path,
        required=True,
        help='楽曲ディレクトリの親ディレクトリ'
    )
    parser.add_argument(
        '--output-db',
        type=Path,
        required=True,
        help='出力SQLiteデータベース'
    )
    parser.add_argument(
        '--source-name',
        type=str,
        default='local_wav',
        help='データソース名'
    )
    parser.add_argument(
        '--wav-features-dir',
        type=Path,
        default=Path('data/local_lamda_wav_features'),
        help='WAV特徴量JSON出力ディレクトリ'
    )
    parser.add_argument(
        '--max-songs',
        type=int,
        default=-1,
        help='処理する最大楽曲数（-1=全曲）'
    )
    parser.add_argument(
        '--sr',
        type=int,
        default=22050,
        help='サンプリングレート'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='詳細ログ'
    )
    
    args = parser.parse_args()
    
    # 実行
    integrator = LocalLAMDAIntegrator(
        db_path=args.output_db,
        wav_features_dir=args.wav_features_dir,
        source_name=args.source_name,
        sr=args.sr
    )
    
    results = integrator.process_dataset(
        input_dir=args.input_dir,
        max_songs=args.max_songs,
        verbose=args.verbose
    )
    
    # サマリー
    print(f"\n{'='*70}")
    print(f"Processing Summary - {results['source']}")
    print(f"{'='*70}")
    print(f"✅ Songs processed: {results['success']}")
    print(f"🎵 Total stems: {results['total_stems']}")
    print(f"❌ Failed: {results['failed']}")
    print(f"{'='*70}")
    
    # メタデータJSONL
    meta_output = args.output_db.with_suffix('.jsonl')
    with open(meta_output, 'w', encoding='utf-8') as f:
        for item in results['processed_songs']:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"📄 Metadata: {meta_output}")


if __name__ == '__main__':
    main()
