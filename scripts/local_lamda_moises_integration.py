#!/usr/bin/env python3
"""
LOCAL LAMDA MoisesDB WAV版統合システム
- 細粒度ステム（guitar/piano/keys/strings等）対応
- 複数セグメント統合（manifest形式）
- vocals/mix除外、非GT明示

Usage:
    python scripts/local_lamda_moises_integration.py \
        --input-dir data/Los-Angeles-MIDI/LOCAL_LAMDA/wav_version/moisesdb/moisesdb_v0.1 \
        --output-db data/moisesdb_wav_unified.db \
        --source-name moisesdb \
        --policy-yaml config/stem_policy.yaml \
        --max-songs 2 \
        --verbose
"""

import argparse
import hashlib
import json
import os
import sqlite3
import yaml
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import librosa
import numpy as np
import soundfile as sf
import pandas as pd


# ========== Stem Policy Loader ==========

def load_stem_policy(policy_path: Path, source_name: Optional[str] = None) -> Dict[str, Any]:
    """
    stem_policy.yaml読み込み（プロファイル対応）
    
    Args:
        policy_path: YAMLファイルパス
        source_name: データセット名（moisesdb/musdb18/auto等）
                     'auto'の場合は後でdetect_profile()で推定
    
    Returns:
        ポリシー辞書（version 1形式 or version 2のプロファイル展開後）
        + メタ情報: _profile_name, _policy_version, _weights_digest
    """
    with open(policy_path, 'r', encoding='utf-8') as f:
        policy_data = yaml.safe_load(f)
    
    # Version 2（プロファイル対応）の場合
    if policy_data.get('version') == 2 and 'profiles' in policy_data:
        profiles = policy_data['profiles']
        default_profile = policy_data.get('default_profile', 'moisesdb')
        
        # source_nameからプロファイル選択
        if source_name == 'auto':
            # 後でdetect_profile()で推定する用にデフォルトを返す
            profile_name = default_profile
            print(f"⚙️  Policy: source_name='auto', will detect from stems later (using '{profile_name}' as fallback)")
        elif source_name in profiles:
            profile_name = source_name
        else:
            profile_name = default_profile
            if source_name:
                print(f"⚠️  Policy: source_name='{source_name}' not found, using default '{profile_name}'")
        
        if profile_name not in profiles:
            raise ValueError(
                f"Profile '{profile_name}' not found in policy. "
                f"Available: {list(profiles.keys())}"
            )
        
        selected = profiles[profile_name].copy()
        
        # メタ情報を埋め込み
        selected['_profile_name'] = profile_name
        selected['_policy_version'] = policy_data.get('version', 2)
        
        # weights_digest（再現性確保用）
        harmony_weights = selected.get('weights', {}).get('harmony', {})
        weights_str = ','.join(f"{k}:{v:.2f}" for k, v in sorted(harmony_weights.items()))
        selected['_weights_digest'] = weights_str
        
        # ログ出力（詳細情報）
        print(f"📋 [Policy] profile={profile_name} v{selected['_policy_version']}")
        print(f"   harmony={{{weights_str}}}")
        print(f"   exclude_for_harmony={selected.get('exclude_for_harmony', [])}")
        
        # ランタイム検証（assertions）
        _validate_policy(selected, profile_name)
        
        return selected
    
    # Version 1（従来形式）の場合
    policy_data['_profile_name'] = 'legacy_v1'
    policy_data['_policy_version'] = 1
    policy_data['_weights_digest'] = 'legacy'
    return policy_data


def _validate_policy(policy: Dict[str, Any], profile_name: str):
    """ポリシーのランタイム検証（軽量ユニットテスト）"""
    try:
        if profile_name == 'musdb18':
            # MUSDB18: other優先/beatはdrums最優先
            harmony_weights = policy.get('weights', {}).get('harmony', {})
            assert harmony_weights.get('other', 0) > harmony_weights.get('bass', 0), \
                "MUSDB18: other weight should be > bass"
            
            beat_priority = policy.get('roles_priority', {}).get('beat', [])
            assert beat_priority[0] == 'drums', "MUSDB18: drums should be first in beat priority"
            
        elif profile_name == 'moisesdb':
            # MoisesDB: guitar/pianoが先頭
            harmony_priority = policy.get('roles_priority', {}).get('harmony', [])
            assert 'guitar' in harmony_priority[:3] and 'piano' in harmony_priority[:3], \
                "MoisesDB: guitar/piano should be in top 3 harmony priority"
            
            exclude = policy.get('exclude_for_harmony', [])
            assert 'drums' in exclude, "MoisesDB: drums should be excluded from harmony"
        
        print(f"   ✅ Policy validation passed for '{profile_name}'")
    except AssertionError as e:
        print(f"   ⚠️  Policy validation warning: {e}")


def detect_profile_from_stems(stem_names: Set[str], available_profiles: List[str]) -> str:
    """
    検出したステム名集合からプロファイルを自動推定
    
    Args:
        stem_names: 検出されたステム名のセット（例: {'vocals', 'drums', 'bass', 'other'}）
        available_profiles: 利用可能なプロファイル名のリスト
    
    Returns:
        推定されたプロファイル名
    """
    stem_names_lower = {s.lower() for s in stem_names}
    
    # MUSDB18パターン: vocals, drums, bass, other（またはmixture/mix）の4ステム
    musdb18_pattern = {'vocals', 'drums', 'bass', 'other'}
    musdb18_pattern_alt = {'vocals', 'drums', 'bass', 'mixture'}
    
    # MoisesDBパターン: guitar/piano/keys等の細粒度ステムが含まれる
    moisesdb_indicators = {'guitar', 'piano', 'keys', 'other_keys', 'strings', 'percussion'}
    
    # MUSDB18判定
    if stem_names_lower == musdb18_pattern or stem_names_lower == musdb18_pattern_alt:
        if 'musdb18' in available_profiles:
            print(f"🔍 Auto-detected: MUSDB18 (4-stem pattern: {sorted(stem_names_lower)})")
            return 'musdb18'
    
    # MUSDB18亜種（mix含む）
    if stem_names_lower & musdb18_pattern == musdb18_pattern and len(stem_names_lower) <= 5:
        if 'musdb18' in available_profiles:
            print(f"🔍 Auto-detected: MUSDB18-like (stems: {sorted(stem_names_lower)})")
            return 'musdb18'
    
    # MoisesDB判定
    if stem_names_lower & moisesdb_indicators:
        if 'moisesdb' in available_profiles:
            print(f"🔍 Auto-detected: MoisesDB (fine-grained stems: {sorted(stem_names_lower & moisesdb_indicators)})")
            return 'moisesdb'
    
    # 判定不能の場合
    print(f"⚠️  Could not auto-detect profile from stems: {sorted(stem_names_lower)}")
    return available_profiles[0] if available_profiles else 'moisesdb'


def match_stem_role(stem_name: str, alias_map: Dict[str, List[str]]) -> Optional[str]:
    """ステム名からroleをマッピング"""
    stem_lower = stem_name.lower()
    for role, patterns in alias_map.items():
        for pattern in patterns:
            if pattern in stem_lower:
                return role
    return stem_name.lower()  # パターンに一致しない場合はそのまま使用


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
    role: str,
    segments: List[Dict[str, Any]],
    sr: int,
    channels: int,
    is_ground_truth: bool = False
) -> Dict[str, Any]:
    """Content-based file_id用のマニフェスト生成（セグメント対応）"""
    return {
        "version": "ok-audio-1.0",
        "song_id": song_id,
        "role": role,
        "sr": int(sr),
        "channels": int(channels),
        "is_ground_truth": is_ground_truth,
        "provenance": "MoisesDB",
        "segments": segments
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
    """WAV音声特徴量抽出（セグメント対応）"""
    
    def __init__(self, sr: int = 22050):
        self.sr = sr
    
    def extract_features_from_segments(
        self,
        segment_paths: List[Path],
        verbose: bool = False
    ) -> Dict[str, Any]:
        """複数セグメントから特徴量抽出（結合せずに個別処理）"""
        if verbose and len(segment_paths) > 1:
            print(f"   📊 Processing {len(segment_paths)} segments")
        
        # 全セグメント読み込み＆結合
        y_segments = []
        for seg_path in segment_paths:
            y_seg, _ = librosa.load(str(seg_path), sr=self.sr, mono=True)
            y_segments.append(y_seg)
        
        y = np.concatenate(y_segments)
        duration = len(y) / self.sr
        
        features = {
            'duration': duration,
            'sample_rate': self.sr,
            'num_segments': len(segment_paths)
        }
        
        # 1. テンポ＆ビート
        try:
            tempo, beats = librosa.beat.beat_track(y=y, sr=self.sr)
            beat_times = librosa.frames_to_time(beats, sr=self.sr).tolist()
            features['tempo'] = float(tempo)
            features['num_beats'] = len(beats)
            features['beat_times'] = beat_times
            features['beat_frames'] = beats.tolist() if hasattr(beats, 'tolist') else list(beats)
        except Exception as e:
            if verbose:
                print(f"   ⚠️ Beat detection failed: {e}")
            features['tempo'] = None
            features['num_beats'] = 0
            features['beat_times'] = []
            features['beat_frames'] = []
        
        # 2. オンセット検出
        try:
            onset_frames = librosa.onset.onset_detect(y=y, sr=self.sr)
            onset_times = librosa.frames_to_time(onset_frames, sr=self.sr).tolist()
            features['num_onsets'] = len(onset_frames)
            features['onset_times'] = onset_times
        except:
            features['num_onsets'] = 0
            features['onset_times'] = []
        
        # 3. クロマ特徴量（コード候補）
        try:
            chroma = librosa.feature.chroma_cqt(y=y, sr=self.sr)
            chroma_mean = np.mean(chroma, axis=1)
            top_pitches = np.argsort(chroma_mean)[-3:][::-1]
            pitch_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
            features['chord_candidates'] = [pitch_names[i] for i in top_pitches]
        except:
            features['chord_candidates'] = []
        
        # 4. Activity（RMS）
        try:
            rms = librosa.feature.rms(y=y)
            features['activity_mean'] = float(np.mean(rms))
            features['activity_std'] = float(np.std(rms))
        except:
            features['activity_mean'] = 0.0
            features['activity_std'] = 0.0
        
        # 5. スペクトル特徴量
        try:
            spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=self.sr)
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=self.sr)
            features['spectral_centroid_mean'] = float(np.mean(spectral_centroid))
            features['spectral_rolloff_mean'] = float(np.mean(spectral_rolloff))
        except:
            features['spectral_centroid_mean'] = 0.0
            features['spectral_rolloff_mean'] = 0.0
        
        if verbose:
            print(f"   Onsets: {features['num_onsets']}")
            if features['chord_candidates']:
                print(f"   Chord candidates: {features['chord_candidates']}")
        
        return features


# ========== MoisesDB Integrator ==========

class MoisesDBIntegrator:
    """MoisesDB統合処理"""
    
    def __init__(
        self,
        db_path: Path,
        wav_features_dir: Path,
        source_name: str,
        policy: Dict[str, Any],
        sr: int = 22050
    ):
        self.db_path = db_path
        self.wav_features_dir = wav_features_dir
        self.source_name = source_name
        self.policy = policy
        self.sr = sr
        self.extractor = WAVFeatureExtractor(sr=sr)
        # aggregated outputs for vocals/mix (dataset-level)
        self.vocal_features_agg: List[Dict[str, Any]] = []
        self.mix_diagnostics_agg: List[Dict[str, Any]] = []
        
        # データベース初期化
        self._init_database()
    
    def _init_database(self):
        """SQLiteデータベース初期化"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS wav_features (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                song_id TEXT NOT NULL,
                file_id TEXT NOT NULL,
                file_path TEXT,
                duration REAL,
                tempo REAL,
                num_beats INTEGER,
                num_onsets INTEGER,
                chord_candidates TEXT,
                activity_mean REAL,
                activity_std REAL,
                spectral_centroid_mean REAL,
                spectral_rolloff_mean REAL,
                num_segments INTEGER,
                manifest TEXT,
                features_json TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(song_id, file_id)
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
        """楽曲ディレクトリ処理（MoisesDB構造）"""
        song_id = song_dir.name
        
        # 処理済みチェック: bars.parquetが存在すればスキップ
        out_dir = Path(self.wav_features_dir) / self.source_name / song_id
        bars_parquet = out_dir / f"{song_id}.bars.parquet"
        if bars_parquet.exists():
            if verbose:
                print(f"⏭️  Skipping (already processed): {song_id}")
            return None
        
        # data.json読み込み
        data_json_path = song_dir / "data.json"
        metadata = {}
        if data_json_path.exists():
            try:
                metadata = json.loads(data_json_path.read_text(encoding='utf-8'))
            except:
                pass
        
        if verbose:
            print(f"\n{'='*70}")
            print(f"Processing: {metadata.get('song', song_id)} - {metadata.get('artist', 'Unknown')}")
            print(f"Song ID: {song_id}")
            print(f"{'='*70}")
        
        # ステムディレクトリ走査（MoisesDB形式）またはWAVファイル直接走査（MUSDB18形式）
        stem_dirs = [d for d in song_dir.iterdir() if d.is_dir()]
        direct_wav_files = list(song_dir.glob("*.wav"))
        
        # vocals/mixを除外するステムリスト
        exclude_stems = set(self.policy.get('exclude_for_harmony', ['mix', 'vocals', 'drums', 'percussion']))
        
        processed_stems = []
        role_features: Dict[str, Dict[str, Any]] = {}
        
        # MUSDB18形式（直接WAVファイル）の処理
        if not stem_dirs and direct_wav_files:
            for wav_file in direct_wav_files:
                stem_name = wav_file.stem  # .wavを除いたファイル名
                
                # role判定
                role = match_stem_role(stem_name, self.policy['alias_map'])
                
                # vocals/mixはmain harmony集合から除外するが、別途解析して保存
                if role in exclude_stems:
                    if role in ('vocals', 'mix'):
                        if verbose:
                            print(f"🔎  Processing (separate): {stem_name} (role: {role})")
                        # セグメント情報
                        size = os.path.getsize(wav_file)
                        sha = sha256_file(str(wav_file))
                        segments = [{
                            "relpath": wav_file.name,
                            "size": size,
                            "sha256": sha,
                            "start_sec": 0.0
                        }]
                        # 特徴量抽出
                        wav_features = self.extractor.extract_features_from_segments([wav_file], verbose=False)
                        # 保存（ファイル出力は別途集約）
                        entry = {
                            'song_id': song_id,
                            'role': role,
                            'segments': segments,
                            'features': wav_features
                        }
                        if role == 'vocals':
                            self.vocal_features_agg.append(entry)
                        elif role == 'mix':
                            self.mix_diagnostics_agg.append(entry)
                        # role_featuresにも追加（beat/chordmap生成用）
                        role_features[role] = {
                            'segments': segments,
                            'wav_files': [str(wav_file)],
                            'wav_features': wav_features,
                            'file_id': sha[:12]
                        }
                    else:
                        if verbose:
                            print(f"⏭️  Skipping: {stem_name} (excluded: {role})")
                    continue
                
                # Harmony stem (piano, guitar, bass, etc.)
                if verbose:
                    print(f"🎵 Processing stem: {stem_name} → role: {role}")
                
                # セグメント情報
                size = os.path.getsize(wav_file)
                sha = sha256_file(str(wav_file))
                segments = [{
                    "relpath": wav_file.name,
                    "size": size,
                    "sha256": sha,
                    "start_sec": 0.0
                }]
                
                # 特徴量抽出
                wav_features = self.extractor.extract_features_from_segments([wav_file], verbose=False)
                
                # file_id
                file_id = sha[:12]
                
                if verbose:
                    duration = wav_features.get('duration', 0)
                    print(f"   → File ID: {file_id}, Duration: {duration:.1f}s")
                
                # store processed stem info for Stage3 aggregation
                processed_stems.append({
                    'role': role,
                    'file_id': file_id,
                    'segments': segments,
                    'wav_features': wav_features
                })
                
                # store role-level features for later Stage2 outputs
                role_features[role] = {
                    'segments': segments,
                    'wav_files': [str(wav_file)],
                    'wav_features': wav_features,
                    'file_id': file_id
                }
        
        # MoisesDB形式（サブディレクトリ）の処理
        for stem_dir in stem_dirs:
            stem_name = stem_dir.name
            
            # role判定
            role = match_stem_role(stem_name, self.policy['alias_map'])
            
            # vocals/mixはmain harmony集合から除外するが、別途解析して保存
            if role in exclude_stems:
                if role in ('vocals', 'mix'):
                    if verbose:
                        print(f"🔎  Processing (separate): {stem_name} (role: {role})")
                    # WAVファイル収集
                    wav_files = sorted(stem_dir.glob("*.wav"))
                    if not wav_files:
                        continue
                    # セグメント情報
                    segments = []
                    for wav_path in wav_files:
                        size = os.path.getsize(wav_path)
                        sha = sha256_file(str(wav_path))
                        segments.append({
                            "relpath": f"{stem_name}/{wav_path.name}",
                            "size": size,
                            "sha256": sha,
                            "start_sec": 0.0
                        })
                    # 特徴量抽出
                    wav_features = self.extractor.extract_features_from_segments(wav_files, verbose=False)
                    # 保存（ファイル出力は別途集約）
                    entry = {
                        'song_id': song_id,
                        'role': role,
                        'segments': segments,
                        'features': wav_features
                    }
                    if role == 'vocals':
                        self.vocal_features_agg.append(entry)
                    else:
                        self.mix_diagnostics_agg.append(entry)
                    # continue main loop (do not add to processed_stems)
                    continue
                else:
                    if verbose:
                        print(f"⏭️  Skipping: {stem_name} (excluded: {role})")
                    continue
            
            # WAVファイル収集
            wav_files = sorted(stem_dir.glob("*.wav"))
            if not wav_files:
                continue
            
            if verbose:
                print(f"🎵 Processing stem: {stem_name} → role: {role} ({len(wav_files)} segments)")
            
            # セグメント情報構築
            segments = []
            for i, wav_path in enumerate(wav_files):
                size = os.path.getsize(wav_path)
                sha = sha256_file(str(wav_path))
                segments.append({
                    "relpath": f"{stem_name}/{wav_path.name}",
                    "size": size,
                    "sha256": sha,
                    "start_sec": 0.0  # data.jsonから取得可能なら更新
                })
            
            # 特徴量抽出
            wav_features = self.extractor.extract_features_from_segments(wav_files, verbose=False)
            
            # サンプルレート/チャンネル取得
            try:
                info = sf.info(str(wav_files[0]))
                sr = info.samplerate
                channels = info.channels
            except:
                sr = self.sr
                channels = 1
            
            # Content-based file_id生成
            canonical_manifest = build_canonical_manifest(
                song_id=song_id,
                role=role,
                segments=segments,
                sr=sr,
                channels=channels,
                is_ground_truth=False  # MoisesDBは非GT
            )
            file_id = compute_file_id(canonical_manifest)
            
            if verbose:
                print(f"   → File ID: {file_id}, Duration: {wav_features['duration']:.1f}s")
            
            # データベース保存
            self._save_to_database(
                song_id=song_id,
                file_id=file_id,
                wav_path=stem_dir,
                wav_features=wav_features,
                canonical_manifest=canonical_manifest,
                role=role
            )
            
            # WAV特徴量JSON保存
            self._save_features_json(song_id, file_id, wav_features, role)
            
            processed_stems.append({
                'role': role,
                'file_id': file_id,
                'num_segments': len(segments),
                'duration': wav_features['duration']
            })
            # store role-level features for later Stage2 outputs
            role_features[role] = {
                'segments': segments,
                'wav_files': [str(p) for p in wav_files],
                'wav_features': wav_features,
                'file_id': file_id
            }
        
        if not processed_stems:
            if verbose:
                print(f"⚠️ No valid stems found in {song_id}")
            return None
        
        if verbose:
            print(f"✅ Processed {len(processed_stems)} stems: {[s['role'] for s in processed_stems]}")

        # ---------- Stage2 outputs (beat grid / accent grid / chordmap / bars) ----------
        # Choose beat source according to policy (beat priority, fallback to mix)
        beat_source_role = None
        for r in self.policy.get('roles_priority', {}).get('beat', []):
            if r in role_features:
                beat_source_role = r
                break
        if not beat_source_role and 'mix' in [e['role'] for e in self.mix_diagnostics_agg]:
            beat_source_role = 'mix'

        beat_times = []
        if beat_source_role and beat_source_role in role_features:
            beat_times = role_features[beat_source_role]['wav_features'].get('beat_times', [])
        # write beat_grid.json
        out_dir = Path(self.wav_features_dir) / self.source_name / song_id
        out_dir.mkdir(parents=True, exist_ok=True)
        beat_grid_path = out_dir / 'beat_grid.json'
        with open(beat_grid_path, 'w', encoding='utf-8') as f:
            json.dump({'beat_times': beat_times}, f, ensure_ascii=False, indent=2)

        # accent grid: simple heuristic using onset counts near beats
        accent_grid = []
        onset_times_all = []
        for rinfo in role_features.values():
            onset_times_all.extend(rinfo['wav_features'].get('onset_times', []))
        onset_times_all = sorted(onset_times_all)
        for t in beat_times:
            # count onsets within ±0.2s
            cnt = sum(1 for o in onset_times_all if abs(o - t) <= 0.2)
            weight = 1.0 + 0.1 * cnt
            accent_grid.append({'time': t, 'weight': weight})
        accent_grid_path = out_dir / 'accent_grid.json'
        with open(accent_grid_path, 'w', encoding='utf-8') as f:
            json.dump({'accents': accent_grid}, f, ensure_ascii=False, indent=2)

        # audio_chordmap.yaml: collect harmony roles and apply policy weights
        harmony_roles = self.policy.get('roles_priority', {}).get('harmony', [])
        chordmap = {
            'song_id': song_id,
            'chordmap': [],
            # メタデータ埋め込み（追跡性・再現性確保）
            'policy_metadata': {
                'profile': self.policy.get('_profile_name', 'unknown'),
                'version': self.policy.get('_policy_version', 1),
                'weights_digest': self.policy.get('_weights_digest', 'n/a'),
                'exclude_for_harmony': self.policy.get('exclude_for_harmony', [])
            }
        }
        default_weights = self.policy.get('weights', {}).get('harmony', {})
        for hr in harmony_roles:
            if hr in role_features:
                cands = role_features[hr]['wav_features'].get('chord_candidates', [])
                w = default_weights.get(hr, 0.1)
                chordmap['chordmap'].append({'role': hr, 'weight': float(w), 'chord_candidates': cands})
        chordmap_path = out_dir / 'audio_chordmap.yaml'
        with open(chordmap_path, 'w', encoding='utf-8') as f:
            yaml.safe_dump(chordmap, f, allow_unicode=True, sort_keys=False)

        # bars.parquet (simple 4/4 grouping of beats)
        try:
            if beat_times:
                bars = []
                n = len(beat_times)
                beats_per_bar = 4
                import math
                n_bars = math.ceil(n / beats_per_bar)
                for bi in range(n_bars):
                    start_idx = bi * beats_per_bar
                    end_idx = min((bi + 1) * beats_per_bar, n)
                    bar_beats = beat_times[start_idx:end_idx]
                    if not bar_beats:
                        continue
                    start_sec = bar_beats[0]
                    end_sec = bar_beats[-1]
                    bars.append({'bar_index': bi, 'start_sec': start_sec, 'end_sec': end_sec, 'beats': bar_beats, 'song_id': song_id})
                if bars:
                    df = pd.DataFrame(bars)
                    bars_path = out_dir / f"{song_id}.bars.parquet"
                    df.to_parquet(bars_path)
        except Exception as e:
            if verbose:
                print(f"⚠️ Failed to write bars.parquet for {song_id}: {e}")

        return {
            'status': 'success',
            'song_id': song_id,
            'stems': processed_stems,
            'metadata': metadata
        }
    
    def _save_to_database(
        self,
        song_id: str,
        file_id: str,
        wav_path: Path,
        wav_features: Dict,
        canonical_manifest: Dict,
        role: str = "unknown"
    ):
        """データベース保存"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO wav_features
            (song_id, file_id, file_path, duration, tempo, num_beats, num_onsets,
             chord_candidates, activity_mean, activity_std, spectral_centroid_mean,
             spectral_rolloff_mean, num_segments, manifest, features_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            f"{song_id}#{role}",
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
            wav_features.get('num_segments', 1),
            json.dumps(canonical_manifest),
            json.dumps(wav_features)
        ))
        
        conn.commit()
        conn.close()
    
    def _save_features_json(self, song_id: str, file_id: str, features: Dict, role: str = "unknown"):
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
        song_dirs = [d for d in input_dir.iterdir() if d.is_dir()]
        
        if max_songs > 0:
            song_dirs = song_dirs[:max_songs]
        
        print(f"\n{'='*70}")
        print(f"LOCAL LAMDA MoisesDB Integration - {self.source_name}")
        print(f"{'='*70}")
        print(f"Input dir: {input_dir}")
        print(f"Total song directories: {len(song_dirs)}")
        print(f"Output DB: {self.db_path}")
        print(f"Policy: {self.policy.get('version', 'custom')}")
        print(f"{'='*70}")
        
        results = {
            'source': self.source_name,
            'success': 0,
            'failed': 0,
            'skipped': 0,
            'total_stems': 0,
            'processed_songs': []
        }
        
        for i, song_dir in enumerate(song_dirs, 1):
            if verbose:
                print(f"\n[{i}/{len(song_dirs)}]")
            
            try:
                result = self.process_song_directory(song_dir, verbose)
                
                if result is None:
                    # スキップ（処理済み）
                    results['skipped'] += 1
                elif result and result['status'] == 'success':
                    results['success'] += 1
                    results['total_stems'] += len(result['stems'])
                    results['processed_songs'].append(result)
                else:
                    results['failed'] += 1
            
            except Exception as e:
                print(f"❌ Failed to process {song_dir.name}: {e}")
                results['failed'] += 1
        
        # ---------- Write aggregated vocal / mix diagnostics (dataset-level) ----------
        try:
            agg_dir = Path(self.wav_features_dir) / self.source_name
            agg_dir.mkdir(parents=True, exist_ok=True)

            if self.vocal_features_agg:
                rows = []
                for e in self.vocal_features_agg:
                    f = e['features']
                    rows.append({
                        'song_id': e['song_id'],
                        'role': e['role'],
                        'tempo': f.get('tempo'),
                        'num_beats': f.get('num_beats'),
                        'num_onsets': f.get('num_onsets'),
                        'duration': f.get('duration'),
                        'chord_candidates': json.dumps(f.get('chord_candidates', [])),
                        'num_segments': len(e.get('segments', [])),
                        'features': json.dumps(f)
                    })
                df_v = pd.DataFrame(rows)
                v_parquet = agg_dir / 'vocal_features.parquet'
                v_csv = agg_dir / 'vocal_features.csv'
                df_v.to_parquet(v_parquet)
                df_v.to_csv(v_csv, index=False)

            if self.mix_diagnostics_agg:
                rows = []
                for e in self.mix_diagnostics_agg:
                    f = e['features']
                    rows.append({
                        'song_id': e['song_id'],
                        'role': e['role'],
                        'tempo': f.get('tempo'),
                        'num_beats': f.get('num_beats'),
                        'num_onsets': f.get('num_onsets'),
                        'duration': f.get('duration'),
                        'chord_candidates': json.dumps(f.get('chord_candidates', [])),
                        'num_segments': len(e.get('segments', [])),
                        'features': json.dumps(f)
                    })
                df_m = pd.DataFrame(rows)
                m_parquet = agg_dir / 'mix_diagnostics.parquet'
                m_csv = agg_dir / 'mix_diagnostics.csv'
                df_m.to_parquet(m_parquet)
                df_m.to_csv(m_csv, index=False)
        except Exception as e:
            if verbose:
                print(f"⚠️ Failed to write aggregated vocal/mix files: {e}")

        return results

    # NOTE: unreachable (kept for clarity)


# ========== CLI ==========

def main():
    parser = argparse.ArgumentParser(
        description="LOCAL LAMDA MoisesDB WAV版統合システム"
    )
    parser.add_argument(
        '--input-dir',
        type=Path,
        required=True,
        help='moisesdb_v0.1ディレクトリ'
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
        default='moisesdb',
        help='データソース名'
    )
    parser.add_argument(
        '--policy-yaml',
        type=Path,
        default=Path('config/stem_policy.yaml'),
        help='ステムポリシーYAML'
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
    
    # ポリシー読み込み（source_nameを渡してプロファイル自動切替）
    if args.policy_yaml.exists():
        policy = load_stem_policy(args.policy_yaml, source_name=args.source_name)
    else:
        print(f"⚠️ Policy file not found: {args.policy_yaml}, using defaults")
        policy = {
            'version': 1,
            'alias_map': {
                'guitar': ['guitar', 'gtr'],
                'bass': ['bass'],
                'drums': ['drums', 'drum'],
                'piano': ['piano'],
                'keys': ['keys', 'keyboard'],
                'vocals': ['vocals', 'vox'],
                'mix': ['mix']
            },
            'exclude_for_harmony': ['mix', 'vocals', 'drums', 'percussion']
        }
    
    # 実行
    integrator = MoisesDBIntegrator(
        db_path=args.output_db,
        wav_features_dir=args.wav_features_dir,
        source_name=args.source_name,
        policy=policy,
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
    print(f"⏭️  Skipped (already done): {results['skipped']}")
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
