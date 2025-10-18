#!/usr/bin/env python3
"""
Suno Structure Extractor

Suno AIで生成されたstem（vocal.wav + accomp.wav）から音楽構造を抽出し、
Stage2 Generatorで使用可能なYAMLファイルを生成。

Input:
    - vocal.wav: ボーカルstem
    - accomp.wav: 伴奏stem（またはfull mix）

Output YAML:
    - tempo_map: グローバルテンポとビート位置
    - sections: 構造セクション（Intro/Verse/Chorus/Bridge/Outro）
    - chords: コード進行（セクションごと）
    - drums_hits: キック/スネア/ハイハットの位置
    - bass_contour: ベースラインの輪郭（ピッチとタイミング）

Usage:
    python scripts/audio2score/extract_structure.py \\
        --vocal data/suno_stems/song1/vocal.wav \\
        --accomp data/suno_stems/song1/accomp.wav \\
        --output data/suno_structures/song1.yaml
"""

import argparse
import pathlib
from typing import Dict, List, Any, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import yaml

# Audio processing
try:
    import librosa
    import librosa.display
except ImportError:
    print("[ERROR] librosa not installed: pip install librosa")
    exit(1)


class SunoStructureExtractor:
    """Suno stems → 音楽構造YAML抽出器"""
    
    def __init__(
        self,
        vocal_path: Optional[pathlib.Path] = None,
        accomp_path: Optional[pathlib.Path] = None,
        sr: int = 22050,
        verbose: bool = True,
    ):
        """
        Args:
            vocal_path: ボーカルstem WAV
            accomp_path: 伴奏stem WAV
            sr: サンプリングレート
            verbose: 詳細ログ出力
        """
        self.vocal_path = vocal_path
        self.accomp_path = accomp_path
        self.sr = sr
        self.verbose = verbose
        
        self.vocal = None
        self.accomp = None
        self.full_mix = None  # vocal + accomp合成
        
    def load_audio(self):
        """Audio files読み込み"""
        if self.verbose:
            print("📁 Loading audio files...")
        
        if self.vocal_path and self.vocal_path.exists():
            self.vocal, _ = librosa.load(str(self.vocal_path), sr=self.sr, mono=True)
            if self.verbose:
                print(f"  ✓ Vocal: {len(self.vocal)/self.sr:.1f}s")
        
        if self.accomp_path and self.accomp_path.exists():
            self.accomp, _ = librosa.load(str(self.accomp_path), sr=self.sr, mono=True)
            if self.verbose:
                print(f"  ✓ Accomp: {len(self.accomp)/self.sr:.1f}s")
        
        # Full mix作成（両方あれば合成、なければ片方使用）
        if self.vocal is not None and self.accomp is not None:
            # 長さを揃える
            min_len = min(len(self.vocal), len(self.accomp))
            self.full_mix = self.vocal[:min_len] + self.accomp[:min_len]
        elif self.accomp is not None:
            self.full_mix = self.accomp
        elif self.vocal is not None:
            self.full_mix = self.vocal
        else:
            raise ValueError("No audio files provided")
        
        if self.verbose:
            print(f"  ✓ Full mix: {len(self.full_mix)/self.sr:.1f}s")
    
    def extract_tempo_map(self) -> Dict[str, Any]:
        """
        テンポ・ビート情報抽出
        
        Returns:
            {
                'global_tempo': float,  # BPM
                'beat_times': List[float],  # Beat positions (seconds)
                'downbeat_times': List[float],  # Downbeat positions
                'time_signature': [int, int],  # [numerator, denominator]
                'tempo_confidence': float  # Beat tracking confidence (0.0-1.0)
            }
        """
        if self.verbose:
            print("\n🎵 Extracting tempo map...")
        
        # テンポ推定（accomp優先、なければfull_mix）
        audio = self.accomp if self.accomp is not None else self.full_mix
        
        # Dynamic tempoトラッキング
        tempo_confidence = 0.0
        try:
            # Onset strength envelope
            onset_env = librosa.onset.onset_strength(y=audio, sr=self.sr)
            
            tempo, beat_frames = librosa.beat.beat_track(
                onset_envelope=onset_env,
                sr=self.sr,
                start_bpm=120,
                units='frames'
            )
            
            # Beat times（秒単位）
            beat_times = librosa.frames_to_time(beat_frames, sr=self.sr).tolist()
            
            # Todo #8: Tempo confidence計算
            # Beat強度の一貫性（標準偏差）
            if len(beat_frames) > 0:
                beat_strength = librosa.util.sync(onset_env, beat_frames, aggregate=np.median)
                if len(beat_strength) > 1 and np.mean(beat_strength) > 1e-6:
                    # 変動係数（CV = std/mean）の逆数
                    cv = np.std(beat_strength) / np.mean(beat_strength)
                    # 0.0-1.0にスケール（CV < 0.5で高信頼度）
                    tempo_confidence = max(0.0, min(1.0, 1.0 - cv))
                else:
                    tempo_confidence = 0.3  # Low confidence
            else:
                tempo_confidence = 0.0
        except Exception as e:
            # Fallback: ビート検出失敗時は固定120 BPMでビート生成
            if self.verbose:
                print(f"  ⚠️ Beat tracking failed, using default 120 BPM")
            tempo = 120.0
            duration = len(audio) / self.sr
            beat_interval = 60.0 / tempo
            beat_times = list(np.arange(0, duration, beat_interval))
            tempo_confidence = 0.0  # No confidence
        
        # Downbeat推定（4/4拍子仮定）
        # 簡易版：4拍ごとにdownbeat
        downbeat_times = [beat_times[i] for i in range(0, len(beat_times), 4)]
        
        # Time signature推定（現時点は4/4固定）
        time_signature = [4, 4]
        
        if self.verbose:
            print(f"  ✓ Global tempo: {tempo:.1f} BPM")
            print(f"  ✓ Beats detected: {len(beat_times)}")
            print(f"  ✓ Downbeats: {len(downbeat_times)}")
            print(f"  ✓ Tempo confidence: {tempo_confidence:.3f}")
        
        return {
            'global_tempo': float(tempo) if not isinstance(tempo, (int, float)) else tempo,
            'beat_times': beat_times,
            'downbeat_times': downbeat_times,
            'time_signature': time_signature,
            'tempo_confidence': float(tempo_confidence),
        }
    
    def extract_sections(
        self,
        tempo_map: Dict[str, Any],
        n_sections: int = 5
    ) -> Tuple[List[Dict[str, Any]], float]:
        """
        構造セクション分割（Intro/Verse/Chorus/Bridge/Outro）
        
        エネルギー・スペクトル変化から自動セグメンテーション
        
        Args:
            tempo_map: extract_tempo_map()の出力
            n_sections: 推定セクション数
        
        Returns:
            (sections, section_confidence)
            sections: List[Dict] - セクション情報
            section_confidence: float (0.0-1.0) - セグメンテーション信頼度
        """
        if self.verbose:
            print("\n📐 Extracting sections...")
        
        audio = self.accomp if self.accomp is not None else self.full_mix
        
        # Spectral segmentation
        hop_length = 512
        chroma = librosa.feature.chroma_cqt(y=audio, sr=self.sr, hop_length=hop_length)
        
        # Recurrence matrix → セグメント境界
        R = librosa.segment.recurrence_matrix(
            chroma,
            mode='affinity',
            metric='cosine',
            width=3
        )
        
        # Laplacian segmentation
        boundaries_frames = librosa.segment.agglomerative(
            chroma,
            n_sections
        )
        
        # Boundaries → 秒単位
        boundaries_times = librosa.frames_to_time(
            boundaries_frames,
            sr=self.sr,
            hop_length=hop_length
        )
        
        # Todo #8: Section confidence計算
        # セクション境界の明瞭度（spectral flux）
        section_confidence = self._calc_section_confidence(
            chroma, boundaries_frames, hop_length
        )
        
        # セクション構築
        sections = []
        beat_times = tempo_map['beat_times']
        tempo = tempo_map['global_tempo']
        beats_per_measure = tempo_map['time_signature'][0]
        
        # セクションラベルヒューリスティック
        section_labels = self._estimate_section_labels(
            boundaries_times,
            len(self.full_mix) / self.sr
        )
        
        for i, (start_time, label) in enumerate(zip(boundaries_times, section_labels)):
            # 終了時刻
            if i < len(boundaries_times) - 1:
                end_time = boundaries_times[i + 1]
            else:
                end_time = len(self.full_mix) / self.sr
            
            # 小節数計算
            duration_beats = (end_time - start_time) * (tempo / 60.0)
            duration_measures = int(np.round(duration_beats / beats_per_measure))
            
            # 開始小節（最も近いdownbeat）
            start_measure = self._time_to_measure(
                start_time,
                beat_times,
                beats_per_measure
            )
            
            sections.append({
                'label': label,
                'start_time': float(start_time),
                'end_time': float(end_time),
                'start_measure': start_measure,
                'duration_measures': max(1, duration_measures),
            })
        
        if self.verbose:
            print(f"  ✓ Sections detected: {len(sections)}")
            print(f"  ✓ Section confidence: {section_confidence:.3f}")
            for sec in sections:
                print(f"    {sec['label']}: {sec['start_time']:.1f}s - {sec['end_time']:.1f}s ({sec['duration_measures']} measures)")
        
        return sections, section_confidence
    
    def extract_chords(
        self,
        sections: List[Dict[str, Any]]
    ) -> Tuple[Dict[str, List[Dict[str, Any]]], float]:
        """
        コード進行抽出（セクションごと）
        
        Chromagram → 主要コード推定
        
        Args:
            sections: extract_sections()の出力
        
        Returns:
            (chords_by_section, chord_confidence)
            chords_by_section: Dict[str, List[Dict]] - セクションごとのコード進行
            chord_confidence: float (0.0-1.0) - コード推定信頼度
        """
        if self.verbose:
            print("\n🎹 Extracting chords...")
        
        audio = self.accomp if self.accomp is not None else self.full_mix
        
        # Chromagram
        hop_length = 512
        chroma = librosa.feature.chroma_cqt(y=audio, sr=self.sr, hop_length=hop_length)
        
        chords_by_section = {}
        all_chord_strengths = []
        
        for section in sections:
            label = section['label']
            start_time = section['start_time']
            end_time = section['end_time']
            
            # セクション範囲のchroma抽出
            start_frame = librosa.time_to_frames(start_time, sr=self.sr, hop_length=hop_length)
            end_frame = librosa.time_to_frames(end_time, sr=self.sr, hop_length=hop_length)
            
            section_chroma = chroma[:, start_frame:end_frame]
            
            # 簡易コード推定（最も強いchroma note → root）
            chord_sequence, chord_strengths = self._chroma_to_chords(
                section_chroma,
                start_time,
                hop_length
            )
            
            chords_by_section[label] = chord_sequence
            all_chord_strengths.extend(chord_strengths)
        
        # Todo #8: Chord confidence計算
        # Chromaピーク強度の平均値（0.0-1.0）
        if len(all_chord_strengths) > 0:
            chord_confidence = float(np.mean(all_chord_strengths))
            chord_confidence = max(0.0, min(1.0, chord_confidence))
        else:
            chord_confidence = 0.0
        
        if self.verbose:
            print(f"  ✓ Chord confidence: {chord_confidence:.3f}")
            for label, chords in chords_by_section.items():
                print(f"    {label}: {len(chords)} chord changes")
        
        return chords_by_section, chord_confidence
    
    def extract_drums_hits(
        self,
        tempo_map: Dict[str, Any]
    ) -> Dict[str, List[float]]:
        """
        ドラムヒット位置抽出（kick/snare/hihat）
        
        Onset detection + 周波数帯域フィルタリング
        
        Args:
            tempo_map: extract_tempo_map()の出力
        
        Returns:
            {
                'kick': [0.5, 1.0, 1.5, ...],  # 秒単位
                'snare': [1.0, 3.0, 5.0, ...],
                'hihat': [0.25, 0.5, 0.75, ...]
            }
        """
        if self.verbose:
            print("\n🥁 Extracting drum hits...")
        
        audio = self.accomp if self.accomp is not None else self.full_mix
        
        # Kick（低域: 20-120 Hz）
        kick_audio = self._bandpass_filter(audio, 20, 120)
        kick_onsets = librosa.onset.onset_detect(
            y=kick_audio,
            sr=self.sr,
            units='time',
            backtrack=True
        )
        
        # Snare（中域: 150-300 Hz）
        snare_audio = self._bandpass_filter(audio, 150, 300)
        snare_onsets = librosa.onset.onset_detect(
            y=snare_audio,
            sr=self.sr,
            units='time',
            backtrack=True
        )
        
        # Hihat（高域: 8000-16000 Hz）
        hihat_audio = self._bandpass_filter(audio, 8000, 16000)
        hihat_onsets = librosa.onset.onset_detect(
            y=hihat_audio,
            sr=self.sr,
            units='time',
            backtrack=True,
            hop_length=256  # より細かい解像度
        )
        
        if self.verbose:
            print(f"  ✓ Kick hits: {len(kick_onsets)}")
            print(f"  ✓ Snare hits: {len(snare_onsets)}")
            print(f"  ✓ Hihat hits: {len(hihat_onsets)}")
        
        return {
            'kick': kick_onsets.tolist(),
            'snare': snare_onsets.tolist(),
            'hihat': hihat_onsets.tolist(),
        }
    
    def extract_bass_contour(
        self,
        tempo_map: Dict[str, Any],
        sections: List[Dict[str, Any]]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        ベースライン輪郭抽出（ピッチ+タイミング）
        
        低域エネルギー + ピッチトラッキング
        
        Args:
            tempo_map: extract_tempo_map()の出力
            sections: extract_sections()の出力
        
        Returns:
            {
                'Verse': [
                    {'time': 0.0, 'pitch': 40, 'duration': 0.5, 'velocity': 80},
                    ...
                ],
                ...
            }
        """
        if self.verbose:
            print("\n🎸 Extracting bass contour...")
        
        audio = self.accomp if self.accomp is not None else self.full_mix
        
        # Bass抽出（低域: 40-250 Hz）
        bass_audio = self._bandpass_filter(audio, 40, 250)
        
        # Pitch tracking (pYIN algorithm)
        f0, voiced_flag, voiced_probs = librosa.pyin(
            bass_audio,
            fmin=librosa.note_to_hz('E1'),  # E1 = 41.2 Hz
            fmax=librosa.note_to_hz('E4'),  # E4 = 329.6 Hz
            sr=self.sr,
            frame_length=2048
        )
        
        # Time grid
        hop_length = 512
        times = librosa.times_like(f0, sr=self.sr, hop_length=hop_length)
        
        bass_by_section = {}
        
        for section in sections:
            label = section['label']
            start_time = section['start_time']
            end_time = section['end_time']
            
            # セクション範囲のpitch抽出
            mask = (times >= start_time) & (times < end_time)
            section_times = times[mask]
            section_f0 = f0[mask]
            section_voiced = voiced_flag[mask]
            
            # Voiced segmentsからnotes抽出
            notes = self._f0_to_notes(
                section_times,
                section_f0,
                section_voiced,
                bass_audio
            )
            
            bass_by_section[label] = notes
        
        if self.verbose:
            for label, notes in bass_by_section.items():
                print(f"  ✓ {label}: {len(notes)} bass notes")
        
        return bass_by_section
    
    def extract_all(self) -> Dict[str, Any]:
        """
        全構造要素を一括抽出
        
        Returns:
            {
                'tempo_map': {...},
                'sections': [...],
                'chords': {...},
                'drums_hits': {...},
                'bass_contour': {...},
                'extraction_confidence': {
                    'tempo_confidence': float,
                    'section_confidence': float,
                    'chord_confidence': float
                },
                'quality_indicators': {
                    'signal_quality': str,
                    'beat_sync_loss': float,
                    'tempo_variance': float,
                    'section_clarity': float
                }
            }
        """
        self.load_audio()
        
        tempo_map = self.extract_tempo_map()
        sections, section_confidence = self.extract_sections(tempo_map)
        chords, chord_confidence = self.extract_chords(sections)
        drums_hits = self.extract_drums_hits(tempo_map)
        bass_contour = self.extract_bass_contour(tempo_map, sections)
        
        # Todo #8: 信頼度スコア集約
        extraction_confidence = {
            'tempo_confidence': tempo_map.get('tempo_confidence', 0.0),
            'section_confidence': section_confidence,
            'chord_confidence': chord_confidence,
        }
        
        # Todo #8: 品質指標計算
        quality_indicators = self._calc_quality_indicators(
            tempo_map, sections, extraction_confidence
        )
        
        return {
            'tempo_map': tempo_map,
            'sections': sections,
            'chords': chords,
            'drums_hits': drums_hits,
            'bass_contour': bass_contour,
            'extraction_confidence': extraction_confidence,
            'quality_indicators': quality_indicators,
        }
    
    def save_yaml(self, structure: Dict[str, Any], output_path: pathlib.Path):
        """YAML保存（numpy型をPython型に変換）"""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Todo #8: numpy型をPython型に変換
        def convert_numpy(obj):
            """Recursively convert numpy types to Python types"""
            if isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            elif isinstance(obj, np.bool_):
                return bool(obj)
            else:
                return obj
        
        structure_clean = convert_numpy(structure)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(structure_clean, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
        
        if self.verbose:
            print(f"\n✅ Structure saved: {output_path}")
    
    # ===== Helper Methods =====
    
    def _calc_section_confidence(
        self,
        chroma: np.ndarray,
        boundaries: np.ndarray,
        hop_length: int
    ) -> float:
        """
        Todo #8: セクション境界の明瞭度計算
        
        境界付近のspectral fluxの大きさで評価
        """
        if len(boundaries) < 2:
            return 0.3  # Low confidence
        
        # Spectral contrast at boundaries
        boundary_contrasts = []
        
        for b_frame in boundaries[1:-1]:  # Skip first/last
            if b_frame < 5 or b_frame >= chroma.shape[1] - 5:
                continue
            
            # Before/After比較（5フレーム窓）
            before = chroma[:, max(0, b_frame-5):b_frame]
            after = chroma[:, b_frame:min(chroma.shape[1], b_frame+5)]
            
            if before.shape[1] > 0 and after.shape[1] > 0:
                # Cosine distance
                mean_before = np.mean(before, axis=1)
                mean_after = np.mean(after, axis=1)
                
                # Normalize
                norm_before = np.linalg.norm(mean_before)
                norm_after = np.linalg.norm(mean_after)
                
                if norm_before > 1e-6 and norm_after > 1e-6:
                    cosine_sim = np.dot(mean_before, mean_after) / (norm_before * norm_after)
                    # Distance（0.0=同じ, 2.0=正反対）
                    distance = 1.0 - cosine_sim
                    boundary_contrasts.append(distance)
        
        if len(boundary_contrasts) > 0:
            # 平均contrast（0.0-1.0にスケール）
            mean_contrast = np.mean(boundary_contrasts)
            # Contrast > 0.3で高信頼度
            confidence = max(0.0, min(1.0, mean_contrast / 0.3))
            return float(confidence)
        else:
            return 0.3
    
    def _calc_quality_indicators(
        self,
        tempo_map: Dict[str, Any],
        sections: List[Dict[str, Any]],
        extraction_confidence: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Todo #8: 品質指標計算
        
        Returns:
            {
                'signal_quality': str,  # 'high' / 'medium' / 'low'
                'beat_sync_loss': float,  # Beat deviation (0.0-1.0)
                'tempo_variance': float,  # Tempo stability (0.0-1.0)
                'section_clarity': float  # Section boundary sharpness (0.0-1.0)
            }
        """
        # 1. Signal quality: RMS/SNR推定
        audio = self.accomp if self.accomp is not None else self.full_mix
        rms = np.sqrt(np.mean(audio ** 2))
        
        # RMSベース評価（簡易版）
        if rms > 0.1:
            signal_quality = 'high'
        elif rms > 0.05:
            signal_quality = 'medium'
        else:
            signal_quality = 'low'
        
        # 2. Beat sync loss: Beat間隔の変動
        beat_times = tempo_map.get('beat_times', [])
        if len(beat_times) > 2:
            intervals = np.diff(beat_times)
            mean_interval = np.mean(intervals)
            if mean_interval > 1e-6:
                # 変動係数
                beat_sync_loss = float(np.std(intervals) / mean_interval)
                beat_sync_loss = max(0.0, min(1.0, beat_sync_loss))
            else:
                beat_sync_loss = 1.0
        else:
            beat_sync_loss = 1.0
        
        # 3. Tempo variance: Tempo変化の度合い
        # 現時点は固定テンポ想定なのでtempo_confidenceの逆数
        tempo_confidence = extraction_confidence.get('tempo_confidence', 0.5)
        tempo_variance = 1.0 - tempo_confidence
        
        # 4. Section clarity: セクション信頼度
        section_clarity = extraction_confidence.get('section_confidence', 0.5)
        
        return {
            'signal_quality': signal_quality,
            'beat_sync_loss': float(beat_sync_loss),
            'tempo_variance': float(tempo_variance),
            'section_clarity': float(section_clarity),
        }
    
    def _bandpass_filter(
        self,
        audio: np.ndarray,
        lowcut: float,
        highcut: float
    ) -> np.ndarray:
        """Bandpass filter"""
        # STFTベースのフィルタリング
        D = librosa.stft(audio)
        freqs = librosa.fft_frequencies(sr=self.sr)
        
        # Frequency mask
        mask = (freqs >= lowcut) & (freqs <= highcut)
        D_filtered = D.copy()
        D_filtered[~mask, :] = 0
        
        # ISTFT
        return librosa.istft(D_filtered)
    
    def _estimate_section_labels(
        self,
        boundaries: np.ndarray,
        total_duration: float
    ) -> List[str]:
        """
        セクションラベル推定ヒューリスティック
        
        境界位置から典型的な曲構造を推定
        """
        n_sections = len(boundaries)
        
        # 典型的なポップス構造パターン
        if n_sections <= 2:
            return ['Intro', 'Verse']
        elif n_sections == 3:
            return ['Intro', 'Verse', 'Chorus']
        elif n_sections == 4:
            return ['Intro', 'Verse', 'Chorus', 'Verse']
        elif n_sections == 5:
            return ['Intro', 'Verse', 'Chorus', 'Bridge', 'Chorus']
        elif n_sections == 6:
            return ['Intro', 'Verse', 'Chorus', 'Verse', 'Chorus', 'Outro']
        else:
            # 7+セクション: パターン繰り返し
            labels = ['Intro']
            remaining = n_sections - 2
            for i in range(remaining):
                labels.append(['Verse', 'Chorus'][i % 2])
            labels.append('Outro')
            return labels
    
    def _time_to_measure(
        self,
        time: float,
        beat_times: List[float],
        beats_per_measure: int
    ) -> int:
        """時刻 → 小節番号変換"""
        # 最も近いbeat探索
        beat_idx = np.searchsorted(beat_times, time)
        measure = beat_idx // beats_per_measure
        return int(measure)
    
    def _chroma_to_chords(
        self,
        chroma: np.ndarray,
        start_time: float,
        hop_length: int,
        chord_change_threshold: float = 0.3
    ) -> Tuple[List[Dict[str, Any]], List[float]]:
        """
        Chromagram → コード列変換
        
        簡易版：最も強いchroma note → root note → major/minor推定
        
        Returns:
            (chord_sequence, chord_strengths)
            chord_sequence: List[Dict] - コードリスト
            chord_strengths: List[float] - 各コードのピーク強度（0.0-1.0）
        """
        # Chord templates (major/minor)
        chord_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        
        chords = []
        chord_strengths = []
        prev_root = -1
        chord_start_time = start_time
        
        for i in range(chroma.shape[1]):
            frame_time = start_time + librosa.frames_to_time(i, sr=self.sr, hop_length=hop_length)
            
            # Root note（最大chroma）
            root_idx = np.argmax(chroma[:, i])
            peak_strength = chroma[root_idx, i]
            
            # コード変化検出
            if root_idx != prev_root and prev_root != -1:
                # 前のコード保存
                duration = frame_time - chord_start_time
                if duration > 0.5:  # 最小持続時間0.5s
                    chords.append({
                        'time': chord_start_time,
                        'chord': chord_names[prev_root],  # 簡易版：major/minor判定省略
                        'duration': duration,
                    })
                    # Todo #8: コード強度保存
                    chord_strengths.append(float(peak_strength))
                chord_start_time = frame_time
            
            prev_root = root_idx
        
        # 最後のコード
        if prev_root != -1:
            peak_strength = chroma[prev_root, -1]
            chords.append({
                'time': chord_start_time,
                'chord': chord_names[prev_root],
                'duration': start_time + librosa.frames_to_time(chroma.shape[1], sr=self.sr, hop_length=hop_length) - chord_start_time,
            })
            chord_strengths.append(float(peak_strength))
        
        return chords, chord_strengths
    
    def _f0_to_notes(
        self,
        times: np.ndarray,
        f0: np.ndarray,
        voiced: np.ndarray,
        audio: np.ndarray
    ) -> List[Dict[str, Any]]:
        """
        F0トラッキング結果 → Note events変換
        """
        notes = []
        
        # Voiced segments抽出
        voiced_segments = self._find_voiced_segments(times, voiced)
        
        for start_idx, end_idx in voiced_segments:
            if end_idx - start_idx < 2:
                continue  # Too short
            
            # Segment内の平均pitch
            segment_f0 = f0[start_idx:end_idx]
            mean_f0 = np.nanmean(segment_f0[~np.isnan(segment_f0)])
            
            if np.isnan(mean_f0) or mean_f0 <= 0:
                continue
            
            # F0 → MIDI pitch
            midi_pitch = librosa.hz_to_midi(mean_f0)
            midi_pitch = int(np.round(midi_pitch))
            
            # Bass range check (E1-E4: MIDI 28-64)
            if midi_pitch < 28 or midi_pitch > 64:
                continue
            
            # Velocity推定（簡易版：固定値）
            velocity = 80
            
            # Duration計算（bounds check）
            if end_idx < len(times):
                duration = float(times[end_idx] - times[start_idx])
            else:
                # 最後のsegment: 推定duration
                duration = float(times[-1] - times[start_idx])
            
            notes.append({
                'time': float(times[start_idx]),
                'pitch': midi_pitch,
                'duration': duration,
                'velocity': velocity,
            })
        
        return notes
    
    def _find_voiced_segments(
        self,
        times: np.ndarray,
        voiced: np.ndarray,
        min_duration: float = 0.1
    ) -> List[Tuple[int, int]]:
        """Voiced segments検出"""
        segments = []
        in_segment = False
        start_idx = 0
        
        for i, is_voiced in enumerate(voiced):
            if is_voiced and not in_segment:
                start_idx = i
                in_segment = True
            elif not is_voiced and in_segment:
                # Segment終了
                duration = times[i] - times[start_idx]
                if duration >= min_duration:
                    segments.append((start_idx, i))
                in_segment = False
        
        # 最後のsegment
        if in_segment:
            duration = times[-1] - times[start_idx]
            if duration >= min_duration:
                segments.append((start_idx, len(times)))
        
        return segments


def main():
    parser = argparse.ArgumentParser(description='Suno Structure Extractor')
    parser.add_argument('--vocal', type=pathlib.Path, help='Vocal stem WAV')
    parser.add_argument('--accomp', type=pathlib.Path, help='Accompaniment stem WAV')
    parser.add_argument('--output', type=pathlib.Path, required=True, help='Output YAML path')
    parser.add_argument('--sr', type=int, default=22050, help='Sample rate')
    parser.add_argument('--n-sections', type=int, default=5, help='Number of sections to detect')
    parser.add_argument('--quiet', action='store_true', help='Suppress verbose output')
    
    args = parser.parse_args()
    
    if not args.vocal and not args.accomp:
        print("[ERROR] At least one of --vocal or --accomp must be provided")
        exit(1)
    
    # Extractor初期化
    extractor = SunoStructureExtractor(
        vocal_path=args.vocal,
        accomp_path=args.accomp,
        sr=args.sr,
        verbose=not args.quiet,
    )
    
    # 構造抽出
    structure = extractor.extract_all()
    
    # YAML保存
    extractor.save_yaml(structure, args.output)
    
    print(f"\n🎉 Extraction complete!")
    print(f"   Tempo: {structure['tempo_map']['global_tempo']:.1f} BPM")
    print(f"   Sections: {len(structure['sections'])}")
    print(f"   Total beats: {len(structure['tempo_map']['beat_times'])}")


if __name__ == '__main__':
    main()
