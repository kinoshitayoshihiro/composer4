#!/usr/bin/env python3
"""
Drums Generator Stage2 - リズム生成（kick/snare/hihat/crash/ride対応）

SLAKH/LAMDA等のドラムMIDIから高品質なリズムパターンを抽出し、
新しい曲に適用できる形で保存・推薦します。

GMドラムマップ:
    Bass Drum (Kick):  35, 36
    Snare:             38, 40
    Hi-Hat Closed:     42
    Hi-Hat Open:       46
    Crash Cymbal:      49, 57
    Ride Cymbal:       51, 59

使用方法:
    from generator.drums_generator_stage2 import DrumsGeneratorStage2
    
    gen = DrumsGeneratorStage2(
        patterns_pickle="data/patterns/stage2_drums.pickle",
        default_instrument=music21.instrument.Percussion()
    )
    
    # パターン推薦
    pattern = gen.recommend_pattern(
        tempo=120,
        emotion="energetic",
        section="Chorus",
        technique="rock_basic"
    )
    
    # ドラムパート生成
    drum_part = gen.generate(
        bars=8,
        chords=["C", "G", "Am", "F"],
        tempo=120,
        emotion="energetic",
        technique="rock_basic"
    )
"""

import music21
from music21 import note, stream, instrument, duration
import numpy as np
import pickle
from pathlib import Path
from typing import Optional, List, Dict, Any
from dataclasses import dataclass


# GMドラムマップ
GM_DRUM_MAP = {
    'kick': [35, 36],           # Bass Drum 1, Bass Drum 2
    'snare': [38, 40],          # Acoustic Snare, Electric Snare
    'hihat_closed': [42],       # Closed Hi-Hat
    'hihat_open': [46],         # Open Hi-Hat
    'crash': [49, 57],          # Crash Cymbal 1, Crash Cymbal 2
    'ride': [51, 59],           # Ride Cymbal 1, Ride Cymbal 2
    'tom_low': [41, 43, 45],    # Low Floor Tom, High Floor Tom, Low Tom
    'tom_mid': [47, 48],        # Low-Mid Tom, Hi-Mid Tom
    'tom_high': [50],           # High Tom
}


@dataclass
class DrumPattern:
    """ドラムパターンデータ"""
    id: str
    instrument: str  # 'drums'
    technique: str   # 'rock_basic', 'jazz_swing', 'edm_four_on_floor' etc.
    tempo: float
    bars: int
    emotion: str
    
    # リズムデータ（小節内の相対位置 0.0-4.0）
    kick_hits: List[float]        # キック打点
    snare_hits: List[float]       # スネア打点
    hihat_hits: List[float]       # ハイハット打点
    crash_hits: List[float]       # クラッシュ打点
    ride_hits: List[float]        # ライド打点
    
    # ベロシティ情報
    kick_velocities: List[int]
    snare_velocities: List[int]
    hihat_velocities: List[int]
    crash_velocities: List[int]
    ride_velocities: List[int]
    
    # メタ情報
    density: float               # 密度（hits per bar）
    complexity: float            # 複雑度（0.0-1.0）
    syncopation_rate: float      # シンコペーション率
    
    # 品質スコア
    quality_score: float = 0.0
    
    # MIDI Pitch情報（Todo #7: ハイハット開閉整合性チェック用）
    hihat_pitches: List[int] = None  # 各ハイハットヒットのMIDI pitch（42=Closed, 46=Open, 44=Pedal）
    
    # Duration情報（Todo #7: クラッシュチョーク長制限チェック用）
    crash_durations: List[float] = None  # 各クラッシュノートの長さ（quarter beats）


class DrumsGeneratorStage2:
    """ドラムトラック生成（Stage2: パターンベース）"""
    
    def __init__(
        self,
        patterns_pickle: Optional[Path] = None,
        default_instrument: Optional[music21.instrument.Instrument] = None
    ):
        """
        Args:
            patterns_pickle: Stage2パターンPickleファイル
            default_instrument: デフォルト楽器（Percussion）
        """
        self.patterns: List[DrumPattern] = []
        self.default_instrument = default_instrument or instrument.Percussion()
        
        if patterns_pickle and Path(patterns_pickle).exists():
            self.load_patterns(patterns_pickle)
    
    def load_patterns(self, pickle_path: Path) -> None:
        """パターンをロード"""
        with open(pickle_path, 'rb') as f:
            self.patterns = pickle.load(f)
        print(f"✅ Loaded {len(self.patterns)} drum patterns from {pickle_path}")
    
    def recommend_pattern(
        self,
        tempo: float,
        emotion: str,
        section: str,
        technique: Optional[str] = None,
        top_k: int = 5
    ) -> Optional[DrumPattern]:
        """
        最適なドラムパターンを推薦
        
        Args:
            tempo: テンポ (BPM)
            emotion: 感情タグ
            section: セクション名
            technique: 奏法（rock_basic, jazz_swing, edm_four_on_floor等）
            top_k: 候補数
        
        Returns:
            推薦されたパターン
        """
        if not self.patterns:
            return None
        
        scores = []
        for pattern in self.patterns:
            score = 0.0
            
            # テンポ適合度（±10 BPM以内で最高スコア）
            tempo_diff = abs(pattern.tempo - tempo)
            tempo_score = max(0.0, 1.0 - tempo_diff / 30.0)
            score += tempo_score * 0.4
            
            # 感情適合度
            if pattern.emotion == emotion:
                score += 0.3
            
            # 奏法適合度
            if technique and pattern.technique == technique:
                score += 0.2
            
            # 品質スコア
            score += pattern.quality_score * 0.1
            
            scores.append(score)
        
        # Top-K選択
        top_indices = np.argsort(scores)[-top_k:][::-1]
        best_pattern = self.patterns[top_indices[0]]
        
        return best_pattern
    
    def generate(
        self,
        bars: int,
        chords: List[str],
        tempo: float,
        emotion: str,
        section: str = "Verse",
        technique: Optional[str] = None,
        time_signature: str = "4/4",
        seed: Optional[int] = None
    ) -> music21.stream.Part:
        """
        ドラムトラックを生成
        
        Args:
            bars: 小節数
            chords: コード進行（参考用、ドラムには直接影響なし）
            tempo: テンポ
            emotion: 感情タグ
            section: セクション名
            technique: 奏法
            time_signature: 拍子記号
            seed: 乱数シード
        
        Returns:
            生成されたドラムパート
        """
        if seed is not None:
            np.random.seed(seed)
        
        # パターン推薦
        pattern = self.recommend_pattern(
            tempo=tempo,
            emotion=emotion,
            section=section,
            technique=technique
        )
        
        if pattern is None:
            # フォールバック: 基本的な4つ打ち
            return self._generate_fallback_drums(bars, tempo, time_signature)
        
        # パート作成
        drum_part = stream.Part()
        drum_part.insert(0, self.default_instrument)
        
        # パターンを繰り返し配置
        current_offset = 0.0
        for bar_idx in range(bars):
            self._add_bar_from_pattern(
                drum_part,
                pattern,
                current_offset,
                tempo
            )
            current_offset += 4.0  # 4/4拍子前提
        
        return drum_part
    
    def _add_bar_from_pattern(
        self,
        drum_part: music21.stream.Part,
        pattern: DrumPattern,
        start_offset: float,
        tempo: float
    ) -> None:
        """パターンから1小節分のドラムノートを追加"""
        
        # Kick
        for i, pos in enumerate(pattern.kick_hits):
            vel = pattern.kick_velocities[i] if i < len(pattern.kick_velocities) else 90
            kick_note = note.Note(GM_DRUM_MAP['kick'][0])
            kick_note.volume.velocity = vel
            kick_note.duration = duration.Duration(0.25)  # 16分音符長
            drum_part.insert(start_offset + pos, kick_note)
        
        # Snare
        for i, pos in enumerate(pattern.snare_hits):
            vel = pattern.snare_velocities[i] if i < len(pattern.snare_velocities) else 95
            snare_note = note.Note(GM_DRUM_MAP['snare'][0])
            snare_note.volume.velocity = vel
            snare_note.duration = duration.Duration(0.25)
            drum_part.insert(start_offset + pos, snare_note)
        
        # Hi-Hat
        for i, pos in enumerate(pattern.hihat_hits):
            vel = pattern.hihat_velocities[i] if i < len(pattern.hihat_velocities) else 70
            hihat_note = note.Note(GM_DRUM_MAP['hihat_closed'][0])
            hihat_note.volume.velocity = vel
            hihat_note.duration = duration.Duration(0.25)
            drum_part.insert(start_offset + pos, hihat_note)
        
        # Crash (小節の最初のみ)
        for i, pos in enumerate(pattern.crash_hits):
            vel = pattern.crash_velocities[i] if i < len(pattern.crash_velocities) else 100
            crash_note = note.Note(GM_DRUM_MAP['crash'][0])
            crash_note.volume.velocity = vel
            crash_note.duration = duration.Duration(1.0)  # 全音符長
            drum_part.insert(start_offset + pos, crash_note)
        
        # Ride
        for i, pos in enumerate(pattern.ride_hits):
            vel = pattern.ride_velocities[i] if i < len(pattern.ride_velocities) else 75
            ride_note = note.Note(GM_DRUM_MAP['ride'][0])
            ride_note.volume.velocity = vel
            ride_note.duration = duration.Duration(0.5)  # 8分音符長
            drum_part.insert(start_offset + pos, ride_note)
    
    def _generate_fallback_drums(
        self,
        bars: int,
        tempo: float,
        time_signature: str
    ) -> music21.stream.Part:
        """
        フォールバック: 基本的な4つ打ちドラムパターン
        
        パターン:
            Kick:  1拍目, 3拍目
            Snare: 2拍目, 4拍目
            HH:    全8分音符
        """
        drum_part = stream.Part()
        drum_part.insert(0, self.default_instrument)
        
        current_offset = 0.0
        for bar_idx in range(bars):
            # Kick (1拍目, 3拍目)
            for beat in [0.0, 2.0]:
                kick = note.Note(GM_DRUM_MAP['kick'][0])
                kick.volume.velocity = 90
                kick.duration = duration.Duration(0.25)
                drum_part.insert(current_offset + beat, kick)
            
            # Snare (2拍目, 4拍目)
            for beat in [1.0, 3.0]:
                snare = note.Note(GM_DRUM_MAP['snare'][0])
                snare.volume.velocity = 95
                snare.duration = duration.Duration(0.25)
                drum_part.insert(current_offset + beat, snare)
            
            # Hi-Hat (全8分音符)
            for eighth in np.arange(0.0, 4.0, 0.5):
                hihat = note.Note(GM_DRUM_MAP['hihat_closed'][0])
                hihat.volume.velocity = 70
                hihat.duration = duration.Duration(0.25)
                drum_part.insert(current_offset + eighth, hihat)
            
            current_offset += 4.0
        
        return drum_part


# デモ実行
if __name__ == '__main__':
    print("=" * 60)
    print("  Drums Generator Stage2 Demo")
    print("=" * 60)
    
    # ジェネレーター作成
    gen = DrumsGeneratorStage2()
    
    # フォールバックドラム生成
    print("\n🥁 Generating fallback drums (4 bars, 120 BPM)...")
    drum_part = gen.generate(
        bars=4,
        chords=["C", "G", "Am", "F"],
        tempo=120,
        emotion="energetic",
        section="Verse",
        seed=42
    )
    
    # 統計情報
    all_notes = list(drum_part.flatten().notes)
    print(f"✅ Generated {len(all_notes)} drum notes")
    
    # MIDI出力
    output_path = Path("out/demo_drums.mid")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    drum_part.write('midi', fp=output_path)
    print(f"💾 Saved to: {output_path}")
    
    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)
