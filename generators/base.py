"""
Instrument Generator Base Classes

楽器別Generator基底クラスと共通データ構造
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np


class Emotion(Enum):
    """感情タイプ"""
    JOY = "joy"
    SORROW = "sorrow"
    TENSION = "tension"
    PEACE = "peace"
    EXCITEMENT = "excitement"
    MELANCHOLY = "melancholy"


@dataclass
class EmotionProfile:
    """感情プロファイル"""
    primary: Emotion
    intensity: float  # 0.0-1.0
    secondary: Optional[Emotion] = None
    secondary_weight: float = 0.0
    
    def to_velocity_mod(self) -> int:
        """感情 → Velocity修正値"""
        base_mod = {
            Emotion.JOY: 10,
            Emotion.SORROW: -10,
            Emotion.TENSION: 15,
            Emotion.PEACE: -5,
            Emotion.EXCITEMENT: 20,
            Emotion.MELANCHOLY: -8,
        }
        return int(base_mod[self.primary] * self.intensity)
    
    def to_tempo_mod(self) -> float:
        """感情 → Tempo修正率（0.95 = -5%）"""
        base_mod = {
            Emotion.JOY: 1.05,
            Emotion.SORROW: 0.95,
            Emotion.TENSION: 1.0,
            Emotion.PEACE: 0.92,
            Emotion.EXCITEMENT: 1.10,
            Emotion.MELANCHOLY: 0.93,
        }
        return base_mod[self.primary]


@dataclass
class Chord:
    """コード情報"""
    root: int  # MIDI note (C=0, C#=1, ...)
    quality: str  # major/minor/dim/aug/sus/7th等
    bass: Optional[int] = None  # Bass note（オンコード用）
    
    def get_tones(self, extensions: bool = False) -> List[int]:
        """
        コード構成音を取得
        
        Args:
            extensions: テンション音を含むか
        
        Returns:
            MIDI note番号のリスト
        """
        tones = [self.root]
        
        if self.quality == "major":
            tones.extend([self.root + 4, self.root + 7])
        elif self.quality == "minor":
            tones.extend([self.root + 3, self.root + 7])
        elif self.quality == "dim":
            tones.extend([self.root + 3, self.root + 6])
        elif self.quality == "aug":
            tones.extend([self.root + 4, self.root + 8])
        elif self.quality == "sus4":
            tones.extend([self.root + 5, self.root + 7])
        elif self.quality == "7th":
            tones.extend([self.root + 4, self.root + 7, self.root + 10])
        elif self.quality == "maj7":
            tones.extend([self.root + 4, self.root + 7, self.root + 11])
        elif self.quality == "min7":
            tones.extend([self.root + 3, self.root + 7, self.root + 10])
        
        # Extensions (9th, 11th, 13th)
        if extensions:
            if self.quality in ["7th", "maj7", "min7"]:
                tones.append(self.root + 14)  # 9th
        
        return tones
    
    @property
    def third(self) -> int:
        """3度音"""
        return self.root + (3 if "min" in self.quality else 4)
    
    @property
    def fifth(self) -> int:
        """5度音"""
        return self.root + 7
    
    @property
    def seventh(self) -> Optional[int]:
        """7度音（7thコードのみ）"""
        if "7" in self.quality:
            return self.root + (11 if "maj7" in self.quality else 10)
        return None


@dataclass
class Section:
    """セクション情報"""
    tempo: float  # BPM
    key: str  # C, Dm, F#, etc
    time_signature: Tuple[int, int]  # (4, 4) = 4/4
    chord_progression: List[Chord]
    duration: float  # セクション長（拍数）
    style: str  # pop/jazz/rock/ballad等


@dataclass
class NoteEvent:
    """音符イベント"""
    pitch: int  # MIDI note number (0-127)
    velocity: int  # 0-127
    time: float  # 開始時刻（拍単位）
    duration: float  # 長さ（拍単位）
    channel: int = 0
    
    @property
    def end_time(self) -> float:
        """終了時刻"""
        return self.time + self.duration


@dataclass
class GenerationContext:
    """生成コンテキスト（他楽器との協調用）"""
    previous_section: Optional[Section] = None
    drums_pattern: Optional[List[NoteEvent]] = None
    bass_notes: Optional[List[NoteEvent]] = None
    guitar_notes: Optional[List[NoteEvent]] = None
    strings_notes: Optional[List[NoteEvent]] = None
    piano_notes: Optional[List[NoteEvent]] = None


@dataclass
class ValidationResult:
    """検証結果"""
    passed: bool
    score: float  # 0.0-1.0
    metrics: Dict[str, float]
    issues: List[str]


class InstrumentGeneratorBase:
    """楽器Generator基底クラス"""
    
    def __init__(self, instrument_name: str):
        self.instrument_name = instrument_name
        self.pitch_range = (0, 127)  # デフォルト音域
    
    def generate(
        self,
        section: Section,
        technique: str,
        emotion: EmotionProfile,
        context: GenerationContext,
    ) -> List[NoteEvent]:
        """
        楽器別MIDI生成
        
        Args:
            section: セクション情報
            technique: 奏法
            emotion: 感情プロファイル
            context: 生成コンテキスト
        
        Returns:
            NoteEvent のリスト
        """
        raise NotImplementedError("Subclass must implement generate()")
    
    def apply_technique(
        self,
        notes: List[NoteEvent],
        technique: str,
    ) -> List[NoteEvent]:
        """
        奏法を適用（velocity/timing/articulation調整）
        
        Args:
            notes: 生成されたノート
            technique: 奏法名
        
        Returns:
            調整後のノート
        """
        return notes
    
    def apply_emotion(
        self,
        notes: List[NoteEvent],
        emotion: EmotionProfile,
    ) -> List[NoteEvent]:
        """
        感情表現を適用
        
        Args:
            notes: 生成されたノート
            emotion: 感情プロファイル
        
        Returns:
            調整後のノート
        """
        velocity_mod = emotion.to_velocity_mod()
        
        for note in notes:
            note.velocity = max(1, min(127, note.velocity + velocity_mod))
        
        return notes
    
    def validate(self, notes: List[NoteEvent]) -> ValidationResult:
        """
        生成結果の品質検証
        
        Args:
            notes: 生成されたノート
        
        Returns:
            検証結果
        """
        issues = []
        metrics = {}
        
        # 基本チェック
        if not notes:
            return ValidationResult(
                passed=False,
                score=0.0,
                metrics={},
                issues=["No notes generated"],
            )
        
        # 音域チェック
        pitches = [n.pitch for n in notes]
        if min(pitches) < self.pitch_range[0]:
            issues.append(f"Pitch below range: {min(pitches)}")
        if max(pitches) > self.pitch_range[1]:
            issues.append(f"Pitch above range: {max(pitches)}")
        
        # Velocity範囲チェック
        velocities = [n.velocity for n in notes]
        if min(velocities) < 1:
            issues.append(f"Invalid velocity: {min(velocities)}")
        if max(velocities) > 127:
            issues.append(f"Invalid velocity: {max(velocities)}")
        
        # Duration チェック
        if any(n.duration <= 0 for n in notes):
            issues.append("Invalid duration (<=0)")
        
        metrics["pitch_range"] = max(pitches) - min(pitches)
        metrics["velocity_range"] = max(velocities) - min(velocities)
        metrics["num_notes"] = len(notes)
        metrics["duration"] = max(n.end_time for n in notes) if notes else 0.0
        
        score = 1.0 if not issues else 0.7
        
        return ValidationResult(
            passed=len(issues) == 0,
            score=score,
            metrics=metrics,
            issues=issues,
        )
    
    def _shift_to_octave(
        self,
        notes: List[NoteEvent],
        target_octave: int,
    ) -> List[NoteEvent]:
        """
        音符を指定オクターブに移動
        
        Args:
            notes: ノート
            target_octave: 目標オクターブ (C4 = 4)
        
        Returns:
            調整後のノート
        """
        for note in notes:
            current_octave = note.pitch // 12
            octave_diff = target_octave - current_octave
            note.pitch += octave_diff * 12
            
            # 音域制限
            note.pitch = max(self.pitch_range[0], 
                            min(self.pitch_range[1], note.pitch))
        
        return notes
    
    def _quantize_timing(
        self,
        notes: List[NoteEvent],
        resolution: float = 0.25,  # 16th note
    ) -> List[NoteEvent]:
        """
        タイミングをクオンタイズ
        
        Args:
            notes: ノート
            resolution: 分解能（拍単位、0.25 = 16分音符）
        
        Returns:
            クオンタイズ後のノート
        """
        for note in notes:
            note.time = round(note.time / resolution) * resolution
            note.duration = round(note.duration / resolution) * resolution
        
        return notes
