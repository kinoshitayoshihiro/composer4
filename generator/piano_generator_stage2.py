#!/usr/bin/env python3
"""
Piano Generator Stage2 Integration

既存PianoGeneratorを継承し、Stage2パターン推薦を統合。

Features:
- use_stage2=True時、Pattern Recommenderで高品質パターンを推薦
- Stage2パターンをテンプレートとして使用（Pitch contour + Rhythm + Articulation）
- 既存の humanization/emotion/controls を適用（Velocity/Timing変動維持）
- Fallback: Stage2パターンがない or 推薦失敗 → 既存pattern libraryを使用

Architecture:
    PianoGeneratorStage2
    ├─ MelodyGeneratorStage2: Stage2 melody patterns → emotion適用 → humanize
    └─ CompingGeneratorStage2: Stage2 chords patterns → technique適用 → humanize

Usage:
    from generator.piano_generator_stage2 import PianoGeneratorStage2
    
    gen = PianoGeneratorStage2(use_stage2=True)
    notes = gen.generate(section, technique="pop_comping", emotion=emotion, context=ctx)
"""

from typing import List, Optional
from pathlib import Path
import logging
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from generators.piano import PianoGenerator, MelodyGenerator, CompingGenerator
from generators.base import (
    NoteEvent,
    Section,
    EmotionProfile,
    GenerationContext,
)
from ml.pattern_recommender import PatternRecommender, PatternQuery

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MelodyGeneratorStage2(MelodyGenerator):
    """Stage2統合 Melody Generator"""
    
    def __init__(self, use_stage2: bool = True):
        super().__init__()
        self.use_stage2 = use_stage2
        self.recommender = None
        
        if self.use_stage2:
            patterns_path = Path("data/patterns/stage2_melody.pickle")
            if patterns_path.exists():
                try:
                    self.recommender = PatternRecommender("melody", patterns_path)
                    logger.info(f"✅ Loaded Stage2 melody patterns: {len(self.recommender.patterns)}")
                except Exception as e:
                    logger.warning(f"⚠️  Failed to load Stage2 patterns: {e}. Falling back to library.")
                    self.recommender = None
            else:
                logger.warning(f"⚠️  Stage2 patterns not found: {patterns_path}. Using library.")
    
    def generate(
        self,
        section: Section,
        technique: str,
        emotion: EmotionProfile,
        context: GenerationContext,
    ) -> List[NoteEvent]:
        """
        Melody生成（Stage2優先）
        
        Stage2有効時:
        1. Pattern Recommenderでベストパターンを推薦
        2. パターンのPitch contour + Rhythmを使用
        3. 既存のEmotion/Articulation/Humanizationを適用
        
        Stage2無効 or 推薦失敗時:
        - 親クラスのgenerate()にフォールバック（既存ライブラリ使用）
        """
        if not self.use_stage2 or self.recommender is None:
            # Fallback: 既存ライブラリ
            return super().generate(section, technique, emotion, context)
        
        # Stage2パターン推薦
        try:
            query = PatternQuery(
                tempo=section.tempo,
                duration=section.duration,
                chord_progression=[c.root for c in section.chord_progression] if section.chord_progression else None,
                emotion=emotion.primary.value if emotion.primary else None,
                tempo_tolerance=30.0,  # ±30 BPM
                duration_tolerance=8.0,  # ±8 seconds
            )
            
            results = self.recommender.recommend(query, top_k=3, min_score=0.5)
            
            if not results:
                logger.debug("No Stage2 patterns found. Falling back to library.")
                return super().generate(section, technique, emotion, context)
            
            # Top-1パターンを使用
            best_pattern = results[0]["pattern"]
            source_info = getattr(best_pattern.metadata, 'source_file', 'unknown')
            logger.debug(f"Using Stage2 pattern: {source_info} "
                        f"(score={results[0]['total_score']:.3f})")
            
            # パターンからMelody生成
            melody = self._adapt_pattern_to_section(
                pattern=best_pattern,
                section=section,
                emotion=emotion,
            )
            
            # 既存のEmotion/Articulation適用（親クラスのメソッドを再利用）
            melody = self._apply_articulation(melody, emotion)
            melody = self.apply_emotion(melody, emotion)
            melody = self._quantize_timing(melody, resolution=0.125)
            
            return melody
            
        except Exception as e:
            logger.warning(f"Stage2 pattern generation failed: {e}. Falling back to library.")
            return super().generate(section, technique, emotion, context)
    
    def _adapt_pattern_to_section(
        self,
        pattern,
        section: Section,
        emotion: EmotionProfile,
    ) -> List[NoteEvent]:
        """
        Stage2パターンをセクションに適合
        
        処理:
        1. パターンのTempo調整（元Tempo → セクションTempo）
        2. Durationスケール（パターン長 → セクション長）
        3. Pitch範囲調整（C4-C6範囲に収める）
        4. Chord progressionに合わせてPitch微調整（オプション）
        """
        notes = []
        
        # Tempo/Duration比率
        tempo_ratio = section.tempo / pattern.metadata.tempo
        duration_ratio = section.duration / pattern.metadata.duration
        
        for note_data in pattern.notes:
            # Time/Durationスケール
            # Note: Stage2 NoteEvent uses 'start' attribute, generators.base uses 'time'
            note_start = getattr(note_data, 'start', getattr(note_data, 'time', 0.0))
            scaled_time = note_start * tempo_ratio * duration_ratio
            scaled_duration = note_data.duration * tempo_ratio
            
            # Pitch範囲調整（C4-C6: 60-84）
            pitch = note_data.pitch
            while pitch < self.pitch_range[0]:
                pitch += 12
            while pitch > self.pitch_range[1]:
                pitch -= 12
            
            # Velocity維持（後でemotion適用）
            note = NoteEvent(
                pitch=pitch,
                velocity=note_data.velocity,
                time=scaled_time,
                duration=scaled_duration,
            )
            notes.append(note)
        
        return notes


class CompingGeneratorStage2(CompingGenerator):
    """Stage2統合 Comping Generator"""
    
    def __init__(self, use_stage2: bool = True):
        super().__init__()
        self.use_stage2 = use_stage2
        self.recommender = None
        
        if self.use_stage2:
            patterns_path = Path("data/patterns/stage2_chords.pickle")
            if patterns_path.exists():
                try:
                    self.recommender = PatternRecommender("chords", patterns_path)
                    logger.info(f"✅ Loaded Stage2 chords patterns: {len(self.recommender.patterns)}")
                except Exception as e:
                    logger.warning(f"⚠️  Failed to load Stage2 patterns: {e}. Falling back to library.")
                    self.recommender = None
            else:
                logger.warning(f"⚠️  Stage2 patterns not found: {patterns_path}. Using library.")
    
    def generate(
        self,
        section: Section,
        technique: str,
        emotion: EmotionProfile,
        context: GenerationContext,
    ) -> List[NoteEvent]:
        """
        Comping生成（Stage2優先）
        
        Stage2有効時:
        1. Techniqueに応じたパターンを推薦
        2. Voicing + Rhythmを使用
        3. Emotion適用
        
        Fallback: 親クラス
        """
        if not self.use_stage2 or self.recommender is None:
            return super().generate(section, technique, emotion, context)
        
        try:
            query = PatternQuery(
                tempo=section.tempo,
                technique=self._map_technique_to_stage2(technique),
                duration=section.duration,
                chord_progression=[c.root for c in section.chord_progression] if section.chord_progression else None,
                tempo_tolerance=30.0,
                duration_tolerance=8.0,
            )
            
            results = self.recommender.recommend(query, top_k=3, min_score=0.5)
            
            if not results:
                logger.debug("No Stage2 comping patterns found. Falling back to library.")
                return super().generate(section, technique, emotion, context)
            
            best_pattern = results[0]["pattern"]
            source_info = getattr(best_pattern.metadata, 'source_file', 'unknown')
            logger.debug(f"Using Stage2 comping pattern: {source_info} "
                        f"(score={results[0]['total_score']:.3f})")
            
            # パターンからComping生成
            comping = self._adapt_pattern_to_section(
                pattern=best_pattern,
                section=section,
                emotion=emotion,
            )
            
            # Emotion適用
            comping = self.apply_emotion(comping, emotion)
            comping = self._quantize_timing(comping, resolution=0.125)
            
            return comping
            
        except Exception as e:
            logger.warning(f"Stage2 comping generation failed: {e}. Falling back to library.")
            return super().generate(section, technique, emotion, context)
    
    def _map_technique_to_stage2(self, technique: str) -> Optional[str]:
        """既存technique名 → Stage2 technique名マッピング"""
        mapping = {
            "pop_comping": "block_chords",
            "ballad": "arpeggio",
            "jazz_voicing": "jazz_voicing",
            "arpeggio": "arpeggio",
        }
        return mapping.get(technique)
    
    def _adapt_pattern_to_section(
        self,
        pattern,
        section: Section,
        emotion: EmotionProfile,
    ) -> List[NoteEvent]:
        """Stage2パターンをセクションに適合（Comping用）"""
        notes = []
        
        tempo_ratio = section.tempo / pattern.metadata.tempo
        duration_ratio = section.duration / pattern.metadata.duration
        
        for note_data in pattern.notes:
            # Note: Stage2 NoteEvent uses 'start' attribute, generators.base uses 'time'
            note_start = getattr(note_data, 'start', getattr(note_data, 'time', 0.0))
            scaled_time = note_start * tempo_ratio * duration_ratio
            scaled_duration = note_data.duration * tempo_ratio
            
            # Pitch範囲調整（C2-C5: 36-72）
            pitch = note_data.pitch
            while pitch < self.pitch_range[0]:
                pitch += 12
            while pitch > self.pitch_range[1]:
                pitch -= 12
            
            note = NoteEvent(
                pitch=pitch,
                velocity=note_data.velocity,
                time=scaled_time,
                duration=scaled_duration,
            )
            notes.append(note)
        
        return notes


class PianoGeneratorStage2(PianoGenerator):
    """Stage2統合 Piano Generator"""
    
    def __init__(self, use_stage2: bool = True):
        """
        Initialize Piano Generator with Stage2 support
        
        Args:
            use_stage2: Stage2パターン推薦を使用するか（False=既存ライブラリのみ）
        """
        # Note: Don't call super().__init__() to avoid creating base generators
        # Instead, create Stage2 generators directly
        self.instrument_name = "piano"
        self.melody_gen = MelodyGeneratorStage2(use_stage2=use_stage2)
        self.comping_gen = CompingGeneratorStage2(use_stage2=use_stage2)
        self.pitch_range = (36, 84)  # C2-C6
        self.use_stage2 = use_stage2
        
        logger.info(f"PianoGeneratorStage2 initialized (use_stage2={use_stage2})")
    
    # generate() は親クラス PianoGenerator のものを継承（melody_gen + comping_gen 統合）
    # 必要に応じてオーバーライド可能


# Convenience factory
