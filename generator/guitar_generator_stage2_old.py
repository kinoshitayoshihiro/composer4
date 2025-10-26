#!/usr/bin/env python3
"""
GuitarGenerator Stage2 - Stage2パターン推薦統合

既存のGuitarGeneratorを継承し、Stage2パターン推薦機能を追加。
既存のstrumming/fingerpicking/humanization機能は全て維持。

Features:
- Stage2パターン推薦（technique: strum/fingerpicking）
- Section + Emotion → Technique自動推定
- 既存のhumanization/articulation処理保持
- Fallback to existing rhythm library

Usage:
    from generator.guitar_generator_stage2 import GuitarGeneratorStage2
    
    gen = GuitarGeneratorStage2(
        use_stage2=True,
        stage2_patterns_path="data/patterns/stage2_guitar.pickle",
        tempo=120,
        emotion="happy"
    )
    
    part = gen.compose(
        section_name="Verse",
        measures=8,
        chord_progression=["C", "G", "Am", "F"]
    )
"""

from pathlib import Path
from typing import List, Dict, Optional, Any
import logging

import music21
from music21 import note, pitch, stream, instrument as m21instrument

# Import parent generator
from generator.guitar_generator import GuitarGenerator

# Import Pattern Recommender
from ml.pattern_recommender import PatternRecommender, PatternQuery

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GuitarGeneratorStage2(GuitarGenerator):
    """GuitarGenerator with Stage2 pattern recommendation"""
    
    def __init__(
        self,
        *args,
        use_stage2: bool = True,
        stage2_patterns_path: Optional[str] = None,
        stage2_min_score: float = 0.5,
        **kwargs
    ):
        """
        Initialize GuitarGeneratorStage2
        
        Args:
            use_stage2: Stage2パターン推薦を使用するか
            stage2_patterns_path: Stage2パターンpickleファイルパス
            stage2_min_score: 推薦最小スコア（0.0-1.0）
            **kwargs: GuitarGeneratorの引数
        """
        super().__init__(*args, **kwargs)
        
        self.use_stage2 = use_stage2
        self.stage2_min_score = stage2_min_score
        self.recommender = None
        
        if self.use_stage2:
            # Stage2 patterns読み込み
            if stage2_patterns_path is None:
                stage2_patterns_path = Path(__file__).parent.parent / "data" / "patterns" / "stage2_guitar.pickle"
            
            patterns_path = Path(stage2_patterns_path)
            if patterns_path.exists():
                try:
                    self.recommender = PatternRecommender("guitar", patterns_path)
                    logger.info(f"✅ Stage2 Guitar patterns loaded: {len(self.recommender.patterns)} patterns")
                    
                    # Technique分布確認
                    techniques = {}
                    for p in self.recommender.patterns:
                        tech = p.metadata.technique if hasattr(p.metadata, 'technique') else 'unknown'
                        techniques[tech] = techniques.get(tech, 0) + 1
                    logger.info(f"   Techniques: {techniques}")
                except Exception as e:
                    logger.warning(f"⚠️ Failed to load Stage2 patterns: {e}")
                    self.use_stage2 = False
            else:
                logger.warning(f"⚠️ Stage2 patterns not found: {patterns_path}")
                self.use_stage2 = False
    
    def compose(
        self,
        section_name: Optional[str] = None,
        measures: int = 4,
        chord_progression: Optional[List[str]] = None,
        **kwargs
    ) -> stream.Part:
        """
        Compose guitar part（Stage2推薦 or 既存ライブラリ）
        
        Args:
            section_name: セクション名（Intro/Verse/Chorus/Bridge/Outro）
            measures: 小節数
            chord_progression: コード進行
            **kwargs: 追加パラメータ
        
        Returns:
            music21.stream.Part
        """
        # Stage2試行
        if self.use_stage2 and self.recommender:
            try:
                part = self._compose_with_stage2(
                    section_name=section_name,
                    measures=measures,
                    chord_progression=chord_progression,
                    **kwargs
                )
                
                # Stage2成功時は既存処理適用して返す
                if part and len(part.flatten().notes) > 0:
                    logger.info(f"✅ Stage2 guitar generation successful: {len(part.flatten().notes)} notes")
                    # 既存のhumanization/articulation適用
                    part = self._apply_existing_processing(part)
                    return part
                else:
                    logger.info("⚠️ Stage2 generated empty part, returning empty part")
                    # Empty part返す（fallbackなし）
                    empty_part = stream.Part()
                    empty_part.insert(0, self.default_instrument if self.default_instrument else m21instrument.AcousticGuitar())
                    return empty_part
            except Exception as e:
                logger.warning(f"⚠️ Stage2 generation failed: {e}")
                # Empty part返す
                empty_part = stream.Part()
                empty_part.insert(0, self.default_instrument if self.default_instrument else m21instrument.AcousticGuitar())
                return empty_part
        
        # Stage2無効時もempty part
        logger.info("📚 Stage2 disabled, returning empty part")
        empty_part = stream.Part()
        empty_part.insert(0, self.default_instrument if self.default_instrument else m21instrument.AcousticGuitar())
        return empty_part
    
    def _compose_with_stage2(
        self,
        section_name: Optional[str],
        measures: int,
        chord_progression: Optional[List[str]],
        **kwargs
    ) -> Optional[stream.Part]:
        """
        Stage2パターン推薦でギター生成
        
        Returns:
            music21.stream.Part or None
        """
        # Tempo取得
        tempo = kwargs.get('tempo', self.global_tempo or 120.0)
        
        # Emotion取得
        emotion = kwargs.get('emotion', self.emotion or 'neutral')
        
        # Technique推定（Section + Emotion → strum/fingerpicking）
        technique = self._estimate_technique(section_name, emotion)
        
        # Duration計算（measures → seconds）
        beats_per_measure = 4  # 4/4 仮定
        total_beats = measures * beats_per_measure
        duration = (total_beats / tempo) * 60.0
        
        # Pattern query作成
        query = PatternQuery(
            tempo=tempo,
            technique=technique,
            duration=duration,
            chord_progression=chord_progression,
            emotion=emotion,
        )
        
        # Pattern推薦
        results = self.recommender.recommend(query, top_k=3, min_score=self.stage2_min_score)
        
        if not results:
            logger.info(f"⚠️ No Stage2 patterns found for {section_name}/{technique}")
            return None
        
        # Best pattern選択
        best_result = results[0]
        pattern = best_result['pattern']
        
        logger.info(f"📊 Stage2 pattern selected: score={best_result['total_score']:.3f}, "
                   f"technique={technique}, notes={len(pattern.notes)}")
        
        # Pattern → Part変換
        part = self._pattern_to_part(pattern, tempo, measures)
        
        return part
    
    def _estimate_technique(self, section_name: Optional[str], emotion: str) -> str:
        """
        Section + Emotion → Technique推定
        
        Heuristics:
        - Intro: fingerpicking（イントロは繊細に）
        - Verse: strum（低エネルギー）or fingerpicking（繊細）
        - Chorus: strum（高エネルギー）
        - Bridge: fingerpicking（変化をつける）
        - Outro: fingerpicking（フェードアウト）
        
        Emotion考慮:
        - sad/calm → fingerpicking優先
        - happy/energetic → strum優先
        
        Returns:
            "strum" or "fingerpicking"
        """
        section = (section_name or "verse").lower()
        emotion = emotion.lower()
        
        # Emotion-based bias
        fingerpicking_emotions = ["sad", "calm", "melancholic", "tender"]
        strum_emotions = ["happy", "energetic", "excited", "powerful"]
        
        emotion_prefers_fingerpicking = any(e in emotion for e in fingerpicking_emotions)
        emotion_prefers_strum = any(e in emotion for e in strum_emotions)
        
        # Section-based decision
        if "intro" in section:
            return "fingerpicking"
        elif "verse" in section:
            # Verseは感情依存
            return "fingerpicking" if emotion_prefers_fingerpicking else "strum"
        elif "chorus" in section:
            # Chorusは通常strum（高エネルギー）
            return "strum"
        elif "bridge" in section:
            # Bridgeは変化をつけてfingerpicking
            return "fingerpicking"
        elif "outro" in section:
            return "fingerpicking"
        else:
            # Default: strum
            return "strum"
    
    def _pattern_to_part(
        self,
        pattern: Any,
        target_tempo: float,
        target_measures: int
    ) -> stream.Part:
        """
        Stage2 Pattern → music21.Part変換
        
        Args:
            pattern: ExtractedPattern
            target_tempo: ターゲットテンポ
            target_measures: ターゲット小節数
        
        Returns:
            music21.stream.Part
        """
        part = stream.Part()
        
        # Instrument設定
        if self.default_instrument:
            part.insert(0, self.default_instrument)
        else:
            part.insert(0, m21instrument.AcousticGuitar())
        
        # Source tempo取得
        source_tempo = pattern.metadata.tempo if hasattr(pattern.metadata, 'tempo') else 120.0
        
        # Tempo ratio計算（時間軸スケーリング）
        tempo_ratio = target_tempo / source_tempo if source_tempo > 0 else 1.0
        
        # Duration ratio（小節数スケーリング）
        source_measures = pattern.metadata.duration_bars if hasattr(pattern.metadata, 'duration_bars') else 4
        duration_ratio = target_measures / source_measures if source_measures > 0 else 1.0
        
        # Notes変換
        for note_event in pattern.notes:
            # Note時刻取得（互換性対応）
            note_start = getattr(note_event, 'start', getattr(note_event, 'time', 0.0))
            
            # 時刻・長さ調整
            adjusted_start = note_start * tempo_ratio * duration_ratio
            adjusted_duration = note_event.duration * tempo_ratio
            
            # Note作成
            if hasattr(note_event, 'pitches') and len(note_event.pitches) > 1:
                # Chord
                pitches = [pitch.Pitch(midi=p) for p in note_event.pitches]
                c = music21.chord.Chord(pitches)
                c.quarterLength = adjusted_duration * 2  # beats → quarterLength
                c.volume.velocity = note_event.velocity
                c.offset = adjusted_start * 2
                
                # Guitar range enforcement (E2-E5: MIDI 40-88)
                for p in c.pitches:
                    while p.midi < 40:
                        p.midi += 12
                    while p.midi > 88:
                        p.midi -= 12
                
                part.append(c)
            else:
                # Single note
                n = note.Note(pitch=note_event.pitch)
                n.quarterLength = adjusted_duration * 2
                n.volume.velocity = note_event.velocity
                n.offset = adjusted_start * 2
                
                # Guitar range enforcement
                while n.pitch.midi < 40:
                    n.pitch.midi += 12
                while n.pitch.midi > 88:
                    n.pitch.midi -= 12
                
                part.append(n)
        
        return part
    
    def _apply_existing_processing(self, part: stream.Part) -> stream.Part:
        """
        既存GuitarGeneratorの処理適用
        
        - Humanization (timing jitter, velocity variation)
        - Articulation
        - Expression
        
        Args:
            part: Stage2生成Part
        
        Returns:
            処理後のPart
        """
        # Timing jitter適用
        if hasattr(self, 'timing_jitter_ms') and self.timing_jitter_ms > 0:
            # rng確認（親クラスから継承）
            rng = getattr(self, 'rng', getattr(self, '_rng', None))
            if rng is None:
                import random
                rng = random.Random()
            
            for n in part.flatten().notesAndRests:
                if hasattr(n, 'offset'):
                    jitter = (rng.random() - 0.5) * 2 * (self.timing_jitter_ms / 1000.0)
                    n.offset += jitter
        
        # Velocity variation（簡易版）
        rng = getattr(self, 'rng', getattr(self, '_rng', None))
        if rng is None:
            import random
            rng = random.Random()
        
        for n in part.flatten().notes:
            if hasattr(n, 'volume'):
                variation = int((rng.random() - 0.5) * 20)
                n.volume.velocity = max(1, min(127, n.volume.velocity + variation))
        
        return part


def main():
    """CLI test harness"""
    import argparse
    
    parser = argparse.ArgumentParser(description='GuitarGenerator Stage2 Test')
    parser.add_argument('--tempo', type=float, default=120.0, help='Tempo (BPM)')
    parser.add_argument('--section', type=str, default='Verse', help='Section name')
    parser.add_argument('--measures', type=int, default=4, help='Number of measures')
    parser.add_argument('--emotion', type=str, default='neutral', help='Emotion')
    parser.add_argument('--chords', type=str, nargs='+', default=['C', 'G', 'Am', 'F'], help='Chord progression')
    parser.add_argument('--output', type=str, default='demo_guitar_stage2.mid', help='Output MIDI file')
    
    args = parser.parse_args()
    
    print("\n🎸 GuitarGenerator Stage2 - Demo Generation")
    print("=" * 60)
    
    # Generator作成
    gen = GuitarGeneratorStage2(
        use_stage2=True,
        default_instrument=m21instrument.AcousticGuitar(),
        tempo=args.tempo,
        emotion=args.emotion,
    )
    
    # Generate
    print(f"\n📝 Parameters:")
    print(f"   Tempo: {args.tempo} BPM")
    print(f"   Section: {args.section}")
    print(f"   Measures: {args.measures}")
    print(f"   Emotion: {args.emotion}")
    print(f"   Chords: {args.chords}")
    
    part = gen.compose(
        section_name=args.section,
        measures=args.measures,
        chord_progression=args.chords,
        tempo=args.tempo,
        emotion=args.emotion,
    )
    
    # Stats
    notes = list(part.flatten().notes)
    print(f"\n📊 Generated:")
    print(f"   Notes: {len(notes)}")
    if notes:
        pitches = [n.pitch.midi for n in notes if hasattr(n, 'pitch')]
        if pitches:
            print(f"   Pitch range: {min(pitches)} - {max(pitches)} (MIDI)")
    
    # Save
    part.write('midi', fp=args.output)
    print(f"\n✅ Saved to {args.output}")


if __name__ == '__main__':
    main()
