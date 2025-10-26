#!/usr/bin/env python3
"""
Bass Generator Stage2 Integration

既存BassGeneratorを拡張し、Stage2パターン推薦を統合。

Features:
- use_stage2=True時、Pattern Recommenderで高品質bassパターンを推薦
- Technique選択：walking/pick/slap/fingerstyle（Stage2パターンから学習）
- Kick同期維持：既存のsync_with_kick()機能を保持
- 既存のhumanization/emotion/controls適用
- Fallback: Stage2パターンがない → 既存pattern libraryを使用

Architecture:
    BassGeneratorStage2
    ├─ Pattern Recommender (Stage2 bass patterns)
    ├─ Technique-aware pattern selection
    ├─ Kick sync (既存実装)
    └─ Humanization + Controls (既存実装)

Usage:
    from generator.bass_generator_stage2 import BassGeneratorStage2
    
    gen = BassGeneratorStage2(use_stage2=True, global_tempo=120.0)
    
    # With kick sync
    shared_tracks = {"kick_offsets": [0.0, 1.0, 2.0, 3.0]}
    part = gen.compose(section_data=section, shared_tracks=shared_tracks)
"""

from typing import List, Optional, Any, Dict
from pathlib import Path
import logging
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from generator.bass_generator import BassGenerator
from ml.pattern_recommender import PatternRecommender, PatternQuery
from music21 import stream, note, pitch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BassGeneratorStage2(BassGenerator):
    """Stage2統合 Bass Generator"""
    
    def __init__(
        self,
        *args,
        use_stage2: bool = True,
        stage2_min_score: float = 0.5,
        **kwargs,
    ):
        """
        Initialize Bass Generator with Stage2 support
        
        Args:
            use_stage2: Stage2パターン推薦を使用するか（False=既存ライブラリのみ）
            stage2_min_score: 最小推薦スコア（0.0-1.0）
            **kwargs: BassGeneratorと同じ引数
        """
        super().__init__(*args, **kwargs)
        
        self.use_stage2 = use_stage2
        self.stage2_min_score = stage2_min_score
        self.recommender = None
        
        if self.use_stage2:
            patterns_path = Path("data/patterns/stage2_bass.pickle")
            if patterns_path.exists():
                try:
                    self.recommender = PatternRecommender("bass", patterns_path)
                    logger.info(f"✅ Loaded Stage2 bass patterns: {len(self.recommender.patterns)}")
                    logger.info(f"   Techniques: {', '.join(sorted(self.recommender.techniques))}")
                except Exception as e:
                    logger.warning(f"⚠️  Failed to load Stage2 patterns: {e}. Falling back to library.")
                    self.recommender = None
                    self.use_stage2 = False
            else:
                logger.warning(f"⚠️  Stage2 patterns not found: {patterns_path}. Using library.")
                self.use_stage2 = False
        
        logger.info(f"BassGeneratorStage2 initialized (use_stage2={self.use_stage2})")
    
    def compose(
        self,
        *,
        section_data: dict[str, Any],
        overrides_root: Any | None = None,
        groove_profile_path: str | None = None,
        next_section_data: dict[str, Any] | None = None,
        part_specific_humanize_params: dict[str, Any] | None = None,
        shared_tracks: dict[str, Any] | None = None,
        vocal_metrics: dict | None = None,
        section: str = "Verse",
        emotion_profile: str | None = None,
    ) -> stream.Part:
        """
        Compose bass part with optional Stage2 pattern recommendation
        
        Flow:
        1. Try Stage2 pattern recommendation
        2. If successful → Apply pattern + kick sync + humanization
        3. If failed → Fallback to existing pattern-based generation
        
        Args:
            section_data: Section configuration
            shared_tracks: Shared track data (kick_offsets for sync)
            emotion_profile: Emotion profile name
            **others: Same as BassGenerator.compose()
        
        Returns:
            stream.Part: Bass part
        """
        # Stage2推薦を試みる
        if self.use_stage2 and self.recommender is not None:
            try:
                stage2_part = self._compose_with_stage2(
                    section_data=section_data,
                    shared_tracks=shared_tracks,
                    section=section,
                    emotion_profile=emotion_profile,
                )
                
                if stage2_part is not None:
                    logger.info(f"✅ Using Stage2 pattern for {section}")
                    
                    # 既存のhumanization/kick sync適用
                    stage2_part = self._apply_existing_processing(
                        stage2_part,
                        section_data=section_data,
                        shared_tracks=shared_tracks,
                        part_specific_humanize_params=part_specific_humanize_params,
                    )
                    
                    # Phase 31: Mode/Scale制約適用（Bass用）
                    self._apply_bass_scale_constraint(stage2_part, section_data)
                    
                    return stage2_part
            
            except Exception as e:
                logger.warning(f"Stage2 generation failed: {e}. Falling back to library.")
                import traceback
                logger.debug(traceback.format_exc())
        
        # Fallback: 既存のpattern-based生成
        logger.debug(f"Using default pattern generation for {section}")
        return super().compose(
            section_data=section_data,
            overrides_root=overrides_root,
            groove_profile_path=groove_profile_path,
            next_section_data=next_section_data,
            part_specific_humanize_params=part_specific_humanize_params,
            shared_tracks=shared_tracks,
            vocal_metrics=vocal_metrics,
            section=section,
            emotion_profile=emotion_profile,
        )
    
    def _compose_with_stage2(
        self,
        section_data: dict[str, Any],
        shared_tracks: dict[str, Any] | None,
        section: str,
        emotion_profile: Optional[str],
    ) -> Optional[stream.Part]:
        """
        Stage2パターン推薦による生成
        
        Returns:
            stream.Part or None (失敗時)
        """
        # Extract query parameters
        tempo = section_data.get("tempo", self.global_tempo or 120.0)
        length_measures = section_data.get("length_in_measures", 4)
        duration = length_measures * self.measure_duration  # measures → beats
        
        # Chord progression
        chord_progression = section_data.get("chord_progression", [])
        
        # Technique推定（既存パターンから）
        technique = self._estimate_technique(section, emotion_profile)
        
        # Query作成
        query = PatternQuery(
            tempo=tempo,
            technique=technique,
            duration=duration,
            chord_progression=chord_progression if chord_progression else None,
            emotion=emotion_profile,
            tempo_tolerance=30.0,  # ±30 BPM
            duration_tolerance=8.0,  # ±8 seconds
        )
        
        # Recommendation
        results = self.recommender.recommend(
            query,
            top_k=3,
            min_score=self.stage2_min_score,
        )
        
        # Check if recommendations available
        if not results:
            logger.debug(f"No Stage2 bass patterns found for {technique} @ {tempo} BPM")
            return None
        
        # Select best pattern
        best_result = results[0]
        best_pattern = best_result["pattern"]
        
        source_info = getattr(best_pattern.metadata, 'source_file', 'unknown')
        logger.info(f"  Bass: {source_info}")
        logger.info(f"    Technique: {best_pattern.metadata.technique}")
        logger.info(f"    Score: {best_result['total_score']:.3f}")
        logger.info(f"    Tempo: {best_pattern.metadata.tempo:.1f} BPM → {tempo:.1f} BPM")
        
        # Convert pattern to music21 Part
        part = self._pattern_to_part(
            best_pattern,
            section_data,
        )
        
        return part
    
    def _estimate_technique(
        self,
        section: str,
        emotion_profile: Optional[str],
    ) -> str:
        """
        Technique推定（Section + Emotionから）
        
        Heuristics:
        - Verse/Calm → walking (定常的)
        - Chorus/High → pick (攻撃的)
        - Intro/Outro → fingerstyle (繊細)
        - Bridge → slap (アクセント)
        
        Returns:
            "walking" | "pick" | "slap" | "fingerstyle"
        """
        # Emotion-based
        emotion = (emotion_profile or "neutral").lower()
        
        if "calm" in emotion or "peace" in emotion:
            base_technique = "walking"
        elif "excite" in emotion or "joy" in emotion or "happy" in emotion:
            base_technique = "pick"
        elif "tension" in emotion or "anger" in emotion:
            base_technique = "slap"
        else:
            base_technique = "fingerstyle"
        
        # Section-based override
        if section in ["Intro", "Outro"]:
            return "fingerstyle"
        elif section in ["Chorus", "Bridge"]:
            return "pick"
        elif section in ["Verse"]:
            return "walking"
        
        return base_technique
    
    def _pattern_to_part(
        self,
        pattern: Any,
        section_data: dict[str, Any],
    ) -> stream.Part:
        """
        Stage2パターン → music21 Part変換
        
        Args:
            pattern: ExtractedPattern (bass)
            section_data: Section data
        
        Returns:
            stream.Part
        """
        part = stream.Part()
        part.id = "Bass"
        
        # Instrument設定（既存のdefault_instrumentを使用）
        if self.default_instrument:
            import copy
            inst = copy.deepcopy(self.default_instrument)
            part.insert(0, inst)
        
        # Tempo設定
        target_tempo = section_data.get("tempo", self.global_tempo or 120.0)
        source_tempo = pattern.metadata.tempo
        tempo_ratio = target_tempo / source_tempo if source_tempo > 0 else 1.0
        
        # Duration調整
        target_duration = section_data.get("length_in_measures", 4) * self.measure_duration
        
        # Bass notes追加
        for note_event in pattern.notes:
            # Tempo調整
            # Note: Stage2 NoteEvent uses 'start' attribute
            note_start = getattr(note_event, 'start', getattr(note_event, 'time', 0.0))
            adjusted_start = note_start * tempo_ratio
            adjusted_duration = note_event.duration * tempo_ratio
            
            # Duration制限内に収める
            if adjusted_start >= target_duration:
                break
            
            # Note作成
            n = note.Note(pitch=note_event.pitch)
            n.quarterLength = adjusted_duration * 2  # seconds → quarter length (approx)
            n.volume.velocity = note_event.velocity
            n.offset = adjusted_start * 2
            
            # Bass range調整（E1-E4: MIDI 28-64）
            while n.pitch.midi < 28:
                n.pitch.midi += 12
            while n.pitch.midi > 64:
                n.pitch.midi -= 12
            
            part.append(n)
        
        # Sort by offset
        part = part.sorted()
        
        return part
    
    def _apply_existing_processing(
        self,
        part: stream.Part,
        section_data: dict[str, Any],
        shared_tracks: dict[str, Any] | None,
        part_specific_humanize_params: Optional[dict[str, Any]],
    ) -> stream.Part:
        """
        既存のhumanization/kick sync/controls適用
        
        This applies:
        - Kick sync (kick_lock)
        - Humanization (timing/velocity variation)
        - Velocity random walk
        - Range clamping
        """
        # Kick sync
        if shared_tracks and "kick_offsets" in shared_tracks:
            kick_offsets = list(shared_tracks["kick_offsets"])
            try:
                self._postprocess_notes_for_kick_lock(part, kick_offsets)
                logger.debug(f"Applied kick sync ({len(kick_offsets)} kicks)")
            except Exception as e:
                logger.debug(f"Kick sync skipped: {e}")
        
        # Velocity random walk
        try:
            self._apply_velocity_random_walk(part)
        except Exception as e:
            logger.debug(f"Velocity random walk skipped: {e}")
        
        # Range clamping
        try:
            self._clamp_range(part)
        except Exception as e:
            logger.debug(f"Range clamping skipped: {e}")
        
        # Kick lock (advanced)
        if self.kick_lock_cfg.get("enabled", False) and shared_tracks is not None:
            kicks = shared_tracks.get("kick_offsets_sec", [])
            try:
                self._apply_kick_lock(part, kicks)
                logger.debug(f"Applied advanced kick lock")
            except Exception as e:
                logger.debug(f"Advanced kick lock skipped: {e}")
        
        return part
    
    def _apply_bass_scale_constraint(
        self,
        part: stream.Part,
        section_data: Dict[str, Any],
        strength: float = 0.5  # Bass用デフォルト: 控えめ
    ):
        """
        Phase 31: Bass専用 Mode/Scale制約
        
        Bassは和声の基礎なので、他楽器より控えめに適用（strength=0.5）
        
        Args:
            part: Bass Part
            section_data: セクション情報
            strength: 修正強度（0.0-1.0、デフォルト0.5=50%確率）
        """
        try:
            # ops.scale_modes をインポート
            try:
                from ops.scale_modes import scale_mask_for_point
            except ImportError:
                logger.debug("scale_modes not available, skipping Phase 31")
                return
            
            # mix_context から sections と chordmap を取得
            mix_context = getattr(self, '_overrides', {}).get('mix_context')
            if not mix_context:
                return
            
            sections = mix_context.get('sections')
            chordmap = mix_context.get('chordmap')
            if not sections:
                return
            
            # ql_per_bar 取得
            time_sig = section_data.get('time_signature', '4/4')
            num, denom = map(int, time_sig.split('/'))
            ql_per_bar = float(num)
            
            import random
            
            # 全ノートを走査
            for n in part.flatten().notes:
                pitch_midi = n.pitch.midi
                offset_ql = float(n.offset)
                
                # 現在のコード情報取得
                chord_root = None
                chord_quality = None
                if chordmap:
                    bar_num = int(offset_ql / ql_per_bar)
                    chord_entry = next((c for c in chordmap if c.get("bar") == bar_num), None)
                    if chord_entry:
                        chord_symbol = chord_entry.get("chord", "")
                        if chord_symbol:
                            try:
                                from ops.scale_modes import _parse_chord_root_pc
                                chord_root = _parse_chord_root_pc(chord_symbol)
                                # quality判定
                                cs_lower = chord_symbol.lower()
                                if "maj7" in cs_lower:
                                    chord_quality = "maj7"
                                elif "min7" in cs_lower or "m7" in cs_lower:
                                    chord_quality = "min7"
                                elif "7" in chord_symbol:
                                    chord_quality = "7"
                                elif "maj" in cs_lower:
                                    chord_quality = "maj"
                                elif "min" in cs_lower or "m" in cs_lower:
                                    chord_quality = "min"
                            except Exception:
                                pass
                
                # Scale Mask 取得
                mask = scale_mask_for_point(
                    t_ql=offset_ql,
                    sections=sections,
                    chord_root=chord_root,
                    chord_quality=chord_quality
                )
                
                if not mask:
                    continue
                
                # スケール外音チェック
                pc = pitch_midi % 12
                avg_mask = sum(mask) / len(mask)
                threshold = avg_mask * 0.70
                
                if mask[pc] <= threshold:
                    # 修正強度に応じて確率的に修正
                    if random.random() > strength:
                        continue
                    
                    # 最近接スケール内音を探す
                    candidates = []
                    for offset in [1, -1, 2, -2]:
                        new_pc = (pc + offset) % 12
                        if mask[new_pc] > threshold:
                            candidates.append((abs(offset), pitch_midi + offset))
                    
                    if candidates:
                        candidates.sort()
                        new_pitch = candidates[0][1]
                        n.pitch.midi = new_pitch
                        logger.debug(f"[Bass] Phase 31: {pitch_midi} → {new_pitch} (strength={strength:.2f})")
        
        except Exception as e:
            logger.debug(f"[Bass] Phase 31 failed: {e}")


# Convenience factory
def create_bass_generator(
    use_stage2: bool = True,
    **kwargs,
) -> BassGeneratorStage2:
    """Factory function for Bass Generator"""
    return BassGeneratorStage2(use_stage2=use_stage2, **kwargs)


# =============================================================================
# Testing / CLI
# =============================================================================

def main():
    """Test BassGeneratorStage2"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test BassGeneratorStage2")
    parser.add_argument("--tempo", type=float, default=120.0, help="Tempo (BPM)")
    parser.add_argument("--section", default="Verse", help="Section name")
    parser.add_argument("--emotion", help="Emotion profile")
    parser.add_argument("--technique", help="Force technique (walking/pick/slap/fingerstyle)")
    parser.add_argument("--measures", type=int, default=4, help="Number of measures")
    parser.add_argument("--no-stage2", action="store_true", help="Disable Stage2")
    parser.add_argument("--output", default="test_bass_stage2.mid", help="Output MIDI file")
    parser.add_argument("--with-kicks", action="store_true", help="Add kick offsets for sync test")
    
    args = parser.parse_args()
    
    # Initialize generator
    from music21 import instrument as m21instrument
    
    gen = BassGeneratorStage2(
        use_stage2=not args.no_stage2,
        default_instrument=m21instrument.AcousticBass(),
        global_tempo=args.tempo,
        global_time_signature="4/4",
    )
    
    # Section data
    section_data = {
        "tempo": args.tempo,
        "length_in_measures": args.measures,
        "chord_progression": ["C", "G", "Am", "F"],  # Example
    }
    
    # Shared tracks (kick sync test)
    shared_tracks = None
    if args.with_kicks:
        # Generate kick offsets (on beats 1 and 3)
        kicks = []
        for m in range(args.measures):
            kicks.append(float(m * 4))      # beat 1
            kicks.append(float(m * 4 + 2))  # beat 3
        shared_tracks = {"kick_offsets": kicks}
    
    # Generate
    print(f"\n🎸 Generating Bass Part...")
    print(f"  Tempo: {args.tempo} BPM")
    print(f"  Section: {args.section}")
    print(f"  Emotion: {args.emotion or 'default'}")
    if args.technique:
        print(f"  Technique: {args.technique}")
    print(f"  Measures: {args.measures}")
    print(f"  Stage2: {'enabled' if not args.no_stage2 else 'disabled'}")
    if shared_tracks:
        print(f"  Kick sync: {len(shared_tracks['kick_offsets'])} kicks")
    
    try:
        part = gen.compose(
            section_data=section_data,
            section=args.section,
            emotion_profile=args.emotion,
            shared_tracks=shared_tracks,
        )
        
        # Save MIDI
        part.write("midi", fp=args.output)
        print(f"\n✅ Saved to {args.output}")
        print(f"   Notes: {len(part.flatten().notes)}")
        
        # Show pitch range
        notes = list(part.flatten().notes)
        if notes:
            pitches = [n.pitch.midi for n in notes]
            print(f"   Pitch range: {min(pitches)} - {max(pitches)} (MIDI)")
    
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
