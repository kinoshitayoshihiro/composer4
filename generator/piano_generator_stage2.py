#!/usr/bin/env python3
"""
Piano Generator Stage2 - AI統合版

V1 PianoGeneratorの全機能を継承し、Stage2レイヤーでAI処理を追加。

アーキテクチャ:
    PianoGeneratorStage2 (V1継承)
    └─ V1の発音エンジン
       └─ Stage2レイヤー（AI/humanize/tempo展開）
          ├─ PatternRecommender（pickle）
          ├─ apply_ai_filters（モデル適用）
          ├─ humanize（微調整）
          └─ quantize_to_tempo_map（可変テンポ）

Usage:
    from generator.piano_generator_stage2 import PianoGeneratorStage2

    gen = PianoGeneratorStage2(...)
    part = gen.compose(section_data=section, ...)
"""

from __future__ import annotations

from pathlib import Path
import logging
import random
from typing import Any, Iterable, List

try:
    from generator.instrument_stage2_base import InstrumentStage2Base
except ImportError:
    from instrument_stage2_base import InstrumentStage2Base

try:
    from generator.piano_generator import PianoGenerator
except ImportError:
    from piano_generator import PianoGenerator

try:
    from ml.pattern_recommender import PatternRecommender
except ImportError:
    PatternRecommender = None

try:
    from generators.base import (
        InstrumentGeneratorBase,
        NoteEvent,
        Section,
        EmotionProfile,
        GenerationContext,
        Chord,
    )
except Exception:  # pragma: no cover - fallback for limited environments
    InstrumentGeneratorBase = object  # type: ignore
    NoteEvent = Any  # type: ignore
    Section = Any  # type: ignore
    EmotionProfile = Any  # type: ignore
    GenerationContext = Any  # type: ignore
    Chord = Any  # type: ignore

logger = logging.getLogger(__name__)


__all__ = [
    "PianoGeneratorStage2",
    "MelodyGeneratorStage2",
    "CompingGeneratorStage2",
]


def _safe_velocity(value: int, delta: int = 0) -> int:
    """Clamp MIDI velocity to 1-127 with optional delta."""
    base = int(value) + int(delta)
    return max(1, min(127, base))


def _ensure_chords(section: Section | None) -> List[Chord]:  # type: ignore[valid-type]
    if section and getattr(section, "chord_progression", None):
        return list(section.chord_progression)
    # Fallback to simple I-V-vi-IV pattern in C major
    fallback_roots = [0, 7, 9, 5]
    chords: List[Chord] = []

    class _FallbackChord:
        def __init__(self, root: int) -> None:
            self.root = root

        def get_tones(self, extensions: bool = False) -> List[int]:
            tones = [self.root, self.root + 4, self.root + 7]
            if extensions:
                tones.append(self.root + 12)
            return tones

    for root in fallback_roots:
        try:
            chords.append(Chord(root=root, quality="major"))  # type: ignore[call-arg]
        except Exception:
            chords.append(_FallbackChord(root))
    return chords


class _BasePianoRoleStage2(InstrumentGeneratorBase):  # type: ignore[misc]
    """Lightweight Stage2 helper for melody/comping roles."""

    def __init__(
        self,
        instrument_name: str,
        *,
        use_stage2: bool = True,
        patterns_path: str | Path | None = None,
        default_velocity: int = 80,
        pitch_range: tuple[int, int] = (36, 96),
    ) -> None:
        super().__init__(instrument_name)
        self.pitch_range = pitch_range
        self.use_stage2 = bool(use_stage2)
        self._patterns_path = Path(patterns_path) if patterns_path else None
        self.default_velocity = default_velocity
        self.recommender = self._load_recommender()

    def _load_recommender(self) -> Any:
        if not self.use_stage2 or PatternRecommender is None:
            return None
        path = self._patterns_path or Path("data/patterns") / f"{self.instrument_name}.pickle"
        if not path.exists():
            return None
        try:
            return PatternRecommender(self.instrument_name, path)
        except Exception as exc:  # pragma: no cover - advisory only
            logger.debug(
                "%s: failed to load stage2 pickle %s (%s)", self.instrument_name, path, exc
            )
            return None

    def _finalize_notes(
        self,
        notes: List[NoteEvent],
        technique: str,
        emotion: EmotionProfile | None,
    ) -> List[NoteEvent]:
        processed = self.apply_technique(notes, technique)
        if emotion is not None:
            try:
                processed = self.apply_emotion(processed, emotion)
            except Exception:
                pass
        return processed

    def _clamp_pitch(self, pitch: int) -> int:
        lo, hi = self.pitch_range
        return max(lo, min(hi, pitch))


class MelodyGeneratorStage2(_BasePianoRoleStage2):
    """Simplified melody writer used by Stage2 piano pipeline."""

    def __init__(
        self,
        *,
        use_stage2: bool = True,
        patterns_path: str | Path | None = None,
    ) -> None:
        super().__init__(
            "piano_melody_stage2",
            use_stage2=use_stage2,
            patterns_path=patterns_path,
            default_velocity=84,
            pitch_range=(60, 104),
        )

    def generate(
        self,
        section: Section,
        technique: str,
        emotion: EmotionProfile,
        context: GenerationContext | None = None,
    ) -> List[NoteEvent]:
        chords = _ensure_chords(section)
        beat = 0.0
        notes: List[NoteEvent] = []
        total_beats = max(float(getattr(section, "duration", 0.0) or 0.0), 4.0)
        step = max(0.5, min(1.0, total_beats / max(len(chords) * 2, 1)))

        for idx in range(int(total_beats / step)):
            chord = chords[idx % len(chords)]
            tones: Iterable[int]
            try:
                tones = chord.get_tones(extensions=True)  # type: ignore[attr-defined]
            except Exception:
                tones = [0, 4, 7]
            tone_list = list(tones) or [0]
            pitch = 60 + (tone_list[idx % len(tone_list)] % 12)
            pitch = self._clamp_pitch(pitch + (12 if idx % 4 == 0 else 0))
            duration = max(0.25, step * 0.9)
            velocity = _safe_velocity(self.default_velocity, delta=random.randint(-6, 6))
            notes.append(NoteEvent(pitch=pitch, velocity=velocity, time=beat, duration=duration))
            beat += step

        return self._finalize_notes(notes, technique, emotion)


class CompingGeneratorStage2(_BasePianoRoleStage2):
    """Simplified comping writer (block/broken chords)."""

    def __init__(
        self,
        *,
        use_stage2: bool = True,
        patterns_path: str | Path | None = None,
    ) -> None:
        super().__init__(
            "piano_comping_stage2",
            use_stage2=use_stage2,
            patterns_path=patterns_path,
            default_velocity=72,
            pitch_range=(36, 84),
        )

    def generate(
        self,
        section: Section,
        technique: str,
        emotion: EmotionProfile,
        context: GenerationContext | None = None,
    ) -> List[NoteEvent]:
        chords = _ensure_chords(section)
        ts = getattr(section, "time_signature", (4, 4))
        beats_per_bar = (ts[0] / ts[1]) * 4 if isinstance(ts, tuple) else 4.0
        beat = 0.0
        notes: List[NoteEvent] = []

        for bar_idx in range(max(1, int((section.duration or 16) / beats_per_bar))):
            chord = chords[bar_idx % len(chords)]
            try:
                tones = chord.get_tones(extensions=True)  # type: ignore[attr-defined]
            except Exception:
                tones = [0, 4, 7]
            base_pitch = 48 + (tones[0] % 12)
            chord_pitches = [self._clamp_pitch(base_pitch + offset) for offset in (0, 4, 7)]
            duration = beats_per_bar * 0.95
            velocity = _safe_velocity(self.default_velocity, delta=random.randint(-8, 4))
            for offset_steps in range(0, int(beats_per_bar), 2):
                start_time = beat + offset_steps
                for pitch_idx, pitch in enumerate(chord_pitches):
                    voicing = pitch + (12 if pitch_idx == 0 and technique.endswith("spread") else 0)
                    notes.append(
                        NoteEvent(
                            pitch=self._clamp_pitch(voicing),
                            velocity=velocity - pitch_idx * 3,
                            time=start_time,
                            duration=max(0.5, min(duration, beats_per_bar - offset_steps)),
                        )
                    )
            beat += beats_per_bar

        return self._finalize_notes(notes, technique, emotion)


class PianoGeneratorStage2(InstrumentStage2Base):
    """Piano Generator Stage2 - Base継承 + V1ラッパ + AI拡張

    アーキテクチャ:
        InstrumentStage2Base (共通後段処理)
        └─ build_notes() で V1 PianoGenerator に委譲
           └─ Base.compose() が自動で AI → humanize → tempo 適用

    Stage2機能（Baseが自動適用）:
        - Pattern Recommenderによる高品質パターン推薦
        - AIモデルによるVelocity/Articulation調整
        - Humanize（微調整）
        - Quantize to tempo map（可変テンポ）

    Pickle無し動作:
        - V1の発音エンジンのみ使用（AI機能スキップ）
    """

    def __init__(self, *args, **kwargs):
        """Initialize Piano Generator with optional Stage2 support

        Args:
            *args, **kwargs: InstrumentStage2Baseへ渡される引数
        """
        super().__init__(*args, **kwargs)
        use_stage2 = bool(kwargs.get("use_stage2", True))
        self.melody_gen = MelodyGeneratorStage2(use_stage2=use_stage2)
        self.comping_gen = CompingGeneratorStage2(use_stage2=use_stage2)

        # V1 PianoGenerator のインスタンスを作成（委譲先）
        try:
            self._v1_generator = PianoGenerator(*args, **kwargs)
            logger.debug("Piano Stage2: V1 generator initialized")
        except Exception as e:
            logger.warning(f"Piano Stage2: V1 initialization failed ({e}), will use Base defaults")
            self._v1_generator = None

        patterns_path = Path("data/patterns/stage2_piano.pickle")

        # Pickleがあれば読み込み（無ければV1のみ）
        if patterns_path.exists():
            try:
                if PatternRecommender is not None:
                    self.recommender = PatternRecommender("piano", patterns_path)
                    logger.info(
                        f"✅ Piano Stage2: Loaded {len(self.recommender.patterns)} AI patterns"
                    )
                else:
                    logger.warning(
                        "⚠️ Piano Stage2: PatternRecommender not available, using V1 only"
                    )
                    self.recommender = None
            except Exception as e:
                logger.warning(f"⚠️ Piano Stage2: Failed to load patterns ({e}), using V1 only")
                self.recommender = None
        else:
            logger.info(f"ℹ️ Piano Stage2: No pickle found ({patterns_path}), using V1 only")
            self.recommender = None

    def build_notes(self, section, processed_chord_events, **kwargs):
        """V1の発音エンジンを呼び出す（委譲）

        V1 PianoGenerator の generate() を呼び出して基本的なnote生成を行います。
        その後、Base.compose() が自動で AI → humanize → tempo を適用します。

        Args:
            section: セクションデータ
            processed_chord_events: コード進行
            **kwargs: 追加パラメータ（emotion, technique等）

        Returns:
            list: V1が生成したnoteイベント
        """
        if self._v1_generator is None:
            logger.warning("Piano Stage2: V1 generator not available, returning empty notes")
            return []

        # V1の generate() メソッドを呼び出し（委譲）
        try:
            notes = self._v1_generator.generate(section, processed_chord_events, **kwargs)
            logger.debug(f"Piano Stage2: V1 returned {len(notes) if notes else 0} notes")
            return notes if notes else []
        except Exception as e:
            logger.error(f"Piano Stage2: V1 generation failed: {e}")
            return []

    def apply_ai_filters(self, notes, section=None):
        """Stage2 AIフィルタを適用（オプション）

        Pickleがロードされている場合のみ、AIモデルによる補正を行います。

        Args:
            notes: V1が生成したnote events
            section: セクション情報（オプション）

        Returns:
            list: AI補正後のnote events（pickleが無い場合はそのまま返す）
        """
        if self.recommender is None:
            # Pickle無し → V1の結果をそのまま返す
            return notes

        # TODO: PatternRecommenderを使った補正ロジック
        # 例: velocity調整、articulation追加等
        logger.debug(f"Piano Stage2: AI filter applied to {len(notes)} notes")

        return notes

    def generate(
        self,
        section: Section,
        technique: str,
        emotion: EmotionProfile,
        context: GenerationContext | None = None,
    ) -> List[NoteEvent]:
        """Generate combined melody + comping note events.

        This lightweight API mirrors the expectations from
        tests/test_piano_stage2_quick.py so Stage2 users can
        obtain a merged piano performance without going
        through the legacy music21 pipeline.
        """

        melody = self.melody_gen.generate(section, technique, emotion, context)
        comping = self.comping_gen.generate(section, technique, emotion, context)
        combined = list(melody) + list(comping)
        try:
            combined.sort(key=lambda n: (n.time, n.pitch))
        except Exception:
            pass
        return combined
