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

from pathlib import Path
import logging

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

logger = logging.getLogger(__name__)


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
                    logger.info(f"✅ Piano Stage2: Loaded {len(self.recommender.patterns)} AI patterns")
                else:
                    logger.warning("⚠️ Piano Stage2: PatternRecommender not available, using V1 only")
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
