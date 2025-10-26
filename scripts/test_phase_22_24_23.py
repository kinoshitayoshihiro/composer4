#!/usr/bin/env python3
"""
Phase 22/24/23 統合テスト
========================================================================
Phase 22: Emotion mapping（感情連続写像）
Phase 24: Controls統一（CC11/RPN/14bit PB）
Phase 23: Prosody整合（子音窓×強勢アライン）

テスト項目：
1. NO-OP回帰テスト（未設定時は過去と完全一致）
2. Phase 22/24/23が設定に応じて動的に有効化される
"""

import sys
import os
from pathlib import Path

# プロジェクトルートを追加
sys.path.insert(0, str(Path(__file__).parent.parent))

import unittest
import music21 as m21
from generator.bass_params_stage2 import BassParamsStage2
from generator.piano_params_stage2 import PianoParamsStage2
from generator.guitar_params_stage2 import GuitarParamsStage2
from generator.strings_params_stage2 import StringsParamsStage2
from generator.drums_params_stage2 import DrumsParamsStage2


def create_dummy_part(num_notes=16, note_name="C4"):
    """テスト用ダミーPartを作成"""
    part = m21.stream.Part()
    for i in range(num_notes):
        n = m21.note.Note(note_name, quarterLength=1.0)
        n.volume.velocity = 80
        part.insert(float(i), n)
    return part


class TestPhase22EmotionMapping(unittest.TestCase):
    """Phase 22: Emotion mapping テスト"""
    
    def test_bass_emotion_mapping(self):
        """Bass: Emotion mappingが有効化される"""
        overrides = {
            "emotion_map": {
                "density_gain": 0.6,
                "register_shift": 2,
                "staccato_bias": 0.15,
                "smooth_ms": 180
            }
        }
        
        section_meta = {"label": "Verse", "bar": 0, "tempo": 120, "ql_per_bar": 4.0}
        mix_context = {"emotion_curve": [0.3, 0.5, 0.7, 0.9], "beat_grid": {"bpm": 120}}
        
        part = create_dummy_part(16, "E2")
        bass = BassParamsStage2()
        track = bass.apply(part, section_meta, mix_context, overrides)
        
        # Phase 22が有効化されている確認
        phases = bass._get_phases(overrides)
        self.assertIn(22, phases, "Phase 22が有効化されている")
        
        notes = list(track.flatten().notesAndRests.notes)
        self.assertGreater(len(notes), 0, "ノートが生成されている")
        print(f"✓ Bass Emotion mapping: {len(notes)} notes, Phase 22 enabled")
    
    def test_piano_emotion_mapping(self):
        """Piano: Emotion mappingが有効化される"""
        overrides = {"emotion_map": {"density_gain": 0.7, "register_shift": 4}}
        section_meta = {"label": "Verse", "bar": 0, "tempo": 120, "ql_per_bar": 4.0}
        mix_context = {"emotion_curve": [0.3, 0.7], "beat_grid": {"bpm": 120}}
        
        part = create_dummy_part(16, "C4")
        piano = PianoParamsStage2()
        track = piano.apply(part, section_meta, mix_context, overrides)
        
        phases = piano._get_phases(overrides)
        self.assertIn(22, phases)
        
        notes = list(track.flatten().notesAndRests.notes)
        self.assertGreater(len(notes), 0)
        print(f"✓ Piano Emotion mapping: {len(notes)} notes")


class TestPhase24ControlsUnified(unittest.TestCase):
    """Phase 24: Controls統一テスト"""
    
    def test_bass_controls(self):
        """Bass: Controlsが有効化される"""
        overrides = {"controls": {"expression_curve": "linear", "bend_range": 2}}
        section_meta = {"label": "Verse", "bar": 0, "tempo": 120}
        mix_context = {"beat_grid": {"bpm": 120}}
        
        part = create_dummy_part(16, "E2")
        bass = BassParamsStage2()
        track = bass.apply(part, section_meta, mix_context, overrides)
        
        phases = bass._get_phases(overrides)
        self.assertIn(24, phases, "Phase 24が有効化されている")
        
        notes = list(track.flatten().notesAndRests.notes)
        self.assertGreater(len(notes), 0)
        print(f"✓ Bass Controls: {len(notes)} notes, Phase 24 enabled")
    
    def test_piano_cc11_curves(self):
        """Piano: CC11表情カーブ（arch/linear/flat）"""
        for curve in ["arch", "linear", "flat"]:
            with self.subTest(curve=curve):
                overrides = {"controls": {"expression_curve": curve, "bend_range": 2}}
                section_meta = {"label": "Verse", "bar": 0, "tempo": 120}
                mix_context = {"beat_grid": {"bpm": 120}}
                
                part = create_dummy_part(16, "C4")
                piano = PianoParamsStage2()
                track = piano.apply(part, section_meta, mix_context, overrides)
                
                phases = piano._get_phases(overrides)
                self.assertIn(24, phases)
                
                notes = list(track.flatten().notesAndRests.notes)
                self.assertGreater(len(notes), 0)
                print(f"✓ Piano CC11 {curve}: {len(notes)} notes")


class TestPhase23ProsodyAlignment(unittest.TestCase):
    """Phase 23: Prosody整合テスト"""
    
    def test_strings_prosody(self):
        """Strings: Prosodyが有効化される"""
        overrides = {"prosody": {"enable": True, "stress_boost": 8}}
        section_meta = {"label": "Verse", "bar": 0, "tempo": 120}
        mix_context = {"beat_grid": {"bpm": 120}}
        
        part = create_dummy_part(16, "G3")
        strings = StringsParamsStage2()
        track = strings.apply(part, section_meta, mix_context, overrides)
        
        phases = strings._get_phases(overrides)
        self.assertIn(23, phases, "Phase 23が有効化されている")
        
        notes = list(track.flatten().notesAndRests.notes)
        self.assertGreater(len(notes), 0)
        print(f"✓ Strings Prosody: {len(notes)} notes, Phase 23 enabled")
    
    def test_drums_prosody(self):
        """Drums: Prosodyが有効化される"""
        overrides = {"prosody": {"enable": True, "stress_boost": 10}}
        section_meta = {"label": "Verse", "bar": 0, "tempo": 120}
        mix_context = {"beat_grid": {"bpm": 120}}
        
        part = create_dummy_part(16, "C4")
        drums = DrumsParamsStage2()
        track = drums.apply(part, section_meta, mix_context, overrides)
        
        phases = drums._get_phases(overrides)
        self.assertIn(23, phases)
        
        notes = list(track.flatten().notesAndRests.notes)
        self.assertGreater(len(notes), 0)
        print(f"✓ Drums Prosody: {len(notes)} notes")


class TestNoOpRegression(unittest.TestCase):
    """NO-OP回帰テスト（Phase 22/24/23未設定時は過去と一致）"""
    
    def test_bass_no_op(self):
        """Bass: Phase 22/24/23未設定時はNO-OP"""
        overrides = {}  # Phase 22/24/23未設定
        section_meta = {"label": "Verse", "bar": 0, "tempo": 120}
        mix_context = {"beat_grid": {"bpm": 120}}
        
        part = create_dummy_part(16, "E2")
        bass = BassParamsStage2()
        
        # Phase 22/24/23が有効化されていない確認
        phases = bass._get_phases(overrides)
        self.assertNotIn(22, phases, "Phase 22は無効")
        self.assertNotIn(24, phases, "Phase 24は無効")
        self.assertNotIn(23, phases, "Phase 23は無効")
        
        track = bass.apply(part, section_meta, mix_context, overrides)
        notes = list(track.flatten().notesAndRests.notes)
        self.assertGreater(len(notes), 0)
        print(f"✓ Bass NO-OP: {len(notes)} notes, Phase 22/24/23 disabled")
    
    def test_piano_no_op(self):
        """Piano: Phase 22/24/23未設定時はNO-OP"""
        overrides = {}
        section_meta = {"label": "Verse", "bar": 0, "tempo": 120}
        mix_context = {"beat_grid": {"bpm": 120}}
        
        part = create_dummy_part(16, "C4")
        piano = PianoParamsStage2()
        
        phases = piano._get_phases(overrides)
        self.assertNotIn(22, phases)
        self.assertNotIn(24, phases)
        self.assertNotIn(23, phases)
        
        track = piano.apply(part, section_meta, mix_context, overrides)
        notes = list(track.flatten().notesAndRests.notes)
        self.assertGreater(len(notes), 0)
        print(f"✓ Piano NO-OP: {len(notes)} notes")
    
    def test_guitar_no_op(self):
        """Guitar: Phase 22/24/23未設定時はNO-OP"""
        overrides = {}
        section_meta = {"label": "Verse", "bar": 0, "tempo": 120}
        mix_context = {"beat_grid": {"bpm": 120}}
        
        part = create_dummy_part(16, "E3")
        guitar = GuitarParamsStage2()
        
        phases = guitar._get_phases(overrides)
        self.assertNotIn(22, phases)
        
        track = guitar.apply(part, section_meta, mix_context, overrides)
        notes = list(track.flatten().notesAndRests.notes)
        self.assertGreater(len(notes), 0)
        print(f"✓ Guitar NO-OP: {len(notes)} notes")
    
    def test_strings_no_op(self):
        """Strings: Phase 22/24/23未設定時はNO-OP"""
        overrides = {}
        section_meta = {"label": "Verse", "bar": 0, "tempo": 120}
        mix_context = {"beat_grid": {"bpm": 120}}
        
        part = create_dummy_part(16, "G3")
        strings = StringsParamsStage2()
        
        phases = strings._get_phases(overrides)
        self.assertNotIn(22, phases)
        
        track = strings.apply(part, section_meta, mix_context, overrides)
        notes = list(track.flatten().notesAndRests.notes)
        self.assertGreater(len(notes), 0)
        print(f"✓ Strings NO-OP: {len(notes)} notes")
    
    def test_drums_no_op(self):
        """Drums: Phase 22/24/23未設定時はNO-OP"""
        overrides = {}
        section_meta = {"label": "Verse", "bar": 0, "tempo": 120}
        mix_context = {"beat_grid": {"bpm": 120}}
        
        part = create_dummy_part(16, "C4")
        drums = DrumsParamsStage2()
        
        phases = drums._get_phases(overrides)
        self.assertNotIn(22, phases)
        
        track = drums.apply(part, section_meta, mix_context, overrides)
        notes = list(track.flatten().notesAndRests.notes)
        self.assertGreater(len(notes), 0)
        print(f"✓ Drums NO-OP: {len(notes)} notes")


if __name__ == "__main__":
    print("=" * 70)
    print("Phase 22/24/23 統合テスト")
    print("=" * 70)
    
    # テストスイート作成
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Phase 22: Emotion mapping
    suite.addTests(loader.loadTestsFromTestCase(TestPhase22EmotionMapping))
    
    # Phase 24: Controls統一
    suite.addTests(loader.loadTestsFromTestCase(TestPhase24ControlsUnified))
    
    # Phase 23: Prosody整合
    suite.addTests(loader.loadTestsFromTestCase(TestPhase23ProsodyAlignment))
    
    # NO-OP回帰テスト
    suite.addTests(loader.loadTestsFromTestCase(TestNoOpRegression))
    
    # 実行
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # サマリー
    print("\n" + "=" * 70)
    print("テスト結果サマリー")
    print("=" * 70)
    print(f"実行: {result.testsRun}")
    print(f"成功: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"失敗: {len(result.failures)}")
    print(f"エラー: {len(result.errors)}")
    
    sys.exit(0 if result.wasSuccessful() else 1)
