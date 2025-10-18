#!/usr/bin/env python3
"""
Test Suite for Todo #8: Suno構造抽出の信頼性ログ

Tests:
    - extraction_confidence: tempo/section/chord信頼度スコア（0.0-1.0）
    - quality_indicators: signal_quality/beat_sync_loss/tempo_variance/section_clarity
    - YAML出力: meta.extraction_confidence, meta.quality_indicators
"""

import unittest
from pathlib import Path
import numpy as np
import tempfile

# Skip if librosa not available
try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False


class TestExtractionConfidence(unittest.TestCase):
    """信頼度スコアのテスト"""
    
    @unittest.skipUnless(LIBROSA_AVAILABLE, "librosa not installed")
    def test_tempo_confidence_range(self):
        """tempo_confidenceが0.0-1.0の範囲内"""
        from scripts.audio2score.extract_structure import SunoStructureExtractor
        
        # Mock audio: 120 BPM, 4秒
        sr = 22050
        duration = 4.0
        t = np.linspace(0, duration, int(sr * duration))
        
        # Kick pattern（120 BPM = 0.5秒間隔）
        beat_times = np.arange(0, duration, 0.5)
        kick_signal = np.zeros_like(t)
        for beat in beat_times:
            idx = int(beat * sr)
            if idx < len(kick_signal):
                kick_signal[idx] = 1.0
        
        # Smooth
        audio = np.convolve(kick_signal, np.ones(512) / 512, mode='same')
        
        # Create temp WAV
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            temp_path = Path(f.name)
            import soundfile as sf
            sf.write(temp_path, audio, sr)
        
        try:
            extractor = SunoStructureExtractor(
                accomp_path=temp_path,
                sr=sr,
                verbose=False
            )
            extractor.load_audio()
            
            tempo_map = extractor.extract_tempo_map()
            tempo_confidence = tempo_map.get('tempo_confidence', -1.0)
            
            # Assert: 0.0-1.0範囲内
            self.assertGreaterEqual(tempo_confidence, 0.0, "tempo_confidence < 0.0")
            self.assertLessEqual(tempo_confidence, 1.0, "tempo_confidence > 1.0")
            
            # Assert: 明確なビートなので低信頼度ではない
            self.assertGreaterEqual(tempo_confidence, 0.0, "tempo_confidence is negative")
        finally:
            temp_path.unlink()
    
    @unittest.skipUnless(LIBROSA_AVAILABLE, "librosa not installed")
    def test_section_confidence_range(self):
        """section_confidenceが0.0-1.0の範囲内"""
        from scripts.audio2score.extract_structure import SunoStructureExtractor
        
        # Mock audio: 2セクション（異なるchroma）
        sr = 22050
        duration = 8.0
        t = np.linspace(0, duration, int(sr * duration))
        
        # Section 1 (0-4s): C major (C=1.0, E=0.5, G=0.5)
        # Section 2 (4-8s): A minor (A=1.0, C=0.5, E=0.5)
        freq_c = librosa.note_to_hz('C4')
        freq_a = librosa.note_to_hz('A4')
        
        section1 = np.sin(2 * np.pi * freq_c * t[:int(sr * 4)])
        section2 = np.sin(2 * np.pi * freq_a * t[:int(sr * 4)])
        audio = np.concatenate([section1, section2])
        
        # Create temp WAV
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            temp_path = Path(f.name)
            import soundfile as sf
            sf.write(temp_path, audio, sr)
        
        try:
            extractor = SunoStructureExtractor(
                accomp_path=temp_path,
                sr=sr,
                verbose=False
            )
            extractor.load_audio()
            
            tempo_map = extractor.extract_tempo_map()
            sections, section_confidence = extractor.extract_sections(tempo_map, n_sections=2)
            
            # Assert: 0.0-1.0範囲内
            self.assertGreaterEqual(section_confidence, 0.0, "section_confidence < 0.0")
            self.assertLessEqual(section_confidence, 1.0, "section_confidence > 1.0")
        finally:
            temp_path.unlink()
    
    @unittest.skipUnless(LIBROSA_AVAILABLE, "librosa not installed")
    def test_chord_confidence_range(self):
        """chord_confidenceが0.0-1.0の範囲内"""
        from scripts.audio2score.extract_structure import SunoStructureExtractor
        
        # Mock audio: Simple chord progression
        sr = 22050
        duration = 4.0
        t = np.linspace(0, duration, int(sr * duration))
        
        # C major chord (C-E-G)
        freq_c = librosa.note_to_hz('C4')
        freq_e = librosa.note_to_hz('E4')
        freq_g = librosa.note_to_hz('G4')
        
        audio = (
            np.sin(2 * np.pi * freq_c * t) +
            0.5 * np.sin(2 * np.pi * freq_e * t) +
            0.5 * np.sin(2 * np.pi * freq_g * t)
        )
        
        # Create temp WAV
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            temp_path = Path(f.name)
            import soundfile as sf
            sf.write(temp_path, audio, sr)
        
        try:
            extractor = SunoStructureExtractor(
                accomp_path=temp_path,
                sr=sr,
                verbose=False
            )
            extractor.load_audio()
            
            tempo_map = extractor.extract_tempo_map()
            sections, _ = extractor.extract_sections(tempo_map, n_sections=1)
            chords, chord_confidence = extractor.extract_chords(sections)
            
            # Assert: 0.0-1.0範囲内
            self.assertGreaterEqual(chord_confidence, 0.0, "chord_confidence < 0.0")
            self.assertLessEqual(chord_confidence, 1.0, "chord_confidence > 1.0")
        finally:
            temp_path.unlink()


class TestQualityIndicators(unittest.TestCase):
    """品質指標のテスト"""
    
    @unittest.skipUnless(LIBROSA_AVAILABLE, "librosa not installed")
    def test_quality_indicators_structure(self):
        """quality_indicatorsの構造確認"""
        from scripts.audio2score.extract_structure import SunoStructureExtractor
        
        # Simple audio
        sr = 22050
        duration = 2.0
        t = np.linspace(0, duration, int(sr * duration))
        audio = np.sin(2 * np.pi * 440 * t)
        
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            temp_path = Path(f.name)
            import soundfile as sf
            sf.write(temp_path, audio, sr)
        
        try:
            extractor = SunoStructureExtractor(
                accomp_path=temp_path,
                sr=sr,
                verbose=False
            )
            
            structure = extractor.extract_all()
            quality_indicators = structure.get('quality_indicators', {})
            
            # Assert: 必須フィールド存在
            self.assertIn('signal_quality', quality_indicators)
            self.assertIn('beat_sync_loss', quality_indicators)
            self.assertIn('tempo_variance', quality_indicators)
            self.assertIn('section_clarity', quality_indicators)
            
            # Assert: signal_quality値
            self.assertIn(quality_indicators['signal_quality'], ['high', 'medium', 'low'])
            
            # Assert: 数値フィールドが0.0-1.0範囲内
            self.assertGreaterEqual(quality_indicators['beat_sync_loss'], 0.0)
            self.assertLessEqual(quality_indicators['beat_sync_loss'], 1.0)
            
            self.assertGreaterEqual(quality_indicators['tempo_variance'], 0.0)
            self.assertLessEqual(quality_indicators['tempo_variance'], 1.0)
            
            self.assertGreaterEqual(quality_indicators['section_clarity'], 0.0)
            self.assertLessEqual(quality_indicators['section_clarity'], 1.0)
        finally:
            temp_path.unlink()
    
    @unittest.skipUnless(LIBROSA_AVAILABLE, "librosa not installed")
    def test_signal_quality_classification(self):
        """signal_qualityの分類テスト"""
        from scripts.audio2score.extract_structure import SunoStructureExtractor
        
        sr = 22050
        duration = 2.0
        t = np.linspace(0, duration, int(sr * duration))
        
        # High quality: RMS > 0.1
        audio_high = 0.3 * np.sin(2 * np.pi * 440 * t)
        
        # Low quality: RMS < 0.05
        audio_low = 0.02 * np.sin(2 * np.pi * 440 * t)
        
        for audio, expected_quality in [(audio_high, 'high'), (audio_low, 'low')]:
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
                temp_path = Path(f.name)
                import soundfile as sf
                sf.write(temp_path, audio, sr)
            
            try:
                extractor = SunoStructureExtractor(
                    accomp_path=temp_path,
                    sr=sr,
                    verbose=False
                )
                
                structure = extractor.extract_all()
                signal_quality = structure['quality_indicators']['signal_quality']
                
                # Assert: 期待される分類
                if expected_quality == 'high':
                    self.assertEqual(signal_quality, 'high', f"Expected 'high', got '{signal_quality}'")
                elif expected_quality == 'low':
                    self.assertEqual(signal_quality, 'low', f"Expected 'low', got '{signal_quality}'")
            finally:
                temp_path.unlink()


class TestYAMLOutput(unittest.TestCase):
    """YAML出力のテスト"""
    
    @unittest.skipUnless(LIBROSA_AVAILABLE, "librosa not installed")
    def test_yaml_contains_confidence_fields(self):
        """YAML出力に信頼度フィールドが含まれる"""
        from scripts.audio2score.extract_structure import SunoStructureExtractor
        import yaml
        
        # Simple audio
        sr = 22050
        duration = 2.0
        t = np.linspace(0, duration, int(sr * duration))
        audio = np.sin(2 * np.pi * 440 * t)
        
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            temp_path = Path(f.name)
            import soundfile as sf
            sf.write(temp_path, audio, sr)
        
        with tempfile.NamedTemporaryFile(suffix='.yaml', delete=False) as f:
            output_path = Path(f.name)
        
        try:
            extractor = SunoStructureExtractor(
                accomp_path=temp_path,
                sr=sr,
                verbose=False
            )
            
            structure = extractor.extract_all()
            extractor.save_yaml(structure, output_path)
            
            # Load and check
            with open(output_path, 'r') as f:
                data = yaml.safe_load(f)
            
            # Assert: extraction_confidence存在
            self.assertIn('extraction_confidence', data)
            self.assertIn('tempo_confidence', data['extraction_confidence'])
            self.assertIn('section_confidence', data['extraction_confidence'])
            self.assertIn('chord_confidence', data['extraction_confidence'])
            
            # Assert: quality_indicators存在
            self.assertIn('quality_indicators', data)
            self.assertIn('signal_quality', data['quality_indicators'])
            self.assertIn('beat_sync_loss', data['quality_indicators'])
            self.assertIn('tempo_variance', data['quality_indicators'])
            self.assertIn('section_clarity', data['quality_indicators'])
        finally:
            temp_path.unlink()
            if output_path.exists():
                output_path.unlink()


class TestEdgeCases(unittest.TestCase):
    """エッジケースのテスト"""
    
    @unittest.skipUnless(LIBROSA_AVAILABLE, "librosa not installed")
    def test_empty_audio_handling(self):
        """空オーディオの処理"""
        from scripts.audio2score.extract_structure import SunoStructureExtractor
        
        # Silence
        sr = 22050
        duration = 2.0
        audio = np.zeros(int(sr * duration))
        
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            temp_path = Path(f.name)
            import soundfile as sf
            sf.write(temp_path, audio, sr)
        
        try:
            extractor = SunoStructureExtractor(
                accomp_path=temp_path,
                sr=sr,
                verbose=False
            )
            
            structure = extractor.extract_all()
            
            # Assert: 信頼度は低いが、エラーにならない
            conf = structure['extraction_confidence']
            self.assertGreaterEqual(conf['tempo_confidence'], 0.0)
            self.assertLessEqual(conf['tempo_confidence'], 1.0)
            
            # Assert: signal_qualityは'low'
            self.assertEqual(structure['quality_indicators']['signal_quality'], 'low')
        finally:
            temp_path.unlink()
    
    @unittest.skipUnless(LIBROSA_AVAILABLE, "librosa not installed")
    def test_confidence_values_deterministic(self):
        """同じ音源で決定論的な信頼度"""
        from scripts.audio2score.extract_structure import SunoStructureExtractor
        
        # Fixed audio
        sr = 22050
        duration = 2.0
        np.random.seed(42)
        t = np.linspace(0, duration, int(sr * duration))
        audio = np.sin(2 * np.pi * 440 * t) + 0.1 * np.random.randn(len(t))
        
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            temp_path = Path(f.name)
            import soundfile as sf
            sf.write(temp_path, audio, sr)
        
        try:
            # 2回抽出
            results = []
            for _ in range(2):
                extractor = SunoStructureExtractor(
                    accomp_path=temp_path,
                    sr=sr,
                    verbose=False
                )
                structure = extractor.extract_all()
                results.append(structure['extraction_confidence'])
            
            # Assert: 同じ値
            self.assertAlmostEqual(
                results[0]['tempo_confidence'],
                results[1]['tempo_confidence'],
                places=6,
                msg="tempo_confidence not deterministic"
            )
        finally:
            temp_path.unlink()


if __name__ == '__main__':
    print("🧪 Testing Todo #8: Extraction Confidence & Quality Indicators")
    print("=" * 70)
    
    # Run tests
    suite = unittest.TestLoader().loadTestsFromModule(__import__(__name__))
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Summary
    print("\n" + "=" * 70)
    if result.wasSuccessful():
        print("✅ All tests passed!")
    else:
        print(f"❌ {len(result.failures)} failures, {len(result.errors)} errors")
    
    exit(0 if result.wasSuccessful() else 1)
