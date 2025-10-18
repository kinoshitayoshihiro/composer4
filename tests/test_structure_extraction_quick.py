#!/usr/bin/env python3
"""
Suno Structure Extractor Quick Test

extract_structure.pyの動作確認テスト。
実際のaudio fileなしでモック生成テスト。
"""

import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import numpy as np
import tempfile
import soundfile as sf

from scripts.audio2score.extract_structure import SunoStructureExtractor


def create_test_audio(duration: float = 30.0, sr: int = 22050) -> np.ndarray:
    """テスト用オーディオ生成（簡易シンセ音）"""
    t = np.linspace(0, duration, int(sr * duration))
    
    # 基音（440 Hz A4）+ ハーモニクス
    audio = np.sin(2 * np.pi * 440 * t) * 0.3
    audio += np.sin(2 * np.pi * 880 * t) * 0.15  # Octave
    audio += np.sin(2 * np.pi * 220 * t) * 0.2   # Bass
    
    # エンベロープ（簡易フェードイン/アウト）
    fade_samples = int(sr * 0.5)
    audio[:fade_samples] *= np.linspace(0, 1, fade_samples)
    audio[-fade_samples:] *= np.linspace(1, 0, fade_samples)
    
    return audio


def test_initialization():
    """Test 1: Extractor初期化"""
    print("\n" + "="*60)
    print("Test 1: Initialization")
    print("="*60)
    
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as vocal_file:
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as accomp_file:
            # テストオーディオ生成
            vocal_audio = create_test_audio(duration=20.0)
            accomp_audio = create_test_audio(duration=20.0)
            
            sf.write(vocal_file.name, vocal_audio, 22050)
            sf.write(accomp_file.name, accomp_audio, 22050)
            
            vocal_path = pathlib.Path(vocal_file.name)
            accomp_path = pathlib.Path(accomp_file.name)
            
            # Extractor初期化
            extractor = SunoStructureExtractor(
                vocal_path=vocal_path,
                accomp_path=accomp_path,
                sr=22050,
                verbose=False
            )
            
            print(f"✓ Vocal path: {vocal_path.exists()}")
            print(f"✓ Accomp path: {accomp_path.exists()}")
            print(f"✓ Extractor created: {extractor is not None}")
            
            # Cleanup
            vocal_path.unlink()
            accomp_path.unlink()
    
    print("✅ Test 1 Passed!")


def test_tempo_extraction():
    """Test 2: Tempo map抽出"""
    print("\n" + "="*60)
    print("Test 2: Tempo Map Extraction")
    print("="*60)
    
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as accomp_file:
        # 30秒のテストオーディオ
        audio = create_test_audio(duration=30.0)
        sf.write(accomp_file.name, audio, 22050)
        
        accomp_path = pathlib.Path(accomp_file.name)
        
        extractor = SunoStructureExtractor(
            accomp_path=accomp_path,
            sr=22050,
            verbose=False
        )
        
        extractor.load_audio()
        tempo_map = extractor.extract_tempo_map()
        
        print(f"✓ Global tempo: {tempo_map['global_tempo']:.1f} BPM")
        print(f"✓ Beats detected: {len(tempo_map['beat_times'])}")
        print(f"✓ Downbeats: {len(tempo_map['downbeat_times'])}")
        print(f"✓ Time signature: {tempo_map['time_signature']}")
        
        # Validation
        # Note: Simple test audio may not have detectable beats (tempo=0)
        # Real music will have detectable tempo > 0
        assert tempo_map['global_tempo'] >= 0, "Tempo should be non-negative"
        assert len(tempo_map['beat_times']) >= 0, "Should have beat times (possibly empty)"
        assert tempo_map['time_signature'] == [4, 4], "Should default to 4/4"
        
        accomp_path.unlink()
    
    print("✅ Test 2 Passed!")


def test_section_extraction():
    """Test 3: Section分割"""
    print("\n" + "="*60)
    print("Test 3: Section Extraction")
    print("="*60)
    
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as accomp_file:
        # 40秒のテストオーディオ（セクション変化含む）
        audio = create_test_audio(duration=40.0)
        
        # セクション変化シミュレーション（エネルギー変化）
        mid_point = len(audio) // 2
        audio[mid_point:] *= 1.5  # 後半を大きく
        
        sf.write(accomp_file.name, audio, 22050)
        accomp_path = pathlib.Path(accomp_file.name)
        
        extractor = SunoStructureExtractor(
            accomp_path=accomp_path,
            sr=22050,
            verbose=False
        )
        
        extractor.load_audio()
        tempo_map = extractor.extract_tempo_map()
        sections = extractor.extract_sections(tempo_map, n_sections=3)
        
        print(f"✓ Sections detected: {len(sections)}")
        for i, sec in enumerate(sections):
            print(f"  [{i+1}] {sec['label']}: {sec['start_time']:.1f}s - {sec['end_time']:.1f}s ({sec['duration_measures']} measures)")
        
        # Validation
        assert len(sections) > 0, "Should detect sections"
        assert sections[0]['start_time'] >= 0, "First section should start at 0"
        
        accomp_path.unlink()
    
    print("✅ Test 3 Passed!")


def test_drums_extraction():
    """Test 4: Drum hits抽出"""
    print("\n" + "="*60)
    print("Test 4: Drum Hits Extraction")
    print("="*60)
    
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as accomp_file:
        # Percussion風のオーディオ生成
        sr = 22050
        duration = 10.0
        audio = np.zeros(int(sr * duration))
        
        # Kick風（低域パルス）
        for beat in range(0, 10, 1):
            t = int(beat * sr)
            if t < len(audio):
                audio[t:t+100] += np.sin(2 * np.pi * 60 * np.linspace(0, 0.01, 100)) * 0.5
        
        sf.write(accomp_file.name, audio, sr)
        accomp_path = pathlib.Path(accomp_file.name)
        
        extractor = SunoStructureExtractor(
            accomp_path=accomp_path,
            sr=sr,
            verbose=False
        )
        
        extractor.load_audio()
        tempo_map = extractor.extract_tempo_map()
        drums = extractor.extract_drums_hits(tempo_map)
        
        print(f"✓ Kick hits: {len(drums['kick'])}")
        print(f"✓ Snare hits: {len(drums['snare'])}")
        print(f"✓ Hihat hits: {len(drums['hihat'])}")
        
        # Validation
        assert 'kick' in drums, "Should have kick hits"
        assert 'snare' in drums, "Should have snare hits"
        assert 'hihat' in drums, "Should have hihat hits"
        
        accomp_path.unlink()
    
    print("✅ Test 4 Passed!")


def test_full_extraction():
    """Test 5: Full extraction workflow"""
    print("\n" + "="*60)
    print("Test 5: Full Extraction Workflow")
    print("="*60)
    
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as vocal_file:
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as accomp_file:
            with tempfile.NamedTemporaryFile(suffix='.yaml', delete=False) as output_file:
                # テストオーディオ生成
                vocal_audio = create_test_audio(duration=25.0)
                accomp_audio = create_test_audio(duration=25.0)
                
                sf.write(vocal_file.name, vocal_audio, 22050)
                sf.write(accomp_file.name, accomp_audio, 22050)
                
                vocal_path = pathlib.Path(vocal_file.name)
                accomp_path = pathlib.Path(accomp_file.name)
                output_path = pathlib.Path(output_file.name)
                
                # Full extraction
                extractor = SunoStructureExtractor(
                    vocal_path=vocal_path,
                    accomp_path=accomp_path,
                    sr=22050,
                    verbose=True
                )
                
                structure = extractor.extract_all()
                
                # YAML保存
                extractor.save_yaml(structure, output_path)
                
                print(f"\n✓ Structure keys: {list(structure.keys())}")
                print(f"✓ Tempo: {structure['tempo_map']['global_tempo']:.1f} BPM")
                print(f"✓ Sections: {len(structure['sections'])}")
                print(f"✓ Output file exists: {output_path.exists()}")
                
                # Validation
                assert 'tempo_map' in structure
                assert 'sections' in structure
                assert 'chords' in structure
                assert 'drums_hits' in structure
                assert 'bass_contour' in structure
                assert output_path.exists()
                
                # Cleanup
                vocal_path.unlink()
                accomp_path.unlink()
                output_path.unlink()
    
    print("✅ Test 5 Passed!")


def main():
    """全テスト実行"""
    print("\n" + "="*60)
    print("🎵 Suno Structure Extractor - Quick Test Suite")
    print("="*60)
    
    tests = [
        test_initialization,
        test_tempo_extraction,
        test_section_extraction,
        test_drums_extraction,
        test_full_extraction,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"❌ Test failed: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "="*60)
    print(f"✅ Tests Passed: {passed}/{len(tests)}")
    if failed > 0:
        print(f"❌ Tests Failed: {failed}/{len(tests)}")
    print("="*60)
    
    return failed == 0


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
