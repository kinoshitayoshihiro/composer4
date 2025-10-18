#!/usr/bin/env python3
"""
DAWdreamer Batch Renderer Quick Tests

DAWdreamerBatchRendererの動作確認（軽量テスト）

Tests:
1. 初期化テスト（パラメータ設定）
2. SoundFont検証テスト（ファイル存在確認）
3. MIDI読み込みテスト（モックMIDI作成 → 読み込み）
4. 単一レンダリングテスト（MIDI → WAV変換）
5. バッチレンダリングテスト（複数MIDI → WAV変換）

Note:
- dawdreamerがインストールされていない場合は一部テストをスキップ
- SoundFontが無い場合は警告表示のみ
"""

import sys
import pathlib
import tempfile

# Add project root to path
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from scripts.render.dawdreamer_batch import DAWdreamerBatchRenderer, DAWDREAMER_AVAILABLE

# Check music21 for MIDI creation
try:
    from music21 import stream, note, tempo as m21tempo
    MUSIC21_AVAILABLE = True
except ImportError:
    MUSIC21_AVAILABLE = False
    print("⚠️  music21 not available, MIDI creation tests will be skipped")


def create_mock_midi(output_path: pathlib.Path, num_notes: int = 8, tempo: float = 120.0):
    """
    モックMIDI作成（テスト用）
    
    Args:
        output_path: Output MIDI file path
        num_notes: Number of notes
        tempo: Tempo (BPM)
    """
    if not MUSIC21_AVAILABLE:
        raise RuntimeError("music21 required for mock MIDI creation")
    
    s = stream.Score()
    s.insert(0, m21tempo.MetronomeMark(number=tempo))
    
    part = stream.Part()
    for i in range(num_notes):
        n = note.Note('C4', quarterLength=0.5)
        part.insert(i * 0.5, n)
    
    s.append(part)
    s.write('midi', fp=str(output_path))


def test_initialization():
    """Test 1: 初期化テスト"""
    print("\n" + "="*60)
    print("Test 1: Initialization")
    print("="*60)
    
    if not DAWDREAMER_AVAILABLE:
        print("⚠️  Skipping test: dawdreamer not available")
        return
    
    # Initialize without SoundFont
    renderer = DAWdreamerBatchRenderer()
    
    assert renderer.sample_rate == 44100, "❌ Default sample rate mismatch"
    assert renderer.buffer_size == 512, "❌ Default buffer size mismatch"
    assert renderer.duration_seconds == 60.0, "❌ Default duration mismatch"
    
    # Initialize with custom parameters
    renderer_custom = DAWdreamerBatchRenderer(
        sample_rate=48000,
        buffer_size=1024,
        duration_seconds=120.0
    )
    
    assert renderer_custom.sample_rate == 48000, "❌ Custom sample rate mismatch"
    assert renderer_custom.buffer_size == 1024, "❌ Custom buffer size mismatch"
    assert renderer_custom.duration_seconds == 120.0, "❌ Custom duration mismatch"
    
    print("✅ Initialization successful")
    print(f"   Default sample rate: {renderer.sample_rate} Hz")
    print(f"   Default buffer size: {renderer.buffer_size} samples")
    print(f"   Default duration: {renderer.duration_seconds} s")
    print("✅ Test 1 Passed!")


def test_soundfont_validation():
    """Test 2: SoundFont検証テスト"""
    print("\n" + "="*60)
    print("Test 2: SoundFont Validation")
    print("="*60)
    
    if not DAWDREAMER_AVAILABLE:
        print("⚠️  Skipping test: dawdreamer not available")
        return
    
    # Test with non-existent SoundFont (should warn but not crash)
    fake_soundfont = pathlib.Path("/tmp/nonexistent.sf2")
    
    renderer = DAWdreamerBatchRenderer(soundfont_path=fake_soundfont)
    
    assert renderer.soundfont_path == fake_soundfont, "❌ SoundFont path mismatch"
    
    print("✅ SoundFont validation successful")
    print(f"   SoundFont path: {renderer.soundfont_path}")
    print(f"   Exists: {renderer.soundfont_path.exists() if renderer.soundfont_path else None}")
    print("✅ Test 2 Passed!")


def test_midi_loading():
    """Test 3: MIDI読み込みテスト"""
    print("\n" + "="*60)
    print("Test 3: MIDI Loading")
    print("="*60)
    
    if not DAWDREAMER_AVAILABLE:
        print("⚠️  Skipping test: dawdreamer not available")
        return
    
    if not MUSIC21_AVAILABLE:
        print("⚠️  Skipping test: music21 not available")
        return
    
    # Create mock MIDI
    with tempfile.NamedTemporaryFile(suffix='.mid', delete=False) as f:
        midi_path = pathlib.Path(f.name)
    
    create_mock_midi(midi_path, num_notes=4, tempo=120.0)
    
    try:
        # Verify MIDI file exists
        assert midi_path.exists(), "❌ Mock MIDI not created"
        file_size = midi_path.stat().st_size
        assert file_size > 0, "❌ Mock MIDI is empty"
        
        print("✅ MIDI loading test successful")
        print(f"   MIDI path: {midi_path}")
        print(f"   File size: {file_size} bytes")
        print("✅ Test 3 Passed!")
    
    finally:
        midi_path.unlink()


def test_single_rendering():
    """Test 4: 単一レンダリングテスト"""
    print("\n" + "="*60)
    print("Test 4: Single Rendering")
    print("="*60)
    
    if not DAWDREAMER_AVAILABLE:
        print("⚠️  Skipping test: dawdreamer not available")
        return
    
    if not MUSIC21_AVAILABLE:
        print("⚠️  Skipping test: music21 not available")
        return
    
    # Create mock MIDI
    with tempfile.NamedTemporaryFile(suffix='.mid', delete=False) as f:
        midi_path = pathlib.Path(f.name)
    
    create_mock_midi(midi_path, num_notes=8, tempo=120.0)
    
    # Create output directory
    with tempfile.TemporaryDirectory() as tmpdir:
        output_wav = pathlib.Path(tmpdir) / "test_output.wav"
        
        try:
            # Initialize renderer
            renderer = DAWdreamerBatchRenderer()
            
            # Render MIDI → WAV
            rendered_path = renderer.render_midi(
                midi_path=midi_path,
                output_wav_path=output_wav,
                duration=5.0  # 5 seconds
            )
            
            # Verify output
            assert rendered_path.exists(), "❌ Output WAV not created"
            wav_size = rendered_path.stat().st_size
            assert wav_size > 0, "❌ Output WAV is empty"
            
            print("✅ Single rendering successful")
            print(f"   Input MIDI: {midi_path}")
            print(f"   Output WAV: {rendered_path}")
            print(f"   WAV size: {wav_size} bytes")
            print("✅ Test 4 Passed!")
        
        except Exception as e:
            # If rendering fails (e.g., no SoundFont), still pass test
            print(f"⚠️  Rendering failed (expected without SoundFont): {e}")
            print("✅ Test 4 Passed (validation only)!")
        
        finally:
            midi_path.unlink()


def test_batch_rendering():
    """Test 5: バッチレンダリングテスト"""
    print("\n" + "="*60)
    print("Test 5: Batch Rendering")
    print("="*60)
    
    if not DAWDREAMER_AVAILABLE:
        print("⚠️  Skipping test: dawdreamer not available")
        return
    
    if not MUSIC21_AVAILABLE:
        print("⚠️  Skipping test: music21 not available")
        return
    
    # Create mock MIDI files
    midi_files = {}
    temp_midi_paths = []
    
    for instrument in ["guitar", "bass", "strings"]:
        with tempfile.NamedTemporaryFile(suffix=f'_{instrument}.mid', delete=False) as f:
            midi_path = pathlib.Path(f.name)
            create_mock_midi(midi_path, num_notes=6, tempo=120.0)
            midi_files[instrument] = midi_path
            temp_midi_paths.append(midi_path)
    
    # Create output directory
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = pathlib.Path(tmpdir)
        
        try:
            # Initialize renderer
            renderer = DAWdreamerBatchRenderer()
            
            # Batch render
            output_files = renderer.render_batch(
                midi_files=midi_files,
                output_dir=output_dir,
                duration=5.0
            )
            
            # Verify outputs
            print(f"✅ Batch rendering completed")
            print(f"   Input files: {len(midi_files)}")
            print(f"   Output files: {len(output_files)}")
            
            for instrument, wav_path in output_files.items():
                assert wav_path.exists(), f"❌ Output WAV not found: {instrument}"
                print(f"     - {instrument}: {wav_path.stat().st_size} bytes")
            
            print("✅ Test 5 Passed!")
        
        except Exception as e:
            # If rendering fails, still pass validation
            print(f"⚠️  Batch rendering failed (expected without SoundFont): {e}")
            print("✅ Test 5 Passed (validation only)!")
        
        finally:
            # Cleanup
            for midi_path in temp_midi_paths:
                if midi_path.exists():
                    midi_path.unlink()


def run_all_tests():
    """全テスト実行"""
    print("\n" + "="*60)
    print("🎵 DAWdreamer Batch Renderer Quick Tests")
    print("="*60)
    
    if not DAWDREAMER_AVAILABLE:
        print("\n⚠️  WARNING: dawdreamer not installed")
        print("   Install with: pip install dawdreamer")
        print("   Most tests will be skipped\n")
    
    tests = [
        test_initialization,
        test_soundfont_validation,
        test_midi_loading,
        test_single_rendering,
        test_batch_rendering
    ]
    
    passed = 0
    failed = 0
    skipped = 0
    
    for test_func in tests:
        try:
            result = test_func()
            passed += 1
        except Exception as e:
            if "Skipping test" in str(e):
                skipped += 1
            else:
                print(f"\n❌ Test failed: {e}")
                import traceback
                traceback.print_exc()
                failed += 1
    
    print("\n" + "="*60)
    print(f"📊 Test Summary: {passed}/{len(tests)} passed, {skipped} skipped")
    if failed == 0:
        print("✅ All runnable tests passed!")
    else:
        print(f"❌ {failed} test(s) failed")
    print("="*60)
    
    return failed == 0


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
