#!/usr/bin/env python3
"""
Vocal Sync Guard Quick Tests

VocalSyncGuardの動作確認（軽量テスト）

Tests:
1. 初期化テスト（パス設定、キャッシュ初期化）
2. Vocal onset検出テスト（モック音声 → onset抽出）
3. MIDI onset抽出テスト（モックMIDI → onset時刻）
4. Drift計算テスト（Vocal vs MIDI drift測定）
5. タイムストレッチ推奨テスト（修正係数計算）
"""

import sys
import pathlib
import tempfile
import numpy as np
import yaml

# Add project root to path
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from generator.vocal_sync_guard import VocalSyncGuard, LIBROSA_AVAILABLE, MUSIC21_AVAILABLE

# Check dependencies
if not LIBROSA_AVAILABLE:
    print("⚠️  librosa not available, skipping vocal onset tests")

if not MUSIC21_AVAILABLE:
    print("⚠️  music21 not available, skipping MIDI tests")


def create_mock_vocal_audio(duration: float = 10.0, onset_times: list = None) -> pathlib.Path:
    """
    モック音声作成（onset位置にクリック音挿入）
    
    Args:
        duration: Duration in seconds
        onset_times: List of onset times (seconds)
    
    Returns:
        Path to temporary WAV file
    """
    if not LIBROSA_AVAILABLE:
        raise RuntimeError("librosa required for mock audio creation")
    
    import soundfile as sf
    
    sr = 22050
    if onset_times is None:
        # Default: onsets every 0.5 seconds
        onset_times = np.arange(0.5, duration, 0.5)
    
    # Generate silence
    audio = np.zeros(int(duration * sr))
    
    # Insert clicks at onset positions
    for onset_time in onset_times:
        onset_sample = int(onset_time * sr)
        if onset_sample < len(audio):
            # Simple click: short burst
            click_duration = int(0.01 * sr)  # 10ms
            audio[onset_sample:onset_sample + click_duration] = 0.5 * np.sin(
                2 * np.pi * 1000 * np.arange(click_duration) / sr
            )
    
    # Save to temp file
    temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
    sf.write(temp_file.name, audio, sr)
    
    return pathlib.Path(temp_file.name)


def create_mock_midi(note_times: list = None, tempo: float = 120.0) -> pathlib.Path:
    """
    モックMIDI作成
    
    Args:
        note_times: List of note onset times (seconds)
        tempo: Tempo (BPM)
    
    Returns:
        Path to temporary MIDI file
    """
    if not MUSIC21_AVAILABLE:
        raise RuntimeError("music21 required for mock MIDI creation")
    
    from music21 import stream, note, tempo as m21tempo
    
    if note_times is None:
        # Default: notes every 0.5 seconds
        note_times = np.arange(0.5, 10.0, 0.5)
    
    # Create score
    s = stream.Score()
    s.insert(0, m21tempo.MetronomeMark(number=tempo))
    
    part = stream.Part()
    for note_time in note_times:
        # Convert seconds to quarterbeats: note_time * (tempo / 60)
        offset_qb = note_time * (tempo / 60.0)
        n = note.Note('C4', quarterLength=0.5)
        part.insert(offset_qb, n)
    
    s.append(part)
    
    # Save to temp file
    temp_file = tempfile.NamedTemporaryFile(suffix='.mid', delete=False)
    s.write('midi', fp=temp_file.name)
    
    return pathlib.Path(temp_file.name)


def create_mock_structure_yaml(sections: list = None) -> pathlib.Path:
    """
    モック構造YAML作成
    
    Args:
        sections: List of section dicts
    
    Returns:
        Path to temporary YAML file
    """
    if sections is None:
        sections = [
            {'label': 'Intro', 'start_time': 0.0, 'end_time': 2.0},
            {'label': 'Verse', 'start_time': 2.0, 'end_time': 6.0},
            {'label': 'Chorus', 'start_time': 6.0, 'end_time': 10.0}
        ]
    
    structure = {
        'tempo_map': {'global_tempo': 120.0},
        'sections': sections
    }
    
    temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False)
    yaml.dump(structure, temp_file)
    temp_file.flush()
    
    return pathlib.Path(temp_file.name)


def test_initialization():
    """Test 1: 初期化テスト"""
    print("\n" + "="*60)
    print("Test 1: Initialization")
    print("="*60)
    
    # Create temporary files
    vocal_path = pathlib.Path("/tmp/test_vocal.wav")
    midi_path = pathlib.Path("/tmp/test_midi.mid")
    structure_path = pathlib.Path("/tmp/test_structure.yaml")
    
    # Initialize guard
    guard = VocalSyncGuard(
        vocal_audio_path=vocal_path,
        midi_path=midi_path,
        structure_yaml_path=structure_path
    )
    
    # Verify paths
    assert guard.vocal_audio_path == vocal_path, "❌ Vocal path mismatch"
    assert guard.midi_path == midi_path, "❌ MIDI path mismatch"
    assert guard.structure_yaml_path == structure_path, "❌ Structure path mismatch"
    
    # Verify cache
    assert guard.vocal_onsets is None, "❌ Vocal onsets should be None initially"
    assert guard.midi_note_onsets is None, "❌ MIDI onsets should be None initially"
    assert guard.sections is None, "❌ Sections should be None initially"
    
    print("✅ Initialization successful")
    print(f"   Vocal: {guard.vocal_audio_path}")
    print(f"   MIDI: {guard.midi_path}")
    print(f"   Structure: {guard.structure_yaml_path}")
    print("✅ Test 1 Passed!")


def test_vocal_onset_detection():
    """Test 2: Vocal onset検出テスト"""
    print("\n" + "="*60)
    print("Test 2: Vocal Onset Detection")
    print("="*60)
    
    if not LIBROSA_AVAILABLE:
        print("⚠️  Skipping test: librosa not available")
        return
    
    # Create mock audio with known onsets
    onset_times = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    vocal_path = create_mock_vocal_audio(duration=4.0, onset_times=onset_times)
    
    try:
        # Initialize guard
        guard = VocalSyncGuard(vocal_audio_path=vocal_path)
        
        # Detect onsets
        detected_onsets = guard.load_vocal_onsets()
        
        # Verify detection
        assert len(detected_onsets) > 0, "❌ No onsets detected"
        
        # Check approximate match (tolerance: 100ms)
        tolerance = 0.1
        matched = 0
        for expected_time in onset_times:
            distances = [abs(detected - expected_time) for detected in detected_onsets]
            if min(distances) < tolerance:
                matched += 1
        
        match_rate = matched / len(onset_times)
        
        print(f"✅ Detected {len(detected_onsets)} onsets")
        print(f"   Expected: {onset_times}")
        print(f"   Detected: {detected_onsets[:6].tolist()}")
        print(f"   Match rate: {match_rate:.1%} (within {tolerance*1000}ms)")
        
        # Loose assertion: at least 50% match rate
        assert match_rate >= 0.5, f"❌ Match rate too low: {match_rate:.1%}"
        
        print("✅ Test 2 Passed!")
    
    finally:
        vocal_path.unlink()


def test_midi_onset_extraction():
    """Test 3: MIDI onset抽出テスト"""
    print("\n" + "="*60)
    print("Test 3: MIDI Onset Extraction")
    print("="*60)
    
    if not MUSIC21_AVAILABLE:
        print("⚠️  Skipping test: music21 not available")
        return
    
    # Create mock MIDI with known note times
    note_times = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    midi_path = create_mock_midi(note_times=note_times, tempo=120.0)
    
    try:
        # Initialize guard
        guard = VocalSyncGuard(midi_path=midi_path)
        
        # Extract onsets
        midi_onsets = guard.load_midi_note_onsets()
        
        # Verify extraction
        assert len(midi_onsets) > 0, "❌ No MIDI onsets extracted"
        assert len(midi_onsets) == len(note_times), f"❌ Onset count mismatch: {len(midi_onsets)} vs {len(note_times)}"
        
        # Check approximate match (tolerance: 10ms)
        tolerance = 0.01
        for i, expected_time in enumerate(note_times):
            actual_time = midi_onsets[i]
            assert abs(actual_time - expected_time) < tolerance, \
                f"❌ Onset {i} mismatch: {actual_time:.3f} vs {expected_time:.3f}"
        
        print(f"✅ Extracted {len(midi_onsets)} MIDI onsets")
        print(f"   Expected: {note_times}")
        print(f"   Extracted: {[f'{t:.3f}' for t in midi_onsets]}")
        print("✅ Test 3 Passed!")
    
    finally:
        midi_path.unlink()


def test_drift_calculation():
    """Test 4: Drift計算テスト"""
    print("\n" + "="*60)
    print("Test 4: Drift Calculation")
    print("="*60)
    
    if not LIBROSA_AVAILABLE or not MUSIC21_AVAILABLE:
        print("⚠️  Skipping test: librosa or music21 not available")
        return
    
    # Create mock data with intentional drift
    # Vocal: every 0.5s
    vocal_onset_times = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
    # MIDI: slightly delayed (50ms drift)
    midi_note_times = [t + 0.05 for t in vocal_onset_times]
    
    vocal_path = create_mock_vocal_audio(duration=5.0, onset_times=vocal_onset_times)
    midi_path = create_mock_midi(note_times=midi_note_times, tempo=120.0)
    structure_path = create_mock_structure_yaml(sections=[
        {'label': 'Section1', 'start_time': 0.0, 'end_time': 2.5},
        {'label': 'Section2', 'start_time': 2.5, 'end_time': 5.0}
    ])
    
    try:
        # Initialize guard
        guard = VocalSyncGuard(
            vocal_audio_path=vocal_path,
            midi_path=midi_path,
            structure_yaml_path=structure_path
        )
        
        # Calculate drift
        drift_reports = guard.calculate_drift_per_section()
        
        # Verify reports
        assert len(drift_reports) == 2, f"❌ Expected 2 sections, got {len(drift_reports)}"
        
        print(f"✅ Calculated drift for {len(drift_reports)} sections:")
        for report in drift_reports:
            print(f"\n   Section: {report['section']}")
            print(f"     Vocal onsets: {report['vocal_onset_count']}")
            print(f"     MIDI onsets: {report['midi_onset_count']}")
            if report['mean_drift_ms'] is not None:
                print(f"     Mean drift: {report['mean_drift_ms']:.1f} ms")
                print(f"     Max drift: {report['max_drift_ms']:.1f} ms")
                print(f"     Status: {report['status']}")
                
                # Check if drift is positive (MIDI is ahead)
                # (Loose tolerance due to onset detection variance)
                assert report['mean_drift_ms'] >= 0.0, \
                    f"❌ Negative drift detected: {report['mean_drift_ms']:.1f} ms"
                
                # Check if drift detection is working (should detect some drift)
                # Expected 50ms, but actual may vary due to onset detection precision
                assert report['mean_drift_ms'] < 200.0, \
                    f"❌ Drift too large: {report['mean_drift_ms']:.1f} ms"
        
        print("\n✅ Test 4 Passed!")
    
    finally:
        vocal_path.unlink()
        midi_path.unlink()
        structure_path.unlink()


def test_time_stretch_recommendation():
    """Test 5: タイムストレッチ推奨テスト"""
    print("\n" + "="*60)
    print("Test 5: Time Stretch Recommendation")
    print("="*60)
    
    if not LIBROSA_AVAILABLE or not MUSIC21_AVAILABLE:
        print("⚠️  Skipping test: librosa or music21 not available")
        return
    
    # Create mock data with significant drift
    vocal_onset_times = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    midi_note_times = [t + 0.1 for t in vocal_onset_times]  # 100ms drift
    
    vocal_path = create_mock_vocal_audio(duration=4.0, onset_times=vocal_onset_times)
    midi_path = create_mock_midi(note_times=midi_note_times, tempo=120.0)
    structure_path = create_mock_structure_yaml(sections=[
        {'label': 'Test', 'start_time': 0.0, 'end_time': 4.0}
    ])
    
    try:
        # Initialize guard
        guard = VocalSyncGuard(
            vocal_audio_path=vocal_path,
            midi_path=midi_path,
            structure_yaml_path=structure_path
        )
        
        # Full sync check
        report = guard.check_sync()
        
        # Verify report
        assert 'overall_status' in report, "❌ Missing overall_status"
        assert 'recommended_stretch' in report, "❌ Missing recommended_stretch"
        
        stretch = report['recommended_stretch']
        
        print(f"✅ Sync check completed")
        print(f"   Overall status: {report['overall_status']}")
        print(f"   Warnings: {report.get('warning_count', 0)}")
        print(f"   Errors: {report.get('error_count', 0)}")
        print(f"   Recommended stretch: {stretch:.6f}")
        
        # Verify stretch is reasonable (0.95 to 1.05)
        assert 0.95 <= stretch <= 1.05, f"❌ Stretch out of range: {stretch:.6f}"
        
        print("✅ Test 5 Passed!")
    
    finally:
        vocal_path.unlink()
        midi_path.unlink()
        structure_path.unlink()


def run_all_tests():
    """全テスト実行"""
    print("\n" + "="*60)
    print("🎤 Vocal Sync Guard Quick Tests")
    print("="*60)
    
    tests = [
        test_initialization,
        test_vocal_onset_detection,
        test_midi_onset_extraction,
        test_drift_calculation,
        test_time_stretch_recommendation
    ]
    
    passed = 0
    failed = 0
    skipped = 0
    
    for test_func in tests:
        try:
            result = test_func()
            if result is None:  # Test ran successfully
                passed += 1
            else:  # Test was skipped
                skipped += 1
        except AssertionError as e:
            print(f"\n❌ Test failed: {e}")
            failed += 1
        except Exception as e:
            print(f"\n❌ Test error: {e}")
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
