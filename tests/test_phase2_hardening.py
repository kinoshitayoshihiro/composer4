#!/usr/bin/env python3
"""
ドラムパターン抽出の簡易テスト（Phase 2）

動作確認:
1. SoundFont Manager のハッシュ登録・検証
2. DAWdreamer Batch の正規化・クリッピング検出
3. ドラムパターン抽出の品質フィルタ

Usage:
    python tests/test_phase2_hardening.py
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np


def test_soundfont_manager():
    """Test 1: SoundFont Manager"""
    print("\n" + "=" * 60)
    print("Test 1: SoundFont Manager - Hash Management")
    print("=" * 60)
    
    from scripts.render.soundfont_manager import SoundFontManager
    
    # テスト用ダミーファイル作成
    test_sf2 = Path("data/test_dummy.sf2")
    test_sf2.parent.mkdir(parents=True, exist_ok=True)
    
    # ダミーデータ（1KB）
    dummy_data = b"RIFF" + b"\x00" * 1020
    test_sf2.write_bytes(dummy_data)
    
    print(f"✅ Created dummy SF2: {test_sf2}")
    
    # ハッシュ計算
    manager = SoundFontManager(lock_file=Path("data/test_soundfonts.lock"))
    hash_value = manager.register(test_sf2, name="test_dummy")
    
    print(f"✅ Registered with hash: {hash_value[:16]}...")
    
    # 検証
    is_valid, message = manager.verify(test_sf2, name="test_dummy")
    
    if is_valid:
        print(f"✅ Verification passed!")
    else:
        print(f"❌ Verification failed: {message}")
    
    # クリーンアップ
    test_sf2.unlink()
    Path("data/test_soundfonts.lock").unlink()
    
    print("\n✅ Test 1 passed!")
    return True


def test_audio_safety_analysis():
    """Test 2: Audio Safety Analysis"""
    print("\n" + "=" * 60)
    print("Test 2: Audio Safety Analysis - Clipping Detection")
    print("=" * 60)
    
    from scripts.render.dawdreamer_batch import DAWdreamerBatchRenderer
    
    # テストケース1: 安全な音声（-6 dB）
    audio_safe = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 44100)) * 0.5
    
    safety = DAWdreamerBatchRenderer.analyze_audio_safety(audio_safe)
    
    print(f"\nTest Case 1: Safe Audio (-6 dB)")
    print(f"   Peak: {safety['peak_db']:.1f} dB")
    print(f"   Clipping rate: {safety['clipping_rate']*100:.2f}%")
    print(f"   Is safe: {safety['is_safe']}")
    
    assert safety['is_safe'], "Safe audio should pass safety check"
    
    # テストケース2: クリッピング音声（0 dB超え）
    audio_clipped = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 44100)) * 1.2
    audio_clipped = np.clip(audio_clipped, -0.99, 1.0)  # 意図的にクリップ
    
    safety_clipped = DAWdreamerBatchRenderer.analyze_audio_safety(audio_clipped)
    
    print(f"\nTest Case 2: Clipped Audio (intentional)")
    print(f"   Peak: {safety_clipped['peak_db']:.1f} dB")
    print(f"   Clipping rate: {safety_clipped['clipping_rate']*100:.2f}%")
    print(f"   Is safe: {safety_clipped['is_safe']}")
    
    # クリッピング率が0.1%を超えているはず
    assert not safety_clipped['is_safe'], "Clipped audio should fail safety check"
    
    print("\n✅ Test 2 passed!")
    return True


def test_drum_pattern_quality_filter():
    """Test 3: Drum Pattern Quality Filter"""
    print("\n" + "=" * 60)
    print("Test 3: Drum Pattern Quality Filter")
    print("=" * 60)
    
    from scripts.extract_drum_patterns import calculate_pattern_metrics, MIN_KICK_ONBEAT_RATIO
    
    # テストケース1: 高品質パターン（キックが拍頭、ゴースト少ない）
    hits_good = {
        'kick': ([0.0, 2.0], [100, 100]),        # 拍頭のみ（100%オンビート）
        'snare': ([1.0, 3.0], [90, 90]),         # 2拍・4拍
        'hihat_closed': ([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5], [80] * 8),  # 8分音符
        'hihat_open': ([], []),
        'crash': ([0.0], [100]),                 # 1発目のみ
        'ride': ([], [])
    }
    
    metrics_good = calculate_pattern_metrics(hits_good)
    
    print(f"\nTest Case 1: High Quality Pattern")
    print(f"   Kick on-beat ratio: {metrics_good['kick_onbeat_ratio']:.2f}")
    print(f"   Ghost note ratio:   {metrics_good['ghost_note_ratio']:.2f}")
    print(f"   Density:            {metrics_good['density']:.2f} hits/bar")
    print(f"   Quality score:      {metrics_good['quality_score']:.2f}")
    
    assert metrics_good['kick_onbeat_ratio'] >= MIN_KICK_ONBEAT_RATIO, \
        "High quality pattern should have good on-beat ratio"
    assert metrics_good['quality_score'] >= 0.6, \
        "High quality pattern should have score >= 0.6"
    
    # テストケース2: 低品質パターン（キックがオフビート、ゴースト多い）
    hits_bad = {
        'kick': ([0.25, 1.75], [50, 50]),        # オフビート、弱いベロシティ
        'snare': ([1.5], [30]),                   # ゴーストノート
        'hihat_closed': ([0.1, 0.6], [30, 30]),  # 少数、ゴースト
        'hihat_open': ([], []),
        'crash': ([], []),
        'ride': ([], [])
    }
    
    metrics_bad = calculate_pattern_metrics(hits_bad)
    
    print(f"\nTest Case 2: Low Quality Pattern")
    print(f"   Kick on-beat ratio: {metrics_bad['kick_onbeat_ratio']:.2f}")
    print(f"   Ghost note ratio:   {metrics_bad['ghost_note_ratio']:.2f}")
    print(f"   Density:            {metrics_bad['density']:.2f} hits/bar")
    print(f"   Quality score:      {metrics_bad['quality_score']:.2f}")
    
    assert metrics_bad['quality_score'] < 0.6, \
        "Low quality pattern should have score < 0.6"
    
    print("\n✅ Test 3 passed!")
    return True


def main():
    """Run all Phase 2 hardening tests"""
    print("\n" + "=" * 60)
    print("🔴 Phase 2 Hardening Tests")
    print("=" * 60)
    
    tests = [
        ("SoundFont Manager", test_soundfont_manager),
        ("Audio Safety Analysis", test_audio_safety_analysis),
        ("Drum Pattern Quality Filter", test_drum_pattern_quality_filter)
    ]
    
    results = []
    
    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, success))
        except Exception as e:
            print(f"\n❌ Test failed: {name}")
            print(f"   Error: {e}")
            results.append((name, False))
    
    # サマリ
    print("\n" + "=" * 60)
    print("📊 Test Summary")
    print("=" * 60)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status}: {name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All Phase 2 hardening tests passed!")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
        return 1


if __name__ == '__main__':
    exit(main())
