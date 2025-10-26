#!/usr/bin/env python3
"""
MoisesDB Integration テストスクリプト

Usage:
    python scripts/test_moisesdb_integration.py
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

from scripts.moisesdb_integration import (
    MoisesDBIntegrator,
    SegmentMerger,
    HarmonicStemSelector,
)


def create_test_segment(output_path: Path, duration: float = 1.0, sr: int = 22050):
    """テスト用WAVセグメント生成"""
    # 440Hz サイン波
    t = np.linspace(0, duration, int(sr * duration))
    audio = 0.5 * np.sin(2 * np.pi * 440 * t)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output_path), audio, sr)
    
    return output_path


def test_segment_merger():
    """セグメント統合テスト"""
    print("\n" + "="*70)
    print("Test: Segment Merger")
    print("="*70)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # テストセグメント作成
        segments = []
        for i in range(3):
            seg_path = tmpdir / f"segment_{i:04d}_guitar.wav"
            create_test_segment(seg_path, duration=0.5)
            segments.append(seg_path)
        
        # 統合
        merger = SegmentMerger(sr=22050)
        output_path = tmpdir / "merged.wav"
        
        result = merger.merge_segments(segments, output_path)
        
        # 検証
        assert output_path.exists(), "Merged file not created"
        assert result['num_segments'] == 3, f"Expected 3 segments, got {result['num_segments']}"
        assert result['duration'] > 1.4, f"Expected ~1.5s, got {result['duration']}"
        
        print(f"✅ Merged {result['num_segments']} segments")
        print(f"   Duration: {result['duration']:.2f}s")
        print(f"   Sample rate: {result['sample_rate']} Hz")


def test_harmonic_stem_selector():
    """ステム選択テスト"""
    print("\n" + "="*70)
    print("Test: Harmonic Stem Selector")
    print("="*70)
    
    selector = HarmonicStemSelector()
    
    # Test 1: piano優先
    stems = ['vocals', 'drums', 'guitar', 'piano']
    selected = selector.select_best_stem(stems)
    assert selected == 'piano', f"Expected 'piano', got '{selected}'"
    print(f"✅ Test 1: Selected '{selected}' from {stems}")
    
    # Test 2: vocals/drums除外
    stems = ['vocals', 'drums', 'bass']
    selected = selector.select_best_stem(stems)
    assert selected == 'bass', f"Expected 'bass', got '{selected}'"
    print(f"✅ Test 2: Selected '{selected}' from {stems}")
    
    # Test 3: 優先度順
    stems = ['other', 'strings', 'keys']
    selected = selector.select_best_stem(stems)
    assert selected == 'keys', f"Expected 'keys', got '{selected}'"
    print(f"✅ Test 3: Selected '{selected}' from {stems}")
    
    # Test 4: 除外ステムのみ
    stems = ['vocals', 'drums', 'percussion']
    selected = selector.select_best_stem(stems)
    assert selected is None, f"Expected None, got '{selected}'"
    print(f"✅ Test 4: No harmonic stem (returned None)")
    
    # Test 5: 重み付き選択
    stems = ['guitar', 'piano', 'bass', 'drums']
    harmonic_stems, weights = selector.select_harmonic_stems_with_weights(stems)
    
    assert 'guitar' in harmonic_stems, "guitar should be selected"
    assert 'piano' in harmonic_stems, "piano should be selected"
    assert 'drums' not in harmonic_stems, "drums should be excluded"
    
    # 重み検証
    assert weights['piano'] > weights['bass'], "piano weight should be higher than bass"
    assert weights['guitar'] > 0, "guitar should have positive weight"
    
    # 正規化検証（合計≈1.0）
    total_weight = sum(weights.values())
    assert 0.99 < total_weight < 1.01, f"Weights should sum to 1.0, got {total_weight}"
    
    print(f"✅ Test 5: Weighted selection")
    print(f"   Harmonic stems: {harmonic_stems}")
    print(f"   Weights: {weights}")
    print(f"   Total weight: {total_weight:.3f}")


def test_database_integration():
    """データベース統合テスト"""
    print("\n" + "="*70)
    print("Test: Database Integration")
    print("="*70)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # テストデータセット作成
        song_dir = tmpdir / "input" / "song_001"
        song_dir.mkdir(parents=True)
        
        # セグメント作成（guitar/drums/vocals）
        for stem in ['guitar', 'drums', 'vocals']:
            for i in range(2):
                seg_path = song_dir / f"segment_{i:04d}_{stem}.wav"
                create_test_segment(seg_path, duration=0.3)
        
        # Integrator実行
        db_path = tmpdir / "test.db"
        midi_dir = tmpdir / "midi"
        
        integrator = MoisesDBIntegrator(
            db_path=db_path,
            midi_output_dir=midi_dir,
            sr=22050
        )
        
        result = integrator.process_song_directory(song_dir, verbose=True)
        
        # 検証
        assert result['status'] == 'success', f"Expected success, got {result['status']}"
        assert result['selected_stem'] == 'guitar', f"Expected 'guitar', got {result['selected_stem']}"
        print(f"✅ Song processed: {result['song_id']}")
        print(f"   Selected stem: {result['selected_stem']}")
        print(f"   Duration: {result['duration']:.2f}s")
        
        # クエリテスト
        print("\n--- Query Tests ---")
        
        # hash検索
        hash_result = integrator.query_by_hash(result['hash_id'])
        assert hash_result is not None, "Hash query failed"
        print(f"✅ Query by hash: {hash_result['song_id']}")
        
        # stem検索
        stem_results = integrator.query_by_stem('guitar', limit=5)
        assert len(stem_results) > 0, "Stem query failed"
        print(f"✅ Query by stem: {len(stem_results)} results")
        
        # 統計
        stats = integrator.get_statistics()
        print(f"✅ Statistics:")
        print(f"   Total songs: {stats['total_songs']}")
        print(f"   Stem counts: {stats['stem_counts']}")


def test_lamda_compatibility():
    """LAMDA互換性テスト"""
    print("\n" + "="*70)
    print("Test: LAMDA Compatibility")
    print("="*70)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        db_path = tmpdir / "test.db"
        midi_dir = tmpdir / "midi"
        
        integrator = MoisesDBIntegrator(
            db_path=db_path,
            midi_output_dir=midi_dir,
            sr=22050
        )
        
        # データベーススキーマ検証
        import sqlite3
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # progressions テーブル存在確認
        cursor.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name='progressions'
        """)
        assert cursor.fetchone() is not None, "progressions table not found"
        print("✅ progressions table exists")
        
        # moisesdb_meta テーブル存在確認
        cursor.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name='moisesdb_meta'
        """)
        assert cursor.fetchone() is not None, "moisesdb_meta table not found"
        print("✅ moisesdb_meta table exists")
        
        # インデックス確認
        cursor.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='index' AND name='idx_hash_id'
        """)
        assert cursor.fetchone() is not None, "idx_hash_id index not found"
        print("✅ idx_hash_id index exists")
        
        conn.close()


def main():
    print("\n" + "="*70)
    print("MoisesDB Integration Test Suite")
    print("="*70)
    
    try:
        test_segment_merger()
        test_harmonic_stem_selector()
        test_database_integration()
        test_lamda_compatibility()
        
        print("\n" + "="*70)
        print("✅ All Tests Passed!")
        print("="*70)
    
    except AssertionError as e:
        print(f"\n❌ Test Failed: {e}")
        raise
    
    except Exception as e:
        print(f"\n❌ Unexpected Error: {e}")
        raise


if __name__ == '__main__':
    main()
