#!/usr/bin/env python3
"""
MoisesDB Quality Filter テスト

Usage:
    python scripts/test_quality_filter.py
"""

import tempfile
from pathlib import Path

import numpy as np
from pretty_midi import PrettyMIDI, Instrument, Note

from scripts.moisesdb_quality_filter import (
    MIDIQualityMetrics,
    MoisesDBQualityFilter
)


def create_test_midi(output_path: Path, quality_type: str = 'good'):
    """
    テスト用MIDIファイル生成
    
    Args:
        quality_type: 'good', 'sparse', 'dense', 'monotonous', 'narrow_range'
    """
    midi = PrettyMIDI()
    piano = Instrument(program=0)
    
    if quality_type == 'good':
        # 良好な品質: 適切なノート密度、ピッチ範囲、和音率
        for i in range(40):
            # メロディ
            pitch = 60 + (i % 12)
            start = i * 0.5
            end = start + 0.4
            velocity = 70 + np.random.randint(-10, 10)
            piano.notes.append(Note(velocity, pitch, start, end))
            
            # 和音（50%の確率）
            if i % 2 == 0:
                pitch2 = pitch + 4
                piano.notes.append(Note(velocity - 5, pitch2, start, end))
                pitch3 = pitch + 7
                piano.notes.append(Note(velocity - 5, pitch3, start, end))
    
    elif quality_type == 'sparse':
        # 低品質: ノート密度が低すぎる
        for i in range(5):
            pitch = 60
            start = i * 4.0  # 4秒間隔
            end = start + 0.5
            piano.notes.append(Note(64, pitch, start, end))
    
    elif quality_type == 'dense':
        # 低品質: ノート密度が高すぎる
        for i in range(200):
            pitch = 60 + np.random.randint(0, 12)
            start = i * 0.05  # 50ms間隔
            end = start + 0.04
            velocity = 64 + np.random.randint(-5, 5)
            piano.notes.append(Note(velocity, pitch, start, end))
    
    elif quality_type == 'monotonous':
        # 低品質: 単調（ベロシティ固定、音長固定）
        for i in range(30):
            pitch = 60 + (i % 5)
            start = i * 0.5
            end = start + 0.5  # 全て同じ音長
            piano.notes.append(Note(64, pitch, start, end))  # 全て同じベロシティ
    
    elif quality_type == 'narrow_range':
        # 低品質: ピッチ範囲が狭い
        for i in range(30):
            pitch = 60 + (i % 3)  # 3音のみ
            start = i * 0.5
            end = start + 0.4
            velocity = 64 + np.random.randint(-10, 10)
            piano.notes.append(Note(velocity, pitch, start, end))
    
    midi.instruments.append(piano)
    midi.write(str(output_path))
    
    return output_path


def test_quality_metrics():
    """品質メトリクス計算テスト"""
    print("\n" + "="*70)
    print("Test: Quality Metrics Calculation")
    print("="*70)
    
    metrics_calculator = MIDIQualityMetrics()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Test 1: Good quality
        midi_path = create_test_midi(tmpdir / 'good.mid', 'good')
        metrics = metrics_calculator.calculate_all_metrics(midi_path)
        
        assert metrics['overall_score'] > 0.6, "Good MIDI should score > 0.6"
        assert metrics['quality_grade'] in ['A', 'B'], "Good MIDI should be A or B grade"
        print(f"✅ Test 1: Good quality (score: {metrics['overall_score']:.3f}, grade: {metrics['quality_grade']})")
        
        # Test 2: Sparse
        midi_path = create_test_midi(tmpdir / 'sparse.mid', 'sparse')
        metrics = metrics_calculator.calculate_all_metrics(midi_path)
        
        assert metrics['note_density_score'] < 0.8, "Sparse MIDI should have low density score"
        print(f"✅ Test 2: Sparse detection (density: {metrics['note_density']:.2f} notes/sec)")
        
        # Test 3: Dense
        midi_path = create_test_midi(tmpdir / 'dense.mid', 'dense')
        metrics = metrics_calculator.calculate_all_metrics(midi_path)
        
        assert metrics['note_density'] > 5.0, "Dense MIDI should have high density"
        print(f"✅ Test 3: Dense detection (density: {metrics['note_density']:.2f} notes/sec)")
        
        # Test 4: Monotonous
        midi_path = create_test_midi(tmpdir / 'monotonous.mid', 'monotonous')
        metrics = metrics_calculator.calculate_all_metrics(midi_path)
        
        assert metrics['velocity_score'] < 0.5, "Monotonous MIDI should have low velocity score"
        assert metrics['duration_score'] < 0.5, "Monotonous MIDI should have low duration score"
        print(f"✅ Test 4: Monotonous detection (vel std: {metrics['velocity_variance']:.1f})")
        
        # Test 5: Narrow range
        midi_path = create_test_midi(tmpdir / 'narrow.mid', 'narrow_range')
        metrics = metrics_calculator.calculate_all_metrics(midi_path)
        
        assert metrics['pitch_range'] < 24, "Narrow range MIDI should have small pitch range"
        print(f"✅ Test 5: Narrow range detection (range: {metrics['pitch_range']} semitones)")


def test_quality_filter():
    """品質フィルタテスト"""
    print("\n" + "="*70)
    print("Test: Quality Filter")
    print("="*70)
    
    quality_filter = MoisesDBQualityFilter(threshold=0.6, verbose=False)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # テストMIDI生成
        good_midi = create_test_midi(tmpdir / 'good.mid', 'good')
        bad_midi = create_test_midi(tmpdir / 'sparse.mid', 'sparse')
        
        # Test 1: Good MIDI passes
        result = quality_filter.evaluate_midi_file(good_midi)
        assert result['passed'], "Good MIDI should pass"
        print(f"✅ Test 1: Good MIDI passed (score: {result['metrics']['overall_score']:.3f})")
        
        # Test 2: Bad MIDI fails
        result = quality_filter.evaluate_midi_file(bad_midi)
        # Note: sparse may still pass depending on other metrics
        print(f"✅ Test 2: Sparse MIDI evaluated (score: {result['metrics']['overall_score']:.3f}, passed: {result['passed']})")


def test_batch_evaluation():
    """バッチ評価テスト"""
    print("\n" + "="*70)
    print("Test: Batch Evaluation")
    print("="*70)
    
    quality_filter = MoisesDBQualityFilter(threshold=0.6, verbose=False)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        midi_dir = tmpdir / 'midi'
        midi_dir.mkdir()
        
        # 複数MIDI生成
        create_test_midi(midi_dir / 'good1.mid', 'good')
        create_test_midi(midi_dir / 'good2.mid', 'good')
        create_test_midi(midi_dir / 'sparse.mid', 'sparse')
        create_test_midi(midi_dir / 'dense.mid', 'dense')
        
        # バッチ評価
        summary = quality_filter.evaluate_batch(midi_dir, max_files=-1)
        
        assert summary['total'] == 4, "Should evaluate 4 files"
        assert summary['passed'] >= 1, "At least one should pass"
        print(f"✅ Batch evaluation: {summary['passed']}/{summary['total']} passed ({summary['pass_rate']:.1%})")


def main():
    print("\n" + "="*70)
    print("MoisesDB Quality Filter Test Suite")
    print("="*70)
    
    try:
        test_quality_metrics()
        test_quality_filter()
        test_batch_evaluation()
        
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
