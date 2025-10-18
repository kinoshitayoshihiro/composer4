#!/usr/bin/env python3
"""
Tests for Hi-Hat Open/Closed Exclusivity and Crash Choke Duration

Todo #7: ハイハット開閉整合性とクラッシュチョーク長制限のテスト
"""

import pytest
from scripts.quality_gate_drums import (
    check_hihat_exclusivity,
    check_crash_choke_duration,
    HIHAT_CLOSED_PITCH,
    HIHAT_OPEN_PITCH,
    HIHAT_PEDAL_PITCH,
)


class TestHihatExclusivity:
    """Hi-Hat Open/Closed 相互排他チェックのテスト"""
    
    def test_no_conflict_different_times(self):
        """異なるタイミングでのOpen/Closedは問題なし"""
        hits = [0.0, 1.0, 2.0, 3.0]
        pitches = [
            HIHAT_CLOSED_PITCH,  # 0.0: Closed
            HIHAT_OPEN_PITCH,    # 1.0: Open
            HIHAT_CLOSED_PITCH,  # 2.0: Closed
            HIHAT_OPEN_PITCH,    # 3.0: Open
        ]
        
        violations = check_hihat_exclusivity(hits, pitches)
        assert len(violations) == 0
    
    def test_conflict_same_time(self):
        """同一タイミング（±0.05内）でのOpen/Closedは違反"""
        hits = [0.0, 0.01, 1.0]  # 0.0と0.01は近接（許容誤差内）
        pitches = [
            HIHAT_OPEN_PITCH,    # 0.0: Open
            HIHAT_CLOSED_PITCH,  # 0.01: Closed (conflict!)
            HIHAT_OPEN_PITCH,    # 1.0: Open
        ]
        
        violations = check_hihat_exclusivity(hits, pitches)
        assert len(violations) == 1
        assert "Open/Closed conflict" in violations[0]
        assert "time 0.00" in violations[0] or "time 0.01" in violations[0]
    
    def test_pedal_allowed_with_open(self):
        """Pedal（44）はOpenと同時発音可能"""
        hits = [0.0, 0.01, 1.0]
        pitches = [
            HIHAT_OPEN_PITCH,   # 0.0: Open
            HIHAT_PEDAL_PITCH,  # 0.01: Pedal (OK!)
            HIHAT_CLOSED_PITCH, # 1.0: Closed
        ]
        
        violations = check_hihat_exclusivity(hits, pitches)
        assert len(violations) == 0
    
    def test_pedal_allowed_with_closed(self):
        """Pedal（44）はClosedと同時発音可能"""
        hits = [0.0, 0.01, 1.0]
        pitches = [
            HIHAT_CLOSED_PITCH, # 0.0: Closed
            HIHAT_PEDAL_PITCH,  # 0.01: Pedal (OK!)
            HIHAT_OPEN_PITCH,   # 1.0: Open
        ]
        
        violations = check_hihat_exclusivity(hits, pitches)
        assert len(violations) == 0
    
    def test_empty_lists(self):
        """空リストはエラーなし"""
        violations = check_hihat_exclusivity([], [])
        assert len(violations) == 0
    
    def test_mismatched_lengths(self):
        """hits と pitches の長さが不一致ならエラー"""
        hits = [0.0, 1.0]
        pitches = [HIHAT_CLOSED_PITCH]  # 長さ不一致
        
        violations = check_hihat_exclusivity(hits, pitches)
        assert len(violations) == 1
        assert "Mismatched" in violations[0]
    
    def test_multiple_conflicts(self):
        """複数の違反を検出"""
        hits = [0.0, 0.01, 1.0, 1.02, 2.0]
        pitches = [
            HIHAT_OPEN_PITCH,    # 0.0: Open
            HIHAT_CLOSED_PITCH,  # 0.01: Closed (conflict #1)
            HIHAT_CLOSED_PITCH,  # 1.0: Closed
            HIHAT_OPEN_PITCH,    # 1.02: Open (conflict #2)
            HIHAT_OPEN_PITCH,    # 2.0: Open
        ]
        
        violations = check_hihat_exclusivity(hits, pitches)
        assert len(violations) == 2
    
    def test_tolerance_boundary(self):
        """許容誤差境界テスト（0.05）"""
        # 0.05以内は同時と判定
        hits_close = [0.0, 0.04]
        pitches_close = [HIHAT_OPEN_PITCH, HIHAT_CLOSED_PITCH]
        violations_close = check_hihat_exclusivity(hits_close, pitches_close, tolerance=0.05)
        assert len(violations_close) == 1  # 違反
        
        # 0.05を超えれば別タイミング
        hits_far = [0.0, 0.06]
        pitches_far = [HIHAT_OPEN_PITCH, HIHAT_CLOSED_PITCH]
        violations_far = check_hihat_exclusivity(hits_far, pitches_far, tolerance=0.05)
        assert len(violations_far) == 0  # 問題なし


class TestCrashChokeDuration:
    """クラッシュチョーク長制限チェックのテスト"""
    
    def test_normal_short_choke(self):
        """通常の短いチョーク（200ms）は問題なし"""
        # 120 BPM: 1 quarter beat = 500ms
        # 0.4 quarter beats = 200ms
        hits = [0.0]
        durations = [0.4]  # 200ms @ 120 BPM
        
        violations = check_crash_choke_duration(
            hits, durations, max_duration_ms=500.0, tempo=120.0
        )
        assert len(violations) == 0
    
    def test_choke_too_long(self):
        """長すぎるチョーク（1000ms）は違反"""
        # 120 BPM: 1 quarter beat = 500ms
        # 2.0 quarter beats = 1000ms
        hits = [0.0]
        durations = [2.0]  # 1000ms @ 120 BPM
        
        violations = check_crash_choke_duration(
            hits, durations, max_duration_ms=500.0, tempo=120.0
        )
        assert len(violations) == 1
        assert "too long" in violations[0]
        assert "1000.0ms" in violations[0]
    
    def test_long_crash_ignored(self):
        """非常に長いクラッシュ（通常音）はチェック対象外"""
        # 120 BPM: 4.0 quarter beats = 2000ms
        # max_duration_ms * 2 = 1000ms を超えるのでチェック対象外
        hits = [0.0]
        durations = [4.0]  # 2000ms @ 120 BPM
        
        violations = check_crash_choke_duration(
            hits, durations, max_duration_ms=500.0, tempo=120.0
        )
        assert len(violations) == 0  # チェック対象外なので違反なし
    
    def test_tempo_dependency(self):
        """テンポによって判定が変わる"""
        hits = [0.0]
        durations = [1.0]  # 1 quarter beat
        
        # 120 BPM: 1 quarter = 500ms → ギリギリ許容
        violations_120 = check_crash_choke_duration(
            hits, durations, max_duration_ms=500.0, tempo=120.0
        )
        assert len(violations_120) == 0
        
        # 60 BPM: 1 quarter = 1000ms → 超過
        violations_60 = check_crash_choke_duration(
            hits, durations, max_duration_ms=500.0, tempo=60.0
        )
        assert len(violations_60) == 1
    
    def test_empty_lists(self):
        """空リストはエラーなし"""
        violations = check_crash_choke_duration([], [], max_duration_ms=500.0, tempo=120.0)
        assert len(violations) == 0
    
    def test_mismatched_lengths(self):
        """hits と durations の長さが不一致ならエラー"""
        hits = [0.0, 1.0]
        durations = [0.5]  # 長さ不一致
        
        violations = check_crash_choke_duration(
            hits, durations, max_duration_ms=500.0, tempo=120.0
        )
        assert len(violations) == 1
        assert "Mismatched" in violations[0]
    
    def test_multiple_violations(self):
        """複数の違反を検出"""
        hits = [0.0, 1.0, 2.0, 3.0]
        durations = [
            0.4,  # 200ms OK
            2.0,  # 1000ms NG
            0.6,  # 300ms OK
            1.8,  # 900ms NG
        ]
        
        violations = check_crash_choke_duration(
            hits, durations, max_duration_ms=500.0, tempo=120.0
        )
        assert len(violations) == 2  # 2つ違反
    
    def test_boundary_case(self):
        """境界値テスト（ちょうど500ms）"""
        # 120 BPM: 1.0 quarter beat = 500ms
        hits = [0.0]
        durations = [1.0]
        
        # ちょうど500msは許容（< でなく <=）
        violations = check_crash_choke_duration(
            hits, durations, max_duration_ms=500.0, tempo=120.0
        )
        assert len(violations) == 0
    
    def test_custom_max_duration(self):
        """カスタム最大長（300ms）"""
        hits = [0.0]
        durations = [0.8]  # 400ms @ 120 BPM
        
        violations = check_crash_choke_duration(
            hits, durations, max_duration_ms=300.0, tempo=120.0
        )
        assert len(violations) == 1
        assert "400.0ms" in violations[0]
        assert "300.0ms" in violations[0]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
