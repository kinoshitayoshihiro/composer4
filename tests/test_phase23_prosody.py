#!/usr/bin/env python3
"""
tests/test_phase23_prosody.py - Phase 23 Prosody制御の統合テスト
"""
import sys
from pathlib import Path
import json

# プロジェクトルートをパスに追加
sys.path.insert(0, str(Path(__file__).parent.parent))

from generator.prosody_controller import ProsodyController, Anchor


def test_anchor_loading():
    """アンカー読み込みテスト"""
    anchors_data = {
        "unit": "sec",
        "anchors": [
            {
                "time": 1.0,
                "token": "test",
                "class": ["sibilant", "stress"],
                "section": "verse",
                "time_ql": 4.0,
                "window_ms": {"pre": 30.0, "post": 20.0}
            },
            {
                "time": 2.0,
                "token": "test2",
                "class": ["plosive"],
                "section": "verse",
                "time_ql": 8.0,
                "window_ms": {"pre": 10.0, "post": 60.0}
            }
        ]
    }
    
    controller = ProsodyController(anchors_data=anchors_data)
    
    assert len(controller.anchors) == 2
    assert controller.anchors[0].time == 1.0
    assert "sibilant" in controller.anchors[0].classes
    assert "stress" in controller.anchors[0].classes
    
    print("✅ test_anchor_loading: PASS")


def test_window_detection():
    """窓範囲検出テスト"""
    anchors_data = {
        "unit": "sec",
        "anchors": [
            {
                "time": 5.0,
                "token": "test",
                "class": ["sibilant"],
                "section": None,
                "time_ql": 0.0,
                "window_ms": {"pre": 100.0, "post": 100.0}  # ±100ms
            }
        ]
    }
    
    controller = ProsodyController(anchors_data=anchors_data, config={"merge_threshold_ms": 0})
    
    # 窓範囲内の時刻
    assert len(controller.get_anchors_for_time(5.0)) == 1
    assert len(controller.get_anchors_for_time(4.95)) == 1
    assert len(controller.get_anchors_for_time(5.05)) == 1
    
    # 窓範囲外
    assert len(controller.get_anchors_for_time(4.89)) == 0
    assert len(controller.get_anchors_for_time(5.11)) == 0
    
    print("✅ test_window_detection: PASS")


def test_prosody_application():
    """Prosody適用テスト"""
    anchors_data = {
        "unit": "sec",
        "anchors": [
            {
                "time": 1.0,
                "token": "sibilant_test",
                "class": ["sibilant"],
                "section": None,
                "time_ql": 0.0,
                "window_ms": {"pre": 50.0, "post": 50.0}
            },
            {
                "time": 2.0,
                "token": "stress_test",
                "class": ["stress"],
                "section": None,
                "time_ql": 0.0,
                "window_ms": {"pre": 50.0, "post": 50.0}
            },
            {
                "time": 3.0,
                "token": "plosive_test",
                "class": ["plosive"],
                "section": None,
                "time_ql": 0.0,
                "window_ms": {"pre": 50.0, "post": 50.0}
            }
        ]
    }
    
    controller = ProsodyController(anchors_data=anchors_data, config={"merge_threshold_ms": 0})
    
    # テストノート
    notes = [
        {"time": 1.0, "pitch": 60, "vel": 80, "dur": 0.5},  # sibilant窓内
        {"time": 2.0, "pitch": 60, "vel": 80, "dur": 0.5},  # stress窓内
        {"time": 3.0, "pitch": 60, "vel": 80, "dur": 0.5},  # plosive窓内
        {"time": 4.0, "pitch": 60, "vel": 80, "dur": 0.5},  # 窓外
    ]
    
    controller.apply_prosody(notes, role="guitar", tempo=120.0)
    
    # Sibilant: Velocity減少
    assert notes[0]["vel"] < 80, f"Sibilant should reduce velocity: {notes[0]['vel']}"
    
    # Stress: Velocity増加
    assert notes[1]["vel"] > 80, f"Stress should increase velocity: {notes[1]['vel']}"
    
    # Plosive: Duration減少
    assert notes[2]["dur"] < 0.5, f"Plosive should reduce duration: {notes[2]['dur']}"
    
    # 窓外: 変化なし
    assert notes[3]["vel"] == 80, f"Outside window should not change velocity: {notes[3]['vel']}"
    assert notes[3]["dur"] == 0.5, f"Outside window should not change duration: {notes[3]['dur']}"
    
    print("✅ test_prosody_application: PASS")


def test_overlap_merging():
    """窓重なりマージテスト"""
    anchors_data = {
        "unit": "sec",
        "anchors": [
            {
                "time": 1.0,
                "token": "test1",
                "class": ["sibilant"],
                "section": None,
                "time_ql": 0.0,
                "window_ms": {"pre": 30.0, "post": 30.0}
            },
            {
                "time": 1.04,  # 40ms後（近接）
                "token": "test2",
                "class": ["stress"],
                "section": None,
                "time_ql": 0.0,
                "window_ms": {"pre": 30.0, "post": 30.0}
            }
        ]
    }
    
    # マージ有効（閾値50ms）
    controller = ProsodyController(anchors_data=anchors_data, config={"merge_threshold_ms": 50})
    assert len(controller.anchors) == 1, f"Should merge overlapping anchors: {len(controller.anchors)}"
    assert "sibilant" in controller.anchors[0].classes
    assert "stress" in controller.anchors[0].classes
    
    # マージ無効
    controller2 = ProsodyController(anchors_data=anchors_data, config={"merge_threshold_ms": 0})
    assert len(controller2.anchors) == 2, f"Should not merge when disabled: {len(controller2.anchors)}"
    
    print("✅ test_overlap_merging: PASS")


def test_statistics():
    """統計情報テスト"""
    anchors_data = {
        "unit": "sec",
        "anchors": [
            {
                "time": 1.0,
                "token": "test1",
                "class": ["sibilant", "stress"],
                "section": "verse",
                "time_ql": 0.0,
                "window_ms": {"pre": 30.0, "post": 40.0}
            },
            {
                "time": 2.0,
                "token": "test2",
                "class": ["plosive"],
                "section": "chorus",
                "time_ql": 0.0,
                "window_ms": {"pre": 10.0, "post": 60.0}
            }
        ]
    }
    
    controller = ProsodyController(anchors_data=anchors_data, config={"merge_threshold_ms": 0})
    stats = controller.get_statistics()
    
    assert stats["total_anchors"] == 2
    assert stats["class_distribution"]["sibilant"] == 1
    assert stats["class_distribution"]["stress"] == 1
    assert stats["class_distribution"]["plosive"] == 1
    assert "verse" in stats["sections"]
    assert "chorus" in stats["sections"]
    
    print("✅ test_statistics: PASS")


if __name__ == "__main__":
    print("=" * 60)
    print("Phase 23: Prosody Control - Unit Tests")
    print("=" * 60)
    
    test_anchor_loading()
    test_window_detection()
    test_prosody_application()
    test_overlap_merging()
    test_statistics()
    
    print("=" * 60)
    print("✅ All tests passed!")
    print("=" * 60)
