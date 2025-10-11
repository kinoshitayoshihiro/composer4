"""Test guard_retry_accept.py audio boost logic."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from guard_retry_accept import should_accept


def test_audio_boost_acceptance():
    """Test that audio.text_audio_cos improvement >= +0.02 helps borderline cases."""
    prev = {
        "score_total": 45.0,
        "axes_raw": {"velocity": 0.40, "timing": 0.70},
        "audio": {"text_audio_cos": 0.60},
    }
    post = {
        "score_total": 46.5,  # +1.5 delta (meets threshold)
        "axes_raw": {"velocity": 0.39, "timing": 0.70},  # velocity down (fails axes)
        "audio": {"text_audio_cos": 0.63},  # +0.03 delta (meets boost)
    }
    control = {"min_delta": {"score_total": 1.0, "axes_raw": {"velocity": 0.02}}}

    ok, meta = should_accept(prev, post, control)

    assert ok is True, "Should accept via audio boost"
    assert meta["ok_total"] is True
    assert meta["ok_axes"] is False  # velocity degraded
    assert meta["audio_boost"] is True
    assert abs(meta["delta_audio_cos"] - 0.03) < 1e-6


def test_no_audio_boost_when_small():
    """Test that audio delta < +0.02 does not trigger boost."""
    prev = {
        "score_total": 45.0,
        "axes_raw": {"velocity": 0.40},
        "audio": {"text_audio_cos": 0.60},
    }
    post = {
        "score_total": 46.5,
        "axes_raw": {"velocity": 0.39},  # fails axes
        "audio": {"text_audio_cos": 0.61},  # +0.01 delta (too small)
    }
    control = {"min_delta": {"score_total": 1.0, "axes_raw": {"velocity": 0.02}}}

    ok, meta = should_accept(prev, post, control)

    assert ok is False, "Should reject when audio boost too small"
    assert meta["ok_total"] is True
    assert meta["ok_axes"] is False
    assert meta["audio_boost"] is False
    assert abs(meta["delta_audio_cos"] - 0.01) < 1e-6


def test_normal_acceptance_path():
    """Test that normal score+axes path still works."""
    prev = {
        "score_total": 45.0,
        "axes_raw": {"velocity": 0.40},
        "audio": {"text_audio_cos": 0.60},
    }
    post = {
        "score_total": 47.0,
        "axes_raw": {"velocity": 0.43},
        "audio": {"text_audio_cos": 0.60},  # no change
    }
    control = {"min_delta": {"score_total": 1.0, "axes_raw": {"velocity": 0.02}}}

    ok, meta = should_accept(prev, post, control)

    assert ok is True, "Should accept via normal path"
    assert meta["ok_total"] is True
    assert meta["ok_axes"] is True
    assert meta["audio_boost"] is False  # not needed
    assert meta["delta_audio_cos"] == 0.0
