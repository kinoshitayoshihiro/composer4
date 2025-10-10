#!/usr/bin/env python3
"""Test guard_retry_accept.py logic."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from scripts.guard_retry_accept import should_accept


def test_guard_accept_dict_format():
    """Test min_delta dict format with score_total and axes_raw."""
    prev = {
        "loop_id": "test1",
        "score": 50.0,
        "axes_raw": {"velocity": 0.30, "structure": 0.40},
    }
    post = {
        "loop_id": "test1",
        "score": 52.5,
        "axes_raw": {"velocity": 0.36, "structure": 0.45},
    }
    control = {
        "min_delta": {
            "score_total": 2.0,
            "axes_raw": {"velocity": 0.05, "structure": 0.04},
        }
    }

    ok, meta = should_accept(prev, post, control)
    assert ok is True
    assert meta["ok_total"] is True
    assert meta["ok_axes"] is True
    assert meta["delta_total"] == pytest.approx(2.5, abs=1e-6)
    assert meta["deltas_axes"]["velocity"] == pytest.approx(0.06, abs=1e-6)
    assert meta["deltas_axes"]["structure"] == pytest.approx(0.05, abs=1e-6)


def test_guard_reject_insufficient_delta():
    """Test rejection when delta is insufficient."""
    prev = {
        "loop_id": "test2",
        "score": 50.0,
        "axes_raw": {"velocity": 0.30},
    }
    post = {
        "loop_id": "test2",
        "score": 50.5,
        "axes_raw": {"velocity": 0.32},
    }
    control = {
        "min_delta": {
            "score_total": 1.0,
            "axes_raw": {"velocity": 0.05},
        }
    }

    ok, meta = should_accept(prev, post, control)
    assert ok is False
    assert meta["ok_total"] is False
    assert meta["ok_axes"] is False


def test_guard_legacy_float_format():
    """Test backward compat with old float format."""
    prev = {"loop_id": "test3", "score": 45.0}
    post = {"loop_id": "test3", "score": 47.5}
    control = {"min_delta": 2.0}

    ok, meta = should_accept(prev, post, control)
    assert ok is True
    assert meta["delta_total"] == 2.5


def test_guard_no_criteria():
    """Test accept when no min_delta criteria."""
    prev = {"loop_id": "test4", "score": 45.0}
    post = {"loop_id": "test4", "score": 46.0}
    control = {}

    ok, meta = should_accept(prev, post, control)
    assert ok is True
    assert meta["reason"] == "no_min_delta_criteria"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
