#!/usr/bin/env python3
"""Test ab_summarize_v2.py stratified analysis."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from scripts.ab_summarize_v2 import _bpm_bin, _conf_bin, _strata_key, _summary


def test_bpm_binning():
    """Test BPM binning logic."""
    assert _bpm_bin(80) == "≤95"
    assert _bpm_bin(120) == "≤130"
    assert _bpm_bin(150) == ">130"
    assert _bpm_bin(None) == "unknown"


def test_conf_binning():
    """Test confidence binning logic."""
    assert _conf_bin(0.3) == "<0.5"
    assert _conf_bin(0.6) == "0.5–0.7"
    assert _conf_bin(0.75) == "0.7–0.85"
    assert _conf_bin(0.9) == "≥0.85"
    assert _conf_bin(None) == "unknown"


def test_strata_key_generation():
    """Test stratification key generation."""
    row = {
        "bpm": 110,
        "audio": {"min_confidence": 0.72},
        "_retry_state": {"last_preset": "vel_chain"},
    }
    strata_names = ["bpm_bin", "audio.min_confidence_bin", "preset_applied"]
    key = _strata_key(row, strata_names)
    assert key == ("≤130", "0.7–0.85", "vel_chain")


def test_summary_stats():
    """Test summary statistics computation."""
    rows = [
        {"score": 45.0, "axes_raw": {"velocity": 0.25, "structure": 0.30}},
        {"score": 55.0, "axes_raw": {"velocity": 0.35, "structure": 0.40}},
        {"score": 65.0, "axes_raw": {"velocity": 0.45, "structure": 0.50}},
    ]
    stats = _summary(rows)
    assert stats["n"] == 3
    assert stats["pass_rate"] == 2 / 3
    assert stats["p50"] == 55.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
