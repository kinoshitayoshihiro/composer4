from __future__ import annotations

import math
import sys
import importlib.util
from pathlib import Path
from types import SimpleNamespace

_MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "lamda_stage2_extractor.py"
_SPEC = importlib.util.spec_from_file_location(
    "lamda_stage2_extractor",
    _MODULE_PATH,
)
assert _SPEC and _SPEC.loader
sys.path.append(str(_MODULE_PATH.parent))
_STAGE2 = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _STAGE2
_SPEC.loader.exec_module(_STAGE2)


def test_swing_alignment_score_prefers_straight_and_swing():
    straight = _STAGE2.swing_alignment_score(1.0)
    swing = _STAGE2.swing_alignment_score(1.5)
    mid = _STAGE2.swing_alignment_score(0.7)
    none = _STAGE2.swing_alignment_score(None)

    assert straight is not None and math.isclose(straight, 1.0, abs_tol=1e-2)
    assert swing is not None and math.isclose(swing, 1.0, abs_tol=1e-2)
    assert mid is not None and mid < 0.6
    assert none is None


def test_fingerprint_similarity_matches_reference():
    reference = {key: value for key, value in _STAGE2.GROOVE_REFERENCE_VECTOR.items()}
    similarity = _STAGE2.fingerprint_similarity(reference)
    assert similarity is not None
    assert math.isclose(similarity, 1.0, abs_tol=1e-6)


def test_score_axes_balances_groove_and_cohesion():
    metrics = SimpleNamespace(
        swing_confidence=0.9,
        microtiming_std=10.0,
        fill_density=0.28,
        microtiming_rms=8.0,
        ghost_rate=0.18,
        accent_rate=0.22,
        velocity_range=80.0,
        unique_velocity_steps=7.0,
        swing_ratio=1.45,
        syncopation_rate=0.33,
        rhythm_fingerprint={
            "eighth": 0.38,
            "sixteenth": 0.32,
            "triplet": 0.18,
            "quarter": 0.12,
        },
        drum_collision_rate=0.08,
        role_separation=0.72,
        hat_transition_rate=0.52,
        repeat_rate=0.5,
        variation_factor=0.45,
        breakpoint_count=2.0,
    )
    axes = _STAGE2.score_axes(metrics)
    assert axes["groove_harmony"] > 0.7
    assert axes["drum_cohesion"] > 0.7
    assert 0.0 <= axes["timing"] <= 1.0
    assert 0.0 <= axes["velocity"] <= 1.0
    assert 0.0 <= axes["structure"] <= 1.0


def test_score_axes_penalises_collisions():
    base_metrics = SimpleNamespace(
        swing_confidence=0.9,
        microtiming_std=10.0,
        fill_density=0.28,
        microtiming_rms=8.0,
        ghost_rate=0.18,
        accent_rate=0.22,
        velocity_range=80.0,
        unique_velocity_steps=7.0,
        swing_ratio=1.45,
        syncopation_rate=0.33,
        rhythm_fingerprint={
            "eighth": 0.38,
            "sixteenth": 0.32,
            "triplet": 0.18,
            "quarter": 0.12,
        },
        drum_collision_rate=0.08,
        role_separation=0.72,
        hat_transition_rate=0.52,
        repeat_rate=0.5,
        variation_factor=0.45,
        breakpoint_count=2.0,
    )
    high_collision = SimpleNamespace(**base_metrics.__dict__)
    high_collision.drum_collision_rate = 0.4
    axes_base = _STAGE2.score_axes(base_metrics)
    axes_collision = _STAGE2.score_axes(high_collision)
    assert axes_collision["drum_cohesion"] < axes_base["drum_cohesion"]
