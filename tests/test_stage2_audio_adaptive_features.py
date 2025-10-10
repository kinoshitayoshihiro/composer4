"""Targeted tests for audio adaptive weight features."""

from typing import Any, Dict

import pytest

from scripts.lamda_stage2_extractor import (
    AudioAdaptiveFusion,
    AudioAdaptiveFusionSource,
    AudioAdaptiveRule,
    AudioAdaptiveState,
    AudioAdaptiveWeights,
    _apply_audio_adaptive_weights,
    _summarise_adaptive_details,
    _summarise_adaptive_flags,
    _temperature_scale_value,
)


def test_temperature_scaling_applies_to_fused_pivot() -> None:
    base = {"timing": 1.0}
    rule = AudioAdaptiveRule(
        operator=">=",
        threshold=0.3,
        multipliers={"timing": 1.0},
        name="temp_rule",
    )
    fusion = AudioAdaptiveFusion(
        sources=(
            AudioAdaptiveFusionSource(
                path=("metrics", "raw_score"),
                weight=1.0,
            ),
        ),
        clamp_min=0.0,
        clamp_max=1.0,
        temperature=2.0,
    )
    adaptive = AudioAdaptiveWeights(
        pivot_path=("metrics", "raw_score"),
        rules=(rule,),
        fusion=fusion,
    )
    context = {
        "metrics": {
            "raw_score": 0.9,
        }
    }

    _, applied, pivot, _, details = _apply_audio_adaptive_weights(
        base,
        adaptive,
        context,
    )

    assert applied is rule
    expected = _temperature_scale_value(0.9, 2.0, 0.0, 1.0)
    assert expected is not None
    assert pivot is not None
    assert pytest.approx(expected) == pivot
    temperature_flag = details["flags"].get("temperature")
    assert temperature_flag is not None
    assert pytest.approx(2.0) == temperature_flag


def test_pivot_exponential_moving_average_smoothing() -> None:
    base = {"timing": 1.0}
    rule = AudioAdaptiveRule(
        operator=">=",
        threshold=0.1,
        multipliers={"timing": 1.1},
        name="ema",
    )
    adaptive = AudioAdaptiveWeights(
        pivot_path=("metrics", "confidence"),
        rules=(rule,),
        pivot_ema_alpha=0.5,
    )
    state = AudioAdaptiveState()

    first = _apply_audio_adaptive_weights(
        base,
        adaptive,
        {"metrics": {"confidence": 0.8}},
        state=state,
    )
    _, applied_first, pivot_first, state, details_first = first
    assert applied_first is rule
    assert pivot_first is not None
    assert pytest.approx(0.8) == pivot_first
    ema_flag = details_first["flags"].get("pivot_ema")
    assert ema_flag is not None
    assert pytest.approx(0.8) == ema_flag

    second = _apply_audio_adaptive_weights(
        base,
        adaptive,
        {"metrics": {"confidence": 0.2}},
        state=state,
    )
    _, applied_second, pivot_second, _, details_second = second

    expected_smoothed = 0.5 * 0.2 + 0.5 * 0.8
    assert applied_second is rule
    assert pivot_second is not None
    assert pytest.approx(expected_smoothed) == pivot_second
    ema_flag_next = details_second["flags"].get("pivot_ema")
    assert ema_flag_next is not None
    assert pytest.approx(expected_smoothed) == ema_flag_next


def test_higher_priority_rule_is_selected_first() -> None:
    base = {"timing": 1.0}
    low_priority = AudioAdaptiveRule(
        operator=">=",
        threshold=0.4,
        multipliers={"timing": 1.1},
        name="low",
        priority=1,
    )
    high_priority = AudioAdaptiveRule(
        operator=">=",
        threshold=0.4,
        multipliers={"timing": 1.2},
        name="high",
        priority=5,
    )
    adaptive = AudioAdaptiveWeights(
        pivot_path=("metrics", "score"),
        rules=(low_priority, high_priority),
    )

    _, applied, _, _, details = _apply_audio_adaptive_weights(
        base,
        adaptive,
        {"metrics": {"score": 0.8}},
    )

    assert applied is high_priority
    matched = details["flags"].get("matched_rules")
    assert matched == ["high", "low"]


def test_adaptive_detail_serialisation_respects_log_levels() -> None:
    rule = AudioAdaptiveRule(
        operator=">=",
        threshold=0.5,
        multipliers={"timing": 1.1},
        name="serialise",
    )
    details: Dict[str, Any] = {
        "rule_name": "serialise",
        "pivot": 0.6,
        "total_delta": 0.2,
        "total_delta_limited": False,
        "total_delta_ratio": None,
        "hysteresis_applied": False,
        "cooldown_active": False,
        "cooldown_remaining": 0,
        "missing_policy_applied": None,
        "below_min_confidence": False,
        "rule_cooldown_blocked": None,
        "flags": {
            "initial_rule_name": "serialise",
            "cooldown_before": 0,
            "extra": "debug",
        },
        "weights": {
            "before": {"timing": 1.0},
            "after": {"timing": 1.1},
        },
    }

    summary = _summarise_adaptive_details(details, rule, "summary")
    assert "weights_before" not in summary
    assert pytest.approx(1.1) == summary["weights_after"]["timing"]
    assert "flags" not in summary

    debug = _summarise_adaptive_details(details, rule, "debug")
    assert pytest.approx(1.0) == debug["weights_before"]["timing"]
    assert debug["flags"]["extra"] == "debug"

    flags_summary = _summarise_adaptive_flags(details["flags"], "summary")
    assert flags_summary["initial_rule_name"] == "serialise"
    assert "extra" not in flags_summary
