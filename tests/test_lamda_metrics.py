from math import isclose
from typing import Any, Dict, List, cast

from lamda_tools.metrics import (
    MetricConfig,
    MetricsAggregator,
    compute_loop_metrics,
)


def _make_sample_notes() -> List[List[object]]:
    # [type, start, duration, channel, pitch, velocity]
    return [
        ["note", 0, 60, 9, 36, 110],
        ["note", 90, 60, 9, 42, 30],
        ["note", 135, 60, 9, 38, 95],
        ["note", 180, 60, 9, 38, 120],
        ["note", 270, 60, 9, 46, 80],
        ["note", 360, 60, 9, 42, 70],
    ]


def test_compute_loop_metrics_basic():
    notes = _make_sample_notes()
    metrics = compute_loop_metrics(notes, config=MetricConfig())

    assert metrics.note_count == 6
    assert metrics.base_step is not None
    assert isclose(metrics.base_step, 90.0, rel_tol=1e-3)
    assert isclose(metrics.ghost_rate, 1 / 6, rel_tol=1e-3)
    assert isclose(metrics.accent_rate, 2 / 6, rel_tol=1e-3)
    assert metrics.syncopation_rate is not None
    assert isclose(metrics.syncopation_rate, 1 / 6, rel_tol=1e-3)
    assert metrics.hat_open_ratio is not None
    assert isclose(metrics.hat_open_ratio, 1 / 3, rel_tol=1e-3)
    assert metrics.hat_transition_rate is not None
    assert isclose(metrics.hat_transition_rate, 1.0, rel_tol=1e-3)
    assert metrics.instrument_distribution["kick"] == 1
    assert metrics.instrument_distribution["snare"] == 2
    assert metrics.instrument_distribution["open_hat"] == 1

    metrics_dict = cast(Dict[str, Any], metrics.to_dict())
    assert "instrument_distribution" in metrics_dict
    assert metrics_dict["instrument_distribution"]["kick"] == 1


def test_metrics_aggregator_summary():
    notes = _make_sample_notes()
    metrics = compute_loop_metrics(notes, config=MetricConfig())

    aggregator = MetricsAggregator()
    aggregator.add(metrics)

    summary = aggregator.summary()
    assert summary["count"] == 1
    averages = cast(Dict[str, float], summary["averages"])
    instrument_distribution = cast(Dict[str, int], summary["instrument_distribution"])
    assert isclose(
        float(averages["ghost_rate"]),
        metrics.ghost_rate,
        abs_tol=1e-4,
    )
    assert isclose(
        float(averages["accent_rate"]),
        metrics.accent_rate,
        abs_tol=1e-4,
    )
    assert instrument_distribution["snare"] == 2


def test_compute_loop_metrics_handles_empty():
    metrics = compute_loop_metrics([], config=MetricConfig())
    assert metrics.note_count == 0
    assert metrics.base_step is None
    assert metrics.ghost_rate == 0.0
    assert metrics.instrument_distribution == {}
