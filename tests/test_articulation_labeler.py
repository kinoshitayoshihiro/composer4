import math
from pathlib import Path
from typing import Any, Dict, Optional

from scripts.lamda_stage2_extractor import (
    ArticulationLabeler,
    ArticulationObservation,
    ArticulationResult,
    ArticulationThresholds,
)


def _make_thresholds(*, mode: str = "fixed") -> ArticulationThresholds:
    fixed: Dict[str, Any] = {
        "drums": {
            "ghost_rate": {"high": 0.2, "min_support": 4},
            "flam_rate": {"high": 0.03, "min_support": 8},
        },
        "strings": {
            "detache_ratio": {"high": 0.5, "min_support": 8},
            "pizzicato_ratio": {"high": 0.1, "min_support": 8},
        },
    }
    auto: Dict[str, Any] = {
        "bins": {"tempo": [0, 90, 110, 130, 999]},
        "min_support": {"drums": 8, "strings": 8},
        "quantiles": {"hysteresis_drop_iqr": 0.1},
    }
    weights: Dict[str, float] = {
        "ghost": 0.15,
        "flam": 0.1,
        "detache": 0.15,
        "pizzicato": 0.1,
    }
    return ArticulationThresholds(
        mode=mode,
        fixed=fixed,
        auto=auto,
        weights=weights,
        path=Path("dummy"),
    )


def _make_observation(
    ghost_rate: float,
    snare_count: int = 16,
    tempo_bpm: Optional[float] = 120.0,
    flam_rate: float = 0.0,
    detache_ratio: Optional[float] = None,
    pizzicato_ratio: Optional[float] = None,
    violin_count: int = 0,
    string_track: Optional[bool] = None,
    drum_track: Optional[bool] = None,
    pizzicato_labeled: bool = False,
) -> ArticulationObservation:
    metrics: Dict[str, Optional[float]] = {
        "articulation.snare_ghost_rate": ghost_rate,
        "articulation.snare_flam_rate": flam_rate,
        "articulation.detache_ratio": detache_ratio,
        "articulation.pizzicato_ratio": pizzicato_ratio,
    }
    string_flag = string_track if string_track is not None else violin_count > 0
    drum_flag = drum_track if drum_track is not None else snare_count > 0
    support: Dict[str, Any] = {
        "snare_count": snare_count,
        "violin_count": violin_count,
        "snare_notes_per_sec": None,
        "loop_duration_seconds": None,
        "tempo_bpm": tempo_bpm,
        "string_track": string_flag,
        "drum_track": drum_flag,
        "pizzicato_labeled": pizzicato_labeled,
    }
    metrics_full: Dict[str, Any] = {}
    return ArticulationObservation(
        loop_id="test",
        metrics=metrics,
        support=support,
        tempo_bpm=tempo_bpm,
        metrics_full=metrics_full,
    )


def test_articulation_labeler_trigger_ghost() -> None:
    labeler = ArticulationLabeler(_make_thresholds())
    observation = _make_observation(ghost_rate=0.3, snare_count=12)
    result = labeler.evaluate(observation)

    assert isinstance(result, ArticulationResult)
    assert "ghost" in result.labels
    assert result.presence["ghost"] == 1.0
    # Weighted sum should equal the configured weight for ghost
    assert math.isclose(result.score, 0.15, rel_tol=1e-6)
    assert math.isclose(result.axis_value, 0.3, rel_tol=1e-6)


def test_articulation_labeler_insufficient_support() -> None:
    labeler = ArticulationLabeler(_make_thresholds())
    observation = _make_observation(ghost_rate=0.5, snare_count=2)
    result = labeler.evaluate(observation)

    assert result.labels == []
    assert result.presence.get("ghost", 0.0) == 0.0
    assert result.score == 0.0
    assert result.axis_value == 0.0


def test_articulation_labeler_auto_threshold_generation() -> None:
    labeler = ArticulationLabeler(_make_thresholds(mode="auto"))
    for rate in [0.1, 0.2, 0.3]:
        result = labeler.evaluate(_make_observation(ghost_rate=rate, snare_count=16))
        assert result.presence["ghost"] in {0.0, 1.0}
    final = labeler.evaluate(_make_observation(ghost_rate=0.4, snare_count=32))

    ghost_details = final.thresholds["ghost"]
    assert ghost_details["source"].startswith("auto")
    auto_info = ghost_details["auto"]
    assert auto_info["count"] == 4
    assert auto_info["bin"] in {"110-130", "all"}
    assert math.isclose(ghost_details["threshold"], 0.3625, rel_tol=1e-3)
    assert final.presence["ghost"] == 1.0


def test_articulation_labeler_requires_string_track_for_detache() -> None:
    labeler = ArticulationLabeler(_make_thresholds())
    observation = _make_observation(
        ghost_rate=0.0,
        snare_count=0,
        detache_ratio=0.8,
        violin_count=32,
        string_track=False,
    )
    result = labeler.evaluate(observation)

    assert result.presence.get("detache", 0.0) == 0.0
    detache_details = result.thresholds["detache"]
    assert detache_details["reason"] == "not_string_track"
