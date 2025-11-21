"""Regression test for EmotionAI -> DurationHumanizeAI integration."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

DEMO_PLAN = Path("sandbox/humanize_demo/piano_plan_duration_ai.json")


@pytest.mark.skipif(not DEMO_PLAN.exists(), reason="demo plan json missing")
def test_demo_plan_has_emotion_tracking() -> None:
    data = json.loads(DEMO_PLAN.read_text(encoding="utf-8"))
    metadata = data.get("metadata") or {}
    tracking = metadata.get("emotion_tracking") or {}
    per_bar = tracking.get("per_bar") or {}
    assert per_bar, "emotion_tracking.per_bar should not be empty"
    for bar_idx, snapshot in per_bar.items():
        assert "energy" in snapshot, f"bar {bar_idx} missing energy"
        assert "section" in snapshot, f"bar {bar_idx} missing section label"

    events = data.get("events") or []
    assert events, "demo plan must contain events"

    emotion_payloads = [
        (idx, (event.get("humanize") or {}).get("emotion"))
        for idx, event in enumerate(events)
        if isinstance(event, dict)
    ]
    payloads = [payload for _, payload in emotion_payloads if payload]
    assert payloads, "no events carried humanize.emotion payloads"
    for payload in payloads:
        assert "bar_idx" in payload, "emotion payload missing bar index"
        assert "energy" in payload, "emotion payload missing energy"
        assert "tension" in payload, "emotion payload missing tension"
