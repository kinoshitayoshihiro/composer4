from typing import Dict, Any

import pytest

from scripts.validate_audio_adaptive_config import validate_stage2_config


@pytest.fixture()
def base_payload() -> Dict[str, Any]:
    return {
        "score": {
            "audio_adaptive_weights": {
                "enabled": True,
                "min_confidence": 0.2,
                "missing_policy": "noop",
                "cooldown_loops": 3,
                "cooldown_by_rule": {"rule_a": 1},
                "max_total_delta": 1.5,
                "normalize": {
                    "enabled": True,
                    "target_sum": 8.0,
                },
                "caps": {
                    "min_scale": 0.9,
                    "max_scale": 1.2,
                    "axes": {"timing": {"min_scale": 0.9, "max_scale": 1.1}},
                },
            }
        }
    }


def test_validate_stage2_config_success(base_payload: Dict[str, Any]) -> None:
    assert validate_stage2_config(base_payload) == []


def test_validate_stage2_config_errors(base_payload: Dict[str, Any]) -> None:
    payload = base_payload
    audio_cfg = payload["score"]["audio_adaptive_weights"]
    audio_cfg["enabled"] = "true"
    audio_cfg["min_confidence"] = 1.5
    audio_cfg["missing_policy"] = "invalid"
    audio_cfg["cooldown_loops"] = -1
    audio_cfg["cooldown_by_rule"] = {"rule_a": -2}
    audio_cfg["max_total_delta"] = -0.1
    audio_cfg["normalize"]["target_sum"] = 0.0
    audio_cfg["caps"]["axes"]["timing"]["min_scale"] = 1.3
    errors = validate_stage2_config(payload)
    assert errors
    messages = "\n".join(errors)
    assert "enabled" in messages
    assert "min_confidence" in messages
    assert "missing_policy" in messages
    assert "cooldown_loops" in messages
    assert "cooldown_by_rule" in messages
    assert "max_total_delta" in messages
    assert "target_sum" in messages
    assert "caps.axes.timing" in messages
