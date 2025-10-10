import importlib.util
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

_MODULE_DIR = Path(__file__).resolve().parents[1] / "scripts"
_MODULE_PATH = _MODULE_DIR / "lamda_stage2_extractor.py"
_SPEC = importlib.util.spec_from_file_location(
    "lamda_stage2_extractor",
    _MODULE_PATH,
)
assert _SPEC is not None and _SPEC.loader is not None
sys.path.append(str(_MODULE_DIR))
_STAGE2 = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _STAGE2
_SPEC.loader.exec_module(_STAGE2)  # type: ignore[arg-type]

load_retry_presets = getattr(_STAGE2, "_load_retry_presets")
select_retry_rule_fn = getattr(_STAGE2, "_select_retry_rule")


def build_rules(raw: List[Dict[str, Any]]) -> Dict[str, List[Any]]:
    return load_retry_presets(raw)  # type: ignore[no-any-return]


def select_rule(
    rules: Dict[str, List[Any]],
    velocity_value: float,
) -> Optional[Any]:
    axes_raw = {"velocity": velocity_value}
    score_breakdown = {"velocity": velocity_value}
    return select_retry_rule_fn(  # type: ignore[no-any-return]
        reason_key="velocity",
        axes_raw=axes_raw,
        score_total=45.0,
        score_breakdown=score_breakdown,
        rules=rules,
    )


def test_retry_rule_sequence_priority():
    raw: List[Dict[str, Any]] = [
        {
            "reason": "velocity_chain",
            "when": {"axes_raw.velocity": "< 0.25"},
            "action": {"preset": "vel_chain"},
        },
        {
            "reason": "velocity",
            "when": {"axes_raw.velocity": "< 0.35"},
            "action": {"preset": "vel_smooth"},
        },
    ]
    rules = build_rules(raw)
    candidate = select_rule(rules, 0.2)
    assert candidate is not None
    assert candidate.name == "velocity_chain"

    fallback = select_rule(rules, 0.3)
    assert fallback is not None
    assert fallback.name == "velocity"


@pytest.mark.parametrize("value", [0.36, 0.45])
def test_retry_rule_none_when_conditions_fail(value: float):
    raw: List[Dict[str, Any]] = [
        {
            "reason": "velocity_chain",
            "when": {"axes_raw.velocity": "< 0.25"},
            "action": {"preset": "vel_chain"},
        }
    ]
    rules = build_rules(raw)
    result = select_rule(rules, value)
    assert result is None


def test_retry_rule_audio_context_condition():
    raw: List[Dict[str, Any]] = [
        {
            "reason": "velocity_audio_low",
            "when": {
                "axes_raw.velocity": "< 0.4",
                "audio.text_audio_cos": "< 0.35",
            },
            "action": {"preset": "vel_chain"},
        }
    ]
    rules = build_rules(raw)
    axes_raw = {"velocity": 0.32}
    score_breakdown = {"velocity": 0.32}
    candidate = select_retry_rule_fn(  # type: ignore[no-any-return]
        reason_key="velocity",
        axes_raw=axes_raw,
        score_total=42.0,
        score_breakdown=score_breakdown,
        rules=rules,
        audio_context={"text_audio_cos": 0.3},
    )
    assert candidate is not None
    assert candidate.name == "velocity_audio_low"

    rejected = select_retry_rule_fn(  # type: ignore[no-any-return]
        reason_key="velocity",
        axes_raw=axes_raw,
        score_total=42.0,
        score_breakdown=score_breakdown,
        rules=rules,
        audio_context={"text_audio_cos": 0.6},
    )
    assert rejected is None
