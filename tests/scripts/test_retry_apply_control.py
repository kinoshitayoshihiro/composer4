import json
from pathlib import Path
from typing import Dict, List

import pytest
import yaml

from scripts import retry_apply


def _write_jsonl(path: Path, rows: List[Dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def _single_row(lines: List[str]) -> Dict[str, object]:
    assert len(lines) == 1
    return json.loads(lines[0])


@pytest.fixture()
def control_preset(tmp_path: Path) -> Path:
    config = {
        "presets": [
            {
                "id": "velocity_chain_audio_v1",
                "axis": "velocity",
                "when": {"axes_raw.velocity": "< 0.40"},
                "control": {
                    "cooldown_runs": 1,
                    "max_attempts": 2,
                    "min_delta": 0.05,
                    "cooldown_key": "velocity_chain",
                },
                "apply": [{"type": "velocity_boost", "boost_lo": 1.05}],
            }
        ]
    }
    path = tmp_path / "retry_presets.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


def test_retry_control_cooldown_and_limits(tmp_path: Path, control_preset: Path) -> None:
    metrics_path = tmp_path / "metrics.jsonl"
    output_path = tmp_path / "out.jsonl"

    base_row: Dict[str, object] = {
        "loop_id": "loop-1",
        "score": 40.0,
        "axes_raw": {"velocity": 0.20},
    }

    # First invocation should apply the preset.
    _write_jsonl(metrics_path, [base_row])
    lines, applied_total, _ = retry_apply.process(
        control_preset,
        metrics_path,
        output_path,
    )
    assert applied_total == 1
    first_row = _single_row(lines)
    assert first_row.get("_retry_ops")
    state = first_row.get("_retry_state")
    assert isinstance(state, dict)
    attempts = state.get("attempts", {})
    assert attempts.get("velocity_chain_audio_v1") == 1
    cooldowns = state.get("cooldowns", {})
    assert cooldowns.get("velocity_chain") == 1

    # Second invocation should be blocked by cooldown and leave attempts unchanged.
    second_row = dict(first_row)
    axes_raw = dict(second_row.get("axes_raw", {}))
    axes_raw["velocity"] = 0.23
    second_row["axes_raw"] = axes_raw
    _write_jsonl(metrics_path, [second_row])
    lines, applied_total, _ = retry_apply.process(
        control_preset,
        metrics_path,
        output_path,
    )
    assert applied_total == 0
    blocked_row = _single_row(lines)
    state = blocked_row.get("_retry_state")
    assert isinstance(state, dict)
    assert state.get("attempts", {}).get("velocity_chain_audio_v1") == 1
    assert state.get("cooldowns", {}).get("velocity_chain") == 0

    # Third invocation with sufficient delta should apply second attempt.
    third_row = dict(blocked_row)
    axes_raw = dict(third_row.get("axes_raw", {}))
    axes_raw["velocity"] = 0.28
    third_row["axes_raw"] = axes_raw
    _write_jsonl(metrics_path, [third_row])
    lines, applied_total, _ = retry_apply.process(
        control_preset,
        metrics_path,
        output_path,
    )
    assert applied_total == 1
    applied_row = _single_row(lines)
    state = applied_row.get("_retry_state")
    assert isinstance(state, dict)
    assert state.get("attempts", {}).get("velocity_chain_audio_v1") == 2

    # Fourth invocation should be blocked by cooldown (cooldown_runs=1).
    fourth_row = dict(applied_row)
    axes_raw = dict(fourth_row.get("axes_raw", {}))
    axes_raw["velocity"] = 0.34
    fourth_row["axes_raw"] = axes_raw
    _write_jsonl(metrics_path, [fourth_row])
    lines, applied_total, _ = retry_apply.process(
        control_preset,
        metrics_path,
        output_path,
    )
    assert applied_total == 0
    final_row = _single_row(lines)
    state = final_row.get("_retry_state")
    assert isinstance(state, dict)
    assert state.get("attempts", {}).get("velocity_chain_audio_v1") == 2
    assert state.get("cooldowns", {}).get("velocity_chain") == 0
    # Fifth invocation should now hit max_attempts after cooldown clears.
    fifth_row = dict(final_row)
    axes_raw = dict(fifth_row.get("axes_raw", {}))
    axes_raw["velocity"] = 0.36
    fifth_row["axes_raw"] = axes_raw
    _write_jsonl(metrics_path, [fifth_row])
    lines, applied_total, _ = retry_apply.process(
        control_preset,
        metrics_path,
        output_path,
    )
    assert applied_total == 0
    terminal_row = _single_row(lines)
    state = terminal_row.get("_retry_state")
    assert isinstance(state, dict)
    assert state.get("attempts", {}).get("velocity_chain_audio_v1") == 2
    control_meta = terminal_row.get("_retry_control")
    assert isinstance(control_meta, dict)
    blocked = control_meta.get("blocked", [])
    assert blocked, "max_attempts should be recorded as a blocked reason"
    reasons = {entry.get("reason") for entry in blocked}
    assert "max_attempts" in reasons
