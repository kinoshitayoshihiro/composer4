#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Validate Stage2 audio_adaptive_weights configuration files."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

_VALID_MISSING_POLICIES = {"noop", "zero", "last"}


def _as_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return None


def _validate_caps(caps_cfg: Dict[str, Any], errors: List[str]) -> None:
    min_scale = _as_float(caps_cfg.get("min_scale"))
    max_scale = _as_float(caps_cfg.get("max_scale"))
    if None not in (min_scale, max_scale) and min_scale > max_scale:
        errors.append(
            "caps.min_scale must be <= caps.max_scale",
        )

    axes_cfg = caps_cfg.get("axes")
    if isinstance(axes_cfg, dict):
        for axis, axis_caps in axes_cfg.items():
            if not isinstance(axis_caps, dict):
                errors.append(
                    f"caps.axes.{axis} must be a mapping",
                )
                continue
            axis_min = _as_float(axis_caps.get("min_scale"))
            axis_max = _as_float(axis_caps.get("max_scale"))
            if None not in (axis_min, axis_max) and axis_min > axis_max:
                errors.append(
                    f"caps.axes.{axis}.min_scale must be <= max_scale "
                    f"(got min={axis_min}, max={axis_max})",
                )


def _validate_audio_adaptive(config: Dict[str, Any]) -> List[str]:
    errors: List[str] = []

    enabled = config.get("enabled")
    if not isinstance(enabled, bool):
        errors.append("audio_adaptive_weights.enabled must be a boolean")

    min_conf = _as_float(config.get("min_confidence"))
    if min_conf is None or not (0.0 <= min_conf <= 1.0):
        errors.append("min_confidence must be within [0.0, 1.0]")

    missing_policy = config.get("missing_policy")
    if missing_policy not in _VALID_MISSING_POLICIES:
        policy_list = ", ".join(sorted(_VALID_MISSING_POLICIES))
        errors.append(
            f"missing_policy must be one of {{{policy_list}}}",
        )

    cooldown_loops = config.get("cooldown_loops")
    if cooldown_loops is not None:
        try:
            if int(cooldown_loops) < 0:
                errors.append("cooldown_loops must be non-negative")
        except (TypeError, ValueError):
            errors.append("cooldown_loops must be an integer")

    cooldown_by_rule = config.get("cooldown_by_rule")
    if isinstance(cooldown_by_rule, dict):
        for rule_name, value in cooldown_by_rule.items():
            try:
                if int(value) < 0:
                    errors.append(
                        f"cooldown_by_rule.{rule_name} must be non-negative",
                    )
            except (TypeError, ValueError):
                errors.append(
                    f"cooldown_by_rule.{rule_name} must be an integer",
                )

    max_total_delta = config.get("max_total_delta")
    if max_total_delta is not None:
        value = _as_float(max_total_delta)
        if value is None or value < 0.0:
            errors.append("max_total_delta must be >= 0")

    # Validate max_total_delta_per_axis
    per_axis_delta = config.get("max_total_delta_per_axis")
    if isinstance(per_axis_delta, dict):
        for axis_name, limit_val in per_axis_delta.items():
            limit_f = _as_float(limit_val)
            if limit_f is None or limit_f < 0.0:
                errors.append(f"max_total_delta_per_axis.{axis_name} must be >= 0")

    # Cross-constraint: missing_policy='zero' + cooldown=0 is risky
    missing_policy = config.get("missing_policy")
    cooldown_loops = config.get("cooldown_loops")
    if missing_policy == "zero" and cooldown_loops == 0:
        errors.append(
            "missing_policy='zero' with cooldown_loops=0 may cause "
            "instability; consider cooldown_loops >= 1"
        )

    normalize_cfg = config.get("normalize")
    if isinstance(normalize_cfg, dict) and normalize_cfg.get("enabled"):
        target_sum = _as_float(normalize_cfg.get("target_sum"))
        if target_sum is None or target_sum <= 0.0:
            errors.append("normalize.target_sum must be > 0 when enabled")

    caps_cfg = config.get("caps")
    if isinstance(caps_cfg, dict):
        _validate_caps(caps_cfg, errors)

    return errors


def validate_stage2_config(payload: Dict[str, Any]) -> List[str]:
    score_cfg = payload.get("score")
    if not isinstance(score_cfg, dict):
        return ["score section must be present"]
    audio_cfg = score_cfg.get("audio_adaptive_weights")
    if not isinstance(audio_cfg, dict):
        return ["score.audio_adaptive_weights must be a mapping"]
    return _validate_audio_adaptive(audio_cfg)


def _load_yaml(path: Path) -> Dict[str, Any]:
    content = path.read_text(encoding="utf-8")
    data = yaml.safe_load(content)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return data


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Validate Stage2 audio adaptive configuration",
    )
    parser.add_argument(
        "paths",
        nargs="+",
        help="Config file(s) to validate",
    )
    args = parser.parse_args(argv)

    failures = 0
    for path_text in args.paths:
        config_path = Path(path_text)
        try:
            payload = _load_yaml(config_path)
        except (OSError, ValueError) as exc:
            print(f"[ERROR] {config_path}: {exc}")
            failures += 1
            continue

        issues = validate_stage2_config(payload)
        if issues:
            print(f"[FAIL] {config_path}")
            for issue in issues:
                print(f"  - {issue}")
            failures += 1
        else:
            print(f"[OK] {config_path}")

    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
