#!/usr/bin/env python3
"""Apply retry presets to Stage2 metrics JSONL files."""

from __future__ import annotations

import argparse
import gzip
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import (
    Any,
    Dict,
    Mapping,
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
    TextIO,
    cast,
)

import yaml


@dataclass(frozen=True)
class AxisLimit:
    axis: str
    operator: str
    threshold: float


PresetStats = Dict[str, Dict[str, int]]


_LIMIT_PATTERN = re.compile(
    r"^\s*([a-zA-Z0-9_]+)\s*(<=|>=|<|>|==|!=)\s*([0-9.]+)\s*$",
)
_CONDITION_PATTERN = re.compile(
    r"^\s*([a-zA-Z0-9_.]+)\s*(<=|>=|<|>|==|!=)\s*([0-9.]+)\s*$",
)


def _match_operator(value: float, operator: str, threshold: float) -> bool:
    if operator == "<":
        return value < threshold
    if operator == "<=":
        return value <= threshold
    if operator == ">":
        return value > threshold
    if operator == ">=":
        return value >= threshold
    if operator == "==":
        return value == threshold
    if operator == "!=":
        return value != threshold
    raise ValueError(f"Unsupported operator: {operator}")


def _match_expression(value: float, expression: str) -> bool:
    match = re.match(r"\s*([<>]=?)\s*([0-9.]+)\s*$", expression)
    if not match:
        return False
    operator, threshold_text = match.groups()
    threshold = float(threshold_text)
    return _match_operator(value, operator, threshold)


def _parse_axis_limit(raw: str) -> AxisLimit:
    match = _LIMIT_PATTERN.match(raw)
    if not match:
        raise ValueError(
            "Invalid axis limit " f"'{raw}'. " "Expected format 'axis<value' or 'axis<=value'."
        )
    axis, operator, threshold_text = match.groups()
    return AxisLimit(
        axis=axis,
        operator=operator,
        threshold=float(threshold_text),
    )


def _passes_axis_limits(
    axes_raw: Dict[str, Any],
    limits: Iterable[AxisLimit],
) -> bool:
    for limit in limits:
        raw_value = axes_raw.get(limit.axis, 0.0)
        try:
            value = float(raw_value)
        except (TypeError, ValueError):
            return False
        if not _match_operator(value, limit.operator, limit.threshold):
            return False
    return True


def _resolve_axis_key(key: str) -> Tuple[str, str]:
    axis_key = key.strip()
    if not axis_key:
        return key, ""
    _, _, axis_name = axis_key.partition(".")
    return axis_key, axis_name or axis_key


def _resolve_nested_value(
    context: Mapping[str, Any],
    path: str,
) -> Any:
    current_value: Any = context
    for part in path.split("."):
        if isinstance(current_value, Mapping) and part in current_value:
            current_value = cast(Any, current_value[part])
        else:
            return None
    return current_value


def _parse_bool_literal(text: str) -> Optional[bool]:
    normalized = text.strip().lower()
    if normalized in {"true", "1", "yes", "on"}:
        return True
    if normalized in {"false", "0", "no", "off"}:
        return False
    return None


def _evaluate_condition(
    condition: Any,
    loop_row: Mapping[str, Any],
    axes_raw: Dict[str, Any],
) -> Tuple[bool, List[Dict[str, Any]]]:
    if condition is None:
        return False, [
            {
                "key": "<missing>",
                "expression": None,
                "value": None,
                "ok": False,
                "note": "preset.when not provided",
            },
        ]

    clauses: List[Dict[str, Any]] = []
    matched = True

    items: Iterable[Tuple[str, str]]
    if isinstance(condition, dict):
        items = [
            (str(cond_key), str(cond_value))
            for cond_key, cond_value in cast(Dict[str, Any], condition).items()
        ]
    elif isinstance(condition, str):
        match = _CONDITION_PATTERN.match(condition)
        if not match:
            return False, [
                {
                    "key": condition,
                    "expression": None,
                    "value": None,
                    "ok": False,
                    "note": "unable to parse preset.when expression",
                },
            ]
        key_text, operator_text, threshold_text = match.groups()
        items = [(key_text, f"{operator_text}{threshold_text}")]
    else:
        return False, [
            {
                "key": str(condition),
                "expression": None,
                "value": None,
                "ok": False,
                "note": "unsupported preset.when type",
            },
        ]

    context: Dict[str, Any] = dict(loop_row)
    context.setdefault("axes_raw", axes_raw)

    for item_key, expression in items:
        key_text = str(item_key).strip()
        exists_check = False
        exists_path = ""
        if key_text.lower().startswith("exists(") and key_text.endswith(")"):
            exists_check = True
            exists_path = key_text[key_text.find("(") + 1 : -1].strip()

        if exists_check:
            resolved = _resolve_nested_value(context, exists_path)
            exists_value = resolved is not None
            expected = _parse_bool_literal(expression)
            ok = exists_value if expected is None else exists_value is expected
            clauses.append(
                {
                    "key": key_text,
                    "expression": expression,
                    "value": exists_value,
                    "ok": ok,
                },
            )
            if not ok:
                matched = False
            continue

        value_obj = _resolve_nested_value(context, key_text)
        if value_obj is None and "." in key_text:
            _, axis_name = _resolve_axis_key(key_text)
            if axis_name and axis_name != key_text:
                value_obj = axes_raw.get(axis_name)

        try:
            numeric_value = float(value_obj) if value_obj is not None else None
        except (TypeError, ValueError):
            numeric_value = None

        ok = False
        if numeric_value is not None:
            ok = _match_expression(numeric_value, expression)

        clauses.append(
            {
                "key": key_text,
                "expression": expression,
                "value": numeric_value,
                "ok": ok,
            },
        )

        if not ok:
            matched = False

    return matched, clauses


def _apply_steps(
    loop_row: Dict[str, Any],
    steps: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    existing_ops = loop_row.setdefault("_retry_ops", [])
    operations: List[Dict[str, Any]]
    if isinstance(existing_ops, list):
        operations = cast(List[Dict[str, Any]], existing_ops)
    else:
        operations = []
        loop_row["_retry_ops"] = operations
    operations.extend(list(steps))
    return loop_row


def _record_preset_stat(stats: PresetStats, name: str, matched: bool) -> None:
    entry = stats.setdefault(
        name,
        {"total": 0, "matched": 0, "unmatched": 0},
    )
    entry["total"] += 1
    if matched:
        entry["matched"] += 1
    else:
        entry["unmatched"] += 1


def _append_limited(
    sequence: List[Dict[str, Any]],
    item: Dict[str, Any],
    *,
    limit: int = 20,
) -> None:
    sequence.append(item)
    if len(sequence) > limit:
        del sequence[:-limit]


def _ensure_retry_state(loop_row: Dict[str, Any]) -> Dict[str, Any]:
    state_obj = loop_row.get("_retry_state")
    if not isinstance(state_obj, dict):
        state_obj = {}
    state = cast(Dict[str, Any], state_obj)
    for key in (
        "attempts",
        "cooldowns",
        "last_axis",
        "last_score",
        "last_delta",
    ):
        sub = state.get(key)
        if not isinstance(sub, dict):
            state[key] = {}
    loop_row["_retry_state"] = state
    return state


def _ensure_retry_control(loop_row: Dict[str, Any]) -> Dict[str, Any]:
    payload = loop_row.get("_retry_control")
    if not isinstance(payload, dict):
        payload = {}
    control = cast(Dict[str, Any], payload)
    blocked = control.get("blocked")
    if not isinstance(blocked, list):
        control["blocked"] = []
    applied = control.get("applied")
    if not isinstance(applied, list):
        control["applied"] = []
    loop_row["_retry_control"] = control
    return control


def _coerce_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return None


def _is_gzip_path(path: Path) -> bool:
    suffixes = path.suffixes
    if not suffixes:
        return False
    return suffixes[-1] == ".gz"


def _write_explain_summary(stream: TextIO, stats: PresetStats) -> None:
    if not stats:
        return
    payload: Dict[str, Any] = {
        "summary": "preset_counts",
        "presets": [
            {
                "name": name,
                "total": counts.get("total", 0),
                "matched": counts.get("matched", 0),
                "unmatched": counts.get("unmatched", 0),
            }
            for name, counts in sorted(stats.items())
        ],
    }
    stream.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _format_preset_summary(stats: PresetStats) -> str:
    if not stats:
        return ""

    total_evals = 0
    total_hits = 0
    total_misses = 0
    parts: List[str] = []
    for name in sorted(stats):
        counts = stats[name]
        total = counts.get("total", 0)
        matched = counts.get("matched", 0)
        unmatched = counts.get("unmatched", total - matched)
        total_evals += total
        total_hits += matched
        total_misses += unmatched
        parts.append(
            f"{name}: total={total} hit={matched} miss={unmatched}",
        )

    headline = (
        "[retry_apply] preset_summary " f"total={total_evals} hit={total_hits} miss={total_misses}"
    )
    detail = "; ".join(parts)
    return f"{headline} | {detail}"


def load_presets(path: Path) -> List[Dict[str, Any]]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        return []
    data_dict = cast(Dict[str, Any], data)
    presets_obj = data_dict.get("presets", [])
    if not isinstance(presets_obj, list):
        return []
    presets: List[Dict[str, Any]] = []
    presets_seq = cast(List[Any], presets_obj)
    for preset in presets_seq:
        if isinstance(preset, dict):
            presets.append(cast(Dict[str, Any], preset))
    return presets


def process(
    presets_path: Path,
    input_path: Path,
    output_path: Path,
    *,
    min_score: Optional[float] = None,
    max_score: Optional[float] = None,
    axis_limits: Optional[List[AxisLimit]] = None,
    round_index: Optional[int] = None,
    max_rounds: Optional[int] = None,
    explain_writer: Optional[TextIO] = None,
) -> Tuple[List[str], int, PresetStats]:
    _ = output_path  # preserved for API compatibility
    presets = load_presets(presets_path)
    if not presets:
        return [], 0, {}

    axis_limits = list(axis_limits or [])

    output_lines: List[str] = []
    applied_total = 0
    preset_stats: PresetStats = {}
    with input_path.open("r", encoding="utf-8") as stream:
        for line in stream:
            text = line.strip()
            if not text:
                continue
            row_obj = json.loads(text)
            if not isinstance(row_obj, dict):
                continue
            loop_row = cast(Dict[str, Any], row_obj)

            score_obj = loop_row.get("score")
            score_value: Optional[float]
            if score_obj is None:
                score_value = None
            else:
                try:
                    score_value = float(score_obj)  # type: ignore[arg-type]
                except (TypeError, ValueError):
                    score_value = None

            below_min = min_score is not None and (score_value is None or score_value < min_score)
            if below_min:
                output_lines.append(json.dumps(loop_row, ensure_ascii=False))
                continue
            above_max = max_score is not None and (score_value is None or score_value > max_score)
            if above_max:
                output_lines.append(json.dumps(loop_row, ensure_ascii=False))
                continue

            axes_raw_candidate = loop_row.get("axes_raw")
            axes_raw = (
                cast(Dict[str, Any], axes_raw_candidate)
                if isinstance(axes_raw_candidate, dict)
                else {}
            )

            if axis_limits and not _passes_axis_limits(axes_raw, axis_limits):
                output_lines.append(json.dumps(loop_row, ensure_ascii=False))
                continue

            existing_round_obj = loop_row.get("_retry_round")
            if isinstance(existing_round_obj, (int, float)):
                existing_round = int(existing_round_obj)
            elif isinstance(existing_round_obj, str):
                try:
                    existing_round = int(existing_round_obj.strip() or "0")
                except ValueError:
                    existing_round = 0
            else:
                existing_round = 0

            if max_rounds is not None and existing_round >= max_rounds:
                output_lines.append(json.dumps(loop_row, ensure_ascii=False))
                continue

            applied_any = False
            retry_state = _ensure_retry_state(loop_row)
            retry_control = _ensure_retry_control(loop_row)
            blocked_notes = cast(
                List[Dict[str, Any]],
                retry_control["blocked"],
            )
            applied_notes = cast(
                List[Dict[str, Any]],
                retry_control["applied"],
            )
            attempts_state = cast(Dict[str, Any], retry_state["attempts"])
            cooldown_state = cast(Dict[str, Any], retry_state["cooldowns"])
            last_axis_state = cast(Dict[str, Any], retry_state["last_axis"])
            last_score_state = cast(Dict[str, Any], retry_state["last_score"])
            last_delta_state = cast(Dict[str, Any], retry_state["last_delta"])
            for preset in presets:
                preset_id = str(preset.get("id") or preset.get("name") or "<unnamed>")
                matched, clauses = _evaluate_condition(
                    preset.get("when"),
                    loop_row,
                    axes_raw,
                )

                _record_preset_stat(preset_stats, preset_id, matched)

                if explain_writer is not None:
                    explain_writer.write(
                        json.dumps(
                            {
                                "loop_id": loop_row.get("loop_id"),
                                "preset": preset_id,
                                "matched": matched,
                                "clauses": clauses,
                            },
                            ensure_ascii=False,
                        )
                        + "\n",
                    )

                if matched:
                    control_raw = preset.get("control")
                    if isinstance(control_raw, Mapping):
                        control_cfg = cast(Mapping[str, Any], control_raw)
                    else:
                        control_cfg = None
                    blocked_by_control = False
                    cooldown_key = preset_id
                    axis_name = str(preset.get("axis") or "").strip()
                    current_axis_value: Optional[float] = None
                    previous_axis_value: Optional[float] = None
                    delta_value: Optional[float] = None
                    cooldown_runs = 0
                    attempts_done = int(_coerce_float(attempts_state.get(preset_id)) or 0)
                    max_attempts_limit: Optional[int] = None
                    min_delta_threshold: Optional[float] = None
                    min_delta_dict: Optional[Dict[str, Any]] = None
                    priority_value: int = 0

                    if control_cfg is not None:
                        cooldown_key = str(control_cfg.get("cooldown_key") or preset_id)
                        axis_name = str(control_cfg.get("axis") or axis_name).strip()
                        cooldown_runs_value = _coerce_float(control_cfg.get("cooldown_runs")) or 0
                        cooldown_runs = int(cooldown_runs_value)

                        # Priority support
                        priority_raw = control_cfg.get("priority")
                        if priority_raw is not None:
                            try:
                                priority_value = int(priority_raw)
                            except (TypeError, ValueError):
                                priority_value = 0

                        max_attempts_raw = control_cfg.get("max_attempts")
                        if max_attempts_raw is not None:
                            try:
                                max_attempts_limit = int(max_attempts_raw)
                            except (TypeError, ValueError):
                                max_attempts_limit = None

                        # min_delta: support both float (backward compat) and dict (new)
                        min_delta_raw = control_cfg.get("min_delta")
                        if min_delta_raw is not None:
                            if isinstance(min_delta_raw, dict):
                                min_delta_dict = cast(Dict[str, Any], min_delta_raw)
                            else:
                                try:
                                    min_delta_threshold = float(min_delta_raw)
                                except (TypeError, ValueError):
                                    min_delta_threshold = None

                        if cooldown_runs > 0:
                            cooldown_remaining_value = (
                                _coerce_float(cooldown_state.get(cooldown_key)) or 0
                            )
                            remaining = int(cooldown_remaining_value)
                            if remaining > 0:
                                cooldown_state[cooldown_key] = max(
                                    remaining - 1,
                                    0,
                                )
                                blocked_by_control = True
                                _append_limited(
                                    blocked_notes,
                                    {
                                        "preset": preset_id,
                                        "reason": "cooldown",
                                        "cooldown_key": cooldown_key,
                                        "remaining": remaining,
                                    },
                                )

                        if (
                            not blocked_by_control
                            and max_attempts_limit is not None
                            and max_attempts_limit > 0
                        ):
                            if attempts_done >= max_attempts_limit:
                                blocked_by_control = True
                                _append_limited(
                                    blocked_notes,
                                    {
                                        "preset": preset_id,
                                        "reason": "max_attempts",
                                        "cooldown_key": cooldown_key,
                                        "attempts": attempts_done,
                                        "limit": max_attempts_limit,
                                    },
                                )

                        if axis_name:
                            current_axis_value = _coerce_float(axes_raw.get(axis_name))
                            previous_axis_value = _coerce_float(last_axis_state.get(cooldown_key))
                            if current_axis_value is not None and previous_axis_value is not None:
                                delta_value = current_axis_value - previous_axis_value
                            if (
                                not blocked_by_control
                                and min_delta_threshold is not None
                                and current_axis_value is not None
                                and previous_axis_value is not None
                                and delta_value is not None
                                and delta_value < min_delta_threshold
                            ):
                                blocked_by_control = True
                                _append_limited(
                                    blocked_notes,
                                    {
                                        "preset": preset_id,
                                        "reason": "min_delta",
                                        "cooldown_key": cooldown_key,
                                        "delta": delta_value,
                                        "threshold": min_delta_threshold,
                                        "axis": axis_name,
                                    },
                                )

                        # min_delta dict support (score_total + axes_raw)
                        if not blocked_by_control and min_delta_dict is not None:
                            block_reasons = []
                            score_total_req = _coerce_float(min_delta_dict.get("score_total"))
                            if score_total_req is not None:
                                prev_score = _coerce_float(last_score_state.get(cooldown_key))
                                if score_value is not None and prev_score is not None:
                                    score_delta = score_value - prev_score
                                    if score_delta < score_total_req:
                                        block_reasons.append(
                                            f"score_delta={score_delta:.3f}<{score_total_req:.3f}"
                                        )

                            axes_raw_req = min_delta_dict.get("axes_raw")
                            if isinstance(axes_raw_req, dict):
                                for ax_name, ax_threshold in axes_raw_req.items():
                                    ax_threshold_f = _coerce_float(ax_threshold)
                                    if ax_threshold_f is None:
                                        continue
                                    curr_val = _coerce_float(axes_raw.get(ax_name))
                                    prev_val = _coerce_float(
                                        last_axis_state.get(f"{cooldown_key}:{ax_name}")
                                    )
                                    if curr_val is not None and prev_val is not None:
                                        ax_delta = curr_val - prev_val
                                        if ax_delta < ax_threshold_f:
                                            block_reasons.append(
                                                f"{ax_name}_delta={ax_delta:.3f}<{ax_threshold_f:.3f}"
                                            )

                            if block_reasons:
                                blocked_by_control = True
                                _append_limited(
                                    blocked_notes,
                                    {
                                        "preset": preset_id,
                                        "reason": "min_delta_dict",
                                        "cooldown_key": cooldown_key,
                                        "details": block_reasons,
                                    },
                                )

                        if axis_name and current_axis_value is not None:
                            last_axis_state[cooldown_key] = current_axis_value
                        if delta_value is not None:
                            last_delta_state[cooldown_key] = delta_value

                        # Store axes_raw values for min_delta_dict
                        if min_delta_dict and "axes_raw" in min_delta_dict:
                            axes_raw_req = min_delta_dict["axes_raw"]
                            if isinstance(axes_raw_req, dict):
                                for ax_name in axes_raw_req.keys():
                                    curr_val = _coerce_float(axes_raw.get(ax_name))
                                    if curr_val is not None:
                                        last_axis_state[f"{cooldown_key}:{ax_name}"] = curr_val

                        if blocked_by_control:
                            continue
                    else:
                        if axis_name:
                            current_axis_value = _coerce_float(axes_raw.get(axis_name))
                            if current_axis_value is not None:
                                last_axis_state.setdefault(
                                    axis_name,
                                    current_axis_value,
                                )

                    steps_raw = preset.get("apply", [])
                    if isinstance(steps_raw, list):
                        steps_seq = cast(List[Any], steps_raw)
                        typed_steps = [
                            cast(Dict[str, Any], step)
                            for step in steps_seq
                            if isinstance(step, dict)
                        ]
                        if typed_steps:
                            loop_row = _apply_steps(loop_row, typed_steps)
                            applied_any = True
                            applied_total += 1
                            if control_cfg is not None:
                                attempts_state[preset_id] = attempts_done + 1
                                if cooldown_runs > 0:
                                    cooldown_state[cooldown_key] = cooldown_runs
                                elif cooldown_key in cooldown_state:
                                    cooldown_state[cooldown_key] = 0
                                if axis_name:
                                    if current_axis_value is not None:
                                        last_axis_state[cooldown_key] = current_axis_value
                                if delta_value is not None:
                                    last_delta_state[cooldown_key] = delta_value
                                if score_value is not None:
                                    last_score_state[cooldown_key] = score_value

                                # Store axes_raw for min_delta_dict
                                if min_delta_dict and "axes_raw" in min_delta_dict:
                                    axes_raw_req = min_delta_dict["axes_raw"]
                                    if isinstance(axes_raw_req, dict):
                                        for ax_name in axes_raw_req.keys():
                                            curr_val = _coerce_float(axes_raw.get(ax_name))
                                            if curr_val is not None:
                                                key = f"{cooldown_key}:{ax_name}"
                                                last_axis_state[key] = curr_val

                                if round_index is not None:
                                    applied_round = round_index
                                else:
                                    applied_round = max(existing_round, 0) + 1
                                _append_limited(
                                    applied_notes,
                                    {
                                        "preset": preset_id,
                                        "cooldown_key": cooldown_key,
                                        "round": applied_round,
                                        "attempt": attempts_state[preset_id],
                                        "axis": axis_name or None,
                                        "axis_value": current_axis_value,
                                        "delta": delta_value,
                                        "priority": priority_value,
                                    },
                                )

            if applied_any:
                if round_index is not None:
                    loop_row["_retry_round"] = round_index
                else:
                    loop_row["_retry_round"] = max(existing_round, 0) + 1

            output_lines.append(json.dumps(loop_row, ensure_ascii=False))

    return output_lines, applied_total, preset_stats


def _legacy_main(argv: Sequence[str]) -> None:
    presets_path = Path(argv[1])
    input_path = Path(argv[2])
    output_path = Path(argv[3])
    lines, _, _ = process(presets_path, input_path, output_path)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Optional[Sequence[str]] = None) -> None:
    argv = tuple(argv or sys.argv)
    if len(argv) == 4 and not argv[1].startswith("-"):
        _legacy_main(argv)
        return

    parser = argparse.ArgumentParser(
        description="Apply retry presets to Stage2 metrics JSONL files",
    )
    parser.add_argument(
        "--presets",
        required=True,
        help="Path to retry_presets.yaml",
    )
    parser.add_argument(
        "--scores",
        required=True,
        help="Input metrics JSONL path",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Output JSONL path",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        help="Skip loops below this Stage2 score",
    )
    parser.add_argument(
        "--max-score",
        type=float,
        help="Skip loops above this Stage2 score",
    )
    parser.add_argument(
        "--axis-limit",
        action="append",
        default=[],
        help=("Filter by axes_raw condition, e.g. 'velocity<0.35'. " "May be repeated."),
    )
    parser.add_argument(
        "--round",
        type=int,
        default=None,
        help="Retry round index to assign when presets are applied",
    )
    parser.add_argument(
        "--max-rounds",
        type=int,
        default=None,
        help=("Stop applying presets if existing _retry_round is at or " "above this value"),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Evaluate presets without writing output file",
    )
    parser.add_argument(
        "--explain",
        action="store_true",
        help="Emit JSONL explanations for preset matches",
    )
    parser.add_argument(
        "--explain-out",
        default=None,
        help="Path to write explanation JSONL (default: alongside --out)",
    )

    args = parser.parse_args(argv[1:])

    axis_limits = [_parse_axis_limit(text) for text in args.axis_limit]

    out_path = Path(args.out)
    explain_path: Optional[Path]
    if args.explain:
        if args.explain_out:
            explain_path = Path(args.explain_out)
        else:
            explain_filename = f"{out_path.stem}.explain.jsonl.gz"
            explain_path = out_path.with_name(explain_filename)
        explain_path.parent.mkdir(parents=True, exist_ok=True)
        if _is_gzip_path(explain_path):
            explain_stream = gzip.open(explain_path, "wt", encoding="utf-8")
        else:
            explain_stream = explain_path.open("w", encoding="utf-8")
    else:
        explain_path = None
        explain_stream = None

    lines, applied_total, preset_stats = process(
        Path(args.presets),
        Path(args.scores),
        out_path,
        min_score=args.min_score,
        max_score=args.max_score,
        axis_limits=axis_limits,
        round_index=args.round,
        max_rounds=args.max_rounds,
        explain_writer=explain_stream,
    )

    if explain_stream is not None:
        _write_explain_summary(explain_stream, preset_stats)
        explain_stream.close()

    if explain_path is not None:
        print(f"[retry_apply] explain -> {explain_path}")

    summary_line = _format_preset_summary(preset_stats)
    if summary_line:
        print(summary_line)

    if args.dry_run:
        print("[retry_apply] DRY-RUN -- applied_presets=" f"{applied_total} / records={len(lines)}")
        return

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("[retry_apply] applied_presets=" f"{applied_total} / records={len(lines)} -> {out_path}")


if __name__ == "__main__":
    main()
