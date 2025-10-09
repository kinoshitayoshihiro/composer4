#!/usr/bin/env python3
"""Apply retry presets to Stage2 metrics JSONL files."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, cast

import yaml


def _match_expression(value: float, expression: str) -> bool:
    match = re.match(r"\s*([<>]=?)\s*([0-9.]+)\s*$", expression)
    if not match:
        return False
    operator, threshold_text = match.groups()
    threshold = float(threshold_text)
    if operator == "<":
        return value < threshold
    if operator == "<=":
        return value <= threshold
    if operator == ">":
        return value > threshold
    if operator == ">=":
        return value >= threshold
    return False


def _should_apply(preset: Dict[str, Any], axes_raw: Dict[str, Any]) -> bool:
    condition_obj = preset.get("when")
    if not isinstance(condition_obj, dict) or not condition_obj:
        return False
    condition = cast(Dict[str, Any], condition_obj)
    key_obj, expression_obj = next(iter(condition.items()))
    key = str(key_obj)
    expression = str(expression_obj)
    _, _, axis_name = key.partition(".")
    if not axis_name:
        return False
    raw_value = axes_raw.get(axis_name, 0.0)
    try:
        value = float(raw_value)
    except (TypeError, ValueError):
        return False
    return _match_expression(value, expression)


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


def process(presets_path: Path, input_path: Path, output_path: Path) -> None:
    presets = load_presets(presets_path)
    if not presets:
        output_path.write_text("", encoding="utf-8")
        return

    output_lines: List[str] = []
    with input_path.open("r", encoding="utf-8") as stream:
        for line in stream:
            text = line.strip()
            if not text:
                continue
            row_obj = json.loads(text)
            if not isinstance(row_obj, dict):
                continue
            loop_row = cast(Dict[str, Any], row_obj)
            axes_raw_candidate = loop_row.get("axes_raw")
            axes_raw = (
                cast(Dict[str, Any], axes_raw_candidate)
                if isinstance(axes_raw_candidate, dict)
                else {}
            )
            for preset in presets:
                if _should_apply(preset, axes_raw):
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
            output_lines.append(json.dumps(loop_row, ensure_ascii=False))

    output_path.write_text("\n".join(output_lines) + "\n", encoding="utf-8")


def main() -> None:
    if len(sys.argv) != 4:
        sys.stderr.write(
            "Usage: retry_apply.py <presets.yaml> " "<metrics.jsonl> <out.jsonl>\n",
        )
        sys.exit(1)

    presets_path = Path(sys.argv[1])
    input_path = Path(sys.argv[2])
    output_path = Path(sys.argv[3])
    process(presets_path, input_path, output_path)


if __name__ == "__main__":
    main()
