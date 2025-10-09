#!/usr/bin/env python3
"""Validate Stage2 articulation threshold configuration files.

The validator enforces the contract documented in
`configs/thresholds/schema.thresholds.v1.yaml`.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, cast

try:
    import yaml
except ImportError as exc:  # pragma: no cover - PyYAML is optional.
    raise SystemExit(
        "PyYAML is required to validate thresholds files",
    ) from exc


class ValidationError(Exception):
    """Raised when a configuration violates the schema."""


def _load_yaml(path: Path) -> Dict[str, Any]:
    raw = path.read_text(encoding="utf-8")
    obj = yaml.safe_load(raw)
    if obj is None:
        return {}
    if not isinstance(obj, dict):
        raise ValidationError("Top-level document must be a mapping")
    return cast(Dict[str, Any], obj)


def _ensure(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationError(message)


def _as_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _validate_bins_structure(
    bins: Iterable[Any],
    *,
    context: str,
) -> List[float]:
    floats: List[float] = []
    for idx, raw_value in enumerate(bins):
        float_value = _as_float(raw_value)
        if float_value is None:
            raise ValidationError(f"{context}: tempo[{idx}] must be numeric")
        floats.append(float_value)
    _ensure(len(floats) >= 2, f"{context}: need at least two tempo bounds")
    for prev, current in zip(floats, floats[1:]):
        _ensure(
            current > prev,
            (f"{context}: tempo bounds must be strictly increasing " f"(got {prev} -> {current})"),
        )
    return floats


def _validate_axis_bin(
    payload: Dict[str, Any],
    *,
    axis: str,
    index: int,
) -> None:
    bin_label = f"per_axis.{axis}.bins[{index}]"
    bounds: Optional[List[float]] = None
    if "bin" in payload:
        raw_bounds = payload["bin"]
        if not isinstance(raw_bounds, (list, tuple)):
            raise ValidationError(
                f"{bin_label}.bin must be a list of two floats",
            )
        bounds_iter = cast(Sequence[Any], raw_bounds)
        if len(bounds_iter) != 2:
            raise ValidationError(
                f"{bin_label}.bin must contain exactly two elements",
            )
        bounds = []
        for pos, raw_bound in enumerate(bounds_iter):
            bound_value = _as_float(raw_bound)
            if bound_value is None:
                raise ValidationError(
                    f"{bin_label}.bin[{pos}] must be numeric",
                )
            bounds.append(bound_value)
        lower, upper = bounds[0], bounds[1]
        if not lower < upper:
            raise ValidationError(
                f"{bin_label}.bin lower bound must be less than upper bound",
            )
    for key in ("low", "high"):
        if key not in payload or payload[key] is None:
            continue
        value = _as_float(payload[key])
        if value is None:
            raise ValidationError(
                f"{bin_label}.{key} must be numeric or null",
            )
        if not 0.0 <= value <= 1.0:
            raise ValidationError(
                f"{bin_label}.{key} must be within [0.0, 1.0]",
            )
    if "count" in payload and payload["count"] is not None:
        if not isinstance(payload["count"], int):
            raise ValidationError(
                f"{bin_label}.count must be an integer or null",
            )
        if payload["count"] < 0:
            raise ValidationError(f"{bin_label}.count must be >= 0")
    if "q1q2q3I" in payload and payload["q1q2q3I"] is not None:
        stats = payload["q1q2q3I"]
        stats_seq = cast(Sequence[Any], stats)
        if not isinstance(stats, (list, tuple)) or len(stats_seq) != 4:
            raise ValidationError(
                f"{bin_label}.q1q2q3I must be a list of four floats or null",
            )
        for pos, entry in enumerate(stats_seq):
            if _as_float(entry) is None:
                raise ValidationError(
                    f"{bin_label}.q1q2q3I[{pos}] must be numeric",
                )


def _validate_axis_block(
    axis: str,
    payload: Dict[str, Any],
    *,
    expected_bins: int,
) -> None:
    bins_raw = payload.get("bins")
    if not isinstance(bins_raw, list):
        raise ValidationError(f"per_axis.{axis}.bins must be a list")
    bins = cast(List[Any], bins_raw)
    if len(bins) != expected_bins:
        raise ValidationError(
            (f"per_axis.{axis}.bins must contain {expected_bins} entries " "(tempo bins - 1)"),
        )
    for index, item in enumerate(bins):
        if not isinstance(item, dict):
            raise ValidationError(
                f"per_axis.{axis}.bins[{index}] must be a mapping",
            )
        _validate_axis_bin(
            cast(Dict[str, Any], item),
            axis=axis,
            index=index,
        )


def validate_thresholds(path: Path) -> None:
    config = _load_yaml(path)

    mode_raw = str(config.get("mode", "auto")).lower()
    if mode_raw == "fixed":
        mode_raw = "manual"
    _ensure(mode_raw in {"auto", "manual"}, "mode must be 'auto' or 'manual'")

    bins_section = config.get("bins", {})
    _ensure(isinstance(bins_section, dict), "bins must be a mapping")
    tempo_bins_raw = bins_section.get("tempo")
    _ensure(tempo_bins_raw is not None, "bins.tempo is required")
    tempo_bins = _validate_bins_structure(
        tempo_bins_raw,
        context="bins.tempo",
    )

    per_axis_raw = config.get("per_axis")
    if not isinstance(per_axis_raw, dict) or not per_axis_raw:
        raise ValidationError("per_axis must be a non-empty mapping")
    per_axis = cast(Dict[str, Any], per_axis_raw)

    expected_bins = len(tempo_bins) - 1
    for axis_name, axis_payload in per_axis.items():
        axis_str = str(axis_name)
        if not axis_str:
            raise ValidationError("per_axis keys must be non-empty strings")
        _validate_axis_block(
            axis_str,
            cast(Dict[str, Any], axis_payload),
            expected_bins=expected_bins,
        )

    metadata = config.get("metadata")
    if metadata is not None:
        if not isinstance(metadata, dict):
            raise ValidationError("metadata must be a mapping when present")


def _summarise_ok(path: Path) -> None:
    print(f"OK: {path}")


def _summarise_error(path: Path, error: ValidationError) -> None:
    print(f"ERROR: {path}: {error}", file=sys.stderr)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate Stage2 threshold files",
    )
    parser.add_argument(
        "paths",
        nargs="+",
        type=Path,
        help="Threshold YAML/JSON files to validate",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print OK/ERROR summary (default prints nothing on success)",
    )
    args = parser.parse_args(argv)

    failures = False
    for path in args.paths:
        try:
            validate_thresholds(path)
        except ValidationError as exc:
            failures = True
            _summarise_error(path, exc)
        else:
            if args.summary:
                _summarise_ok(path)

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
