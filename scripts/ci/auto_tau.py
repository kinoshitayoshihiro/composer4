#!/usr/bin/env python3
"""Propose tau thresholds from Stage2 metrics JSONL."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import (
    Any,
    Dict,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    cast,
)


_MISSING = object()


def _read_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                yield payload


def _pluck(
    payload: Mapping[str, Any],
    dotted_key: str,
    default: Any = None,
) -> Any:
    current_value: Any = payload
    for part in dotted_key.split("."):
        if not isinstance(current_value, Mapping):
            return default
        current_map = cast(Mapping[str, Any], current_value)
        next_value = current_map.get(part, _MISSING)
        if next_value is _MISSING:
            return default
        current_value = cast(Any, next_value)
    return current_value


def _as_float(value: Any) -> Optional[float]:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric):
        return None
    return numeric


def _quantile(values: Sequence[float], q: float) -> Optional[float]:
    if not values:
        return None
    sorted_vals = sorted(values)
    if len(sorted_vals) == 1:
        return float(sorted_vals[0])
    index = q * (len(sorted_vals) - 1)
    lower = int(math.floor(index))
    upper = int(math.ceil(index))
    if lower == upper:
        return float(sorted_vals[lower])
    fraction = index - lower
    blended = (sorted_vals[lower] * (1.0 - fraction)) + (sorted_vals[upper] * fraction)
    return float(blended)


def _tempo_bucket(bpm: Optional[float], edges: Sequence[float]) -> str:
    if bpm is None:
        return "NA"
    if not edges:
        return "all"
    previous: Optional[float] = None
    for edge in edges:
        if bpm < edge:
            if previous is None:
                return f"<{edge:g}"
            return f"{previous:g}-{edge:g}"
        previous = edge
    return f">={edges[-1]:g}"


def _tau_for(values: Sequence[float], target_pct: float) -> Optional[float]:
    if not values:
        return None
    q = min(max(target_pct / 100.0, 0.0), 1.0)
    return _quantile(values, q)


def _format_yaml_block(data: Dict[str, Any], indent: int = 0) -> List[str]:
    pad = "  " * indent
    lines: List[str] = []
    for key in sorted(data):
        value = data[key]
        if isinstance(value, dict):
            typed_value = cast(Dict[str, Any], value)
            lines.append(f"{pad}{key}:")
            lines.extend(_format_yaml_block(typed_value, indent + 1))
        else:
            if value is None:
                rendered = "null"
            else:
                rendered = f"{float(value):.6f}"
            lines.append(f"{pad}{key}: {rendered}")
    return lines


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Propose tau thresholds.")
    parser.add_argument("--input", required=True, help="Metrics JSONL path")
    parser.add_argument(
        "--target-mismatch",
        type=float,
        default=15.0,
        help="Target mismatch rate percentage (default: 15)",
    )
    parser.add_argument(
        "--by-instrument",
        action="store_true",
        help="Emit per-instrument tau proposals",
    )
    parser.add_argument(
        "--by-tempo",
        action="store_true",
        help="Emit per-tempo-bin tau proposals",
    )
    parser.add_argument(
        "--tempo-edges",
        default="95,110,130,150",
        help="Comma separated tempo boundaries (default: 95,110,130,150)",
    )
    parser.add_argument(
        "--instrument-key",
        default="instrument",
        help="Dotted key for instrument name (default: instrument)",
    )
    parser.add_argument(
        "--bpm-key",
        default="tempo.bpm",
        help="Dotted key for BPM (default: tempo.bpm)",
    )
    parser.add_argument(
        "--cos-key",
        default="metrics.text_audio_cos",
        help=("Dotted key for text-audio cosine " "(default: metrics.text_audio_cos)"),
    )
    parser.add_argument(
        "--out",
        default="artifacts/auto_tau.yaml",
        help="Output YAML path (default: artifacts/auto_tau.yaml)",
    )
    return parser


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Metrics JSONL not found: {input_path}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    tempo_edges: List[float] = []
    for token in args.tempo_edges.split(","):
        value = _as_float(token)
        if value is not None:
            tempo_edges.append(value)
    tempo_edges.sort()

    all_values: List[float] = []
    by_instrument: Dict[str, List[float]] = defaultdict(list)
    by_tempo: Dict[str, List[float]] = defaultdict(list)
    by_instrument_tempo: Dict[str, Dict[str, List[float]]] = defaultdict(
        lambda: defaultdict(list),
    )

    for row in _read_jsonl(input_path):
        cos_value = _as_float(_pluck(row, args.cos_key))
        if cos_value is None:
            continue
        all_values.append(cos_value)

        instrument: Optional[str] = None
        if args.by_instrument:
            inst = _pluck(row, args.instrument_key)
            instrument = str(inst) if inst not in (None, "") else "NA"
            by_instrument[instrument].append(cos_value)

        tempo_bucket: Optional[str] = None
        if args.by_tempo:
            tempo_val = _as_float(_pluck(row, args.bpm_key))
            tempo_bucket = _tempo_bucket(tempo_val, tempo_edges)
            by_tempo[tempo_bucket].append(cos_value)

        if args.by_instrument and args.by_tempo:
            key_inst = instrument or "NA"
            key_tempo = tempo_bucket or "NA"
            by_instrument_tempo[key_inst][key_tempo].append(cos_value)

    result: Dict[str, Any] = {}
    result["global"] = _tau_for(all_values, args.target_mismatch)

    if args.by_instrument:
        result["by_instrument"] = {
            name: _tau_for(values, args.target_mismatch)
            for name, values in sorted(by_instrument.items())
        }

    if args.by_tempo:
        result["by_tempo"] = {
            bucket: _tau_for(values, args.target_mismatch)
            for bucket, values in sorted(by_tempo.items())
        }

    if args.by_instrument and args.by_tempo:
        nested: Dict[str, Dict[str, Optional[float]]] = {}
        for inst, tempo_map in sorted(by_instrument_tempo.items()):
            nested[inst] = {
                tempo_key: _tau_for(values, args.target_mismatch)
                for tempo_key, values in sorted(tempo_map.items())
            }
        result["by_instrument_tempo"] = nested

    yaml_lines = ["# auto_tau proposal"]
    yaml_lines.extend(_format_yaml_block(result))
    out_path.write_text("\n".join(yaml_lines) + "\n", encoding="utf-8")
    print(f"[auto_tau] wrote {out_path}")


if __name__ == "__main__":
    main()
