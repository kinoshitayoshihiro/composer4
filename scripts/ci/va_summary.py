#!/usr/bin/env python3
"""Summarize EMOPIA valence/arousal metrics."""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
from typing import Any, Dict, Iterable, List, Optional, cast


def read_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def pluck(data: Dict[str, Any], dotted: str) -> Any:
    current: Any = data
    for part in dotted.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def _as_valid(value: Any) -> Optional[float]:
    if isinstance(value, (int, float)) and math.isfinite(value):
        return float(value)
    return None


def _median(values: Iterable[Optional[float]]) -> Optional[float]:
    filtered = [v for v in values if v is not None]
    if not filtered:
        return None
    return float(statistics.median(filtered))


def _iqr(values: Iterable[Optional[float]]) -> Optional[float]:
    filtered = sorted(v for v in values if v is not None)
    if not filtered:
        return None
    q1_index = int(round(0.25 * (len(filtered) - 1)))
    q3_index = int(round(0.75 * (len(filtered) - 1)))
    return float(filtered[q3_index] - filtered[q1_index])


def _scalar_or_seq(entry: Dict[str, Any], key: str) -> Optional[float]:
    scalar = _as_valid(pluck(entry, f"metrics.emopia.{key}"))
    if scalar is not None:
        return scalar
    seq = pluck(entry, f"metrics.emopia.{key}_seq")
    if isinstance(seq, list):
        converted: List[Optional[float]] = []
        for item in cast(List[Any], seq):
            value = _as_valid(item)
            if value is not None:
                converted.append(value)
        values = converted
        if values:
            return _median(values)
    return None


def summarize(entries: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    valence: List[Optional[float]] = []
    arousal: List[Optional[float]] = []
    for entry in entries:
        valence.append(_scalar_or_seq(entry, "valence"))
        arousal.append(_scalar_or_seq(entry, "arousal"))

    valence_clean = [v for v in valence if v is not None]
    arousal_clean = [v for v in arousal if v is not None]

    return {
        "n_valence": len(valence_clean),
        "n_arousal": len(arousal_clean),
        "valence_p50": _median(valence_clean),
        "valence_iqr": _iqr(valence_clean),
        "arousal_p50": _median(arousal_clean),
        "arousal_iqr": _iqr(arousal_clean),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        required=True,
        help="JSONL with EMOPIA metrics",
    )
    parser.add_argument(
        "--out", default="artifacts/va_summary.json", help="output JSON path"
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    entries = list(read_jsonl(args.input))
    stats = summarize(entries)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(stats, handle, ensure_ascii=False, indent=2)
    print(f"[va_summary] wrote {args.out} -> {stats}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
