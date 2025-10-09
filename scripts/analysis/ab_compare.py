#!/usr/bin/env python3
"""Compare Stage2 metrics between baseline and trial runs."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class MetricsRow:
    score: Optional[float]
    axes_raw: Dict[str, float]
    tempo_bpm: Optional[float]


def _parse_float(value: object) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _load_metrics(path: Path) -> List[MetricsRow]:
    rows: List[MetricsRow] = []
    if not path.exists():
        raise FileNotFoundError(f"Metrics file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            payload = json.loads(text)
            if not isinstance(payload, dict):
                continue
            score = (
                _parse_float(payload.get("score"))
                or _parse_float(payload.get("stage2_score"))
                or _parse_float(payload.get("final_score"))
            )
            axes_raw_obj = payload.get("axes_raw")
            axes_raw: Dict[str, float] = {}
            if isinstance(axes_raw_obj, dict):
                for key, value in axes_raw_obj.items():
                    parsed = _parse_float(value)
                    if parsed is not None:
                        axes_raw[str(key)] = parsed
            tempo = (
                _parse_float(payload.get("tempo"))
                or _parse_float(payload.get("tempo_bpm"))
                or _parse_float(payload.get("bpm"))
            )
            rows.append(MetricsRow(score=score, axes_raw=axes_raw, tempo_bpm=tempo))
    if not rows:
        raise ValueError(f"No valid metrics rows parsed from {path}")
    return rows


def _percentile(values: Sequence[float], percentile: float) -> Optional[float]:
    if not values:
        return None
    return float(np.percentile(values, percentile))


def _format_percentile(values: Sequence[float], percentile: float) -> str:
    result = _percentile(values, percentile)
    return f"{result:.2f}" if result is not None else "n/a"


def _tempo_bin_labels(edges: Sequence[float]) -> List[str]:
    labels: List[str] = []
    if not edges:
        return ["all"]
    labels.append(f"<= {edges[0]:.1f}")
    for start, end in zip(edges[:-1], edges[1:]):
        labels.append(f"{start:.1f}-{end:.1f}")
    labels.append(f"> {edges[-1]:.1f}")
    return labels


def _assign_tempo_bin(tempo: Optional[float], edges: Sequence[float]) -> int:
    if not edges:
        return 0
    if tempo is None:
        return len(edges)  # treat unknown tempos as highest bin
    for idx, bound in enumerate(edges):
        if tempo <= bound:
            return idx
    return len(edges)


def _collect_scores(rows: Iterable[MetricsRow]) -> List[float]:
    scores = [row.score for row in rows if row.score is not None]
    return [float(value) for value in scores]


def _axis_values(rows: Iterable[MetricsRow], axis: str) -> List[float]:
    values = []
    for row in rows:
        value = row.axes_raw.get(axis)
        if value is not None:
            values.append(float(value))
    return values


def _describe_group(rows: Sequence[MetricsRow], threshold: float) -> Dict[str, float]:
    scores = _collect_scores(rows)
    if not scores:
        return {
            "samples": 0,
            "pass_rate": float("nan"),
            "p50": float("nan"),
            "p75": float("nan"),
            "lt30": 0.0,
        }
    arr = np.asarray(scores, dtype=np.float64)
    pass_rate = float(np.mean(arr >= threshold))
    lt30 = float(np.sum(arr < 30.0))
    return {
        "samples": float(arr.size),
        "pass_rate": pass_rate,
        "p50": float(np.percentile(arr, 50.0)),
        "p75": float(np.percentile(arr, 75.0)),
        "lt30": lt30,
    }


def _print_summary(
    name: str,
    rows: Sequence[MetricsRow],
    threshold: float,
    tempo_edges: Sequence[float],
) -> None:
    scores = _collect_scores(rows)
    total = len(rows)
    print(f"=== {name} ===")
    print(f"samples: {total}")
    if scores:
        arr = np.asarray(scores, dtype=np.float64)
        pass_rate = np.mean(arr >= threshold)
        print(f"pass_rate (@{threshold:.1f}): {pass_rate*100:.1f}%")
        print(
            "p50/p75: " f"{_format_percentile(scores, 50.0)} / {_format_percentile(scores, 75.0)}",
        )
        print(f"<30 count: {int(np.sum(arr < 30.0))}")
    else:
        print("No valid scores found.")

    for axis in ("velocity", "structure"):
        axis_values = _axis_values(rows, axis)
        if axis_values:
            print(
                f"axis {axis}: p50={_format_percentile(axis_values, 50.0)} "
                f"p75={_format_percentile(axis_values, 75.0)}",
            )

    if tempo_edges:
        labels = _tempo_bin_labels(tempo_edges)
        print("Tempo buckets:")
        groups: List[List[MetricsRow]] = [list() for _ in labels]
        for row in rows:
            idx = _assign_tempo_bin(row.tempo_bpm, tempo_edges)
            groups[idx].append(row)
        for label, group in zip(labels, groups):
            stats = _describe_group(group, threshold)
            print(
                f"  {label:>10}: n={int(stats['samples']):4d} "
                f"pass={stats['pass_rate']*100:.1f}% "
                f"p50={stats['p50']:.2f} lt30={int(stats['lt30'])}",
            )
    print()


def _diff(a: Sequence[float], b: Sequence[float]) -> Optional[float]:
    if not a or not b:
        return None
    return float(np.percentile(b, 50.0) - np.percentile(a, 50.0))


def _print_delta(
    baseline: Sequence[MetricsRow],
    trial: Sequence[MetricsRow],
    threshold: float,
) -> None:
    base_scores = _collect_scores(baseline)
    trial_scores = _collect_scores(trial)
    if base_scores and trial_scores:
        delta_pass = np.mean(trial_scores >= threshold) - np.mean(base_scores >= threshold)
        print(
            "Δ pass_rate: " f"{delta_pass*100:.2f}% (trial - baseline)",
        )
        print(
            "Δ p50: " f"{_diff(base_scores, trial_scores) or 0.0:.2f}",
        )
    for axis in ("velocity", "structure"):
        base_axis = _axis_values(baseline, axis)
        trial_axis = _axis_values(trial, axis)
        if base_axis and trial_axis:
            delta = _diff(base_axis, trial_axis) or 0.0
            print(f"Δ axis {axis} p50: {delta:.2f}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare Stage2 metrics JSONL outputs (baseline vs trial).",
    )
    parser.add_argument("baseline", type=Path, help="Baseline metrics JSONL")
    parser.add_argument("trial", type=Path, help="Trial metrics JSONL")
    parser.add_argument(
        "--threshold",
        type=float,
        default=50.0,
        help="Pass/fail threshold for score comparisons (default: 50)",
    )
    parser.add_argument(
        "--tempo-bins",
        type=str,
        default="95,130",
        help=(
            "Comma-separated tempo boundaries (e.g. '95,130'). "
            "Provide empty string to disable tempo grouping."
        ),
    )
    args = parser.parse_args()

    tempo_edges: Tuple[float, ...]
    tempo_arg = args.tempo_bins.strip()
    if tempo_arg:
        tempo_edges = tuple(float(value) for value in tempo_arg.split(",") if value.strip())
    else:
        tempo_edges = ()

    baseline_rows = _load_metrics(args.baseline)
    trial_rows = _load_metrics(args.trial)

    print("Stage2 metrics comparison")
    print("==========================")
    print()
    _print_summary("Baseline", baseline_rows, args.threshold, tempo_edges)
    _print_summary("Trial", trial_rows, args.threshold, tempo_edges)
    _print_delta(baseline_rows, trial_rows, args.threshold)


if __name__ == "__main__":
    main()
