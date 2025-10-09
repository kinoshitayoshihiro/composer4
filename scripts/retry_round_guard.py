#!/usr/bin/env python3
"""Evaluate retry round effectiveness for Stage2 metrics."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, cast


@dataclass
class LoopMetrics:
    loop_id: str
    score: float
    axes_raw: Dict[str, float]
    retry_round: Optional[int]


def _load_metrics(path: Path) -> Dict[str, LoopMetrics]:
    loops: Dict[str, LoopMetrics] = {}
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            row = json.loads(line)
            loop_id = str(row.get("loop_id"))
            if not loop_id:
                continue
            score_obj = row.get("score", 0.0)
            try:
                score = float(score_obj)
            except (TypeError, ValueError):
                score = 0.0
            axes_raw_obj = row.get("axes_raw")
            axes_raw: Dict[str, float] = {}
            if isinstance(axes_raw_obj, dict):
                axes_raw_map = cast(Dict[Any, Any], axes_raw_obj)
                for key, value in axes_raw_map.items():
                    numeric: Optional[float] = None
                    if isinstance(value, (int, float)):
                        numeric = float(value)
                    elif isinstance(value, str):
                        text = value.strip()
                        if text:
                            try:
                                numeric = float(text)
                            except ValueError:
                                numeric = None
                    if numeric is not None:
                        axes_raw[str(key)] = float(numeric)
            retry_round_obj = row.get("_retry_round")
            retry_round: Optional[int]
            if isinstance(retry_round_obj, (int, float)):
                retry_round = int(retry_round_obj)
            elif isinstance(retry_round_obj, str):
                retry_round = int(retry_round_obj.strip() or 0)
            else:
                retry_round = None
            loops[loop_id] = LoopMetrics(
                loop_id=loop_id,
                score=score,
                axes_raw=axes_raw,
                retry_round=retry_round,
            )
    return loops


def _pick_targets(
    after: Dict[str, LoopMetrics],
    ops: Optional[Dict[str, LoopMetrics]],
) -> Iterable[str]:
    if ops:
        for loop_id, metrics in ops.items():
            if metrics.retry_round and metrics.retry_round > 0:
                yield loop_id
    else:
        yield from after.keys()


def _quantiles(values: Sequence[float]) -> Dict[str, float]:
    if not values:
        return {"p50": 0.0, "p75": 0.0, "p90": 0.0}
    sorted_vals = sorted(values)
    p50 = statistics.median(sorted_vals)
    p75 = sorted_vals[int(len(sorted_vals) * 0.75)]
    p90 = sorted_vals[int(len(sorted_vals) * 0.9)]
    return {"p50": p50, "p75": p75, "p90": p90}


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Check retry improvement against minimum thresholds.",
    )
    parser.add_argument(
        "--before",
        required=True,
        help="Baseline metrics JSONL path",
    )
    parser.add_argument(
        "--after",
        required=True,
        help="Post-retry metrics JSONL path",
    )
    parser.add_argument(
        "--ops",
        help=("Retry annotation JSONL (output of retry_apply) " "to scope target loops"),
    )
    parser.add_argument(
        "--axis",
        action="append",
        default=["velocity", "structure"],
        help="Axes to evaluate (default: velocity, structure)",
    )
    parser.add_argument(
        "--max-regression",
        type=float,
        default=None,
        help=(
            "Maximum allowed negative delta (if provided, values smaller than "
            "-max-regression trigger a failure)"
        ),
    )
    parser.add_argument(
        "--min-improvement",
        type=float,
        default=0.05,
        help="Minimum required delta for each axis (axes_raw units)",
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=5.0,
        help="Minimum Stage2 score delta expected",
    )
    parser.add_argument(
        "--report",
        help="Optional JSON report path",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with status 1 if any axis fails the improvement threshold",
    )

    args = parser.parse_args(argv)

    before = _load_metrics(Path(args.before))
    after = _load_metrics(Path(args.after))
    ops = _load_metrics(Path(args.ops)) if args.ops else None

    targets = list(_pick_targets(after, ops))
    missing = [loop_id for loop_id in targets if loop_id not in before]
    if missing:
        sys.stderr.write(
            (
                f"Warning: {len(missing)} loops not found in baseline "
                "metrics; they will be skipped.\n"
            ),
        )
    axes = [axis.strip() for axis in args.axis if axis.strip()]

    summary: Dict[str, Dict[str, float]] = {}
    failures: Dict[str, List[str]] = {}
    reasons: List[Dict[str, Any]] = []

    for axis in axes:
        deltas: List[float] = []
        failing: List[str] = []
        for loop_id in targets:
            base = before.get(loop_id)
            updated = after.get(loop_id)
            if not base or not updated:
                continue
            delta = updated.axes_raw.get(axis, 0.0) - base.axes_raw.get(axis, 0.0)
            deltas.append(delta)
            reason: Optional[str] = None
            if args.max_regression is not None and delta < -abs(args.max_regression):
                reason = "max_regression"
            elif delta < args.min_improvement:
                reason = "min_improvement_not_met"

            if reason:
                failing.append(loop_id)
                reasons.append(
                    {
                        "loop_id": loop_id,
                        "axis": axis,
                        "reason": reason,
                        "delta": float(delta),
                    },
                )
        quant = _quantiles(deltas)
        summary[axis] = {
            "count": float(len(deltas)),
            "mean": float(sum(deltas) / len(deltas)) if deltas else 0.0,
            "min": float(min(deltas)) if deltas else 0.0,
            "max": float(max(deltas)) if deltas else 0.0,
            "failure_count": float(len(failing)),
            **quant,
        }
        failures[axis] = failing

    combined_failures: List[str] = []
    for axis in axes:
        combined_failures.extend(failures.get(axis, []))
    if combined_failures:
        unique_combined = sorted(set(combined_failures))
    else:
        unique_combined = []
    summary["combined"] = {"failure_count": float(len(unique_combined))}

    score_deltas: List[float] = []
    score_failures: List[str] = []
    for loop_id in targets:
        base = before.get(loop_id)
        updated = after.get(loop_id)
        if not base or not updated:
            continue
        delta = updated.score - base.score
        score_deltas.append(delta)
        if args.score_threshold is not None and delta < args.score_threshold:
            score_failures.append(loop_id)
            reasons.append(
                {
                    "loop_id": loop_id,
                    "axis": "<score>",
                    "reason": "score_threshold",
                    "delta": float(delta),
                },
            )
    summary["score"] = {
        "count": float(len(score_deltas)),
        "mean": (float(sum(score_deltas) / len(score_deltas)) if score_deltas else 0.0),
        "min": float(min(score_deltas)) if score_deltas else 0.0,
        "max": float(max(score_deltas)) if score_deltas else 0.0,
        "failure_count": float(len(score_failures)),
        **_quantiles(score_deltas),
    }
    failures["score"] = score_failures

    if reasons:
        reason_counts: Dict[str, int] = {}
        for entry in reasons:
            key = entry.get("reason", "unknown")
            reason_counts[key] = reason_counts.get(key, 0) + 1
        summary["reasons_count"] = {name: float(count) for name, count in reason_counts.items()}

    payload: Dict[str, Any] = {
        "axes": summary,
        "failures": failures,
        "reasons": reasons,
        "score_threshold": args.score_threshold,
        "min_improvement": args.min_improvement,
    }

    if args.report:
        Path(args.report).write_text(
            json.dumps(payload, indent=2),
            encoding="utf-8",
        )

    json.dump(payload, sys.stdout, indent=2)
    sys.stdout.write("\n")

    if args.strict:
        axis_fail = any(failures.get(axis) for axis in axes)
        score_fail = bool(score_failures)
        if axis_fail or score_fail:
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
