#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import statistics as stats
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, cast

try:
    import yaml  # type: ignore
    from yaml import YAMLError  # type: ignore
except ImportError:
    yaml = None  # type: ignore

    class YAMLError(Exception):
        """Fallback error when PyYAML isn't installed."""


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            text = line.strip()
            if not text:
                continue
            rows.append(json.loads(text))
    return rows


def load_summary_bins(
    summary_path: Optional[Path],
    cli_bins: Optional[List[float]],
) -> List[float]:
    if cli_bins:
        return cli_bins
    if summary_path and summary_path.exists():
        data = json.loads(summary_path.read_text(encoding="utf-8"))
        try:
            tempo_bins = data["articulation"]["auto"]["bins"]["tempo"]
            return [float(value) for value in tempo_bins]
        except (KeyError, TypeError, ValueError):
            pass
    return [0.0, 90.0, 110.0, 130.0, 999.0]


def bin_index(value: float, bins: List[float]) -> int:
    for index in range(len(bins) - 1):
        if bins[index] <= value < bins[index + 1]:
            return index
    return max(0, len(bins) - 2)


def iqr(values: List[float]) -> Tuple[float, float, float, float]:
    if not values:
        return (math.nan, math.nan, math.nan, math.nan)
    quartiles = stats.quantiles(values, n=4, method="inclusive")
    q1 = quartiles[0]
    q3 = quartiles[2]
    q2 = stats.median(values)
    spread = q3 - q1
    return (q1, q2, q3, spread)


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def load_previous_thresholds(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    raw = path.read_text(encoding="utf-8")
    if yaml is not None:
        try:
            loaded = yaml.safe_load(raw)
            if isinstance(loaded, dict):
                return cast(Dict[str, Any], loaded)
        except YAMLError:
            pass
    try:
        loaded_json = json.loads(raw)
        if isinstance(loaded_json, dict):
            return cast(Dict[str, Any], loaded_json)
    except (json.JSONDecodeError, TypeError):
        pass
    return {}


def dump_thresholds(data: Dict[str, Any], path: Path, dry_run: bool) -> None:
    if dry_run:
        sys.stdout.write("---- DRY RUN THRESHOLDS ----\n")
        if yaml is not None:
            sys.stdout.write(yaml.safe_dump(data, allow_unicode=True, sort_keys=False))
        else:
            sys.stdout.write(json.dumps(data, ensure_ascii=False, indent=2))
        sys.stdout.write("\n----------------------------\n")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    if yaml is not None:
        text = yaml.safe_dump(data, allow_unicode=True, sort_keys=False)
    else:
        text = json.dumps(data, ensure_ascii=False, indent=2)
    path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="IQR-based auto calibration for articulation thresholds."
    )
    parser.add_argument(
        "--in",
        dest="in_path",
        required=True,
        help="metrics_score.jsonl path",
    )
    parser.add_argument(
        "--summary",
        dest="summary_path",
        default=None,
        help="stage2_summary.json path",
    )
    parser.add_argument(
        "--tempo-bins",
        dest="tempo_bins",
        default=None,
        help="comma separated tempo bins",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="high = q3 + alpha * IQR",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.0,
        help="low  = q1 - beta * IQR",
    )
    parser.add_argument(
        "--min-support",
        dest="min_support",
        type=int,
        default=30,
        help="minimum samples per bin",
    )
    parser.add_argument(
        "--hysteresis",
        type=float,
        default=0.01,
        help="skip update if |new-old| < hysteresis",
    )
    parser.add_argument(
        "--out",
        dest="out_path",
        required=True,
        help="output thresholds path",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print thresholds without writing",
    )
    args = parser.parse_args()

    input_path = Path(args.in_path)
    summary_path = Path(args.summary_path) if args.summary_path else None
    cli_bins = None
    if args.tempo_bins:
        cli_bins = [float(item) for item in args.tempo_bins.split(",") if item.strip()]
    bins = load_summary_bins(summary_path, cli_bins)

    rows = load_jsonl(input_path)
    per_bin_axis: Dict[int, Dict[str, List[float]]] = {}
    axes_seen: Set[str] = set()
    for row in rows:
        tempo = float(row.get("tempo", 120.0))
        bucket = bin_index(tempo, bins)
        axes_raw = cast(Dict[str, Any], row.get("axes_raw") or {})
        if not axes_raw:
            continue
        bin_map = per_bin_axis.setdefault(bucket, {})
        for raw_name, value in axes_raw.items():
            axis_name = str(raw_name)
            try:
                float_value = float(value)
            except (TypeError, ValueError):
                continue
            axes_seen.add(axis_name)
            bin_map.setdefault(axis_name, []).append(float_value)

    previous_thresholds = load_previous_thresholds(Path(args.out_path))
    prev_per_axis_raw = previous_thresholds.get("per_axis")
    prev_per_axis = (
        cast(Dict[str, Any], prev_per_axis_raw) if isinstance(prev_per_axis_raw, dict) else {}
    )

    output: Dict[str, Any] = {
        "mode": "auto",
        "bins": {"tempo": bins},
        "per_axis": {},
    }

    updated = 0
    for axis_name in sorted(axes_seen):
        axis_info: Dict[str, Any] = {"bins": []}
        for bucket_index in range(len(bins) - 1):
            samples = per_bin_axis.get(bucket_index, {}).get(axis_name, [])
            sample_count = len(samples)
            if sample_count >= args.min_support:
                q1, q2, q3, spread = iqr(samples)
                if math.isnan(q1):
                    low_value: Optional[float] = None
                    high_value: Optional[float] = None
                else:
                    low_value = clamp01(q1 - args.beta * spread)
                    high_value = clamp01(q3 + args.alpha * spread)
                    if low_value > high_value:
                        low_value, high_value = high_value, low_value
            else:
                prev_axis = prev_per_axis.get(axis_name)
                prev_axis_dict = (
                    cast(Dict[str, Any], prev_axis) if isinstance(prev_axis, dict) else {}
                )
                prev_bins_raw = prev_axis_dict.get("bins")
                prev_bins = (
                    cast(List[Dict[str, Any]], prev_bins_raw)
                    if isinstance(prev_bins_raw, list)
                    else []
                )
                if bucket_index < len(prev_bins):
                    prev_entry_dict = prev_bins[bucket_index]
                else:
                    prev_entry_dict = {}
                low_value = cast(Optional[float], prev_entry_dict.get("low"))
                high_value = cast(Optional[float], prev_entry_dict.get("high"))
                q1 = q2 = q3 = spread = math.nan

            prev_low: Optional[float] = None
            prev_high: Optional[float] = None
            try:
                prev_low = cast(
                    Optional[float],
                    prev_per_axis[axis_name]["bins"][bucket_index]["low"],
                )
                prev_high = cast(
                    Optional[float],
                    prev_per_axis[axis_name]["bins"][bucket_index]["high"],
                )
            except (KeyError, IndexError, TypeError):
                pass

            def hysteresis_apply(
                new_value: Optional[float],
                old_value: Optional[float],
            ) -> Optional[float]:
                if new_value is None or old_value is None:
                    return new_value
                if abs(new_value - old_value) < args.hysteresis:
                    return old_value
                return new_value

            low_value = hysteresis_apply(low_value, prev_low)
            high_value = hysteresis_apply(high_value, prev_high)
            if low_value != prev_low or high_value != prev_high:
                updated += 1

            axis_info["bins"].append(
                {
                    "bin": [bins[bucket_index], bins[bucket_index + 1]],
                    "count": sample_count,
                    "q1q2q3I": None if math.isnan(q1) else [q1, q2, q3, spread],
                    "low": low_value,
                    "high": high_value,
                }
            )
        output["per_axis"][axis_name] = axis_info

    dump_thresholds(output, Path(args.out_path), args.dry_run)
    print(
        f"[calibrate] axes={len(axes_seen)} bins={len(bins) - 1} "
        f"updates={updated} -> {args.out_path}"
    )


if __name__ == "__main__":
    main()
