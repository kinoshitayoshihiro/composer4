from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, cast

import yaml

DEFAULT_OUTPUT_PATH = Path("configs/thresholds/articulation.auto.yaml")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calibrate articulation thresholds from loop summaries."
    )
    parser.add_argument("input", type=Path, help="Input JSONL file with loop summaries")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Destination YAML file (default: %(default)s)",
    )
    parser.add_argument(
        "--metrics",
        nargs="*",
        default=[
            "articulation.snare_ghost_rate",
            "articulation.snare_flam_rate",
            "articulation.detache_ratio",
            "articulation.pizzicato_ratio",
        ],
        help="Metric names to calibrate",
    )
    parser.add_argument(
        "--bins",
        nargs="*",
        type=float,
        default=[0, 90, 110, 130, 999],
        help="Tempo bin edges",
    )
    parser.add_argument(
        "--iqr-factor",
        type=float,
        default=0.25,
        help="Multiplier applied to IQR for high threshold",
    )
    parser.add_argument(
        "--drop-ratio",
        type=float,
        default=0.1,
        help="Multiple of IQR subtracted from median for hysteresis drop threshold",
    )
    parser.add_argument(
        "--min-support",
        type=int,
        default=8,
        help="Minimum observation count required per bin",
    )
    return parser.parse_args()


def load_records(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if isinstance(payload, dict):
                records.append(cast(Dict[str, Any], payload))
    return records


def tempo_bin_label(value: float, bins: List[float]) -> str:
    ordered = sorted(bins)
    for start, end in zip(ordered[:-1], ordered[1:]):
        if value < end:
            return f"{int(start)}-{int(end)}"
    return f"{int(ordered[-2])}-{int(ordered[-1])}"


def percentile(values: List[float], ratio: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = ratio * (len(ordered) - 1)
    lower = int(position)
    upper = min(len(ordered) - 1, lower + 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def collect_metric(
    values: Iterable[Dict[str, Any]], metrics: List[str], bins: List[float]
) -> Dict[str, Dict[str, List[float]]]:
    bucketed: Dict[str, Dict[str, List[float]]] = {}
    for metric in metrics:
        bucketed[metric] = defaultdict(list)

    for record in values:
        tempo = record.get("tempo_bpm") or record.get("tempo")
        tempo_value = _coerce_float(tempo)
        if tempo_value is None:
            continue
        label = tempo_bin_label(tempo_value, bins)
        metrics_dict = cast(Dict[str, Any], record.get("metrics", {}))
        for metric in metrics:
            metric_value = metrics_dict.get(metric) or metrics_dict.get(metric.split(".")[-1])
            value = _coerce_float(metric_value)
            if value is None:
                continue
            bucketed[metric][label].append(value)
    return bucketed


def calibrate(
    bucketed: Dict[str, Dict[str, List[float]]],
    iqr_factor: float,
    drop_ratio: float,
    min_support: int,
) -> Dict[str, Any]:
    calibrated: Dict[str, Any] = {}
    for metric, buckets in bucketed.items():
        metric_summary: Dict[str, Any] = {}
        for bin_label, values in buckets.items():
            if len(values) < min_support:
                continue
            q1 = percentile(values, 0.25)
            q2 = percentile(values, 0.5)
            q3 = percentile(values, 0.75)
            iqr = q3 - q1
            metric_summary[bin_label] = {
                "count": len(values),
                "q1": q1,
                "q2": q2,
                "q3": q3,
                "iqr": iqr,
                "high": q3 + iqr_factor * iqr,
                "drop": q2 - drop_ratio * iqr,
            }
        if metric_summary:
            calibrated[metric] = metric_summary
    return calibrated


def save_yaml(path: Path, data: Dict[str, Any], bins: List[float]) -> None:
    payload: Dict[str, Any] = {
        "mode": "auto",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "bins": {"tempo": bins},
        "metrics": data,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    dump = yaml.safe_dump(payload, sort_keys=True, allow_unicode=True)
    path.write_text(dump, encoding="utf-8")


def _coerce_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def main() -> None:
    args = parse_args()
    records = load_records(args.input)
    bucketed = collect_metric(records, args.metrics, args.bins)
    calibrated = calibrate(
        bucketed,
        args.iqr_factor,
        args.drop_ratio,
        args.min_support,
    )
    save_yaml(args.output, calibrated, args.bins)
    print(
        f"Wrote auto thresholds for {len(calibrated)} metrics to {args.output}",
    )


if __name__ == "__main__":
    main()
