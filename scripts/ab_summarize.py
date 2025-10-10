#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A/B 集計レポート自動生成（Markdown）。"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple, cast


@dataclass
class RunArtifacts:
    metrics_path: Path
    summary_path: Optional[Path]


def _safe_get(
    payload: Mapping[str, Any],
    path: str,
    default: Any = None,
) -> Any:
    current: Any = payload
    for part in path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return default
        current = cast(Any, current[part])
    return current


def _quantiles(values: List[float]) -> Dict[str, float]:
    if not values:
        return {
            "min": math.nan,
            "p50": math.nan,
            "p90": math.nan,
            "max": math.nan,
            "mean": math.nan,
        }
    ordered = sorted(values)
    if len(ordered) == 1:
        unique = ordered[0]
        return {
            "min": unique,
            "p50": unique,
            "p90": unique,
            "max": unique,
            "mean": unique,
        }

    def percentile(percent: float) -> float:
        if percent <= 0:
            return ordered[0]
        if percent >= 100:
            return ordered[-1]
        position = (len(ordered) - 1) * (percent / 100.0)
        lower = math.floor(position)
        upper = math.ceil(position)
        if lower == upper:
            return ordered[int(position)]
        weight = position - lower
        return ordered[lower] * (1 - weight) + ordered[upper] * weight

    avg = sum(ordered) / len(ordered)
    return {
        "min": ordered[0],
        "p50": percentile(50.0),
        "p90": percentile(90.0),
        "max": ordered[-1],
        "mean": avg,
    }


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            text = line.strip()
            if not text:
                continue
            try:
                payload = json.loads(text)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                rows.append(cast(Dict[str, Any], payload))
    return rows


def _resolve_run_paths(root: Path) -> RunArtifacts:
    if not root.exists():
        raise FileNotFoundError(
            f"Path does not exist: {root}\n"
            f"Please provide a valid directory containing metrics_score.jsonl "
            f"or a direct path to the JSONL file."
        )
    if root.is_dir():
        metrics_candidates = list(root.glob("**/metrics_score.jsonl"))
        summary_candidates = list(root.glob("**/stage2_summary.json"))
        if metrics_candidates:
            metrics_path = metrics_candidates[0]
        else:
            metrics_path = root / "metrics_score.jsonl"
        summary_path: Optional[Path]
        if summary_candidates:
            summary_path = summary_candidates[0]
        else:
            direct = root / "stage2_summary.json"
            summary_path = direct if direct.exists() else None
    else:
        metrics_path = root
        sibling = root.parent / "stage2_summary.json"
        summary_path = sibling if sibling.exists() else None
    return RunArtifacts(metrics_path=metrics_path, summary_path=summary_path)


def _collect_stats(
    run: RunArtifacts,
    *,
    threshold: float,
) -> Dict[str, Any]:
    rows = _load_jsonl(run.metrics_path)
    scores = [
        float(value)
        for value in (_safe_get(row, "score", math.nan) for row in rows)
        if isinstance(value, (int, float))
    ]
    axes_velocity = [
        float(value)
        for value in (_safe_get(row, "axes_raw.velocity", math.nan) for row in rows)
        if isinstance(value, (int, float))
    ]
    axes_structure = [
        float(value)
        for value in (_safe_get(row, "axes_raw.structure", math.nan) for row in rows)
        if isinstance(value, (int, float))
    ]
    axes_timing = [
        float(value)
        for value in (_safe_get(row, "axes_raw.timing", math.nan) for row in rows)
        if isinstance(value, (int, float))
    ]

    passed = sum(1 for score in scores if score >= threshold)
    total = len(scores)
    pass_rate_est = (passed / total) if total else math.nan
    low_score_lt30 = sum(1 for score in scores if not math.isnan(score) and score < 30.0)

    audio_total = 0
    adaptive_hits = 0
    rule_counts: Dict[str, int] = {}
    for row in rows:
        audio_payload = row.get("audio")
        if not isinstance(audio_payload, dict):
            continue
        audio_total += 1
        audio_map = cast(Dict[str, Any], audio_payload)
        applied = _safe_get(audio_map, "adaptive.applied")
        rule_name = _safe_get(audio_map, "adaptive.rule")
        if not applied:
            details = row.get("audio.adaptive_details") or row.get("audio_adaptive_details")
            if isinstance(details, dict):
                details_map = cast(Dict[str, Any], details)
                applied = details_map.get("applied")
                rule_name = details_map.get("rule_name") or details_map.get("rule") or rule_name
        if applied:
            adaptive_hits += 1
            if isinstance(rule_name, str):
                rule_counts[rule_name] = rule_counts.get(rule_name, 0) + 1

    stats: Dict[str, Any] = {
        "n": total,
        "score": _quantiles([s for s in scores if not math.isnan(s)]),
        "pass_rate_est": pass_rate_est,
        "low_score_lt30": low_score_lt30,
        "axes_raw": {
            "timing.p50": statistics.median(axes_timing) if axes_timing else math.nan,
            "velocity.p50": statistics.median(axes_velocity) if axes_velocity else math.nan,
            "structure.p50": statistics.median(axes_structure) if axes_structure else math.nan,
        },
        "audio": {
            "has_audio": audio_total,
            "adaptive_applied": adaptive_hits,
            "adaptive_rate": (adaptive_hits / audio_total) if audio_total else 0.0,
            "rules_top": sorted(
                rule_counts.items(),
                key=lambda item: item[1],
                reverse=True,
            )[:5],
        },
    }

    if run.summary_path is not None and run.summary_path.exists():
        try:
            summary_payload = json.loads(run.summary_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            summary_payload = None
        if isinstance(summary_payload, dict):
            summary_map = cast(Dict[str, Any], summary_payload)
            stats["summary"] = {
                "pass_rate": _safe_get(summary_map, "outputs.pass_rate"),
                "passed_loops": _safe_get(
                    summary_map,
                    "outputs.passed_loops",
                ),
                "score_distribution": _safe_get(
                    summary_map,
                    "score_distribution",
                ),
            }

    return stats


def _format_delta(base: float, challenger: float) -> str:
    if any(math.isnan(x) for x in (base, challenger)):
        return "—"
    delta = challenger - base
    sign = "+" if delta >= 0 else ""
    return f"{challenger:.2f} ({sign}{delta:.2f})"


def _format_rules(rules: Iterable[Tuple[str, int]]) -> str:
    parts = [f"{name}×{count}" for name, count in rules]
    return ", ".join(parts) if parts else "—"


def _append_row(
    lines: List[str],
    label: str,
    value_a: str,
    value_b: str,
) -> None:
    lines.append(f"| {label} | {value_a} | {value_b} |")


def _write_report(
    report_path: Optional[Path],
    *,
    body: str,
) -> None:
    if report_path is None or report_path == Path("-"):
        print(body)
        return
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(body, encoding="utf-8")
    print(f"Wrote: {report_path}")


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--a",
        required=True,
        help="Run A (dir or metrics JSONL)",
    )
    parser.add_argument(
        "--b",
        required=True,
        help="Run B (dir or metrics JSONL)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=50.0,
        help="Pass threshold used for pass-rate estimate",
    )
    parser.add_argument(
        "--out",
        default="-",
        help="Output Markdown path (default: stdout)",
    )
    args = parser.parse_args(argv)

    run_a = _resolve_run_paths(Path(args.a))
    run_b = _resolve_run_paths(Path(args.b))

    stats_a = _collect_stats(run_a, threshold=args.threshold)
    stats_b = _collect_stats(run_b, threshold=args.threshold)

    lines: List[str] = []
    lines.append("# Stage2 A/B 比較レポート")
    lines.append("")
    lines.append(f"- 閾値(推定): **{args.threshold:.2f}**")
    lines.append(f"- A: `{run_a.metrics_path}`")
    lines.append(f"- B: `{run_b.metrics_path}`")
    lines.append("")

    lines.append("## 1) スコア概況")
    lines.append("| 指標 | A | B (Δ) |")
    lines.append("|---|---:|---:|")
    delta_pass = _format_delta(
        stats_a["pass_rate_est"],
        stats_b["pass_rate_est"],
    )
    value_pass = f"{stats_a['pass_rate_est']:.4f}"
    _append_row(lines, "pass_rate(推定)", value_pass, delta_pass)
    for label in ("min", "p50", "p90", "max", "mean"):
        delta_value = _format_delta(
            stats_a["score"][label],
            stats_b["score"][label],
        )
        value_a = f"{stats_a['score'][label]:.2f}"
        _append_row(lines, f"score.{label}", value_a, delta_value)
    diff_low = stats_b["low_score_lt30"] - stats_a["low_score_lt30"]
    low_value = f"{stats_b['low_score_lt30']} ({diff_low:+d})"
    _append_row(
        lines,
        "low(<30)件数",
        str(stats_a["low_score_lt30"]),
        low_value,
    )

    lines.append("")
    lines.append("## 2) 主要軸（中央値）")
    lines.append("| 軸 | A | B (Δ) |")
    lines.append("|---|---:|---:|")
    for axis in ("timing", "velocity", "structure"):
        key = f"{axis}.p50"
        delta_axis = _format_delta(
            stats_a["axes_raw"][key],
            stats_b["axes_raw"][key],
        )
        value_axis = f"{stats_a['axes_raw'][key]:.3f}"
        _append_row(lines, axis, value_axis, delta_axis)

    lines.append("")
    lines.append("## 3) 音声適応の稼働状況")
    lines.append("| 指標 | A | B (Δ) |")
    lines.append("|---|---:|---:|")
    audio_count_delta = stats_b["audio"]["has_audio"] - stats_a["audio"]["has_audio"]
    audio_count_value = f"{stats_b['audio']['has_audio']} ({audio_count_delta:+d})"
    _append_row(
        lines,
        "audio付き件数",
        str(stats_a["audio"]["has_audio"]),
        audio_count_value,
    )
    audio_applied_delta = (
        stats_b["audio"]["adaptive_applied"] - stats_a["audio"]["adaptive_applied"]
    )
    audio_applied_value = f"{stats_b['audio']['adaptive_applied']} ({audio_applied_delta:+d})"
    _append_row(
        lines,
        "適応発動件数",
        str(stats_a["audio"]["adaptive_applied"]),
        audio_applied_value,
    )
    audio_rate_delta = _format_delta(
        stats_a["audio"]["adaptive_rate"],
        stats_b["audio"]["adaptive_rate"],
    )
    audio_rate_value = f"{stats_a['audio']['adaptive_rate']:.3f}"
    _append_row(lines, "適応発動率", audio_rate_value, audio_rate_delta)
    lines.append("")
    lines.append(f"- A ルール上位: {_format_rules(stats_a['audio']['rules_top'])}")
    lines.append(f"- B ルール上位: {_format_rules(stats_b['audio']['rules_top'])}")

    if "summary" in stats_a or "summary" in stats_b:
        lines.append("")
        lines.append("## 4) Stage2 summary")
        tag_summary_pairs = (
            ("A", stats_a.get("summary")),
            ("B", stats_b.get("summary")),
        )
        for tag, summary in tag_summary_pairs:
            if not summary:
                continue
            lines.append(
                f"- **{tag}** pass_rate: {summary.get('pass_rate')} / "
                f"passed_loops: {summary.get('passed_loops')}"
            )

    report_body = "\n".join(lines) + "\n"
    report_path = None if args.out in {None, "-"} else Path(args.out)
    _write_report(report_path, body=report_body)


if __name__ == "__main__":
    main()
