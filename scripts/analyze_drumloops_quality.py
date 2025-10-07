#!/usr/bin/env python3
"""Quality inspection utilities for the deduplicated drum loops.

The script samples MIDI files from ``output/drumloops_cleaned`` (by default),
parses them via TMIDIX, and reports basic statistics alongside high-level
quality heuristics (length, velocity dynamics, density, etc.).

Usage examples
--------------

Inspect 200 random loops and print a concise report::

    ./composer2/bin/python scripts/analyze_drumloops_quality.py --sample 200

Generate a JSON report for the entire cleaned dataset::

    ./composer2/bin/python scripts/analyze_drumloops_quality.py --sample -1 \
        --report-path reports/drumloops_quality_report.json

"""
# pylint: disable=broad-except
from __future__ import annotations

import argparse
import importlib
import json
import math
import random
import statistics
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, cast

# -------------------------------------------------------------------------------------
# TMIDIX bootstrap
# -------------------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TMIDIX_PATH = PROJECT_ROOT / "data" / "Los-Angeles-MIDI" / "CODE"

if not TMIDIX_PATH.exists():
    raise SystemExit("TMIDIX module not found. Expected it at data/Los-Angeles-MIDI/CODE.")

sys.path.append(str(TMIDIX_PATH))

try:  # pragma: no cover - import validation happens at runtime
    TMIDIX = importlib.import_module("TMIDIX")
except ModuleNotFoundError as exc:  # pragma: no cover - import guard
    raise SystemExit(f"Failed to import TMIDIX: {exc}") from exc


def _list_str_factory() -> list[str]:
    return []


@dataclass
class DrumLoopMetrics:
    path: Path
    status: str
    reason: Optional[str] = None
    ticks_per_beat: Optional[int] = None
    note_count: Optional[int] = None
    unique_pitches: Optional[int] = None
    unique_channels: Optional[int] = None
    dominant_channel: Optional[int] = None
    min_velocity: Optional[int] = None
    max_velocity: Optional[int] = None
    avg_velocity: Optional[float] = None
    velocity_std: Optional[float] = None
    loop_length_ticks: Optional[int] = None
    loop_length_beats: Optional[float] = None
    tempo_bpm: Optional[float] = None
    density_notes_per_beat: Optional[float] = None
    polyphony_ratio: Optional[float] = None
    high_velocity_ratio: Optional[float] = None
    low_velocity_ratio: Optional[float] = None
    issues: list[str] = field(default_factory=_list_str_factory)

    def to_summary(self) -> Dict[str, object]:
        return {
            "path": str(self.path),
            "status": self.status,
            "reason": self.reason,
            "ticks_per_beat": self.ticks_per_beat,
            "note_count": self.note_count,
            "unique_pitches": self.unique_pitches,
            "unique_channels": self.unique_channels,
            "dominant_channel": self.dominant_channel,
            "min_velocity": self.min_velocity,
            "max_velocity": self.max_velocity,
            "avg_velocity": self.avg_velocity,
            "velocity_std": self.velocity_std,
            "loop_length_ticks": self.loop_length_ticks,
            "loop_length_beats": self.loop_length_beats,
            "tempo_bpm": self.tempo_bpm,
            "density_notes_per_beat": self.density_notes_per_beat,
            "polyphony_ratio": self.polyphony_ratio,
            "high_velocity_ratio": self.high_velocity_ratio,
            "low_velocity_ratio": self.low_velocity_ratio,
            "issues": self.issues,
        }


def list_midi_files(root: Path) -> List[Path]:
    midi_files = [p for p in root.rglob("*.mid") if p.is_file()]
    midi_files.extend(p for p in root.rglob("*.midi") if p.is_file())
    return sorted(set(midi_files))


def percentile(values: Iterable[float], pct: float) -> float:
    data = sorted(values)
    if not data:
        raise ValueError("Empty values")
    if len(data) == 1:
        return data[0]
    pct = max(0.0, min(100.0, pct))
    k = (len(data) - 1) * (pct / 100.0)
    lower = math.floor(k)
    upper = math.ceil(k)
    if lower == upper:
        return data[int(k)]
    weight = k - lower
    return data[lower] * (1.0 - weight) + data[upper] * weight


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze drum loop quality metrics.")
    parser.add_argument(
        "--root",
        type=Path,
        default=PROJECT_ROOT / "output" / "drumloops_cleaned",
        help="Root directory containing deduplicated drum loops.",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=200,
        help=(
            "Number of files to sample. Use a negative value to analyze "
            "all files. Defaults to 200."
        ),
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling order.")
    parser.add_argument(
        "--report-path",
        type=Path,
        default=PROJECT_ROOT / "reports" / "drumloops_quality_report.json",
        help=("Optional path to write a JSON report. Directory will be created" " if needed."),
    )
    parser.add_argument(
        "--max-length-beats",
        type=float,
        default=64.0,
        help="Flag loops whose length exceeds this number of beats.",
    )
    parser.add_argument(
        "--min-length-beats",
        type=float,
        default=1.0,
        help="Flag loops shorter than this number of beats.",
    )
    parser.add_argument(
        "--max-density",
        type=float,
        default=32.0,
        help=("Flag loops with note density (notes per beat) above this " "threshold."),
    )
    parser.add_argument(
        "--min-velocity-std",
        type=float,
        default=3.0,
        help=("Flag loops whose velocity standard deviation is below this " "threshold."),
    )
    parser.add_argument(
        "--avg-velocity-soft-threshold",
        type=float,
        default=20.0,
        help="Average velocity below which a loop is considered too soft.",
    )
    parser.add_argument(
        "--avg-velocity-loud-threshold",
        type=float,
        default=110.0,
        help="Average velocity above which a loop is considered too loud.",
    )
    parser.add_argument(
        "--max-high-velocity-ratio",
        type=float,
        default=0.5,
        help=("Flag loops if the share of velocities >= 125 exceeds this " "ratio."),
    )
    parser.add_argument(
        "--min-low-velocity-ratio",
        type=float,
        default=0.3,
        help=("Flag loops if the share of velocities <= 10 exceeds this " "ratio."),
    )
    parser.add_argument(
        "--show-issues",
        type=int,
        default=15,
        help="Number of problematic files to list in the console output.",
    )
    parser.add_argument(
        "--no-report",
        action="store_true",
        help="Skip writing the JSON report even if --report-path is provided.",
    )
    return parser.parse_args()


def analyze_file(path: Path, thresholds: argparse.Namespace) -> DrumLoopMetrics:
    try:
        fdata = path.read_bytes()
    except OSError as exc:
        return DrumLoopMetrics(path=path, status="error", reason=str(exc))

    try:
        score = cast(List[Any], TMIDIX.midi2score(fdata))  # type: ignore[attr-defined]
    except Exception as exc:  # noqa: BLE001
        return DrumLoopMetrics(path=path, status="error", reason=str(exc))

    if not score:
        return DrumLoopMetrics(path=path, status="error", reason="Empty TMIDIX score")

    ticks_per_beat = int(score[0])
    events: List[List[Any]] = [
        event for track in score[1:] for event in track if isinstance(event, list)
    ]
    notes: List[List[Any]] = [event for event in events if event and event[0] == "note"]

    if not notes:
        return DrumLoopMetrics(path=path, status="empty", reason="No note events found")

    tempos: List[int] = [
        event[2] for event in events if event and event[0] == "set_tempo" and len(event) >= 3
    ]
    tempo_bpm: Optional[float] = None
    if tempos:
        avg_microseconds = sum(tempos) / len(tempos)
        if avg_microseconds > 0:
            tempo_bpm = 60_000_000 / avg_microseconds

    times: List[int] = [int(note[1]) for note in notes]
    end_times: List[int] = [int(note[1] + note[2]) for note in notes]
    loop_length_ticks = max(end_times)
    loop_length_beats = loop_length_ticks / ticks_per_beat if ticks_per_beat else None

    velocities: List[int] = [int(note[5]) for note in notes]
    velocity_std = 0.0
    if len(velocities) > 1:
        try:
            velocity_std = statistics.pstdev(velocities)
        except statistics.StatisticsError:
            velocity_std = 0.0

    channels: List[int] = [int(note[3]) for note in notes]
    channel_counts = Counter(channels)
    dominant_channel, _ = channel_counts.most_common(1)[0]

    density = None
    if loop_length_beats and loop_length_beats > 0:
        density = len(notes) / loop_length_beats

    polyphony_ratio = None
    if times:
        simultaneous_counts = Counter(times)
        overlaps = sum(count for count in simultaneous_counts.values() if count > 1)
        polyphony_ratio = overlaps / len(times)

    high_velocity_ratio = sum(1 for v in velocities if v >= 125) / len(velocities)
    low_velocity_ratio = sum(1 for v in velocities if v <= 10) / len(velocities)

    metrics = DrumLoopMetrics(
        path=path,
        status="ok",
        ticks_per_beat=ticks_per_beat,
        note_count=len(notes),
        unique_pitches=len(set(note[4] for note in notes)),
        unique_channels=len(channel_counts),
        dominant_channel=dominant_channel,
        min_velocity=min(velocities),
        max_velocity=max(velocities),
        avg_velocity=sum(velocities) / len(velocities),
        velocity_std=velocity_std,
        loop_length_ticks=loop_length_ticks,
        loop_length_beats=loop_length_beats,
        tempo_bpm=tempo_bpm,
        density_notes_per_beat=density,
        polyphony_ratio=polyphony_ratio,
        high_velocity_ratio=high_velocity_ratio,
        low_velocity_ratio=low_velocity_ratio,
    )

    issues: List[str] = []
    if loop_length_beats is not None:
        if loop_length_beats > thresholds.max_length_beats:
            issues.append("too_long")
        if loop_length_beats < thresholds.min_length_beats:
            issues.append("too_short")
    if density is not None and density > thresholds.max_density:
        issues.append("too_dense")
    if velocity_std < thresholds.min_velocity_std:
        issues.append("flat_velocity")
    if (
        metrics.avg_velocity is not None
        and metrics.avg_velocity >= thresholds.avg_velocity_loud_threshold
        and high_velocity_ratio >= thresholds.max_high_velocity_ratio
    ):
        issues.append("very_loud")
    if (
        metrics.avg_velocity is not None
        and metrics.avg_velocity <= thresholds.avg_velocity_soft_threshold
        and low_velocity_ratio >= thresholds.min_low_velocity_ratio
    ):
        issues.append("very_soft")
    if (
        metrics.unique_channels
        and metrics.dominant_channel is not None
        and metrics.dominant_channel != 9
    ):
        issues.append("non_drum_channel")

    metrics.issues = issues
    if issues:
        metrics.status = "review"

    return metrics


def summarize_metrics(metrics: Iterable[DrumLoopMetrics]) -> Dict[str, object]:
    collected = list(metrics)
    ok_items = [m for m in collected if m.status == "ok"]
    review_items = [m for m in collected if m.status == "review"]
    problem_items = [m for m in collected if m.status not in {"ok", "review"}]

    def collect_values(name: str) -> List[float]:
        values: List[float] = []
        for item in ok_items + review_items:
            value = getattr(item, name)
            if value is not None:
                values.append(float(value))
        return values

    stats = {}
    for name in [
        "note_count",
        "unique_pitches",
        "avg_velocity",
        "velocity_std",
        "loop_length_beats",
        "density_notes_per_beat",
        "polyphony_ratio",
        "high_velocity_ratio",
        "low_velocity_ratio",
    ]:
        values = collect_values(name)
        if values:
            stats[name] = {
                "min": min(values),
                "p5": percentile(values, 5),
                "median": statistics.median(values),
                "p95": percentile(values, 95),
                "max": max(values),
                "mean": statistics.fmean(values),
            }

    issue_counter: Dict[str, int] = defaultdict(int)
    for item in review_items:
        for issue in item.issues:
            issue_counter[issue] += 1

    return {
        "total_analyzed": len(collected),
        "ok": len(ok_items),
        "review": len(review_items),
        "errors": len(problem_items),
        "issue_counts": dict(sorted(issue_counter.items(), key=lambda kv: kv[1], reverse=True)),
        "stats": stats,
    }


def print_summary(
    summary: Dict[str, object],
    review_items: List[DrumLoopMetrics],
    show_issues: int,
) -> None:
    print("=" * 80)
    print("Drum Loop Quality Report")
    print("=" * 80)
    print(f"Analyzed files : {summary['total_analyzed']}")
    print(f"OK            : {summary['ok']}")
    print(f"Needs review  : {summary['review']}")
    print(f"Errors        : {summary['errors']}")

    issue_counts = summary.get("issue_counts", {})
    if issue_counts:
        print("\nIssue breakdown:")
        for issue, count in cast(Dict[str, int], issue_counts).items():
            print(f"  - {issue:15s}: {count}")

    stats = cast(Dict[str, Dict[str, float]], summary.get("stats", {}))
    if stats:
        print("\nAggregate metrics (min | p5 | median | p95 | max | mean):")
        for name, values in stats.items():
            print(
                f"  {name:20s}: {values['min']:.2f} | {values['p5']:.2f} | "
                f"{values['median']:.2f} | {values['p95']:.2f} | "
                f"{values['max']:.2f} | {values['mean']:.2f}"
            )

    if review_items:
        print("\nHighlighted files (needs review):")
        for item in review_items[:show_issues]:
            beats = "n/a"
            if item.loop_length_beats:
                beats = f"{item.loop_length_beats:.2f}"

            density = "0.00"
            if item.density_notes_per_beat:
                density = f"{item.density_notes_per_beat:.2f}"
            issues = ", ".join(item.issues)
            print(
                f"  - {item.path} | beats={beats} notes={item.note_count} "
                f"density={density} | issues: {issues}"
            )

    if summary.get("errors"):
        print("\n⚠️  Some files could not be processed." " Consider inspecting them manually.")


def main() -> None:  # pragma: no cover - CLI entry point
    args = parse_args()

    if not args.root.exists():
        raise SystemExit(f"Input directory not found: {args.root}")

    midi_files = list_midi_files(args.root)
    if not midi_files:
        raise SystemExit(f"No MIDI files found under {args.root}")

    if args.sample >= 0 and args.sample < len(midi_files):
        random.seed(args.seed)
        midi_files = random.sample(midi_files, args.sample)
    else:
        random.seed(args.seed)
        random.shuffle(midi_files)

    metrics = [analyze_file(path, args) for path in midi_files]
    summary = summarize_metrics(metrics)

    review_items = [m for m in metrics if m.status == "review"]
    print_summary(summary, review_items, args.show_issues)

    if not args.no_report:
        report_path = args.report_path
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with report_path.open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "summary": summary,
                    "files": [m.to_summary() for m in metrics],
                },
                handle,
                indent=2,
                ensure_ascii=False,
            )
        print(f"\nReport written to: {report_path}")


if __name__ == "__main__":  # pragma: no cover - script entry point
    main()
