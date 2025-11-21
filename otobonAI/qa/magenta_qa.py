#!/usr/bin/env python3
"""
Magenta QA Validator — Phase 4

Validates Magenta-generated fills against quality gates.

Usage:
    from otobonAI.qa.magenta_qa import MagentaQA

    qa = MagentaQA.from_yaml("config/quality_gates.yaml")
    results = qa.validate(fills, section="chorus")
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class QAResult:
    """QA validation result."""

    passed: bool
    violations: list[str]
    metrics: dict[str, float]


class MagentaQA:
    """Validate Magenta fills against quality gates."""

    def __init__(self, gates: dict[str, Any]):
        """Initialize QA validator.

        Args:
            gates: Quality gate configuration
        """
        self.gates = gates.get("magenta", {})
        self.defaults = self.gates.get("defaults", {})
        self.overrides = self.gates.get("section_overrides", {})

    @classmethod
    def from_yaml(cls, config_path: Path) -> MagentaQA:
        """Load QA gates from YAML."""
        with open(config_path) as f:
            gates = yaml.safe_load(f)
        return cls(gates)

    def validate(
        self,
        fills: list[dict[str, Any]],
        all_events: list[dict[str, Any]],
        section: str = "default",
    ) -> QAResult:
        """Validate Magenta fills.

        Args:
            fills: Magenta-generated fill events
            all_events: All events in the arrangement
            section: Section label for override rules

        Returns:
            QAResult with pass/fail status
        """
        # Get effective thresholds (apply section overrides)
        thresholds = self._get_thresholds(section)

        violations = []
        metrics = {}

        # Check event ratio
        magenta_count = len(fills)
        total_count = len(all_events)
        if total_count > 0:
            event_ratio = magenta_count / total_count
            metrics["event_ratio"] = event_ratio

            max_ratio = thresholds.get("max_event_ratio", 0.25)
            if event_ratio > max_ratio:
                violations.append(f"Event ratio {event_ratio:.2f} exceeds max {max_ratio}")

        # Check consecutive bars
        consecutive = self._count_consecutive_bars(fills)
        metrics["max_consecutive_bars"] = consecutive

        max_consecutive = thresholds.get("max_consecutive_bars", 4)
        if consecutive > max_consecutive:
            violations.append(f"Consecutive bars {consecutive} exceeds max {max_consecutive}")

        # Check rest gaps
        min_rest = thresholds.get("min_rest_bars", 2)
        rest_violations = self._check_rest_gaps(fills, min_rest)
        if rest_violations:
            violations.extend(rest_violations)
            metrics["rest_gap_violations"] = len(rest_violations)

        # Check pitch outliers
        max_outlier = thresholds.get("max_pitch_outlier", 12)
        outlier_count = self._count_pitch_outliers(fills, max_outlier)
        metrics["pitch_outliers"] = outlier_count

        if outlier_count > len(fills) * 0.1:  # More than 10%
            violations.append(f"Pitch outliers {outlier_count} exceeds 10% threshold")

        # Check velocity range
        velocity_range = thresholds.get("velocity_range", [40, 110])
        velocity_violations = self._check_velocity_range(fills, velocity_range)
        if velocity_violations:
            violations.extend(velocity_violations)
            metrics["velocity_violations"] = len(velocity_violations)

        # Check event density
        max_density = thresholds.get("max_density", 8.0)
        density = self._calculate_density(fills)
        metrics["event_density"] = density

        if density > max_density:
            violations.append(f"Event density {density:.1f} exceeds max {max_density}")

        passed = len(violations) == 0
        return QAResult(passed=passed, violations=violations, metrics=metrics)

    def _get_thresholds(self, section: str) -> dict[str, Any]:
        """Get effective thresholds for section."""
        thresholds = dict(self.defaults)

        if section in self.overrides:
            thresholds.update(self.overrides[section])

        return thresholds

    def _count_consecutive_bars(self, fills: list[dict[str, Any]]) -> int:
        """Count maximum consecutive bars with fills."""
        if not fills:
            return 0

        # Extract bar indices
        bars = set()
        for fill in fills:
            bar = fill.get("bar_start", fill.get("bar", 0))
            bars.add(bar)

        sorted_bars = sorted(bars)

        max_consecutive = 1
        current_consecutive = 1

        for i in range(1, len(sorted_bars)):
            if sorted_bars[i] == sorted_bars[i - 1] + 1:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 1

        return max_consecutive

    def _check_rest_gaps(self, fills: list[dict[str, Any]], min_rest: int) -> list[str]:
        """Check rest gaps between fills."""
        if not fills:
            return []

        # Extract bar indices
        bars = []
        for fill in fills:
            bar = fill.get("bar_start", fill.get("bar", 0))
            bars.append(bar)

        sorted_bars = sorted(set(bars))

        violations = []
        for i in range(1, len(sorted_bars)):
            gap = sorted_bars[i] - sorted_bars[i - 1] - 1
            if gap < min_rest:
                violations.append(
                    f"Rest gap {gap} bars between {sorted_bars[i-1]} and {sorted_bars[i]} "
                    f"is less than minimum {min_rest}"
                )

        return violations

    def _count_pitch_outliers(self, fills: list[dict[str, Any]], max_outlier: int) -> int:
        """Count pitch outliers."""
        if not fills:
            return 0

        # Calculate median pitch
        pitches = []
        for fill in fills:
            events = fill.get("events", [])
            for event in events:
                if "pitch" in event:
                    pitches.append(event["pitch"])

        if not pitches:
            return 0

        median_pitch = sorted(pitches)[len(pitches) // 2]

        # Count outliers
        outliers = 0
        for pitch in pitches:
            if abs(pitch - median_pitch) > max_outlier:
                outliers += 1

        return outliers

    def _check_velocity_range(
        self, fills: list[dict[str, Any]], velocity_range: list[int]
    ) -> list[str]:
        """Check velocity range violations."""
        min_vel, max_vel = velocity_range
        violations = []

        for fill in fills:
            events = fill.get("events", [])
            for i, event in enumerate(events):
                velocity = event.get("velocity", 80)
                if velocity < min_vel or velocity > max_vel:
                    violations.append(
                        f"Velocity {velocity} outside range [{min_vel}, {max_vel}] "
                        f"in fill bar {fill.get('bar_start', '?')} event {i}"
                    )

        return violations

    def _calculate_density(self, fills: list[dict[str, Any]]) -> float:
        """Calculate event density (events per beat)."""
        if not fills:
            return 0.0

        total_events = 0
        total_beats = 0

        for fill in fills:
            events = fill.get("events", [])
            total_events += len(events)

            bar_start = fill.get("bar_start", 0)
            bar_end = fill.get("bar_end", bar_start + 1)
            total_beats += (bar_end - bar_start) * 4  # Assume 4/4

        if total_beats == 0:
            return 0.0

        return total_events / total_beats


def main():
    """Example QA validation."""
    import json
    import sys

    if len(sys.argv) < 2:
        print("Usage: python magenta_qa.py <fills.json>")
        sys.exit(1)

    # Load fills
    fills_path = Path(sys.argv[1])
    with open(fills_path) as f:
        data = json.load(f)

    fills = data.get("fills", [])

    # Load QA gates
    qa = MagentaQA.from_yaml(Path("config/quality_gates.yaml"))

    # Validate (assuming fills are 30% of total events)
    all_events = fills + [{"dummy": True}] * (len(fills) * 2)

    result = qa.validate(fills, all_events, section="chorus")

    print(f"\n{'✅' if result.passed else '❌'} QA Result: {'PASS' if result.passed else 'FAIL'}")
    print(f"\nMetrics:")
    for key, value in result.metrics.items():
        print(f"  {key}: {value}")

    if result.violations:
        print(f"\nViolations ({len(result.violations)}):")
        for v in result.violations:
            print(f"  - {v}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
