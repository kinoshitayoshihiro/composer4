#!/usr/bin/env python3
"""CI evaluation gate: validate quick_eval_stage2 output and enforce quality thresholds.

This script is designed to run in CI pipelines and provides automated pass/fail
decisions based on:

1. JSON schema validation (structural correctness)
2. KPI threshold checks (quality gates)
3. Sample size warnings (statistical reliability)
4. Error rate limits

Exit codes:
  0: All checks passed
  1: Quality gate failed (KPI below threshold or errors excessive)
  2: Schema validation failed or report file missing
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

LOGGER = logging.getLogger("ci_eval_gate")

# Quality gate thresholds (configurable via CLI)
DEFAULT_THRESHOLDS = {
    "overall_pass_rate_min": 0.70,  # 70% of samples must pass Stage2 (score >= 50)
    "overall_p50_min": 55.0,  # Median score >= 55
    "overall_p90_min": 70.0,  # 90th percentile >= 70
    "error_rate_max": 0.20,  # Max 20% evaluation errors
    "bar_violation_rate_max": 0.10,  # Max 10% bar/beat violations
    "min_sample_count": 5,  # Minimum samples per stratified bucket for reliability
}


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(levelname)s | %(message)s",
    )


def load_report(report_path: Path) -> Dict[str, Any]:
    """Load and parse JSON evaluation report."""
    if not report_path.exists():
        raise FileNotFoundError(f"Report file not found: {report_path}")

    with report_path.open() as f:
        return json.load(f)


def validate_schema(report: Dict[str, Any], schema_path: Path) -> bool:
    """Validate report against JSON schema. Returns True if valid."""
    try:
        import jsonschema
    except ImportError:
        LOGGER.warning("jsonschema not installed, skipping schema validation")
        return True

    if not schema_path.exists():
        LOGGER.error("Schema file not found: %s", schema_path)
        return False

    schema = json.loads(schema_path.read_text())
    try:
        jsonschema.validate(report, schema)
        LOGGER.info("✅ Schema validation passed")
        return True
    except jsonschema.ValidationError as exc:
        LOGGER.error("❌ Schema validation FAILED")
        LOGGER.error("  Path: %s", list(exc.path))
        LOGGER.error("  Error: %s", exc.message)
        return False


def check_overall_kpis(
    report: Dict[str, Any],
    thresholds: Dict[str, float],
) -> tuple[bool, List[str]]:
    """Check overall KPIs against thresholds. Returns (pass, failures)."""
    overall = report.get("overall", {})
    errors = report.get("errors", {})
    meta = report.get("meta", {})

    n_total = meta.get("n", 0)
    n_errors = errors.get("total", 0)

    failures: List[str] = []
    pass_rate = overall.get("pass_rate", 0.0)
    p50 = overall.get("p50", 0.0)
    p90 = overall.get("p90", 0.0)
    bar_viol_rate = overall.get("bar_beat_violation_rate", 0.0)
    error_rate = n_errors / n_total if n_total > 0 else 1.0

    if pass_rate < thresholds["overall_pass_rate_min"]:
        failures.append(f"Pass rate {pass_rate:.2%} < {thresholds['overall_pass_rate_min']:.2%}")

    if p50 < thresholds["overall_p50_min"]:
        failures.append(f"P50 {p50:.1f} < {thresholds['overall_p50_min']:.1f}")

    if p90 < thresholds["overall_p90_min"]:
        failures.append(f"P90 {p90:.1f} < {thresholds['overall_p90_min']:.1f}")

    if error_rate > thresholds["error_rate_max"]:
        failures.append(f"Error rate {error_rate:.2%} > {thresholds['error_rate_max']:.2%}")

    if bar_viol_rate > thresholds["bar_violation_rate_max"]:
        failures.append(
            f"Bar violation rate {bar_viol_rate:.2%} > {thresholds['bar_violation_rate_max']:.2%}"
        )

    return (len(failures) == 0, failures)


def check_stratified_warnings(
    report: Dict[str, Any],
    min_sample_count: int,
) -> List[str]:
    """Check stratified KPIs for low sample count warnings."""
    warnings: List[str] = []
    stratified = report.get("stratified", {})

    for dimension, buckets in stratified.items():
        if not isinstance(buckets, dict):
            continue
        for bucket_key, kpi in buckets.items():
            if not isinstance(kpi, dict):
                continue
            n = kpi.get("n", 0)
            if n < min_sample_count:
                warning_msg = kpi.get("_warning", "")
                warnings.append(
                    f"Stratified '{dimension}/{bucket_key}': n={n} < {min_sample_count} ({warning_msg})"
                )

    return warnings


def print_summary(
    report: Dict[str, Any],
    schema_valid: bool,
    kpi_pass: bool,
    kpi_failures: List[str],
    warnings: List[str],
) -> None:
    """Print human-readable summary."""
    print("\n" + "=" * 70)
    print("CI Evaluation Gate Summary")
    print("=" * 70)

    meta = report.get("meta", {})
    overall = report.get("overall", {})

    print(f"Report:        {meta.get('created_at', 'unknown')}")
    print(f"Model:         {meta.get('model_commit', 'unknown')[:12]}")
    print(f"Tokenizer:     {meta.get('tokenizer_hash', 'unknown')[:12]}")
    print(f"Stage2:        {meta.get('stage2_version', 'unknown')}")
    print(f"Samples:       {meta.get('n', 0)}")
    print()
    print(f"Pass Rate:     {overall.get('pass_rate', 0.0):.2%}")
    print(f"P50:           {overall.get('p50', 0.0):.1f}")
    print(f"P90:           {overall.get('p90', 0.0):.1f}")
    print(f"Violations:    {overall.get('bar_beat_violation_rate', 0.0):.2%}")
    print()

    print(f"Schema Valid:  {'✅ PASS' if schema_valid else '❌ FAIL'}")
    print(f"KPI Gate:      {'✅ PASS' if kpi_pass else '❌ FAIL'}")

    if kpi_failures:
        print("\nKPI Failures:")
        for failure in kpi_failures:
            print(f"  ❌ {failure}")

    if warnings:
        print(f"\n⚠️  Stratified Sample Warnings ({len(warnings)}):")
        for warning in warnings[:5]:  # Limit output
            print(f"  {warning}")
        if len(warnings) > 5:
            print(f"  ... and {len(warnings) - 5} more")

    print("=" * 70)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CI evaluation quality gate")
    parser.add_argument("report", type=Path, help="Path to quick_eval JSON report")
    parser.add_argument(
        "--schema",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "eval" / "schema" / "quick_eval_v1.json",
        help="JSON schema path",
    )
    parser.add_argument(
        "--overall-pass-rate-min",
        type=float,
        default=DEFAULT_THRESHOLDS["overall_pass_rate_min"],
        help="Minimum overall pass rate",
    )
    parser.add_argument(
        "--overall-p50-min",
        type=float,
        default=DEFAULT_THRESHOLDS["overall_p50_min"],
        help="Minimum overall P50 score",
    )
    parser.add_argument(
        "--overall-p90-min",
        type=float,
        default=DEFAULT_THRESHOLDS["overall_p90_min"],
        help="Minimum overall P90 score",
    )
    parser.add_argument(
        "--error-rate-max",
        type=float,
        default=DEFAULT_THRESHOLDS["error_rate_max"],
        help="Maximum error rate",
    )
    parser.add_argument(
        "--bar-violation-rate-max",
        type=float,
        default=DEFAULT_THRESHOLDS["bar_violation_rate_max"],
        help="Maximum bar/beat violation rate",
    )
    parser.add_argument(
        "--min-sample-count",
        type=int,
        default=DEFAULT_THRESHOLDS["min_sample_count"],
        help="Minimum samples per stratified bucket",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    setup_logging(args.verbose)

    try:
        report = load_report(args.report)
    except Exception as exc:
        LOGGER.error("Failed to load report: %s", exc)
        return 2

    schema_valid = validate_schema(report, args.schema)
    if not schema_valid:
        LOGGER.error("Schema validation failed, aborting gate check")
        return 2

    thresholds = {
        "overall_pass_rate_min": args.overall_pass_rate_min,
        "overall_p50_min": args.overall_p50_min,
        "overall_p90_min": args.overall_p90_min,
        "error_rate_max": args.error_rate_max,
        "bar_violation_rate_max": args.bar_violation_rate_max,
        "min_sample_count": args.min_sample_count,
    }

    kpi_pass, kpi_failures = check_overall_kpis(report, thresholds)
    warnings = check_stratified_warnings(report, args.min_sample_count)

    print_summary(report, schema_valid, kpi_pass, kpi_failures, warnings)

    if not kpi_pass:
        LOGGER.error("❌ Quality gate FAILED")
        return 1

    LOGGER.info("✅ All quality gates PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
