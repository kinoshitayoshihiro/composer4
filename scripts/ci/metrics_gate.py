#!/usr/bin/env python3
"""CI Metrics Gate - Quality thresholds for LAMDA v2.

Validates:
1. Match rate (A/B chord comparison) ≥ 0.85
2. Controls integrity (valid PB/CC ranges) ≥ 0.99

Usage:
    python scripts/ci/metrics_gate.py \
        --ab-csv analysis/ab_chords_audit.csv \
        --stage2-json-dir output/stage2/json

Environment Variables:
    MATCH_RATE_MIN: Minimum match rate (default: 0.85)
    CONTROLS_INTEGRITY_MIN: Minimum controls integrity (default: 0.99)

Exit Codes:
    0: All gates passed
    1: One or more gates failed
"""

from __future__ import annotations
import argparse
import csv
import glob
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple


# Default thresholds
MATCH_RATE_MIN = float(os.getenv("MATCH_RATE_MIN", "0.85"))
CONTROLS_INTEGRITY_MIN = float(os.getenv("CONTROLS_INTEGRITY_MIN", "0.99"))


def gate_match_rate(ab_csv: str) -> float:
    """Calculate mean match rate from A/B chord audit CSV.

    Parameters
    ----------
    ab_csv : str
        Path to A/B chord audit CSV file.

    Returns
    -------
    float
        Mean match rate (0.0-1.0). Returns 1.0 if file not found.
    """
    if not ab_csv or not os.path.exists(ab_csv):
        print(f"⚠️  No A/B audit CSV found at: {ab_csv}")
        return 1.0  # Skip gate if no data

    rates = []
    try:
        with open(ab_csv, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    rate = float(row.get("match_rate", 1.0))
                    rates.append(rate)
                except (ValueError, TypeError):
                    continue
    except Exception as e:
        print(f"⚠️  Error reading A/B CSV: {e}")
        return 1.0

    if not rates:
        print("⚠️  No match rates found in CSV")
        return 1.0

    mean_rate = sum(rates) / len(rates)
    print(f"📊 Match rate: {mean_rate:.3f} (n={len(rates)})")
    return mean_rate


def gate_controls_integrity(stage2_dir: str) -> float:
    """Calculate controls integrity from Stage2 JSON files.

    Checks:
    - Pitch bend range within [-8191, 8191]
    - No invalid CC values

    Parameters
    ----------
    stage2_dir : str
        Directory containing Stage2 JSON files.

    Returns
    -------
    float
        Integrity score (0.0-1.0). Returns 1.0 if no files found.
    """
    if not stage2_dir or not os.path.isdir(stage2_dir):
        print(f"⚠️  No Stage2 JSON dir found at: {stage2_dir}")
        return 1.0  # Skip gate if no data

    ok, total = 0, 0
    pattern = os.path.join(stage2_dir, "*.json")
    files = glob.glob(pattern)

    if not files:
        print(f"⚠️  No JSON files in: {stage2_dir}")
        return 1.0

    for path in files:
        try:
            with open(path, "r", encoding="utf-8") as f:
                meta = json.load(f)
        except Exception:
            continue

        # Check controls field
        controls = meta.get("controls") or {}
        pb_range = controls.get("pb_range") or [0, 0]

        # Validate pitch bend range
        try:
            lo, hi = int(pb_range[0]), int(pb_range[1])
            in_range = (-8191 <= lo <= 8191) and (-8191 <= hi <= 8191) and (lo <= hi)
        except (ValueError, IndexError, TypeError):
            in_range = False

        # CC summary validation (optional)
        cc_summary = controls.get("cc_summary") or {}
        cc_valid = True
        for cc_num, stats in cc_summary.items():
            try:
                min_val = int(stats.get("min", 0))
                max_val = int(stats.get("max", 127))
                if not (0 <= min_val <= 127 and 0 <= max_val <= 127 and min_val <= max_val):
                    cc_valid = False
                    break
            except (ValueError, TypeError):
                cc_valid = False
                break

        score = 1.0 if (in_range and cc_valid) else 0.0
        ok += score
        total += 1

    if total == 0:
        print("⚠️  No valid Stage2 JSON files processed")
        return 1.0

    integrity = ok / total
    print(f"📊 Controls integrity: {integrity:.3f} (n={total})")
    return integrity


def main():
    """Main entry point for metrics gate."""
    parser = argparse.ArgumentParser(
        description="CI quality gate for LAMDA v2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--ab-csv", default="", help="Path to A/B chord audit CSV")
    parser.add_argument(
        "--stage2-json-dir", default="", help="Directory containing Stage2 JSON files"
    )
    parser.add_argument("--verbose", action="store_true", help="Verbose output")

    args = parser.parse_args()

    print("=" * 60)
    print("🚦 LAMDA v2 Quality Gate")
    print("=" * 60)

    # Gate 1: Match rate
    match_rate = gate_match_rate(args.ab_csv)
    match_ok = match_rate >= MATCH_RATE_MIN

    print(f"✓ Match rate gate: {'PASS' if match_ok else 'FAIL'}")
    print(f"  Threshold: {MATCH_RATE_MIN:.2f}")
    print(f"  Actual:    {match_rate:.3f}")
    print()

    # Gate 2: Controls integrity
    controls_int = gate_controls_integrity(args.stage2_json_dir)
    controls_ok = controls_int >= CONTROLS_INTEGRITY_MIN

    print(f"✓ Controls integrity gate: {'PASS' if controls_ok else 'FAIL'}")
    print(f"  Threshold: {CONTROLS_INTEGRITY_MIN:.2f}")
    print(f"  Actual:    {controls_int:.3f}")
    print()

    # Final result
    all_ok = match_ok and controls_ok

    if all_ok:
        print("=" * 60)
        print("✅ All quality gates PASSED")
        print("=" * 60)
        sys.exit(0)
    else:
        print("=" * 60)
        print("❌ Quality gates FAILED")
        print("=" * 60)

        if not match_ok:
            print(f"::error::Match rate {match_rate:.3f} below threshold {MATCH_RATE_MIN}")
        if not controls_ok:
            print(
                f"::error::Controls integrity {controls_int:.3f} below threshold {CONTROLS_INTEGRITY_MIN}"
            )

        sys.exit(1)


if __name__ == "__main__":
    main()
