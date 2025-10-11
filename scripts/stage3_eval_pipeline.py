#!/usr/bin/env python
"""Stage3 Automated Evaluation Pipeline (v1.0).

Workflow:
    1. Generate MIDI from prompts (stage3_infer.py)
    2. Evaluate with Stage2 model (lamda_stage2_extractor.py)
    3. Gate by thresholds (guard_retry_accept.py)
    4. A/B comparison and KPI report (ab_summarize_v2.py)

Exit Codes:
    0: Success (p50>=60, p90>=75, violations=0)
    1: Failed KPI thresholds
    2: File/dependency errors
    3: Constraint violations detected
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def run_command(
    cmd: list[str], *, cwd: Path | None = None, capture_output: bool = False
) -> subprocess.CompletedProcess:
    """Run shell command with error handling."""
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            check=True,
            capture_output=capture_output,
            text=True,
        )
        return result
    except subprocess.CalledProcessError as e:
        logging.error("Command failed: %s", " ".join(cmd))
        logging.error("Exit code: %d", e.returncode)
        if e.stdout:
            logging.error("STDOUT: %s", e.stdout)
        if e.stderr:
            logging.error("STDERR: %s", e.stderr)
        raise


def load_prompts(prompts_path: Path) -> list[dict[str, Any]]:
    """Load evaluation prompts from YAML."""
    with prompts_path.open() as f:
        data = yaml.safe_load(f)
    return data.get("prompts", [])


def stage1_generate(
    checkpoint_path: Path,
    prompts_path: Path,
    output_dir: Path,
    num_samples: int = 3,
) -> Path:
    """Step 1: Generate MIDI files from prompts."""
    logging.info("=== Step 1: Generate MIDI from prompts ===")
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "ml/stage3_infer.py",
        "--checkpoint",
        str(checkpoint_path),
        "--prompts",
        str(prompts_path),
        "--output-dir",
        str(output_dir),
        "--num-samples",
        str(num_samples),
        "--max-bars",
        "8",
        "--enforce-bar-constraint",
    ]

    run_command(cmd)
    logging.info("Generated MIDI files in: %s", output_dir)
    return output_dir


def stage2_evaluate(midi_dir: Path, output_json: Path) -> Path:
    """Step 2: Evaluate generated MIDI with Stage2 model."""
    logging.info("=== Step 2: Evaluate with Stage2 model ===")

    # Find all MIDI files
    midi_files = list(midi_dir.glob("*.mid"))
    if not midi_files:
        raise FileNotFoundError(f"No MIDI files found in {midi_dir}")

    logging.info("Found %d MIDI files to evaluate", len(midi_files))

    # Run Stage2 extractor
    cmd = [
        sys.executable,
        "ml/lamda_stage2_extractor.py",
        "--mode",
        "eval",
        "--input-dir",
        str(midi_dir),
        "--output",
        str(output_json),
    ]

    run_command(cmd)
    logging.info("Evaluation results saved to: %s", output_json)
    return output_json


def stage3_gate(
    eval_json: Path,
    gate_output: Path,
    threshold_p50: int = 60,
    threshold_p90: int = 75,
) -> dict[str, Any]:
    """Step 3: Gate results by thresholds and check constraints."""
    logging.info("=== Step 3: Gate by thresholds ===")

    with eval_json.open() as f:
        results = json.load(f)

    # Calculate percentiles
    scores = [r.get("score", 0) for r in results if "score" in r]
    if not scores:
        raise ValueError("No valid scores in evaluation results")

    scores_sorted = sorted(scores)
    n = len(scores_sorted)
    p50_idx = int(n * 0.5)
    p90_idx = int(n * 0.9)
    p50 = scores_sorted[p50_idx]
    p90 = scores_sorted[p90_idx]

    # Check violations
    violations = sum(1 for r in results if r.get("time_signature_violations", 0) > 0)
    bar_overflows = sum(1 for r in results if r.get("bar_overflow", False))

    # Gate decision
    passed = (
        p50 >= threshold_p50 and p90 >= threshold_p90 and violations == 0 and bar_overflows == 0
    )

    gate_result = {
        "p50": p50,
        "p90": p90,
        "threshold_p50": threshold_p50,
        "threshold_p90": threshold_p90,
        "violations": violations,
        "bar_overflows": bar_overflows,
        "total_samples": len(results),
        "passed": passed,
    }

    with gate_output.open("w") as f:
        json.dump(gate_result, f, indent=2)

    logging.info("Gate result: %s", "PASS" if passed else "FAIL")
    logging.info("  p50: %d (threshold: %d)", p50, threshold_p50)
    logging.info("  p90: %d (threshold: %d)", p90, threshold_p90)
    logging.info("  Violations: %d", violations)
    logging.info("  Bar overflows: %d", bar_overflows)

    return gate_result


def stage4_ab_summary(
    current_json: Path, baseline_json: Path | None, report_path: Path
) -> dict[str, Any]:
    """Step 4: A/B comparison and KPI report."""
    logging.info("=== Step 4: A/B comparison and KPI report ===")

    if baseline_json is None or not baseline_json.exists():
        logging.warning("No baseline provided, skipping A/B comparison")
        with current_json.open() as f:
            current_data = json.load(f)
        summary = {
            "mode": "current_only",
            "total_samples": len(current_data),
            "note": "No baseline for comparison",
        }
    else:
        # Run A/B summarizer
        cmd = [
            sys.executable,
            "eval/ab_summarize_v2.py",
            "--current",
            str(current_json),
            "--baseline",
            str(baseline_json),
            "--output",
            str(report_path),
        ]
        run_command(cmd)

        with report_path.open() as f:
            summary = json.load(f)

    logging.info("KPI report saved to: %s", report_path)
    return summary


def main() -> int:
    """Run full evaluation pipeline."""
    parser = argparse.ArgumentParser(description="Stage3 Automated Evaluation Pipeline")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Stage3 model checkpoint path",
    )
    parser.add_argument(
        "--prompts",
        type=Path,
        default=Path("configs/stage3/prompts_eval.yaml"),
        help="Prompts YAML file",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/stage3_eval"),
        help="Output directory for generated MIDI and reports",
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        help="Baseline evaluation JSON for A/B comparison",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=3,
        help="Number of samples to generate per prompt",
    )
    parser.add_argument(
        "--threshold-p50",
        type=int,
        default=60,
        help="Minimum p50 score to pass",
    )
    parser.add_argument(
        "--threshold-p90",
        type=int,
        default=75,
        help="Minimum p90 score to pass",
    )

    args = parser.parse_args()

    # Verify dependencies
    if not args.checkpoint.exists():
        logging.error("Checkpoint not found: %s", args.checkpoint)
        return 2

    if not args.prompts.exists():
        logging.error("Prompts file not found: %s", args.prompts)
        return 2

    # Create output directories
    midi_dir = args.output_dir / "generated_midi"
    reports_dir = args.output_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Step 1: Generate
        stage1_generate(args.checkpoint, args.prompts, midi_dir, args.num_samples)

        # Step 2: Evaluate
        eval_json = reports_dir / "stage2_eval.json"
        stage2_evaluate(midi_dir, eval_json)

        # Step 3: Gate
        gate_json = reports_dir / "gate_result.json"
        gate_result = stage3_gate(eval_json, gate_json, args.threshold_p50, args.threshold_p90)

        # Step 4: A/B summary
        ab_report = reports_dir / "ab_summary.json"
        stage4_ab_summary(eval_json, args.baseline, ab_report)

        # Final decision
        if not gate_result["passed"]:
            logging.error("Pipeline FAILED: KPI thresholds not met")
            if gate_result["violations"] > 0:
                logging.error("  %d time signature violations", gate_result["violations"])
                return 3
            if gate_result["bar_overflows"] > 0:
                logging.error("  %d bar overflows", gate_result["bar_overflows"])
                return 3
            return 1

        logging.info("Pipeline SUCCESS: All KPIs met")
        return 0

    except Exception as e:
        logging.exception("Pipeline failed with exception: %s", e)
        return 2


if __name__ == "__main__":
    sys.exit(main())
