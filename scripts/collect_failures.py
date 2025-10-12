#!/usr/bin/env python3
"""Collect failed samples from Stage3 generation for retry processing.

This script identifies samples that failed quality checks and prepares them
for automatic retry with appropriate presets.

Usage:
    PYTHONPATH=. python scripts/collect_failures.py \
        --eval-report outputs/stage3/eval_report.json \
        --config configs/failure_criteria.yaml \
        --output outputs/stage3/failed_cases.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


DEFAULT_CRITERIA = {
    "score_threshold": 45.0,
    "text_audio_cos_threshold": 0.50,
    "check_emotion_mismatch": True,
    "check_structure_violations": True,
    "max_failures_per_category": 50,
}


def load_criteria(config_path: Path | None) -> dict[str, Any]:
    """Load failure criteria from YAML config."""
    if config_path is None or not config_path.exists():
        logging.warning("No config found, using defaults")
        return DEFAULT_CRITERIA
    
    with open(config_path, encoding="utf-8") as f:
        criteria = yaml.safe_load(f)
    
    return {**DEFAULT_CRITERIA, **criteria}


def load_eval_report(report_path: Path) -> pd.DataFrame:
    """Load evaluation report as DataFrame."""
    with open(report_path, encoding="utf-8") as f:
        report = json.load(f)
    
    # Extract sample-level results
    samples = report.get("samples", [])
    if not samples:
        logging.warning("No samples found in report")
        return pd.DataFrame()
    
    return pd.DataFrame(samples)


def identify_failures(
    df: pd.DataFrame,
    criteria: dict[str, Any],
) -> pd.DataFrame:
    """Identify failed samples based on criteria."""
    failures = []
    
    for _, row in df.iterrows():
        reasons = []
        recommended_preset = None
        
        # Check score
        score = row.get("score_total", 100)
        if score < criteria["score_threshold"]:
            reasons.append(f"low_score:{score:.1f}")
            recommended_preset = "velocity_chain_audio"
        
        # Check text-audio alignment
        text_audio_cos = row.get("text_audio_cos")
        if text_audio_cos is not None and text_audio_cos < criteria["text_audio_cos_threshold"]:
            reasons.append(f"low_text_audio_cos:{text_audio_cos:.3f}")
            if recommended_preset is None:
                recommended_preset = "audio_adaptive"
        
        # Check emotion mismatch
        if criteria.get("check_emotion_mismatch"):
            emotion_pred = row.get("emotion_predicted")
            emotion_cond = row.get("emotion_condition")
            if emotion_pred and emotion_cond and emotion_pred != emotion_cond:
                reasons.append(f"emotion_mismatch:{emotion_cond}→{emotion_pred}")
                if recommended_preset is None:
                    recommended_preset = "emotion_correction"
        
        # Check structure violations
        if criteria.get("check_structure_violations"):
            bar_violations = row.get("bar_violations", 0)
            beat_violations = row.get("beat_violations", 0)
            if bar_violations > 0 or beat_violations > 0:
                reasons.append(f"structure_violations:bar={bar_violations},beat={beat_violations}")
                if recommended_preset is None:
                    recommended_preset = "structure_repair"
        
        if reasons:
            failures.append({
                "loop_id": row.get("loop_id"),
                "file_digest": row.get("file_digest"),
                "reasons": reasons,
                "metrics": {
                    "score_total": score,
                    "text_audio_cos": text_audio_cos,
                    "emotion_predicted": emotion_pred if "emotion_pred" in locals() else None,
                    "emotion_condition": emotion_cond if "emotion_cond" in locals() else None,
                },
                "recommended_preset": recommended_preset,
            })
    
    return pd.DataFrame(failures)


def balance_failures(
    df: pd.DataFrame,
    max_per_category: int,
) -> pd.DataFrame:
    """Balance failures across reason categories."""
    if df.empty:
        return df
    
    # Extract primary reason category
    df["primary_reason"] = df["reasons"].apply(
        lambda x: x[0].split(":")[0] if x else "unknown"
    )
    
    # Sample from each category
    balanced = []
    for category in df["primary_reason"].unique():
        category_df = df[df["primary_reason"] == category]
        sample_size = min(len(category_df), max_per_category)
        balanced.append(category_df.sample(n=sample_size, random_state=42))
    
    result = pd.concat(balanced, ignore_index=True)
    result = result.drop(columns=["primary_reason"])
    
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect failed samples for retry")
    parser.add_argument(
        "--eval-report",
        type=Path,
        required=True,
        help="Stage3 evaluation report JSON",
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Failure criteria YAML config",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output JSONL file for failed cases",
    )
    parser.add_argument(
        "--balance",
        action="store_true",
        help="Balance failures across categories",
    )
    
    args = parser.parse_args()
    
    # Load criteria
    criteria = load_criteria(args.config)
    logging.info("Using criteria: %s", criteria)
    
    # Load evaluation report
    df = load_eval_report(args.eval_report)
    logging.info("Loaded %d samples from evaluation report", len(df))
    
    # Identify failures
    failures = identify_failures(df, criteria)
    logging.info("Identified %d failures", len(failures))
    
    # Balance if requested
    if args.balance and not failures.empty:
        failures = balance_failures(failures, criteria["max_failures_per_category"])
        logging.info("Balanced to %d failures", len(failures))
    
    # Save output
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for _, row in failures.iterrows():
            f.write(json.dumps(row.to_dict(), ensure_ascii=False) + "\n")
    
    logging.info("Saved %d failed cases to %s", len(failures), args.output)
    
    # Print summary
    if not failures.empty:
        print("\nFailure Summary:")
        reason_counts = {}
        for reasons in failures["reasons"]:
            for reason in reasons:
                category = reason.split(":")[0]
                reason_counts[category] = reason_counts.get(category, 0) + 1
        
        for category, count in sorted(reason_counts.items(), key=lambda x: -x[1]):
            print(f"  {category}: {count}")


if __name__ == "__main__":
    main()
