#!/usr/bin/env python3
"""Generate a dummy evaluation report for testing ci_eval_gate.py.

This script creates a synthetic JSON report that conforms to the quick_eval_v1 schema,
useful for testing the CI gate logic without running full Stage3 generation.
"""

from __future__ import annotations

import argparse
import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def generate_dummy_report(
    n_samples: int = 32,
    pass_rate: float = 0.75,
    p50: float = 60.0,
    p90: float = 75.0,
    error_rate: float = 0.10,
) -> Dict[str, Any]:
    """Generate a synthetic evaluation report."""

    n_errors = int(n_samples * error_rate)
    n_ok = n_samples - n_errors
    n_passed = int(n_ok * pass_rate)

    # Generate individual item records
    items = []

    # Generate successful evaluations
    for i in range(n_ok):
        # Generate score around target percentiles
        if i < n_passed:
            # Passed items: scores >= 50
            score = random.uniform(50, 95)
        else:
            # Failed items: scores < 50
            score = random.uniform(20, 49)

        items.append(
            {
                "gen_id": f"gen_{i:03d}",
                "file": f"outputs/eval_stage2/midi/sample_{i:03d}.mid",
                "status": "ok",
                "error_reason": None,
                "score": round(score, 2),
                "passed": score >= 50,
                "axes_raw": {
                    "melody": round(random.uniform(40, 95), 2),
                    "harmony": round(random.uniform(40, 95), 2),
                    "rhythm": round(random.uniform(40, 95), 2),
                    "stage2_version": "1.2.3",
                },
                "diagnostics": {
                    "time_sig": random.choice(["4/4", "3/4", "6/8"]),
                    "tempo_bin": random.choice(["slow", "medium", "fast"]),
                    "genre": random.choice(["pop", "jazz", "classical"]),
                    "emotion": random.choice(["happy", "sad", "neutral"]),
                    "bar_beat_violation": random.random() < 0.05,
                    "audio": {
                        "adaptive_enabled": random.random() < 0.3,
                        "failsafe_reason": None,
                    },
                },
            }
        )

    # Generate error records
    error_reasons = ["timeout", "parse_error", "stage2_crash", "invalid_midi"]
    for i in range(n_errors):
        items.append(
            {
                "gen_id": f"gen_err_{i:03d}",
                "file": f"outputs/eval_stage2/midi/error_{i:03d}.mid",
                "status": "error",
                "error_reason": random.choice(error_reasons),
                "score": None,
                "passed": False,
                "axes_raw": None,
                "diagnostics": {
                    "time_sig": "unknown",
                    "tempo_bin": "unknown",
                    "genre": "unknown",
                    "emotion": "unknown",
                    "bar_beat_violation": False,
                    "audio": {
                        "adaptive_enabled": False,
                        "failsafe_reason": None,
                    },
                },
            }
        )

    # Shuffle items to mix successes and errors
    random.shuffle(items)

    # Compute stratified KPIs
    def stratify(items, key_path):
        from collections import defaultdict

        buckets = defaultdict(list)
        for item in items:
            if item["status"] != "ok":
                continue
            keys = key_path.split(".")
            val = item
            for k in keys:
                val = val.get(k, "unknown")
                if not isinstance(val, dict):
                    break
            buckets[str(val)].append(item)

        result = {}
        for bucket_key, bucket_items in buckets.items():
            scores = [item["score"] for item in bucket_items]
            n = len(bucket_items)
            result[bucket_key] = {
                "pass_rate": sum(1 for item in bucket_items if item["passed"]) / max(n, 1),
                "p50": sorted(scores)[n // 2] if scores else 0.0,
                "p90": sorted(scores)[int(n * 0.9)] if scores else 0.0,
                "mean": sum(scores) / max(len(scores), 1),
                "bar_beat_violation_rate": sum(
                    1 for item in bucket_items if item["diagnostics"]["bar_beat_violation"]
                )
                / max(n, 1),
                "n": n,
            }
            if n < 5:
                result[bucket_key]["_warning"] = f"low_sample_count (n={n} < 5)"
        return result

    # Build report structure
    ok_items = [item for item in items if item["status"] == "ok"]
    scores = [item["score"] for item in ok_items]
    scores_sorted = sorted(scores)

    report = {
        "meta": {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "prompt_file": "configs/stage3/prompts_eval.yaml",
            "model_commit": "abc123def456",
            "tokenizer_hash": "hash789xyz",
            "n": n_samples,
            "stage2_version": "1.2.3",
        },
        "overall": {
            "pass_rate": round(pass_rate, 6),
            "p50": round(scores_sorted[len(scores_sorted) // 2] if scores_sorted else 0.0, 2),
            "p90": round(scores_sorted[int(len(scores_sorted) * 0.9)] if scores_sorted else 0.0, 2),
            "mean": round(sum(scores) / max(len(scores), 1), 2),
            "bar_beat_violation_rate": round(
                sum(1 for item in ok_items if item["diagnostics"]["bar_beat_violation"])
                / max(len(ok_items), 1),
                6,
            ),
        },
        "errors": {
            "total": n_errors,
            "by_reason": {
                reason: sum(1 for item in items if item.get("error_reason") == reason)
                for reason in error_reasons
            },
        },
        "stratified": {
            "time_sig": stratify(items, "diagnostics.time_sig"),
            "tempo_bin": stratify(items, "diagnostics.tempo_bin"),
            "genre": stratify(items, "diagnostics.genre"),
            "emotion": stratify(items, "diagnostics.emotion"),
            "audio_adaptive": stratify(items, "diagnostics.audio.adaptive_enabled"),
        },
        "items": items,
    }

    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate dummy evaluation report for testing")
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "eval_stage2" / "eval_report_dummy.json",
        help="Output JSON path",
    )
    parser.add_argument("--n-samples", type=int, default=32, help="Number of samples")
    parser.add_argument("--pass-rate", type=float, default=0.75, help="Overall pass rate")
    parser.add_argument("--p50", type=float, default=60.0, help="Target P50 score")
    parser.add_argument("--p90", type=float, default=75.0, help="Target P90 score")
    parser.add_argument("--error-rate", type=float, default=0.10, help="Error rate (0.0-1.0)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    random.seed(args.seed)

    report = generate_dummy_report(
        n_samples=args.n_samples,
        pass_rate=args.pass_rate,
        p50=args.p50,
        p90=args.p90,
        error_rate=args.error_rate,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))

    print(f"✅ Generated dummy report: {args.output}")
    print(f"   Samples: {args.n_samples}")
    print(f"   Pass rate: {report['overall']['pass_rate']:.2%}")
    print(f"   P50: {report['overall']['p50']:.1f}")
    print(f"   P90: {report['overall']['p90']:.1f}")
    print(f"   Errors: {report['errors']['total']}")
    print()
    print("Test with:")
    print(f"  python scripts/ci_eval_gate.py {args.output} --verbose")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
