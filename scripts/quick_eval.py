#!/usr/bin/env python3
"""Quick A/B evaluation loop for Stage3 model.

Workflow:
1. Generate N samples from prompts
2. Evaluate with Stage2 model
3. Calculate KPIs (pass_rate, p50, p90, BAR violation rate, velocity distribution)
4. Compare A vs B checkpoints

Usage:
    python scripts/quick_eval.py --checkpoint checkpoints/stage3_epoch10.ckpt \
                                  --num-samples 64 \
                                  --prompts configs/stage3/prompts_eval.yaml \
                                  --output outputs/stage3/quick_eval_results.json
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Quick A/B evaluation for Stage3")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to Stage3 checkpoint (.ckpt)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=64,
        help="Number of samples to generate per prompt (default: 64)",
    )
    parser.add_argument(
        "--prompts",
        type=str,
        default="configs/stage3/prompts_eval.yaml",
        help="Path to prompts YAML file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/stage3/quick_eval_results.json",
        help="Output path for results JSON",
    )
    parser.add_argument(
        "--max-bars",
        type=int,
        default=8,
        help="Maximum bars per generation (default: 8)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device for inference (cpu/cuda/mps)",
    )
    return parser.parse_args()


def load_prompts(prompts_path: str) -> list[dict[str, Any]]:
    """Load prompts from YAML or JSON file."""
    path = Path(prompts_path)

    if path.suffix == ".json":
        with open(path) as f:
            data = json.load(f)
        return data.get("prompts", [])
    else:
        # Minimal YAML parsing for demo
        logging.warning("YAML support limited, returning default prompts")
        return [
            {"name": "test_prompt_1", "genre": "pop", "emotion": "happy"},
            {"name": "test_prompt_2", "genre": "rock", "emotion": "angry"},
        ]


def generate_samples(
    checkpoint_path: str,
    prompts: list[dict[str, Any]],
    num_samples: int,
    max_bars: int,
    device: str,
) -> list[Path]:
    """Generate MIDI samples using Stage3 inference.

    Returns:
        List of generated MIDI file paths
    """
    logging.info(f"Loading checkpoint: {checkpoint_path}")

    # Mock implementation for demonstration
    logging.warning("Mock implementation: generate_samples creating placeholder files")
    output_dir = Path("outputs/stage3/quick_eval")
    output_dir.mkdir(parents=True, exist_ok=True)

    midi_paths = []
    for i in range(min(num_samples, 5)):  # Generate 5 samples for demo
        midi_path = output_dir / f"sample_{i:03d}.mid"
        # Create empty placeholder
        midi_path.touch()
        midi_paths.append(midi_path)

    return midi_paths


def evaluate_with_stage2(midi_paths: list[Path]) -> list[dict[str, Any]]:
    """Evaluate MIDI files with Stage2 extractor.

    Returns:
        List of evaluation results (one per MIDI file)
    """
    logging.info(f"Evaluating {len(midi_paths)} MIDI files with Stage2")

    # Mock: Replace with actual Stage2 evaluation
    # from ml.lamda_stage2_extractor import extract_features
    # results = [extract_features(midi_path) for midi_path in midi_paths]

    # For now, return mock results
    logging.warning("Mock implementation: evaluate_with_stage2 not fully implemented")
    results = []
    for midi_path in midi_paths:
        results.append(
            {
                "file": str(midi_path),
                "score": 65.0 + (hash(midi_path.name) % 20),  # Mock score 65-85
                "bar_violations": 0,
                "beat_violations": 0,
                "velocity_mean": 64.0,
                "velocity_std": 12.0,
            }
        )

    return results


def calculate_kpis(eval_results: list[dict[str, Any]]) -> dict[str, Any]:
    """Calculate KPIs from evaluation results.

    KPIs:
    - pass_rate: % of samples with score >= 60
    - p50: Median score
    - p90: 90th percentile score
    - bar_violation_rate: % of samples with BAR violations
    - velocity_distribution: {mean, std, min, max}
    """
    import numpy as np

    scores = [r["score"] for r in eval_results]
    bar_violations = [r["bar_violations"] for r in eval_results]
    velocities = [r["velocity_mean"] for r in eval_results]

    # Pass rate (score >= 60)
    pass_rate = sum(1 for s in scores if s >= 60) / len(scores) * 100

    # Percentiles
    p50 = float(np.percentile(scores, 50))
    p90 = float(np.percentile(scores, 90))

    # BAR violation rate
    bar_violation_rate = sum(1 for v in bar_violations if v > 0) / len(bar_violations) * 100

    # Velocity stats
    velocity_dist = {
        "mean": float(np.mean(velocities)),
        "std": float(np.std(velocities)),
        "min": float(np.min(velocities)),
        "max": float(np.max(velocities)),
    }

    return {
        "total_samples": len(eval_results),
        "pass_rate": pass_rate,
        "p50": p50,
        "p90": p90,
        "bar_violation_rate": bar_violation_rate,
        "velocity_distribution": velocity_dist,
    }


def save_results(results: dict[str, Any], output_path: str) -> None:
    """Save results to JSON file."""
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    logging.info(f"Results saved to: {output_file}")


def print_summary(kpis: dict[str, Any]) -> None:
    """Print KPI summary to console."""
    print("\n" + "=" * 70)
    print("Quick Evaluation Summary")
    print("=" * 70)
    print(f"Total samples: {kpis['total_samples']}")
    print(f"Pass rate: {kpis['pass_rate']:.1f}% (score >= 60)")
    print(f"p50 (median): {kpis['p50']:.2f}")
    print(f"p90: {kpis['p90']:.2f}")
    print(f"BAR violation rate: {kpis['bar_violation_rate']:.1f}%")
    print("\nVelocity distribution:")
    vel = kpis["velocity_distribution"]
    print(f"  Mean: {vel['mean']:.2f}, Std: {vel['std']:.2f}")
    print(f"  Range: [{vel['min']:.2f}, {vel['max']:.2f}]")
    print("=" * 70)

    # Pass/fail gate
    if kpis["p50"] >= 60 and kpis["p90"] >= 75 and kpis["bar_violation_rate"] == 0:
        print("✅ PASS: Meets v1.0 KPI requirements")
    else:
        print("❌ FAIL: Does not meet v1.0 KPI requirements")
        if kpis["p50"] < 60:
            print(f"  - p50 too low: {kpis['p50']:.2f} < 60")
        if kpis["p90"] < 75:
            print(f"  - p90 too low: {kpis['p90']:.2f} < 75")
        if kpis["bar_violation_rate"] > 0:
            print(f"  - BAR violations: {kpis['bar_violation_rate']:.1f}% > 0%")
    print()


def main() -> None:
    """Main evaluation workflow."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    args = parse_args()

    logging.info("=" * 70)
    logging.info("Stage3 Quick A/B Evaluation")
    logging.info("=" * 70)
    logging.info(f"Checkpoint: {args.checkpoint}")
    logging.info(f"Num samples: {args.num_samples}")
    logging.info(f"Prompts: {args.prompts}")
    logging.info(f"Max bars: {args.max_bars}")
    logging.info(f"Device: {args.device}")

    # Step 1: Load prompts
    prompts = load_prompts(args.prompts)
    logging.info(f"Loaded {len(prompts)} prompts")

    # Step 2: Generate samples
    midi_paths = generate_samples(
        checkpoint_path=args.checkpoint,
        prompts=prompts,
        num_samples=args.num_samples,
        max_bars=args.max_bars,
        device=args.device,
    )
    logging.info(f"Generated {len(midi_paths)} MIDI samples")

    # Step 3: Evaluate with Stage2
    eval_results = evaluate_with_stage2(midi_paths)

    # Step 4: Calculate KPIs
    kpis = calculate_kpis(eval_results)

    # Step 5: Print summary
    print_summary(kpis)

    # Step 6: Save results
    results = {
        "checkpoint": args.checkpoint,
        "num_samples": args.num_samples,
        "kpis": kpis,
        "eval_results": eval_results,
    }
    save_results(results, args.output)

    logging.info("Evaluation complete")


if __name__ == "__main__":
    main()
