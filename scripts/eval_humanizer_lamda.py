#!/usr/bin/env python3
"""
Evaluate humanization impact on Lamda metrics (Stage3 v1.1 Day 2)

Compares MIDI files before/after humanization on Lamda Velocity and Timing scores.
Validates the +5-8pt improvement target from evaluation response.

Usage:
    python scripts/eval_humanizer_lamda.py \\
        --input-dir output/drumloops_cleaned/9 \\
        --num-samples 20 \\
        --humanize-velocity-std 12.0 \\
        --humanize-timing-jitter 0.018

Requires:
    - lamda_unified_analyzer.py for metric computation
    - scripts/humanize_midi.py
"""

import argparse
import json
import logging
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pretty_midi
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.humanize_midi import MIDIHumanizer

# Import Lamda analyzer if available
try:
    from lamda_unified_analyzer import LamdaUnifiedAnalyzer
    LAMDA_AVAILABLE = True
except ImportError:
    LAMDA_AVAILABLE = False
    logging.warning("lamda_unified_analyzer not available - will use mock metrics")


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def compute_lamda_metrics(midi_path: Path) -> dict[str, float]:
    """
    Compute Lamda Velocity and Timing metrics for a MIDI file.
    
    Returns:
        Dict with keys: velocity_score, timing_score, overall_score
    """
    if not LAMDA_AVAILABLE:
        # Mock metrics for testing without Lamda
        midi = pretty_midi.PrettyMIDI(str(midi_path))
        velocities = [n.velocity for inst in midi.instruments for n in inst.notes]
        velocity_std = np.std(velocities)
        
        # Simple heuristics
        velocity_score = min(100.0, velocity_std * 2.5)  # Higher std = higher score
        timing_score = 60.0  # Placeholder
        
        return {
            "velocity_score": velocity_score,
            "timing_score": timing_score,
            "overall_score": (velocity_score + timing_score) / 2,
        }
    
    # Real Lamda analysis
    analyzer = LamdaUnifiedAnalyzer()
    result = analyzer.analyze_file(str(midi_path))
    
    return {
        "velocity_score": result.get("velocity_score", 0.0),
        "timing_score": result.get("timing_score", 0.0),
        "overall_score": result.get("overall_score", 0.0),
    }


def evaluate_humanization(
    input_path: Path,
    *,
    velocity_std: float = 12.0,
    timing_jitter: float = 0.018,
    seed: int = 42,
) -> dict[str, Any]:
    """
    Evaluate humanization impact on a single MIDI file.
    
    Returns:
        Dict with original_metrics, humanized_metrics, improvement
    """
    # Compute original metrics
    original_metrics = compute_lamda_metrics(input_path)
    
    # Apply humanization
    humanizer = MIDIHumanizer(
        velocity_std=velocity_std,
        timing_jitter_seconds=timing_jitter,
        accent_strength=1.3,
        seed=seed,
    )
    
    with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    
    try:
        midi = pretty_midi.PrettyMIDI(str(input_path))
        humanized = humanizer.humanize(midi)
        humanized.write(str(tmp_path))
        
        # Compute humanized metrics
        humanized_metrics = compute_lamda_metrics(tmp_path)
    finally:
        tmp_path.unlink(missing_ok=True)
    
    # Calculate improvement
    improvement = {
        "velocity_delta": humanized_metrics["velocity_score"] - original_metrics["velocity_score"],
        "timing_delta": humanized_metrics["timing_score"] - original_metrics["timing_score"],
        "overall_delta": humanized_metrics["overall_score"] - original_metrics["overall_score"],
    }
    
    return {
        "file": input_path.name,
        "original": original_metrics,
        "humanized": humanized_metrics,
        "improvement": improvement,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate humanization impact on Lamda metrics")
    parser.add_argument("--input-dir", type=Path, required=True, help="Directory with MIDI files")
    parser.add_argument("--num-samples", type=int, default=20, help="Number of samples to evaluate")
    parser.add_argument("--humanize-velocity-std", type=float, default=12.0, help="Velocity std for humanization")
    parser.add_argument("--humanize-timing-jitter", type=float, default=0.018, help="Timing jitter for humanization")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", type=Path, default=Path("outputs/humanizer_eval_results.json"), help="Output JSON path")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    
    args = parser.parse_args()
    setup_logging(args.verbose)
    
    # Find MIDI files
    midi_files = list(args.input_dir.glob("*.mid"))[:args.num_samples]
    
    if not midi_files:
        logging.error("No MIDI files found in %s", args.input_dir)
        return
    
    logging.info("Evaluating %d MIDI files from %s", len(midi_files), args.input_dir)
    
    # Evaluate each file
    results = []
    for midi_path in tqdm(midi_files, desc="Evaluating humanization"):
        try:
            result = evaluate_humanization(
                midi_path,
                velocity_std=args.humanize_velocity_std,
                timing_jitter=args.humanize_timing_jitter,
                seed=args.seed,
            )
            results.append(result)
        except Exception as exc:
            logging.warning("Failed to evaluate %s: %s", midi_path, exc)
    
    # Aggregate statistics
    velocity_deltas = [r["improvement"]["velocity_delta"] for r in results]
    timing_deltas = [r["improvement"]["timing_delta"] for r in results]
    overall_deltas = [r["improvement"]["overall_delta"] for r in results]
    
    summary = {
        "num_samples": len(results),
        "humanization_params": {
            "velocity_std": args.humanize_velocity_std,
            "timing_jitter": args.humanize_timing_jitter,
            "seed": args.seed,
        },
        "aggregate_improvement": {
            "velocity_mean": float(np.mean(velocity_deltas)),
            "velocity_std": float(np.std(velocity_deltas)),
            "velocity_median": float(np.median(velocity_deltas)),
            "timing_mean": float(np.mean(timing_deltas)),
            "timing_std": float(np.std(timing_deltas)),
            "timing_median": float(np.median(timing_deltas)),
            "overall_mean": float(np.mean(overall_deltas)),
            "overall_std": float(np.std(overall_deltas)),
            "overall_median": float(np.median(overall_deltas)),
        },
        "target_validation": {
            "velocity_target": 5.0,  # Minimum +5pt improvement
            "velocity_achieved": float(np.mean(velocity_deltas)),
            "velocity_pass": float(np.mean(velocity_deltas)) >= 5.0,
            "timing_target": 3.0,  # Minimum +3pt improvement
            "timing_achieved": float(np.mean(timing_deltas)),
            "timing_pass": float(np.mean(timing_deltas)) >= 3.0,
        },
        "per_file_results": results,
    }
    
    # Save results
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    logging.info("Results saved to %s", args.output)
    
    # Print summary
    print("\n" + "=" * 60)
    print("Humanization Evaluation Summary")
    print("=" * 60)
    print(f"Samples: {len(results)}")
    print(f"Velocity improvement: {summary['aggregate_improvement']['velocity_mean']:.2f} ± {summary['aggregate_improvement']['velocity_std']:.2f} pt")
    print(f"Timing improvement: {summary['aggregate_improvement']['timing_mean']:.2f} ± {summary['aggregate_improvement']['timing_std']:.2f} pt")
    print(f"Overall improvement: {summary['aggregate_improvement']['overall_mean']:.2f} ± {summary['aggregate_improvement']['overall_std']:.2f} pt")
    print()
    print("Target Validation:")
    print(f"  Velocity: {'✅ PASS' if summary['target_validation']['velocity_pass'] else '❌ FAIL'} (target: ≥5.0pt, achieved: {summary['target_validation']['velocity_achieved']:.2f}pt)")
    print(f"  Timing: {'✅ PASS' if summary['target_validation']['timing_pass'] else '❌ FAIL'} (target: ≥3.0pt, achieved: {summary['target_validation']['timing_achieved']:.2f}pt)")
    print("=" * 60)


if __name__ == "__main__":
    main()
