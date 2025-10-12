#!/usr/bin/env python3
"""
A/B Ablation Study for REMI Tokenizer (Day 7-8)

Compares different REMI configurations to isolate component contributions:
1. Baseline (legacy): remi_enabled=False
2. DUR only: DURATION tokens only
3. DUR+ROLE: DURATION + ROLE tokens
4. DUR+ROLE+CHORD: Full REMI (v1.1)

寸評推奨: Ablation分析で各コンポーネントの寄与を分解
"""

import argparse
import json
import logging
import statistics
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import pretty_midi


class REMIMode(str, Enum):
    """REMI configuration modes for ablation."""
    LEGACY = "legacy"
    DUR_ONLY = "dur_only"
    DUR_ROLE = "dur_role"
    FULL_REMI = "full_remi"


@dataclass
class AblationMetrics:
    """Metrics for A/B comparison."""
    
    mode: str
    
    # Quality metrics
    avg_score: float
    median_score: float
    p90_score: float
    
    # Bar violation
    bar_violation_rate: float
    
    # Harmonic validity
    harmonic_validity: float
    
    # Sequence length
    avg_sequence_length: float
    p95_sequence_length: float
    
    # Velocity diversity
    velocity_std: float
    
    # Timing
    timing_jitter_std: float
    
    # File count
    total_files: int


@dataclass
class AblationComparison:
    """Comparison between two modes."""
    
    mode_a: str
    mode_b: str
    
    # Relative improvements
    score_delta: float  # percentage points
    bar_violation_delta: float
    harmonic_validity_delta: float
    sequence_length_delta: float  # percentage
    
    # Statistical significance
    significant: bool
    p_value: float | None = None


class REMIAblationEvaluator:
    """Evaluate REMI tokenizer with ablation study (寸評推奨)."""
    
    def __init__(
        self,
        baseline_dir: Path,
        dur_only_dir: Path,
        dur_role_dir: Path,
        full_remi_dir: Path,
        output_dir: Path,
        verbose: bool = False,
    ):
        self.dirs = {
            REMIMode.LEGACY: baseline_dir,
            REMIMode.DUR_ONLY: dur_only_dir,
            REMIMode.DUR_ROLE: dur_role_dir,
            REMIMode.FULL_REMI: full_remi_dir,
        }
        self.output_dir = output_dir
        self.verbose = verbose
        
        self.logger = logging.getLogger(__name__)
        if verbose:
            self.logger.setLevel(logging.DEBUG)
    
    def evaluate(self) -> dict[str, AblationMetrics]:
        """Evaluate all modes and compute metrics."""
        results = {}
        
        for mode, directory in self.dirs.items():
            self.logger.info(f"Evaluating mode: {mode.value}")
            metrics = self._evaluate_mode(directory, mode.value)
            results[mode.value] = metrics
        
        return results
    
    def _evaluate_mode(self, directory: Path, mode: str) -> AblationMetrics:
        """Evaluate a single mode."""
        midi_files = list(directory.glob("*.mid"))
        
        if not midi_files:
            self.logger.warning(f"No MIDI files found in {directory}")
            return AblationMetrics(
                mode=mode,
                avg_score=0.0,
                median_score=0.0,
                p90_score=0.0,
                bar_violation_rate=1.0,
                harmonic_validity=0.0,
                avg_sequence_length=0.0,
                p95_sequence_length=0.0,
                velocity_std=0.0,
                timing_jitter_std=0.0,
                total_files=0,
            )
        
        # Evaluate each file
        scores = []
        bar_violations = []
        harmonic_scores = []
        sequence_lengths = []
        velocity_stds = []
        timing_jitters = []
        
        for midi_file in midi_files:
            try:
                midi = pretty_midi.PrettyMIDI(str(midi_file))
                
                # Mock scores (replace with actual evaluation)
                scores.append(70.0 + hash(midi_file.name) % 20)
                
                # Bar violations
                violations = self._compute_bar_violations(midi)
                bar_violations.append(violations)
                
                # Harmonic validity
                harmonic = self._compute_harmonic_score(midi)
                harmonic_scores.append(harmonic)
                
                # Sequence length
                seq_len = sum(len(inst.notes) for inst in midi.instruments) * 10
                sequence_lengths.append(seq_len)
                
                # Velocity diversity
                velocities = [n.velocity for inst in midi.instruments for n in inst.notes]
                if velocities:
                    velocity_stds.append(statistics.stdev(velocities) if len(velocities) > 1 else 0.0)
                
                # Timing jitter
                timing_jitters.append(15.0)  # Mock
                
            except Exception as e:
                self.logger.error(f"Error evaluating {midi_file}: {e}")
        
        return AblationMetrics(
            mode=mode,
            avg_score=statistics.mean(scores) if scores else 0.0,
            median_score=statistics.median(scores) if scores else 0.0,
            p90_score=statistics.quantiles(scores, n=10)[8] if len(scores) >= 10 else max(scores, default=0.0),
            bar_violation_rate=statistics.mean(bar_violations) if bar_violations else 1.0,
            harmonic_validity=statistics.mean(harmonic_scores) if harmonic_scores else 0.0,
            avg_sequence_length=statistics.mean(sequence_lengths) if sequence_lengths else 0.0,
            p95_sequence_length=statistics.quantiles(sequence_lengths, n=20)[18] if len(sequence_lengths) >= 20 else max(sequence_lengths, default=0.0),
            velocity_std=statistics.mean(velocity_stds) if velocity_stds else 0.0,
            timing_jitter_std=statistics.mean(timing_jitters) if timing_jitters else 0.0,
            total_files=len(midi_files),
        )
    
    def _compute_bar_violations(self, midi: pretty_midi.PrettyMIDI) -> float:
        """Compute bar violation rate."""
        beats_per_bar = 4
        tempo = midi.estimate_tempo()
        bar_duration = (60 / tempo) * beats_per_bar
        
        violations = 0
        total_notes = 0
        
        for inst in midi.instruments:
            for note in inst.notes:
                total_notes += 1
                start_bar = int(note.start / bar_duration)
                end_bar = int(note.end / bar_duration)
                if end_bar > start_bar:
                    violations += 1
        
        return violations / max(1, total_notes)
    
    def _compute_harmonic_score(self, midi: pretty_midi.PrettyMIDI) -> float:
        """Compute harmonic validity score."""
        # Mock: 70-90% range
        return 72.1 + (hash(midi.get_end_time()) % 20)
    
    def compare_modes(
        self,
        results: dict[str, AblationMetrics],
    ) -> list[AblationComparison]:
        """Compare modes pairwise (寸評推奨: 寄与分解)."""
        comparisons = []
        
        # Legacy vs DUR only
        if REMIMode.LEGACY.value in results and REMIMode.DUR_ONLY.value in results:
            comparisons.append(self._compare_pair(
                results[REMIMode.LEGACY.value],
                results[REMIMode.DUR_ONLY.value],
            ))
        
        # DUR only vs DUR+ROLE
        if REMIMode.DUR_ONLY.value in results and REMIMode.DUR_ROLE.value in results:
            comparisons.append(self._compare_pair(
                results[REMIMode.DUR_ONLY.value],
                results[REMIMode.DUR_ROLE.value],
            ))
        
        # DUR+ROLE vs Full REMI
        if REMIMode.DUR_ROLE.value in results and REMIMode.FULL_REMI.value in results:
            comparisons.append(self._compare_pair(
                results[REMIMode.DUR_ROLE.value],
                results[REMIMode.FULL_REMI.value],
            ))
        
        # Legacy vs Full REMI (total improvement)
        if REMIMode.LEGACY.value in results and REMIMode.FULL_REMI.value in results:
            comparisons.append(self._compare_pair(
                results[REMIMode.LEGACY.value],
                results[REMIMode.FULL_REMI.value],
            ))
        
        return comparisons
    
    def _compare_pair(
        self,
        metrics_a: AblationMetrics,
        metrics_b: AblationMetrics,
    ) -> AblationComparison:
        """Compare two modes."""
        score_delta = metrics_b.avg_score - metrics_a.avg_score
        bar_violation_delta = metrics_b.bar_violation_rate - metrics_a.bar_violation_rate
        harmonic_validity_delta = metrics_b.harmonic_validity - metrics_a.harmonic_validity
        sequence_length_delta = (
            (metrics_b.avg_sequence_length - metrics_a.avg_sequence_length) 
            / max(1, metrics_a.avg_sequence_length) * 100
        )
        
        # Simple significance test (mock)
        significant = abs(score_delta) > 2.0  # 2 points threshold
        
        return AblationComparison(
            mode_a=metrics_a.mode,
            mode_b=metrics_b.mode,
            score_delta=score_delta,
            bar_violation_delta=bar_violation_delta,
            harmonic_validity_delta=harmonic_validity_delta,
            sequence_length_delta=sequence_length_delta,
            significant=significant,
            p_value=None,  # TODO: Statistical test
        )
    
    def save_report(
        self,
        results: dict[str, AblationMetrics],
        comparisons: list[AblationComparison],
    ) -> Path:
        """Save ablation study report."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        report_path = self.output_dir / "remi_ablation_report.json"
        
        report = {
            "results": {mode: asdict(metrics) for mode, metrics in results.items()},
            "comparisons": [asdict(comp) for comp in comparisons],
            "summary": self._generate_summary(results, comparisons),
        }
        
        report_path.write_text(json.dumps(report, indent=2))
        self.logger.info(f"Report saved to {report_path}")
        return report_path
    
    def _generate_summary(
        self,
        results: dict[str, AblationMetrics],
        comparisons: list[AblationComparison],
    ) -> dict[str, Any]:
        """Generate summary of ablation study."""
        # Find best mode
        best_mode = max(results.items(), key=lambda x: x[1].avg_score)
        
        # Total improvement (legacy → full REMI)
        total_improvement = next(
            (c for c in comparisons if c.mode_a == REMIMode.LEGACY.value and c.mode_b == REMIMode.FULL_REMI.value),
            None
        )
        
        return {
            "best_mode": best_mode[0],
            "best_score": best_mode[1].avg_score,
            "total_improvement": asdict(total_improvement) if total_improvement else None,
            "component_contributions": [
                {
                    "component": comp.mode_b.replace(comp.mode_a, "").strip("_"),
                    "score_delta": comp.score_delta,
                    "bar_violation_delta": comp.bar_violation_delta,
                    "harmonic_validity_delta": comp.harmonic_validity_delta,
                }
                for comp in comparisons
                if comp.mode_a != REMIMode.LEGACY.value or comp.mode_b != REMIMode.FULL_REMI.value
            ],
        }


def main():
    parser = argparse.ArgumentParser(description="A/B Ablation Study for REMI (Day 7-8)")
    parser.add_argument("--baseline-dir", type=Path, required=True, help="Legacy mode output directory")
    parser.add_argument("--dur-only-dir", type=Path, required=True, help="DUR only mode output directory")
    parser.add_argument("--dur-role-dir", type=Path, required=True, help="DUR+ROLE mode output directory")
    parser.add_argument("--full-remi-dir", type=Path, required=True, help="Full REMI mode output directory")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/ablation"), help="Output directory")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    
    evaluator = REMIAblationEvaluator(
        baseline_dir=args.baseline_dir,
        dur_only_dir=args.dur_only_dir,
        dur_role_dir=args.dur_role_dir,
        full_remi_dir=args.full_remi_dir,
        output_dir=args.output_dir,
        verbose=args.verbose,
    )
    
    results = evaluator.evaluate()
    comparisons = evaluator.compare_modes(results)
    report_path = evaluator.save_report(results, comparisons)
    
    # Print summary
    print("\n" + "=" * 60)
    print("REMI Ablation Study Results")
    print("=" * 60)
    
    for mode, metrics in results.items():
        print(f"\n{mode.upper()}:")
        print(f"  Avg Score: {metrics.avg_score:.1f}")
        print(f"  Bar Violation: {metrics.bar_violation_rate:.2%}")
        print(f"  Harmonic Validity: {metrics.harmonic_validity:.1f}%")
    
    print("\n" + "-" * 60)
    print("Component Contributions:")
    print("-" * 60)
    
    for comp in comparisons:
        print(f"\n{comp.mode_a} → {comp.mode_b}:")
        print(f"  Score Δ: {comp.score_delta:+.1f}")
        print(f"  Bar Violation Δ: {comp.bar_violation_delta:+.2%}")
        print(f"  Harmonic Validity Δ: {comp.harmonic_validity_delta:+.1f}%")
        print(f"  Sequence Length Δ: {comp.sequence_length_delta:+.1f}%")
    
    print(f"\nReport saved to: {report_path}")


if __name__ == "__main__":
    main()
