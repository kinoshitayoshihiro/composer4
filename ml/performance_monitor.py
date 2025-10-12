#!/usr/bin/env python3
"""
Performance Monitor for Stage3 Inference (Day 9-10)

Tracks latency, sequence length, and memory usage for Performer evaluation.

寸評推奨: p95レイテンシと最大長をログ化
"""

import json
import logging
import statistics
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore


@dataclass
class InferenceMetrics:
    """Metrics for a single inference run."""
    
    # Latency (ms)
    total_latency_ms: float
    per_token_latency_ms: float
    
    # Sequence length
    prompt_length: int
    generated_length: int
    total_length: int
    
    # Memory (MB)
    peak_memory_mb: float
    avg_memory_mb: float
    
    # Metadata
    timestamp: float
    model_type: str  # "standard" or "performer"


@dataclass
class PerformanceReport:
    """Aggregated performance report."""
    
    model_type: str
    total_runs: int
    
    # Latency statistics (ms)
    latency_mean: float
    latency_median: float
    latency_p95: float
    latency_p99: float
    
    # Per-token latency (ms)
    per_token_mean: float
    per_token_p95: float
    
    # Sequence length statistics
    max_sequence_length: int
    avg_sequence_length: float
    p95_sequence_length: int
    p99_sequence_length: int
    
    # Memory statistics (MB)
    peak_memory_mean: float
    peak_memory_p95: float
    
    # Raw metrics for detailed analysis
    raw_metrics: list[dict[str, Any]] = field(default_factory=list)


class PerformanceMonitor:
    """Monitor inference performance (寸評推奨).
    
    Usage:
        monitor = PerformanceMonitor()
        
        with monitor.track_inference(model_type="performer"):
            output = model.generate(...)
        
        monitor.log_metrics(prompt_length=10, generated_length=100)
        report = monitor.generate_report()
    """
    
    def __init__(self):
        self.metrics: list[InferenceMetrics] = []
        self.current_run: dict[str, Any] = {}
        self.logger = logging.getLogger(__name__)
    
    def track_inference(self, model_type: str = "standard"):
        """Context manager for tracking inference run."""
        return _InferenceTracker(self, model_type)
    
    def log_metrics(
        self,
        total_latency_ms: float,
        prompt_length: int,
        generated_length: int,
        peak_memory_mb: float,
        avg_memory_mb: float,
        model_type: str = "standard",
    ):
        """Log metrics for a completed inference run."""
        metrics = InferenceMetrics(
            total_latency_ms=total_latency_ms,
            per_token_latency_ms=total_latency_ms / max(1, generated_length),
            prompt_length=prompt_length,
            generated_length=generated_length,
            total_length=prompt_length + generated_length,
            peak_memory_mb=peak_memory_mb,
            avg_memory_mb=avg_memory_mb,
            timestamp=time.time(),
            model_type=model_type,
        )
        
        self.metrics.append(metrics)
        
        self.logger.debug(
            f"Logged inference: {model_type}, latency={total_latency_ms:.1f}ms, "
            f"seq_len={metrics.total_length}, memory={peak_memory_mb:.1f}MB"
        )
    
    def generate_report(self, model_type: str | None = None) -> PerformanceReport:
        """Generate performance report (寸評推奨: p95レイテンシ監視)."""
        filtered_metrics = self.metrics
        if model_type:
            filtered_metrics = [m for m in self.metrics if m.model_type == model_type]
        
        if not filtered_metrics:
            raise ValueError(f"No metrics found for model_type={model_type}")
        
        # Extract values
        latencies = [m.total_latency_ms for m in filtered_metrics]
        per_token_latencies = [m.per_token_latency_ms for m in filtered_metrics]
        seq_lengths = [m.total_length for m in filtered_metrics]
        peak_memories = [m.peak_memory_mb for m in filtered_metrics]
        
        return PerformanceReport(
            model_type=model_type or "all",
            total_runs=len(filtered_metrics),
            latency_mean=statistics.mean(latencies),
            latency_median=statistics.median(latencies),
            latency_p95=statistics.quantiles(latencies, n=20)[18] if len(latencies) >= 20 else max(latencies),
            latency_p99=statistics.quantiles(latencies, n=100)[98] if len(latencies) >= 100 else max(latencies),
            per_token_mean=statistics.mean(per_token_latencies),
            per_token_p95=statistics.quantiles(per_token_latencies, n=20)[18] if len(per_token_latencies) >= 20 else max(per_token_latencies),
            max_sequence_length=max(seq_lengths),
            avg_sequence_length=statistics.mean(seq_lengths),
            p95_sequence_length=int(statistics.quantiles(seq_lengths, n=20)[18]) if len(seq_lengths) >= 20 else max(seq_lengths),
            p99_sequence_length=int(statistics.quantiles(seq_lengths, n=100)[98]) if len(seq_lengths) >= 100 else max(seq_lengths),
            peak_memory_mean=statistics.mean(peak_memories),
            peak_memory_p95=statistics.quantiles(peak_memories, n=20)[18] if len(peak_memories) >= 20 else max(peak_memories),
            raw_metrics=[asdict(m) for m in filtered_metrics],
        )
    
    def save_report(self, output_path: Path, model_type: str | None = None):
        """Save performance report to JSON."""
        report = self.generate_report(model_type=model_type)
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(asdict(report), indent=2))
        
        self.logger.info(f"Performance report saved to {output_path}")
    
    def compare_models(
        self,
        baseline_type: str = "standard",
        comparison_type: str = "performer",
    ) -> dict[str, Any]:
        """Compare two model types (寸評推奨: Standard vs Performer).
        
        Returns:
            Comparison dict with speedup/memory improvement metrics
        """
        baseline_report = self.generate_report(model_type=baseline_type)
        comparison_report = self.generate_report(model_type=comparison_type)
        
        return {
            "baseline": asdict(baseline_report),
            "comparison": asdict(comparison_report),
            "improvements": {
                "latency_speedup": baseline_report.latency_mean / comparison_report.latency_mean,
                "p95_latency_speedup": baseline_report.latency_p95 / comparison_report.latency_p95,
                "memory_reduction": (baseline_report.peak_memory_mean - comparison_report.peak_memory_mean) / baseline_report.peak_memory_mean,
                "max_sequence_improvement": comparison_report.max_sequence_length - baseline_report.max_sequence_length,
            },
            "summary": {
                "faster": comparison_report.latency_mean < baseline_report.latency_mean,
                "less_memory": comparison_report.peak_memory_mean < baseline_report.peak_memory_mean,
                "longer_sequences": comparison_report.max_sequence_length > baseline_report.max_sequence_length,
            },
        }


class _InferenceTracker:
    """Context manager for tracking a single inference run."""
    
    def __init__(self, monitor: PerformanceMonitor, model_type: str):
        self.monitor = monitor
        self.model_type = model_type
        self.start_time = 0.0
        self.start_memory = 0.0
        self.memory_samples: list[float] = []
    
    def __enter__(self):
        self.start_time = time.time()
        if torch is not None and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            self.start_memory = torch.cuda.memory_allocated() / (1024 ** 2)  # MB
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed_ms = (time.time() - self.start_time) * 1000
        
        peak_memory_mb = 0.0
        avg_memory_mb = 0.0
        if torch is not None and torch.cuda.is_available():
            peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
            avg_memory_mb = torch.cuda.memory_allocated() / (1024 ** 2)
        
        # Store for later logging
        self.monitor.current_run = {
            "total_latency_ms": elapsed_ms,
            "peak_memory_mb": peak_memory_mb,
            "avg_memory_mb": avg_memory_mb,
            "model_type": self.model_type,
        }
        
        return False  # Don't suppress exceptions


def main():
    """Example usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Performance Monitor Demo")
    parser.add_argument("--report", type=Path, help="Load and display report")
    
    args = parser.parse_args()
    
    if args.report:
        report_data = json.loads(args.report.read_text())
        
        print("\n" + "=" * 60)
        print(f"Performance Report: {report_data['model_type']}")
        print("=" * 60)
        print(f"Total runs: {report_data['total_runs']}")
        print(f"\nLatency (ms):")
        print(f"  Mean: {report_data['latency_mean']:.1f}")
        print(f"  Median: {report_data['latency_median']:.1f}")
        print(f"  P95: {report_data['latency_p95']:.1f}")
        print(f"  P99: {report_data['latency_p99']:.1f}")
        print(f"\nSequence Length:")
        print(f"  Max: {report_data['max_sequence_length']}")
        print(f"  Avg: {report_data['avg_sequence_length']:.0f}")
        print(f"  P95: {report_data['p95_sequence_length']}")
        print(f"\nMemory (MB):")
        print(f"  Peak Mean: {report_data['peak_memory_mean']:.1f}")
        print(f"  Peak P95: {report_data['peak_memory_p95']:.1f}")
    else:
        print("Usage: python -m ml.performance_monitor --report <path_to_report.json>")


def compare_models(
    baseline_report: PerformanceReport,
    comparison_report: PerformanceReport,
) -> dict[str, Any]:
    """Compare two performance reports (standalone helper).
    
    Args:
        baseline_report: Baseline model report (e.g., Standard)
        comparison_report: Comparison model report (e.g., Performer)
    
    Returns:
        Comparison metrics including speedup and memory reduction
    """
    speedup = baseline_report.latency_mean / comparison_report.latency_mean
    latency_delta = baseline_report.latency_mean - comparison_report.latency_mean
    memory_delta = baseline_report.peak_memory_mean - comparison_report.peak_memory_mean
    memory_reduction_pct = (memory_delta / baseline_report.peak_memory_mean) * 100
    
    return {
        "speedup": speedup,
        "latency_delta_ms": latency_delta,
        "memory_delta_mb": memory_delta,
        "memory_reduction": memory_reduction_pct,
        "p95_speedup": baseline_report.latency_p95 / comparison_report.latency_p95,
        "faster": comparison_report.latency_mean < baseline_report.latency_mean,
        "less_memory": comparison_report.peak_memory_mean < baseline_report.peak_memory_mean,
    }


if __name__ == "__main__":
    main()
