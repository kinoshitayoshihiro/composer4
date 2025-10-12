#!/usr/bin/env python3
"""Real-time Performer Linear Attention Benchmark.

This script measures ACTUAL inference performance (not dummy metrics)
between Standard and Performer attention on Stage3 models.

Usage:
    python scripts/benchmark_performer_realtime.py \\
        --num-samples 20 \\
        --max-new-tokens 512 \\
        --device cuda \\
        --output results/performer_realtime_benchmark.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import torch
from transformers import GPT2Config, GPT2LMHeadModel

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ml.attention_performer import replace_attention_layers

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def create_dummy_model(n_embd: int = 768, n_head: int = 12, n_layer: int = 12) -> GPT2LMHeadModel:
    """Create a dummy GPT-2 model for benchmarking."""
    config = GPT2Config(
        vocab_size=1000,
        n_positions=2048,
        n_embd=n_embd,
        n_layer=n_layer,
        n_head=n_head,
    )
    model = GPT2LMHeadModel(config)
    return model


def measure_inference_time(
    model: GPT2LMHeadModel,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    device: str,
) -> tuple[float, int, float, float]:
    """Measure actual inference time and memory.
    
    Returns:
        (latency_ms, total_length, peak_memory_mb, avg_memory_mb)
    """
    model.eval()
    
    # GPU synchronization if available
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        start_memory = torch.cuda.memory_allocated() / (1024 ** 2)
    else:
        start_memory = 0.0
    
    # Measure inference time
    start_time = time.time()
    
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.9,
            top_p=0.95,
            pad_token_id=0,
            use_cache=False,  # Disable KV cache for Performer compatibility
        )
    
    # GPU synchronization for accurate timing
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()
    
    elapsed_ms = (time.time() - start_time) * 1000
    
    # Measure memory
    if device == "cuda" and torch.cuda.is_available():
        peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
        current_memory_mb = torch.cuda.memory_allocated() / (1024 ** 2)
        avg_memory_mb = (start_memory + current_memory_mb) / 2
    else:
        peak_memory_mb = 0.0
        avg_memory_mb = 0.0
    
    total_length = output.shape[1]
    
    return elapsed_ms, total_length, peak_memory_mb, avg_memory_mb


def benchmark_model(
    model: GPT2LMHeadModel,
    model_type: str,
    num_samples: int,
    prompt_length: int,
    max_new_tokens: int,
    device: str,
) -> dict:
    """Run real-time inference benchmark on a model.
    
    Returns:
        Performance statistics dict
    """
    model.eval()
    model.to(device)
    
    logger.info(f"🚀 Benchmarking {model_type} attention ({num_samples} samples)")
    logger.info(f"   Device: {device}")
    logger.info(f"   Prompt length: {prompt_length}")
    logger.info(f"   Max new tokens: {max_new_tokens}")
    
    latencies = []
    sequence_lengths = []
    peak_memories = []
    avg_memories = []
    
    for i in range(num_samples):
        # Create random input
        input_ids = torch.randint(0, 1000, (1, prompt_length), device=device)
        
        # Measure inference
        latency_ms, total_length, peak_mem_mb, avg_mem_mb = measure_inference_time(
            model=model,
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            device=device,
        )
        
        latencies.append(latency_ms)
        sequence_lengths.append(total_length)
        peak_memories.append(peak_mem_mb)
        avg_memories.append(avg_mem_mb)
        
        if (i + 1) % 5 == 0:
            logger.info(f"   Progress: {i + 1}/{num_samples} (avg latency: {sum(latencies) / len(latencies):.2f}ms)")
    
    # Calculate statistics
    import statistics
    
    latency_mean = statistics.mean(latencies)
    latency_median = statistics.median(latencies)
    latency_p95 = statistics.quantiles(latencies, n=20)[18] if len(latencies) >= 20 else max(latencies)
    latency_p99 = statistics.quantiles(latencies, n=100)[98] if len(latencies) >= 100 else max(latencies)
    
    per_token_latencies = [lat / max(1, seq - prompt_length) for lat, seq in zip(latencies, sequence_lengths)]
    per_token_mean = statistics.mean(per_token_latencies)
    
    seq_length_max = max(sequence_lengths)
    seq_length_avg = statistics.mean(sequence_lengths)
    seq_length_p95 = int(statistics.quantiles(sequence_lengths, n=20)[18]) if len(sequence_lengths) >= 20 else max(sequence_lengths)
    
    peak_memory_mean = statistics.mean(peak_memories) if any(peak_memories) else 0.0
    peak_memory_p95 = statistics.quantiles(peak_memories, n=20)[18] if len(peak_memories) >= 20 and any(peak_memories) else (max(peak_memories) if peak_memories else 0.0)
    
    return {
        "model_type": model_type,
        "total_runs": num_samples,
        "latency_mean": latency_mean,
        "latency_median": latency_median,
        "latency_p95": latency_p95,
        "latency_p99": latency_p99,
        "per_token_mean": per_token_mean,
        "max_sequence_length": seq_length_max,
        "avg_sequence_length": seq_length_avg,
        "p95_sequence_length": seq_length_p95,
        "peak_memory_mean": peak_memory_mean,
        "peak_memory_p95": peak_memory_p95,
        "raw_latencies": latencies,
        "raw_sequence_lengths": sequence_lengths,
    }


def main() -> None:
    """Main benchmark execution."""
    parser = argparse.ArgumentParser(description="Real-time Performer vs Standard benchmark")
    parser.add_argument("--num-samples", type=int, default=20, help="Number of samples")
    parser.add_argument("--prompt-length", type=int, default=64, help="Prompt length")
    parser.add_argument("--max-new-tokens", type=int, default=512, help="Max new tokens")
    parser.add_argument("--num-random-features", type=int, default=256, help="Performer random features")
    parser.add_argument("--output", type=str, default="results/performer_realtime_benchmark.json")
    parser.add_argument("--n-embd", type=int, default=768, help="Embedding dimension")
    parser.add_argument("--n-layer", type=int, default=12, help="Number of layers")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"], help="Device")
    args = parser.parse_args()
    
    # Check CUDA availability
    if args.device == "cuda" and not torch.cuda.is_available():
        logger.warning("⚠️  CUDA not available, falling back to CPU")
        args.device = "cpu"
    
    logger.info("=" * 80)
    logger.info("🎯 Performer Linear Attention Real-time Benchmark")
    logger.info("=" * 80)
    logger.info(f"Device: {args.device}")
    if args.device == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
    
    # Create models
    logger.info(f"🔧 Creating dummy GPT-2 model (n_embd={args.n_embd}, n_layer={args.n_layer})")
    standard_model = create_dummy_model(n_embd=args.n_embd, n_layer=args.n_layer)
    
    logger.info(f"🎭 Creating Performer model (num_random_features={args.num_random_features})")
    performer_model = GPT2LMHeadModel(standard_model.config)
    performer_model.load_state_dict(standard_model.state_dict())
    replace_attention_layers(performer_model, num_random_features=args.num_random_features)
    logger.info("✅ Attention layers replaced with Performer")
    
    # Benchmark Standard
    logger.info("")
    logger.info("=" * 80)
    logger.info("📊 Benchmark 1/2: Standard GPT-2 Attention")
    logger.info("=" * 80)
    standard_stats = benchmark_model(
        model=standard_model,
        model_type="standard",
        num_samples=args.num_samples,
        prompt_length=args.prompt_length,
        max_new_tokens=args.max_new_tokens,
        device=args.device,
    )
    
    # Benchmark Performer
    logger.info("")
    logger.info("=" * 80)
    logger.info("📊 Benchmark 2/2: Performer Linear Attention")
    logger.info("=" * 80)
    performer_stats = benchmark_model(
        model=performer_model,
        model_type="performer",
        num_samples=args.num_samples,
        prompt_length=args.prompt_length,
        max_new_tokens=args.max_new_tokens,
        device=args.device,
    )
    
    # Compare results
    logger.info("")
    logger.info("=" * 80)
    logger.info("📈 Performance Report")
    logger.info("=" * 80)
    
    logger.info("")
    logger.info("🔵 Standard Attention:")
    logger.info(f"   Latency (mean):     {standard_stats['latency_mean']:.2f} ms")
    logger.info(f"   Latency (p95):      {standard_stats['latency_p95']:.2f} ms")
    logger.info(f"   Per-token (mean):   {standard_stats['per_token_mean']:.2f} ms")
    logger.info(f"   Sequence (max):     {standard_stats['max_sequence_length']}")
    if args.device == "cuda":
        logger.info(f"   Memory (peak):      {standard_stats['peak_memory_mean']:.2f} MB")
    
    logger.info("")
    logger.info("🟢 Performer Attention:")
    logger.info(f"   Latency (mean):     {performer_stats['latency_mean']:.2f} ms")
    logger.info(f"   Latency (p95):      {performer_stats['latency_p95']:.2f} ms")
    logger.info(f"   Per-token (mean):   {performer_stats['per_token_mean']:.2f} ms")
    logger.info(f"   Sequence (max):     {performer_stats['max_sequence_length']}")
    if args.device == "cuda":
        logger.info(f"   Memory (peak):      {performer_stats['peak_memory_mean']:.2f} MB")
    
    # Comparison
    logger.info("")
    logger.info("=" * 80)
    logger.info("🎯 Comparison: Performer vs Standard")
    logger.info("=" * 80)
    
    speedup = standard_stats['latency_mean'] / performer_stats['latency_mean']
    latency_delta = standard_stats['latency_mean'] - performer_stats['latency_mean']
    
    if args.device == "cuda":
        memory_delta = standard_stats['peak_memory_mean'] - performer_stats['peak_memory_mean']
        memory_reduction_pct = (memory_delta / standard_stats['peak_memory_mean']) * 100 if standard_stats['peak_memory_mean'] > 0 else 0.0
    else:
        memory_delta = 0.0
        memory_reduction_pct = 0.0
    
    speedup_icon = "🚀" if speedup > 1.1 else ("🐌" if speedup < 0.9 else "➡️")
    memory_icon = "💚" if memory_reduction_pct > 10 else ("💛" if memory_reduction_pct > 0 else "➡️")
    
    logger.info(f"{speedup_icon} Speedup:          {speedup:.2f}x")
    logger.info(f"   Latency delta:    {latency_delta:+.2f} ms")
    if args.device == "cuda":
        logger.info(f"{memory_icon} Memory reduction: {memory_reduction_pct:+.1f}%")
        logger.info(f"   Memory delta:     {memory_delta:+.2f} MB")
    
    # Save report
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    report_data = {
        "benchmark_config": {
            "num_samples": args.num_samples,
            "prompt_length": args.prompt_length,
            "max_new_tokens": args.max_new_tokens,
            "num_random_features": args.num_random_features,
            "n_embd": args.n_embd,
            "n_layer": args.n_layer,
            "device": args.device,
        },
        "standard": {k: v for k, v in standard_stats.items() if k != "raw_latencies" and k != "raw_sequence_lengths"},
        "performer": {k: v for k, v in performer_stats.items() if k != "raw_latencies" and k != "raw_sequence_lengths"},
        "comparison": {
            "speedup": speedup,
            "latency_delta_ms": latency_delta,
            "memory_delta_mb": memory_delta,
            "memory_reduction_pct": memory_reduction_pct,
        },
    }
    
    with output_path.open("w") as f:
        json.dump(report_data, f, indent=2)
    
    logger.info("")
    logger.info(f"💾 Report saved to: {output_path}")
    logger.info("")
    logger.info("=" * 80)
    logger.info("✅ Benchmark complete!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
