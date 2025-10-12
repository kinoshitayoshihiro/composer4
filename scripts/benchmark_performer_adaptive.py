#!/usr/bin/env python3
"""Adaptive Attention Benchmark - SDPA/Standard/Performer.

This script uses the Adaptive Attention Selector to automatically choose
the best attention mechanism based on device and sequence length.

Updated for SDPA (PyTorch 2.x Flash Attention):
- SDPA is now the default for GPU (proven 2-4x faster)
- Performer is kept for compatibility but NOT selected by default
- Standard is used for CPU or as fallback

Usage:
    # Auto mode (SDPA for GPU, Standard for CPU)
    python scripts/benchmark_performer_adaptive.py \\
        --device cuda \\
        --num-samples 10 \\
        --prompt-length 64 \\
        --max-new-tokens 512 \\
        --attn auto \\
        --output results/adaptive_attn_n576.json

    # Force SDPA (Flash Attention)
    python scripts/benchmark_performer_adaptive.py \\
        --device cuda \\
        --attn sdpa \\
        --output results/forced_sdpa.json

    # Force Standard
    python scripts/benchmark_performer_adaptive.py \\
        --device cuda \\
        --attn standard \\
        --output results/forced_standard.json
        
    # Force Performer (experimental, proven slower)
    python scripts/benchmark_performer_adaptive.py \\
        --device cuda \\
        --attn performer \\
        --output results/forced_performer.json
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

from ml.attention_performer import replace_attention_layers as replace_attention_layers_performer
from ml.attention_sdpa import replace_attention_layers_sdpa, log_sdpa_backend_info
from ml.attn_selector import apply_adaptive_attention, AttnAutoConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def create_dummy_model(n_embd: int = 768, n_head: int = 12, n_layer: int = 12) -> GPT2LMHeadModel:
    """Create a dummy GPT-2 model for benchmarking.
    
    Note: n_embd must be divisible by n_head.
    """
    # Auto-adjust n_head if needed
    if n_embd % n_head != 0:
        # Find largest divisor <= 12
        for h in range(min(12, n_embd), 0, -1):
            if n_embd % h == 0:
                n_head = h
                break
    
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


def run_benchmark(
    model: GPT2LMHeadModel,
    num_samples: int,
    prompt_length: int,
    max_new_tokens: int,
    device: str,
) -> dict:
    """Run inference benchmark on a model.
    
    Returns:
        Performance statistics dict
    """
    model.eval()
    model.to(device)
    
    logger.info(f"🚀 Running benchmark ({num_samples} samples)")
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
    
    per_token_latencies = [lat / max(1, seq - prompt_length) for lat, seq in zip(latencies, sequence_lengths)]
    per_token_mean = statistics.mean(per_token_latencies)
    
    seq_length_max = max(sequence_lengths)
    seq_length_avg = statistics.mean(sequence_lengths)
    
    peak_memory_mean = statistics.mean(peak_memories) if any(peak_memories) else 0.0
    peak_memory_p95 = statistics.quantiles(peak_memories, n=20)[18] if len(peak_memories) >= 20 and any(peak_memories) else (max(peak_memories) if peak_memories else 0.0)
    
    return {
        "total_runs": num_samples,
        "latency_mean": latency_mean,
        "latency_median": latency_median,
        "latency_p95": latency_p95,
        "per_token_mean": per_token_mean,
        "max_sequence_length": seq_length_max,
        "avg_sequence_length": seq_length_avg,
        "peak_memory_mean": peak_memory_mean,
        "peak_memory_p95": peak_memory_p95,
        "raw_latencies": latencies,
        "raw_sequence_lengths": sequence_lengths,
    }


def main() -> None:
    """Main benchmark execution with adaptive attention selection."""
    parser = argparse.ArgumentParser(description="Adaptive Attention Benchmark (SDPA/Standard/Performer)")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="Device")
    parser.add_argument("--num-samples", type=int, default=10, help="Number of samples")
    parser.add_argument("--prompt-length", type=int, default=64, help="Prompt length")
    parser.add_argument("--max-new-tokens", type=int, default=512, help="Max new tokens")
    parser.add_argument("--n-embd", type=int, default=768, help="Embedding dimension")
    parser.add_argument("--n-layer", type=int, default=12, help="Number of layers")
    parser.add_argument("--num-random-features", type=int, default=128, help="Performer random features")
    parser.add_argument("--attn", choices=["auto", "standard", "sdpa", "performer"], default="auto", 
                        help="Attention selection: auto (SDPA for GPU), standard, sdpa (Flash), performer (slow)")
    parser.add_argument("--attn-threshold", type=int, default=512, help="Threshold for auto mode (unused for SDPA)")
    parser.add_argument("--output", required=True, help="Output JSON file")
    args = parser.parse_args()
    
    # Check CUDA availability
    if args.device == "cuda" and not torch.cuda.is_available():
        logger.warning("⚠️  CUDA not available, falling back to CPU")
        args.device = "cpu"
    
    seq_len = args.prompt_length + args.max_new_tokens
    
    logger.info("=" * 80)
    logger.info("🎯 Adaptive Attention Benchmark (SDPA/Standard/Performer)")
    logger.info("=" * 80)
    logger.info(f"Device: {args.device}")
    if args.device == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
        
        # Log SDPA backend availability
        logger.info("")
        log_sdpa_backend_info()
    
    logger.info(f"Sequence length: {seq_len} (prompt: {args.prompt_length}, new: {args.max_new_tokens})")
    logger.info(f"Attention mode: {args.attn}")
    if args.attn == "auto":
        logger.info(f"   Threshold: {args.attn_threshold}")
    
    # Create model
    logger.info("")
    logger.info(f"🔧 Creating GPT-2 model (n_embd={args.n_embd}, n_layer={args.n_layer})")
    model = create_dummy_model(n_embd=args.n_embd, n_layer=args.n_layer)
    
    # Apply adaptive attention
    logger.info("")
    logger.info("🎭 Applying Adaptive Attention Selector...")
    
    if args.attn == "auto":
        cfg = AttnAutoConfig(
            threshold=args.attn_threshold,
            num_random_features=args.num_random_features,
            idempotent=True
        )
        applied_kind = apply_adaptive_attention(
            model,
            device=args.device,
            seq_len=seq_len,
            replace_sdpa_fn=replace_attention_layers_sdpa,
            replace_performer_fn=replace_attention_layers_performer,
            cfg=cfg
        )
    elif args.attn == "sdpa":
        num_replaced = replace_attention_layers_sdpa(model, causal=True)
        applied_kind = "sdpa"
        logger.info(f"   SDPA replaced {num_replaced} layers")
        try:
            setattr(model, "_attn_kind", "sdpa")
        except Exception:
            pass
    elif args.attn == "performer":
        replace_attention_layers_performer(model, num_random_features=args.num_random_features)
        applied_kind = "performer"
        try:
            setattr(model, "_attn_kind", "performer")
        except Exception:
            pass
    else:  # standard
        applied_kind = "standard"
        try:
            setattr(model, "_attn_kind", "standard")
        except Exception:
            pass
    
    logger.info(f"✅ Attention selected: {applied_kind.upper()}")
    logger.info(f"   Device: {args.device}")
    logger.info(f"   Seq len: {seq_len}")
    if applied_kind == "performer":
        logger.info(f"   Random features: {args.num_random_features}")
        logger.info(f"   ⚠️  WARNING: Performer proven 2-3x SLOWER in empirical tests")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
    logger.info(f"Sequence length: {seq_len} (prompt: {args.prompt_length}, new: {args.max_new_tokens})")
    logger.info(f"Attention mode: {args.attn}")
    if args.attn == "auto":
        logger.info(f"   Threshold: {args.attn_threshold}")
        logger.info(f"   Random features: {args.num_random_features}")
    
    # Create model
    logger.info("")
    logger.info(f"🔧 Creating GPT-2 model (n_embd={args.n_embd}, n_layer={args.n_layer})")
    model = create_dummy_model(n_embd=args.n_embd, n_layer=args.n_layer)
    
    # Apply adaptive attention
    logger.info("")
    logger.info("🎭 Applying Adaptive Attention Selector...")
    
    if args.attn == "auto":
        cfg = AttnAutoConfig(
            threshold=args.attn_threshold,
            num_random_features=args.num_random_features,
            idempotent=True
        )
        applied_kind = apply_adaptive_attention(
            model,
            device=args.device,
            seq_len=seq_len,
            replace_fn=replace_attention_layers,
            cfg=cfg
        )
    elif args.attn == "performer":
        replace_attention_layers(model, num_random_features=args.num_random_features)
        applied_kind = "performer"
        try:
            setattr(model, "_attn_kind", "performer")
        except Exception:
            pass
    else:  # standard
        applied_kind = "standard"
        try:
            setattr(model, "_attn_kind", "standard")
        except Exception:
            pass
    
    logger.info(f"✅ Attention selected: {applied_kind.upper()}")
    logger.info(f"   Device: {args.device}")
    logger.info(f"   Seq len: {seq_len}")
    if applied_kind == "performer":
        logger.info(f"   Random features: {args.num_random_features}")
    
    # Run benchmark
    logger.info("")
    logger.info("=" * 80)
    logger.info(f"📊 Running Benchmark ({applied_kind.upper()} attention)")
    logger.info("=" * 80)
    
    result = run_benchmark(
        model=model,
        num_samples=args.num_samples,
        prompt_length=args.prompt_length,
        max_new_tokens=args.max_new_tokens,
        device=args.device,
    )
    
    # Print results
    logger.info("")
    logger.info("=" * 80)
    logger.info("📈 Benchmark Results")
    logger.info("=" * 80)
    logger.info(f"Attention type:     {applied_kind.upper()}")
    logger.info(f"Latency (mean):     {result['latency_mean']:.2f} ms")
    logger.info(f"Latency (p95):      {result['latency_p95']:.2f} ms")
    logger.info(f"Per-token (mean):   {result['per_token_mean']:.2f} ms")
    logger.info(f"Sequence (max):     {result['max_sequence_length']}")
    if args.device == "cuda":
        logger.info(f"Memory (peak):      {result['peak_memory_mean']:.2f} MB")
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    meta = {
        "attn": applied_kind,
        "device": args.device,
        "seq_len": seq_len,
        "num_random_features": args.num_random_features if applied_kind == "performer" else None,
        "threshold": args.attn_threshold,
        "mode": args.attn,
    }
    
    report_data = {
        "adaptive_meta": meta,
        "benchmark_config": {
            "num_samples": args.num_samples,
            "prompt_length": args.prompt_length,
            "max_new_tokens": args.max_new_tokens,
            "n_embd": args.n_embd,
            "n_layer": args.n_layer,
        },
        "results": {k: v for k, v in result.items() if k not in ["raw_latencies", "raw_sequence_lengths"]},
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
