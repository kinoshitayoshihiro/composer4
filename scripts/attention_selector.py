#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Attention Mechanism Auto-Selector (Phase 4.4)

Automatically selects the optimal attention mechanism based on:
- Sequence length
- Precision (FP32/FP16/BF16)
- Device (CPU/CUDA/MPS)

Usage:
    from attention_selector import AttentionSelector
    
    selector = AttentionSelector(threshold=1024)
    mechanism = selector.select(seq_len=2048, precision=torch.bfloat16, device=torch.device('cuda'))
    print(f"Selected: {mechanism}")  # 'flash', 'sdpa', or 'standard'
"""

import torch
from typing import Optional, Literal
import logging

AttentionType = Literal['standard', 'sdpa', 'flash']

logger = logging.getLogger(__name__)


class AttentionSelector:
    """
    Select optimal attention mechanism based on runtime conditions.
    
    Attributes:
        threshold: Sequence length threshold for efficient attention (default: 1024)
        force_mechanism: Override auto-selection (default: None)
    """
    
    def __init__(
        self,
        threshold: int = 1024,
        force_mechanism: Optional[AttentionType] = None,
        verbose: bool = True
    ):
        """
        Initialize AttentionSelector.
        
        Args:
            threshold: Sequence length threshold (N < threshold uses standard attention)
            force_mechanism: Force specific mechanism ('standard', 'sdpa', 'flash', or None)
            verbose: Enable logging
        """
        self.threshold = threshold
        self.force_mechanism = force_mechanism
        self.verbose = verbose
        
        # Check availability
        self._flash_available = self._check_flash_attn()
        self._sdpa_available = self._check_sdpa()
        
        if self.verbose:
            logger.info(f"AttentionSelector initialized (threshold={threshold})")
            logger.info(f"  Flash Attention available: {self._flash_available}")
            logger.info(f"  SDPA available: {self._sdpa_available}")
    
    def _check_flash_attn(self) -> bool:
        """Check if Flash Attention is available."""
        try:
            import flash_attn
            return True
        except ImportError:
            return False
    
    def _check_sdpa(self) -> bool:
        """Check if PyTorch SDPA is available."""
        return hasattr(torch.nn.functional, 'scaled_dot_product_attention')
    
    def select(
        self,
        seq_len: int,
        precision: torch.dtype,
        device: torch.device
    ) -> AttentionType:
        """
        Select optimal attention mechanism.
        
        Args:
            seq_len: Sequence length
            precision: torch.float32, torch.float16, or torch.bfloat16
            device: torch.device
        
        Returns:
            'standard', 'sdpa', or 'flash'
        """
        # Manual override
        if self.force_mechanism is not None:
            if self.verbose:
                logger.info(f"[attention] Force override: {self.force_mechanism}")
            return self.force_mechanism
        
        # Short sequences: always use standard (most efficient for small N)
        if seq_len < self.threshold:
            if self.verbose:
                logger.debug(f"[attention] seq_len={seq_len} < {self.threshold}, using standard")
            return 'standard'
        
        # CPU: no efficient attention available
        if device.type == 'cpu':
            if self.verbose:
                logger.info(f"[attention] CPU device, using standard")
            return 'standard'
        
        # Long sequences on GPU
        if device.type == 'cuda':
            # Flash Attention (best for BF16 + long sequences)
            if precision == torch.bfloat16 and self._flash_available:
                if self.verbose:
                    logger.info(f"[attention] BF16 + CUDA, using flash (seq_len={seq_len})")
                return 'flash'
            
            # SDPA (good for FP16/FP32 + long sequences)
            if self._sdpa_available:
                if self.verbose:
                    logger.info(f"[attention] CUDA, using sdpa (seq_len={seq_len})")
                return 'sdpa'
        
        # Fallback
        if self.verbose:
            logger.warning(f"[attention] No efficient attention available, using standard")
        return 'standard'
    
    def configure_model(
        self,
        model,
        mechanism: AttentionType,
        **kwargs
    ):
        """
        Configure model to use specified attention mechanism.
        
        Args:
            model: HuggingFace model instance
            mechanism: Attention mechanism to use
            **kwargs: Additional configuration
        
        Returns:
            Configured model
        """
        if mechanism == 'sdpa':
            # PyTorch 2.0+ SDPA
            if hasattr(model.config, 'attn_implementation'):
                model.config.attn_implementation = "sdpa"
                if self.verbose:
                    logger.info("[attention] Configured model for SDPA")
        
        elif mechanism == 'flash':
            # Flash Attention 2
            if hasattr(model.config, 'attn_implementation'):
                model.config.attn_implementation = "flash_attention_2"
                if self.verbose:
                    logger.info("[attention] Configured model for Flash Attention 2")
            else:
                logger.warning("[attention] Model does not support flash_attention_2 config")
        
        return model
    
    def get_memory_estimate(
        self,
        seq_len: int,
        mechanism: AttentionType,
        hidden_size: int = 768,
        num_heads: int = 12
    ) -> dict:
        """
        Estimate memory usage for different attention mechanisms.
        
        Args:
            seq_len: Sequence length
            mechanism: Attention mechanism
            hidden_size: Model hidden size
            num_heads: Number of attention heads
        
        Returns:
            Dictionary with memory estimates (in MB)
        """
        # Attention matrix size
        attn_matrix_size = seq_len * seq_len * num_heads * 4  # FP32 bytes
        
        # Hidden states
        hidden_states = seq_len * hidden_size * 4
        
        if mechanism == 'standard':
            # O(N²) memory for attention matrix
            total = attn_matrix_size + hidden_states
        elif mechanism == 'sdpa':
            # Reduced memory (~0.8x)
            total = attn_matrix_size * 0.8 + hidden_states
        elif mechanism == 'flash':
            # O(N) memory (~0.5x)
            total = attn_matrix_size * 0.5 + hidden_states
        else:
            total = 0
        
        return {
            'mechanism': mechanism,
            'total_mb': total / (1024 * 1024),
            'attention_mb': attn_matrix_size / (1024 * 1024),
            'hidden_mb': hidden_states / (1024 * 1024),
        }
    
    def benchmark(
        self,
        seq_lengths: list = [512, 1024, 2048, 4096],
        precision: torch.dtype = torch.float32,
        device: torch.device = torch.device('cuda')
    ) -> dict:
        """
        Benchmark attention mechanisms for different sequence lengths.
        
        Args:
            seq_lengths: List of sequence lengths to test
            precision: Precision to use
            device: Device to test on
        
        Returns:
            Dictionary with benchmark results
        """
        results = {}
        
        for seq_len in seq_lengths:
            mechanism = self.select(seq_len, precision, device)
            mem_est = self.get_memory_estimate(seq_len, mechanism)
            
            results[seq_len] = {
                'mechanism': mechanism,
                'memory_mb': mem_est['total_mb'],
                'precision': str(precision),
                'device': str(device),
            }
        
        return results
    
    def __repr__(self) -> str:
        return (
            f"AttentionSelector("
            f"threshold={self.threshold}, "
            f"force={self.force_mechanism}, "
            f"flash_avail={self._flash_available}, "
            f"sdpa_avail={self._sdpa_available})"
        )


def main():
    """Example usage and benchmarking."""
    import sys
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("=" * 60)
    print("Attention Mechanism Auto-Selector (Phase 4.4)")
    print("=" * 60)
    print()
    
    # Initialize selector
    selector = AttentionSelector(threshold=1024, verbose=True)
    print(f"Selector: {selector}")
    print()
    
    # Test cases
    test_cases = [
        (512, torch.float32, torch.device('cuda')),
        (1024, torch.float32, torch.device('cuda')),
        (2048, torch.float16, torch.device('cuda')),
        (2048, torch.bfloat16, torch.device('cuda')),
        (4096, torch.bfloat16, torch.device('cuda')),
        (1024, torch.float32, torch.device('cpu')),
    ]
    
    print("Test Cases:")
    print("-" * 60)
    for seq_len, precision, device in test_cases:
        mechanism = selector.select(seq_len, precision, device)
        mem = selector.get_memory_estimate(seq_len, mechanism)
        print(f"seq_len={seq_len:4d}, {str(precision):16s}, {str(device):12s} → {mechanism:8s} ({mem['total_mb']:.1f} MB)")
    
    print()
    
    # Benchmark
    print("Benchmark Results:")
    print("-" * 60)
    benchmark_results = selector.benchmark(
        seq_lengths=[512, 1024, 2048, 4096],
        precision=torch.bfloat16,
        device=torch.device('cuda')
    )
    
    for seq_len, result in benchmark_results.items():
        print(f"seq_len={seq_len:4d}: {result['mechanism']:8s} ({result['memory_mb']:.1f} MB)")
    
    print()
    print("=" * 60)


if __name__ == '__main__':
    main()
