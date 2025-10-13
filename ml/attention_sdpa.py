"""PyTorch SDPA (Scaled Dot-Product Attention) Implementation.

Uses F.scaled_dot_product_attention from PyTorch 2.x, which automatically
selects the best kernel:
- Flash Attention (if available, GPU with fp16/bfloat16)
- Memory-efficient attention (fallback)
- Math attention (CPU/fallback)

This provides significant speedup (2-4x) and memory reduction (20-40%)
compared to standard attention, especially for long sequences.

Usage:
    from ml.attention_sdpa import replace_attention_layers_sdpa
    
    # Replace attention in model
    num_replaced = replace_attention_layers_sdpa(model, causal=True)
    print(f"Replaced {num_replaced} attention layers with SDPA")

Requirements:
    - PyTorch >= 2.0
    - CUDA GPU with Flash Attention support (recommended)
    - fp16/bfloat16 + autocast for best performance
"""

from __future__ import annotations

from typing import Iterable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class SDPAAttn(nn.Module):
    """Scaled-Dot-Product Attention using PyTorch 2.x SDPA.
    
    Automatically selects the best backend:
    - Flash Attention: 2-4x faster, 20-40% less memory (GPU, fp16/bfloat16)
    - Memory-efficient: Good for long sequences
    - Math: CPU/fallback
    
    Args:
        causal: If True, use causal masking (autoregressive)
        dropout_p: Dropout probability (0.0 for inference)
        
    Example:
        >>> attn = SDPAAttn(causal=True, dropout_p=0.0)
        >>> q = torch.randn(2, 8, 512, 64)  # (B, H, T, D)
        >>> k = torch.randn(2, 8, 512, 64)
        >>> v = torch.randn(2, 8, 512, 64)
        >>> out = attn(q, k, v)  # (2, 8, 512, 64)
    """
    
    def __init__(self, causal: bool = True, dropout_p: float = 0.0):
        super().__init__()
        self.causal = causal
        self.dropout_p = dropout_p
        
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass with SDPA.
        
        Args:
            q: Query tensor (B, H, T, D)
            k: Key tensor (B, H, T, D)
            v: Value tensor (B, H, T, D)
            attn_mask: Optional attention mask (None recommended for packed)
            
        Returns:
            Output tensor (B, H, T, D)
        """
        return F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.dropout_p,
            is_causal=self.causal
        )


def _iter_attn_cores(m: nn.Module) -> Iterable[tuple[nn.Module, str]]:
    """Iterate over attention core modules in the model.
    
    Looks for common attribute names:
    - attn_core
    - self_attn_core
    - attention_core
    
    Compatible with Performer's replace_attention_layers() pattern.
    
    Args:
        m: Model to search
        
    Yields:
        (module, attribute_name) tuples
    """
    for mod in m.modules():
        for name in ("attn_core", "self_attn_core", "attention_core"):
            if hasattr(mod, name) and isinstance(getattr(mod, name), nn.Module):
                yield mod, name


def replace_attention_layers_sdpa(
    model: nn.Module,
    *,
    causal: bool = True,
    dropout_p: float = 0.0,
) -> int:
    """Replace attention layers with SDPA implementation.
    
    Args:
        model: Model to modify (in-place)
        causal: Use causal masking (True for autoregressive)
        dropout_p: Dropout probability (0.0 for inference)
        
    Returns:
        Number of layers replaced
        
    Example:
        >>> from transformers import GPT2LMHeadModel
        >>> model = GPT2LMHeadModel.from_pretrained("gpt2")
        >>> num = replace_attention_layers_sdpa(model, causal=True)
        >>> print(f"Replaced {num} layers")
    """
    n = 0
    for mod, name in _iter_attn_cores(model):
        setattr(mod, name, SDPAAttn(causal=causal, dropout_p=dropout_p))
        n += 1
    return n


def sdpa_kernel_availability() -> dict[str, bool]:
    """Check which SDPA kernels are available.
    
    Returns:
        Dictionary with kernel availability:
        - flash: Flash Attention (fastest, GPU fp16/bfloat16)
        - mem_efficient: Memory-efficient attention
        - math: Math attention (CPU/fallback)
        
    Example:
        >>> avail = sdpa_kernel_availability()
        >>> print(avail)
        {'flash': True, 'mem_efficient': True, 'math': True}
    """
    avail = {}
    try:
        # Test each backend by attempting to use it
        if torch.cuda.is_available() and hasattr(torch.backends.cuda, 'sdp_kernel'):
            from torch.backends.cuda import sdp_kernel, SDPBackend
            
            # Create test tensors on GPU
            q = torch.randn(1, 2, 4, 8, device='cuda', dtype=torch.float16)
            k = q.clone()
            v = q.clone()
            
            # Test Flash Attention
            try:
                with sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
                    _ = F.scaled_dot_product_attention(q, k, v)
                avail["flash"] = True
            except Exception:
                avail["flash"] = False
            
            # Test Memory Efficient
            try:
                with sdp_kernel(enable_flash=False, enable_math=False, enable_mem_efficient=True):
                    _ = F.scaled_dot_product_attention(q, k, v)
                avail["mem_efficient"] = True
            except Exception:
                avail["mem_efficient"] = False
            
            # Test Math
            try:
                with sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False):
                    _ = F.scaled_dot_product_attention(q, k, v)
                avail["math"] = True
            except Exception:
                avail["math"] = False
                
        elif hasattr(F, 'scaled_dot_product_attention'):
            # CUDA not available but SDPA exists (CPU only, math backend)
            avail = {"flash": False, "mem_efficient": False, "math": True}
        else:
            # PyTorch < 2.0 or SDPA not available
            avail = {"flash": False, "mem_efficient": False, "math": False}
    except Exception as e:
        # Fallback
        avail = {"flash": False, "mem_efficient": False, "math": False, "error": str(e)}
    return avail


def log_sdpa_backend_info() -> None:
    """Log information about available SDPA backends.
    
    Useful for debugging and performance optimization.
    """
    print("=" * 60)
    print("PyTorch SDPA Backend Availability")
    print("=" * 60)
    
    avail = sdpa_kernel_availability()
    
    for kernel, is_avail in avail.items():
        if kernel == "error":
            print(f"⚠️  Error: {is_avail}")
            continue
        
        icon = "✅" if is_avail else "❌"
        print(f"{icon} {kernel.upper()}: {is_avail}")
    
    print("=" * 60)
    
    # Recommendations
    if avail.get("flash"):
        print("🚀 Flash Attention available! Use fp16/bfloat16 + autocast for best performance.")
    elif avail.get("mem_efficient"):
        print("⚡ Memory-efficient attention available. Good for long sequences.")
    else:
        print("💡 Using math fallback. Consider upgrading PyTorch >= 2.0 for speedup.")
    
    print("=" * 60)
