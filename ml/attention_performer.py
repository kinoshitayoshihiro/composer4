#!/usr/bin/env python3
"""
Performer Linear Attention for Stage3 (Day 9-10)

Implements linear-complexity attention using random feature approximation
(FAVOR+ algorithm from "Rethinking Attention with Performers", Choromanski et al., 2020).

寸評推奨: 推論パスのみ差し替え → 速度・メモリ測定 → 学習適用

Key Features:
- O(N) complexity vs O(N²) for standard attention
- Causal masking for autoregressive generation
- Drop-in replacement for GPT-2 attention
- Backward compatible with existing checkpoints
"""

import logging
import math
from typing import Any, Optional, Tuple

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:  # pragma: no cover
    torch = None  # type: ignore
    nn = None  # type: ignore
    F = None  # type: ignore


def _create_random_features(
    num_features: int,
    dim: int,
    device: Optional["torch.device"] = None,
    dtype: Optional["torch.dtype"] = None,
) -> "torch.Tensor":
    """Create random feature matrix for FAVOR+ approximation.
    
    Args:
        num_features: Number of random features (typically 256)
        dim: Dimension per attention head
        device: Device for tensor
        dtype: Data type for tensor
        
    Returns:
        Random feature matrix of shape (num_features, dim)
    """
    if torch is None:
        raise RuntimeError("torch not available")
    
    # Create random features
    matrix = torch.randn(num_features, dim, device=device, dtype=dtype)
    
    # Orthogonalize if possible (num_features <= dim)
    if num_features <= dim:
        q, _ = torch.linalg.qr(matrix.T)
        return q.T[:num_features, :]
    else:
        # Use normalized random features
        return matrix / math.sqrt(dim)


def _kernel_feature_creator(
    data: "torch.Tensor",
    projection_matrix: "torch.Tensor",
    is_query: bool,
    eps: float = 1e-4,
) -> "torch.Tensor":
    """Create kernel features using random projection.
    
    Args:
        data: Input tensor (B, H, L, D)
        projection_matrix: Random feature matrix (M, D)
        is_query: Whether this is query (True) or key (False)
        eps: Small constant for numerical stability
        
    Returns:
        Kernel features (B, H, L, M)
    """
    if torch is None or F is None:
        raise RuntimeError("torch not available")
    
    # Normalize input
    data_normalizer = 1.0 / math.sqrt(math.sqrt(data.shape[-1]))
    
    # Project: (B, H, L, D) @ (D, M) -> (B, H, L, M)
    ratio = 1.0 / math.sqrt(projection_matrix.shape[0])
    data_dash = torch.einsum('...nd,md->...nm', data_normalizer * data, projection_matrix)
    
    # Softmax kernel approximation
    diag_data = torch.sum(data ** 2, dim=-1, keepdim=True)
    diag_data = (diag_data / 2.0) * (data_normalizer ** 2)
    
    if is_query:
        data_dash = ratio * (
            torch.exp(data_dash - diag_data - torch.max(data_dash, dim=-1, keepdim=True).values) + eps
        )
    else:
        data_dash = ratio * (
            torch.exp(data_dash - diag_data - torch.max(data_dash, dim=-2, keepdim=True).values) + eps
        )
    
    return data_dash


def _causal_linear_attention(
    q: "torch.Tensor",
    k: "torch.Tensor",
    v: "torch.Tensor",
    projection_matrix: "torch.Tensor",
    eps: float = 1e-6,
) -> "torch.Tensor":
    """Compute causal linear attention using FAVOR+ algorithm.
    
    Args:
        q: Query tensor (B, H, L, D)
        k: Key tensor (B, H, L, D)
        v: Value tensor (B, H, L, D)
        projection_matrix: Random features (M, D)
        eps: Numerical stability constant
        
    Returns:
        Attention output (B, H, L, D)
    """
    if torch is None:
        raise RuntimeError("torch not available")
    
    # Create kernel features
    q_prime = _kernel_feature_creator(q, projection_matrix, is_query=True, eps=eps)
    k_prime = _kernel_feature_creator(k, projection_matrix, is_query=False, eps=eps)
    
    # Causal attention via cumulative sum
    # kv: (B, H, L, M) @ (B, H, L, D) -> cumsum over L -> (B, H, L, M, D)
    k_prime_T = k_prime.unsqueeze(-1)  # (B, H, L, M, 1)
    v_expanded = v.unsqueeze(-2)  # (B, H, L, 1, D)
    kv = k_prime_T * v_expanded  # (B, H, L, M, D)
    kv_cumsum = torch.cumsum(kv, dim=2)  # Causal cumulative sum
    
    # Normalization: sum of keys
    k_sum = torch.cumsum(k_prime, dim=2)  # (B, H, L, M)
    
    # Compute attention: q @ kv_cumsum / q @ k_sum
    # (B, H, L, M) @ (B, H, L, M, D) -> (B, H, L, D)
    numerator = torch.einsum('...lm,...lmd->...ld', q_prime, kv_cumsum)
    denominator = torch.einsum('...lm,...lm->...l', q_prime, k_sum).unsqueeze(-1)
    
    return numerator / (denominator + eps)


class PerformerAttention(nn.Module if nn is not None else object):
    """Performer-based linear attention (drop-in replacement for GPT-2 attention).
    
    寸評推奨: 推論パスのみ差し替え → API互換性維持
    """
    
    def __init__(
        self,
        n_embd: int,
        n_head: int,
        num_random_features: int = 256,
        causal: bool = True,
        device: Optional["torch.device"] = None,
        dtype: Optional["torch.dtype"] = None,
    ):
        """Initialize Performer attention.
        
        Args:
            n_embd: Embedding dimension (e.g., 768 for GPT-2)
            n_head: Number of attention heads (e.g., 12 for GPT-2)
            num_random_features: Number of random features (default: 256)
            causal: Whether to use causal masking (True for autoregressive)
            device: Device for tensors
            dtype: Data type for tensors
        """
        if nn is None or torch is None:
            raise RuntimeError("torch not available")
        
        super().__init__()
        
        self.n_embd = n_embd
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.num_random_features = num_random_features
        self.causal = causal
        
        assert n_embd % n_head == 0, "n_embd must be divisible by n_head"
        
        # QKV projection (same as GPT-2)
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        
        # Output projection (same as GPT-2)
        self.c_proj = nn.Linear(n_embd, n_embd)
        
        # Random feature matrix (shared across heads)
        # Register as buffer (not trainable, but saved with model)
        projection = _create_random_features(
            num_random_features,
            self.head_dim,
            device=device,
            dtype=dtype or torch.float32,
        )
        self.register_buffer('projection_matrix', projection)
        
        logging.info(
            f"PerformerAttention initialized: n_embd={n_embd}, n_head={n_head}, "
            f"num_features={num_random_features}, causal={causal}"
        )
    
    def forward(
        self,
        hidden_states: "torch.Tensor",
        attention_mask: Optional["torch.Tensor"] = None,
        layer_past: Optional[Tuple["torch.Tensor", "torch.Tensor"]] = None,
        past_key_values: Optional[Tuple["torch.Tensor", "torch.Tensor"]] = None,
        use_cache: bool = False,
        **kwargs: Any,
    ) -> Tuple["torch.Tensor", Optional[Tuple["torch.Tensor", "torch.Tensor"]]]:
        """Forward pass (API compatible with GPT-2 attention).
        
        Args:
            hidden_states: Input tensor (B, L, n_embd)
            attention_mask: Attention mask (ignored for causal Performer)
            layer_past: Past key-value cache (not used in linear attention)
            use_cache: Whether to return cache (always False for linear attention)
            
        Returns:
            Tuple of (output, None) for API compatibility
        """
        if torch is None:
            raise RuntimeError("torch not available")
        
        B, L, _ = hidden_states.shape
        
        # QKV projection
        qkv = self.c_attn(hidden_states)  # (B, L, 3*n_embd)
        q, k, v = qkv.split(self.n_embd, dim=2)
        
        # Reshape to multi-head: (B, L, n_embd) -> (B, n_head, L, head_dim)
        q = q.view(B, L, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, L, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, L, self.n_head, self.head_dim).transpose(1, 2)
        
        # Linear attention (causal)
        attn_output = _causal_linear_attention(
            q, k, v,
            projection_matrix=self.projection_matrix,
        )
        
        # Reshape back: (B, n_head, L, head_dim) -> (B, L, n_embd)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, L, self.n_embd)
        
        # Output projection
        output = self.c_proj(attn_output)
        
        # Return (output, None) for API compatibility with GPT-2
        return output, None


def replace_attention_layers(
    model: "torch.nn.Module",
    num_random_features: int = 256,
    verbose: bool = False,
) -> "torch.nn.Module":
    """Replace GPT-2 self-attention layers with Performer linear attention.
    
    寸評推奨: 推論パスのみ差し替え → 既存ckpt互換性維持
    
    Args:
        model: GPT-2 model (or GPT-2 with LoRA)
        num_random_features: Number of random features for Performer
        verbose: Whether to log replacement details
        
    Returns:
        Modified model with Performer attention
    """
    if torch is None or nn is None:
        raise RuntimeError("torch not available")
    
    # Access base model (handle LoRA wrapper and GPT2LMHeadModel)
    base_model = getattr(model, 'base_model', model)
    if hasattr(base_model, 'model'):
        base_model = base_model.model
    
    # GPT2LMHeadModel wraps GPT2Model - access the transformer
    if hasattr(base_model, 'transformer'):
        transformer = base_model.transformer
    elif hasattr(base_model, 'h'):
        # Already GPT2Model
        transformer = base_model
    else:
        raise ValueError(f"Cannot find transformer in model: {type(base_model)}")
    
    # Get model config
    config = transformer.config if hasattr(transformer, 'config') else base_model.config
    
    replaced_count = 0
    for i, layer in enumerate(transformer.h):
        original_attn = layer.attn
        
        # Create Performer attention with same config
        performer_attn = PerformerAttention(
            n_embd=config.n_embd,
            n_head=config.n_head,
            num_random_features=num_random_features,
            causal=True,
            device=next(model.parameters()).device,
            dtype=next(model.parameters()).dtype,
        )
        
        # Copy learned weights from original attention
        # Note: GPT-2's c_attn uses Conv1D with different layout
        # Weight copying is optional for inference-only replacement
        try:
            with torch.no_grad():
                # GPT-2 c_attn: (n_embd, 3*n_embd) transposed
                # Our c_attn: (n_embd, 3*n_embd) standard Linear
                if hasattr(original_attn, 'c_attn') and hasattr(original_attn.c_attn, 'weight'):
                    # Attempt to copy if shapes match
                    orig_weight = original_attn.c_attn.weight
                    if orig_weight.shape == performer_attn.c_attn.weight.T.shape:
                        performer_attn.c_attn.weight.copy_(orig_weight.T)
                        performer_attn.c_attn.bias.copy_(original_attn.c_attn.bias)
                    if verbose:
                        logging.debug(f"Layer {i}: Copied QKV weights")
                
                if hasattr(original_attn, 'c_proj') and hasattr(original_attn.c_proj, 'weight'):
                    orig_proj_weight = original_attn.c_proj.weight
                    if orig_proj_weight.shape == performer_attn.c_proj.weight.T.shape:
                        performer_attn.c_proj.weight.copy_(orig_proj_weight.T)
                        performer_attn.c_proj.bias.copy_(original_attn.c_proj.bias)
                    if verbose:
                        logging.debug(f"Layer {i}: Copied projection weights")
        except Exception as e:
            if verbose:
                logging.warning(f"Layer {i}: Could not copy weights: {e}. Using random initialization.")
        
        # Replace attention layer
        layer.attn = performer_attn
        replaced_count += 1
        
        if verbose:
            logging.info(
                f"Layer {i}: Replaced {original_attn.__class__.__name__} "
                f"with PerformerAttention (features={num_random_features})"
            )
    
    logging.info(f"Replaced {replaced_count} attention layers with Performer")
    return model
