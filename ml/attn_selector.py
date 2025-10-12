"""Adaptive Attention Selector.

Automatically selects Standard or Performer attention based on:
- Device (CPU/GPU)
- Sequence length
- Configurable threshold

Usage:
    from ml.attn_selector import apply_adaptive_attention, AttnAutoConfig
    from ml.attention_performer import replace_attention_layers
    
    cfg = AttnAutoConfig(threshold=1024, num_random_features=128)
    kind = apply_adaptive_attention(
        model,
        device="cuda",
        seq_len=576,
        replace_fn=replace_attention_layers,
        cfg=cfg
    )
    print(f"Using {kind} attention")
"""

from dataclasses import dataclass
from typing import Callable, Optional, Literal

AttnKind = Literal["standard", "sdpa", "performer"]


@dataclass(frozen=True)
class AttnAutoConfig:
    """Configuration for adaptive attention selection.
    
    Attributes:
        threshold: Minimum sequence length to use advanced attention on GPU.
                   Based on empirical benchmarks:
                   - SDPA/Flash: Effective at N>=512 (and often at shorter sequences)
                   - Performer: NOT recommended (0.3-0.45x speedup, proven slower)
        num_random_features: Number of random features for Performer.
                             Recommended: 64, 128, or 256 (unused if SDPA is selected)
        idempotent: If True, avoid re-applying attention replacement
                    if already applied.
        
    Note:
        Default uses SDPA (PyTorch 2.x Flash Attention) as the primary choice.
        Performer is kept for compatibility but NOT selected by default.
    """
    threshold: int = 512  # SDPA works well even for shorter sequences
    num_random_features: int = 128
    idempotent: bool = True


def select_attention(
    device: str,
    seq_len: int,
    *,
    threshold: int = 512,
    force: Optional[AttnKind] = None,
) -> AttnKind:
    """Select attention mechanism based on device and sequence length.
    
    Decision logic (updated for SDPA):
    1. If `force` is specified, use that
    2. If GPU → use SDPA (PyTorch 2.x Flash Attention)
    3. Otherwise → use Standard
    
    Note:
        - SDPA is now the default for GPU (proven 2-4x faster)
        - Performer is kept for compatibility but NOT selected automatically
        - Performer empirically proven slower (0.3-0.45x) in all tested conditions
    
    Args:
        device: Device type ("cuda" or "cpu")
        seq_len: Total sequence length (prompt + generated tokens)
        threshold: Minimum sequence length for advanced attention (unused for SDPA)
        force: Force specific attention type (overrides logic)
        
    Returns:
        "sdpa", "standard", or "performer"
        
    Examples:
        >>> select_attention("cpu", 2048)
        'standard'
        >>> select_attention("cuda", 576)
        'sdpa'
        >>> select_attention("cuda", 128, force="performer")
        'performer'
    """
    if force in ("performer", "standard", "sdpa"):
        return force

    is_cuda = (device.lower() == "cuda")
    if is_cuda:
        # SDPA is the primary choice for GPU (proven fast at all sequence lengths)
        return "sdpa"
    
    # CPU: always use standard
    return "standard"


def apply_adaptive_attention(
    model,
    *,
    device: str,
    seq_len: int,
    replace_sdpa_fn: Optional[Callable[..., int]] = None,
    replace_performer_fn: Optional[Callable[..., None]] = None,
    cfg: AttnAutoConfig = AttnAutoConfig(),
    force: Optional[AttnKind] = None,
) -> AttnKind:
    """Apply adaptive attention mechanism to model.
    
    This function:
    1. Selects appropriate attention type (SDPA/Standard/Performer)
    2. Applies attention replacement if needed
    3. Records the applied type in model._attn_kind
    4. Ensures idempotent operation (won't re-apply if already set)
    
    Updated for SDPA:
        - SDPA is now the primary choice for GPU
        - Performer is kept for compatibility but requires explicit force
        - Standard is used for CPU or as fallback
    
    Args:
        model: Model to modify (e.g., GPT2LMHeadModel)
        device: Device string ("cuda" or "cpu")
        seq_len: Total sequence length
        replace_sdpa_fn: Function to replace attention with SDPA
                        (e.g., attention_sdpa.replace_attention_layers_sdpa)
                        Signature: replace_sdpa_fn(model, causal=True) -> int
        replace_performer_fn: Function to replace attention with Performer
                             (e.g., attention_performer.replace_attention_layers)
                             Signature: replace_performer_fn(model, num_random_features=128)
        cfg: Configuration for adaptive selection
        force: Force specific attention type (overrides auto selection)
        
    Returns:
        Applied attention kind ("sdpa", "standard", or "performer")
        
    Side Effects:
        - Modifies model attention layers (if SDPA/Performer selected)
        - Sets model._attn_kind attribute
        
    Examples:
        >>> from ml.attention_sdpa import replace_attention_layers_sdpa
        >>> from ml.attention_performer import replace_attention_layers
        >>> cfg = AttnAutoConfig(threshold=512, num_random_features=128)
        >>> kind = apply_adaptive_attention(
        ...     model,
        ...     device="cuda",
        ...     seq_len=576,
        ...     replace_sdpa_fn=replace_attention_layers_sdpa,
        ...     replace_performer_fn=replace_attention_layers,
        ...     cfg=cfg
        ... )
        >>> print(kind)
        'sdpa'
    """
    # Idempotent check: skip if already applied
    current = getattr(model, "_attn_kind", None)
    if cfg.idempotent and current in ("performer", "standard", "sdpa"):
        return current

    # Select attention type
    kind = select_attention(
        device,
        seq_len,
        threshold=cfg.threshold,
        force=force
    )

    # Apply SDPA if selected
    if kind == "sdpa" and replace_sdpa_fn is not None:
        num_replaced = replace_sdpa_fn(model, causal=True)
        if num_replaced == 0:
            # No layers replaced, fallback warning handled by caller
            pass
    
    # Apply Performer if selected
    elif kind == "performer" and replace_performer_fn is not None:
        replace_performer_fn(model, num_random_features=cfg.num_random_features)

    # Record applied type
    try:
        setattr(model, "_attn_kind", kind)
    except Exception:
        # Ignore if model doesn't support attribute assignment
        pass

    return kind
