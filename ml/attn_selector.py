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

AttnKind = Literal["standard", "performer"]


@dataclass(frozen=True)
class AttnAutoConfig:
    """Configuration for adaptive attention selection.
    
    Attributes:
        threshold: Minimum sequence length to use Performer on GPU.
                   Based on empirical benchmarks:
                   - N=576: Standard is 2.2-3.3x faster
                   - N=1024: Recommended threshold
        num_random_features: Number of random features for Performer.
                             Recommended: 64, 128, or 256
        idempotent: If True, avoid re-applying attention replacement
                    if already applied.
    """
    threshold: int = 1024
    num_random_features: int = 128
    idempotent: bool = True


def select_attention(
    device: str,
    seq_len: int,
    *,
    threshold: int = 1024,
    force: Optional[AttnKind] = None,
) -> AttnKind:
    """Select attention mechanism based on device and sequence length.
    
    Decision logic:
    1. If `force` is specified, use that
    2. If GPU and seq_len >= threshold, use Performer
    3. Otherwise, use Standard
    
    Args:
        device: Device type ("cuda" or "cpu")
        seq_len: Total sequence length (prompt + generated tokens)
        threshold: Minimum sequence length for Performer on GPU
        force: Force specific attention type (overrides logic)
        
    Returns:
        "performer" or "standard"
        
    Examples:
        >>> select_attention("cpu", 2048)
        'standard'
        >>> select_attention("cuda", 576, threshold=1024)
        'standard'
        >>> select_attention("cuda", 1024, threshold=1024)
        'performer'
        >>> select_attention("cuda", 128, force="performer")
        'performer'
    """
    if force in ("performer", "standard"):
        return force

    is_cuda = (device.lower() == "cuda")
    if is_cuda and seq_len >= threshold:
        return "performer"
    return "standard"


def apply_adaptive_attention(
    model,
    *,
    device: str,
    seq_len: int,
    replace_fn: Callable[..., None],
    cfg: AttnAutoConfig = AttnAutoConfig(),
    force: Optional[AttnKind] = None,
) -> AttnKind:
    """Apply adaptive attention mechanism to model.
    
    This function:
    1. Selects appropriate attention type (Standard/Performer)
    2. Applies attention replacement if needed
    3. Records the applied type in model._attn_kind
    4. Ensures idempotent operation (won't re-apply if already set)
    
    Args:
        model: Model to modify (e.g., GPT2LMHeadModel)
        device: Device string ("cuda" or "cpu")
        seq_len: Total sequence length
        replace_fn: Function to replace attention layers
                    (e.g., attention_performer.replace_attention_layers)
                    Signature: replace_fn(model, num_random_features=128)
        cfg: Configuration for adaptive selection
        force: Force specific attention type (overrides auto selection)
        
    Returns:
        Applied attention kind ("performer" or "standard")
        
    Side Effects:
        - Modifies model attention layers (if Performer selected)
        - Sets model._attn_kind attribute
        
    Examples:
        >>> from ml.attention_performer import replace_attention_layers
        >>> cfg = AttnAutoConfig(threshold=1024, num_random_features=128)
        >>> kind = apply_adaptive_attention(
        ...     model,
        ...     device="cuda",
        ...     seq_len=2048,
        ...     replace_fn=replace_attention_layers,
        ...     cfg=cfg
        ... )
        >>> print(kind)
        'performer'
    """
    # Idempotent check: skip if already applied
    current = getattr(model, "_attn_kind", None)
    if cfg.idempotent and current in ("performer", "standard"):
        return current

    # Select attention type
    kind = select_attention(
        device,
        seq_len,
        threshold=cfg.threshold,
        force=force
    )

    # Apply Performer if selected
    if kind == "performer":
        replace_fn(model, num_random_features=cfg.num_random_features)

    # Record applied type
    try:
        setattr(model, "_attn_kind", kind)
    except Exception:
        # Ignore if model doesn't support attribute assignment
        pass

    return kind
