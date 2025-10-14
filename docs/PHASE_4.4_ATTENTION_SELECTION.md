# Phase 4.4: Attention Mechanism Auto-Selection

## Overview

シーケンス長に応じて最適なAttentionメカニズムを自動選択する機能を実装します。

## Attention Mechanisms

### 1. Standard Attention
- **Use case**: N < 1024
- **Memory**: O(N²)
- **Speed**: 最速（小規模）
- **Compatibility**: すべてのデバイス・精度で動作

### 2. SDPA (Scaled Dot-Product Attention)
- **Use case**: N ≥ 1024
- **Memory**: O(N²) → O(N) (memory-efficient variant)
- **Speed**: 高速（PyTorch最適化）
- **Requirements**: PyTorch 2.0+, CUDA
- **Compatibility**: FP16/BF16/FP32

### 3. Flash Attention
- **Use case**: N ≥ 1024 and BF16
- **Memory**: O(N)
- **Speed**: 最速（大規模）
- **Requirements**: `flash-attn` package, BF16
- **Compatibility**: Modern GPUs (Ampere+)

## Auto-Selection Policy

```python
def select_attention_mechanism(seq_len: int, precision: str, device: str) -> str:
    """
    Auto-select attention mechanism based on sequence length and environment.
    
    Args:
        seq_len: Sequence length
        precision: 'fp32', 'fp16', or 'bf16'
        device: 'cuda', 'cpu', or 'mps'
    
    Returns:
        'standard', 'sdpa', or 'flash'
    """
    if seq_len < 1024:
        return 'standard'
    
    if device == 'cpu':
        return 'standard'
    
    # Try Flash Attention (best performance for long sequences)
    if precision == 'bf16' and device == 'cuda':
        try:
            import flash_attn
            return 'flash'
        except ImportError:
            pass
    
    # Fallback to SDPA (PyTorch native)
    if device == 'cuda':
        return 'sdpa'
    
    return 'standard'
```

## Implementation

### 1. Config Schema

Add to `configs/piano_transformer.yaml`:

```yaml
model:
  attention:
    # Auto-selection policy
    auto_select: true
    threshold: 1024  # Switch to efficient attention at N >= 1024
    
    # Manual override (optional)
    force_mechanism: null  # 'standard', 'sdpa', 'flash', or null for auto
    
    # Flash Attention specific
    flash_causal: true
    flash_dropout: 0.0
```

### 2. Model Code

Add to `piano_train.py` or create `attention_selector.py`:

```python
import torch
from typing import Optional, Literal

AttentionType = Literal['standard', 'sdpa', 'flash']

class AttentionSelector:
    """Select optimal attention mechanism based on runtime conditions."""
    
    def __init__(
        self,
        threshold: int = 1024,
        force_mechanism: Optional[AttentionType] = None
    ):
        self.threshold = threshold
        self.force_mechanism = force_mechanism
        self._flash_available = self._check_flash_attn()
    
    def _check_flash_attn(self) -> bool:
        try:
            import flash_attn
            return True
        except ImportError:
            return False
    
    def select(
        self,
        seq_len: int,
        precision: torch.dtype,
        device: torch.device
    ) -> AttentionType:
        """Select attention mechanism."""
        
        # Manual override
        if self.force_mechanism is not None:
            return self.force_mechanism
        
        # Short sequences: always use standard
        if seq_len < self.threshold:
            return 'standard'
        
        # CPU: no efficient attention available
        if device.type == 'cpu':
            return 'standard'
        
        # Long sequences on GPU
        if device.type == 'cuda':
            # Flash Attention (best for BF16)
            if precision == torch.bfloat16 and self._flash_available:
                return 'flash'
            
            # SDPA (good for FP16/FP32)
            return 'sdpa'
        
        return 'standard'
    
    def get_attention_impl(self, mechanism: AttentionType):
        """Get attention implementation."""
        if mechanism == 'flash':
            from flash_attn import flash_attn_func
            return flash_attn_func
        elif mechanism == 'sdpa':
            return torch.nn.functional.scaled_dot_product_attention
        else:
            return None  # Use model's default implementation
```

### 3. Integration with Training

Modify `piano_train.py`:

```python
from attention_selector import AttentionSelector

# Initialize selector
attention_selector = AttentionSelector(
    threshold=config.get('attention', {}).get('threshold', 1024),
    force_mechanism=config.get('attention', {}).get('force_mechanism')
)

# Before training loop
def get_model_with_attention(model, seq_len, precision, device):
    mechanism = attention_selector.select(seq_len, precision, device)
    print(f"[attention] Using {mechanism} attention for seq_len={seq_len}")
    
    # Configure model
    if mechanism == 'sdpa':
        # Enable SDPA in Transformers (automatic in PyTorch 2.0+)
        model.config.attn_implementation = "sdpa"
    elif mechanism == 'flash':
        model.config.attn_implementation = "flash_attention_2"
    
    return model
```

## Benchmarks

Expected performance improvements:

| Seq Length | Standard | SDPA | Flash (BF16) |
|------------|----------|------|--------------|
| 512 | 1.0x (baseline) | 1.0x | - |
| 1024 | 1.0x | 1.3x | 1.5x |
| 2048 | 1.0x | 1.8x | 2.2x |
| 4096 | OOM | 2.5x | 3.0x |

Memory usage (relative to standard):

- SDPA: ~0.8x
- Flash: ~0.5x

## Testing

```bash
# Test with different sequence lengths
python scripts/piano_train.py \
  --config configs/piano_transformer.yaml \
  --max-seq-len 512  # Uses standard

python scripts/piano_train.py \
  --config configs/piano_transformer.yaml \
  --max-seq-len 2048  # Uses SDPA/Flash

# Force specific mechanism
python scripts/piano_train.py \
  --config configs/piano_transformer.yaml \
  --force-attention flash
```

## Compatibility Matrix

| Attention | CPU | CUDA (FP32) | CUDA (FP16) | CUDA (BF16) |
|-----------|-----|-------------|-------------|-------------|
| Standard | ✅ | ✅ | ✅ | ✅ |
| SDPA | ✅ | ✅ | ✅ | ✅ |
| Flash | ❌ | ❌ | ⚠️ | ✅ |

Legend:
- ✅ Fully supported
- ⚠️ Supported but not optimal
- ❌ Not supported

## Migration Path

### Phase 1 (Current)
- Implement `AttentionSelector` class
- Add config schema
- Manual testing

### Phase 2 (Next Sprint)
- Integrate with `piano_train.py`
- Add automatic benchmarking
- CI integration

### Phase 3 (Future)
- Profile-guided optimization
- Dynamic switching during training
- Custom kernels for edge cases

## Related Issues

- Phase 4.1: Training robustness (AdamW, cosine schedule)
- Phase 4.2: Data quality (stratified splits)
- Phase 4.3: External benchmarks

## References

- [Flash Attention Paper](https://arxiv.org/abs/2205.14135)
- [PyTorch SDPA Docs](https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)
- [HuggingFace Efficient Attention](https://huggingface.co/docs/transformers/perf_infer_gpu_one#efficient-attention)
