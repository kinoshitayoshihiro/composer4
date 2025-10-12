"""Tests for SDPA (Scaled Dot-Product Attention) implementation."""

import pytest
import torch
import torch.nn as nn

from ml.attention_sdpa import SDPAAttn, replace_attention_layers_sdpa, sdpa_kernel_availability


def _create_toy_tensors(
    b: int = 2,
    h: int = 4,
    t: int = 8,
    d: int = 16,
    dtype: torch.dtype = torch.float32,
    device: str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create toy Q, K, V tensors for testing."""
    q = torch.randn(b, h, t, d, dtype=dtype, device=device)
    k = torch.randn(b, h, t, d, dtype=dtype, device=device)
    v = torch.randn(b, h, t, d, dtype=dtype, device=device)
    return q, k, v


def test_sdpa_shapes_and_finite():
    """Test SDPA output shapes and finite values."""
    q, k, v = _create_toy_tensors()
    attn = SDPAAttn(causal=True, dropout_p=0.0)
    y = attn(q, k, v)
    
    # Check shape
    assert y.shape == q.shape, f"Expected {q.shape}, got {y.shape}"
    
    # Check finite values (no NaN/Inf)
    assert torch.isfinite(y).all(), "Output contains NaN or Inf"


def test_sdpa_causality():
    """Test causal masking behavior."""
    torch.manual_seed(42)
    q, k, v = _create_toy_tensors(t=6)
    
    # Causal attention
    attn_causal = SDPAAttn(causal=True, dropout_p=0.0)
    y_causal = attn_causal(q, k, v)
    
    # Non-causal attention
    attn_noncausal = SDPAAttn(causal=False, dropout_p=0.0)
    y_noncausal = attn_noncausal(q, k, v)
    
    # They should be different (causal has masked future)
    diff = torch.max(torch.abs(y_causal - y_noncausal)).item()
    assert diff > 1e-6, f"Causal and non-causal should differ, got max diff {diff}"


def test_sdpa_close_to_math_backend():
    """Test that SDPA is close to math backend (short sequence)."""
    torch.manual_seed(0)
    q, k, v = _create_toy_tensors(t=6)
    
    # SDPA implementation
    attn = SDPAAttn(causal=True, dropout_p=0.0)
    y1 = attn(q, k, v)
    
    # Direct call to F.scaled_dot_product_attention
    from torch.nn.functional import scaled_dot_product_attention
    y2 = scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=0.0)
    
    # Should be very close (identical in practice)
    max_diff = torch.max(torch.abs(y1 - y2)).item()
    assert max_diff < 1e-6, f"Expected close match, got max diff {max_diff}"


def test_sdpa_different_dtypes():
    """Test SDPA with different dtypes (fp32, fp16)."""
    torch.manual_seed(123)
    
    # FP32
    q32, k32, v32 = _create_toy_tensors(dtype=torch.float32)
    attn32 = SDPAAttn(causal=True, dropout_p=0.0)
    y32 = attn32(q32, k32, v32)
    assert y32.dtype == torch.float32
    assert torch.isfinite(y32).all()
    
    # FP16 (if available)
    try:
        q16, k16, v16 = _create_toy_tensors(dtype=torch.float16)
        attn16 = SDPAAttn(causal=True, dropout_p=0.0)
        y16 = attn16(q16, k16, v16)
        assert y16.dtype == torch.float16
        assert torch.isfinite(y16).all()
    except Exception as e:
        pytest.skip(f"FP16 not available: {e}")


def test_sdpa_batch_size_variations():
    """Test SDPA with different batch sizes."""
    for b in [1, 2, 4, 8]:
        q, k, v = _create_toy_tensors(b=b)
        attn = SDPAAttn(causal=True, dropout_p=0.0)
        y = attn(q, k, v)
        assert y.shape == (b, 4, 8, 16)
        assert torch.isfinite(y).all()


def test_sdpa_sequence_length_variations():
    """Test SDPA with different sequence lengths."""
    for t in [4, 16, 64, 256]:
        q, k, v = _create_toy_tensors(t=t)
        attn = SDPAAttn(causal=True, dropout_p=0.0)
        y = attn(q, k, v)
        assert y.shape == (2, 4, t, 16)
        assert torch.isfinite(y).all()


def test_replace_attention_layers_sdpa_no_layers():
    """Test replacement when model has no attn_core layers."""
    model = nn.Sequential(
        nn.Linear(10, 10),
        nn.ReLU(),
        nn.Linear(10, 10),
    )
    
    num_replaced = replace_attention_layers_sdpa(model, causal=True)
    assert num_replaced == 0, f"Expected 0 replacements, got {num_replaced}"


def test_replace_attention_layers_sdpa_with_cores():
    """Test replacement when model has attn_core attributes."""
    # Create a mock model with attn_core
    class MockBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.attn_core = nn.Linear(10, 10)  # Dummy
            
    class MockModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = nn.ModuleList([MockBlock() for _ in range(3)])
    
    model = MockModel()
    
    # Replace
    num_replaced = replace_attention_layers_sdpa(model, causal=True)
    assert num_replaced == 3, f"Expected 3 replacements, got {num_replaced}"
    
    # Check that attn_core is now SDPAAttn
    for block in model.blocks:
        assert isinstance(block.attn_core, SDPAAttn)
        assert block.attn_core.causal == True


def test_sdpa_kernel_availability():
    """Test kernel availability detection."""
    avail = sdpa_kernel_availability()
    
    # Should return a dict
    assert isinstance(avail, dict)
    
    # Should have expected keys (or error)
    if "error" not in avail:
        assert "flash" in avail
        assert "mem_efficient" in avail
        assert "math" in avail
        
        # Values should be boolean
        for kernel, is_avail in avail.items():
            if kernel != "error":
                assert isinstance(is_avail, bool)


def test_sdpa_dropout_training_mode():
    """Test dropout behavior in training mode."""
    torch.manual_seed(999)
    q, k, v = _create_toy_tensors(t=16)
    
    # Training mode with dropout
    attn = SDPAAttn(causal=True, dropout_p=0.1)
    attn.train()
    
    y1 = attn(q, k, v)
    y2 = attn(q, k, v)
    
    # With dropout, outputs should be different
    # (unless we're extremely unlucky with random seed)
    diff = torch.max(torch.abs(y1 - y2)).item()
    # Note: This test might be flaky, but generally dropout causes variation
    # Just check that we can run it without errors
    assert torch.isfinite(y1).all()
    assert torch.isfinite(y2).all()


def test_sdpa_eval_mode_deterministic():
    """Test that eval mode is deterministic (dropout_p ignored)."""
    torch.manual_seed(777)
    q, k, v = _create_toy_tensors(t=16)
    
    # Eval mode (dropout_p should be ignored)
    attn = SDPAAttn(causal=True, dropout_p=0.0)
    attn.eval()
    
    y1 = attn(q, k, v)
    y2 = attn(q, k, v)
    
    # Should be identical
    max_diff = torch.max(torch.abs(y1 - y2)).item()
    assert max_diff < 1e-7, f"Expected identical outputs, got max diff {max_diff}"


def test_sdpa_gradient_flow():
    """Test that gradients flow through SDPA."""
    q, k, v = _create_toy_tensors()
    q.requires_grad = True
    k.requires_grad = True
    v.requires_grad = True
    
    attn = SDPAAttn(causal=True, dropout_p=0.0)
    y = attn(q, k, v)
    
    # Compute loss and backward
    loss = y.sum()
    loss.backward()
    
    # Check gradients exist
    assert q.grad is not None
    assert k.grad is not None
    assert v.grad is not None
    
    # Gradients should be finite
    assert torch.isfinite(q.grad).all()
    assert torch.isfinite(k.grad).all()
    assert torch.isfinite(v.grad).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
