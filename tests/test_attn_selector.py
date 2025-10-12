"""Tests for adaptive attention selector."""

import types
import pytest
from ml.attn_selector import (
    select_attention,
    apply_adaptive_attention,
    AttnAutoConfig,
)


def test_select_cpu_always_standard():
    """CPU should always use Standard attention."""
    assert select_attention("cpu", 128) == "standard"
    assert select_attention("cpu", 8192) == "standard"
    assert select_attention("CPU", 2048) == "standard"  # case-insensitive


def test_select_cuda_threshold():
    """GPU should respect threshold for Performer selection."""
    # Below threshold -> Standard
    assert select_attention("cuda", 256, threshold=1024) == "standard"
    assert select_attention("cuda", 1023, threshold=1024) == "standard"
    
    # At or above threshold -> Performer
    assert select_attention("cuda", 1024, threshold=1024) == "performer"
    assert select_attention("cuda", 2048, threshold=1024) == "performer"


def test_force_override():
    """Force parameter should override automatic selection."""
    # Force Performer on CPU
    assert select_attention("cpu", 0, force="performer") == "performer"
    
    # Force Standard on GPU with long sequence
    assert select_attention("cuda", 9999, force="standard") == "standard"


def test_apply_adaptive_calls_replace_once():
    """Verify replace_fn is called with correct parameters."""
    calls = []
    
    def fake_replace(model, num_random_features=128):
        calls.append(num_random_features)
        model.replaced = True

    model = types.SimpleNamespace()
    cfg = AttnAutoConfig(
        threshold=1024,
        num_random_features=256,
        idempotent=True
    )
    
    kind = apply_adaptive_attention(
        model,
        device="cuda",
        seq_len=2048,
        replace_fn=fake_replace,
        cfg=cfg
    )
    
    assert kind == "performer"
    assert calls == [256]
    assert getattr(model, "_attn_kind") == "performer"


def test_apply_adaptive_idempotent():
    """Second call should not re-apply if idempotent=True."""
    calls = []
    
    def fake_replace(model, num_random_features=128):
        calls.append(num_random_features)

    model = types.SimpleNamespace()
    cfg = AttnAutoConfig(
        threshold=1024,
        num_random_features=256,
        idempotent=True
    )
    
    # First call
    kind1 = apply_adaptive_attention(
        model,
        device="cuda",
        seq_len=2048,
        replace_fn=fake_replace,
        cfg=cfg
    )
    assert kind1 == "performer"
    assert calls == [256]
    
    # Second call (should be skipped)
    kind2 = apply_adaptive_attention(
        model,
        device="cuda",
        seq_len=2048,
        replace_fn=fake_replace,
        cfg=cfg
    )
    assert kind2 == "performer"
    assert calls == [256]  # No additional calls


def test_apply_adaptive_standard_path_no_replace():
    """Standard path should not call replace_fn."""
    calls = []
    
    def fake_replace(model, num_random_features=128):
        calls.append(num_random_features)
    
    model = types.SimpleNamespace()
    cfg = AttnAutoConfig(threshold=1024, num_random_features=128)
    
    kind = apply_adaptive_attention(
        model,
        device="cuda",
        seq_len=128,  # Below threshold
        replace_fn=fake_replace,
        cfg=cfg
    )
    
    assert kind == "standard"
    assert calls == []  # replace_fn should not be called
    assert getattr(model, "_attn_kind") == "standard"


def test_apply_adaptive_force_performer():
    """Force parameter should work with apply_adaptive_attention."""
    calls = []
    
    def fake_replace(model, num_random_features=128):
        calls.append(num_random_features)
    
    model = types.SimpleNamespace()
    cfg = AttnAutoConfig(threshold=1024, num_random_features=128)
    
    # Force Performer on CPU with short sequence
    kind = apply_adaptive_attention(
        model,
        device="cpu",
        seq_len=64,
        replace_fn=fake_replace,
        cfg=cfg,
        force="performer"
    )
    
    assert kind == "performer"
    assert calls == [128]


def test_apply_adaptive_force_standard():
    """Force standard should skip replacement even on GPU+long seq."""
    calls = []
    
    def fake_replace(model, num_random_features=128):
        calls.append(num_random_features)
    
    model = types.SimpleNamespace()
    cfg = AttnAutoConfig(threshold=1024, num_random_features=128)
    
    kind = apply_adaptive_attention(
        model,
        device="cuda",
        seq_len=4096,  # Long sequence
        replace_fn=fake_replace,
        cfg=cfg,
        force="standard"  # Force Standard
    )
    
    assert kind == "standard"
    assert calls == []  # No replacement


def test_config_defaults():
    """Verify default configuration values."""
    cfg = AttnAutoConfig()
    
    assert cfg.threshold == 1024
    assert cfg.num_random_features == 128
    assert cfg.idempotent is True


def test_config_custom():
    """Verify custom configuration values."""
    cfg = AttnAutoConfig(
        threshold=2048,
        num_random_features=64,
        idempotent=False
    )
    
    assert cfg.threshold == 2048
    assert cfg.num_random_features == 64
    assert cfg.idempotent is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
