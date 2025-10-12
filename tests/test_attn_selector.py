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
    """GPU should default to SDPA (not Performer)."""
    # GPU always returns SDPA by default (regardless of threshold)
    assert select_attention("cuda", 256, threshold=1024) == "sdpa"
    assert select_attention("cuda", 1023, threshold=1024) == "sdpa"
    assert select_attention("cuda", 1024, threshold=1024) == "sdpa"
    assert select_attention("cuda", 2048, threshold=1024) == "sdpa"


def test_force_override():
    """Force parameter should override automatic selection."""
    # Force Performer on CPU
    assert select_attention("cpu", 0, force="performer") == "performer"
    
    # Force Standard on GPU with long sequence
    assert select_attention("cuda", 9999, force="standard") == "standard"
    
    # Force SDPA explicitly
    assert select_attention("cpu", 128, force="sdpa") == "sdpa"


def test_apply_adaptive_calls_replace_once():
    """Verify replace_sdpa_fn is called with correct parameters."""
    sdpa_calls = []
    
    def fake_replace_sdpa(model, causal=True):
        sdpa_calls.append(causal)
        model.replaced = True
        return 12  # Simulated number of replaced layers

    model = types.SimpleNamespace()
    cfg = AttnAutoConfig(
        threshold=512,
        num_random_features=256,
        idempotent=True
    )
    
    kind = apply_adaptive_attention(
        model,
        device="cuda",
        seq_len=2048,
        replace_sdpa_fn=fake_replace_sdpa,
        cfg=cfg
    )
    
    assert kind == "sdpa"
    assert sdpa_calls == [True]  # causal=True
    assert getattr(model, "_attn_kind") == "sdpa"


def test_apply_adaptive_idempotent():
    """Second call should not re-apply if idempotent=True."""
    sdpa_calls = []
    
    def fake_replace_sdpa(model, causal=True):
        sdpa_calls.append(causal)
        return 12

    model = types.SimpleNamespace()
    cfg = AttnAutoConfig(
        threshold=512,
        num_random_features=256,
        idempotent=True
    )
    
    # First call
    kind1 = apply_adaptive_attention(
        model,
        device="cuda",
        seq_len=2048,
        replace_sdpa_fn=fake_replace_sdpa,
        cfg=cfg
    )
    assert kind1 == "sdpa"
    assert sdpa_calls == [True]
    
    # Second call (should be skipped)
    kind2 = apply_adaptive_attention(
        model,
        device="cuda",
        seq_len=2048,
        replace_sdpa_fn=fake_replace_sdpa,
        cfg=cfg
    )
    assert kind2 == "sdpa"
    assert sdpa_calls == [True]  # No additional calls


def test_apply_adaptive_standard_path_no_replace():
    """Standard path (CPU) should not call replace functions."""
    sdpa_calls = []
    performer_calls = []
    
    def fake_replace_sdpa(model, causal=True):
        sdpa_calls.append(causal)
        return 12
    
    def fake_replace_performer(model, num_random_features=128):
        performer_calls.append(num_random_features)
    
    model = types.SimpleNamespace()
    cfg = AttnAutoConfig(threshold=512, num_random_features=128)
    
    kind = apply_adaptive_attention(
        model,
        device="cpu",  # CPU -> standard
        seq_len=2048,
        replace_sdpa_fn=fake_replace_sdpa,
        replace_performer_fn=fake_replace_performer,
        cfg=cfg
    )
    
    assert kind == "standard"
    assert sdpa_calls == []  # No SDPA replacement on CPU
    assert performer_calls == []  # No Performer replacement
    assert getattr(model, "_attn_kind") == "standard"


def test_apply_adaptive_force_performer():
    """Force parameter should work with apply_adaptive_attention."""
    performer_calls = []
    
    def fake_replace_performer(model, num_random_features=128):
        performer_calls.append(num_random_features)
    
    model = types.SimpleNamespace()
    cfg = AttnAutoConfig(threshold=512, num_random_features=128)
    
    # Force Performer on CPU with short sequence
    kind = apply_adaptive_attention(
        model,
        device="cpu",
        seq_len=64,
        replace_performer_fn=fake_replace_performer,
        cfg=cfg,
        force="performer"
    )
    
    assert kind == "performer"
    assert performer_calls == [128]


def test_apply_adaptive_force_standard():
    """Force standard should skip replacement even on GPU+long seq."""
    sdpa_calls = []
    performer_calls = []
    
    def fake_replace_sdpa(model, causal=True):
        sdpa_calls.append(causal)
        return 12
    
    def fake_replace_performer(model, num_random_features=128):
        performer_calls.append(num_random_features)
    
    model = types.SimpleNamespace()
    cfg = AttnAutoConfig(threshold=512, num_random_features=128)
    
    kind = apply_adaptive_attention(
        model,
        device="cuda",
        seq_len=4096,  # Long sequence
        replace_sdpa_fn=fake_replace_sdpa,
        replace_performer_fn=fake_replace_performer,
        cfg=cfg,
        force="standard"  # Force Standard
    )
    
    assert kind == "standard"
    assert sdpa_calls == []  # No replacement
    assert performer_calls == []  # No replacement


def test_config_defaults():
    """Verify default configuration values."""
    cfg = AttnAutoConfig()
    
    assert cfg.threshold == 512  # Updated default for SDPA
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
