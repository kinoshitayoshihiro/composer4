#!/usr/bin/env python3
"""
Tests for Performer Linear Attention (Day 9-10)
"""

import tempfile
from pathlib import Path

import pytest
import torch

from ml.attention_performer import (
    PerformerAttention,
    _causal_linear_attention,
    _create_random_features,
    _kernel_feature_creator,
    replace_attention_layers,
)
from ml.performance_monitor import InferenceMetrics, PerformanceMonitor


class TestPerformerAttention:
    """Test Performer linear attention implementation."""
    
    def test_create_random_features(self):
        """Test random feature matrix creation."""
        num_features = 256
        dim = 64
        
        features = _create_random_features(num_features, dim)
        
        assert features.shape == (num_features, dim)
        assert features.dtype == torch.float32
    
    def test_kernel_feature_creator(self):
        """Test kernel feature transformation."""
        B, H, L, D = 2, 8, 10, 64
        M = 256
        
        data = torch.randn(B, H, L, D)
        projection = _create_random_features(M, D)
        
        features = _kernel_feature_creator(data, projection, is_query=True)
        
        assert features.shape == (B, H, L, M)
        assert torch.all(features >= 0)  # Should be positive (exp output)
    
    def test_causal_linear_attention(self):
        """Test causal linear attention computation."""
        B, H, L, D = 2, 8, 10, 64
        M = 256
        
        q = torch.randn(B, H, L, D)
        k = torch.randn(B, H, L, D)
        v = torch.randn(B, H, L, D)
        projection = _create_random_features(M, D)
        
        output = _causal_linear_attention(q, k, v, projection)
        
        assert output.shape == (B, H, L, D)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    def test_performer_attention_initialization(self):
        """Test PerformerAttention initialization."""
        n_embd = 768
        n_head = 12
        
        attn = PerformerAttention(
            n_embd=n_embd,
            n_head=n_head,
            num_random_features=256,
            causal=True,
        )
        
        assert attn.n_embd == n_embd
        assert attn.n_head == n_head
        assert attn.head_dim == n_embd // n_head
        assert attn.projection_matrix.shape == (256, attn.head_dim)
    
    def test_performer_attention_forward(self):
        """Test PerformerAttention forward pass."""
        B, L, n_embd = 2, 10, 768
        n_head = 12
        
        attn = PerformerAttention(
            n_embd=n_embd,
            n_head=n_head,
            num_random_features=256,
            causal=True,
        )
        
        hidden_states = torch.randn(B, L, n_embd)
        
        output, cache = attn(hidden_states)
        
        assert output.shape == (B, L, n_embd)
        assert cache is None  # Linear attention doesn't use cache
        assert not torch.isnan(output).any()
    
    def test_performer_attention_causality(self):
        """Test that Performer attention is causal."""
        B, L, n_embd = 1, 10, 768
        n_head = 12
        
        attn = PerformerAttention(
            n_embd=n_embd,
            n_head=n_head,
            num_random_features=256,
            causal=True,
        )
        attn.eval()
        
        # Create input with distinct pattern
        hidden_states = torch.randn(B, L, n_embd)
        
        # Full forward pass
        with torch.no_grad():
            full_output, _ = attn(hidden_states)
        
        # Truncated forward pass (should give same results for prefix)
        with torch.no_grad():
            truncated_output, _ = attn(hidden_states[:, :5, :])
        
        # First 5 positions should be similar (within numerical tolerance)
        # Note: Not exact due to numerical differences in cumsum
        similarity = torch.nn.functional.cosine_similarity(
            full_output[:, :5, :].flatten(),
            truncated_output[:, :5, :].flatten(),
            dim=0,
        )
        
        assert similarity > 0.95  # High similarity indicates causality
    
    def test_replace_attention_layers(self):
        """Test replacing attention layers in GPT-2 model."""
        from transformers import GPT2Config, GPT2LMHeadModel
        
        config = GPT2Config(
            vocab_size=1000,
            n_layer=2,
            n_head=4,
            n_embd=128,
            n_positions=256,
        )
        model = GPT2LMHeadModel(config)
        
        # Replace attention layers
        modified_model = replace_attention_layers(model, num_random_features=64, verbose=True)
        
        # Check that attention layers were replaced
        for layer in modified_model.transformer.h:
            assert isinstance(layer.attn, PerformerAttention)
        
        # Test forward pass with random input (weights not copied, just shape check)
        input_ids = torch.randint(0, 1000, (2, 10))
        
        with torch.no_grad():
            output = modified_model(input_ids)
        
        assert output.logits.shape == (2, 10, 1000)


class TestPerformanceMonitor:
    """Test performance monitoring tools."""
    
    def test_inference_metrics_creation(self):
        """Test InferenceMetrics dataclass."""
        metrics = InferenceMetrics(
            total_latency_ms=100.0,
            per_token_latency_ms=1.0,
            prompt_length=10,
            generated_length=100,
            total_length=110,
            peak_memory_mb=500.0,
            avg_memory_mb=400.0,
            timestamp=1234567890.0,
            model_type="performer",
        )
        
        assert metrics.total_latency_ms == 100.0
        assert metrics.model_type == "performer"
    
    def test_performance_monitor_logging(self):
        """Test logging metrics."""
        monitor = PerformanceMonitor()
        
        # Log some metrics
        for i in range(10):
            monitor.log_metrics(
                total_latency_ms=100.0 + i * 10,
                prompt_length=10,
                generated_length=100 + i * 5,
                peak_memory_mb=500.0,
                avg_memory_mb=400.0,
                model_type="performer",
            )
        
        assert len(monitor.metrics) == 10
    
    def test_performance_report_generation(self):
        """Test generating performance report."""
        monitor = PerformanceMonitor()
        
        # Log metrics
        for i in range(20):
            monitor.log_metrics(
                total_latency_ms=100.0 + i,
                prompt_length=10,
                generated_length=100,
                peak_memory_mb=500.0 + i,
                avg_memory_mb=400.0,
                model_type="performer",
            )
        
        report = monitor.generate_report(model_type="performer")
        
        assert report.total_runs == 20
        assert report.model_type == "performer"
        assert report.latency_mean > 0
        assert report.latency_p95 > report.latency_mean
        assert report.max_sequence_length == 110
    
    def test_performance_monitor_comparison(self):
        """Test comparing two model types."""
        monitor = PerformanceMonitor()
        
        # Log baseline metrics
        for i in range(10):
            monitor.log_metrics(
                total_latency_ms=200.0,
                prompt_length=10,
                generated_length=100,
                peak_memory_mb=600.0,
                avg_memory_mb=500.0,
                model_type="standard",
            )
        
        # Log performer metrics (faster, less memory)
        for i in range(10):
            monitor.log_metrics(
                total_latency_ms=150.0,
                prompt_length=10,
                generated_length=100,
                peak_memory_mb=400.0,
                avg_memory_mb=350.0,
                model_type="performer",
            )
        
        comparison = monitor.compare_models(baseline_type="standard", comparison_type="performer")
        
        assert comparison["improvements"]["latency_speedup"] > 1.0  # Faster
        assert comparison["improvements"]["memory_reduction"] > 0.0  # Less memory
        assert comparison["summary"]["faster"] is True
        assert comparison["summary"]["less_memory"] is True
    
    def test_save_report(self):
        """Test saving report to JSON."""
        monitor = PerformanceMonitor()
        
        # Log some metrics
        for i in range(5):
            monitor.log_metrics(
                total_latency_ms=100.0,
                prompt_length=10,
                generated_length=100,
                peak_memory_mb=500.0,
                avg_memory_mb=400.0,
                model_type="performer",
            )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "report.json"
            monitor.save_report(output_path, model_type="performer")
            
            assert output_path.exists()
            
            import json
            report_data = json.loads(output_path.read_text())
            assert report_data["total_runs"] == 5
            assert report_data["model_type"] == "performer"
    
    def test_inference_tracker_context_manager(self):
        """Test inference tracker context manager."""
        monitor = PerformanceMonitor()
        
        with monitor.track_inference(model_type="performer"):
            # Simulate some work
            import time
            time.sleep(0.01)
        
        # Check that metrics were recorded
        assert "total_latency_ms" in monitor.current_run
        assert monitor.current_run["total_latency_ms"] >= 10.0  # At least 10ms
        assert monitor.current_run["model_type"] == "performer"
