"""Test suite for Stage3 tokenizer, dataset, and inference."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

# Assuming ml.stage3_generator is importable
try:
    from ml.stage3_generator import Stage3Tokenizer, quantize, tokenize_caption
except ImportError:
    pytest.skip("ml.stage3_generator not available", allow_module_level=True)


class TestStage3Tokenizer:
    """Test Stage3Tokenizer vocabulary and encoding."""

    def test_init_creates_base_vocab(self) -> None:
        """Verify base vocabulary initialization."""
        tokenizer = Stage3Tokenizer()
        assert tokenizer.vocab_size > 0
        assert "<pad>" in tokenizer.token_to_id
        assert "<bos>" in tokenizer.token_to_id
        assert "<eos>" in tokenizer.token_to_id
        assert "<cond_end>" in tokenizer.token_to_id
        assert "<BAR>" in tokenizer.token_to_id
        assert "<BEAT>" in tokenizer.token_to_id

    def test_note_tokens_present(self) -> None:
        """Verify all MIDI notes are tokenized."""
        tokenizer = Stage3Tokenizer()
        for note in range(128):
            assert f"NOTE_{note}" in tokenizer.token_to_id

    def test_tempo_tokens_present(self) -> None:
        """Verify tempo tokens cover expected range."""
        tokenizer = Stage3Tokenizer()
        for bpm in range(40, 250, 10):
            assert f"TEMPO_{bpm}" in tokenizer.token_to_id

    def test_time_signature_tokens_present(self) -> None:
        """Verify common time signatures are tokenized."""
        tokenizer = Stage3Tokenizer()
        assert "TSIG_4/4" in tokenizer.token_to_id
        assert "TSIG_3/4" in tokenizer.token_to_id
        assert "TSIG_6/8" in tokenizer.token_to_id

    def test_bar_position_tokens_present(self) -> None:
        """Verify bar position tokens are created."""
        tokenizer = Stage3Tokenizer(max_bars=8)
        for bar_num in range(8):
            assert f"BAR_{bar_num}" in tokenizer.token_to_id

    def test_velocity_bins(self) -> None:
        """Verify velocity quantization into bins."""
        tokenizer = Stage3Tokenizer(velocity_bins=16)
        # Test low velocity
        tok_low = tokenizer.velocity_token(10)
        # Test mid velocity
        tok_mid = tokenizer.velocity_token(64)
        # Test high velocity
        tok_high = tokenizer.velocity_token(120)
        assert tok_low != tok_mid
        assert tok_mid != tok_high

    def test_save_and_load_roundtrip(self, tmp_path: Path) -> None:
        """Verify tokenizer can be saved and loaded."""
        tokenizer = Stage3Tokenizer(beat_division=24, max_bars=12)
        save_path = tmp_path / "tokenizer.json"
        tokenizer.save(save_path)

        assert save_path.exists()
        data = json.loads(save_path.read_text())
        assert data["beat_division"] == 24
        assert data["max_bars"] == 12
        assert len(data["token_to_id"]) == tokenizer.vocab_size


class TestUtilityFunctions:
    """Test helper functions."""

    def test_quantize_clamps_to_zero_one(self) -> None:
        """Verify quantization clamps values to [0, 1]."""
        assert quantize(-0.5, buckets=10) == 0
        assert quantize(1.5, buckets=10) == 10

    def test_quantize_buckets(self) -> None:
        """Verify quantization creates expected buckets."""
        assert quantize(0.0, buckets=10) == 0
        assert quantize(0.5, buckets=10) == 5
        assert quantize(1.0, buckets=10) == 10

    def test_quantize_nan_returns_negative_one(self) -> None:
        """Verify NaN handling."""
        assert quantize(float("nan"), buckets=10) == -1

    def test_tokenize_caption_japanese(self) -> None:
        """Verify Japanese caption tokenization."""
        tokens = tokenize_caption("パワフルなロックビート", limit=8)
        assert len(tokens) > 0
        assert "パワフルなロックビート" in tokens

    def test_tokenize_caption_mixed(self) -> None:
        """Verify mixed Japanese/English tokenization."""
        tokens = tokenize_caption("激しいintenseビート", limit=10)
        assert len(tokens) >= 2

    def test_tokenize_caption_deduplicates(self) -> None:
        """Verify duplicate tokens are removed."""
        tokens = tokenize_caption("テスト、テスト、テスト", limit=10)
        assert len(tokens) == 2  # "テスト" and "、"

    def test_tokenize_caption_respects_limit(self) -> None:
        """Verify token limit is enforced."""
        tokens = tokenize_caption("あいうえおかきくけこさしすせそ", limit=5)
        assert len(tokens) <= 5


class TestStage3Inference:
    """Smoke tests for inference (requires model files - skip if unavailable)."""

    @pytest.mark.skip(reason="Requires trained model files")
    def test_inference_smoke(self, tmp_path: Path) -> None:
        """Smoke test for end-to-end inference."""
        # This would require a trained model checkpoint
        # Placeholder for integration testing
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
