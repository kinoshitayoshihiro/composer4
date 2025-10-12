"""
Test cases for caption_to_attrs.py

Run with: pytest tests/test_caption_to_attrs.py -v
"""

import json
import tempfile
from pathlib import Path

import pytest

# Import the module
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from caption_to_attrs import AttributeNormalizer, DEFAULT_VOCAB


class TestAttributeNormalizer:
    """Test AttributeNormalizer class."""

    def setup_method(self):
        """Initialize normalizer for each test."""
        self.normalizer = AttributeNormalizer()

    def test_genre_extraction(self):
        """Test genre detection."""
        caption = "A beautiful jazz piano piece"
        attrs = self.normalizer.normalize(caption)
        assert attrs["genre"] == "jazz"

    def test_mood_extraction(self):
        """Test mood detection."""
        caption = "A cheerful and upbeat melody"
        attrs = self.normalizer.normalize(caption)
        assert attrs["mood"] == "cheerful"

    def test_tempo_extraction(self):
        """Test tempo detection."""
        caption = "A very slow adagio movement"
        attrs = self.normalizer.normalize(caption)
        assert attrs["tempo"] == "very_slow"

    def test_multi_word_tempo(self):
        """Test multi-word tempo phrases."""
        # "very slow" should match before "slow"
        caption = "A very slow piece"
        attrs = self.normalizer.normalize(caption)
        assert attrs["tempo"] == "very_slow"

    def test_intensity_extraction(self):
        """Test intensity detection."""
        caption = "A powerful and loud performance"
        attrs = self.normalizer.normalize(caption)
        assert attrs["intensity"] == "high"

    def test_texture_extraction(self):
        """Test texture detection."""
        caption = "A dense orchestral arrangement"
        attrs = self.normalizer.normalize(caption)
        assert attrs["texture"] == "dense"

    def test_multiple_attributes(self):
        """Test extraction of multiple attributes."""
        caption = "A cheerful jazz piece with fast tempo and rich layered texture"
        attrs = self.normalizer.normalize(caption)
        assert attrs["genre"] == "jazz"
        assert attrs["mood"] == "cheerful"
        assert attrs["tempo"] == "fast"
        assert attrs["texture"] == "dense"  # "layered" -> dense

    def test_unknown_attributes(self):
        """Test handling of unknown attributes."""
        caption = "A piece of music"
        attrs = self.normalizer.normalize(caption)
        # Should return "unknown" for missing attributes
        assert attrs["genre"] == "unknown"
        assert attrs["mood"] == "unknown"

    def test_case_insensitive(self):
        """Test case-insensitive matching."""
        caption = "A JAZZ piece with CHEERFUL mood"
        attrs = self.normalizer.normalize(caption)
        assert attrs["genre"] == "jazz"
        assert attrs["mood"] == "cheerful"

    def test_word_boundaries(self):
        """Test word boundary matching."""
        # "pop" should not match "popular"
        caption = "A popular classical symphony"
        attrs = self.normalizer.normalize(caption)
        assert attrs["genre"] == "classical"  # Not "pop"

    def test_to_token_string(self):
        """Test token string generation."""
        attrs = {
            "genre": "jazz",
            "mood": "cheerful",
            "tempo": "fast",
            "intensity": "high",
            "texture": "sparse",
        }
        tokens = self.normalizer.to_token_string(attrs)
        assert tokens == "[jazz][cheerful][fast][high][sparse]"

    def test_to_token_string_with_unknown(self):
        """Test token string with unknown values."""
        attrs = {
            "genre": "unknown",
            "mood": "cheerful",
            "tempo": "unknown",
            "intensity": "medium",
            "texture": "unknown",
        }
        tokens = self.normalizer.to_token_string(attrs)
        assert tokens == "[unknown][cheerful][unknown][medium][unknown]"


class TestEndToEnd:
    """End-to-end integration tests."""

    def test_process_captions(self):
        """Test full caption processing pipeline."""
        # Create temporary input file
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False
        ) as infile:
            input_path = Path(infile.name)
            infile.write(
                json.dumps(
                    {
                        "loop_id": "test001",
                        "caption": "A cheerful jazz piano with fast tempo",
                    }
                )
                + "\n"
            )
            infile.write(
                json.dumps(
                    {
                        "loop_id": "test002",
                        "caption": "A melancholic classical piece with slow tempo and dense texture",
                    }
                )
                + "\n"
            )

        # Create temporary output file
        output_path = Path(tempfile.mktemp(suffix=".jsonl"))

        try:
            # Process
            from caption_to_attrs import process_captions

            normalizer = AttributeNormalizer()
            total, unknown = process_captions(input_path, output_path, normalizer)

            assert total == 2
            # Both entries have some unknown attributes (intensity/texture)
            assert unknown == 2

            # Verify output
            results = []
            with open(output_path, "r") as f:
                for line in f:
                    results.append(json.loads(line))

            assert len(results) == 2
            assert results[0]["loop_id"] == "test001"
            assert results[0]["attributes"]["genre"] == "jazz"
            assert results[0]["attributes"]["mood"] == "cheerful"
            assert results[0]["attributes"]["tempo"] == "fast"
            assert results[0]["tokens"] == "[jazz][cheerful][fast][unknown][unknown]"

            assert results[1]["loop_id"] == "test002"
            assert results[1]["attributes"]["genre"] == "classical"
            assert results[1]["attributes"]["mood"] == "melancholic"
            assert results[1]["attributes"]["tempo"] == "slow"
            assert results[1]["attributes"]["texture"] == "dense"

        finally:
            # Cleanup
            input_path.unlink(missing_ok=True)
            output_path.unlink(missing_ok=True)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
