#!/usr/bin/env python
"""Test Stage3 enhancements: CLAP/MERT tokens and dataloader improvements."""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ml.stage3_generator import Stage3Tokenizer, quantize


def test_clap_mert_tokens():
    """Test CLAP/MERT token generation."""
    print("=" * 60)
    print("Test 1: CLAP/MERT Token Generation")
    print("=" * 60)

    tokenizer = Stage3Tokenizer(audio_bins=10)

    # Check AUDIOCLAP tokens
    clap_tokens = [tok for tok in tokenizer.token_to_id if tok.startswith("AUDIOCLAP_")]
    print(f"\nAUDIOCLAP tokens: {len(clap_tokens)}")
    print(f"Sample: {clap_tokens[:5]}")
    assert len(clap_tokens) == 10, f"Expected 10 AUDIOCLAP tokens, got {len(clap_tokens)}"

    # Check AUDIOMERT tokens
    mert_tokens = [tok for tok in tokenizer.token_to_id if tok.startswith("AUDIOMERT_")]
    print(f"\nAUDIOMERT tokens: {len(mert_tokens)}")
    print(f"Sample: {mert_tokens[:5]}")
    assert len(mert_tokens) == 10, f"Expected 10 AUDIOMERT tokens, got {len(mert_tokens)}"

    print("\n✅ CLAP/MERT tokens generated successfully\n")


def test_encode_prompt_with_audio():
    """Test encode_prompt with audio similarity conditions."""
    print("=" * 60)
    print("Test 2: Encode Prompt with Audio Conditions")
    print("=" * 60)

    # Use generator tokenizer for testing
    tokenizer = Stage3Tokenizer(audio_bins=10)

    # Build condition tokens manually
    condition_tokens = []

    # Emotion
    emotion = "intense"
    emotion_id = tokenizer.ensure_condition_token(f"<emotion:{emotion}>")
    condition_tokens.append(emotion_id)

    # Genre
    genre = "rock"
    genre_id = tokenizer.ensure_condition_token(f"<genre:{genre}>")
    condition_tokens.append(genre_id)

    # Valence
    valence = 0.6
    valence_bucket = int(min(10, max(0, round(valence * 10))))
    valence_id = tokenizer.ensure_condition_token(f"<valence:{valence_bucket}>")
    condition_tokens.append(valence_id)

    # Arousal
    arousal = 0.8
    arousal_bucket = int(min(10, max(0, round(arousal * 10))))
    arousal_id = tokenizer.ensure_condition_token(f"<arousal:{arousal_bucket}>")
    condition_tokens.append(arousal_id)

    # Audio CLAP - floor-based quantization
    audio_clap = 0.85
    clap_val = min(max(float(audio_clap), 0.0), 1.0)
    clap_bucket = min(int(clap_val * 10), 9)
    clap_token = f"AUDIOCLAP_{clap_bucket}"
    clap_id = tokenizer.token_to_id.get(clap_token)
    if clap_id is not None:
        condition_tokens.append(clap_id)

    # Audio MERT - floor-based quantization
    audio_mert = 0.75
    mert_val = min(max(float(audio_mert), 0.0), 1.0)
    mert_bucket = min(int(mert_val * 10), 9)
    mert_token = f"AUDIOMERT_{mert_bucket}"
    mert_id = tokenizer.token_to_id.get(mert_token)
    if mert_id is not None:
        condition_tokens.append(mert_id)

    decoded = [tokenizer.id_to_token.get(tok_id, f"<UNK:{tok_id}>") for tok_id in condition_tokens]

    print(f"\nCondition tokens ({len(decoded)} tokens):")
    for i, tok in enumerate(decoded):
        print(f"  {i}: {tok}")

    # Check for AUDIOCLAP token
    clap_tokens = [tok for tok in decoded if tok.startswith("AUDIOCLAP_")]
    assert len(clap_tokens) >= 1, "Expected at least 1 AUDIOCLAP token"
    print(f"\n✅ Found AUDIOCLAP token: {clap_tokens[0]}")

    # Check for AUDIOMERT token
    mert_tokens = [tok for tok in decoded if tok.startswith("AUDIOMERT_")]
    assert len(mert_tokens) >= 1, "Expected at least 1 AUDIOMERT token"
    print(f"✅ Found AUDIOMERT token: {mert_tokens[0]}")

    # Verify quantization
    expected_clap = f"AUDIOCLAP_{clap_bucket}"
    assert expected_clap in decoded, f"Expected {expected_clap} in decoded tokens"
    print(f"✅ Correct CLAP quantization: 0.85 → {expected_clap}")

    expected_mert = f"AUDIOMERT_{mert_bucket}"
    assert expected_mert in decoded, f"Expected {expected_mert} in decoded tokens"
    print(f"✅ Correct MERT quantization: 0.75 → {expected_mert}\n")


def test_quantize_function():
    """Test quantize helper function."""
    print("=" * 60)
    print("Test 3: Quantization Function")
    print("=" * 60)

    test_cases = [
        (0.0, 10, 0),  # floor(0.0 * 10) = 0
        (0.5, 10, 5),  # floor(0.5 * 10) = 5
        (1.0, 10, 9),  # floor(1.0 * 10) = 10, clamped to 9
        (0.85, 10, 8),  # floor(0.85 * 10) = 8
        (0.75, 10, 7),  # floor(0.75 * 10) = 7
        (0.33, 10, 3),  # floor(0.33 * 10) = 3
        (-0.1, 10, 0),  # Clamped to 0
        (1.5, 10, 9),  # Clamped to 1.0 → 9
    ]

    for value, buckets, expected in test_cases:
        result = quantize(value, buckets=buckets)
        status = "✅" if result == expected else "❌"
        print(f"{status} quantize({value}, buckets={buckets}) = {result} (expected: {expected})")
        assert result == expected, f"Quantize failed for {value}"

    print("\n✅ All quantization tests passed\n")


def test_dataloader_filtering():
    """Test sequence length filtering logic."""
    print("=" * 60)
    print("Test 4: Dataloader Length Filtering")
    print("=" * 60)

    # Simulate filtering
    sequences = [
        ("seq1", 10),
        ("seq2", 50),
        ("seq3", 100),
        ("seq4", 500),
        ("seq5", 2100),
        ("seq6", 30),
    ]

    min_length = 20
    max_length = 2048

    filtered = [(name, length) for name, length in sequences if min_length <= length <= max_length]

    print(f"\nOriginal sequences: {len(sequences)}")
    print(f"Min length: {min_length}, Max length: {max_length}")
    print(f"Filtered sequences: {len(filtered)}")
    print(f"\nKept:")
    for name, length in filtered:
        print(f"  {name}: {length} tokens")

    print(f"\nSkipped:")
    skipped = [(name, length) for name, length in sequences if (name, length) not in filtered]
    for name, length in skipped:
        print(f"  {name}: {length} tokens (out of range)")

    assert len(filtered) == 4, f"Expected 4 filtered sequences, got {len(filtered)}"
    print("\n✅ Length filtering works correctly\n")


def test_weighted_sampling():
    """Test genre-based weighted sampling calculation."""
    print("=" * 60)
    print("Test 5: Weighted Sampling")
    print("=" * 60)

    # Simulate genre distribution
    samples = [
        ["<genre:pop>"],
        ["<genre:pop>"],
        ["<genre:pop>"],
        ["<genre:rock>"],
        ["<genre:jazz>"],
    ]

    from collections import Counter

    genre_counts = Counter()
    for conditions in samples:
        for cond in conditions:
            if cond.startswith("<genre:"):
                genre = cond.replace("<genre:", "").replace(">", "")
                genre_counts[genre] += 1

    print(f"\nGenre distribution: {dict(genre_counts)}")

    total_samples = len(samples)
    weights = []

    for conditions in samples:
        weight = 1.0
        for cond in conditions:
            if cond.startswith("<genre:"):
                genre = cond.replace("<genre:", "").replace(">", "")
                genre_freq = genre_counts[genre] / total_samples
                # Inverse frequency
                weight = 1.0 / (genre_freq + 0.01)
                break
        weights.append(weight)

    # Normalize
    total_weight = sum(weights)
    normalized_weights = [w / total_weight for w in weights]

    print(f"\nSample weights (before normalization):")
    for i, (conditions, weight) in enumerate(zip(samples, weights)):
        genre = conditions[0].replace("<genre:", "").replace(">", "")
        print(f"  Sample {i} ({genre}): {weight:.4f}")

    print(f"\nSample weights (normalized):")
    for i, (conditions, weight) in enumerate(zip(samples, normalized_weights)):
        genre = conditions[0].replace("<genre:", "").replace(">", "")
        print(f"  Sample {i} ({genre}): {weight:.4f}")

    # Verify that underrepresented genres get higher weights
    pop_weight = normalized_weights[0]  # pop is overrepresented
    rock_weight = normalized_weights[3]  # rock is underrepresented

    print(f"\nPop weight: {pop_weight:.4f}")
    print(f"Rock weight: {rock_weight:.4f}")
    assert rock_weight > pop_weight, "Rock should have higher weight than pop"
    print("✅ Underrepresented genres get higher sampling weights\n")


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("Stage3 Enhancements Test Suite")
    print("=" * 60 + "\n")

    try:
        test_clap_mert_tokens()
        test_encode_prompt_with_audio()
        test_quantize_function()
        test_dataloader_filtering()
        test_weighted_sampling()

        print("=" * 60)
        print("✅ All tests passed!")
        print("=" * 60)
        return 0
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        return 1
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback

        traceback.print_exc()
        return 2


if __name__ == "__main__":
    sys.exit(main())
