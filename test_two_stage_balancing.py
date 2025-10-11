#!/usr/bin/env python3
"""Manual test for two-stage balancing (genre × emotion)."""

from collections import Counter


def test_two_stage_balancing():
    """Test (genre, emotion) pair weighting."""
    print("=" * 70)
    print("Two-Stage Balancing - (Genre, Emotion) Pair Weights")
    print("=" * 70)

    # Mock dataset
    samples = [
        {"genre": "pop", "emotion": "happy"},
        {"genre": "pop", "emotion": "happy"},
        {"genre": "pop", "emotion": "happy"},
        {"genre": "pop", "emotion": "happy"},
        {"genre": "pop", "emotion": "happy"},
        {"genre": "rock", "emotion": "angry"},
        {"genre": "rock", "emotion": "angry"},
        {"genre": "rock", "emotion": "sad"},  # Rare pair
        {"genre": "jazz", "emotion": "calm"},
        {"genre": "jazz", "emotion": "calm"},
    ]

    # Count (genre, emotion) pairs
    pair_counts: Counter[tuple[str, str]] = Counter()
    for sample in samples:
        pair = (sample["genre"], sample["emotion"])
        pair_counts[pair] += 1

    total_samples = len(samples)
    epsilon = 0.01

    # Compute weights (inverse frequency)
    weights = []
    for sample in samples:
        pair = (sample["genre"], sample["emotion"])
        pair_freq = pair_counts[pair] / total_samples
        weight = 1.0 / (pair_freq + epsilon)
        weights.append(weight)

    # Normalize
    total_weight = sum(weights)
    weights = [w / total_weight for w in weights]

    # Aggregate by pair
    pair_weights: dict[tuple[str, str], list[float]] = {}
    for sample, weight in zip(samples, weights):
        pair = (sample["genre"], sample["emotion"])
        if pair not in pair_weights:
            pair_weights[pair] = []
        pair_weights[pair].append(weight)

    # Report
    print("\nPair Distribution:")
    print(f"{'Genre':<10} {'Emotion':<10} {'Count':<8} {'Frequency':<12} {'Avg Weight':<12}")
    print("-" * 70)

    for pair, count in pair_counts.most_common():
        genre, emotion = pair
        freq = count / total_samples
        avg_weight = sum(pair_weights[pair]) / len(pair_weights[pair])
        print(f"{genre:<10} {emotion:<10} {count:<8} {freq:<12.4f} {avg_weight:<12.6f}")

    print("\n" + "=" * 70)
    print("Key Insights:")
    print("=" * 70)

    # Most common pair
    most_common_pair = pair_counts.most_common(1)[0]
    most_common_weight = sum(pair_weights[most_common_pair[0]]) / len(
        pair_weights[most_common_pair[0]]
    )

    # Rarest pair
    rarest_pair = pair_counts.most_common()[-1]
    rarest_weight = sum(pair_weights[rarest_pair[0]]) / len(pair_weights[rarest_pair[0]])

    print(
        f"Most common: {most_common_pair[0]} (count={most_common_pair[1]}, weight={most_common_weight:.6f})"
    )
    print(f"Rarest: {rarest_pair[0]} (count={rarest_pair[1]}, weight={rarest_weight:.6f})")

    # Weight ratio
    weight_ratio = rarest_weight / most_common_weight
    print(f"\nWeight boost for rare pairs: {weight_ratio:.2f}x")
    print(
        f"✅ Rare (genre, emotion) combinations get {weight_ratio:.2f}x higher sampling probability"
    )

    # Verify normalization
    total_prob = sum(weights)
    print(f"\nTotal probability: {total_prob:.6f} (should be ~1.0)")
    assert abs(total_prob - 1.0) < 0.001, "Weights not normalized"
    print("✅ Weights properly normalized\n")


if __name__ == "__main__":
    test_two_stage_balancing()
