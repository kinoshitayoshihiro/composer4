#!/usr/bin/env python3
"""Example: Integrate AudioEmbeddingCache with Stage3 training.

This demonstrates how to use the embedding cache to:
1. Pre-compute and cache audio embeddings
2. Load normalized embeddings during training
3. Speed up training by avoiding repeated normalization

Integration points:
- Stage3Dataset: Load cached embeddings by file hash
- Stage3Tokenizer: Use cached bucket tokens for AUDIOCLAP/AUDIOMERT
"""

import logging
from pathlib import Path

import numpy as np


def create_example_cache():
    """Create example cache with mock audio embeddings."""
    import sys

    sys.path.insert(0, str(Path(__file__).parent.parent))
    from ml.audio_embedding_cache import AudioEmbeddingCache, compute_file_hash

    logging.basicConfig(level=logging.INFO)

    # Initialize cache
    cache = AudioEmbeddingCache(
        cache_dir="outputs/stage3/embedding_cache",
        sigma_clip=3.0,
        auto_save=True,
    )

    print("=" * 70)
    print("Stage3 Audio Embedding Cache Integration Example")
    print("=" * 70)

    # Simulate adding embeddings for training files
    print("\nStep 1: Adding embeddings for training files")
    print("-" * 70)

    # Mock: In real usage, you would:
    # 1. Extract CLAP/MERT embeddings from audio files
    # 2. Compute file hash for each MIDI file
    # 3. Cache the embeddings

    mock_files = [
        ("outputs/stage3/data/song001.mid", np.random.randn(512)),  # CLAP dim
        ("outputs/stage3/data/song002.mid", np.random.randn(512)),
        ("outputs/stage3/data/song003.mid", np.random.randn(512)),
        ("outputs/stage3/data/song004.mid", np.random.randn(512)),
        ("outputs/stage3/data/song005.mid", np.random.randn(512)),
    ]

    for file_path, clap_emb in mock_files:
        # Compute file hash (would use actual file in production)
        file_hash = f"mock_hash_{hash(file_path) % 10000:04d}"

        # Mock MERT embedding
        mert_emb = np.random.randn(768)  # MERT dim

        cache.add_embedding(
            file_hash=file_hash,
            clap_embedding=clap_emb,
            mert_embedding=mert_emb,
        )

        print(f"Cached: {file_hash} ({Path(file_path).name})")

    # Compute normalization statistics
    print("\nStep 2: Computing normalization statistics")
    print("-" * 70)

    cache.compute_statistics()

    summary = cache.summary()
    print(f"Cache summary:")
    print(f"  Entries: {summary['num_entries']}")
    print(f"  CLAP: mean={summary['clap_mean']:.4f}, std={summary['clap_std']:.4f}")
    print(f"  MERT: mean={summary['mert_mean']:.4f}, std={summary['mert_std']:.4f}")
    print(f"  Sigma clip: {summary['sigma_clip']}")

    # Retrieve normalized embeddings
    print("\nStep 3: Retrieving normalized embeddings")
    print("-" * 70)

    file_hash = f"mock_hash_{hash(mock_files[0][0]) % 10000:04d}"
    clap_norm, mert_norm = cache.get_normalized_embeddings(file_hash)

    print(f"File hash: {file_hash}")
    print(f"CLAP normalized shape: {clap_norm.shape}")
    print(f"CLAP range: [{np.min(clap_norm):.4f}, {np.max(clap_norm):.4f}]")
    print(f"MERT normalized shape: {mert_norm.shape}")
    print(f"MERT range: [{np.min(mert_norm):.4f}, {np.max(mert_norm):.4f}]")

    # Quantize to bucket tokens
    print("\nStep 4: Quantizing to bucket tokens")
    print("-" * 70)

    for file_path, _ in mock_files:
        file_hash = f"mock_hash_{hash(file_path) % 10000:04d}"
        clap_bucket, mert_bucket = cache.get_bucket_tokens(file_hash, buckets=10)

        print(f"{Path(file_path).name}: " f"AUDIOCLAP_{clap_bucket}, AUDIOMERT_{mert_bucket}")

    # Integration with Stage3Tokenizer
    print("\nStep 5: Integration with Stage3Tokenizer")
    print("-" * 70)

    print("In Stage3Dataset.__getitem__():")
    print("  1. Load MIDI file")
    print("  2. Compute file_hash = compute_file_hash(midi_path)")
    print("  3. Get buckets: clap_b, mert_b = cache.get_bucket_tokens(file_hash)")
    print("  4. Add tokens: conditions.extend([f'AUDIOCLAP_{clap_b}', f'AUDIOMERT_{mert_b}'])")
    print("\nBenefits:")
    print("  ✅ Embeddings computed once and cached")
    print("  ✅ Fast lookup during training (hash-based)")
    print("  ✅ Consistent normalization across runs")
    print("  ✅ Reduced memory footprint (cached on disk)")

    print("\n" + "=" * 70)
    print("Cache saved to: outputs/stage3/embedding_cache")
    print("=" * 70)
    print("\nUsage in training:")
    print("  cache = AudioEmbeddingCache('outputs/stage3/embedding_cache')")
    print("  clap_b, mert_b = cache.get_bucket_tokens(file_hash)")
    print()


if __name__ == "__main__":
    create_example_cache()
