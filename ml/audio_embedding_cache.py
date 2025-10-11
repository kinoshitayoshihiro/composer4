#!/usr/bin/env python3
"""Audio embedding normalization and caching for Stage3.

This module provides:
1. Z-score normalization of CLAP/MERT embeddings
2. Clipping to [-3, 3] sigma range
3. Projection to [0, 1] unit interval
4. Hash-based caching for fast lookup during training/inference

Usage:
    # Initialize cache
    cache = AudioEmbeddingCache(cache_dir="outputs/stage3/embedding_cache")

    # Normalize and cache embeddings
    cache.add_embedding(
        file_hash="abc123",
        clap_embedding=np.array([0.1, 0.2, ...]),
        mert_embedding=np.array([0.3, 0.4, ...])
    )

    # Retrieve normalized embeddings
    clap_norm, mert_norm = cache.get_normalized_embeddings("abc123")

    # Quantize to discrete buckets
    clap_bucket = cache.quantize_to_bucket(clap_norm, buckets=10)
"""

import hashlib
import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np


class AudioEmbeddingCache:
    """Cache for normalized audio embeddings with z-score normalization."""

    def __init__(
        self,
        cache_dir: str = "outputs/stage3/embedding_cache",
        sigma_clip: float = 3.0,
        auto_save: bool = True,
    ):
        """Initialize embedding cache.

        Args:
            cache_dir: Directory to store cache files
            sigma_clip: Number of standard deviations for clipping (default: 3.0)
            auto_save: Automatically save cache on updates
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.sigma_clip = sigma_clip
        self.auto_save = auto_save

        # Cache storage: {file_hash: {"clap": array, "mert": array}}
        self.embeddings: dict[str, dict[str, np.ndarray]] = {}

        # Normalization statistics (computed from all cached embeddings)
        self.clap_mean: Optional[float] = None
        self.clap_std: Optional[float] = None
        self.mert_mean: Optional[float] = None
        self.mert_std: Optional[float] = None

        # Load existing cache if available
        self._load_cache()

    def add_embedding(
        self,
        file_hash: str,
        clap_embedding: Optional[np.ndarray] = None,
        mert_embedding: Optional[np.ndarray] = None,
    ) -> None:
        """Add raw embeddings to cache.

        Args:
            file_hash: Unique hash identifier for the audio file
            clap_embedding: Raw CLAP embedding vector
            mert_embedding: Raw MERT embedding vector
        """
        if file_hash not in self.embeddings:
            self.embeddings[file_hash] = {}

        if clap_embedding is not None:
            self.embeddings[file_hash]["clap"] = np.array(clap_embedding)

        if mert_embedding is not None:
            self.embeddings[file_hash]["mert"] = np.array(mert_embedding)

        if self.auto_save:
            self._save_cache()

    def compute_statistics(self) -> None:
        """Compute z-score normalization statistics from all cached embeddings."""
        if not self.embeddings:
            logging.warning("No embeddings in cache, cannot compute statistics")
            return

        # Collect all CLAP embeddings
        clap_values = []
        for entry in self.embeddings.values():
            if "clap" in entry:
                clap_values.extend(entry["clap"].flatten())

        # Collect all MERT embeddings
        mert_values = []
        for entry in self.embeddings.values():
            if "mert" in entry:
                mert_values.extend(entry["mert"].flatten())

        # Compute statistics
        if clap_values:
            self.clap_mean = float(np.mean(clap_values))
            self.clap_std = float(np.std(clap_values))
            logging.info(f"CLAP stats: mean={self.clap_mean:.4f}, std={self.clap_std:.4f}")

        if mert_values:
            self.mert_mean = float(np.mean(mert_values))
            self.mert_std = float(np.std(mert_values))
            logging.info(f"MERT stats: mean={self.mert_mean:.4f}, std={self.mert_std:.4f}")

        if self.auto_save:
            self._save_cache()

    def normalize_embedding(
        self,
        embedding: np.ndarray,
        mean: float,
        std: float,
    ) -> np.ndarray:
        """Apply z-score normalization with clipping and [0,1] projection.

        Pipeline:
            1. Z-score: (x - mean) / std
            2. Clip: [-sigma_clip, sigma_clip]
            3. Project: [0, 1] via linear mapping

        Args:
            embedding: Raw embedding vector
            mean: Mean for z-score normalization
            std: Standard deviation for z-score normalization

        Returns:
            Normalized embedding in [0, 1] range
        """
        # Avoid division by zero
        if std == 0:
            std = 1.0

        # Z-score normalization
        z_score = (embedding - mean) / std

        # Clip to [-sigma_clip, sigma_clip]
        clipped = np.clip(z_score, -self.sigma_clip, self.sigma_clip)

        # Project to [0, 1]
        # Map [-sigma_clip, sigma_clip] -> [0, 1]
        normalized = (clipped + self.sigma_clip) / (2 * self.sigma_clip)

        return normalized

    def get_normalized_embeddings(
        self,
        file_hash: str,
    ) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Retrieve normalized embeddings for a file.

        Args:
            file_hash: File hash identifier

        Returns:
            Tuple of (clap_normalized, mert_normalized), or (None, None) if not found
        """
        if file_hash not in self.embeddings:
            return None, None

        entry = self.embeddings[file_hash]

        # Normalize CLAP
        clap_norm = None
        if "clap" in entry and self.clap_mean is not None:
            clap_norm = self.normalize_embedding(
                entry["clap"],
                self.clap_mean,
                self.clap_std or 1.0,
            )

        # Normalize MERT
        mert_norm = None
        if "mert" in entry and self.mert_mean is not None:
            mert_norm = self.normalize_embedding(
                entry["mert"],
                self.mert_mean,
                self.mert_std or 1.0,
            )

        return clap_norm, mert_norm

    def quantize_to_bucket(
        self,
        normalized_value: float,
        buckets: int = 10,
    ) -> int:
        """Quantize normalized value to discrete bucket.

        Uses floor-based quantization for monotonic mapping.

        Args:
            normalized_value: Value in [0, 1] range
            buckets: Number of discrete buckets (default: 10)

        Returns:
            Bucket index in [0, buckets-1]
        """
        clamped = min(max(float(normalized_value), 0.0), 1.0)
        return min(int(clamped * buckets), buckets - 1)

    def get_bucket_tokens(
        self,
        file_hash: str,
        buckets: int = 10,
    ) -> tuple[Optional[int], Optional[int]]:
        """Get quantized bucket indices for CLAP and MERT.

        Args:
            file_hash: File hash identifier
            buckets: Number of buckets for quantization

        Returns:
            Tuple of (clap_bucket, mert_bucket), or (None, None) if not found
        """
        clap_norm, mert_norm = self.get_normalized_embeddings(file_hash)

        clap_bucket = None
        if clap_norm is not None:
            # Use mean of embedding vector
            clap_bucket = self.quantize_to_bucket(float(np.mean(clap_norm)), buckets)

        mert_bucket = None
        if mert_norm is not None:
            mert_bucket = self.quantize_to_bucket(float(np.mean(mert_norm)), buckets)

        return clap_bucket, mert_bucket

    def _save_cache(self) -> None:
        """Save cache to disk."""
        cache_file = self.cache_dir / "embeddings.npz"
        stats_file = self.cache_dir / "statistics.json"

        # Save embeddings as .npz (compressed numpy format)
        np.savez_compressed(cache_file, **self.embeddings)

        # Save statistics as JSON
        stats = {
            "clap_mean": self.clap_mean,
            "clap_std": self.clap_std,
            "mert_mean": self.mert_mean,
            "mert_std": self.mert_std,
            "sigma_clip": self.sigma_clip,
            "num_entries": len(self.embeddings),
        }

        with open(stats_file, "w") as f:
            json.dump(stats, f, indent=2)

        logging.debug(f"Cache saved: {len(self.embeddings)} entries")

    def _load_cache(self) -> None:
        """Load cache from disk."""
        cache_file = self.cache_dir / "embeddings.npz"
        stats_file = self.cache_dir / "statistics.json"

        # Load embeddings
        if cache_file.exists():
            data = np.load(cache_file, allow_pickle=True)
            self.embeddings = {key: data[key].item() for key in data.files}
            logging.info(f"Loaded cache: {len(self.embeddings)} entries")

        # Load statistics
        if stats_file.exists():
            with open(stats_file) as f:
                stats = json.load(f)

            self.clap_mean = stats.get("clap_mean")
            self.clap_std = stats.get("clap_std")
            self.mert_mean = stats.get("mert_mean")
            self.mert_std = stats.get("mert_std")
            self.sigma_clip = stats.get("sigma_clip", 3.0)

            logging.info("Loaded normalization statistics")

    def summary(self) -> dict:
        """Return cache summary statistics."""
        return {
            "num_entries": len(self.embeddings),
            "clap_mean": self.clap_mean,
            "clap_std": self.clap_std,
            "mert_mean": self.mert_mean,
            "mert_std": self.mert_std,
            "sigma_clip": self.sigma_clip,
            "cache_dir": str(self.cache_dir),
        }


def compute_file_hash(file_path: str) -> str:
    """Compute SHA256 hash of a file for cache key.

    Args:
        file_path: Path to file

    Returns:
        Hex digest of file hash
    """
    hasher = hashlib.sha256()

    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hasher.update(chunk)

    return hasher.hexdigest()
