#!/usr/bin/env python3
"""Generation logging for Stage3 causality tracking.

This module provides:
1. Prompt + model metadata logging to JSONL
2. Generation ID assignment for reproducibility
3. Hash-based join with Stage2 evaluation results
4. Metadata embedding in MIDI files

Usage:
    # Initialize logger
    logger = GenerationLogger(log_file="outputs/stage3/generation_log.jsonl")

    # Log generation
    gen_id = logger.log_generation(
        prompt={"genre": "rock", "emotion": "happy"},
        model_commit="abc123",
        tokenizer_hash="def456",
        output_file="outputs/song.mid"
    )

    # Embed metadata in MIDI
    logger.embed_metadata_in_midi("outputs/song.mid", gen_id)

    # Query by generation ID
    metadata = logger.get_generation_metadata(gen_id)
"""

import hashlib
import json
import logging
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Optional


class GenerationLogger:
    """Logger for tracking generation provenance and enabling A/B comparison."""

    def __init__(
        self,
        log_file: str = "outputs/stage3/generation_log.jsonl",
        auto_commit_hash: bool = True,
    ):
        """Initialize generation logger.

        Args:
            log_file: Path to JSONL log file
            auto_commit_hash: Automatically detect git commit hash
        """
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)

        self.auto_commit_hash = auto_commit_hash

        # In-memory cache of logged generations
        self.generations: dict[str, dict[str, Any]] = {}

        # Load existing log
        self._load_log()

    def log_generation(
        self,
        prompt: dict[str, Any],
        output_file: str,
        model_checkpoint: Optional[str] = None,
        model_commit: Optional[str] = None,
        tokenizer_hash: Optional[str] = None,
        num_tokens: Optional[int] = None,
        generation_params: Optional[dict[str, Any]] = None,
    ) -> str:
        """Log a generation event.

        Args:
            prompt: Prompt dictionary (genre, emotion, etc.)
            output_file: Path to generated MIDI file
            model_checkpoint: Model checkpoint path
            model_commit: Git commit hash of model code
            tokenizer_hash: Hash of tokenizer configuration
            num_tokens: Number of generated tokens
            generation_params: Generation parameters (temperature, top_p, etc.)

        Returns:
            Generation ID (unique hash)
        """
        # Auto-detect git commit if enabled
        if model_commit is None and self.auto_commit_hash:
            model_commit = self._get_git_commit()

        # Compute generation ID
        gen_id = self._compute_generation_id(
            prompt=prompt,
            model_checkpoint=model_checkpoint,
            timestamp=datetime.now().isoformat(),
        )

        # Create log entry
        entry = {
            "generation_id": gen_id,
            "timestamp": datetime.now().isoformat(),
            "prompt": prompt,
            "output_file": str(output_file),
            "model_checkpoint": model_checkpoint,
            "model_commit": model_commit,
            "tokenizer_hash": tokenizer_hash,
            "num_tokens": num_tokens,
            "generation_params": generation_params or {},
        }

        # Add to cache and append to log
        self.generations[gen_id] = entry
        self._append_to_log(entry)

        logging.info(f"Logged generation: {gen_id}")

        return gen_id

    def get_generation_metadata(self, gen_id: str) -> Optional[dict[str, Any]]:
        """Retrieve metadata for a generation ID.

        Args:
            gen_id: Generation ID

        Returns:
            Metadata dictionary or None if not found
        """
        return self.generations.get(gen_id)

    def embed_metadata_in_midi(
        self,
        midi_file: str,
        gen_id: str,
    ) -> None:
        """Embed generation ID in MIDI file metadata.

        Note: This is a placeholder. Real implementation would use
        pretty_midi or mido to add metadata to MIDI file.

        Args:
            midi_file: Path to MIDI file
            gen_id: Generation ID to embed
        """
        logging.info(f"Embedding gen_id={gen_id} in {midi_file}")

        # Placeholder: In real implementation, you would:
        # 1. Load MIDI with pretty_midi or mido
        # 2. Add meta message with generation_id
        # 3. Save modified MIDI

        # For now, create a sidecar JSON file
        sidecar_file = Path(midi_file).with_suffix(".meta.json")
        with open(sidecar_file, "w") as f:
            json.dump({"generation_id": gen_id}, f, indent=2)

        logging.info(f"Metadata saved to: {sidecar_file}")

    def compute_tokenizer_hash(self, tokenizer_config: dict[str, Any]) -> str:
        """Compute hash of tokenizer configuration for versioning.

        Args:
            tokenizer_config: Tokenizer configuration dict

        Returns:
            SHA256 hash of config
        """
        config_str = json.dumps(tokenizer_config, sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()[:16]

    def query_by_prompt(self, prompt_filter: dict[str, Any]) -> list[dict[str, Any]]:
        """Query generations by prompt attributes.

        Args:
            prompt_filter: Filter dict (e.g., {"genre": "rock"})

        Returns:
            List of matching generation entries
        """
        results = []

        for entry in self.generations.values():
            prompt = entry.get("prompt", {})

            # Check if all filter keys match
            match = all(prompt.get(key) == value for key, value in prompt_filter.items())

            if match:
                results.append(entry)

        return results

    def export_for_ab_comparison(
        self,
        checkpoint_a: str,
        checkpoint_b: str,
        output_file: str,
    ) -> None:
        """Export generations for A/B comparison.

        Args:
            checkpoint_a: Path to checkpoint A
            checkpoint_b: Path to checkpoint B
            output_file: Output CSV path for comparison
        """
        import csv

        # Find generations for each checkpoint
        gens_a = [g for g in self.generations.values() if g.get("model_checkpoint") == checkpoint_a]

        gens_b = [g for g in self.generations.values() if g.get("model_checkpoint") == checkpoint_b]

        logging.info(f"Checkpoint A: {len(gens_a)} generations")
        logging.info(f"Checkpoint B: {len(gens_b)} generations")

        # Export to CSV
        with open(output_file, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["checkpoint", "generation_id", "prompt", "output_file"],
            )
            writer.writeheader()

            for gen in gens_a:
                writer.writerow(
                    {
                        "checkpoint": "A",
                        "generation_id": gen["generation_id"],
                        "prompt": json.dumps(gen["prompt"]),
                        "output_file": gen["output_file"],
                    }
                )

            for gen in gens_b:
                writer.writerow(
                    {
                        "checkpoint": "B",
                        "generation_id": gen["generation_id"],
                        "prompt": json.dumps(gen["prompt"]),
                        "output_file": gen["output_file"],
                    }
                )

        logging.info(f"A/B comparison exported to: {output_file}")

    def _compute_generation_id(
        self,
        prompt: dict[str, Any],
        model_checkpoint: Optional[str],
        timestamp: str,
    ) -> str:
        """Compute unique generation ID.

        Args:
            prompt: Prompt dict
            model_checkpoint: Model checkpoint path
            timestamp: ISO timestamp

        Returns:
            SHA256 hash (first 16 chars)
        """
        id_str = json.dumps(
            {
                "prompt": prompt,
                "model_checkpoint": model_checkpoint,
                "timestamp": timestamp,
            },
            sort_keys=True,
        )

        return hashlib.sha256(id_str.encode()).hexdigest()[:16]

    def _get_git_commit(self) -> Optional[str]:
        """Get current git commit hash.

        Returns:
            Git commit hash or None if not in git repo
        """
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                check=True,
            )
            return result.stdout.strip()[:8]  # Short hash
        except (subprocess.CalledProcessError, FileNotFoundError):
            return None

    def _append_to_log(self, entry: dict[str, Any]) -> None:
        """Append entry to JSONL log file.

        Args:
            entry: Log entry dict
        """
        with open(self.log_file, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def _load_log(self) -> None:
        """Load existing log from file."""
        if not self.log_file.exists():
            return

        with open(self.log_file) as f:
            for line in f:
                if line.strip():
                    entry = json.loads(line)
                    gen_id = entry.get("generation_id")
                    if gen_id:
                        self.generations[gen_id] = entry

        logging.info(f"Loaded {len(self.generations)} generations from log")

    def summary(self) -> dict[str, Any]:
        """Return logger summary statistics.

        Returns:
            Summary dict
        """
        # Count by checkpoint
        checkpoint_counts: dict[str, int] = {}
        for entry in self.generations.values():
            ckpt = entry.get("model_checkpoint", "unknown")
            checkpoint_counts[ckpt] = checkpoint_counts.get(ckpt, 0) + 1

        # Count by prompt genre
        genre_counts: dict[str, int] = {}
        for entry in self.generations.values():
            genre = entry.get("prompt", {}).get("genre", "unknown")
            genre_counts[genre] = genre_counts.get(genre, 0) + 1

        return {
            "total_generations": len(self.generations),
            "log_file": str(self.log_file),
            "checkpoint_counts": checkpoint_counts,
            "genre_counts": genre_counts,
        }
