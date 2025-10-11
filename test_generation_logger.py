#!/usr/bin/env python3
"""Test generation logger."""

import json
import tempfile
from pathlib import Path


def test_generation_logger():
    """Test generation logging and metadata tracking."""
    print("=" * 70)
    print("Generation Logger Test")
    print("=" * 70)

    # Import logger
    import sys

    sys.path.insert(0, str(Path(__file__).parent.parent))
    from ml.generation_logger import GenerationLogger

    # Create temporary log file
    with tempfile.TemporaryDirectory() as tmp_dir:
        log_file = Path(tmp_dir) / "generation_log.jsonl"

        logger = GenerationLogger(log_file=str(log_file), auto_commit_hash=True)

        # Test 1: Log generations
        print("\nTest 1: Logging generations")
        print("-" * 70)

        gen_id_1 = logger.log_generation(
            prompt={"genre": "rock", "emotion": "happy", "tempo": 120},
            output_file="outputs/song001.mid",
            model_checkpoint="checkpoints/epoch_10.ckpt",
            num_tokens=512,
            generation_params={"temperature": 0.9, "top_p": 0.9},
        )

        print(f"Gen ID 1: {gen_id_1}")

        gen_id_2 = logger.log_generation(
            prompt={"genre": "jazz", "emotion": "calm", "tempo": 100},
            output_file="outputs/song002.mid",
            model_checkpoint="checkpoints/epoch_10.ckpt",
            num_tokens=480,
        )

        print(f"Gen ID 2: {gen_id_2}")

        gen_id_3 = logger.log_generation(
            prompt={"genre": "rock", "emotion": "angry", "tempo": 140},
            output_file="outputs/song003.mid",
            model_checkpoint="checkpoints/epoch_20.ckpt",
            num_tokens=600,
        )

        print(f"Gen ID 3: {gen_id_3}")

        assert gen_id_1 != gen_id_2 != gen_id_3, "Generation IDs should be unique"
        print("✅ Unique generation IDs created")

        # Test 2: Retrieve metadata
        print("\nTest 2: Retrieving metadata")
        print("-" * 70)

        metadata = logger.get_generation_metadata(gen_id_1)

        assert metadata is not None, "Metadata not found"
        assert metadata["prompt"]["genre"] == "rock", "Prompt not stored correctly"
        assert metadata["num_tokens"] == 512, "Token count not stored"

        print(f"Generation ID: {metadata['generation_id']}")
        print(f"Prompt: {metadata['prompt']}")
        print(f"Output: {metadata['output_file']}")
        print(f"Checkpoint: {metadata['model_checkpoint']}")
        print(f"Tokens: {metadata['num_tokens']}")
        print("✅ Metadata retrieval working")

        # Test 3: Query by prompt
        print("\nTest 3: Querying by prompt")
        print("-" * 70)

        rock_gens = logger.query_by_prompt({"genre": "rock"})

        assert len(rock_gens) == 2, f"Expected 2 rock generations, got {len(rock_gens)}"

        print(f"Found {len(rock_gens)} generations with genre='rock':")
        for gen in rock_gens:
            print(f"  - {gen['generation_id']}: {gen['prompt']['emotion']}")

        print("✅ Prompt filtering working")

        # Test 4: Embed metadata in MIDI
        print("\nTest 4: Embedding metadata in MIDI")
        print("-" * 70)

        midi_file = Path(tmp_dir) / "test_song.mid"
        midi_file.touch()  # Create dummy file

        logger.embed_metadata_in_midi(str(midi_file), gen_id_1)

        # Check sidecar file
        sidecar_file = midi_file.with_suffix(".meta.json")
        assert sidecar_file.exists(), "Sidecar metadata file not created"

        with open(sidecar_file) as f:
            sidecar_data = json.load(f)

        assert sidecar_data["generation_id"] == gen_id_1, "Generation ID not embedded"
        print(f"Metadata embedded in: {sidecar_file}")
        print(f"  generation_id: {sidecar_data['generation_id']}")
        print("✅ Metadata embedding working")

        # Test 5: A/B comparison export
        print("\nTest 5: A/B comparison export")
        print("-" * 70)

        ab_file = Path(tmp_dir) / "ab_comparison.csv"

        logger.export_for_ab_comparison(
            checkpoint_a="checkpoints/epoch_10.ckpt",
            checkpoint_b="checkpoints/epoch_20.ckpt",
            output_file=str(ab_file),
        )

        assert ab_file.exists(), "A/B comparison file not created"

        with open(ab_file) as f:
            lines = f.readlines()

        # Header + 2 generations from epoch_10 + 1 from epoch_20
        assert len(lines) == 4, f"Expected 4 lines (header + 3 data), got {len(lines)}"

        print(f"A/B comparison exported: {ab_file}")
        print(f"  Total entries: {len(lines) - 1}")
        print("✅ A/B export working")

        # Test 6: Summary statistics
        print("\nTest 6: Summary statistics")
        print("-" * 70)

        summary = logger.summary()

        print(f"Total generations: {summary['total_generations']}")
        print(f"Checkpoint counts: {summary['checkpoint_counts']}")
        print(f"Genre counts: {summary['genre_counts']}")

        assert summary["total_generations"] == 3, "Total count mismatch"
        assert summary["genre_counts"]["rock"] == 2, "Genre count mismatch"

        print("✅ Summary statistics working")

        # Test 7: Persistence (save and reload)
        print("\nTest 7: Persistence (save and reload)")
        print("-" * 70)

        # Create new logger instance
        logger2 = GenerationLogger(log_file=str(log_file), auto_commit_hash=False)

        assert len(logger2.generations) == 3, "Log not loaded correctly"

        metadata2 = logger2.get_generation_metadata(gen_id_1)
        assert metadata2 is not None, "Metadata not persisted"
        assert metadata2["prompt"]["genre"] == "rock", "Prompt not persisted"

        print(f"Reloaded {len(logger2.generations)} generations from log")
        print("✅ Persistence working")

    print("\n" + "=" * 70)
    print("All tests passed!")
    print("=" * 70)
    print("\nKey Features Validated:")
    print("✅ Unique generation ID computation")
    print("✅ Prompt + model metadata logging")
    print("✅ JSONL persistence")
    print("✅ Metadata retrieval by generation ID")
    print("✅ Query by prompt attributes")
    print("✅ MIDI metadata embedding (sidecar)")
    print("✅ A/B comparison export")
    print("✅ Summary statistics\n")


if __name__ == "__main__":
    test_generation_logger()
