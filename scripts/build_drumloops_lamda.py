#!/usr/bin/env python3
"""CLI wrapper for the reusable drum-loop LAMDa builder."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

from lamda_tools import DrumLoopBuildConfig, build_drumloops


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LAMDa drum-loop dataset builder")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/loops"),
        help="Directory containing source MIDI files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/drumloops_cleaned"),
        help="Directory to store cleaned MIDI files.",
    )
    parser.add_argument(
        "--metadata-dir",
        type=Path,
        default=Path("output/drumloops_metadata"),
        help="Directory to store intermediate metadata pickles.",
    )
    parser.add_argument(
        "--tmidix-path",
        type=Path,
        default=Path("data/Los-Angeles-MIDI/CODE"),
        help="Path to the TMIDIX module from LAMDa.",
    )
    parser.add_argument(
        "--min-notes",
        type=int,
        default=16,
        help="Minimum number of notes required to treat a file as a loop.",
    )
    parser.add_argument(
        "--max-file-size",
        type=int,
        default=1_000_000,
        help="Maximum MIDI file size (bytes) to accept.",
    )
    parser.add_argument(
        "--save-interval",
        type=int,
        default=5_000,
        help="Number of accepted loops between checkpoint saves.",
    )
    parser.add_argument(
        "--start-file-number",
        type=int,
        default=0,
        help="Skip this many files from the shuffled list before processing.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for shuffling input files (omit to disable).",
    )
    parser.add_argument(
        "--allow-monophonic",
        action="store_true",
        help="Do not require chord/polyphonic content in drum loops.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print summary information without performing dedupe.",
    )
    return parser.parse_args()


def _build_config(args: argparse.Namespace) -> DrumLoopBuildConfig:
    kwargs: Dict[str, Any] = {
        "dataset_name": "Drum Loops",
        "input_dir": args.input_dir,
        "output_dir": args.output_dir,
        "metadata_dir": args.metadata_dir,
        "tmidix_path": args.tmidix_path,
        "min_notes": args.min_notes,
        "max_file_size": args.max_file_size,
        "save_interval": args.save_interval,
        "start_file_number": args.start_file_number,
        "random_seed": args.seed,
        "require_polyphony": not args.allow_monophonic,
    }
    return DrumLoopBuildConfig(**kwargs)


def main() -> None:
    args = _parse_args()
    config = _build_config(args)
    build_drumloops(config, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
