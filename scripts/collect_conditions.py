#!/usr/bin/env python3
"""Collect and merge all condition data for Stage3.

This script aggregates:
- XMIDI emotion/genre labels
- MetaScore captions
- VPTT technique metadata
- CLAP/MERT audio embeddings

into a unified conditions/*.parquet format.

Usage:
    PYTHONPATH=. python scripts/collect_conditions.py \
        --stage2-summary output/drumloops_stage2/loop_summary.csv \
        --xmidi-labels outputs/stage3/xmidi_labels.csv \
        --captions outputs/stage3/music_captions.jsonl \
        --technique-meta outputs/stage3/technique_metadata.jsonl \
        --audio-cache outputs/stage3/embedding_cache \
        --output conditions/stage3_conditions.parquet
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def _normalize_midi_key(value: Any) -> str | None:
    """Return a normalized MIDI stem (e.g. XMIDI_angry_123) used for joins."""
    if not isinstance(value, str):
        return None
    cleaned = value.strip().replace("\r", "").replace("\n", "")
    if not cleaned:
        return None
    cleaned = cleaned.replace("\\", "/")
    stem = Path(cleaned).stem
    return stem or None


def load_stage2_summary(path: Path) -> pd.DataFrame:
    """Load Stage2 loop summary as base."""
    logging.info("Loading Stage2 summary from %s", path)
    df = pd.read_csv(path)
    df["xmidi_key"] = None
    if "file" in df.columns:
        df["xmidi_key"] = df["file"].map(_normalize_midi_key)
    if "loop_id" in df.columns:
        df["xmidi_key"] = df["xmidi_key"].fillna(df["loop_id"].map(_normalize_midi_key))

    if "loop_id" not in df.columns:
        logging.info("Deriving loop_id from available columns")
        fallback = df["xmidi_key"]
        if "file" in df.columns:
            fallback = fallback.fillna(df["file"].astype(str).str.strip())
        df["loop_id"] = fallback

    if "file_digest" not in df.columns:
        if "loop_id" in df.columns:
            logging.info("Populating file_digest with loop_id values")
            df["file_digest"] = df["loop_id"]
        elif "file" in df.columns:
            logging.info("Populating file_digest with 'file' column")
            df["file_digest"] = df["file"].astype(str).str.strip()
    logging.info("Loaded %d rows from Stage2", len(df))
    return df


def merge_xmidi_labels(df: pd.DataFrame, path: Path | None) -> pd.DataFrame:
    """Merge XMIDI emotion/genre labels."""
    if path is None or not path.exists():
        logging.warning("XMIDI labels not found, skipping")
        return df

    logging.info("Merging XMIDI labels from %s", path)
    xmidi = pd.read_csv(path)

    # Ensure required columns
    required = ["emotion", "genre", "valence", "arousal"]
    missing = [col for col in required if col not in xmidi.columns]
    if missing:
        logging.warning("XMIDI missing columns %s, skipping", missing)
        return df

    stage_key = None
    for candidate in ("xmidi_key", "loop_id", "file"):
        if candidate in df.columns:
            stage_key = candidate
            break

    if stage_key is None:
        logging.warning("No suitable key in base data for XMIDI merge, skipping")
        return df

    join_key_col = "xmidi_key"
    path_columns = [
        "midi_path",
        "midi_file",
        "relative_path",
        "file",
    ]
    for col in path_columns:
        if col in xmidi.columns:
            xmidi[join_key_col] = xmidi[col].map(_normalize_midi_key)
            break
    else:
        if "loop_id" in xmidi.columns:
            xmidi[join_key_col] = xmidi["loop_id"].map(_normalize_midi_key)
        else:
            xmidi[join_key_col] = None

    if xmidi[join_key_col].isna().all():
        logging.warning("XMIDI labels missing join key, skipping")
        return df

    subset_cols = [join_key_col] + required
    if "loop_id" in xmidi.columns:
        subset_cols.append("loop_id")
    xmidi_subset = (
        xmidi[subset_cols]
        .dropna(subset=[join_key_col])
        .drop_duplicates(subset=[join_key_col])
        .rename(columns={join_key_col: "__xmidi_join_key", "loop_id": "xmidi_loop_id"})
    )

    df = df.merge(
        xmidi_subset,
        left_on=stage_key,
        right_on="__xmidi_join_key",
        how="left",
        suffixes=("", "_xmidi"),
    )
    df = df.drop(columns=["__xmidi_join_key"])

    logging.info("XMIDI labels merged using %s", stage_key)
    return df


def merge_captions(df: pd.DataFrame, path: Path | None) -> pd.DataFrame:
    """Merge MetaScore captions."""
    if path is None or not path.exists():
        logging.warning("Captions not found, skipping")
        return df

    logging.info("Merging captions from %s", path)
    captions = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    captions.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    if not captions:
        logging.warning("No valid captions found")
        return df

    caption_df = pd.DataFrame(captions)

    # Extract loop_id from filename or digest
    if "filename" in caption_df.columns:
        caption_df["loop_id"] = caption_df["filename"].str.replace(".mid", "")
    elif "digest" in caption_df.columns:
        caption_df["loop_id"] = caption_df["digest"]

    if "caption" in caption_df.columns and "loop_id" in caption_df.columns:
        df = df.merge(
            caption_df[["loop_id", "caption"]],
            on="loop_id",
            how="left",
            suffixes=("", "_caption"),
        )
        logging.info("Captions merged")
    else:
        logging.warning("Caption format invalid, skipping")

    return df


def merge_techniques(df: pd.DataFrame, path: Path | None) -> pd.DataFrame:
    """Merge VPTT technique metadata."""
    if path is None or not path.exists():
        logging.warning("Technique metadata not found, skipping")
        return df

    logging.info("Merging technique metadata from %s", path)
    techniques: dict[str, list[str]] = {}

    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    obj = json.loads(line)
                    digest = Path(obj.get("input_file", "")).stem
                    tech = obj.get("technique")
                    if digest and tech:
                        if digest not in techniques:
                            techniques[digest] = []
                        techniques[digest].append(tech)
                except json.JSONDecodeError:
                    continue

    if not techniques:
        logging.warning("No valid techniques found")
        return df

    # Create technique column
    df["technique"] = df["file_digest"].map(
        lambda x: ",".join(techniques.get(x, [])) if x in techniques else None
    )

    logging.info("Techniques merged for %d samples", df["technique"].notna().sum())
    return df


def merge_audio_embeddings(df: pd.DataFrame, cache_dir: Path | None) -> pd.DataFrame:
    """Merge CLAP/MERT embeddings from cache."""
    if cache_dir is None or not cache_dir.exists():
        logging.warning("Audio cache not found, skipping")
        return df

    logging.info("Merging audio embeddings from %s", cache_dir)

    # Look for cache files
    clap_cache = cache_dir / "clap_embeddings.parquet"
    mert_cache = cache_dir / "mert_embeddings.parquet"

    if clap_cache.exists():
        clap_df = pd.read_parquet(clap_cache)
        if "file_digest" in clap_df.columns:
            df = df.merge(
                clap_df[["file_digest", "clap_embedding"]],
                on="file_digest",
                how="left",
            )
            logging.info("CLAP embeddings merged")

    if mert_cache.exists():
        mert_df = pd.read_parquet(mert_cache)
        if "file_digest" in mert_df.columns:
            df = df.merge(
                mert_df[["file_digest", "mert_embedding"]],
                on="file_digest",
                how="left",
            )
            logging.info("MERT embeddings merged")

    return df


def validate_output(df: pd.DataFrame) -> dict[str, Any]:
    """Validate merged conditions and return statistics."""
    stats: dict[str, Any] = {
        "total_rows": len(df),
        "null_rates": {},
        "value_counts": {},
    }

    # Check null rates
    for col in ["emotion", "genre", "caption", "technique"]:
        if col in df.columns:
            null_rate = df[col].isna().sum() / len(df)
            stats["null_rates"][col] = float(null_rate)
            if null_rate > 0.5:
                logging.warning("High null rate in %s: %.1f%%", col, null_rate * 100)

    # Check value distributions
    if "emotion" in df.columns:
        stats["value_counts"]["emotion"] = df["emotion"].value_counts().to_dict()
    if "genre" in df.columns:
        stats["value_counts"]["genre"] = df["genre"].value_counts().to_dict()

    # Check embedding coverage
    if "clap_embedding" in df.columns:
        stats["clap_coverage"] = float(df["clap_embedding"].notna().sum() / len(df))
    if "mert_embedding" in df.columns:
        stats["mert_coverage"] = float(df["mert_embedding"].notna().sum() / len(df))

    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect Stage3 conditions")
    parser.add_argument(
        "--stage2-summary",
        type=Path,
        required=True,
        help="Stage2 loop summary CSV",
    )
    parser.add_argument(
        "--xmidi-labels",
        type=Path,
        help="XMIDI emotion/genre labels CSV",
    )
    parser.add_argument(
        "--captions",
        type=Path,
        help="MetaScore captions JSONL",
    )
    parser.add_argument(
        "--technique-meta",
        type=Path,
        help="VPTT technique metadata JSONL",
    )
    parser.add_argument(
        "--audio-cache",
        type=Path,
        help="Audio embedding cache directory",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output parquet file",
    )
    parser.add_argument(
        "--stats-output",
        type=Path,
        help="Output JSON file for statistics",
    )

    args = parser.parse_args()

    # Load base data
    df = load_stage2_summary(args.stage2_summary)

    # Merge all conditions
    df = merge_xmidi_labels(df, args.xmidi_labels)
    df = merge_captions(df, args.captions)
    df = merge_techniques(df, args.technique_meta)
    df = merge_audio_embeddings(df, args.audio_cache)

    # Validate
    stats = validate_output(df)
    logging.info("Validation stats: %s", json.dumps(stats, indent=2, ensure_ascii=False))

    # Save output
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.output, index=False)
    logging.info("Saved %d rows to %s", len(df), args.output)

    # Save stats if requested
    if args.stats_output:
        args.stats_output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.stats_output, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        logging.info("Saved statistics to %s", args.stats_output)


if __name__ == "__main__":
    main()
