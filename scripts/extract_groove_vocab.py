#!/usr/bin/env python3
"""Groove vocabulary extractor for RhythmAI / Drumify.

This scaffolding script ingests Stage2 loop summaries (e.g.
``outputs/stage2_drums_iter8_100PCT/loop_summary.csv``) and emits a compact
parquet/JSON pair describing groove traits per loop. The resulting
``data/groove_vocab.parquet`` can be loaded by RhythmAI to recommend patterns
per section/emotion without parsing the heavyweight Stage2 exports each time.

Typical usage::

    PYTHONPATH=. .venv311/bin/python scripts/extract_groove_vocab.py \
        --stage2-dir outputs/stage2_drums_iter8_100PCT \
        --output-parquet data/groove_vocab.parquet \
        --output-stats data/groove_vocab_stats.json

The script is intentionally lightweight: it only depends on pandas/numpy and is
safe to run as soon as Stage2 completes.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

LOGGER = logging.getLogger("groove_vocab")
DEFAULT_STAGE2 = Path("outputs/stage2_drums_iter8_100PCT")
DEFAULT_VOCAB = Path("data/groove_vocab.parquet")
DEFAULT_STATS = Path("data/groove_vocab_stats.json")


@dataclass(slots=True)
class GrooveExtractionConfig:
    stage2_dir: Path
    loop_summary: Optional[Path]
    labels_csv: Optional[Path]
    output_parquet: Path
    output_stats: Path
    min_score: float
    max_rows: Optional[int]


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------


def parse_args() -> GrooveExtractionConfig:
    parser = argparse.ArgumentParser(description="Extract groove vocabulary metadata")
    parser.add_argument(
        "--stage2-dir",
        type=Path,
        default=DEFAULT_STAGE2,
        help="Stage2 output directory containing loop_summary.csv",
    )
    parser.add_argument(
        "--loop-summary",
        type=Path,
        help="Explicit path to loop_summary.csv (overrides --stage2-dir)",
    )
    parser.add_argument(
        "--labels-csv",
        type=Path,
        help="Optional CSV mapping loop_id -> drum_label/emotion (e.g. XMIDI labels)",
    )
    parser.add_argument(
        "--output-parquet",
        type=Path,
        default=DEFAULT_VOCAB,
        help="Destination parquet file for RhythmAI consumption",
    )
    parser.add_argument(
        "--output-stats",
        type=Path,
        default=DEFAULT_STATS,
        help="Destination JSON summary (counts, histograms, etc.)",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=55.0,
        help="Minimum Stage2 score.total required to keep a groove",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        help="Optional cap for smoke tests / sampling",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Python logging level (DEBUG, INFO, WARNING, ...)",
    )

    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    stage2_dir = args.stage2_dir.expanduser().resolve()
    loop_summary = args.loop_summary.expanduser().resolve() if args.loop_summary else None
    labels_csv = args.labels_csv.expanduser().resolve() if args.labels_csv else None
    output_parquet = args.output_parquet.expanduser().resolve()
    output_stats = args.output_stats.expanduser().resolve()

    return GrooveExtractionConfig(
        stage2_dir=stage2_dir,
        loop_summary=loop_summary,
        labels_csv=labels_csv,
        output_parquet=output_parquet,
        output_stats=output_stats,
        min_score=args.min_score,
        max_rows=args.max_rows,
    )


# ---------------------------------------------------------------------------
# Core extraction logic
# ---------------------------------------------------------------------------


def load_loop_summary(cfg: GrooveExtractionConfig) -> pd.DataFrame:
    csv_path = cfg.loop_summary or (cfg.stage2_dir / "loop_summary.csv")
    if not csv_path.exists():
        raise FileNotFoundError(f"loop_summary.csv not found at {csv_path}")

    LOGGER.info("Loading Stage2 loop summary: %s", csv_path)
    df = pd.read_csv(csv_path)

    if cfg.max_rows:
        df = df.head(cfg.max_rows)
        LOGGER.info("Truncated loop summary to %d rows (max_rows)", len(df))

    # Ensure expected baseline columns.
    ensure_column(df, "score.total", np.nan)
    ensure_column(df, "genre", "unknown")
    ensure_column(df, "metrics.note_density_per_bar", np.nan)
    ensure_column(df, "metrics.swing_ratio", np.nan)
    ensure_column(df, "metrics.syncopation_rate", np.nan)
    ensure_column(df, "metrics.fill_density", np.nan)
    ensure_column(df, "metrics.layering_rate", np.nan)
    ensure_column(df, "metrics.velocity_mean", np.nan)
    ensure_column(df, "metrics.velocity_std", np.nan)
    ensure_column(df, "metrics.hat_open_ratio", np.nan)
    ensure_column(df, "metrics.rhythm_hash", "")
    ensure_column(df, "metrics.rhythm_fingerprint", "")

    # Filter by Stage2 score threshold if available.
    mask = df["score.total"].fillna(0) >= cfg.min_score
    before = len(df)
    df = df.loc[mask].copy()
    LOGGER.info(
        "Filtered grooves by score.total ≥ %.1f (%d → %d rows)", cfg.min_score, before, len(df)
    )

    return df


def ensure_column(df: pd.DataFrame, name: str, default: Any) -> None:
    if name not in df.columns:
        df[name] = default


def load_labels(labels_csv: Optional[Path]) -> Optional[pd.DataFrame]:
    if labels_csv is None:
        return None
    if not labels_csv.exists():
        raise FileNotFoundError(f"labels CSV not found: {labels_csv}")
    LOGGER.info("Loading label map: %s", labels_csv)
    labels = pd.read_csv(labels_csv)
    if "loop_id" not in labels.columns:
        raise ValueError("labels CSV must include a 'loop_id' column")
    return labels


def enrich_grooves(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["groove_id"] = "groove::" + df["loop_id"].astype(str)
    df["swing_class"] = df["metrics.swing_ratio"].apply(classify_swing)
    df["density_bucket"] = df["metrics.note_density_per_bar"].apply(classify_density)
    df["pattern_family"] = df.apply(classify_pattern_family, axis=1)
    df["energy_tag"] = df["metrics.velocity_mean"].apply(classify_energy)
    df["confidence"] = df["score.total"].apply(lambda x: float(x) / 100.0 if pd.notna(x) else 0.0)
    df["section_hint"] = df["metrics.fill_density"].apply(classify_section_hint)
    df["velocity_iqr"] = df["metrics.velocity_std"].fillna(0) * 1.349  # approx IQR from std
    df["swing_ratio"] = df["metrics.swing_ratio"].fillna(np.nan)
    df["syncopation_rate"] = df["metrics.syncopation_rate"].fillna(np.nan)

    keep_cols = [
        "groove_id",
        "loop_id",
        "source",
        "genre",
        "bpm",
        "duration_ticks",
        "bar_count",
        "pattern_family",
        "swing_class",
        "density_bucket",
        "energy_tag",
        "section_hint",
        "metrics.note_density_per_bar",
        "metrics.swing_ratio",
        "metrics.syncopation_rate",
        "metrics.fill_density",
        "metrics.layering_rate",
        "metrics.velocity_mean",
        "metrics.velocity_std",
        "velocity_iqr",
        "metrics.hat_open_ratio",
        "metrics.rhythm_hash",
        "metrics.rhythm_fingerprint",
        "score.total",
        "confidence",
    ]

    missing = [col for col in keep_cols if col not in df.columns]
    for col in missing:
        df[col] = np.nan

    return df[keep_cols].copy()


def classify_swing(value: Any) -> str:
    try:
        val = float(value)
    except (TypeError, ValueError):
        return "unknown"
    if math.isnan(val):
        return "unknown"
    if val < 1.05:
        return "straight"
    if val < 1.25:
        return "shuffle"
    return "swing"


def classify_density(value: Any) -> str:
    try:
        val = float(value)
    except (TypeError, ValueError):
        return "unknown"
    if math.isnan(val):
        return "unknown"
    if val < 16:
        return "sparse"
    if val < 48:
        return "medium"
    if val < 80:
        return "dense"
    return "wall"


def classify_energy(value: Any) -> str:
    try:
        val = float(value)
    except (TypeError, ValueError):
        return "unknown"
    if math.isnan(val):
        return "unknown"
    if val < 55:
        return "low"
    if val < 85:
        return "mid"
    return "high"


def classify_section_hint(fill_density: Any) -> str:
    try:
        val = float(fill_density)
    except (TypeError, ValueError):
        return "generic"
    if math.isnan(val):
        return "generic"
    if val >= 0.5:
        return "fill"
    if val >= 0.25:
        return "pre_chorus"
    return "groove"


def classify_pattern_family(row: pd.Series) -> str:
    swing = row.get("metrics.swing_ratio", np.nan)
    sync = row.get("metrics.syncopation_rate", np.nan)
    genre = str(row.get("genre", ""))
    hat_open = row.get("metrics.hat_open_ratio", 0.0)
    density = row.get("metrics.note_density_per_bar", np.nan)

    if pd.notna(swing) and swing >= 1.25:
        return "swing"
    if pd.notna(swing) and 1.05 <= swing < 1.25:
        return "shuffle"
    if "disco" in genre or "house" in genre or hat_open > 0.25:
        return "four_on_floor"
    if pd.notna(sync) and sync >= 0.6:
        return "syncopated"
    if pd.notna(density) and density >= 80:
        return "blast"
    if pd.notna(density) and density <= 16:
        return "minimal"
    return "backbeat"


def merge_labels(grooves: pd.DataFrame, labels: Optional[pd.DataFrame]) -> pd.DataFrame:
    if labels is None:
        return grooves
    cols = [col for col in labels.columns if col != "loop_id"]
    merged = grooves.merge(labels, on="loop_id", how="left", suffixes=("", "_label"))
    LOGGER.info("Merged label columns: %s", cols)
    return merged


def save_outputs(df: pd.DataFrame, cfg: GrooveExtractionConfig) -> Dict[str, Any]:
    cfg.output_parquet.parent.mkdir(parents=True, exist_ok=True)
    cfg.output_stats.parent.mkdir(parents=True, exist_ok=True)

    df.to_parquet(cfg.output_parquet, index=False)
    LOGGER.info("Wrote groove vocab parquet → %s (%d rows)", cfg.output_parquet, len(df))

    stats = {
        "rows": int(len(df)),
        "source": str(cfg.loop_summary or (cfg.stage2_dir / "loop_summary.csv")),
        "min_score": cfg.min_score,
        "density_counts": df["density_bucket"].value_counts(dropna=False).to_dict(),
        "pattern_counts": df["pattern_family"].value_counts(dropna=False).to_dict(),
        "swing_counts": df["swing_class"].value_counts(dropna=False).to_dict(),
    }
    cfg.output_stats.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    LOGGER.info("Wrote groove stats JSON → %s", cfg.output_stats)
    return stats


def main() -> None:
    cfg = parse_args()
    df_loops = load_loop_summary(cfg)
    labels = load_labels(cfg.labels_csv)
    grooves = enrich_grooves(df_loops)
    grooves = merge_labels(grooves, labels)
    save_outputs(grooves, cfg)


if __name__ == "__main__":
    main()
