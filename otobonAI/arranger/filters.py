#!/usr/bin/env python3
"""Arranger filtering helpers for Stage3 emotion/genre metadata.

Provides query utilities for XMIDI labels and Stage3 conditions.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass
class EmotionRecord:
    """XMIDI emotion metadata for a single loop."""

    loop_id: str
    emotion: str
    genre: str
    valence: float
    arousal: float
    drum_label: str | None = None
    drum_traits: dict[str, Any] | None = None
    axis_bias: dict[str, float] | None = None


class EmotionCatalog:
    """Query interface for XMIDI emotion/genre metadata."""

    def __init__(self, records: dict[str, EmotionRecord]):
        self.records = records

    @classmethod
    def from_csv(cls, labels_csv: Path) -> EmotionCatalog:
        """Load from xmidi_labels.csv."""
        df = pd.read_csv(labels_csv)
        records = {}

        for row in df.itertuples(index=False):
            # Parse JSON columns if present
            drum_traits = None
            axis_bias = None
            if hasattr(row, "drum_traits_json") and pd.notna(row.drum_traits_json):
                try:
                    drum_traits = json.loads(row.drum_traits_json)
                except json.JSONDecodeError:
                    pass
            if hasattr(row, "axis_bias_json") and pd.notna(row.axis_bias_json):
                try:
                    axis_bias = json.loads(row.axis_bias_json)
                except json.JSONDecodeError:
                    pass

            records[row.loop_id] = EmotionRecord(
                loop_id=row.loop_id,
                emotion=row.emotion,
                genre=row.genre,
                valence=row.valence,
                arousal=row.arousal,
                drum_label=getattr(row, "drum_label", None),
                drum_traits=drum_traits,
                axis_bias=axis_bias,
            )

        return cls(records)

    def get(self, loop_id: str) -> EmotionRecord | None:
        """Get emotion record for a loop."""
        return self.records.get(loop_id)

    def filter(
        self,
        emotion: str | list[str] | None = None,
        genre: str | list[str] | None = None,
        drum_label: str | list[str] | None = None,
        valence_min: float | None = None,
        valence_max: float | None = None,
        arousal_min: float | None = None,
        arousal_max: float | None = None,
    ) -> list[EmotionRecord]:
        """Filter records by emotion/genre/valence/arousal."""
        results = []

        for record in self.records.values():
            # Emotion filter
            if emotion is not None:
                emotions = [emotion] if isinstance(emotion, str) else emotion
                if record.emotion not in emotions:
                    continue

            # Genre filter
            if genre is not None:
                genres = [genre] if isinstance(genre, str) else genre
                if record.genre not in genres:
                    continue

            # Drum label filter
            if drum_label is not None:
                labels = [drum_label] if isinstance(drum_label, str) else drum_label
                if record.drum_label not in labels:
                    continue

            # Valence bounds
            if valence_min is not None and record.valence < valence_min:
                continue
            if valence_max is not None and record.valence > valence_max:
                continue

            # Arousal bounds
            if arousal_min is not None and record.arousal < arousal_min:
                continue
            if arousal_max is not None and record.arousal > arousal_max:
                continue

            results.append(record)

        return results

    def get_context(self, loop_id: str) -> dict[str, Any]:
        """Get arranger context dict for a loop."""
        record = self.get(loop_id)
        if record is None:
            return {}

        context = {
            "emotion": record.emotion,
            "genre": record.genre,
            "valence": record.valence,
            "arousal": record.arousal,
        }

        if record.drum_label:
            context["drum_label"] = record.drum_label
        if record.drum_traits:
            context["drum_traits"] = record.drum_traits
        if record.axis_bias:
            context["axis_bias"] = record.axis_bias

        return context


def load_emotion_catalog(labels_csv: Path) -> EmotionCatalog:
    """Load emotion catalog from xmidi_labels.csv."""
    return EmotionCatalog.from_csv(labels_csv)


def apply_qa_filters(
    df: pd.DataFrame,
    emotion_catalog: EmotionCatalog,
    min_score: float = 70.0,
    exclude_retry: bool = True,
    allowed_drum_labels: list[str] | None = None,
) -> pd.DataFrame:
    """Apply QA filters to Stage2 loop_summary.

    Args:
        df: Stage2 loop_summary DataFrame
        emotion_catalog: Emotion metadata catalog
        min_score: Minimum score.total threshold
        exclude_retry: Exclude loops flagged for retry
        allowed_drum_labels: Optional whitelist of drum labels

    Returns:
        Filtered DataFrame
    """
    # Score filter
    if "score.total" in df.columns:
        df = df[df["score.total"] >= min_score]

    # Retry filter
    if exclude_retry and "retry.preset_id" in df.columns:
        df = df[df["retry.preset_id"].isna()]

    # Drum label filter
    if allowed_drum_labels is not None:

        def has_allowed_label(loop_id):
            record = emotion_catalog.get(loop_id)
            return record and record.drum_label in allowed_drum_labels

        if "loop_id" in df.columns:
            mask = df["loop_id"].apply(has_allowed_label)
            df = df[mask]

    return df


def main():
    """Example usage."""
    import sys

    if len(sys.argv) < 2:
        print("Usage: python filters.py <xmidi_labels.csv>")
        sys.exit(1)

    catalog = load_emotion_catalog(Path(sys.argv[1]))
    print(f"Loaded {len(catalog.records)} emotion records")

    # Example queries
    print("\nHigh-energy loops (arousal > 0.7):")
    high_energy = catalog.filter(arousal_min=0.7)
    print(f"  Found {len(high_energy)} loops")

    print("\nShuffle drum patterns:")
    shuffle = catalog.filter(drum_label="sparkle_shuffle")
    print(f"  Found {len(shuffle)} loops")

    print("\nAngry metal:")
    angry_metal = catalog.filter(emotion="angry", genre="metal")
    print(f"  Found {len(angry_metal)} loops")


if __name__ == "__main__":
    main()
