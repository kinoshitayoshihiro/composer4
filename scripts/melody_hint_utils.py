#!/usr/bin/env python3
"""Shared helpers for CREPE-driven melody hints and instrument filtering."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

DEFAULT_SHORT_RATIO = 0.35
DEFAULT_LONG_RATIO = 0.65
DEFAULT_SLIDE_THRESHOLD = 1.2  # 120 cents per-beat slide activity
BEATS_PER_BAR = 4.0
MIN_HZ = 1e-3


@dataclass(frozen=True)
class MelodyHint:
    """Single-bar metadata derived from CREPE vocal analysis."""

    bar_idx: int
    section_label: str
    voiced_ratio: float
    duration_beats: float
    slide_activity: float
    tag: str
    source: str = "crepe"

    def as_dict(self) -> Dict[str, Any]:
        return {
            "bar_index": self.bar_idx,
            "section_label": self.section_label,
            "voiced_ratio": round(self.voiced_ratio, 3),
            "duration_beats": round(self.duration_beats, 2),
            "slide_activity": round(self.slide_activity, 3),
            "tag": self.tag,
            "source": self.source,
        }


def _resolve_bar_index(row: pd.Series) -> int:
    if "bar_idx" in row:
        return int(row["bar_idx"])
    if "bar_index" in row:
        return int(row["bar_index"])
    return int(row.get("bar", 0))


def _normalize_section(value: Any) -> str:
    if isinstance(value, str) and value.strip():
        return value.strip().lower()
    return "unknown"


def _first_non_null(series: pd.Series) -> Optional[float]:
    if series is None:
        return None
    valid = series.dropna()
    if valid.empty:
        return None
    return float(valid.iloc[0])


def _estimate_slide_activity(group: pd.DataFrame) -> float:
    if "slide_activity" in group.columns:
        preset = _first_non_null(group["slide_activity"])
        if preset is not None:
            return float(preset)

    if "f0_hz" not in group.columns:
        return 0.0

    ordered = group
    if "time_sec" in group.columns:
        ordered = group.sort_values("time_sec")
    elif "time" in group.columns:
        ordered = group.sort_values("time")

    series = ordered["f0_hz"].astype(float).replace([np.inf, -np.inf], np.nan).dropna()
    if series.empty:
        return 0.0

    if "voiced" in ordered.columns:
        voiced_mask = ordered["voiced"].astype(bool).reset_index(drop=True)
        series = series.reset_index(drop=True)
        series = series[voiced_mask[: len(series)]]
        if series.empty:
            return 0.0

    hz = np.clip(series.to_numpy(dtype=float), MIN_HZ, None)
    if hz.size < 2:
        return 0.0

    cents = np.abs(np.diff(np.log2(hz))) * 1200.0
    if cents.size == 0:
        return 0.0

    return float(np.mean(cents) / 100.0)


def _aggregate_vocal_metrics(f0: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, float]] = []
    for bar_idx, group in f0.groupby("bar_idx"):
        ratio = None
        if "f0_voiced_ratio" in group.columns:
            ratio = _first_non_null(group["f0_voiced_ratio"])
        if ratio is None and "voiced_ratio" in group.columns:
            ratio = _first_non_null(group["voiced_ratio"])
        if ratio is None and "voiced" in group.columns:
            try:
                ratio = float(group["voiced"].astype(float).mean())
            except Exception:
                ratio = None
        if ratio is None:
            ratio = 0.0

        slide = _estimate_slide_activity(group)

        rows.append(
            {"bar_idx": int(bar_idx), "f0_voiced_ratio": float(ratio), "slide_activity": slide}
        )

    return pd.DataFrame(rows)


def build_melody_hint_table(
    bars_df: pd.DataFrame,
    vocal_f0_df: Optional[pd.DataFrame],
    short_ratio: float = DEFAULT_SHORT_RATIO,
    long_ratio: float = DEFAULT_LONG_RATIO,
    slide_threshold: float = DEFAULT_SLIDE_THRESHOLD,
) -> Dict[int, MelodyHint]:
    """Create bar-indexed melody hints from bars + CREPE statistics."""

    if bars_df is None or vocal_f0_df is None or vocal_f0_df.empty:
        return {}

    bars = bars_df.copy()
    if "bar_idx" not in bars.columns:
        if "bar_index" in bars.columns:
            bars = bars.rename(columns={"bar_index": "bar_idx"})
        else:
            bars["bar_idx"] = np.arange(len(bars))

    f0 = vocal_f0_df.copy()
    if "bar_idx" not in f0.columns:
        if "bar_index" in f0.columns:
            f0 = f0.rename(columns={"bar_index": "bar_idx"})
        else:
            f0["bar_idx"] = np.arange(len(f0))

    bar_metrics = _aggregate_vocal_metrics(f0)
    merged = bars.merge(bar_metrics, on="bar_idx", how="left")

    hints: Dict[int, MelodyHint] = {}
    for _, row in merged.iterrows():
        bar_idx = int(row["bar_idx"])
        ratio = float(row.get("f0_voiced_ratio", 0.0) or 0.0)
        slide = float(row.get("slide_activity", 0.0) or 0.0)
        duration_beats = ratio * BEATS_PER_BAR
        section_label = _normalize_section(row.get("section_label", ""))

        tag = ""
        if ratio >= long_ratio:
            tag = "melody_hint_long"
        elif ratio >= short_ratio:
            tag = "melody_hint_phrase"
        elif slide >= slide_threshold:
            tag = "melody_hint_gliss"

        if not tag:
            continue

        hints[bar_idx] = MelodyHint(
            bar_idx=bar_idx,
            section_label=section_label,
            voiced_ratio=ratio,
            duration_beats=duration_beats,
            slide_activity=slide,
            tag=tag,
        )

    return hints


def summarize_melody_hints(hints: Dict[int, MelodyHint]) -> Dict[str, Dict[str, Any]]:
    """Aggregate counts per section and tag for logging or manifest export."""

    summary: Dict[str, Dict[str, Any]] = {}
    for hint in hints.values():
        section = hint.section_label
        section_stats = summary.setdefault(
            section,
            {
                "bars": 0,
                "long": 0,
                "phrase": 0,
                "gliss": 0,
                "avg_duration_beats": 0.0,
            },
        )
        section_stats["bars"] += 1
        section_stats["avg_duration_beats"] += hint.duration_beats
        if hint.tag == "melody_hint_long":
            section_stats["long"] += 1
        elif hint.tag == "melody_hint_phrase":
            section_stats["phrase"] += 1
        elif hint.tag == "melody_hint_gliss":
            section_stats["gliss"] += 1

    for stats in summary.values():
        bars = max(1, stats["bars"])
        stats["avg_duration_beats"] = round(stats["avg_duration_beats"] / bars, 2)

    return summary


def apply_melody_hint_filter(
    events: List[Dict[str, Any]],
    hints: Dict[int, MelodyHint],
    *,
    instrument: str,
    drop_tags: Iterable[str] = ("melody_hint_long",),
    drop_threshold_beats: float = 2.0,
    annotate: bool = True,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """Annotate events with CREPE hints and optionally drop conflicts."""

    if not events or not hints:
        return events, {"annotated": 0, "removed": 0}

    drop_tags_set = {tag for tag in drop_tags if tag}
    kept: List[Dict[str, Any]] = []
    annotated = 0
    removed = 0

    for ev in events:
        bar_idx = ev.get("bar_idx") or ev.get("bar") or ev.get("measure")
        bar_idx = int(bar_idx) if bar_idx is not None else None
        hint = hints.get(bar_idx) if bar_idx is not None else None

        if hint and annotate:
            tags = list(ev.get("tags", []))
            tags.extend([hint.tag, "crepe_hint"])
            ev["tags"] = sorted({t for t in tags if t})
            ev["melody_hint"] = hint.as_dict()
            annotated += 1

        should_drop = False
        if hint and drop_tags_set:
            if hint.tag in drop_tags_set and hint.duration_beats >= drop_threshold_beats:
                should_drop = instrument.lower().startswith("strings")

        if should_drop:
            removed += 1
            continue

        kept.append(ev)

    return kept, {"annotated": annotated, "removed": removed}


def build_melody_hint_manifest_payload(
    hints: Dict[int, MelodyHint],
    *,
    bars_total: int,
    song_id: Optional[str],
    bars_path: Path,
    vocal_f0_path: Optional[Path],
    out_path: Path,
) -> Dict[str, Any]:
    """Assemble a serializable payload summarizing melody hints."""

    summary = summarize_melody_hints(hints)
    manifest = {
        "metadata": {
            "song_id": song_id,
            "bars_total": int(bars_total),
            "hints_total": len(hints),
            "generator": "melody_hint_utils.build_melody_hint_manifest_payload",
            "inputs": {
                "bars": str(bars_path),
                "vocal_f0": str(vocal_f0_path) if vocal_f0_path else None,
            },
        },
        "schema": {
            "hint_fields": [
                "bar_index",
                "section_label",
                "voiced_ratio",
                "duration_beats",
                "slide_activity",
                "tag",
                "source",
            ],
            "paths": {
                "melody_manifest": str(out_path),
                "vocal_f0_parquet": str(vocal_f0_path) if vocal_f0_path else None,
            },
        },
        "hints": [hint.as_dict() for hint in hints.values()],
        "sections": summary,
    }

    return manifest
