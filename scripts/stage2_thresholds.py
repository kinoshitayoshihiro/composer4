#!/usr/bin/env python3
"""Utilities for Stage2 tempo binning and threshold band application."""
from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, cast


@dataclass(frozen=True)
class Band:
    """Represents a low/high band for an axis within a tempo bucket."""

    low: Optional[float]
    high: Optional[float]
    count: Optional[int] = None
    stats: Optional[Sequence[float]] = None


class TempoBinner:
    """Assigns a tempo (BPM) to an index based on ordered edges."""

    def __init__(self, edges: Sequence[float]) -> None:
        if len(edges) < 2:
            raise ValueError(
                "Tempo bin edges must contain at least two values",
            )
        ordered = list(edges)
        if ordered != sorted(ordered):
            raise ValueError(
                "Tempo bin edges must be sorted in ascending order",
            )
        self._edges = ordered

    @property
    def edges(self) -> List[float]:
        return list(self._edges)

    def __len__(self) -> int:
        return max(0, len(self._edges) - 1)

    def index(self, tempo: float) -> int:
        if tempo != tempo:  # NaN check
            return 0
        insertion = bisect_left(self._edges, tempo)
        if insertion <= 0:
            return 0
        if insertion >= len(self._edges):
            return len(self._edges) - 2
        return insertion - 1

    def bin_pair(self, tempo: float) -> Tuple[float, float]:
        idx = self.index(tempo)
        return self._edges[idx], self._edges[idx + 1]


class ThresholdsProvider:
    """Returns precomputed bands for each axis given a tempo."""

    def __init__(
        self,
        *,
        bins: TempoBinner,
        per_axis: Dict[str, List[Band]],
    ) -> None:
        self._bins = bins
        self._per_axis = per_axis

    @property
    def bins(self) -> TempoBinner:
        return self._bins

    def axes(self) -> Iterable[str]:
        return self._per_axis.keys()

    def band_for(self, axis: str, tempo: float) -> Band:
        bins = self._per_axis.get(axis)
        if not bins:
            return Band(low=None, high=None)
        idx = self._bins.index(tempo)
        if idx >= len(bins):
            idx = len(bins) - 1
        return bins[idx]

    def snapshot(self) -> Dict[str, List[Band]]:
        return self._per_axis


def _parse_band(entry: Dict[str, Any]) -> Band:
    low = entry.get("low")
    high = entry.get("high")
    count = entry.get("count")
    stats = entry.get("q1q2q3I")
    stats_seq: Optional[List[float]] = None
    if isinstance(stats, (list, tuple)):
        stats_seq = []
        for value in cast(Sequence[Any], stats):
            if isinstance(value, (int, float)):
                stats_seq.append(float(value))
    return Band(
        low=float(low) if isinstance(low, (int, float)) else None,
        high=float(high) if isinstance(high, (int, float)) else None,
        count=int(count) if isinstance(count, int) else None,
        stats=stats_seq,
    )


def build_thresholds_provider(
    document: Dict[str, object],
) -> ThresholdsProvider:
    bins_section = document.get("bins", {})
    tempo_edges: List[float] = []
    if isinstance(bins_section, dict):
        bins_mapping = cast(Dict[str, Any], bins_section)
        tempo_candidates_raw = bins_mapping.get("tempo", [])
        if isinstance(tempo_candidates_raw, Sequence):
            tempo_candidates = cast(Sequence[Any], tempo_candidates_raw)
            for raw_value in tempo_candidates:
                if isinstance(raw_value, (int, float)):
                    tempo_edges.append(float(raw_value))
    if not tempo_edges:
        raise ValueError("Threshold document must define bins.tempo")
    binner = TempoBinner(tempo_edges)

    per_axis_raw = document.get("per_axis", {})
    if not isinstance(per_axis_raw, dict) or not per_axis_raw:
        raise ValueError("Threshold document must include per_axis entries")

    per_axis_mapping = cast(Dict[str, Any], per_axis_raw)
    per_axis: Dict[str, List[Band]] = {}
    for axis_raw, payload in per_axis_mapping.items():
        if not axis_raw:
            continue
        axis = axis_raw
        bins_payload_raw: Sequence[Any] = []
        if isinstance(payload, dict):
            payload_map = cast(Dict[str, Any], payload)
            candidate_bins = payload_map.get("bins", [])
            if isinstance(candidate_bins, Sequence):
                bins_payload_raw = cast(Sequence[Any], candidate_bins)
        if not bins_payload_raw:
            continue
        bands: List[Band] = []
        for entry in bins_payload_raw:
            if isinstance(entry, dict):
                bands.append(_parse_band(cast(Dict[str, Any], entry)))
        if bands:
            # Normalize length to tempo bins - 1
            target_len = len(binner) or len(bands)
            if len(bands) < target_len:
                last_band = bands[-1]
                bands.extend([last_band] * (target_len - len(bands)))
            per_axis[axis] = bands[:target_len]

    if not per_axis:
        raise ValueError(
            "No valid per_axis bands found in thresholds document",
        )

    return ThresholdsProvider(bins=binner, per_axis=per_axis)


def score_axis(raw: float, band: Band) -> float:
    """Convert a raw metric in [0, 1] to a 0–100 score relative to a band."""
    raw = max(0.0, min(1.0, raw))
    if band.low is None or band.high is None:
        return raw * 100.0
    low = min(band.low, band.high)
    high = max(band.low, band.high)
    if raw < low:
        if low <= 0:
            return 0.0
        scale = (low - raw) / low
        return max(0.0, 100.0 * (1.0 - scale))
    if raw > high:
        if high >= 1.0:
            return 0.0
        scale = (raw - high) / (1.0 - high)
        return max(0.0, 100.0 * (1.0 - scale))
    return 100.0
