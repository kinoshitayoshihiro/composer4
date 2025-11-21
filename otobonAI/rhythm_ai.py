#!/usr/bin/env python3
"""RhythmAI — lightweight groove recommender for Drumify.

The class in this module consumes ``data/groove_vocab.parquet`` (emitted by
``scripts/extract_groove_vocab.py``) and exposes a simple API that drum plan
scripts can call per bar/section. When the parquet file is absent the class
falls back to deterministic heuristics so the pipeline keeps running.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import pandas as pd
import yaml

LOGGER = logging.getLogger(__name__)
DEFAULT_VOCAB = Path("data/groove_vocab.parquet")
DEFAULT_RHYTHM_MANIFEST = Path("data/rhythm_vocab.yaml")


@dataclass(slots=True)
class RhythmContext:
    """Minimal context required to request a groove candidate."""

    section_label: str
    tempo_bpm: float
    energy: float
    drum_label: str | None = None
    emotion: str | None = None
    style_hint: str | None = None
    fill_slot: bool = False
    riff_slot: bool = False
    vocal_voiced_ratio: float = 0.0
    vocal_profile: str | None = None
    vocal_profile_confidence: float = 0.0


@dataclass(slots=True)
class RhythmCandidate:
    """Rhythm pattern recommendation returned by RhythmAI."""

    pattern_family: str
    swing_class: str
    density_bucket: str
    confidence: float
    source: str
    metadata: dict[str, Any]


@dataclass(slots=True)
class RhythmVocabEntry:
    """Structured entry from rhythm_vocab.yaml manifest."""

    id: str
    instrument: str
    source: str
    pattern_ref: str
    density: str | None
    energy: str | None
    grid: float | None
    sections: tuple[str, ...]
    descriptors: tuple[str, ...]
    ai_hooks: dict[str, Any]


class RhythmAI:
    """Recommend groove templates from a precomputed vocabulary."""

    def __init__(
        self,
        vocab_path: Path | None = None,
        rhythm_manifest_path: Path | None = None,
        *,
        enable_logging: bool = True,
    ) -> None:
        self.vocab_path = (vocab_path or DEFAULT_VOCAB).expanduser().resolve()
        self.rhythm_manifest_path = (
            (rhythm_manifest_path or DEFAULT_RHYTHM_MANIFEST).expanduser().resolve()
        )
        self._df: pd.DataFrame | None = None
        self._manifest: dict[str, list[RhythmVocabEntry]] = {}
        self._load_vocab()
        self._load_manifest()
        if enable_logging:
            LOGGER.info(
                "RhythmAI initialized (vocab=%s, rows=%s)",
                self.vocab_path if self._df is not None else "<memory>",
                len(self._df) if self._df is not None else 0,
            )

    # ------------------------------------------------------------------
    def _load_vocab(self) -> None:
        if not self.vocab_path.exists():
            LOGGER.warning(
                "Groove vocab not found at %s — falling back to heuristics", self.vocab_path
            )
            return
        try:
            self._df = pd.read_parquet(self.vocab_path)
        except Exception as exc:  # pragma: no cover - best effort loader
            LOGGER.error("Failed to load groove vocab: %s", exc)
            self._df = None

    # ------------------------------------------------------------------
    def _load_manifest(self) -> None:
        if not self.rhythm_manifest_path.exists():
            LOGGER.debug("Rhythm vocab manifest not found at %s", self.rhythm_manifest_path)
            return
        try:
            with self.rhythm_manifest_path.open("r", encoding="utf-8") as handle:
                payload = cast(dict[str, Any], yaml.safe_load(handle) or {})
        except Exception as exc:  # pragma: no cover - best effort loader
            LOGGER.error("Failed to load rhythm vocab manifest: %s", exc)
            return

        vocab = cast(dict[str, Any], payload.get("vocabulary", {}) or {})
        manifest: dict[str, list[RhythmVocabEntry]] = {}
        for instrument, entries in vocab.items():
            normalized_instrument = instrument.lower()
            manifest[normalized_instrument] = []
            for raw in entries or []:
                try:
                    entry = RhythmVocabEntry(
                        id=str(raw.get("id", "")),
                        instrument=str(raw.get("instrument", normalized_instrument)),
                        source=str(raw.get("source", "")),
                        pattern_ref=str(raw.get("pattern_ref", "")),
                        density=raw.get("density"),
                        energy=raw.get("energy"),
                        grid=float(raw.get("grid")) if raw.get("grid") is not None else None,
                        sections=tuple(raw.get("sections") or []),
                        descriptors=tuple(raw.get("descriptors") or []),
                        ai_hooks=dict(raw.get("ai_hooks") or {}),
                    )
                except Exception as exc:  # pragma: no cover - defensive parse
                    LOGGER.warning("Skipping invalid rhythm vocab entry (%s): %s", raw, exc)
                    continue
                if not entry.id:
                    continue
                manifest[normalized_instrument].append(entry)

        self._manifest = {k: v for k, v in manifest.items() if v}
        LOGGER.debug(
            "Rhythm vocab manifest loaded (%s instruments, file=%s)",
            len(self._manifest),
            self.rhythm_manifest_path,
        )

    # ------------------------------------------------------------------
    def is_ready(self) -> bool:
        return self._df is not None and not self._df.empty

    # ------------------------------------------------------------------
    def has_manifest(self) -> bool:
        return bool(self._manifest)

    # ------------------------------------------------------------------
    def recommend(self, context: RhythmContext, top_k: int = 3) -> list[RhythmCandidate]:
        """Return up to ``top_k`` groove candidates for the supplied context."""

        if self.is_ready():
            subset = self._filter_vocab(context)
            if subset.empty and self._df is not None:
                subset = self._df.copy()
            subset = subset.head(top_k)
            return [self._row_to_candidate(row) for _, row in subset.iterrows()]

        # Fall back to heuristics when no vocab is available.
        return [self._fallback_candidate(context)]

    # ------------------------------------------------------------------
    def choose_pattern(self, context: RhythmContext) -> RhythmCandidate:
        """Convenience wrapper that returns the highest ranked candidate."""

        candidates = self.recommend(context, top_k=1)
        return candidates[0]

    # ------------------------------------------------------------------
    def _filter_vocab(self, context: RhythmContext) -> pd.DataFrame:
        assert self._df is not None
        df = self._df
        subset = df.copy()

        def _matches(series: pd.Series, value: str | None) -> pd.Series:
            if not value:
                return pd.Series([True] * len(series))
            return series.fillna("").str.contains(value, case=False, na=False)

        filters: list[pd.Series] = []
        filters.append(_matches(subset["pattern_family"], infer_pattern_family(context)))
        filters.append(_matches(subset["swing_class"], infer_swing_class(context)))

        if context.drum_label:
            if "drum_label" in subset.columns:
                filters.append(_matches(subset["drum_label"], context.drum_label))
        if context.emotion and "emotion" in subset.columns:
            filters.append(_matches(subset["emotion"], context.emotion))

        mask = filters[0]
        for extra in filters[1:]:
            mask &= extra
        subset = subset.loc[mask].copy()

        if subset.empty:
            return df.sort_values(by="confidence", ascending=False)

        # Rank by proximity to tempo/energy/density targets.
        target_density = suggest_density_bucket(context)
        subset["tempo_delta"] = (subset["bpm"].fillna(context.tempo_bpm) - context.tempo_bpm).abs()
        subset["energy_delta"] = (
            subset["metrics.velocity_mean"].fillna(context.energy) - context.energy
        ).abs()
        density_rank = subset["density_bucket"].map(_density_rank)
        density_rank = density_rank.fillna(_density_rank(target_density))
        subset["density_rank"] = density_rank
        subset["density_delta"] = (density_rank - _density_rank(target_density)).abs()
        subset["score"] = (
            subset["tempo_delta"] * 0.5
            + subset["energy_delta"] * 0.3
            + subset["density_delta"] * 0.7
        )
        return subset.sort_values(by=["score", "confidence"], ascending=[True, False])

    # ------------------------------------------------------------------
    def manifest_entries(self, instrument: str) -> list[RhythmVocabEntry]:
        return list(self._manifest.get(instrument.lower(), ()))

    # ------------------------------------------------------------------
    def choose_vocab_entry(
        self,
        instrument: str,
        *,
        section_label: str | None = None,
        density_hint: str | None = None,
        descriptors: Sequence[str] | None = None,
        preferred_ids: Sequence[str] | None = None,
    ) -> RhythmVocabEntry | None:
        """Return the best matching manifest entry for the supplied instrument."""

        entries = self.manifest_entries(instrument)
        if not entries:
            return None

        section_label = (section_label or "").lower()
        density_hint = _coerce_density_label(density_hint)
        descriptor_set = {d.lower() for d in (descriptors or [])}
        preferred_ids = [pid for pid in (preferred_ids or []) if pid]

        ranked: list[tuple[float, RhythmVocabEntry]] = []
        for entry in entries:
            score = 0.0

            if preferred_ids:
                if entry.id in preferred_ids:
                    score -= 1.5
                elif any(entry.id.endswith(pref) for pref in preferred_ids):
                    score -= 0.5

            if section_label and entry.sections:
                sections = {s.lower() for s in entry.sections}
                if section_label in sections:
                    score -= 0.3
                else:
                    score += 0.4

            if density_hint and entry.density:
                entry_density = _coerce_density_label(entry.density)
                score += abs(_density_rank(entry_density) - _density_rank(density_hint)) * 0.2

            if descriptor_set and entry.descriptors:
                overlap = descriptor_set.intersection({d.lower() for d in entry.descriptors})
                score -= 0.1 * len(overlap)

            ranked.append((score, entry))

        ranked.sort(key=lambda item: (item[0], item[1].id))
        return ranked[0][1] if ranked else None

    # ------------------------------------------------------------------
    @staticmethod
    def _row_to_candidate(row: pd.Series) -> RhythmCandidate:
        return RhythmCandidate(
            pattern_family=str(row.get("pattern_family", "backbeat")),
            swing_class=str(row.get("swing_class", "straight")),
            density_bucket=str(row.get("density_bucket", "medium")),
            confidence=float(row.get("confidence", 0.5) or 0.0),
            source=str(row.get("groove_id", row.get("loop_id", "unknown"))),
            metadata={
                "bpm": row.get("bpm"),
                "energy_tag": row.get("energy_tag"),
                "section_hint": row.get("section_hint"),
                "note_density_per_bar": row.get("metrics.note_density_per_bar"),
                "swing_ratio": row.get("metrics.swing_ratio"),
                "syncopation_rate": row.get("metrics.syncopation_rate"),
            },
        )

    # ------------------------------------------------------------------
    @staticmethod
    def _fallback_candidate(context: RhythmContext) -> RhythmCandidate:
        pattern = infer_pattern_family(context)
        swing = infer_swing_class(context)
        density = suggest_density_bucket(context)
        confidence = 0.35 if context.fill_slot else 0.5
        return RhythmCandidate(
            pattern_family=pattern,
            swing_class=swing,
            density_bucket=density,
            confidence=confidence,
            source="heuristic",
            metadata={
                "tempo_bpm": context.tempo_bpm,
                "energy": context.energy,
                "drum_label": context.drum_label,
                "vocal_voiced_ratio": context.vocal_voiced_ratio,
                "vocal_profile": context.vocal_profile,
                "vocal_profile_confidence": context.vocal_profile_confidence,
            },
        )


# ---------------------------------------------------------------------------
# Lightweight heuristics used by both the extractor and the fallback path
# ---------------------------------------------------------------------------


def infer_pattern_family(context: RhythmContext) -> str:
    section = context.section_label.lower()
    label = (context.drum_label or "").lower()
    tokens = _style_tokens(context.style_hint)
    if "fill" in tokens:
        return "fill"
    if "groove" in tokens:
        return "backbeat"
    if "sync" in tokens and context.tempo_bpm >= 110:
        return "four_on_floor"
    if "counter" in tokens and context.tempo_bpm < 100:
        return "half_time"
    style_hint = (context.style_hint or "").lower()
    if style_hint in {"four_on_floor", "double_time", "half_time", "shuffle", "swing"}:
        return style_hint
    if context.fill_slot:
        return "fill"
    if "shuffle" in label or "swing" in label:
        return "shuffle"
    if section in {"bridge", "breakdown"}:
        return "half_time"
    if section in {"chorus", "drop"}:
        return "four_on_floor"
    if context.tempo_bpm >= 140:
        return "double_time"
    return "backbeat"


def infer_swing_class(context: RhythmContext) -> str:
    label = (context.drum_label or "").lower()
    if "shuffle" in label or "swing" in label:
        return "swing"
    if "latin" in label:
        return "straight"
    if context.section_label.lower() in {"bridge", "solo"} and context.tempo_bpm < 90:
        return "shuffle"
    return "straight"


def infer_density_bucket(energy: float) -> str:
    if energy < 40:
        return "sparse"
    if energy < 70:
        return "medium"
    if energy < 95:
        return "dense"
    return "wall"


def _coerce_density_label(bucket: str | None) -> str:
    if not bucket:
        return "medium"
    label = bucket.lower()
    mapping = {
        "low": "sparse",
        "lo": "sparse",
        "mid": "medium",
        "mid_low": "medium",
        "mid_high": "dense",
        "high": "dense",
        "hi": "dense",
        "wall": "wall",
    }
    return mapping.get(label, label)


def _density_rank(bucket: str | None) -> float:
    order = {"sparse": 0, "medium": 1, "dense": 2, "wall": 3}
    label = _coerce_density_label(bucket)
    return order.get(label, 1.5)


def _shift_density_bucket(bucket: str, steps: int) -> str:
    order = ["sparse", "medium", "dense", "wall"]
    try:
        idx = order.index((bucket or "medium").lower())
    except ValueError:
        idx = 1
    idx = max(0, min(len(order) - 1, idx + steps))
    return order[idx]


def suggest_density_bucket(context: RhythmContext) -> str:
    ratio_raw = max(0.0, min(float(context.vocal_voiced_ratio or 0.0), 1.2))
    confidence = max(0.0, min(float(context.vocal_profile_confidence or 0.0), 1.0))
    neutral_ratio = 0.35
    effective_ratio = ratio_raw * confidence + neutral_ratio * (1.0 - confidence)
    effective_ratio = max(0.0, min(effective_ratio, 1.0))

    tokens = _style_tokens(context.style_hint)
    tokens |= _style_tokens(getattr(context, "vocal_profile", None))

    base = infer_density_bucket(context.energy)
    if "sync" in tokens:
        base = "dense" if context.energy >= 55 else "medium"
    elif "counter" in tokens:
        base = "medium" if context.energy >= 65 else "sparse"
    elif "groove" in tokens:
        if context.energy >= 70:
            base = "dense"
        elif context.energy >= 50:
            base = "medium"
        else:
            base = "sparse"
    elif "fill" in tokens or context.fill_slot:
        base = "dense" if context.energy >= 55 else "medium"

    ratio_shift = 0
    if effective_ratio <= 0.05:
        ratio_shift = 2
    elif effective_ratio <= 0.15:
        ratio_shift = 1
    elif effective_ratio >= 0.8:
        ratio_shift = -2
    elif effective_ratio >= 0.6:
        ratio_shift = -1

    if ratio_shift:
        max_shift = ratio_shift if ratio_shift > 0 else (-1 if context.fill_slot else ratio_shift)
        base = _shift_density_bucket(base, max_shift)

    if effective_ratio >= 0.65 and base == "wall":
        base = "dense"
    if effective_ratio <= 0.25 and base == "sparse" and context.energy >= 45:
        base = "medium"

    return base


def _style_tokens(hint: str | None) -> set[str]:
    if not hint:
        return set()
    cleaned = hint.replace("|", "_").replace("+", "_").replace("-", "_").replace(" ", "_").lower()
    return {token for token in cleaned.split("_") if token}


__all__ = [
    "RhythmAI",
    "RhythmCandidate",
    "RhythmContext",
    "RhythmVocabEntry",
]
