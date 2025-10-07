"""Metric extraction utilities for LAMDa drum datasets."""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class MetricConfig:
    """Configuration for loop metric extraction."""

    ghost_velocity_threshold: int = 45
    accent_velocity_threshold: int = 100
    microtiming_tolerance_ratio: float = 0.2
    swing_min_pairs: int = 2
    min_base_step: int = 1
    max_breakpoints: int = 3
    round_digits: int = 4


@dataclass
class LoopMetrics:
    """Container for per-loop metric results."""

    note_count: int
    duration_ticks: int
    base_step: Optional[float]
    swing_ratio: Optional[float]
    swing_confidence: float
    ghost_rate: float
    accent_rate: float
    microtiming_mean: Optional[float]
    microtiming_std: Optional[float]
    microtiming_rms: Optional[float]
    onbeat_velocity_avg: Optional[float]
    offbeat_velocity_avg: Optional[float]
    on_off_velocity_ratio: Optional[float]
    syncopation_rate: Optional[float]
    fill_density: Optional[float]
    note_density_per_bar: Optional[float]
    layering_rate: float
    velocity_mean: float
    velocity_median: float
    velocity_std: float
    velocity_range: int
    unique_velocity_steps: int
    instrument_distribution: Dict[str, int] = field(default_factory=dict)
    hat_open_ratio: Optional[float] = None
    hat_transition_rate: Optional[float] = None
    repeat_rate: Optional[float] = None
    variation_factor: Optional[float] = None
    breakpoint_count: int = 0
    breakpoints: Tuple[float, ...] = ()
    drum_collision_rate: Optional[float] = None
    role_separation: Optional[float] = None
    rhythm_fingerprint: Optional[Dict[str, float]] = None
    rhythm_hash: Optional[str] = None
    tempo_stability: Optional[float] = None

    def to_dict(self, *, digits: int = 4) -> Dict[str, object]:
        def _round(value: Optional[float]) -> Optional[float]:
            if value is None:
                return None
            return round(value, digits)

        return {
            "note_count": self.note_count,
            "duration_ticks": self.duration_ticks,
            "base_step": _round(self.base_step),
            "swing_ratio": _round(self.swing_ratio),
            "swing_confidence": _round(self.swing_confidence),
            "ghost_rate": _round(self.ghost_rate),
            "accent_rate": _round(self.accent_rate),
            "microtiming_mean": _round(self.microtiming_mean),
            "microtiming_std": _round(self.microtiming_std),
            "microtiming_rms": _round(self.microtiming_rms),
            "onbeat_velocity_avg": _round(self.onbeat_velocity_avg),
            "offbeat_velocity_avg": _round(self.offbeat_velocity_avg),
            "on_off_velocity_ratio": _round(self.on_off_velocity_ratio),
            "syncopation_rate": _round(self.syncopation_rate),
            "fill_density": _round(self.fill_density),
            "note_density_per_bar": _round(self.note_density_per_bar),
            "layering_rate": _round(self.layering_rate),
            "velocity_mean": _round(self.velocity_mean),
            "velocity_median": _round(self.velocity_median),
            "velocity_std": _round(self.velocity_std),
            "velocity_range": self.velocity_range,
            "unique_velocity_steps": self.unique_velocity_steps,
            "instrument_distribution": dict(self.instrument_distribution),
            "hat_open_ratio": _round(self.hat_open_ratio),
            "hat_transition_rate": _round(self.hat_transition_rate),
            "repeat_rate": _round(self.repeat_rate),
            "variation_factor": _round(self.variation_factor),
            "breakpoint_count": self.breakpoint_count,
            "breakpoints": [round(b, digits) for b in self.breakpoints],
            "drum_collision_rate": _round(self.drum_collision_rate),
            "role_separation": _round(self.role_separation),
            "rhythm_fingerprint": self.rhythm_fingerprint,
            "rhythm_hash": self.rhythm_hash,
            "tempo_stability": _round(self.tempo_stability),
        }

    @classmethod
    def numeric_fields(cls) -> Tuple[str, ...]:
        return (
            "note_count",
            "duration_ticks",
            "base_step",
            "swing_ratio",
            "swing_confidence",
            "ghost_rate",
            "accent_rate",
            "microtiming_mean",
            "microtiming_std",
            "microtiming_rms",
            "onbeat_velocity_avg",
            "offbeat_velocity_avg",
            "on_off_velocity_ratio",
            "syncopation_rate",
            "fill_density",
            "note_density_per_bar",
            "layering_rate",
            "velocity_mean",
            "velocity_median",
            "velocity_std",
            "velocity_range",
            "unique_velocity_steps",
            "hat_open_ratio",
            "hat_transition_rate",
            "repeat_rate",
            "variation_factor",
            "breakpoint_count",
            "drum_collision_rate",
            "role_separation",
            "tempo_stability",
        )


class MetricsAggregator:
    """Helper to consolidate metrics across loops."""

    def __init__(self) -> None:
        self._count = 0
        self._totals: Dict[str, float] = defaultdict(float)
        self._observations: Dict[str, int] = defaultdict(int)
        self._instrument_totals: Counter[str] = Counter()

    @property
    def count(self) -> int:
        return self._count

    def add(self, metrics: LoopMetrics) -> None:
        self._count += 1
        for field_name in LoopMetrics.numeric_fields():
            value = getattr(metrics, field_name)
            if value is None:
                continue
            self._totals[field_name] += float(value)
            self._observations[field_name] += 1
        self._instrument_totals.update(metrics.instrument_distribution)

    def summary(self, *, digits: int = 4) -> Dict[str, object]:
        averages = {
            field: round(
                self._totals[field] / self._observations[field],
                digits,
            )
            for field in self._totals
            if self._observations[field]
        }
        return {
            "count": self._count,
            "averages": averages,
            "instrument_distribution": dict(self._instrument_totals),
        }


DRUM_CATEGORIES = {
    "kick": {35, 36},
    "snare": {38, 40},
    "closed_hat": {42, 44},
    "open_hat": {46},
    "tom": {41, 43, 45, 47, 48, 50},
    "cymbal": {49, 51, 52, 53, 55, 57, 59},
    "perc": {37, 39, 54, 56, 58},
}


def _categorise_pitch(pitch: int) -> str:
    for name, pitches in DRUM_CATEGORIES.items():
        if pitch in pitches:
            return name
    return "other"


def _group_by_timestamp(starts: Sequence[int]) -> Dict[int, int]:
    counts: Dict[int, int] = defaultdict(int)
    for start in starts:
        counts[start] += 1
    return counts


def _compute_rhythm_fingerprint(
    iois: Sequence[int],
    ticks_per_beat: Optional[int],
) -> Optional[Dict[str, float]]:
    if not iois or not ticks_per_beat or ticks_per_beat <= 0:
        return None
    reference = {
        "quarter": float(ticks_per_beat),
        "eighth": ticks_per_beat / 2.0,
        "triplet": ticks_per_beat / 3.0,
        "sixteenth": ticks_per_beat / 4.0,
    }
    counts = {key: 0 for key in reference}
    other = 0
    for value in iois:
        best_key: Optional[str] = None
        best_error: Optional[float] = None
        for key, ref in reference.items():
            if ref <= 0:
                continue
            error = abs(float(value) - ref) / ref
            if best_error is None or error < best_error:
                best_error = error
                best_key = key
        if best_key is not None and best_error is not None and best_error <= 0.25:
            counts[best_key] += 1
        else:
            other += 1
    total = sum(counts.values()) + other
    if total == 0:
        return None
    fingerprint: Dict[str, float] = {key: counts[key] / total for key in counts}
    if other:
        fingerprint["other"] = other / total
    return fingerprint


def _fingerprint_hash(fingerprint: Optional[Dict[str, float]]) -> Optional[str]:
    if not fingerprint:
        return None
    serialised = json.dumps(fingerprint, sort_keys=True, ensure_ascii=False)
    return hashlib.sha1(serialised.encode("utf-8")).hexdigest()


def _compute_base_step(
    iois: Sequence[int],
    config: MetricConfig,
) -> Optional[float]:
    if not iois:
        return None
    filtered = [ioi for ioi in iois if ioi > 0]
    if not filtered:
        return None
    median = statistics.median(filtered)
    if median < config.min_base_step:
        return float(config.min_base_step)
    return float(median)


def compute_loop_metrics(
    notes: Sequence[Sequence[Any]],
    *,
    config: MetricConfig = MetricConfig(),
    ticks_per_beat: Optional[int] = None,
    tempo_events: Optional[Sequence[int]] = None,
) -> LoopMetrics:
    if not notes:
        return LoopMetrics(
            note_count=0,
            duration_ticks=0,
            base_step=None,
            swing_ratio=None,
            swing_confidence=0.0,
            ghost_rate=0.0,
            accent_rate=0.0,
            microtiming_mean=None,
            microtiming_std=None,
            microtiming_rms=None,
            onbeat_velocity_avg=None,
            offbeat_velocity_avg=None,
            on_off_velocity_ratio=None,
            syncopation_rate=None,
            fill_density=None,
            note_density_per_bar=None,
            layering_rate=0.0,
            velocity_mean=0.0,
            velocity_median=0.0,
            velocity_std=0.0,
            velocity_range=0,
            unique_velocity_steps=0,
            instrument_distribution={},
            hat_open_ratio=None,
            hat_transition_rate=None,
            repeat_rate=None,
            variation_factor=None,
            breakpoint_count=0,
            breakpoints=(),
            drum_collision_rate=None,
            role_separation=None,
            rhythm_fingerprint=None,
            rhythm_hash=None,
            tempo_stability=None,
        )

    starts = sorted(int(note[1]) for note in notes)
    velocities = [int(note[5]) for note in notes]
    durations = [int(note[2]) for note in notes]
    pitches = [int(note[4]) for note in notes]

    loop_end = max(start + duration for start, duration in zip(starts, durations))
    loop_duration = loop_end - min(starts)

    iois = [b - a for a, b in zip(starts, starts[1:]) if b - a >= 0]
    base_step = _compute_base_step(iois, config)
    tolerance = base_step * config.microtiming_tolerance_ratio if base_step else None

    # Swing metrics
    swing_ratio = None
    swing_confidence = 0.0
    if iois and len(iois) >= 2:
        even = [d for idx, d in enumerate(iois) if idx % 2 == 0 and d > 0]
        odd = [d for idx, d in enumerate(iois) if idx % 2 == 1 and d > 0]
        if even and odd:
            mean_even = statistics.mean(even)
            mean_odd = statistics.mean(odd)
            if mean_even > 0:
                swing_ratio = mean_odd / mean_even
            swing_confidence = min(
                1.0,
                len(iois) / (config.swing_min_pairs * 2),
            )

    # Dynamics metrics
    ghost_threshold = config.ghost_velocity_threshold
    accent_threshold = config.accent_velocity_threshold
    ghost_rate = sum(1 for v in velocities if v <= ghost_threshold) / len(velocities)
    accent_rate = sum(1 for v in velocities if v >= accent_threshold) / len(velocities)

    velocity_mean = statistics.mean(velocities)
    velocity_median = statistics.median(velocities)
    velocity_std = statistics.pstdev(velocities) if len(velocities) > 1 else 0.0
    velocity_range = max(velocities) - min(velocities)
    unique_velocity_steps = len(set(velocities))

    # Microtiming
    microtiming: List[float] = []
    on_velocities: List[int] = []
    off_velocities: List[int] = []
    syncopated = 0
    microtiming_mean: Optional[float]
    microtiming_std: Optional[float]
    microtiming_rms: Optional[float]
    syncopation_rate: Optional[float]
    onbeat_velocity_avg: Optional[float]
    offbeat_velocity_avg: Optional[float]
    on_off_ratio: Optional[float]
    fill_density: Optional[float]
    note_density_per_bar: Optional[float]
    if base_step and tolerance is not None and base_step > 0:
        for start, velocity in zip(starts, velocities):
            nearest = round(start / base_step) * base_step
            residual = start - nearest
            microtiming.append(abs(residual))
            if abs(residual) <= tolerance:
                on_velocities.append(velocity)
            else:
                off_velocities.append(velocity)
                syncopated += 1
        microtiming_mean = statistics.mean(microtiming) if microtiming else 0.0
        microtiming_std = statistics.pstdev(microtiming) if len(microtiming) > 1 else 0.0
        microtiming_rms = (
            math.sqrt(statistics.fmean(value**2 for value in microtiming)) if microtiming else 0.0
        )
        syncopation_rate = syncopated / len(starts)
        onbeat_velocity_avg = statistics.mean(on_velocities) if on_velocities else None
        offbeat_velocity_avg = statistics.mean(off_velocities) if off_velocities else None
        on_off_ratio = None
        if onbeat_velocity_avg and offbeat_velocity_avg:
            on_off_ratio = offbeat_velocity_avg / onbeat_velocity_avg

        steps = loop_duration / base_step if base_step else 0
        fill_density = len(starts) / steps if steps else None
        note_density_per_bar = None
        if steps:
            bars = steps / 16.0
            if bars:
                note_density_per_bar = len(starts) / bars
    else:
        microtiming_mean = None
        microtiming_std = None
        microtiming_rms = None
        syncopation_rate = None
        onbeat_velocity_avg = None
        offbeat_velocity_avg = None
        on_off_ratio = None
        fill_density = None
        note_density_per_bar = None

    # Layering
    timestamp_counts = _group_by_timestamp(starts)
    layered_hits = sum(count for count in timestamp_counts.values() if count > 1)
    layering_rate = layered_hits / len(starts)

    # Instrument distribution
    instrument_distribution: Counter[str] = Counter()
    hat_sequence: List[str] = []
    for pitch in pitches:
        category = _categorise_pitch(pitch)
        instrument_distribution[category] += 1
        if category in {"closed_hat", "open_hat"}:
            hat_sequence.append(category)

    total_hats = sum(instrument_distribution[c] for c in ("closed_hat", "open_hat"))
    hat_open_ratio = instrument_distribution["open_hat"] / total_hats if total_hats else None
    hat_transition_rate = None
    if len(hat_sequence) >= 2:
        transitions = sum(1 for a, b in zip(hat_sequence, hat_sequence[1:]) if a != b)
        hat_transition_rate = transitions / (len(hat_sequence) - 1)

    # Repetition analysis
    repeat_rate = None
    variation_factor = None
    breakpoints: Tuple[float, ...] = ()
    breakpoint_count = 0
    if base_step and loop_duration > 0:
        segment_len = base_step * 4
        if segment_len > 0:
            segments: Dict[int, List[Tuple[float, str]]] = defaultdict(list)
            for start, pitch in zip(starts, pitches):
                idx = int(start // segment_len)
                pos = (start % segment_len) / base_step if base_step else 0.0
                segments[idx].append((round(pos, 3), _categorise_pitch(pitch)))
            fingerprints: List[Tuple[Tuple[float, str], ...]] = []
            max_index = int(math.ceil(loop_duration / segment_len))
            for idx in range(max_index + 1):
                pattern = tuple(sorted(segments.get(idx, [])))
                fingerprints.append(pattern)
            if fingerprints:
                unique_patterns = len(set(fingerprints))
                variation_factor = unique_patterns / len(fingerprints)
                repeat_rate = 1.0 - variation_factor

            gaps = [gap for gap in iois if gap > segment_len]
            breakpoint_count = len(gaps)
            if gaps:
                positions: List[float] = []
                for gap, start in zip(iois, starts[1:]):
                    if gap > segment_len:
                        positions.append(start / loop_duration)
                breakpoints = tuple(positions[: config.max_breakpoints])

    # Drum collisions & role separation ------------------------------------
    timestamp_categories: Dict[int, List[str]] = defaultdict(list)
    for start, pitch in zip(starts, pitches):
        timestamp_categories[start].append(_categorise_pitch(pitch))
    target_categories = {"kick", "snare", "closed_hat", "open_hat"}
    collision_timestamps = 0
    active_timestamps = 0
    for categories in timestamp_categories.values():
        active = any(cat in target_categories for cat in categories)
        if active:
            active_timestamps += 1
            unique = {cat for cat in categories if cat in target_categories}
            if len(unique) >= 2:
                collision_timestamps += 1
    drum_collision_rate = collision_timestamps / active_timestamps if active_timestamps else 0.0
    total_notes = len(pitches)
    role_separation = None
    if total_notes:
        proportions = [count / total_notes for count in instrument_distribution.values()]
        role_separation = 1.0 - sum(prop**2 for prop in proportions)

    rhythm_fingerprint = _compute_rhythm_fingerprint(iois, ticks_per_beat)
    rhythm_hash = _fingerprint_hash(rhythm_fingerprint)

    tempo_stability = None
    if tempo_events:
        try:
            bpms = [60_000_000 / max(1, value) for value in tempo_events]
            if len(bpms) > 1:
                tempo_stability = statistics.pstdev(bpms)
        except (ValueError, ZeroDivisionError):
            tempo_stability = None

    return LoopMetrics(
        note_count=len(starts),
        duration_ticks=loop_duration,
        base_step=base_step,
        swing_ratio=swing_ratio,
        swing_confidence=swing_confidence,
        ghost_rate=ghost_rate,
        accent_rate=accent_rate,
        microtiming_mean=microtiming_mean,
        microtiming_std=microtiming_std,
        microtiming_rms=microtiming_rms,
        onbeat_velocity_avg=onbeat_velocity_avg,
        offbeat_velocity_avg=offbeat_velocity_avg,
        on_off_velocity_ratio=on_off_ratio,
        syncopation_rate=syncopation_rate,
        fill_density=fill_density,
        note_density_per_bar=note_density_per_bar,
        layering_rate=layering_rate,
        velocity_mean=velocity_mean,
        velocity_median=velocity_median,
        velocity_std=velocity_std,
        velocity_range=velocity_range,
        unique_velocity_steps=unique_velocity_steps,
        instrument_distribution=dict(instrument_distribution),
        hat_open_ratio=hat_open_ratio,
        hat_transition_rate=hat_transition_rate,
        repeat_rate=repeat_rate,
        variation_factor=variation_factor,
        breakpoint_count=breakpoint_count,
        breakpoints=breakpoints,
        drum_collision_rate=drum_collision_rate,
        role_separation=role_separation,
        rhythm_fingerprint=rhythm_fingerprint,
        rhythm_hash=rhythm_hash,
        tempo_stability=tempo_stability,
    )
