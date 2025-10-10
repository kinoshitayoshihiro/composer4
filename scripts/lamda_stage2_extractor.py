#!/usr/bin/env python3
"""Stage 2 extractor for LAMDa drum loops."""

from __future__ import annotations

import argparse
import importlib
import json
import math
import statistics
import sys
import hashlib
import subprocess
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import (
    Any,
    Dict,
    FrozenSet,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
    cast,
)

import pandas as pd
import yaml
from tqdm import tqdm
import numpy as np
from numpy.typing import NDArray

from lamda_tools import MetricConfig, compute_loop_metrics
from lamda_tools.metadata_io import iter_loop_records, load_metadata_index
from lamda_tools.metrics import DRUM_CATEGORIES

DEFAULT_CONFIG_PATH = Path("configs/lamda/drums_stage2.yaml")
DEFAULT_METADATA_INDEX = Path(
    "output/drumloops_metadata/drumloops_metadata_v2.pickle",
)
DEFAULT_METADATA_DIR = Path("output/drumloops_metadata")
DEFAULT_INPUT_DIR = Path("output/drumloops_cleaned")
DEFAULT_TMIDIX_PATH = Path("data/Los-Angeles-MIDI/CODE")
DEFAULT_ARTICULATION_THRESHOLDS_PATH = Path(
    "configs/thresholds/articulation.yaml",
)
DEFAULT_AXIS_ALIAS_PATH = Path("configs/aliases/axes.yaml")

DEFAULT_AXIS_WEIGHTS: Dict[str, float] = {
    "timing": 1.0,
    "velocity": 1.0,
    "groove_harmony": 1.0,
    "drum_cohesion": 1.0,
    "structure": 1.0,
    "articulation": 1.0,
}

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]

STRUCTURE_ROLE_INDEX: Dict[str, int] = {
    "none": 0,
    "kick": 1,
    "snare": 2,
    "closed_hat": 3,
    "open_hat": 4,
    "tom": 5,
    "perc": 6,
    "cymbal": 7,
    "other": 8,
}

ROLE_STABILITY_ROLES: Tuple[str, ...] = ("kick", "snare", "closed_hat")
ROLE_STABILITY_INDICES: Tuple[int, ...] = tuple(
    STRUCTURE_ROLE_INDEX[role] for role in ROLE_STABILITY_ROLES if role in STRUCTURE_ROLE_INDEX
)

BUILTIN_AXIS_ALIASES: Dict[str, str] = {
    "groove": "groove_harmony",
    "harmony": "groove_harmony",
    "cohesion": "drum_cohesion",
    "drums": "drum_cohesion",
    "art": "articulation",
    "artic": "articulation",
}

DEFAULT_TICKS_PER_BEAT = 480
FLAM_WINDOW_RATIO = 0.05
DETACHE_DURATION_RANGE = (0.9, 2.2)
DETACHE_GAP_RATIO = 0.15
PIZZICATO_DURATION_RATIO = 0.65
VIOLIN_PITCH_RANGE = (55, 103)

_CONDITION_PATTERN = re.compile(r"^\s*(<=|>=|<|>|==|!=)\s*(-?\d+(?:\.\d+)?)\s*$")


@dataclass
class VelocityScoringConfig:
    nbins: int
    targets: Dict[str, FloatArray]
    weights: Dict[str, float]
    metric_weights: Dict[str, float]
    prefill_window_beats: float = 0.5
    tempo_bins: Tuple[Dict[str, Any], ...] = ()
    tempo_bin_targets: Tuple[
        Tuple[Optional[float], Dict[str, FloatArray]],
        ...,
    ] = ()
    normalize_dynamic_range: bool = False
    dynamic_range: Dict[str, float] = field(default_factory=dict)
    tempo_dynamic_ranges: Tuple[
        Tuple[Optional[float], Dict[str, float]],
        ...,
    ] = ()
    phase_compensation: bool = False
    phase_adjust_db: Dict[str, float] = field(default_factory=dict)
    tempo_phase_adjust: Tuple[
        Tuple[Optional[float], Dict[str, float]],
        ...,
    ] = ()

    def target(self, key: str) -> FloatArray:
        default = self.targets.get("global")
        if default is None:
            raise ValueError("velocity targets must define 'global'")
        if key in self.targets:
            return self.targets[key]
        return default


@dataclass
class DensityBandSpec:
    max_bpm: Optional[float]
    low: float
    high: float


@dataclass
class AudioAdaptiveRule:
    operator: str
    threshold: float
    multipliers: Dict[str, float]
    name: str
    priority: int = 0
    extras: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AudioAdaptiveAxisCap:
    min_scale: Optional[float] = None
    max_scale: Optional[float] = None


@dataclass
class AudioAdaptiveFusionSource:
    path: Tuple[str, ...]
    weight: float
    required: bool = False
    bias: float = 0.0


@dataclass
class AudioAdaptiveFusion:
    sources: Tuple[AudioAdaptiveFusionSource, ...]
    bias: float = 0.0
    normalise_weights: bool = True
    default: Optional[float] = None
    clamp_min: Optional[float] = None
    clamp_max: Optional[float] = None
    temperature: Optional[float] = None


@dataclass
class AudioAdaptiveWeights:
    pivot_path: Tuple[str, ...]
    rules: Tuple[AudioAdaptiveRule, ...]
    enabled: bool = True
    min_confidence: Optional[float] = None
    missing_policy: str = "noop"
    fusion: Optional[AudioAdaptiveFusion] = None
    min_scale: Optional[float] = None
    max_scale: Optional[float] = None
    axis_caps: Dict[str, AudioAdaptiveAxisCap] = field(default_factory=dict)
    normalize_sum: bool = False
    normalize_target: Optional[float] = None
    hysteresis_margin: float = 0.0
    cooldown_loops: int = 0
    cooldown_by_rule: Dict[str, int] = field(default_factory=dict)
    max_total_delta: Optional[float] = None
    max_total_delta_per_axis: Dict[str, float] = field(default_factory=dict)
    pivot_ema_alpha: Optional[float] = None
    log_level: str = "summary"


@dataclass
class AudioAdaptiveState:
    last_rule_name: Optional[str] = None
    last_pivot: Optional[float] = None
    cooldown_remaining: int = 0
    rule_cooldowns: Dict[str, int] = field(default_factory=dict)
    pivot_ema: Optional[float] = None


@dataclass
class AudioAdaptiveEvaluation:
    rule: Optional[AudioAdaptiveRule]
    pivot: Optional[float]
    state: Optional[AudioAdaptiveState]
    flags: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StructureScoringConfig:
    bands: List[DensityBandSpec]
    weights: Dict[str, float]
    grid_divisions: int = 16
    periodicity_candidates: Tuple[int, ...] = (4, 8, 12, 16)

    def expected_density(self, bpm: Optional[float]) -> Tuple[float, float]:
        if not self.bands:
            return 0.0, 0.0
        if bpm is None:
            return self.bands[0].low, self.bands[0].high
        for band in self.bands:
            if band.max_bpm is None or bpm <= band.max_bpm:
                return band.low, band.high
        last = self.bands[-1]
        return last.low, last.high


def _build_feature_context(
    notes: Sequence[Sequence[Any]],
    *,
    ticks_per_beat: int,
    time_signature: Tuple[int, int],
    channel_programs: Dict[int, int],
    duration_ticks: int,
    bpm_hint: Optional[float],
    grid_divisions: int = 16,
) -> Optional[LoopFeatureContext]:
    if not notes:
        return None

    ts_numerator, ts_denominator = time_signature
    if ts_numerator <= 0:
        ts_numerator = 4
    if ts_denominator <= 0:
        ts_denominator = 4

    effective_tpb = float(max(ticks_per_beat, 1))
    ts_ratio = 4.0 / float(ts_denominator)
    beat_ticks = effective_tpb * ts_ratio
    if beat_ticks <= 0:
        return None
    bar_ticks = beat_ticks * float(ts_numerator)
    if bar_ticks <= 0:
        bar_ticks = beat_ticks * 4.0

    velocities_list: List[float] = []
    beats_list: List[float] = []
    beat_positions_list: List[float] = []

    onset_counts = np.zeros(grid_divisions, dtype=np.float64)
    num_roles = len(STRUCTURE_ROLE_INDEX)
    role_bins = np.zeros((grid_divisions, num_roles), dtype=np.float64)

    def _create_role_slot_sets() -> List[Set[int]]:
        return [set() for _ in range(num_roles)]

    role_slot_sets: Dict[int, List[Set[int]]] = defaultdict(_create_role_slot_sets)

    for note in notes:
        start = int(note[1])
        velocity = float(note[5])
        channel = int(note[3])
        pitch = int(note[4])
        program = channel_programs.get(channel, 0)

        velocities_list.append(velocity)
        if beat_ticks > 0:
            beats_list.append(start / beat_ticks)
        else:
            beats_list.append(0.0)

        if bar_ticks > 0:
            bar_index = int(start // bar_ticks)
            slot_position = (start % bar_ticks) / bar_ticks
        else:
            bar_index = 0
            slot_position = (start / beat_ticks) % 1.0 if beat_ticks else 0.0
        beat_positions_list.append(slot_position)

        slot_float = slot_position * grid_divisions
        slot_index = int(math.floor(slot_float))
        slot_index = max(0, min(grid_divisions - 1, slot_index))

        onset_counts[slot_index] += 1.0
        role, _ = _infer_role(channel, program, pitch)
        role_index = STRUCTURE_ROLE_INDEX.get(role, STRUCTURE_ROLE_INDEX["other"])
        role_bins[slot_index, role_index] += 1.0
        role_slot_sets[bar_index][role_index].add(slot_index)

    velocities = np.asarray(velocities_list, dtype=np.float64)
    beats = np.asarray(beats_list, dtype=np.float64)
    beat_positions = np.asarray(beat_positions_list, dtype=np.float64)

    bars_float = 1.0
    if bar_ticks > 0 and duration_ticks > 0:
        bars_float = max(1.0, math.ceil(duration_ticks / bar_ticks))
    onset_counts = onset_counts / bars_float
    role_tokens = np.argmax(role_bins, axis=1).astype(np.int64)

    beats_per_bar = float(ts_numerator) if ts_numerator > 0 else 4.0
    bars_int = max(1, int(bars_float))

    role_slots_per_bar_list: List[Tuple[FrozenSet[int], ...]] = []
    for bar_idx in range(bars_int):
        slot_sets = role_slot_sets.get(bar_idx)
        if slot_sets is None:
            slot_sets = _create_role_slot_sets()
        frozen_slots = tuple(frozenset(slot_sets[idx]) for idx in range(num_roles))
        role_slots_per_bar_list.append(frozen_slots)
    role_slots_per_bar = tuple(role_slots_per_bar_list)

    return LoopFeatureContext(
        velocities=velocities,
        beats=beats,
        beat_positions=beat_positions,
        bpm=bpm_hint,
        onset_counts_16th=onset_counts,
        role_tokens_16th=role_tokens,
        beats_per_bar=beats_per_bar,
        bars=bars_int,
        role_slots_per_bar=role_slots_per_bar,
    )


@dataclass
class LoopFeatureContext:
    velocities: FloatArray
    beats: FloatArray
    beat_positions: FloatArray
    bpm: Optional[float]
    onset_counts_16th: FloatArray
    role_tokens_16th: IntArray
    beats_per_bar: float
    bars: int
    role_slots_per_bar: Tuple[Tuple[FrozenSet[int], ...], ...]

    @property
    def total_weight(self) -> float:
        return sum(self.weights.values())

    def tempo_bins(self) -> List[float]:
        if self.provider is not None:
            return self.provider.bins.edges
        tempo_cfg = cast(Dict[str, Any], self.auto.get("bins", {}))
        bins_obj = tempo_cfg.get("tempo", [])
        if not isinstance(bins_obj, list):
            return []
        result: List[float] = []
        for bound in cast(List[Any], bins_obj):
            try:
                result.append(float(bound))
            except (TypeError, ValueError):
                continue
        return result

    def auto_min_support(self, category: str) -> Optional[int]:
        values = cast(Dict[str, Any], self.auto.get("min_support", {}))
        if category not in values:
            return None
        try:
            return int(values[category])
        except (TypeError, ValueError):
            return None

    def hysteresis_drop_ratio(self) -> float:
        quantiles_cfg = cast(Dict[str, Any], self.auto.get("quantiles", {}))
        value = quantiles_cfg.get("hysteresis_drop_iqr", 0.1)
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.1


@dataclass
class ArticulationObservation:
    loop_id: str
    metrics: Dict[str, Optional[float]]
    support: Dict[str, Any]
    tempo_bpm: Optional[float]
    metrics_full: Dict[str, Any]


@dataclass
class ArticulationResult:
    score: float
    labels: List[str]
    presence: Dict[str, float]
    thresholds: Dict[str, Any]
    axis_value: float

    @classmethod
    def empty(cls) -> "ArticulationResult":
        return cls(
            score=0.0,
            labels=[],
            presence={},
            thresholds={},
            axis_value=0.0,
        )


@dataclass
class Stage2Paths:
    events_parquet: Path
    events_csv_sample: Optional[Path]
    loop_summary_csv: Path
    metrics_jsonl: Path
    retry_dir: Path
    summary_out: Optional[Path]
    sample_event_rows: int = 5000
    audio_embeddings_parquet: Optional[Path] = None


AUDIO_LOOP_DEFAULTS: Dict[str, Optional[Any]] = {
    "text_audio_cos": None,
    "text_audio_cos_mert": None,
    "caption": None,
    "caption_en": None,
    "audio.render_path": None,
    "audio.embedding_path": None,
}


def _as_optional_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or None
    text = str(value)
    return text.strip() or None


def _to_optional_float(value: Any) -> Optional[float]:
    try:
        if value is None or (isinstance(value, str) and not value.strip()):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_float_array(value: Any) -> Optional[np.ndarray]:
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        if value.ndim != 1:
            return None
        if value.dtype != np.float32:
            return value.astype(np.float32)
        return value
    if isinstance(value, (list, tuple)):
        try:
            array = np.asarray(value, dtype=np.float32)
        except (TypeError, ValueError):
            return None
        if array.ndim != 1:
            return None
        return array
    return None


def _split_path(path: str) -> Tuple[str, ...]:
    return tuple(part for part in path.split(".") if part)


def _resolve_nested_value(
    context: Mapping[str, Any],
    path_parts: Tuple[str, ...],
) -> Any:
    value: Any = context
    for part in path_parts:
        if isinstance(value, Mapping) and part in value:
            value = value[part]
        else:
            return None
    return value


def _parse_threshold_expression(
    expression: str,
) -> Optional[Tuple[str, float]]:
    match = _CONDITION_PATTERN.match(expression)
    if not match:
        return None
    operator, literal = match.groups()
    try:
        return operator, float(literal)
    except (TypeError, ValueError):
        return None


def _resolve_audio_adaptive_rule(
    rules: Tuple[AudioAdaptiveRule, ...],
    name: Optional[str],
) -> Optional[AudioAdaptiveRule]:
    if name is None:
        return None
    for rule in rules:
        if rule.name == name:
            return rule
    return None


def _temperature_scale_value(
    value: Optional[float],
    temperature: Optional[float],
    clamp_min: Optional[float] = None,
    clamp_max: Optional[float] = None,
) -> Optional[float]:
    if value is None or temperature is None:
        return value
    try:
        temp = float(temperature)
    except (TypeError, ValueError):
        return value
    if temp <= 0.0 or math.isclose(temp, 1.0):
        return value
    epsilon = 1e-6
    lower = clamp_min if clamp_min is not None else 0.0
    upper = clamp_max if clamp_max is not None else 1.0
    span = max(epsilon, float(upper) - float(lower))
    normalised = (float(value) - lower) / span
    normalised = min(max(normalised, epsilon), 1.0 - epsilon)
    logit = math.log(normalised / (1.0 - normalised))
    scaled = 1.0 / (1.0 + math.exp(-logit / temp))
    result = lower + scaled * span
    result = min(max(result, lower), upper)
    return result


def _fuse_audio_adaptive_sources(
    fusion: AudioAdaptiveFusion,
    context: Mapping[str, Any],
) -> Optional[float]:
    values: List[float] = []
    weights: List[float] = []
    for source in fusion.sources:
        raw_value = _resolve_nested_value(context, source.path)
        numeric = _to_optional_float(raw_value)
        if numeric is None:
            if source.required:
                return fusion.default
            continue
        adjusted = numeric + source.bias
        values.append(adjusted)
        weights.append(float(source.weight))

    if not values:
        return fusion.default

    if fusion.normalise_weights:
        weight_sum = sum(weights)
        if weight_sum <= 0.0:
            weights = [1.0 / float(len(values))] * len(values)
        else:
            weights = [w / weight_sum for w in weights]

    fused = sum(value * weight for value, weight in zip(values, weights)) + fusion.bias
    if fusion.clamp_min is not None:
        fused = max(fused, fusion.clamp_min)
    if fusion.clamp_max is not None:
        fused = min(fused, fusion.clamp_max)
    fused = _temperature_scale_value(fused, fusion.temperature, fusion.clamp_min, fusion.clamp_max)
    return fused


def _resolve_audio_adaptive_pivot(
    adaptive: AudioAdaptiveWeights,
    context: Mapping[str, Any],
) -> Optional[float]:
    if adaptive.fusion is not None:
        fused = _fuse_audio_adaptive_sources(adaptive.fusion, context)
        if fused is not None:
            return fused
    pivot_raw = _resolve_nested_value(context, adaptive.pivot_path)
    return _to_optional_float(pivot_raw)


def _apply_audio_caps(
    weights: Dict[str, float],
    base_weights: Mapping[str, float],
    adaptive: AudioAdaptiveWeights,
) -> None:
    for axis_key, value in weights.items():
        base_value = base_weights.get(axis_key)
        if base_value is None:
            continue
        min_scale = adaptive.min_scale
        max_scale = adaptive.max_scale
        axis_cap = adaptive.axis_caps.get(axis_key)
        if axis_cap is not None:
            if axis_cap.min_scale is not None:
                min_scale = axis_cap.min_scale
            if axis_cap.max_scale is not None:
                max_scale = axis_cap.max_scale
        if min_scale is not None:
            weights[axis_key] = max(value, base_value * float(min_scale))
            value = weights[axis_key]
        if max_scale is not None:
            weights[axis_key] = min(value, base_value * float(max_scale))


def _normalise_audio_weights(
    weights: Dict[str, float],
    base_weights: Mapping[str, float],
    adaptive: AudioAdaptiveWeights,
) -> None:
    if not adaptive.normalize_sum:
        return
    target_sum = adaptive.normalize_target
    if target_sum is None:
        target_sum = sum(float(value) for value in base_weights.values())
    if target_sum <= 0.0:
        return
    bounds: Dict[str, Tuple[float, float]] = {}
    tiny = 1e-9
    for axis_key, value in weights.items():
        base_value = base_weights.get(axis_key, value)
        min_scale = adaptive.min_scale
        max_scale = adaptive.max_scale
        axis_cap = adaptive.axis_caps.get(axis_key)
        if axis_cap is not None:
            if axis_cap.min_scale is not None:
                min_scale = axis_cap.min_scale
            if axis_cap.max_scale is not None:
                max_scale = axis_cap.max_scale
        if min_scale is not None:
            lower = float(base_value) * float(min_scale)
        else:
            lower = 0.0
        if max_scale is not None:
            upper = float(base_value) * float(max_scale)
        else:
            upper = float("inf")
        bounds[axis_key] = (lower, upper)

    for _ in range(16):
        current_sum = sum(float(value) for value in weights.values())
        delta = target_sum - current_sum
        if abs(delta) <= tiny:
            break
        if delta > 0.0:
            unbounded = [axis for axis, (_, upper) in bounds.items() if math.isinf(upper)]
            if unbounded:
                share = delta / float(len(unbounded))
                for axis in unbounded:
                    weights[axis] += share
                continue
            adjustable = {
                axis: bounds[axis][1] - weights[axis]
                for axis in weights
                if bounds[axis][1] - weights[axis] > tiny
            }
            capacity = sum(adjustable.values())
            if capacity <= tiny:
                break
            for axis, room in adjustable.items():
                portion = delta * (room / capacity)
                weights[axis] += min(portion, room)
        else:
            adjustable = {
                axis: weights[axis] - bounds[axis][0]
                for axis in weights
                if weights[axis] - bounds[axis][0] > tiny
            }
            if not adjustable:
                break
            capacity = sum(adjustable.values())
            if capacity <= tiny:
                break
            for axis, room in adjustable.items():
                portion = (-delta) * (room / capacity)
                weights[axis] -= min(portion, room)

    for axis, (lower, upper) in bounds.items():
        if weights[axis] < lower:
            weights[axis] = lower
        if weights[axis] > upper:
            weights[axis] = upper


def _evaluate_audio_adaptive_rule(
    adaptive: Optional[AudioAdaptiveWeights],
    context: Mapping[str, Any],
    state: Optional[AudioAdaptiveState],
) -> AudioAdaptiveEvaluation:
    flags: Dict[str, Any] = {}
    if adaptive is None:
        return AudioAdaptiveEvaluation(None, None, state, flags)

    if state is None:
        state = AudioAdaptiveState()
        flags["state_initialised"] = True

    state.rule_cooldowns = dict(state.rule_cooldowns or {})

    cooldown_before = state.cooldown_remaining
    flags["cooldown_before"] = cooldown_before
    flags["rule_cooldowns_before"] = dict(state.rule_cooldowns)

    if not adaptive.enabled:
        state.last_rule_name = None
        state.last_pivot = None
        state.cooldown_remaining = 0
        state.rule_cooldowns.clear()
        state.pivot_ema = None
        flags.update(
            {
                "disabled": True,
                "selected_rule_name": None,
                "initial_rule_name": None,
                "hysteresis_applied": False,
                "cooldown_active": False,
                "cooldown_remaining": 0,
                "rule_cooldowns": {},
            }
        )
        return AudioAdaptiveEvaluation(None, None, state, flags)

    if state.cooldown_remaining > 0:
        state.cooldown_remaining = max(0, state.cooldown_remaining - 1)
        flags["cooldown_decremented"] = True

    cooldown_active = state.cooldown_remaining > 0
    flags["cooldown_active_before"] = cooldown_before > 0

    if state.rule_cooldowns:
        decremented: Dict[str, int] = {}
        for key, remaining in list(state.rule_cooldowns.items()):
            new_value = max(0, int(remaining) - 1)
            if new_value <= 0:
                state.rule_cooldowns.pop(key, None)
            else:
                state.rule_cooldowns[key] = new_value
                decremented[key] = new_value
        if decremented:
            flags["rule_cooldowns_decremented"] = decremented
    flags["rule_cooldowns_after"] = dict(state.rule_cooldowns)

    pivot_value = _resolve_audio_adaptive_pivot(adaptive, context)
    if adaptive.fusion is not None and adaptive.fusion.temperature is not None:
        try:
            flags["temperature"] = float(adaptive.fusion.temperature)
        except (TypeError, ValueError):
            flags["temperature"] = adaptive.fusion.temperature
    missing_policy = adaptive.missing_policy.lower()
    flags["missing_policy"] = missing_policy
    if pivot_value is None:
        if missing_policy == "zero":
            pivot_value = 0.0
            flags["missing_policy_applied"] = "zero"
        elif missing_policy == "last" and state.last_pivot is not None:
            pivot_value = state.last_pivot
            flags["missing_policy_applied"] = "last"
        else:
            state.last_rule_name = None
            state.last_pivot = None
            state.cooldown_remaining = 0
            state.rule_cooldowns.clear()
            state.pivot_ema = None
            flags.update(
                {
                    "pivot_missing": True,
                    "selected_rule_name": None,
                    "initial_rule_name": None,
                    "hysteresis_applied": False,
                    "cooldown_active": cooldown_active,
                    "cooldown_remaining": state.cooldown_remaining,
                    "rule_cooldowns": {},
                }
            )
            return AudioAdaptiveEvaluation(None, None, state, flags)
    else:
        flags["missing_policy_applied"] = None

    pivot_numeric: Optional[float]
    try:
        pivot_numeric = float(pivot_value) if pivot_value is not None else None
    except (TypeError, ValueError):
        pivot_numeric = None

    min_conf = adaptive.min_confidence
    if min_conf is not None and pivot_numeric is not None and pivot_numeric < float(min_conf):
        state.last_rule_name = None
        state.last_pivot = pivot_numeric
        state.cooldown_remaining = 0
        state.rule_cooldowns.clear()
        state.pivot_ema = None
        flags.update(
            {
                "below_min_confidence": True,
                "selected_rule_name": None,
                "initial_rule_name": None,
                "hysteresis_applied": False,
                "cooldown_active": cooldown_active,
                "cooldown_remaining": state.cooldown_remaining,
                "rule_cooldowns": {},
            }
        )
        return AudioAdaptiveEvaluation(None, pivot_numeric, state, flags)

    pivot_raw = pivot_numeric
    if pivot_numeric is not None:
        flags["pivot_raw"] = pivot_numeric
        alpha = adaptive.pivot_ema_alpha
        alpha_value: Optional[float]
        try:
            alpha_value = float(alpha) if alpha is not None else None
        except (TypeError, ValueError):
            alpha_value = None
        if alpha_value is not None and 0.0 < alpha_value <= 1.0:
            previous = state.pivot_ema
            if previous is None:
                state.pivot_ema = pivot_numeric
            else:
                state.pivot_ema = (alpha_value * pivot_numeric) + ((1.0 - alpha_value) * previous)
            pivot_numeric = state.pivot_ema
            flags["pivot_ema"] = state.pivot_ema
        pivot_value = pivot_numeric
    else:
        flags["pivot_raw"] = pivot_value

    candidate_rule: Optional[AudioAdaptiveRule] = None
    matched_rules: List[Tuple[int, AudioAdaptiveRule]] = []
    for index, rule in enumerate(adaptive.rules):
        if _compare_numeric(pivot_value, rule.operator, rule.threshold):
            matched_rules.append((index, rule))

    if matched_rules:
        matched_rules.sort(key=lambda pair: (pair[1].priority, -pair[0]), reverse=True)
        candidate_rule = matched_rules[0][1]
        flags["matched_rules"] = [rule.name for _, rule in matched_rules]

    initial_rule = candidate_rule
    flags["initial_rule_name"] = initial_rule.name if initial_rule else None

    last_rule = _resolve_audio_adaptive_rule(adaptive.rules, state.last_rule_name)
    margin = adaptive.hysteresis_margin
    hysteresis_applied = False
    rule_specific_cooldowns = adaptive.cooldown_by_rule or {}
    rule_cooldown_blocked: Optional[Dict[str, Any]] = None

    if last_rule is not None:
        if cooldown_active:
            candidate_rule = last_rule
        elif candidate_rule is None and margin > 0.0 and state.last_pivot is not None:
            if abs(pivot_value - state.last_pivot) < margin:
                candidate_rule = last_rule
                hysteresis_applied = True
        elif (
            candidate_rule is not None
            and candidate_rule.name != last_rule.name
            and margin > 0.0
            and state.last_pivot is not None
            and abs(pivot_value - state.last_pivot) < margin
        ):
            candidate_rule = last_rule
            hysteresis_applied = True

    if candidate_rule is not None and rule_specific_cooldowns:
        remaining = int(state.rule_cooldowns.get(candidate_rule.name, 0))
        if remaining > 0:
            rule_cooldown_blocked = {
                "name": candidate_rule.name,
                "remaining": remaining,
            }
            flags["rule_cooldown_blocked"] = rule_cooldown_blocked
            candidate_rule = None

    if candidate_rule is None:
        state.last_rule_name = None
        state.last_pivot = pivot_value
        state.cooldown_remaining = 0
        flags.update(
            {
                "selected_rule_name": None,
                "hysteresis_applied": hysteresis_applied,
                "cooldown_active": cooldown_active,
                "cooldown_remaining": state.cooldown_remaining,
                "rule_cooldowns": dict(state.rule_cooldowns),
                "rule_cooldown_blocked": rule_cooldown_blocked,
            }
        )
        return AudioAdaptiveEvaluation(None, pivot_value, state, flags)

    if candidate_rule.name != state.last_rule_name:
        state.cooldown_remaining = adaptive.cooldown_loops
    state.last_rule_name = candidate_rule.name
    state.last_pivot = pivot_value

    configured_cooldown = int(rule_specific_cooldowns.get(candidate_rule.name, 0) or 0)
    if configured_cooldown > 0:
        state.rule_cooldowns[candidate_rule.name] = configured_cooldown

    flags.update(
        {
            "selected_rule_name": candidate_rule.name,
            "hysteresis_applied": hysteresis_applied,
            "cooldown_active": cooldown_active,
            "cooldown_remaining": state.cooldown_remaining,
            "rule_cooldowns": dict(state.rule_cooldowns),
        }
    )

    return AudioAdaptiveEvaluation(candidate_rule, pivot_value, state, flags)


def _scale_velocity_phase_compensation(
    config: Optional[VelocityScoringConfig],
    factor: Optional[float],
) -> Optional[VelocityScoringConfig]:
    if config is None or factor is None:
        return config
    try:
        factor_value = float(factor)
    except (TypeError, ValueError):
        return config
    if not config.phase_compensation:
        return config
    if math.isclose(factor_value, 1.0):
        return config

    def _scaled_map(source: Mapping[str, Any]) -> Dict[str, float]:
        scaled: Dict[str, float] = {}
        for key, value in source.items():
            try:
                scaled[str(key)] = float(value) * factor_value
            except (TypeError, ValueError):
                continue
        return scaled

    base_adjust = _scaled_map(config.phase_adjust_db)
    tempo_adjust: Tuple[Tuple[Optional[float], Dict[str, float]], ...] = tuple(
        (
            tempo_limit,
            _scaled_map(adjust_map),
        )
        for tempo_limit, adjust_map in config.tempo_phase_adjust
    )

    return replace(
        config,
        phase_adjust_db=base_adjust,
        tempo_phase_adjust=tempo_adjust,
    )


def _apply_audio_adaptive_weights(
    base_weights: Dict[str, float],
    adaptive: Optional[AudioAdaptiveWeights],
    context: Mapping[str, Any],
    state: Optional[AudioAdaptiveState] = None,
    evaluated: Optional[AudioAdaptiveEvaluation] = None,
) -> Tuple[
    Dict[str, float],
    Optional[AudioAdaptiveRule],
    Optional[float],
    Optional[AudioAdaptiveState],
    Dict[str, Any],
]:
    weights_before = dict(base_weights)
    weights = dict(base_weights)

    evaluation: Optional[AudioAdaptiveEvaluation] = evaluated
    if adaptive is None:
        details = {
            "rule": None,
            "rule_name": None,
            "pivot": None,
            "weights": {"before": weights_before, "after": dict(weights)},
            "hysteresis_applied": False,
            "cooldown_active": False,
            "cooldown_remaining": 0,
            "disabled": True,
            "missing_policy_applied": None,
            "below_min_confidence": False,
            "rule_cooldown_blocked": None,
            "total_delta": 0.0,
            "total_delta_limited": False,
            "flags": {},
        }
        return weights, None, None, state, details

    if evaluation is None:
        evaluation = _evaluate_audio_adaptive_rule(
            adaptive,
            context,
            state,
        )

    candidate_rule = evaluation.rule
    pivot_value = evaluation.pivot
    state = evaluation.state
    flags = dict(evaluation.flags)

    cooldown_remaining = state.cooldown_remaining if state is not None else 0
    cooldown_active = bool(flags.get("cooldown_active", cooldown_remaining > 0))

    details: Dict[str, Any] = {
        "rule": candidate_rule,
        "rule_name": candidate_rule.name if candidate_rule else None,
        "pivot": pivot_value,
        "weights": {"before": weights_before, "after": dict(weights)},
        "hysteresis_applied": bool(flags.get("hysteresis_applied", False)),
        "cooldown_active": cooldown_active,
        "cooldown_remaining": cooldown_remaining,
        "disabled": bool(flags.get("disabled", False)),
        "missing_policy_applied": flags.get("missing_policy_applied"),
        "below_min_confidence": bool(flags.get("below_min_confidence", False)),
        "rule_cooldown_blocked": flags.get("rule_cooldown_blocked"),
        "total_delta": 0.0,
        "total_delta_limited": False,
        "flags": flags,
        "log_level": adaptive.log_level,
    }

    if candidate_rule is None:
        details["weights"]["after"] = dict(weights)
        return weights, None, pivot_value, state, details

    pending_updates: Dict[str, float] = {}
    total_delta = 0.0
    for axis_key, multiplier in candidate_rule.multipliers.items():
        if axis_key in weights:
            base_value = weights[axis_key]
            try:
                target_value = base_value * float(multiplier)
            except (TypeError, ValueError):
                continue
            pending_updates[axis_key] = target_value
            total_delta += abs(target_value - base_value)

    details["total_delta"] = float(total_delta)

    # Per-axis delta limiting (priority over global max_total_delta)
    per_axis_limits = adaptive.max_total_delta_per_axis
    if per_axis_limits:
        for axis_key, target_value in list(pending_updates.items()):
            axis_limit = per_axis_limits.get(axis_key)
            if axis_limit is not None:
                base_value = weights[axis_key]
                delta = abs(target_value - base_value)
                if delta > axis_limit:
                    ratio_axis = axis_limit / delta
                    adjusted_value = base_value + (target_value - base_value) * ratio_axis
                    pending_updates[axis_key] = adjusted_value
                    details.setdefault("per_axis_limited", {})[axis_key] = {
                        "original_delta": float(delta),
                        "limit": float(axis_limit),
                        "ratio": float(ratio_axis),
                    }
        # Recalculate total_delta after per-axis limiting
        total_delta = sum(abs(pending_updates[axis] - weights[axis]) for axis in pending_updates)
        details["total_delta"] = float(total_delta)

    # Global max_total_delta (fallback if no per-axis limits)
    max_total_delta = adaptive.max_total_delta
    ratio = 1.0
    if max_total_delta is not None:
        try:
            max_total_delta_value = float(max_total_delta)
        except (TypeError, ValueError):
            max_total_delta_value = None
        if (
            max_total_delta_value is not None
            and max_total_delta_value >= 0.0
            and total_delta > max_total_delta_value > 0.0
        ):
            ratio = max_total_delta_value / total_delta
            details["total_delta_limited"] = True
            details["total_delta_ratio"] = ratio

    if ratio != 1.0:
        for axis_key, target_value in list(pending_updates.items()):
            base_value = weights[axis_key]
            adjusted_value = base_value + (target_value - base_value) * ratio
            pending_updates[axis_key] = adjusted_value

        total_delta = sum(abs(pending_updates[axis] - weights[axis]) for axis in pending_updates)
        details["total_delta"] = float(total_delta)

    for axis_key, target_value in pending_updates.items():
        weights[axis_key] = target_value

    _apply_audio_caps(weights, base_weights, adaptive)
    _normalise_audio_weights(weights, base_weights, adaptive)
    # Re-apply per-axis caps after normalisation to enforce hard limits.
    _apply_audio_caps(weights, base_weights, adaptive)

    details["weights"]["after"] = dict(weights)

    return weights, candidate_rule, pivot_value, state, details


def _adaptive_rule_to_dict(rule: Optional[AudioAdaptiveRule]) -> Optional[Dict[str, Any]]:
    if rule is None:
        return None
    return {
        "name": rule.name,
        "operator": rule.operator,
        "threshold": rule.threshold,
        "multipliers": dict(rule.multipliers),
        "extras": dict(rule.extras),
        "priority": rule.priority,
    }


def _summarise_adaptive_flags(flags: Mapping[str, Any], log_level: str) -> Dict[str, Any]:
    if log_level == "debug":
        return dict(flags)
    summary_keys = (
        "initial_rule_name",
        "selected_rule_name",
        "hysteresis_applied",
        "cooldown_before",
        "cooldown_remaining",
        "cooldown_active",
        "rule_cooldown_blocked",
        "pivot_raw",
        "pivot_ema",
        "temperature",
        "matched_rules",
    )
    return {key: flags.get(key) for key in summary_keys if key in flags}


def _summarise_adaptive_details(
    details: Mapping[str, Any],
    applied_rule: Optional[AudioAdaptiveRule],
    log_level: str,
) -> Dict[str, Any]:
    base: Dict[str, Any] = {
        "rule": _adaptive_rule_to_dict(applied_rule),
        "rule_name": details.get("rule_name"),
        "pivot": details.get("pivot"),
        "total_delta": details.get("total_delta"),
        "total_delta_limited": details.get("total_delta_limited"),
        "total_delta_ratio": details.get("total_delta_ratio"),
        "hysteresis_applied": details.get("hysteresis_applied"),
        "cooldown_active": details.get("cooldown_active"),
        "cooldown_remaining": details.get("cooldown_remaining"),
        "missing_policy_applied": details.get("missing_policy_applied"),
        "below_min_confidence": details.get("below_min_confidence"),
        "rule_cooldown_blocked": details.get("rule_cooldown_blocked"),
        "log_level": log_level,
    }

    weights_entry = details.get("weights")
    if isinstance(weights_entry, Mapping):
        after_weights = cast(Mapping[str, Any], weights_entry.get("after"))
        if isinstance(after_weights, Mapping):
            base["weights_after"] = {
                str(axis): float(value)
                for axis, value in after_weights.items()
                if isinstance(value, (int, float))
            }
        if log_level == "debug":
            before_weights = cast(Mapping[str, Any], weights_entry.get("before"))
            if isinstance(before_weights, Mapping):
                base["weights_before"] = {
                    str(axis): float(value)
                    for axis, value in before_weights.items()
                    if isinstance(value, (int, float))
                }

    if log_level == "debug":
        base["flags"] = details.get("flags")

    return base


def _build_audio_summary(
    loop_id: str,
    audio_row: Optional[AudioGuidanceRow],
) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
    if audio_row is None:
        return {}, None

    summary: Dict[str, Any] = {
        "text_audio_cos": audio_row.text_audio_cos,
        "text_audio_cos_mert": audio_row.text_audio_cos_mert,
        "caption": audio_row.caption,
        "caption_en": audio_row.caption_en or audio_row.caption,
        "render_path": audio_row.render_path or audio_row.audio_path,
        "embedding_path": audio_row.embedding_path,
        "model": audio_row.audio_model,
    }

    clap_embed = audio_row.audio_embed_clap
    if clap_embed is None and audio_row.extra:
        clap_embed = _coerce_float_array(
            audio_row.extra.get("audio_embed_clap")
            or audio_row.extra.get("audio_embedding_clap")
            or audio_row.extra.get("audio_embedding")
        )
    mert_embed = audio_row.audio_embed_mert
    if mert_embed is None and audio_row.extra:
        mert_embed = _coerce_float_array(audio_row.extra.get("audio_embed_mert"))
    text_embed = audio_row.text_embed
    if text_embed is None and audio_row.extra:
        text_embed = _coerce_float_array(
            audio_row.extra.get("text_embed") or audio_row.extra.get("text_embedding")
        )

    embed_entry: Optional[Dict[str, Any]] = None
    if any(embed is not None for embed in (clap_embed, mert_embed, text_embed)):
        embed_entry = {
            "loop_id": loop_id,
            "model": summary.get("model"),
            "audio_embed_clap": clap_embed,
            "audio_embed_mert": mert_embed,
            "text_embed": text_embed,
        }

    return summary, embed_entry


def _write_audio_embeddings(
    rows: Sequence[Dict[str, Any]],
    out_path: Path,
) -> None:
    if not rows:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError:  # pragma: no cover - optional dependency
        sys.stderr.write("[warn] pyarrow not available; skipping audio embeddings parquet output\n")
        return

    def _embed_column(key: str) -> List[Optional[List[float]]]:
        values: List[Optional[List[float]]] = []
        for row in rows:
            embed = row.get(key)
            if embed is None:
                values.append(None)
            else:
                array = _coerce_float_array(embed)
                if array is None:
                    values.append(None)
                else:
                    values.append([float(x) for x in array.tolist()])
        return values

    table = pa.table(
        {
            "loop_id": pa.array([row.get("loop_id") for row in rows], type=pa.string()),
            "model": pa.array([row.get("model") for row in rows], type=pa.string()),
            "audio_embed_clap": pa.array(
                _embed_column("audio_embed_clap"),
                type=pa.list_(pa.float32()),
            ),
            "audio_embed_mert": pa.array(
                _embed_column("audio_embed_mert"),
                type=pa.list_(pa.float32()),
            ),
            "text_embed": pa.array(
                _embed_column("text_embed"),
                type=pa.list_(pa.float32()),
            ),
        }
    )

    pq.write_table(table, out_path)


@dataclass
class AudioGuidanceRow:
    loop_id: Optional[str] = None
    file_digest: Optional[str] = None
    filename: Optional[str] = None
    text_audio_cos: Optional[float] = None
    text_audio_cos_mert: Optional[float] = None
    caption: Optional[str] = None
    caption_en: Optional[str] = None
    audio_path: Optional[str] = None
    render_path: Optional[str] = None
    embedding_path: Optional[str] = None
    audio_embed_clap: Optional[np.ndarray] = None
    audio_embed_mert: Optional[np.ndarray] = None
    text_embed: Optional[np.ndarray] = None
    audio_model: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def update_from(self, other: "AudioGuidanceRow") -> None:
        for field_name in (
            "loop_id",
            "file_digest",
            "filename",
            "text_audio_cos",
            "text_audio_cos_mert",
            "caption",
            "caption_en",
            "audio_path",
            "render_path",
            "embedding_path",
            "audio_embed_clap",
            "audio_embed_mert",
            "text_embed",
            "audio_model",
        ):
            value = getattr(other, field_name)
            if value is not None:
                setattr(self, field_name, value)
        if other.extra:
            self.extra.update(other.extra)

    def as_loop_row(self) -> Dict[str, Any]:
        payload: Dict[str, Optional[Any]] = dict(AUDIO_LOOP_DEFAULTS)
        payload["text_audio_cos"] = self.text_audio_cos
        payload["text_audio_cos_mert"] = self.text_audio_cos_mert
        payload["caption"] = self.caption
        payload["caption_en"] = self.caption_en or self.caption
        payload["audio.render_path"] = self.render_path or self.audio_path
        payload["audio.embedding_path"] = self.embedding_path
        return payload

    def as_retry_context(self) -> Dict[str, Any]:
        context: Dict[str, Any] = {}
        if self.text_audio_cos is not None:
            context["text_audio_cos"] = self.text_audio_cos
        if self.text_audio_cos_mert is not None:
            context["text_audio_cos_mert"] = self.text_audio_cos_mert
        if self.loop_id:
            context["loop_id"] = self.loop_id
        if self.file_digest:
            context["file_digest"] = self.file_digest
        if self.caption_en or self.caption:
            context["caption_en"] = self.caption_en or self.caption
        if self.caption:
            context["caption"] = self.caption
        if self.audio_path:
            context["audio_path"] = self.audio_path
        if self.render_path:
            context["render_path"] = self.render_path
        return context


@dataclass
class AudioGuidanceStore:
    rows: Dict[str, AudioGuidanceRow] = field(default_factory=dict)
    aliases: Dict[str, str] = field(default_factory=dict)

    def _register_alias(self, alias: Optional[str], key: str) -> None:
        if alias:
            self.aliases[alias] = key

    def _resolve_key(
        self,
        loop_id: Optional[str],
        file_digest: Optional[str],
        filename: Optional[str],
    ) -> Optional[str]:
        for candidate in (loop_id, file_digest, filename):
            if candidate and candidate in self.aliases:
                return self.aliases[candidate]
        for candidate in (loop_id, file_digest, filename):
            if candidate and candidate in self.rows:
                return candidate
        return None

    def merge(self, row: AudioGuidanceRow) -> None:
        canonical = self._resolve_key(
            row.loop_id,
            row.file_digest,
            row.filename,
        )
        if canonical is None:
            canonical = row.loop_id or row.file_digest or row.filename
            if canonical is None:
                return
        current = self.rows.get(canonical)
        if current is None:
            current = AudioGuidanceRow()
            self.rows[canonical] = current
        current.update_from(row)
        self._register_alias(current.loop_id, canonical)
        self._register_alias(current.file_digest, canonical)
        self._register_alias(current.filename, canonical)

    def get(
        self,
        loop_id: Optional[str],
        file_digest: Optional[str],
        filename: Optional[str],
    ) -> Optional[AudioGuidanceRow]:
        key = self._resolve_key(loop_id, file_digest, filename)
        if key is None:
            return None
        return self.rows.get(key)

    def has_rows(self) -> bool:
        return bool(self.rows)


def _resolve_optional_config_path(base_dir: Path, raw: Any) -> Optional[Path]:
    path_str = _as_optional_str(raw)
    if path_str is None:
        return None
    candidate = Path(path_str)
    if not candidate.is_absolute():
        candidate = (base_dir / candidate).resolve()
    return candidate


def _load_audio_alignment(
    store: AudioGuidanceStore,
    path: Path,
    config: Dict[str, Any],
) -> None:
    if not path.exists():
        return
    loop_field = _as_optional_str(config.get("loop_id_field")) or "loop_id"
    digest_field = _as_optional_str(config.get("file_digest_field")) or "file_digest"
    filename_field = _as_optional_str(config.get("filename_field")) or "filename"
    basename_field = _as_optional_str(config.get("basename_field")) or "basename"
    clap_field = _as_optional_str(config.get("clap_field")) or "text_audio_cos"
    clap_alt_field = _as_optional_str(config.get("clap_fallback_field")) or "cos_mean"
    mert_field = _as_optional_str(config.get("mert_field")) or "text_audio_cos_mert"
    caption_field = _as_optional_str(config.get("caption_field")) or "caption"
    caption_en_field = _as_optional_str(config.get("caption_en_field")) or "caption_en"
    audio_path_field = _as_optional_str(config.get("audio_path_field")) or "audio_path"
    render_path_field = _as_optional_str(config.get("render_path_field")) or "render_path"
    embedding_field = _as_optional_str(config.get("embedding_field")) or "embedding_path"

    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            payload = raw.strip()
            if not payload:
                continue
            try:
                data = json.loads(payload)
            except json.JSONDecodeError:
                continue
            loop_id = _as_optional_str(data.get(loop_field))
            file_digest = _as_optional_str(data.get(digest_field))
            filename = _as_optional_str(data.get(filename_field))
            if filename is None:
                filename = _as_optional_str(data.get(basename_field))
            filtered_keys = {loop_field, digest_field, filename_field, basename_field}
            row = AudioGuidanceRow(
                loop_id=loop_id,
                file_digest=file_digest,
                filename=filename,
                text_audio_cos=_to_optional_float(data.get(clap_field)),
                text_audio_cos_mert=_to_optional_float(data.get(mert_field)),
                caption=_as_optional_str(data.get(caption_field)),
                caption_en=_as_optional_str(data.get(caption_en_field)),
                audio_path=_as_optional_str(data.get(audio_path_field)),
                render_path=_as_optional_str(data.get(render_path_field)),
                embedding_path=_as_optional_str(data.get(embedding_field)),
                extra={k: v for k, v in data.items() if k not in filtered_keys},
            )
            if row.text_audio_cos is None and clap_alt_field:
                row.text_audio_cos = _to_optional_float(data.get(clap_alt_field))
            clap_embed = _coerce_float_array(
                data.get("audio_embed_clap")
                or data.get("audio_embedding_clap")
                or data.get("audio_embedding")
            )
            if clap_embed is not None:
                row.audio_embed_clap = clap_embed
            mert_embed = _coerce_float_array(data.get("audio_embed_mert"))
            if mert_embed is not None:
                row.audio_embed_mert = mert_embed
            text_embed = _coerce_float_array(data.get("text_embed") or data.get("text_embedding"))
            if text_embed is not None:
                row.text_embed = text_embed
            model_name = _as_optional_str(
                data.get("audio_model") or data.get("model") or data.get("guidance_model")
            )
            if model_name is not None:
                row.audio_model = model_name
            store.merge(row)


def _load_audio_captions(
    store: AudioGuidanceStore,
    path: Path,
    config: Dict[str, Any],
) -> None:
    if not path.exists():
        return
    caption_locale = _as_optional_str(config.get("caption_locale")) or "en"
    with path.open("r", encoding="utf-8") as handle:
        try:
            payload = json.load(handle)
        except json.JSONDecodeError:
            return
    if not isinstance(payload, dict):
        return
    for key, value in payload.items():
        text_default: Optional[str] = None
        text_en: Optional[str] = None
        if isinstance(value, str):
            text_en = value
        elif isinstance(value, dict):
            text_default = _as_optional_str(value.get("caption"))
            text_en = _as_optional_str(
                value.get("caption_en") or value.get(caption_locale) or text_default
            )
        elif value is not None:
            text_en = _as_optional_str(value)
        key_str = _as_optional_str(key)
        if key_str is None:
            continue
        row = AudioGuidanceRow(
            loop_id=key_str if len(key_str) == 32 else None,
            file_digest=key_str if len(key_str) == 32 else None,
            filename=key_str if key_str.endswith((".mid", ".midi", ".wav")) else None,
            caption=text_default,
            caption_en=text_en or text_default,
        )
        store.merge(row)


def _build_audio_guidance(
    config: Dict[str, Any],
    base_dir: Path,
) -> Optional[AudioGuidanceStore]:
    if not config:
        return None
    store = AudioGuidanceStore()
    alignment_path = _resolve_optional_config_path(base_dir, config.get("alignment_jsonl"))
    if alignment_path is not None:
        _load_audio_alignment(store, alignment_path, config)
    captions_path = _resolve_optional_config_path(base_dir, config.get("captions_json"))
    if captions_path is not None:
        _load_audio_captions(store, captions_path, config)
    return store if store.has_rows() else None


@dataclass
class RetryPresetRule:
    name: str
    when: Dict[str, str]
    action: Dict[str, Any]


@dataclass
class Stage2Settings:
    pipeline_version: str
    threshold: float
    axis_weights: Dict[str, float]
    retry_presets: Dict[str, List[RetryPresetRule]]
    metrics: MetricConfig
    paths: Stage2Paths
    limit: Optional[int]
    print_summary: bool
    articulation_thresholds: Optional[ArticulationThresholds]
    velocity_scoring: Optional[VelocityScoringConfig]
    structure_scoring: Optional[StructureScoringConfig]
    audio_adaptive_weights: Optional[AudioAdaptiveWeights] = None
    audio_guidance: Optional[AudioGuidanceStore] = None


def _primary_retry_key(reason: str) -> str:
    if "_" in reason:
        return reason.split("_", 1)[0]
    return reason


def _stringify_when(when_obj: Optional[Dict[str, Any]]) -> Dict[str, str]:
    if when_obj is None:
        return {}
    result: Dict[str, str] = {}
    for key, value in when_obj.items():
        result[str(key)] = str(value)
    return result


def _load_retry_presets(
    raw_retry: Sequence[Any],
) -> Dict[str, List[RetryPresetRule]]:
    grouped: Dict[str, List[RetryPresetRule]] = defaultdict(list)
    for item in raw_retry:
        if not isinstance(item, dict):
            continue
        entry = cast(Dict[str, Any], item)
        reason_obj = entry.get("reason")
        if not reason_obj:
            continue
        reason = str(reason_obj)
        axis_override = entry.get("axis")
        axis_key = str(axis_override) if axis_override else _primary_retry_key(reason)
        when_map = _stringify_when(cast(Optional[Dict[str, Any]], entry.get("when")))
        action = cast(Dict[str, Any], entry.get("action", {}))
        grouped[axis_key].append(
            RetryPresetRule(
                name=reason,
                when=when_map,
                action=action,
            )
        )
    return {key: value for key, value in grouped.items()}


def _resolve_condition_value(
    path: str,
    context: Dict[str, Any],
) -> Optional[float]:
    value: Any = context
    for part in path.split("."):
        if isinstance(value, dict) and part in value:
            value = value[part]
        else:
            return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _compare_numeric(value: float, operator: str, target: float) -> bool:
    if operator == "<":
        return value < target
    if operator == "<=":
        return value <= target
    if operator == ">":
        return value > target
    if operator == ">=":
        return value >= target
    if operator == "==":
        return value == target
    if operator == "!=":
        return value != target
    return False


def _conditions_met(
    conditions: Dict[str, str],
    context: Dict[str, Any],
) -> bool:
    if not conditions:
        return True
    for key, expression in conditions.items():
        value = _resolve_condition_value(key, context)
        if value is None:
            return False
        match = _CONDITION_PATTERN.match(expression)
        if match:
            operator, literal = match.groups()
            numeric = float(literal)
            if not _compare_numeric(value, operator, numeric):
                return False
            continue
        try:
            numeric = float(expression)
        except ValueError:
            return False
        if value != numeric:
            return False
    return True


def _select_retry_rule(
    reason_key: str,
    axes_raw: Dict[str, float],
    score_total: float,
    score_breakdown: Dict[str, float],
    rules: Dict[str, List[RetryPresetRule]],
    audio_context: Optional[Dict[str, Any]] = None,
) -> Optional[RetryPresetRule]:
    candidates = rules.get(reason_key, [])
    context: Dict[str, Any] = {
        "axes_raw": axes_raw,
        "score": score_total,
        "score_breakdown": score_breakdown,
        "reason": reason_key,
    }
    if audio_context:
        context["audio"] = audio_context
    for rule in candidates:
        if _conditions_met(rule.when, context):
            return rule
    fallback = rules.get("default")
    if fallback:
        for rule in fallback:
            if _conditions_met(rule.when, context):
                return rule
    return None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Stage 2 artefacts for LAMDa loops.",
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--metadata-index", type=Path)
    parser.add_argument("--metadata-dir", type=Path)
    parser.add_argument("--input-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--summary-out", type=Path)
    parser.add_argument("--print-summary", action="store_true")
    parser.add_argument("--sample-events", type=int)
    parser.add_argument("--articulation-thresholds", type=Path)
    return parser.parse_args()


def _load_config(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError("Stage 2 config must be a mapping")
    return cast(Dict[str, Any], raw)


def _load_axis_aliases(path: Optional[Path]) -> Dict[str, str]:
    if path is None or not path.exists():
        return {}
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError("Axis alias config must be a mapping")
    payload = cast(Dict[str, Any], raw)
    aliases_root = payload.get("aliases", {})
    if not isinstance(aliases_root, dict):
        raise ValueError("aliases section must be a mapping")

    lookup: Dict[str, str] = {}
    for canonical_raw, aliases in cast(Dict[Any, Any], aliases_root).items():
        canonical = str(canonical_raw)
        lookup.setdefault(canonical, canonical)
        if isinstance(aliases, (list, tuple)):
            alias_entries = cast(Sequence[Any], aliases)
            for entry in alias_entries:
                alias_key = str(entry)
                lookup.setdefault(alias_key, canonical)
    return lookup


def _apply_axis_aliases(
    config: Dict[str, Any],
    aliases: Dict[str, str],
) -> Dict[str, Any]:
    if not aliases:
        return config
    result: Dict[str, Any] = {}
    for key, value in config.items():
        canonical = aliases.get(str(key), str(key))
        result[canonical] = value
    return result


def _safe_normalised_histogram(
    values: Sequence[float],
    *,
    nbins: int,
) -> FloatArray:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size != nbins:
        raise ValueError(
            f"velocity histogram expected {nbins} entries, found {array.size}",
        )
    total = float(array.sum())
    if total > 0:
        array = array / total
    else:
        array = np.full(nbins, 1.0 / nbins, dtype=np.float64)
    return array


def _parse_range_phase_config(
    payload: Any,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    dynamic: Dict[str, float] = {}
    phase: Dict[str, float] = {}
    if not isinstance(payload, dict):
        return dynamic, phase
    dynamic_payload = payload.get("dynamic_range")
    if isinstance(dynamic_payload, dict):
        for key, value in dynamic_payload.items():
            try:
                dynamic[str(key)] = float(value)
            except (TypeError, ValueError):
                continue
    if not dynamic:
        global_obj = payload.get("global")
        if isinstance(global_obj, dict):
            for key in ("min", "max"):
                candidate = global_obj.get(key)
                try:
                    if candidate is not None:
                        dynamic[key] = float(candidate)
                except (TypeError, ValueError):
                    continue
        for key in ("min", "max"):
            if key in dynamic:
                continue
            candidate = payload.get(key)
            try:
                if candidate is not None:
                    dynamic[key] = float(candidate)
            except (TypeError, ValueError):
                continue
    phase_payload = payload.get("phase_adjust_db")
    if isinstance(phase_payload, dict):
        for key, value in phase_payload.items():
            try:
                phase[str(key)] = float(value)
            except (TypeError, ValueError):
                continue
    for key, value in payload.items():
        if key in {"global", "dynamic_range", "phase_adjust_db"}:
            continue
        if isinstance(value, (int, float)) and str(key).endswith("_db"):
            phase[str(key)] = float(value)
    return dynamic, phase


def _resolve_config_path(base_dir: Path, value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return path


def _load_velocity_scoring_config(
    velocity_cfg: Dict[str, Any],
    *,
    config_dir: Path,
) -> Optional[VelocityScoringConfig]:
    if not velocity_cfg:
        return None
    target_path_value = velocity_cfg.get("target_histograms_path")
    if not target_path_value:
        return None
    target_path = _resolve_config_path(config_dir, str(target_path_value))
    if not target_path.exists():
        raise FileNotFoundError(
            f"Velocity target histogram file not found: {target_path}",
        )
    target_text = target_path.read_text(encoding="utf-8")
    raw_loaded: Any = yaml.safe_load(target_text) or {}
    if not isinstance(raw_loaded, dict):
        raise ValueError(
            f"Velocity targets file {target_path} must contain a mapping",
        )
    raw = cast(Dict[str, Any], raw_loaded)
    nbins = int(velocity_cfg.get("nbins") or raw.get("nbins") or 16)
    targets_root = cast(Dict[str, Any], raw.get("targets", raw))
    targets: Dict[str, FloatArray] = {}
    for key, values in targets_root.items():
        if not isinstance(values, Sequence):
            continue
        try:
            sequence_values = [float(item) for item in values]
        except (TypeError, ValueError):
            continue
        targets[key] = _safe_normalised_histogram(sequence_values, nbins=nbins)
    if "global" not in targets:
        message = "Velocity targets file " f"{target_path} missing 'global' histogram"
        raise ValueError(message)
    for key in ("downbeat", "offbeat", "prefill"):
        if key not in targets:
            targets[key] = targets["global"]

    default_dynamic, default_phase = _parse_range_phase_config(
        raw.get("range_factors"),
    )

    tempo_target_entries: List[Tuple[Optional[float], Dict[str, FloatArray]]] = []
    tempo_dynamic_entries: List[Tuple[Optional[float], Dict[str, float]]] = []
    tempo_phase_entries: List[Tuple[Optional[float], Dict[str, float]]] = []
    tempo_targets_root = raw.get("tempo_bins")
    if tempo_targets_root is None:
        tempo_targets_root = targets_root.get("tempo_bins")
    if isinstance(tempo_targets_root, Sequence):
        tempo_targets_seq = cast(Sequence[Any], tempo_targets_root)
        for entry in tempo_targets_seq:
            if not isinstance(entry, dict):
                continue
            entry_dict = cast(Dict[str, Any], entry)
            max_bpm_raw = entry_dict.get("max_bpm")
            max_bpm: Optional[float]
            if max_bpm_raw is None:
                max_bpm = None
            else:
                try:
                    max_bpm = float(max_bpm_raw)
                except (TypeError, ValueError):
                    max_bpm = None
            targets_obj = entry_dict.get("targets")
            if isinstance(targets_obj, dict):
                targets_obj = cast(Dict[str, Any], targets_obj)
            else:
                targets_obj = {
                    key: entry_dict.get(key, entry_dict.get(f"hist_{key}"))
                    for key in ("global", "downbeat", "offbeat", "prefill")
                }
            bin_targets: Dict[str, FloatArray] = {}
            for target_key in ("global", "downbeat", "offbeat", "prefill"):
                values_obj = targets_obj.get(target_key)
                if not isinstance(values_obj, Sequence):
                    continue
                try:
                    sequence_values = [float(item) for item in values_obj]
                except (TypeError, ValueError):
                    continue
                bin_targets[target_key] = _safe_normalised_histogram(
                    sequence_values,
                    nbins=nbins,
                )
            if not bin_targets:
                continue
            if "global" not in bin_targets:
                bin_targets["global"] = targets["global"].copy()
            for target_key in ("downbeat", "offbeat", "prefill"):
                if target_key not in bin_targets:
                    bin_targets[target_key] = targets.get(
                        target_key,
                        targets["global"],
                    ).copy()
            tempo_target_entries.append((max_bpm, bin_targets))

            range_payload = entry_dict.get("range_factors")
            if range_payload is not None:
                dyn_entry, phase_entry = _parse_range_phase_config(range_payload)
                if dyn_entry:
                    tempo_dynamic_entries.append((max_bpm, dyn_entry))
                if phase_entry:
                    tempo_phase_entries.append((max_bpm, phase_entry))
    weights_root = cast(Dict[str, Any], velocity_cfg.get("weights", {}))
    weights = {
        "global": float(weights_root.get("global", 0.4)),
        "downbeat": float(weights_root.get("downbeat", 0.25)),
        "offbeat": float(weights_root.get("offbeat", 0.2)),
        "prefill": float(weights_root.get("prefill", 0.15)),
    }
    metric_weights_root = cast(
        Dict[str, Any],
        velocity_cfg.get("metric_weights", {}),
    )
    metric_weights = {
        "js": float(metric_weights_root.get("js", 0.5)),
        "cos": float(metric_weights_root.get("cos", 0.5)),
    }
    prefill_window = float(velocity_cfg.get("prefill_window_beats", 0.5))
    tempo_bins: List[Dict[str, Any]] = []
    tempo_root = velocity_cfg.get("tempo_adaptive")
    if isinstance(tempo_root, dict):
        tempo_root_dict = cast(Dict[str, Any], tempo_root)
        bins_raw = tempo_root_dict.get("bins")
        if isinstance(bins_raw, Sequence):
            bins_sequence = cast(Sequence[Any], bins_raw)
            for entry in bins_sequence:
                if not isinstance(entry, dict):
                    continue
                entry_dict = cast(Dict[str, Any], entry)
                max_bpm_raw = entry_dict.get("max_bpm")
                max_bpm = None
                if max_bpm_raw is not None:
                    max_bpm = float(max_bpm_raw)
                tempo_bins.append(
                    {
                        "max_bpm": max_bpm,
                        "alpha": float(entry_dict.get("alpha", 1.0)),
                        "downbeat_boost": float(
                            entry_dict.get("downbeat_boost", 0.0),
                        ),
                    },
                )

    def _tempo_bin_key(spec: Dict[str, Any]) -> float:
        value = spec.get("max_bpm")
        if value is None:
            return float("inf")
        return float(value)

    def _tempo_target_key(entry: Tuple[Optional[float], Any]) -> float:
        max_bpm = entry[0]
        if max_bpm is None:
            return float("inf")
        return float(max_bpm)

    tempo_bins.sort(key=_tempo_bin_key)
    tempo_target_entries.sort(key=_tempo_target_key)
    tempo_dynamic_entries.sort(key=_tempo_target_key)
    tempo_phase_entries.sort(key=_tempo_target_key)

    normalize_dynamic = bool(velocity_cfg.get("normalize_dynamic_range", False))
    phase_comp = bool(velocity_cfg.get("phase_compensation", False))
    dynamic_range = dict(default_dynamic)
    phase_adjust_db = dict(default_phase)

    return VelocityScoringConfig(
        nbins=nbins,
        targets=targets,
        weights=weights,
        metric_weights=metric_weights,
        prefill_window_beats=prefill_window,
        tempo_bins=tuple(tempo_bins),
        tempo_bin_targets=tuple(tempo_target_entries),
        normalize_dynamic_range=normalize_dynamic,
        dynamic_range=dynamic_range,
        tempo_dynamic_ranges=tuple(tempo_dynamic_entries),
        phase_compensation=phase_comp,
        phase_adjust_db=phase_adjust_db,
        tempo_phase_adjust=tuple(tempo_phase_entries),
    )


def _load_structure_scoring_config(
    structure_cfg: Dict[str, Any],
) -> Optional[StructureScoringConfig]:
    if not structure_cfg:
        return None
    bands_config = structure_cfg.get("density_bands")
    bands: List[DensityBandSpec] = []
    if isinstance(bands_config, list):
        bands_list = cast(List[Any], bands_config)
        for entry_obj in bands_list:
            if not isinstance(entry_obj, dict):
                continue
            entry_dict = cast(Dict[str, Any], entry_obj)
            max_bpm_val = entry_dict.get("max_bpm")
            max_bpm = float(max_bpm_val) if max_bpm_val is not None else None
            low = float(entry_dict.get("low", 0.0))
            high = float(entry_dict.get("high", low))
            bands.append(DensityBandSpec(max_bpm=max_bpm, low=low, high=high))
    elif isinstance(bands_config, dict):
        bands_dict = cast(Dict[Any, Any], bands_config)
        for entry_obj in bands_dict.values():
            if not isinstance(entry_obj, dict):
                continue
            entry_dict = cast(Dict[str, Any], entry_obj)
            max_bpm_val = entry_dict.get("max_bpm")
            max_bpm = float(max_bpm_val) if max_bpm_val is not None else None
            low = float(entry_dict.get("low", 0.0))
            high = float(entry_dict.get("high", low))
            bands.append(DensityBandSpec(max_bpm=max_bpm, low=low, high=high))
    if not bands:
        bands = [
            DensityBandSpec(max_bpm=95.0, low=2.0, high=6.0),
            DensityBandSpec(max_bpm=130.0, low=4.0, high=9.0),
            DensityBandSpec(max_bpm=None, low=6.0, high=12.0),
        ]

    def _density_band_key(spec: DensityBandSpec) -> float:
        if spec.max_bpm is None:
            return float("inf")
        return float(spec.max_bpm)

    bands.sort(key=_density_band_key)
    weights_root = cast(Dict[str, Any], structure_cfg.get("weights", {}))
    weights = {
        "periodicity": float(weights_root.get("periodicity", 0.45)),
        "diversity": float(weights_root.get("diversity", 0.35)),
        "density": float(weights_root.get("density", 0.2)),
        "role_stability": float(weights_root.get("role_stability", 0.0)),
    }
    grid_divisions = int(structure_cfg.get("grid_divisions", 16))
    periodicity_candidates_raw = structure_cfg.get("periodicity_candidates")
    if isinstance(periodicity_candidates_raw, Sequence):
        periodicity_sequence = cast(Sequence[Any], periodicity_candidates_raw)
        periodicity_candidates = tuple(
            int(value_obj)
            for value_obj in periodicity_sequence
            if isinstance(value_obj, (int, float))
        )
        if not periodicity_candidates:
            periodicity_candidates = (4, 8, 12, 16)
    else:
        periodicity_candidates = (4, 8, 12, 16)
    return StructureScoringConfig(
        bands=bands,
        weights=weights,
        grid_divisions=grid_divisions,
        periodicity_candidates=periodicity_candidates,
    )


def _load_audio_adaptive_axis_caps(
    axes_cfg: Optional[Any],
) -> Dict[str, AudioAdaptiveAxisCap]:
    if not isinstance(axes_cfg, dict):
        return {}
    caps: Dict[str, AudioAdaptiveAxisCap] = {}
    for axis_key, raw_cfg in axes_cfg.items():
        if not isinstance(raw_cfg, Mapping):
            continue
        axis_map = cast(Mapping[str, Any], raw_cfg)
        min_scale = _to_optional_float(axis_map.get("min_scale"))
        max_scale = _to_optional_float(axis_map.get("max_scale"))
        caps[str(axis_key)] = AudioAdaptiveAxisCap(
            min_scale=min_scale,
            max_scale=max_scale,
        )
    return caps


def _load_audio_adaptive_fusion(
    fusion_cfg: Mapping[str, Any],
) -> Optional[AudioAdaptiveFusion]:
    sources_cfg = fusion_cfg.get("sources")
    sources: List[AudioAdaptiveFusionSource] = []

    if isinstance(sources_cfg, Sequence):
        for entry in cast(Sequence[Any], sources_cfg):
            if not isinstance(entry, Mapping):
                continue
            entry_map = cast(Mapping[str, Any], entry)
            path_text = _as_optional_str(entry_map.get("path"))
            if not path_text:
                continue
            path_parts = _split_path(path_text)
            if not path_parts:
                continue
            weight_value = _to_optional_float(entry_map.get("weight"))
            if weight_value is None:
                continue
            required = bool(entry_map.get("required", False))
            bias = _to_optional_float(entry_map.get("bias")) or 0.0
            sources.append(
                AudioAdaptiveFusionSource(
                    path=path_parts,
                    weight=float(weight_value),
                    required=required,
                    bias=bias,
                )
            )
    else:
        clap_path_text = _as_optional_str(fusion_cfg.get("clap_path")) or "audio.text_audio_cos"
        mert_path_text = (
            _as_optional_str(fusion_cfg.get("mert_path")) or "metrics.text_audio_cos_mert"
        )
        clap_parts = _split_path(clap_path_text) if clap_path_text else ()
        mert_parts = _split_path(mert_path_text) if mert_path_text else ()
        clap_weight = _to_optional_float(fusion_cfg.get("clap_weight"))
        mert_weight = _to_optional_float(fusion_cfg.get("mert_weight"))
        default_weight = 0.5
        if clap_parts:
            sources.append(
                AudioAdaptiveFusionSource(
                    path=clap_parts,
                    weight=float(clap_weight if clap_weight is not None else default_weight),
                )
            )
        if mert_parts:
            sources.append(
                AudioAdaptiveFusionSource(
                    path=mert_parts,
                    weight=float(mert_weight if mert_weight is not None else default_weight),
                )
            )

    if not sources:
        return None

    normalise_weights = bool(fusion_cfg.get("normalise_weights", True))
    bias = _to_optional_float(fusion_cfg.get("bias")) or 0.0
    default_value = _to_optional_float(fusion_cfg.get("default"))
    clamp_min = _to_optional_float(fusion_cfg.get("clamp_min"))
    clamp_max = _to_optional_float(fusion_cfg.get("clamp_max"))
    temperature = _to_optional_float(fusion_cfg.get("temperature"))

    return AudioAdaptiveFusion(
        sources=tuple(sources),
        bias=bias,
        normalise_weights=normalise_weights,
        default=default_value,
        clamp_min=clamp_min,
        clamp_max=clamp_max,
        temperature=temperature,
    )


def _load_audio_adaptive_weights(
    score_cfg: Dict[str, Any],
) -> Optional[AudioAdaptiveWeights]:
    adaptive_root = cast(Dict[str, Any], score_cfg.get("audio_adaptive_weights", {}))
    if not adaptive_root:
        return None

    pivot_text = _as_optional_str(adaptive_root.get("pivot"))
    if not pivot_text:
        return None
    pivot_parts = _split_path(pivot_text)
    if not pivot_parts:
        return None

    rules_raw = adaptive_root.get("rules")
    if not isinstance(rules_raw, list):
        return None

    def _ensure_rule_name(entry: Dict[str, Any], current_index: int) -> str:
        text = _as_optional_str(entry.get("name") or entry.get("id"))
        if text:
            return text
        return f"rule_{current_index}"

    rules: List[AudioAdaptiveRule] = []
    for item in cast(List[Any], rules_raw):
        if not isinstance(item, dict):
            continue
        entry = cast(Dict[str, Any], item)
        condition_text = _as_optional_str(entry.get("if"))
        if not condition_text:
            continue
        condition = _parse_threshold_expression(condition_text)
        if condition is None:
            continue
        operator, threshold = condition
        multipliers_obj: Any = entry.get("axis_multipliers")
        if not isinstance(multipliers_obj, dict):
            multipliers_obj = entry.get("multipliers")
        if not isinstance(multipliers_obj, dict):
            continue
        multipliers: Dict[str, float] = {}
        for axis_key, value in cast(Dict[Any, Any], multipliers_obj).items():
            try:
                multipliers[str(axis_key)] = float(value)
            except (TypeError, ValueError):
                continue
        if not multipliers:
            continue
        name = _ensure_rule_name(entry, len(rules))
        extras_obj = entry.get("extras")
        extras: Dict[str, Any]
        if isinstance(extras_obj, Mapping):
            extras = {str(key): value for key, value in extras_obj.items()}
        else:
            extras = {}
        priority_value = entry.get("priority")
        try:
            priority = int(priority_value) if priority_value is not None else 0
        except (TypeError, ValueError):
            priority = 0
        rules.append(
            AudioAdaptiveRule(
                operator=operator,
                threshold=threshold,
                multipliers=multipliers,
                name=name,
                priority=priority,
                extras=extras,
            )
        )

    if not rules:
        return None

    enabled = bool(adaptive_root.get("enabled", True))

    min_confidence = _to_optional_float(adaptive_root.get("min_confidence"))
    missing_policy = _as_optional_str(adaptive_root.get("missing_policy")) or "noop"
    missing_policy = missing_policy.lower()
    if missing_policy not in {"noop", "zero", "last"}:
        missing_policy = "noop"

    fusion_cfg = cast(Dict[str, Any], adaptive_root.get("fusion", {}))
    fusion = _load_audio_adaptive_fusion(fusion_cfg) if fusion_cfg else None

    caps_cfg = cast(Dict[str, Any], adaptive_root.get("caps", {}))
    min_scale = _to_optional_float(caps_cfg.get("min_scale")) if caps_cfg else None
    max_scale = _to_optional_float(caps_cfg.get("max_scale")) if caps_cfg else None
    axis_caps = _load_audio_adaptive_axis_caps(caps_cfg.get("axes")) if caps_cfg else {}

    normalize_cfg = adaptive_root.get("normalize")
    normalize_sum = False
    normalize_target: Optional[float] = None
    if isinstance(normalize_cfg, dict):
        normalize_sum = bool(normalize_cfg.get("enabled", True))
        normalize_target = _to_optional_float(normalize_cfg.get("target_sum"))
    elif normalize_cfg:
        normalize_sum = True

    hysteresis_margin = _to_optional_float(adaptive_root.get("hysteresis_margin")) or 0.0
    cooldown_loops = int(adaptive_root.get("cooldown_loops", 0))
    cooldown_by_rule_cfg = adaptive_root.get("cooldown_by_rule", {})
    cooldown_by_rule: Dict[str, int] = {}
    if isinstance(cooldown_by_rule_cfg, Mapping):
        for key, value in cooldown_by_rule_cfg.items():
            try:
                loops = int(value)
            except (TypeError, ValueError):
                continue
            if loops > 0:
                cooldown_by_rule[str(key)] = loops

    max_total_delta = _to_optional_float(adaptive_root.get("max_total_delta"))

    # Load max_total_delta_per_axis
    max_total_delta_per_axis: Dict[str, float] = {}
    per_axis_raw = adaptive_root.get("max_total_delta_per_axis")
    if isinstance(per_axis_raw, dict):
        for axis_key, limit_val in per_axis_raw.items():
            limit_f = _to_optional_float(limit_val)
            if limit_f is not None and limit_f >= 0.0:
                max_total_delta_per_axis[str(axis_key)] = limit_f

    pivot_ema_alpha = _to_optional_float(adaptive_root.get("pivot_ema_alpha"))
    log_level_text = _as_optional_str(adaptive_root.get("log_level")) or "summary"
    log_level = log_level_text.lower()
    if log_level not in {"summary", "debug"}:
        log_level = "summary"

    return AudioAdaptiveWeights(
        pivot_path=pivot_parts,
        rules=tuple(rules),
        enabled=enabled,
        min_confidence=min_confidence,
        missing_policy=missing_policy,
        fusion=fusion,
        min_scale=min_scale,
        max_scale=max_scale,
        axis_caps=axis_caps,
        normalize_sum=normalize_sum,
        normalize_target=normalize_target,
        hysteresis_margin=float(hysteresis_margin),
        cooldown_loops=max(0, cooldown_loops),
        cooldown_by_rule=cooldown_by_rule,
        max_total_delta=max_total_delta,
        max_total_delta_per_axis=max_total_delta_per_axis,
        pivot_ema_alpha=pivot_ema_alpha,
        log_level=log_level,
    )


def _resolve_paths(
    config: Dict[str, Any],
    args: argparse.Namespace,
) -> Stage2Paths:
    paths_cfg = cast(Dict[str, Any], config.get("paths", {}))
    base = Path(paths_cfg.get("output_dir", "output/drumloops_stage2"))
    if args.output_dir is not None:
        base = args.output_dir
    base.mkdir(parents=True, exist_ok=True)

    def _resolve(key: str, default: Optional[str]) -> Optional[Path]:
        override = getattr(args, key, None)
        if override is not None:
            return Path(override)
        value = paths_cfg.get(key, default)
        if value is None:
            return None
        path = Path(value)
        if not path.is_absolute():
            path = base / path
        return path

    events_parquet = _resolve("events_parquet", "canonical_events.parquet")
    events_csv_sample = _resolve(
        "events_csv_sample",
        "canonical_events_sample.csv",
    )
    loop_summary_csv = _resolve("loop_summary_csv", "loop_summary.csv")
    metrics_jsonl = _resolve("metrics_jsonl", "metrics_score.jsonl")
    retry_dir = _resolve("retry_dir", "retries")
    summary_out = args.summary_out or _resolve(
        "summary_out",
        "stage2_summary.json",
    )
    audio_embeddings_parquet = _resolve(
        "audio_embeddings_parquet",
        "audio_embeddings.parquet",
    )

    sample_rows = int(paths_cfg.get("sample_event_rows", 5000))
    if args.sample_events is not None:
        sample_rows = max(0, args.sample_events)

    missing_paths_msg = (
        "Config must define events_parquet, loop_summary_csv, " "metrics_jsonl, and retry_dir paths"
    )
    if (
        events_parquet is None
        or loop_summary_csv is None
        or metrics_jsonl is None
        or retry_dir is None
    ):
        raise ValueError(missing_paths_msg)
    if summary_out:
        summary_out.parent.mkdir(parents=True, exist_ok=True)
    if events_csv_sample:
        events_csv_sample.parent.mkdir(parents=True, exist_ok=True)
    if audio_embeddings_parquet:
        audio_embeddings_parquet.parent.mkdir(parents=True, exist_ok=True)

    return Stage2Paths(
        events_parquet=events_parquet,
        events_csv_sample=events_csv_sample,
        loop_summary_csv=loop_summary_csv,
        metrics_jsonl=metrics_jsonl,
        retry_dir=retry_dir,
        summary_out=summary_out,
        sample_event_rows=sample_rows,
        audio_embeddings_parquet=audio_embeddings_parquet,
    )


def _build_settings(
    config: Dict[str, Any],
    args: argparse.Namespace,
) -> Stage2Settings:
    if getattr(args, "config", None):
        config_dir = args.config.resolve().parent
    else:
        config_dir = Path(".")
    pipeline_cfg = cast(Dict[str, Any], config.get("pipeline", {}))
    version = str(pipeline_cfg.get("version", "stage2"))
    threshold = float(pipeline_cfg.get("threshold", 70.0))
    if args.threshold is not None:
        threshold = args.threshold

    score_cfg = cast(Dict[str, Any], config.get("score", {}))
    axis_cfg_raw = cast(Dict[str, Any], score_cfg.get("axes", {}))
    alias_path_cfg = score_cfg.get("aliases_path")
    alias_path: Optional[Path]
    if alias_path_cfg is None:
        alias_path = DEFAULT_AXIS_ALIAS_PATH
    else:
        alias_path = Path(alias_path_cfg)
        if not alias_path.is_absolute():
            alias_path = (args.config.parent / alias_path).resolve()
    alias_lookup = BUILTIN_AXIS_ALIASES.copy()
    alias_lookup.update(_load_axis_aliases(alias_path))
    axis_cfg = _apply_axis_aliases(axis_cfg_raw, alias_lookup)
    axis_weights = DEFAULT_AXIS_WEIGHTS.copy()
    for raw_key, value in axis_cfg.items():
        axis_key = str(raw_key)
        axis_weights[axis_key] = float(value)

    velocity_scoring = _load_velocity_scoring_config(
        cast(Dict[str, Any], score_cfg.get("velocity", {})),
        config_dir=config_dir,
    )
    structure_scoring = _load_structure_scoring_config(
        cast(Dict[str, Any], score_cfg.get("structure", {})),
    )
    audio_adaptive_weights = _load_audio_adaptive_weights(score_cfg)

    audio_cfg = cast(Dict[str, Any], config.get("audio", {}))
    audio_guidance = _build_audio_guidance(audio_cfg, config_dir)

    raw_retry = cast(List[Any], config.get("retry_presets", []))
    retry_map = _load_retry_presets(raw_retry)

    metric_kwargs = cast(Dict[str, Any], config.get("metrics", {}))
    metrics_cfg = MetricConfig(**metric_kwargs)

    articulation_cfg = cast(Dict[str, Any], config.get("articulation", {}))
    thresholds_path_cfg = articulation_cfg.get("thresholds_path")
    thresholds_path: Optional[Path]
    if thresholds_path_cfg is not None:
        thresholds_path = Path(thresholds_path_cfg)
    else:
        thresholds_path = DEFAULT_ARTICULATION_THRESHOLDS_PATH
    if args.articulation_thresholds is not None:
        thresholds_path = args.articulation_thresholds

    articulation_thresholds: Optional[ArticulationThresholds] = None
    if thresholds_path and thresholds_path.exists():
        articulation_thresholds = ArticulationThresholds.load(thresholds_path)
        axis_override = axis_cfg.get("articulation")
        if axis_override is not None:
            axis_weights["articulation"] = float(axis_override)
        else:
            total_weight = articulation_thresholds.total_weight
            if total_weight > 0:
                axis_weights["articulation"] = total_weight
    elif "articulation" not in axis_weights:
        axis_weights["articulation"] = 0.0

    paths = _resolve_paths(config, args)

    return Stage2Settings(
        pipeline_version=version,
        threshold=threshold,
        axis_weights=axis_weights,
        retry_presets=retry_map,
        metrics=metrics_cfg,
        paths=paths,
        limit=args.limit,
        print_summary=args.print_summary,
        articulation_thresholds=articulation_thresholds,
        velocity_scoring=velocity_scoring,
        structure_scoring=structure_scoring,
        audio_adaptive_weights=audio_adaptive_weights,
        audio_guidance=audio_guidance,
    )


def _import_tmidix(path: Path) -> Any:
    resolved = path.resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"TMIDIX module not found: {resolved}")
    if str(resolved) not in sys.path:
        sys.path.append(str(resolved))
    return importlib.import_module("TMIDIX")


def _infer_role(channel: int, program: int, pitch: int) -> Tuple[str, float]:
    channel_is_drum = channel == 9  # MIDI channel 10 (0-based)
    category = None
    if pitch in DRUM_CATEGORIES.get("kick", set()):
        category = "kick"
    else:
        for name, pitches in DRUM_CATEGORIES.items():
            if pitch in pitches:
                category = name
                break

    percussion_hint = 112 <= program <= 119

    confidence = 0.2
    if category is not None:
        confidence += 0.5
    if channel_is_drum:
        confidence += 0.3
        if category is None:
            category = "perc"
    if percussion_hint:
        confidence += 0.1
        if category is None:
            category = "perc"

    if category is None:
        category = "other"
    return category, _clip(confidence, 0.0, 1.0)


def _flatten_events(tracks: Sequence[Sequence[Any]]) -> List[List[Any]]:
    events: List[List[Any]] = []
    for track in tracks:
        for event in track:
            if isinstance(event, list):
                events.append(cast(List[Any], event))
    return events


def _tempo_events(events: Sequence[Sequence[Any]]) -> List[int]:
    tempos: List[int] = []
    for event in events:
        if len(event) >= 3 and event[0] == "set_tempo":
            try:
                tempos.append(int(event[2]))
            except (TypeError, ValueError):
                continue
    return tempos


def _time_signature(events: Sequence[Sequence[Any]]) -> Tuple[int, int]:
    for event in events:
        if len(event) >= 4 and event[0] == "time_signature":
            numerator = int(event[2])
            denominator = 2 ** int(event[3])
            if numerator > 0 and denominator > 0:
                return numerator, denominator
    return 4, 4


def _channel_programs(events: Sequence[Sequence[Any]]) -> Dict[int, int]:
    mapping: Dict[int, int] = {}
    for event in events:
        if len(event) >= 4 and event[0] == "patch_change":
            channel = int(event[2])
            program = int(event[3])
            mapping[channel] = program
    return mapping


def _ioi_bucket(ioi: Optional[int], ticks_per_beat: int) -> Optional[str]:
    if ioi is None or ticks_per_beat <= 0:
        return None
    reference = {
        "quarter": float(ticks_per_beat),
        "eighth": ticks_per_beat / 2.0,
        "triplet": ticks_per_beat / 3.0,
        "sixteenth": ticks_per_beat / 4.0,
    }
    best_label: Optional[str] = None
    best_error: Optional[float] = None
    for label, ref in reference.items():
        if ref <= 0:
            continue
        error = abs(ioi - ref) / ref
        if best_error is None or error < best_error:
            best_error = error
            best_label = label
    if best_error is None or best_error > 0.25:
        return "other"
    return best_label


def _event_rows(
    loop_id: str,
    notes: Sequence[Sequence[Any]],
    ticks_per_bar: float,
    beat_ticks: float,
    base_step: Optional[float],
    channels: Dict[int, int],
    ghost_threshold: int,
    ticks_per_beat: int,
    source: str,
    pipeline_version: str,
    file_digest: Optional[str],
) -> Tuple[List[Dict[str, Any]], List[int]]:
    rows: List[Dict[str, Any]] = []
    programs: List[int] = []
    previous_start: Optional[int] = None
    for note in notes:
        start = int(note[1])
        duration = int(note[2])
        channel = int(note[3])
        pitch = int(note[4])
        velocity = int(note[5])

        if ticks_per_bar:
            bar_index = int(start // ticks_per_bar)
            within_bar = start - bar_index * ticks_per_bar
        else:
            bar_index = 0
            within_bar = 0.0
        beat_float = within_bar / beat_ticks if beat_ticks else 0.0
        beat_index = int(math.floor(beat_float))
        beat_frac = beat_float - beat_index

        grid_onset = start
        microtiming_offset = 0.0
        swing_phase: Optional[str] = None
        if base_step:
            grid_index = round(start / base_step)
            grid_onset = int(round(grid_index * base_step))
            microtiming_offset = float(start - grid_onset)
            swing_phase = "even" if grid_index % 2 == 0 else "odd"

        ioi = start - previous_start if previous_start is not None else None

        program = channels.get(channel, 0)
        program_norm = program % 128
        programs.append(program)

        role, role_confidence = _infer_role(channel, program, pitch)

        rows.append(
            {
                "loop_id": loop_id,
                "source": source,
                "pipeline_version": pipeline_version,
                "bar_index": bar_index,
                "beat_index": beat_index,
                "beat_frac": beat_frac,
                "grid_onset": grid_onset,
                "onset_ticks": start,
                "duration_ticks": duration,
                "channel": channel,
                "program": program,
                "program_norm": program_norm,
                "pitch": pitch,
                "velocity": velocity,
                "intensity": velocity / 127.0,
                "is_ghost": velocity <= ghost_threshold,
                "swing_phase": swing_phase,
                "microtiming_offset": microtiming_offset,
                "ioi_bucket": _ioi_bucket(ioi, ticks_per_beat),
                "instrument_role": role,
                "role_confidence": role_confidence,
                "file_digest": file_digest or loop_id,
            }
        )
        previous_start = start
    return rows, programs


def _normalise_histogram(array: FloatArray) -> FloatArray:
    total = float(array.sum())
    if total <= 0:
        return np.full_like(
            array,
            1.0 / max(1, array.size),
            dtype=np.float64,
        )
    return (array / total).astype(np.float64)


def _jensen_shannon(
    p: FloatArray,
    q: FloatArray,
    eps: float = 1e-9,
) -> float:
    p_safe = np.clip(p, eps, 1.0)
    q_safe = np.clip(q, eps, 1.0)
    p_norm = _normalise_histogram(p_safe)
    q_norm = _normalise_histogram(q_safe)
    m = 0.5 * (p_norm + q_norm)
    kl_pm = float(np.sum(p_norm * (np.log(p_norm) - np.log(m))))
    kl_qm = float(np.sum(q_norm * (np.log(q_norm) - np.log(m))))
    js = 0.5 * (kl_pm + kl_qm)
    return float(js)


def _cosine_similarity(
    p: FloatArray,
    q: FloatArray,
    eps: float = 1e-9,
) -> float:
    numerator = float(np.dot(p, q))
    denominator = float(np.linalg.norm(p) * np.linalg.norm(q)) + eps
    return numerator / denominator if denominator > 0 else 0.0


def _softmax(values: FloatArray) -> FloatArray:
    if values.size == 0:
        return values
    max_value = float(np.max(values))
    shifted = values - max_value
    exps = np.exp(shifted)
    total = float(np.sum(exps))
    if total <= 0:
        return np.full_like(
            values,
            1.0 / max(1, values.size),
            dtype=np.float64,
        )
    return (exps / total).astype(np.float64)


def _pick_tempo_bin(
    bpm: Optional[float],
    bins: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    if not bins:
        return {"alpha": 1.0, "downbeat_boost": 0.0}
    if bpm is None:
        return bins[0]
    for entry in bins:
        max_bpm = entry.get("max_bpm")
        if max_bpm is None or bpm <= float(max_bpm):
            return entry
    return bins[-1]


def _pick_tempo_mapping(
    bpm: Optional[float],
    specs: Tuple[Tuple[Optional[float], Dict[str, float]], ...],
) -> Dict[str, float]:
    if not specs:
        return {}
    if bpm is None:
        return specs[0][1]
    for max_bpm, mapping in specs:
        if max_bpm is None or bpm <= max_bpm:
            return mapping
    return specs[-1][1]


def _pick_tempo_target_overrides(
    bpm: Optional[float],
    specs: Tuple[Tuple[Optional[float], Dict[str, FloatArray]], ...],
) -> Optional[Dict[str, FloatArray]]:
    if not specs:
        return None
    if bpm is None:
        return specs[0][1]
    for max_bpm, mapping in specs:
        if max_bpm is None or bpm <= max_bpm:
            return mapping
    return specs[-1][1]


def _db_to_gain(value_db: float) -> float:
    return float(10.0 ** (value_db / 20.0))


def _adapt_velocity_targets(
    config: VelocityScoringConfig,
    bpm: Optional[float],
) -> Tuple[
    Dict[str, FloatArray],
    Dict[str, float],
    Dict[str, float],
    Dict[str, float],
]:
    targets = {key: value.copy() for key, value in config.targets.items()}
    weights = dict(config.weights)
    dynamic_range = dict(config.dynamic_range)
    if config.tempo_dynamic_ranges:
        dynamic_override = _pick_tempo_mapping(
            bpm,
            config.tempo_dynamic_ranges,
        )
        if dynamic_override:
            dynamic_range.update(dynamic_override)
    phase_adjust = dict(config.phase_adjust_db)
    if config.tempo_phase_adjust:
        phase_override = _pick_tempo_mapping(
            bpm,
            config.tempo_phase_adjust,
        )
        if phase_override:
            phase_adjust.update(phase_override)
    overrides: Optional[Dict[str, FloatArray]] = _pick_tempo_target_overrides(
        bpm,
        config.tempo_bin_targets,
    )
    if overrides is not None:
        for key, histogram in overrides.items():
            targets[key] = histogram.copy()
    if not config.tempo_bins:
        return targets, weights, dynamic_range, phase_adjust
    bin_cfg = _pick_tempo_bin(bpm, config.tempo_bins)
    alpha = float(bin_cfg.get("alpha", 1.0))
    boost = float(bin_cfg.get("downbeat_boost", 0.0))

    for key, histogram in list(targets.items()):
        logits = np.log(np.clip(histogram, 1e-9, 1.0))
        targets[key] = _softmax(logits * alpha)

    weights["downbeat"] = weights.get("downbeat", 0.0) + boost
    weight_total = sum(weights.values())
    if weight_total > 0:
        for key in list(weights.keys()):
            weights[key] /= weight_total
    return targets, weights, dynamic_range, phase_adjust


def _velocity_axis_score(
    ctx: LoopFeatureContext,
    config: VelocityScoringConfig,
) -> float:
    if ctx.velocities.size == 0:
        return 0.0
    (
        targets_map,
        weights_map,
        dynamic_range_map,
        phase_adjust_map,
    ) = _adapt_velocity_targets(config, ctx.bpm)
    nbins = config.nbins
    velocities = ctx.velocities.astype(np.float64)
    if config.normalize_dynamic_range:
        min_level = dynamic_range_map.get("min")
        max_level = dynamic_range_map.get("max")
        if min_level is not None and max_level is not None and max_level > min_level:
            span = max_level - min_level
            velocities = np.clip(
                (velocities - min_level) * (127.0 / span),
                0.0,
                127.0,
            )
    phase_gains: Dict[str, float] = {"global": 1.0}
    if config.phase_compensation:
        for key, value in phase_adjust_map.items():
            try:
                phase_gains[str(key)] = _db_to_gain(float(value))
            except (TypeError, ValueError):
                continue

    def _apply_phase(values: FloatArray, *keys: str) -> FloatArray:
        gain = phase_gains.get("global", 1.0)
        for key in keys:
            gain *= phase_gains.get(key, 1.0)
        if np.isclose(gain, 1.0):
            return values
        return np.clip(values * gain, 0.0, 127.0)

    hist_global = np.histogram(
        _apply_phase(velocities),
        bins=nbins,
        range=(0, 128),
    )[
        0
    ].astype(np.float64)
    p_global = _normalise_histogram(hist_global)

    beat_residual = ctx.beats - np.round(ctx.beats)
    is_downbeat = np.isclose(beat_residual, 0.0, atol=0.1)
    hist_downbeat = np.histogram(
        _apply_phase(velocities[is_downbeat], "downbeat"),
        bins=nbins,
        range=(0, 128),
    )[0].astype(np.float64)
    p_downbeat = _normalise_histogram(hist_downbeat)

    fractional = np.modf(ctx.beats)[0]
    is_offbeat = np.isclose(fractional, 0.5, atol=0.1)
    hist_offbeat = np.histogram(
        _apply_phase(velocities[is_offbeat], "offbeat"),
        bins=nbins,
        range=(0, 128),
    )[0].astype(np.float64)
    p_offbeat = _normalise_histogram(hist_offbeat)

    beats_per_bar = max(1.0, ctx.beats_per_bar)
    window_ratio = config.prefill_window_beats / beats_per_bar
    threshold = max(0.0, 1.0 - window_ratio)
    prefill_mask = ctx.beat_positions >= threshold
    hist_prefill = np.histogram(
        _apply_phase(velocities[prefill_mask], "prefill"),
        bins=nbins,
        range=(0, 128),
    )[0].astype(np.float64)
    p_prefill = _normalise_histogram(hist_prefill)

    def _pair_score(histogram: FloatArray, target_key: str) -> float:
        target = targets_map.get(target_key, targets_map.get("global"))
        if target is None:
            raise ValueError("velocity targets must include 'global'")
        js = _jensen_shannon(histogram, target)
        js_norm = 1.0 - min(js / math.log(2.0), 1.0)
        cos = _cosine_similarity(histogram, target)
        cos_norm = (cos + 1.0) / 2.0
        js_weight = config.metric_weights.get("js", 0.5)
        cos_weight = config.metric_weights.get("cos", 0.5)
        return js_weight * js_norm + cos_weight * cos_norm

    defaults = {
        "global": 0.4,
        "downbeat": 0.25,
        "offbeat": 0.2,
        "prefill": 0.15,
    }
    score = 0.0
    for key, histogram in (
        ("global", p_global),
        ("downbeat", p_downbeat),
        ("offbeat", p_offbeat),
        ("prefill", p_prefill),
    ):
        weight = weights_map.get(key, defaults[key])
        score += weight * _pair_score(histogram, key)
    return float(_clip(score, 0.0, 1.0))


def _smooth_onset_grid(counts: FloatArray, window: int = 4) -> FloatArray:
    if counts.size == 0:
        return counts
    kernel = np.ones(window, dtype=np.float64) / float(window)
    smoothed = np.convolve(counts, kernel, mode="same")
    max_value = float(smoothed.max())
    if max_value <= 0:
        return np.zeros_like(smoothed, dtype=np.float64)
    return (smoothed / max_value).astype(np.float64)


def _periodicity_strength(
    grid: FloatArray,
    candidates: Tuple[int, ...],
) -> float:
    if grid.size == 0:
        return 0.0
    centered = grid - float(np.mean(grid))
    autocorr = np.correlate(
        centered,
        centered,
        mode="full",
    )[grid.size - 1 :]
    if autocorr[0] <= 0:
        return 0.0
    autocorr /= autocorr[0]
    peaks: List[float] = []
    for lag in candidates:
        if 0 < lag < autocorr.size:
            peaks.append(float(autocorr[lag]))
    if not peaks:
        return 0.0
    peaks.sort(reverse=True)
    top = peaks[:2]
    return float(_clip(sum(top) / len(top), 0.0, 1.0))


def _ngram_diversity(tokens: IntArray, n: int = 3) -> float:
    if tokens.size < n:
        return 0.0
    grams = {tuple(tokens[idx : idx + n]) for idx in range(tokens.size - n + 1)}
    total = max(1, tokens.size - n + 1)
    ratio = len(grams) / total
    return float(_clip(ratio, 0.0, 1.0))


def _density_fit(value: float, low: float, high: float) -> float:
    if high <= low:
        return 0.0
    mid = 0.5 * (low + high)
    span = 0.5 * (high - low)
    if span <= 0:
        return 0.0
    distance = abs(value - mid) / span
    return float(_clip(1.0 - distance, 0.0, 1.0))


def _role_stability_score(ctx: LoopFeatureContext) -> float:
    if len(ROLE_STABILITY_INDICES) == 0:
        return 0.0
    if len(ctx.role_slots_per_bar) < 2:
        return 0.0
    scores: List[float] = []
    history: List[Tuple[FrozenSet[int], ...]] = [ctx.role_slots_per_bar[0]]
    window = 4
    for bar_slots in ctx.role_slots_per_bar[1:]:
        similarities: List[float] = []
        for lookback in range(1, min(len(history), window) + 1):
            reference = history[-lookback]
            role_total = 0.0
            support = 0
            for index in ROLE_STABILITY_INDICES:
                ref_slots = reference[index]
                cur_slots = bar_slots[index]
                union = ref_slots | cur_slots
                if not union:
                    role_total += 1.0
                    continue
                intersection = ref_slots & cur_slots
                role_total += len(intersection) / len(union)
                support += 1
            if support == 0:
                similarities.append(1.0)
            else:
                similarities.append(role_total / float(support))
        if similarities:
            scores.append(max(similarities))
        history.append(bar_slots)
        if len(history) > window:
            history.pop(0)
    if not scores:
        return 0.0
    average = sum(scores) / float(len(scores))
    return float(_clip(average, 0.0, 1.0))


def _structure_axis_score(
    ctx: LoopFeatureContext,
    config: StructureScoringConfig,
) -> float:
    grid = _smooth_onset_grid(ctx.onset_counts_16th)
    periodicity = _periodicity_strength(grid, config.periodicity_candidates)
    diversity = _ngram_diversity(ctx.role_tokens_16th, n=3)
    low, high = config.expected_density(ctx.bpm)
    density_value = float(np.sum(ctx.onset_counts_16th))
    density_score = _density_fit(density_value, low, high)
    role_weight = config.weights.get("role_stability", 0.0)
    score = (
        config.weights.get("periodicity", 0.45) * periodicity
        + config.weights.get("diversity", 0.35) * diversity
        + config.weights.get("density", 0.2) * density_score
        + role_weight * _role_stability_score(ctx)
    )
    return float(_clip(score, 0.0, 1.0))


def _clip(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _clamp_opt(
    value: Optional[float],
    lower: float,
    upper: float,
) -> Optional[float]:
    if value is None:
        return None
    return _clip(float(value), lower, upper)


def _invert_scale(value: Optional[float], limit: float) -> Optional[float]:
    if value is None:
        return None
    return max(0.0, 1.0 - float(value) / limit)


def _bandpass(
    value: Optional[float],
    centre: float,
    width: float,
) -> Optional[float]:
    if value is None:
        return None
    delta = abs(float(value) - centre)
    return max(0.0, 1.0 - delta / width)


def _average(values: Iterable[Optional[float]]) -> float:
    items = [float(v) for v in values if v is not None]
    if not items:
        return 0.5
    return _clip(sum(items) / len(items), 0.0, 1.0)


GROOVE_REFERENCE_KEYS = ("eighth", "sixteenth", "triplet", "quarter")
GROOVE_REFERENCE_VECTOR = {
    "eighth": 0.38,
    "sixteenth": 0.32,
    "triplet": 0.18,
    "quarter": 0.12,
}
_GROOVE_REFERENCE_MAGNITUDE = math.sqrt(
    sum(value**2 for value in GROOVE_REFERENCE_VECTOR.values()),
)


def _swing_alignment_score(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    straight = _bandpass(value, 1.0, 0.25)
    swung = _bandpass(value, 1.5, 0.35)
    if straight is None and swung is None:
        return None
    return max(straight or 0.0, swung or 0.0)


def swing_alignment_score(value: Optional[float]) -> Optional[float]:
    return _swing_alignment_score(value)


def _fingerprint_similarity(
    fingerprint: Optional[Dict[str, float]],
) -> Optional[float]:
    if not fingerprint:
        return None
    vector: List[float] = []
    for key in GROOVE_REFERENCE_KEYS:
        value = float(fingerprint.get(key, 0.0) or 0.0)
        vector.append(max(0.0, value))
    magnitude = math.sqrt(sum(component**2 for component in vector))
    if magnitude == 0 or _GROOVE_REFERENCE_MAGNITUDE == 0:
        return None
    normalised: List[float] = [component / magnitude for component in vector]
    reference: List[float] = [
        GROOVE_REFERENCE_VECTOR[key] / _GROOVE_REFERENCE_MAGNITUDE for key in GROOVE_REFERENCE_KEYS
    ]
    similarity = sum(a * b for a, b in zip(normalised, reference))
    return _clip(similarity, 0.0, 1.0)


def _syncopation_alignment(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    return _bandpass(value, 0.35, 0.35)


def fingerprint_similarity(
    fingerprint: Optional[Dict[str, float]],
) -> Optional[float]:
    return _fingerprint_similarity(fingerprint)


def _ghost_rate_window(
    events: Sequence[Tuple[int, int]],
    window_beats: float,
    ticks_per_beat: int,
    velocity_threshold: int,
) -> float:
    if not events or ticks_per_beat <= 0 or window_beats <= 0:
        return 0.0
    window_ticks = max(1, int(round(window_beats * ticks_per_beat)))
    ordered = sorted(events, key=lambda item: item[0])
    best_ratio = 0.0
    left = 0
    ghost_count = 0
    total_count = 0
    for right, (start, velocity) in enumerate(ordered):
        while left <= right and start - ordered[left][0] > window_ticks:
            total_count -= 1
            if ordered[left][1] <= velocity_threshold:
                ghost_count -= 1
            left += 1
        total_count += 1
        if velocity <= velocity_threshold:
            ghost_count += 1
        if total_count > 0:
            window_ratio = ghost_count / total_count
            if window_ratio > best_ratio:
                best_ratio = window_ratio
    return best_ratio


def _ghost_rate(
    events: Sequence[Tuple[int, int]],
    ticks_per_beat: int,
    velocity_threshold: int,
    short_window_beats: float = 0.25,
    long_window_beats: float = 0.5,
    window_velocity_cap: Optional[int] = 24,
) -> float:
    if not events:
        return 0.0
    overall = sum(1 for _, velocity in events if velocity <= velocity_threshold)
    overall_ratio = overall / len(events)
    window_threshold = velocity_threshold
    if window_velocity_cap is not None:
        window_threshold = min(window_threshold, window_velocity_cap)
    short_ratio = _ghost_rate_window(
        events,
        short_window_beats,
        ticks_per_beat,
        window_threshold,
    )
    long_ratio = _ghost_rate_window(
        events,
        long_window_beats,
        ticks_per_beat,
        window_threshold,
    )
    return max(overall_ratio, short_ratio, long_ratio)


def _compute_articulation_metrics(
    notes: Sequence[Sequence[Any]],
    *,
    ghost_threshold: int,
    ticks_per_beat: int,
    base_step: Optional[float],
    loop_duration_ticks: int,
    tempos: Sequence[int],
) -> Tuple[Dict[str, Optional[float]], Dict[str, Any]]:
    if not notes:
        metrics: Dict[str, Optional[float]] = {
            "articulation.snare_ghost_rate": None,
            "articulation.snare_flam_rate": None,
            "articulation.detache_ratio": None,
            "articulation.pizzicato_ratio": None,
        }
        support: Dict[str, Any] = {
            "snare_count": 0,
            "violin_count": 0,
            "snare_notes_per_sec": None,
            "loop_duration_seconds": None,
            "tempo_bpm": None,
            "string_track": False,
            "drum_track": False,
            "pizzicato_labeled": False,
        }
        return metrics, support

    effective_tpb = ticks_per_beat if ticks_per_beat > 0 else DEFAULT_TICKS_PER_BEAT
    flam_window = max(3, int(FLAM_WINDOW_RATIO * effective_tpb))
    snare_pitches = set(DRUM_CATEGORIES.get("snare", set()))

    snare_events: List[Tuple[int, int]] = []
    violin_events: List[Tuple[int, int, int]] = []

    for raw in notes:
        if len(raw) < 6:
            continue
        start = int(raw[1])
        duration = int(raw[2])
        channel = int(raw[3])
        pitch = int(raw[4])
        velocity = int(raw[5])

        if channel == 9 or pitch in snare_pitches:
            snare_events.append((start, velocity))

        violin_lower, violin_upper = VIOLIN_PITCH_RANGE
        if channel != 9 and violin_lower <= pitch <= violin_upper:
            violin_events.append((start, duration, velocity))

    snare_ghost_rate: Optional[float] = None
    snare_flam_rate: Optional[float] = None

    if snare_events:
        total_snare = len(snare_events)
        ghost_ratio = _ghost_rate(
            snare_events,
            effective_tpb,
            ghost_threshold,
        )
        snare_ghost_rate = ghost_ratio if total_snare else None

        ordered = sorted(snare_events, key=lambda item: item[0])
        flam_pairs = 0
        for first, second in zip(ordered, ordered[1:]):
            delta = second[0] - first[0]
            if 0 < delta <= flam_window and second[1] >= first[1]:
                flam_pairs += 1
        snare_flam_rate = flam_pairs / total_snare if total_snare else None

    detache_ratio: Optional[float] = None
    pizzicato_ratio: Optional[float] = None

    if violin_events:
        total_violin = len(violin_events)
        violin_ordered = sorted(violin_events, key=lambda item: item[0])
        reference_step = base_step if base_step and base_step > 0 else effective_tpb / 2.0
        gap_threshold = reference_step * DETACHE_GAP_RATIO
        detache_min = reference_step * DETACHE_DURATION_RANGE[0]
        detache_max = reference_step * DETACHE_DURATION_RANGE[1]
        pizzicato_max = reference_step * PIZZICATO_DURATION_RATIO

        detache_count = 0
        pizz_count = 0

        for idx, (start, duration, velocity) in enumerate(violin_ordered):
            if duration <= 0:
                continue
            if duration <= pizzicato_max:
                pizz_count += 1

            if detache_min <= duration <= detache_max:
                if idx == 0:
                    detache_count += 1
                else:
                    prev_start, prev_duration, _ = violin_ordered[idx - 1]
                    gap = start - (prev_start + prev_duration)
                    if gap >= gap_threshold:
                        detache_count += 1
        if total_violin:
            detache_ratio = detache_count / total_violin
            pizzicato_ratio = pizz_count / total_violin

    tempo_us = tempos[0] if tempos else None
    tempo_bpm = None
    if tempo_us and tempo_us > 0:
        tempo_bpm = 60_000_000 / tempo_us
    loop_duration_seconds = None
    if ticks_per_beat > 0 and tempo_us and tempo_us > 0:
        beats = loop_duration_ticks / ticks_per_beat
        loop_duration_seconds = beats * (tempo_us / 1_000_000)

    snare_count = len(snare_events)
    violin_count = len(violin_events)
    snare_notes_per_sec = None
    if loop_duration_seconds and loop_duration_seconds > 0:
        snare_notes_per_sec = snare_count / loop_duration_seconds

    metrics = {
        "articulation.snare_ghost_rate": snare_ghost_rate,
        "articulation.snare_flam_rate": snare_flam_rate,
        "articulation.detache_ratio": detache_ratio,
        "articulation.pizzicato_ratio": pizzicato_ratio,
    }
    support = {
        "snare_count": snare_count,
        "violin_count": violin_count,
        "snare_notes_per_sec": snare_notes_per_sec,
        "loop_duration_seconds": loop_duration_seconds,
        "tempo_bpm": tempo_bpm,
        "string_track": violin_count > 0,
        "drum_track": snare_count > 0,
        "pizzicato_labeled": False,
    }
    return metrics, support


ARTICULATION_RULES: Dict[str, Dict[str, Any]] = {
    "ghost": {
        "metric": "articulation.snare_ghost_rate",
        "config_key": "ghost_rate",
        "category": "drums",
        "support_key": "snare_count",
        "default_min_support": 8,
    },
    "flam": {
        "metric": "articulation.snare_flam_rate",
        "config_key": "flam_rate",
        "category": "drums",
        "support_key": "snare_count",
        "default_min_support": 24,
    },
    "detache": {
        "metric": "articulation.detache_ratio",
        "config_key": "detache_ratio",
        "category": "strings",
        "support_key": "violin_count",
        "default_min_support": 16,
    },
    "pizzicato": {
        "metric": "articulation.pizzicato_ratio",
        "config_key": "pizzicato_ratio",
        "category": "strings",
        "support_key": "violin_count",
        "default_min_support": 16,
    },
}


def _tempo_bin_label(
    value: Optional[float],
    boundaries: Sequence[float],
) -> str:
    if value is None:
        return "unknown"
    if not boundaries or len(boundaries) < 2:
        return "all"
    ordered = sorted(boundaries)
    for lower, upper in zip(ordered[:-1], ordered[1:]):
        if value < upper or math.isclose(value, upper):
            return f"{int(lower)}-{int(upper)}"
    last_lower = ordered[-2]
    last_upper = ordered[-1]
    return f"{int(last_lower)}-{int(last_upper)}"


class ArticulationLabeler:
    HIGH_IQR_FACTOR = 0.25

    def __init__(self, thresholds: Optional[ArticulationThresholds]) -> None:
        self.thresholds = thresholds
        self._auto_stats: Dict[str, Dict[str, List[float]]] = defaultdict(
            lambda: defaultdict(list),
        )
        self._observation_count = 0

    def evaluate(
        self,
        observation: ArticulationObservation,
    ) -> ArticulationResult:
        self._observation_count += 1
        if not self.thresholds or not self.thresholds.weights:
            return ArticulationResult.empty()

        if self.thresholds.mode == "auto":
            self._register_auto_stat(observation)

        presence: Dict[str, float] = {}
        labels: List[str] = []
        details: Dict[str, Any] = {}
        score_sum = 0.0
        weight_total = sum(self.thresholds.weights.values()) or 1.0

        for label, rule in ARTICULATION_RULES.items():
            if label not in self.thresholds.weights:
                continue
            result = self._evaluate_label(label, rule, observation)
            presence[label] = result["presence"]
            details[label] = result["details"]
            if result["presence"] >= 0.5:
                labels.append(label)
            score_sum += self.thresholds.weights[label] * result["presence"]

        axis_value = _clip(score_sum / weight_total, 0.0, 1.0)
        return ArticulationResult(
            score=score_sum,
            labels=labels,
            presence=presence,
            thresholds=details,
            axis_value=axis_value,
        )

    def summary(self) -> Dict[str, Any]:
        if not self.thresholds or not self.thresholds.weights:
            return {
                "mode": "disabled",
                "count": self._observation_count,
            }
        summary: Dict[str, Any] = {
            "mode": self.thresholds.mode,
            "count": self._observation_count,
            "weights": dict(self.thresholds.weights),
        }
        if self.thresholds.mode == "auto":
            auto_summary: Dict[str, Any] = {}
            for metric_name, bin_map in self._auto_stats.items():
                metric_summary: Dict[str, Any] = {}
                for bin_key, values in bin_map.items():
                    if len(values) < 2:
                        continue
                    ordered = sorted(values)
                    metric_summary[bin_key] = {
                        "q1": _percentile(ordered, 0.25),
                        "q2": _percentile(ordered, 0.5),
                        "q3": _percentile(ordered, 0.75),
                        "count": len(values),
                    }
                if metric_summary:
                    auto_summary[metric_name] = metric_summary
            if auto_summary:
                summary["auto"] = auto_summary
            summary["tempo_bins"] = self.thresholds.tempo_bins()
        return summary

    def _register_auto_stat(
        self,
        observation: ArticulationObservation,
    ) -> None:
        if not self.thresholds or self.thresholds.mode != "auto":
            return
        bins = self.thresholds.tempo_bins()
        bin_key = _tempo_bin_label(observation.tempo_bpm, bins)
        for label, rule in ARTICULATION_RULES.items():
            if label not in self.thresholds.weights:
                continue
            metric_value = observation.metrics.get(rule["metric"])
            if metric_value is None:
                continue
            support_value = observation.support.get(rule["support_key"], 0)
            support_count = int(support_value or 0)
            required = self._auto_support_requirement(rule)
            spec = self._category_spec(rule["category"], rule["config_key"])
            spec_min = spec.get("min_support")
            if spec_min is not None:
                try:
                    required = max(required, int(spec_min))
                except (TypeError, ValueError):
                    pass
            if support_count < required:
                continue
            metric_name = rule["config_key"]
            stats_for_metric = self._auto_stats.setdefault(
                metric_name,
                defaultdict(list),
            )
            stats_for_metric[bin_key].append(float(metric_value))

    def _category_spec(self, category: str, config_key: str) -> Dict[str, Any]:
        if not self.thresholds:
            return {}
        category_cfg = cast(
            Dict[str, Any],
            self.thresholds.fixed.get(category, {}),
        )
        entry = category_cfg.get(config_key, {})
        if isinstance(entry, dict):
            return cast(Dict[str, Any], entry)
        return {}

    def _auto_support_requirement(self, rule: Dict[str, Any]) -> int:
        default = int(rule.get("default_min_support", 0) or 0)
        if not self.thresholds:
            return default
        category = rule["category"]
        auto_value = self.thresholds.auto_min_support(category)
        if auto_value is None:
            return default
        return max(default, int(auto_value))

    def _evaluate_label(
        self,
        label: str,
        rule: Dict[str, Any],
        observation: ArticulationObservation,
    ) -> Dict[str, Any]:
        metric_value = observation.metrics.get(rule["metric"])
        support_value = observation.support.get(rule["support_key"], 0)
        support_count = int(support_value or 0)
        threshold_info = self._resolve_threshold(label, rule, observation)
        high = threshold_info.get("high")
        min_support = int(threshold_info.get("min_support", 0))
        presence = 0.0
        reason = "missing_metric" if metric_value is None else None

        if metric_value is None or high is None:
            presence = 0.0
            if high is None:
                reason = reason or "missing_threshold"
        elif support_count < min_support:
            presence = 0.0
            reason = "insufficient_support"
        else:
            presence = 1.0 if metric_value >= high else 0.0
            reason = "threshold_met" if presence else "threshold_not_met"

        if label in {"detache", "pizzicato"} and not observation.support.get(
            "string_track",
        ):
            presence = 0.0
            reason = "not_string_track"

        if label == "pizzicato" and threshold_info.get("prefer_labeled"):
            if not observation.support.get("pizzicato_labeled"):
                if metric_value is None or metric_value < (high or 0.0):
                    presence = 0.0
                    reason = "missing_explicit_label"

        details: Dict[str, Any] = {
            "metric": metric_value,
            "threshold": high,
            "min_support": min_support,
            "support": support_count,
            "source": threshold_info.get("source"),
            "reason": reason,
        }
        if "auto_summary" in threshold_info:
            details["auto"] = threshold_info["auto_summary"]
        return {"presence": presence, "details": details}

    def _resolve_threshold(
        self,
        label: str,
        rule: Dict[str, Any],
        observation: ArticulationObservation,
    ) -> Dict[str, Any]:
        if not self.thresholds:
            return {"high": None, "min_support": 0, "source": "disabled"}

        category = rule["category"]
        config_key = rule["config_key"]
        spec = self._category_spec(category, config_key)
        high_value: Optional[float] = None
        source = "fixed"
        provider_band: Optional[Band] = None
        provider_bin_label: Optional[str] = None
        tempo_raw = observation.tempo_bpm
        tempo_value = float(tempo_raw) if tempo_raw is not None else 120.0
        provider = self.thresholds.provider
        if provider is not None:
            provider_band = provider.band_for(label, tempo_value)
            if provider_band.low is not None or provider_band.high is not None:
                lo, hi = provider.bins.bin_pair(tempo_value)
                provider_bin_label = f"{int(lo)}-{int(hi)}"
                if provider_band.high is not None:
                    high_value = provider_band.high
                    source = f"provider:{provider_bin_label}"

        if "high" in spec:
            try:
                high_value = float(spec["high"])
            except (TypeError, ValueError):
                high_value = None

        auto_summary = None
        if self.thresholds.mode == "auto":
            auto_value, auto_info = self._auto_threshold_for_metric(
                config_key,
                observation,
            )
            if auto_value is not None:
                high_value = auto_value
                auto_summary = auto_info or {}
                source = f"auto:{auto_summary.get('bin', 'all')}"

        min_support = spec.get(
            "min_support",
            rule.get("default_min_support", 0),
        )
        try:
            min_support_int = int(min_support)
        except (TypeError, ValueError):
            min_support_int = int(rule.get("default_min_support", 0) or 0)

        auto_support = self.thresholds.auto_min_support(category)
        if auto_support is not None:
            min_support_int = max(min_support_int, int(auto_support))

        result: Dict[str, Any] = {
            "high": high_value,
            "min_support": min_support_int,
            "source": source,
            "prefer_labeled": bool(spec.get("prefer_labeled")),
            "label": label,
        }
        if provider_band is not None:
            result["band_low"] = provider_band.low
            result["band_high"] = provider_band.high
            result["band_count"] = provider_band.count
            result["band_bin"] = provider_bin_label
        if auto_summary:
            result["auto_summary"] = auto_summary
        return result

    def _auto_threshold_for_metric(
        self,
        metric_name: str,
        observation: ArticulationObservation,
    ) -> Tuple[Optional[float], Optional[Dict[str, Any]]]:
        if not self.thresholds or self.thresholds.mode != "auto":
            return None, None
        bin_map = self._auto_stats.get(metric_name)
        if not bin_map:
            return None, None
        bins = self.thresholds.tempo_bins()
        bin_key = _tempo_bin_label(observation.tempo_bpm, bins)
        values = list(bin_map.get(bin_key, []))
        if len(values) < 4:
            flattened: List[float] = []
            for entries in bin_map.values():
                flattened.extend(entries)
            values = flattened
            bin_key = "all"
        if len(values) < 4:
            return None, None
        ordered = sorted(values)
        q1 = _percentile(ordered, 0.25)
        q2 = _percentile(ordered, 0.5)
        q3 = _percentile(ordered, 0.75)
        iqr = q3 - q1
        high = q3 + self.HIGH_IQR_FACTOR * iqr
        drop_ratio = self.thresholds.hysteresis_drop_ratio()
        drop = q2 - drop_ratio * iqr
        return high, {
            "bin": bin_key,
            "q1": q1,
            "q2": q2,
            "q3": q3,
            "iqr": iqr,
            "drop": drop,
            "count": len(values),
        }


def _score_axes(
    metrics: Any,
    *,
    ctx: Optional[LoopFeatureContext] = None,
    velocity_cfg: Optional[VelocityScoringConfig] = None,
    structure_cfg: Optional[StructureScoringConfig] = None,
) -> Dict[str, float]:
    def metric(name: str) -> Optional[float]:
        return getattr(metrics, name, None)

    fingerprint_raw = getattr(metrics, "rhythm_fingerprint", {}) or {}
    fingerprint = cast(Dict[str, float], fingerprint_raw)
    groove_eighth = float(fingerprint.get("eighth", 0.0))
    groove_sixteenth = float(fingerprint.get("sixteenth", 0.0))
    groove_triplet = float(fingerprint.get("triplet", 0.0))
    groove_strength = groove_eighth + groove_sixteenth

    # 期待する拍のバランス (裏拍・シンコペーション) を 0–1 で粗く算出
    offbeat_total = groove_sixteenth + groove_triplet
    total_fp = (
        groove_eighth + groove_sixteenth + groove_triplet + float(fingerprint.get("quarter", 0.0))
    )
    if total_fp > 0:
        offbeat_ratio = _clip(offbeat_total / total_fp, 0.0, 1.0)
    else:
        offbeat_ratio = 0.0

    swing_alignment = _swing_alignment_score(metric("swing_ratio"))
    fingerprint_match = _fingerprint_similarity(fingerprint)
    syncopation_match = _syncopation_alignment(metric("syncopation_rate"))
    groove_balance = _bandpass(offbeat_ratio, 0.45, 0.45)

    velocity_legacy = _average(
        [
            _bandpass(metric("ghost_rate"), 0.25, 0.25),
            _bandpass(metric("accent_rate"), 0.25, 0.25),
            _bandpass(metric("velocity_range"), 80.0, 55.0),
            _bandpass(metric("unique_velocity_steps"), 5.5, 4.5),
            _bandpass(metric("velocity_std"), 12.0, 8.0),
        ]
    )
    structure_legacy = _average(
        [
            _bandpass(metric("repeat_rate"), 0.60, 0.45),
            _bandpass(metric("variation_factor"), 0.45, 0.35),
            _invert_scale(metric("breakpoint_count"), 10.0),
            _bandpass(metric("note_density_per_bar"), 10.0, 8.0),
        ]
    )

    axes: Dict[str, float] = {
        "timing": _average(
            [
                _clamp_opt(metric("swing_confidence"), 0.0, 1.0),
                _invert_scale(metric("microtiming_std"), 40.0),
                _invert_scale(metric("microtiming_rms"), 36.0),
                _bandpass(metric("fill_density"), 0.32, 0.28),
            ]
        ),
        "velocity": velocity_legacy,
        "groove_harmony": _average(
            [
                swing_alignment,
                fingerprint_match,
                syncopation_match,
                groove_balance,
                _clamp_opt(groove_strength, 0.0, 1.0),
            ]
        ),
        "drum_cohesion": _average(
            [
                _invert_scale(metric("drum_collision_rate"), 0.45),
                _clamp_opt(metric("role_separation"), 0.0, 1.0),
                _bandpass(metric("hat_transition_rate"), 0.5, 0.45),
            ]
        ),
        "structure": structure_legacy,
    }
    if ctx is not None and velocity_cfg is not None:
        axes["velocity"] = _velocity_axis_score(ctx, velocity_cfg)
    if ctx is not None and structure_cfg is not None:
        axes["structure"] = _structure_axis_score(ctx, structure_cfg)
    return axes


def score_axes(
    metrics: Any,
    *,
    ctx: Optional[LoopFeatureContext] = None,
    velocity_cfg: Optional[VelocityScoringConfig] = None,
    structure_cfg: Optional[StructureScoringConfig] = None,
) -> Dict[str, float]:
    return _score_axes(
        metrics,
        ctx=ctx,
        velocity_cfg=velocity_cfg,
        structure_cfg=structure_cfg,
    )


def _combine_score(
    axes: Dict[str, float],
    weights: Dict[str, float],
) -> Tuple[float, Dict[str, float]]:
    total_weight = sum(weights.values()) or float(len(axes))
    breakdown: Dict[str, float] = {}
    total = 0.0
    for axis, raw_value in axes.items():
        weight = float(weights.get(axis, 0.0))
        share = weight / total_weight if total_weight else 0.0
        clipped = _clip(float(raw_value), 0.0, 1.0)
        contribution = clipped * 100.0 * share
        breakdown[axis] = contribution
        total += contribution
    return total, breakdown


def _resolve_git_commit() -> Optional[str]:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return commit.decode("utf-8").strip() or None


def _compute_data_digest(
    loop_ids: Sequence[str],
    pipeline_version: str,
) -> Optional[str]:
    if not loop_ids:
        return f"stage2:{pipeline_version}:empty"
    hasher = hashlib.sha1()
    for loop_id in sorted(set(loop_ids)):
        hasher.update(loop_id.encode("utf-8"))
    hasher.update(pipeline_version.encode("utf-8"))
    digest = hasher.hexdigest()
    return f"stage2:{pipeline_version}:{digest}"


def _summarise_axes(
    samples: Sequence[Dict[str, float]],
) -> Dict[str, Dict[str, float]]:
    if not samples:
        return {}
    buckets: Dict[str, List[float]] = defaultdict(list)
    for sample in samples:
        for key, value in sample.items():
            buckets[key].append(float(value))
    summary: Dict[str, Dict[str, float]] = {}
    for key, values in buckets.items():
        if not values:
            continue
        ordered = sorted(values)
        median = float(statistics.median(ordered))
        summary[key] = {
            "mean": sum(values) / len(values),
            "min": min(values),
            "max": max(values),
            "p50": median,
            "p90": _percentile(ordered, 0.9),
        }
    return summary


def _tempo_summary(tempos: Sequence[int]) -> Dict[str, float]:
    bpms = [60_000_000 / t for t in tempos if t > 0]
    if not bpms:
        return {}
    summary: Dict[str, float] = {
        "initial_bpm": bpms[0],
        "min": min(bpms),
        "max": max(bpms),
    }
    if len(bpms) > 1:
        summary["std"] = statistics.pstdev(bpms)
    return summary


def _tempo_lock_method(tempos: Sequence[int]) -> str:
    unique = {t for t in tempos if t > 0}
    if not unique:
        return "none"
    if len(unique) == 1:
        return "first"
    return "median"


def _grid_confidence(metrics: Any) -> Optional[float]:
    candidates: List[float] = []
    if getattr(metrics, "swing_confidence", None) is not None:
        candidates.append(_clip(float(metrics.swing_confidence), 0.0, 1.0))
    if getattr(metrics, "microtiming_rms", None) is not None:
        rms = float(metrics.microtiming_rms)
        candidates.append(_clip(1.0 - (rms / 30.0), 0.0, 1.0))
    if not candidates:
        return None
    return sum(candidates) / len(candidates)


def _deterministic_seed(loop_id: str) -> int:
    digest = hashlib.sha1(loop_id.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def _flatten_metrics(metrics: Dict[str, Any]) -> Dict[str, Any]:
    flat: Dict[str, Any] = {}
    for key, value in metrics.items():
        column = f"metrics.{key}"
        if isinstance(value, (dict, list, tuple)):
            flat[column] = json.dumps(value, ensure_ascii=False)
        else:
            flat[column] = value
    return flat


def _percentile(values: Sequence[float], ratio: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    if len(ordered) == 1:
        return ordered[0]
    position = ratio * (len(ordered) - 1)
    low = int(math.floor(position))
    high = int(math.ceil(position))
    if low == high:
        return ordered[low]
    weight = position - low
    return ordered[low] * (1 - weight) + ordered[high] * weight


@dataclass
class Stage2Extractor:
    settings: Stage2Settings
    metadata_index: Path
    metadata_dir: Path
    input_dir: Path
    index_data: Dict[str, Any]
    tmidix_path: Path

    def run(self) -> Dict[str, Any]:
        tmidix_mod = _import_tmidix(self.tmidix_path)
        records = iter_loop_records(
            self.index_data,
            metadata_dir=self.metadata_dir,
            index_path=self.metadata_index,
        )

        events_rows: List[Dict[str, Any]] = []
        loop_rows: List[Dict[str, Any]] = []
        score_rows: List[Dict[str, Any]] = []
        audio_embedding_rows: List[Dict[str, Any]] = []
        loop_ids: List[str] = []
        axis_pass_samples: List[Dict[str, float]] = []
        git_commit = _resolve_git_commit() or "unknown"
        passed_scores: List[float] = []
        retry_count = 0
        tempo_with_events = 0
        tempo_missing = 0
        exclusions: Counter[str] = Counter()
        total_seen = 0
        processed = 0
        ghost_threshold = self.settings.metrics.ghost_velocity_threshold
        articulation_labeler = ArticulationLabeler(
            self.settings.articulation_thresholds,
        )

        adaptive_state: Optional[AudioAdaptiveState]
        adaptive_outcomes: Optional[Counter[str]]
        adaptive_skipped_reasons: Optional[Counter[str]]
        adaptive_rule_counts: Optional[Counter[str]]
        adaptive_total_delta_sum: Optional[float]
        adaptive_total_delta_samples: Optional[int]
        adaptive_total_delta_limited: Optional[int]
        if self.settings.audio_adaptive_weights is not None:
            adaptive_state = AudioAdaptiveState()
            adaptive_outcomes = Counter()
            adaptive_skipped_reasons = Counter()
            adaptive_rule_counts = Counter()
            adaptive_total_delta_sum = 0.0
            adaptive_total_delta_samples = 0
            adaptive_total_delta_limited = 0
        else:
            adaptive_state = None
            adaptive_outcomes = None
            adaptive_skipped_reasons = None
            adaptive_rule_counts = None
            adaptive_total_delta_sum = None
            adaptive_total_delta_samples = None
            adaptive_total_delta_limited = None

        limit = self.settings.limit
        iterator: Iterator[Dict[str, Any]] = iter(records)
        for record in tqdm(iterator, desc="stage2", unit="loop"):
            if limit is not None and processed >= limit:
                break
            total_seen += 1

            loop = cast(Dict[str, Any], record.get("loop", {}))
            loop_id = loop.get("md5")
            if not loop_id:
                exclusions["missing_md5"] += 1
                continue
            output_path = loop.get("output_path")
            if not output_path:
                exclusions["missing_output_path"] += 1
                continue
            midi_path = Path(output_path)
            if not midi_path.is_absolute():
                midi_path = self.input_dir / midi_path
            if not midi_path.exists():
                exclusions["missing_file"] += 1
                continue

            try:
                midi_bytes = midi_path.read_bytes()
                score = tmidix_mod.midi2score(midi_bytes)
            except (OSError, ValueError, RuntimeError):
                exclusions["parse_error"] += 1
                continue

            ticks_per_beat = int(score[0]) if score else 480
            events = _flatten_events(score[1:])
            notes = [e for e in events if len(e) >= 6 and e[0] == "note"]
            if not notes:
                exclusions["no_notes"] += 1
                continue

            tempos = _tempo_events(events)
            if tempos:
                tempo_with_events += 1
            else:
                tempo_missing += 1
            time_signature = _time_signature(events)
            channel_programs = _channel_programs(events)

            loop_source = str(loop.get("source", "drumloops"))
            file_digest = cast(Optional[str], loop.get("file_digest"))

            audio_row: Optional[AudioGuidanceRow] = None
            audio_loop_row: Dict[str, Any] = AUDIO_LOOP_DEFAULTS.copy()
            audio_retry_context: Optional[Dict[str, Any]] = None
            if self.settings.audio_guidance is not None:
                audio_row = self.settings.audio_guidance.get(
                    loop_id,
                    file_digest,
                    cast(Optional[str], loop.get("filename")),
                )
                if audio_row is not None:
                    audio_loop_row = audio_row.as_loop_row()
                    context_candidate = audio_row.as_retry_context()
                    if context_candidate:
                        audio_retry_context = context_candidate
            audio_summary, audio_embedding_entry = _build_audio_summary(loop_id, audio_row)
            if audio_embedding_entry is not None:
                audio_embedding_rows.append(audio_embedding_entry)

            loop_metrics = compute_loop_metrics(
                notes,
                config=self.settings.metrics,
                ticks_per_beat=ticks_per_beat,
                tempo_events=tempos,
            )
            round_digits = self.settings.metrics.round_digits
            metrics_dict = loop_metrics.to_dict(digits=round_digits)
            (
                articulation_metrics,
                articulation_support,
            ) = _compute_articulation_metrics(
                notes,
                ghost_threshold=ghost_threshold,
                ticks_per_beat=ticks_per_beat,
                base_step=loop_metrics.base_step,
                loop_duration_ticks=loop_metrics.duration_ticks,
                tempos=tempos,
            )
            metrics_dict.update(articulation_metrics)

            if audio_summary:
                metrics_dict.setdefault(
                    "text_audio_cos",
                    audio_summary.get("text_audio_cos"),
                )
                metrics_dict.setdefault(
                    "text_audio_cos_mert",
                    audio_summary.get("text_audio_cos_mert"),
                )
            else:
                metrics_dict.setdefault("text_audio_cos", None)
                metrics_dict.setdefault("text_audio_cos_mert", None)
            metrics_dict["audio"] = dict(audio_summary)
            observation = ArticulationObservation(
                loop_id=loop_id,
                metrics=articulation_metrics,
                support=articulation_support,
                tempo_bpm=articulation_support.get("tempo_bpm"),
                metrics_full=metrics_dict,
            )
            articulation_result = articulation_labeler.evaluate(observation)
            metrics_dict["articulation.score_weighted"] = round(
                articulation_result.score, round_digits
            )
            metrics_dict["articulation.score_normalized"] = round(
                articulation_result.axis_value, round_digits
            )
            tempo_summary = _tempo_summary(tempos)
            tempo_lock = _tempo_lock_method(tempos)
            grid_confidence = _grid_confidence(loop_metrics)

            bpm_hint: Optional[float] = None
            if tempos:
                try:
                    bpm_hint = 60_000_000 / float(max(1, tempos[0]))
                except (TypeError, ValueError):
                    bpm_hint = None
            if bpm_hint is None:
                bpm_value = loop.get("bpm")
                if bpm_value is not None:
                    try:
                        bpm_hint = float(bpm_value)
                    except (TypeError, ValueError):
                        bpm_hint = None

            adaptive_rule_preview: Optional[AudioAdaptiveRule] = None
            adaptive_evaluation_preview: Optional[AudioAdaptiveEvaluation] = None
            adaptive_rule_pivot: Optional[float] = None
            phase_comp_factor_used: Optional[float] = None
            velocity_cfg_effective = self.settings.velocity_scoring
            if self.settings.audio_adaptive_weights is not None:
                preview_context = {
                    "audio": audio_summary,
                    "metrics": metrics_dict,
                }
                adaptive_evaluation_preview = _evaluate_audio_adaptive_rule(
                    self.settings.audio_adaptive_weights,
                    preview_context,
                    adaptive_state,
                )
                adaptive_rule_preview = adaptive_evaluation_preview.rule
                adaptive_rule_pivot = adaptive_evaluation_preview.pivot
                adaptive_state = adaptive_evaluation_preview.state
                metrics_dict["audio.adaptive_preview"] = _summarise_adaptive_flags(
                    adaptive_evaluation_preview.flags,
                    self.settings.audio_adaptive_weights.log_level,
                )
                if adaptive_rule_preview is not None and self.settings.velocity_scoring is not None:
                    phase_comp_factor = adaptive_rule_preview.extras.get("phase_comp_factor")
                    velocity_cfg_effective = _scale_velocity_phase_compensation(
                        self.settings.velocity_scoring,
                        phase_comp_factor,
                    )
                    if phase_comp_factor is not None:
                        try:
                            phase_comp_factor_used = float(phase_comp_factor)
                        except (TypeError, ValueError):
                            phase_comp_factor_used = None

            grid_divisions = 16
            if self.settings.structure_scoring is not None:
                grid_divisions = self.settings.structure_scoring.grid_divisions
            feature_ctx = _build_feature_context(
                notes,
                ticks_per_beat=ticks_per_beat,
                time_signature=time_signature,
                channel_programs=channel_programs,
                duration_ticks=loop_metrics.duration_ticks,
                bpm_hint=bpm_hint,
                grid_divisions=grid_divisions,
            )

            if ticks_per_beat > 0:
                ts_numerator, ts_denominator = time_signature
                ts_ratio = 4 / ts_denominator
                bar_ticks = ticks_per_beat * ts_numerator * ts_ratio
                beat_length_ticks = ticks_per_beat * ts_ratio
            else:
                bar_ticks = 0
                beat_length_ticks = 0

            event_rows, program_list = _event_rows(
                loop_id=loop_id,
                notes=notes,
                ticks_per_bar=bar_ticks,
                beat_ticks=beat_length_ticks,
                base_step=loop_metrics.base_step,
                channels=channel_programs,
                ghost_threshold=ghost_threshold,
                ticks_per_beat=ticks_per_beat,
                source=loop_source,
                pipeline_version=self.settings.pipeline_version,
                file_digest=file_digest,
            )
            events_rows.extend(event_rows)

            axes_raw = _score_axes(
                loop_metrics,
                ctx=feature_ctx,
                velocity_cfg=velocity_cfg_effective,
                structure_cfg=self.settings.structure_scoring,
            )
            axes_raw["articulation"] = articulation_result.axis_value
            adaptive_context: Dict[str, Any] = {
                "audio": audio_summary,
                "metrics": metrics_dict,
                "axes_raw": axes_raw,
            }
            if adaptive_evaluation_preview is not None:
                (
                    effective_axis_weights,
                    applied_adaptive_rule,
                    adaptive_pivot,
                    adaptive_state,
                    adaptive_details,
                ) = _apply_audio_adaptive_weights(
                    self.settings.axis_weights,
                    self.settings.audio_adaptive_weights,
                    adaptive_context,
                    state=adaptive_state,
                    evaluated=adaptive_evaluation_preview,
                )
            else:
                (
                    effective_axis_weights,
                    applied_adaptive_rule,
                    adaptive_pivot,
                    adaptive_state,
                    adaptive_details,
                ) = _apply_audio_adaptive_weights(
                    self.settings.axis_weights,
                    self.settings.audio_adaptive_weights,
                    adaptive_context,
                    state=adaptive_state,
                )
            score_total, score_breakdown = _combine_score(
                axes_raw,
                effective_axis_weights,
            )
            metrics_dict["audio.adaptive_axis_weights"] = effective_axis_weights
            metrics_dict["audio.adaptive_pivot"] = adaptive_pivot
            metrics_dict["audio.adaptive_rule"] = _adaptive_rule_to_dict(applied_adaptive_rule)
            details_log_level = self.settings.audio_adaptive_weights.log_level
            metrics_dict["audio.adaptive_details"] = _summarise_adaptive_details(
                adaptive_details,
                applied_adaptive_rule,
                details_log_level,
            )
            if (
                adaptive_details
                and adaptive_outcomes is not None
                and adaptive_rule_counts is not None
                and adaptive_skipped_reasons is not None
                and adaptive_total_delta_sum is not None
                and adaptive_total_delta_samples is not None
                and adaptive_total_delta_limited is not None
            ):
                delta_value = adaptive_details.get("total_delta")
                try:
                    adaptive_total_delta_sum += (
                        float(delta_value) if delta_value is not None else 0.0
                    )
                except (TypeError, ValueError):
                    pass
                adaptive_total_delta_samples += 1
                if adaptive_details.get("total_delta_limited"):
                    adaptive_total_delta_limited += 1

                if applied_adaptive_rule is not None:
                    adaptive_outcomes["applied"] += 1
                    adaptive_rule_counts[applied_adaptive_rule.name] += 1
                else:
                    reason = "no_rule"
                    if adaptive_details.get("disabled"):
                        reason = "disabled"
                    elif adaptive_details.get("below_min_confidence"):
                        reason = "below_min_confidence"
                    elif adaptive_details.get("missing_policy_applied"):
                        reason = f"missing_{adaptive_details['missing_policy_applied']}"
                    elif adaptive_details.get("rule_cooldown_blocked"):
                        reason = "rule_cooldown"
                    elif adaptive_details.get("cooldown_active"):
                        reason = "global_cooldown"
                    elif adaptive_details.get("hysteresis_applied"):
                        reason = "hysteresis_hold"
                    else:
                        flags_map = adaptive_details.get("flags")
                        if isinstance(flags_map, Mapping) and flags_map.get("pivot_missing"):
                            reason = "pivot_missing"
                    adaptive_outcomes["skipped"] += 1
                    adaptive_skipped_reasons[reason] += 1
            if phase_comp_factor_used is not None:
                metrics_dict["audio.phase_compensation_factor"] = phase_comp_factor_used
            if score_total >= self.settings.threshold:
                passed_scores.append(score_total)
                axis_pass_samples.append(axes_raw)

            retry_preset_id: Optional[str] = None
            retry_seed: Optional[int] = None
            timestamp_now = datetime.now(timezone.utc).isoformat()

            exclusion_reason = None
            if score_total < self.settings.threshold:
                reason_key = _failure_reason(score_breakdown)
                exclusion_reason = f"{reason_key}_below_threshold"
                retry_rule = _select_retry_rule(
                    reason_key=reason_key,
                    axes_raw=axes_raw,
                    score_total=score_total,
                    score_breakdown=score_breakdown,
                    rules=self.settings.retry_presets,
                    audio_context=audio_retry_context,
                )
                retry_action: Optional[Dict[str, Any]]
                if retry_rule is not None:
                    retry_action = retry_rule.action
                    retry_preset_id = retry_rule.name
                    retry_seed = _deterministic_seed(loop_id)
                else:
                    retry_action = None
                metrics_before: Dict[str, Any] = {
                    "score": score_total,
                    "axis": reason_key,
                    "axis_score": float(score_breakdown.get(reason_key, 0.0)),
                }
                retry_metrics: Dict[str, Optional[float]] = {
                    "drum_collision_rate": loop_metrics.drum_collision_rate,
                    "microtiming_std": loop_metrics.microtiming_std,
                    "microtiming_rms": loop_metrics.microtiming_rms,
                    "swing_confidence": loop_metrics.swing_confidence,
                    "articulation_score": articulation_result.score,
                    "axis": axes_raw.get(reason_key),
                    "text_audio_cos": cast(
                        Optional[float],
                        metrics_dict.get("text_audio_cos"),
                    ),
                    "text_audio_cos_mert": cast(
                        Optional[float],
                        metrics_dict.get("text_audio_cos_mert"),
                    ),
                }
                retry_payload: Dict[str, Any] = {
                    "loop_id": loop_id,
                    "score": score_total,
                    "reason": exclusion_reason,
                    "metrics": retry_metrics,
                    "preset_id": retry_preset_id,
                    "seed": retry_seed,
                    "applied_at": timestamp_now,
                    "preset": retry_action,
                    "metrics_before": metrics_before,
                    "metrics_after": None,
                }
                if audio_retry_context:
                    retry_payload["audio"] = audio_retry_context
                if retry_action is not None and retry_preset_id is not None:
                    retry_path = self.settings.paths.retry_dir / f"{loop_id}.json"
                    retry_path.parent.mkdir(parents=True, exist_ok=True)
                    retry_text = json.dumps(
                        retry_payload,
                        ensure_ascii=False,
                        indent=2,
                    )
                    retry_path.write_text(retry_text, encoding="utf-8")
                    retry_count += 1

            score_rows.append(
                {
                    "loop_id": loop_id,
                    "score": score_total,
                    "axes": score_breakdown,
                    "axes_raw": axes_raw,
                    "axis_weights_effective": effective_axis_weights,
                    "threshold": self.settings.threshold,
                    "retry_preset_id": retry_preset_id,
                    "seed": retry_seed,
                    "metrics_used": sorted(metrics_dict.keys()),
                    "created_at": timestamp_now,
                    "articulation_score": articulation_result.score,
                    "articulation_labels": articulation_result.labels,
                    "articulation_presence": articulation_result.presence,
                    "articulation_axis": articulation_result.axis_value,
                    "audio_adaptive_rule": metrics_dict["audio.adaptive_rule"],
                    "audio_adaptive_pivot": adaptive_pivot,
                    "audio": dict(audio_summary),
                }
            )

            if bar_ticks > 0:
                duration_ticks = loop.get("duration_ticks", 0)
                bar_count_value = math.ceil(duration_ticks / bar_ticks)
            else:
                bar_count_value = None
            program_set = sorted(set(program_list))
            tempo_events_json = None
            if tempos:
                tempo_events_json = json.dumps(list(tempos))
            tempo_summary_json = None
            if tempo_summary:
                tempo_summary_json = json.dumps(
                    tempo_summary,
                    ensure_ascii=False,
                )
            loop_row: Dict[str, Any] = {
                "loop_id": loop_id,
                "source": loop_source,
                "file_digest": file_digest or loop_id,
                "filename": loop.get("filename"),
                "genre": loop.get("genre"),
                "bpm": loop.get("bpm"),
                "note_count": loop.get("note_count"),
                "duration_ticks": loop.get("duration_ticks"),
                "bar_count": bar_count_value,
                "ticks_per_beat": ticks_per_beat,
                "time_signature": f"{time_signature[0]}/{time_signature[1]}",
                "program_set": program_set,
                "score.total": score_total,
                "score.threshold": self.settings.threshold,
                "score.passed": score_total >= self.settings.threshold,
                "score.threshold_passed": (score_total >= self.settings.threshold),
                "score.breakdown": json.dumps(
                    score_breakdown,
                    ensure_ascii=False,
                ),
                "score.axes_raw": json.dumps(
                    axes_raw,
                    ensure_ascii=False,
                ),
                "tempo.summary": tempo_summary_json,
                "tempo.events": tempo_events_json,
                "tempo.lock_method": tempo_lock,
                "tempo.grid_confidence": grid_confidence,
                "retry.preset_id": retry_preset_id,
                "retry.seed": retry_seed,
                "exclusion_reason": exclusion_reason,
                "pipeline_version": self.settings.pipeline_version,
                "articulation.score": articulation_result.score,
                "articulation.axis_value": articulation_result.axis_value,
                "articulation.labels": json.dumps(
                    articulation_result.labels,
                    ensure_ascii=False,
                ),
                "articulation.presence": json.dumps(
                    articulation_result.presence,
                    ensure_ascii=False,
                ),
                "articulation.thresholds": json.dumps(
                    articulation_result.thresholds,
                    ensure_ascii=False,
                ),
                "git_commit": git_commit,
            }
            loop_row.update(AUDIO_LOOP_DEFAULTS)
            if audio_row is not None:
                loop_row.update(audio_loop_row)
                loop_row.setdefault("text_audio_cos", audio_summary.get("text_audio_cos"))
                loop_row.setdefault(
                    "text_audio_cos_mert",
                    audio_summary.get("text_audio_cos_mert"),
                )
            loop_row.update(_flatten_metrics(metrics_dict))
            adaptive_rule_entry = metrics_dict.get("audio.adaptive_rule")
            if isinstance(adaptive_rule_entry, Mapping):
                loop_row["audio_adaptive_rule_name"] = adaptive_rule_entry.get("name")
            else:
                loop_row["audio_adaptive_rule_name"] = None
            loop_row["audio_adaptive_pivot"] = metrics_dict.get("audio.adaptive_pivot")
            adaptive_detail_entry = metrics_dict.get("audio.adaptive_details")
            if isinstance(adaptive_detail_entry, Mapping):
                loop_row["audio_adaptive_total_delta"] = adaptive_detail_entry.get("total_delta")
                loop_row["audio_adaptive_total_delta_limited"] = adaptive_detail_entry.get(
                    "total_delta_limited",
                )
                loop_row["audio_adaptive_total_delta_ratio"] = adaptive_detail_entry.get(
                    "total_delta_ratio",
                )
            else:
                loop_row["audio_adaptive_total_delta"] = None
                loop_row["audio_adaptive_total_delta_limited"] = None
                loop_row["audio_adaptive_total_delta_ratio"] = None
            loop_rows.append(loop_row)
            loop_ids.append(loop_id)

            processed += 1

        articulation_summary = articulation_labeler.summary()

        # Write artefacts
        data_digest = _compute_data_digest(
            loop_ids,
            self.settings.pipeline_version,
        )

        if data_digest:
            for item in loop_rows:
                item["data_digest"] = data_digest

        if events_rows:
            events_df = pd.DataFrame(events_rows)
            events_df.to_parquet(
                self.settings.paths.events_parquet,
                index=False,
            )

            sample_path = self.settings.paths.events_csv_sample
            sample_rows = self.settings.paths.sample_event_rows
            if sample_path and sample_rows > 0:
                events_df.head(sample_rows).to_csv(sample_path, index=False)

        if loop_rows:
            loop_df = pd.DataFrame(loop_rows)
            loop_df.to_csv(self.settings.paths.loop_summary_csv, index=False)

        if audio_embedding_rows and self.settings.paths.audio_embeddings_parquet is not None:
            _write_audio_embeddings(
                audio_embedding_rows,
                self.settings.paths.audio_embeddings_parquet,
            )

        if score_rows:
            metrics_path = self.settings.paths.metrics_jsonl
            with metrics_path.open("w", encoding="utf-8") as stream:
                for row in score_rows:
                    stream.write(json.dumps(row, ensure_ascii=False) + "\n")

        audio_adaptive_summary: Optional[Dict[str, Any]] = None
        if (
            self.settings.audio_adaptive_weights is not None
            and adaptive_outcomes is not None
            and adaptive_rule_counts is not None
            and adaptive_skipped_reasons is not None
            and adaptive_total_delta_sum is not None
            and adaptive_total_delta_samples is not None
            and adaptive_total_delta_limited is not None
        ):
            average_delta = (
                adaptive_total_delta_sum / adaptive_total_delta_samples
                if adaptive_total_delta_samples
                else 0.0
            )
            audio_adaptive_summary = {
                "outcomes": dict(adaptive_outcomes),
                "skipped_reasons": dict(adaptive_skipped_reasons),
                "applied_rules": dict(adaptive_rule_counts),
                "total_delta": {
                    "sum": adaptive_total_delta_sum,
                    "average": average_delta,
                    "samples": adaptive_total_delta_samples,
                    "limited": adaptive_total_delta_limited,
                },
            }

        summary = _build_summary(
            settings=self.settings,
            processed=processed,
            total=len(loop_rows) + sum(exclusions.values()),
            exclusion_counts=exclusions,
            tempo_with_events=tempo_with_events,
            tempo_missing=tempo_missing,
            passed_scores=passed_scores,
            retry_entries=retry_count,
            articulation_summary=articulation_summary,
            git_commit=git_commit,
            data_digest=data_digest,
            axis_summary=_summarise_axes(axis_pass_samples),
            audio_adaptive=audio_adaptive_summary,
        )

        if self.settings.paths.summary_out:
            summary_text = json.dumps(summary, ensure_ascii=False, indent=2)
            self.settings.paths.summary_out.write_text(
                summary_text,
                encoding="utf-8",
            )

        if self.settings.print_summary:
            _print_summary(summary)

        return summary


def _build_summary(
    *,
    settings: Stage2Settings,
    processed: int,
    total: int,
    exclusion_counts: Counter[str],
    tempo_with_events: int,
    tempo_missing: int,
    passed_scores: Sequence[float],
    retry_entries: int,
    articulation_summary: Optional[Dict[str, Any]] = None,
    git_commit: Optional[str] = None,
    data_digest: Optional[str] = None,
    axis_summary: Optional[Dict[str, Dict[str, float]]] = None,
    audio_adaptive: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    passed_count = len(passed_scores)
    pass_rate = float(passed_count / processed) if processed else 0.0

    scores_list = [float(score) for score in passed_scores]
    distribution: Dict[str, Any] = {
        "population": "passed_loops",
        "count": passed_count,
        "pass_rate": pass_rate,
    }
    if scores_list:
        median_value = float(statistics.median(scores_list))
        distribution.update(
            {
                "min": float(min(scores_list)),
                "median": median_value,
                "p50": median_value,
                "p90": _percentile(scores_list, 0.9),
                "mean": float(statistics.mean(scores_list)),
                "max": float(max(scores_list)),
            }
        )
    else:
        distribution.update(
            {
                "min": None,
                "median": None,
                "p50": None,
                "p90": None,
                "mean": None,
                "max": None,
            }
        )

    axis_stats = axis_summary or {}
    axis_payload: Dict[str, Any] = {
        "population": "passed_loops",
        "count": passed_count,
        "axes": axis_stats,
    }

    summary: Dict[str, Any] = {
        "pipeline_version": settings.pipeline_version,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "total_loops": total,
            "processed_loops": processed,
            "tempo_events": tempo_with_events,
            "tempo_missing": tempo_missing,
        },
        "outputs": {
            "passed_loops": passed_count,
            "pass_rate": pass_rate,
        },
        "exclusions": dict(exclusion_counts),
        "score_distribution": distribution,
        "retry_queue": retry_entries,
        "threshold": settings.threshold,
    }
    if articulation_summary:
        summary["articulation"] = articulation_summary
    summary["git_commit"] = git_commit
    summary["data_digest"] = data_digest
    summary["score_axes"] = axis_payload
    if audio_adaptive:
        summary["audio_adaptive"] = audio_adaptive
    return summary


def _print_summary(summary: Dict[str, Any]) -> None:
    print("=" * 70)
    print("Stage 2 Summary")
    print("=" * 70)
    print(f"Pipeline version : {summary.get('pipeline_version')}")
    inputs = cast(Dict[str, Any], summary.get("inputs", {}))
    outputs = cast(Dict[str, Any], summary.get("outputs", {}))
    print(f"Total loops      : {inputs.get('total_loops')}")
    print(f"Processed loops  : {inputs.get('processed_loops')}")
    if "passed_loops" in outputs:
        print(f"Passed loops     : {outputs.get('passed_loops')}")
    pass_rate = outputs.get("pass_rate")
    if isinstance(pass_rate, (int, float)):
        print(f"Pass rate        : {pass_rate:.3f}")
    print(f"Retry queue size : {summary.get('retry_queue')}")
    distribution = cast(
        Dict[str, Any],
        summary.get("score_distribution") or {},
    )
    if distribution:
        population = distribution.get("population")
        if population:
            print(f"Population      : {population}")
        print("Score distribution:")
        for key in ["min", "median", "mean", "p90", "max"]:
            value = distribution.get(key)
            if value is not None:
                print(f"  {key:>6}: {float(value):.2f}")
    exclusions = cast(Dict[str, Any], summary.get("exclusions") or {})
    if exclusions:
        print("Exclusions:")
        for reason, count in exclusions.items():
            print(f"  {reason}: {count}")
    print("=" * 70)


def _failure_reason(breakdown: Dict[str, float]) -> str:
    if not breakdown:
        return "unknown"
    return min(breakdown.items(), key=lambda item: item[1])[0]


def main() -> None:
    args = _parse_args()
    config = _load_config(args.config)
    settings = _build_settings(config, args)

    metadata_index_path = args.metadata_index or Path(
        config.get("paths", {}).get("metadata_index", DEFAULT_METADATA_INDEX)
    )
    metadata_dir = args.metadata_dir or Path(
        config.get("paths", {}).get("metadata_dir", DEFAULT_METADATA_DIR)
    )
    input_dir = args.input_dir or Path(config.get("paths", {}).get("input_dir", DEFAULT_INPUT_DIR))

    index_data = load_metadata_index(metadata_index_path)
    stage1_config = index_data.get("config", {})
    tmidix_path = Path(stage1_config.get("tmidix_path", DEFAULT_TMIDIX_PATH))

    extractor = Stage2Extractor(
        settings=settings,
        metadata_index=metadata_index_path,
        metadata_dir=metadata_dir,
        input_dir=input_dir,
        index_data=index_data,
        tmidix_path=tmidix_path,
    )
    extractor.run()


if __name__ == "__main__":
    main()
