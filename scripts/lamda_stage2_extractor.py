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
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import (
    Any,
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
    Sequence,
    Tuple,
    cast,
)

import pandas as pd
import yaml
from tqdm import tqdm

from lamda_tools import MetricConfig, compute_loop_metrics
from lamda_tools.metadata_io import iter_loop_records, load_metadata_index
from lamda_tools.metrics import DRUM_CATEGORIES

DEFAULT_CONFIG_PATH = Path("configs/lamda/drums_stage2.yaml")
DEFAULT_METADATA_INDEX = Path("output/drumloops_metadata/drumloops_metadata_v2.pickle")
DEFAULT_METADATA_DIR = Path("output/drumloops_metadata")
DEFAULT_INPUT_DIR = Path("output/drumloops_cleaned")
DEFAULT_TMIDIX_PATH = Path("data/Los-Angeles-MIDI/CODE")
DEFAULT_ARTICULATION_THRESHOLDS_PATH = Path("configs/thresholds/articulation.yaml")

DEFAULT_AXIS_WEIGHTS: Dict[str, float] = {
    "timing": 1.0,
    "velocity": 1.0,
    "groove": 1.0,
    "cohesion": 1.0,
    "structure": 1.0,
}

DEFAULT_TICKS_PER_BEAT = 480
FLAM_WINDOW_RATIO = 0.05
DETACHE_DURATION_RANGE = (0.9, 2.2)
DETACHE_GAP_RATIO = 0.15
PIZZICATO_DURATION_RATIO = 0.65
VIOLIN_PITCH_RANGE = (55, 103)


@dataclass
class ArticulationThresholds:
    mode: str
    fixed: Dict[str, Any]
    auto: Dict[str, Any]
    weights: Dict[str, float]
    path: Path

    @classmethod
    def load(cls, path: Path) -> "ArticulationThresholds":
        if not path.exists():
            hint = str(path)
            raise FileNotFoundError(f"Articulation threshold config not found: {hint}")
        raw_text = path.read_text(encoding="utf-8")
        raw_obj: Any = yaml.safe_load(raw_text) or {}
        if not isinstance(raw_obj, dict):
            raise ValueError("Articulation threshold config must be a mapping")
        raw = cast(Dict[str, Any], raw_obj)
        mode = str(raw.get("mode", "fixed")).lower()
        fixed_cfg = cast(Dict[str, Any], raw.get("fixed", {}))
        auto_cfg = cast(Dict[str, Any], raw.get("auto", {}))
        weights_root = cast(Dict[str, Any], raw.get("weights", {}))
        weights_cfg = cast(
            Dict[str, Any],
            weights_root.get("articulation", {}),
        )
        weights: Dict[str, float] = {
            str(key): float(value) for key, value in weights_cfg.items() if value is not None
        }
        return cls(
            mode=mode,
            fixed=fixed_cfg,
            auto=auto_cfg,
            weights=weights,
            path=path,
        )

    @property
    def total_weight(self) -> float:
        return sum(self.weights.values())

    def tempo_bins(self) -> List[float]:
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


@dataclass
class Stage2Settings:
    pipeline_version: str
    threshold: float
    axis_weights: Dict[str, float]
    retry_presets: Dict[str, Dict[str, Any]]
    metrics: MetricConfig
    paths: Stage2Paths
    limit: Optional[int]
    print_summary: bool
    articulation_thresholds: Optional[ArticulationThresholds]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Stage 2 artefacts for LAMDa loops.")
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

    return Stage2Paths(
        events_parquet=events_parquet,
        events_csv_sample=events_csv_sample,
        loop_summary_csv=loop_summary_csv,
        metrics_jsonl=metrics_jsonl,
        retry_dir=retry_dir,
        summary_out=summary_out,
        sample_event_rows=sample_rows,
    )


def _build_settings(
    config: Dict[str, Any],
    args: argparse.Namespace,
) -> Stage2Settings:
    pipeline_cfg = cast(Dict[str, Any], config.get("pipeline", {}))
    version = str(pipeline_cfg.get("version", "stage2"))
    threshold = float(pipeline_cfg.get("threshold", 70.0))
    if args.threshold is not None:
        threshold = args.threshold

    axis_cfg = cast(Dict[str, Any], config.get("score", {}).get("axes", {}))
    axis_weights = DEFAULT_AXIS_WEIGHTS.copy()
    for key, value in axis_cfg.items():
        axis_weights[str(key)] = float(value)

    retry_map: Dict[str, Dict[str, Any]] = {}
    raw_retry = cast(List[Any], config.get("retry_presets", []))
    for item in raw_retry:
        if not isinstance(item, dict):
            continue
        entry = cast(Dict[str, Any], item)
        reason_obj = entry.get("reason")
        if not reason_obj:
            continue
        reason = str(reason_obj)
        action = cast(Dict[str, Any], entry.get("action", {}))
        retry_map[reason] = action

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
        ghost_hits = sum(1 for _, velocity in snare_events if velocity <= ghost_threshold)
        snare_ghost_rate = ghost_hits / total_snare if total_snare else None

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
        self._auto_stats: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
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
            support_count = int(observation.support.get(rule["support_key"], 0) or 0)
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
        support_count = int(observation.support.get(rule["support_key"], 0) or 0)
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

        if label in {"detache", "pizzicato"} and not observation.support.get("string_track"):
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
        }
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


def _score_axes(metrics: Any) -> Dict[str, float]:
    axes: Dict[str, float] = {}
    axes["timing"] = _average(
        [
            _clamp_opt(metrics.swing_confidence, 0.0, 1.0),
            _invert_scale(metrics.microtiming_std, 30.0),
            _invert_scale(metrics.microtiming_rms, 30.0),
            _bandpass(metrics.syncopation_rate, 0.25, 0.25),
        ]
    )
    axes["velocity"] = _average(
        [
            _bandpass(metrics.ghost_rate, 0.2, 0.2),
            _bandpass(metrics.accent_rate, 0.2, 0.2),
            _bandpass(metrics.velocity_range, 60.0, 40.0),
            _bandpass(metrics.unique_velocity_steps, 6.0, 4.0),
            _bandpass(metrics.velocity_std, 10.0, 6.0),
        ]
    )
    fingerprint = cast(Dict[str, float], metrics.rhythm_fingerprint or {})
    groove_strength = fingerprint.get("eighth", 0.0) + fingerprint.get(
        "sixteenth",
        0.0,
    )
    axes["groove"] = _average(
        [
            _bandpass(metrics.swing_ratio, 1.0, 0.6),
            _bandpass(metrics.syncopation_rate, 0.25, 0.2),
            _clamp_opt(groove_strength, 0.0, 1.0),
        ]
    )
    axes["cohesion"] = _average(
        [
            _invert_scale(metrics.drum_collision_rate, 0.4),
            _clamp_opt(metrics.role_separation, 0.0, 1.0),
            _bandpass(metrics.hat_transition_rate, 0.4, 0.4),
        ]
    )
    axes["structure"] = _average(
        [
            _bandpass(metrics.repeat_rate, 0.5, 0.3),
            _bandpass(metrics.variation_factor, 0.4, 0.3),
            _invert_scale(metrics.breakpoint_count, 8.0),
            _bandpass(metrics.note_density_per_bar, 12.0, 8.0),
        ]
    )
    return axes


def _combine_score(
    axes: Dict[str, float],
    weights: Dict[str, float],
) -> Tuple[float, Dict[str, float]]:
    total_weight = sum(weights.values()) or len(weights)
    breakdown: Dict[str, float] = {}
    for axis, value in axes.items():
        weight = weights.get(axis, 0.0)
        share = weight / total_weight if total_weight else 0.0
        breakdown[axis] = float(value) * 100.0 * share
    return sum(breakdown.values()), breakdown


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
        passed_scores: List[float] = []
        retry_count = 0
        tempo_with_events = 0
        tempo_missing = 0
        exclusions: Counter[str] = Counter()
        total_seen = 0
        processed = 0
        ghost_threshold = self.settings.metrics.ghost_velocity_threshold
        articulation_labeler = ArticulationLabeler(self.settings.articulation_thresholds)

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

            axes = _score_axes(loop_metrics)
            axes["articulation"] = articulation_result.axis_value
            score_total, score_breakdown = _combine_score(
                axes,
                self.settings.axis_weights,
            )
            if score_total >= self.settings.threshold:
                passed_scores.append(score_total)

            retry_preset_id: Optional[str] = None
            retry_seed: Optional[int] = None

            timestamp_now = datetime.now(timezone.utc).isoformat()

            exclusion_reason = None
            if score_total < self.settings.threshold:
                reason_key = _failure_reason(score_breakdown)
                exclusion_reason = f"{reason_key}_below_threshold"
                retry_action = self.settings.retry_presets.get(reason_key, {})
                retry_preset_id = reason_key
                retry_seed = _deterministic_seed(loop_id)
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
                    "threshold": self.settings.threshold,
                    "retry_preset_id": retry_preset_id,
                    "seed": retry_seed,
                    "metrics_used": sorted(metrics_dict.keys()),
                    "created_at": timestamp_now,
                    "articulation_score": articulation_result.score,
                    "articulation_labels": articulation_result.labels,
                    "articulation_presence": articulation_result.presence,
                    "articulation_axis": articulation_result.axis_value,
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
                "score.breakdown": json.dumps(
                    score_breakdown,
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
            }
            loop_row.update(_flatten_metrics(metrics_dict))
            loop_rows.append(loop_row)

            processed += 1

        articulation_summary = articulation_labeler.summary()

        # Write artefacts
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

        if score_rows:
            metrics_path = self.settings.paths.metrics_jsonl
            with metrics_path.open("w", encoding="utf-8") as stream:
                for row in score_rows:
                    stream.write(json.dumps(row, ensure_ascii=False) + "\n")

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
) -> Dict[str, Any]:
    distribution: Dict[str, Any] = {}
    if passed_scores:
        distribution = {
            "population": "passed_loops",
            "min": float(min(passed_scores)),
            "median": float(statistics.median(passed_scores)),
            "p90": _percentile(list(passed_scores), 0.9),
            "max": float(max(passed_scores)),
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
        "exclusions": dict(exclusion_counts),
        "score_distribution": distribution,
        "retry_queue": retry_entries,
        "threshold": settings.threshold,
    }
    if articulation_summary:
        summary["articulation"] = articulation_summary
    return summary


def _print_summary(summary: Dict[str, Any]) -> None:
    print("=" * 70)
    print("Stage 2 Summary")
    print("=" * 70)
    print(f"Pipeline version : {summary.get('pipeline_version')}")
    inputs = cast(Dict[str, Any], summary.get("inputs", {}))
    print(f"Total loops      : {inputs.get('total_loops')}")
    print(f"Processed loops  : {inputs.get('processed_loops')}")
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
        for key in ["min", "median", "p90", "max"]:
            if key in distribution:
                print(f"  {key:>6}: {distribution[key]:.2f}")
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
