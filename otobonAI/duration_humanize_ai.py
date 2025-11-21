#!/usr/bin/env python3
"""DurationHumanizeAI — metadata-aware timing/length annotator for plan events.

This module reads policy-defined humanize profiles and enriches plan events
with per-event ``humanize`` payloads.  The payload blends the global/section
settings from ``policy['humanize']`` with RhythmAI metadata such as
``rhythm_pattern_id`` and manifest descriptors.  The resulting dictionary can
be consumed by downstream converters (e.g., ``json2midi.py`` or DUV pipelines)
to drive timing and duration adjustments deterministically.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import numpy as np
import pandas as pd

try:  # Optional dependency for DUV inference
    import torch
except Exception:  # pragma: no cover - optional dependency
    torch = None  # type: ignore

from utilities.csv_io import coerce_columns

try:
    from utilities.duv_infer import duv_sequence_predict, mask_any
    from utilities.ml_velocity import MLVelocityModel

    _DUV_AVAILABLE = True
    _DUV_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # pragma: no cover - optional dependency
    duv_sequence_predict = None  # type: ignore
    mask_any = None  # type: ignore
    MLVelocityModel = None  # type: ignore
    _DUV_AVAILABLE = False
    _DUV_IMPORT_ERROR = exc
from utilities.rhythm_vocab_loader import PatternEntry, RhythmVocabLoader


@dataclass(slots=True)
class _SectionProfile:
    timing_std_ms: float
    duration_scale: float
    duration_jitter: float
    staccato_prob: float
    phrase_end_extend: float
    max_shift_ms: float


_DEFAULT_PROGRAMS = {
    "piano": 0,
    "guitar": 24,
    "strings": 48,
    "bass": 32,
}
_DEFAULT_RHYTHM_MANIFEST = Path("data/rhythm_vocab.yaml")


def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():  # pragma: no cover - optional GPU
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():  # macOS
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


def _parse_fraction(value: object) -> float:
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return 0.0
        if "/" in text:
            left, right = text.split("/", 1)
            try:
                return float(left) / float(right)
            except Exception:
                return 0.0
        try:
            return float(text)
        except Exception:
            return 0.0
    return 0.0


class DurationHumanizeAI:
    """Annotate plan events with timing/duration metadata derived from policy."""

    def __init__(
        self,
        *,
        instrument: str,
        policy: Mapping[str, Any],
        tempo_bpm: float = 120.0,
        rhythm_manifest_path: Path | None = None,
        vocab_instrument: str | None = None,
    ) -> None:
        self.instrument = instrument
        self.policy = dict(policy or {})
        self.tempo_bpm = max(1.0, float(tempo_bpm or 120.0))
        self.vocab_instrument = (vocab_instrument or instrument).lower()
        self.humanize_cfg = dict(self.policy.get("humanize", {}))
        self.global_profile = dict(self.humanize_cfg.get("global", {}))
        self.section_profiles = {
            str(name).lower(): dict(cfg)
            for name, cfg in (self.humanize_cfg.get("sections", {}) or {}).items()
        }
        self.inst_cfg = dict(self.policy.get("instruments", {}).get(self.instrument, {}))
        manifest_path = rhythm_manifest_path
        if manifest_path is None and _DEFAULT_RHYTHM_MANIFEST.exists():
            manifest_path = _DEFAULT_RHYTHM_MANIFEST
        self.rhythm_manifest_path = Path(manifest_path).expanduser() if manifest_path else None
        self._manifest_index: dict[str, PatternEntry] = {}
        if self.rhythm_manifest_path and self.rhythm_manifest_path.exists():
            loader = RhythmVocabLoader(self.rhythm_manifest_path, validate=False)
            for entry in loader.vocab.entries():
                self._manifest_index[entry.id] = entry

        self._duv_cfg = dict(self.inst_cfg.get("humanize_duv", {}))
        if not self._duv_cfg:
            self._duv_cfg = dict(self.humanize_cfg.get("duv", {}))
        self._duv_model: MLVelocityModel | None = None
        self._duv_device = torch.device("cpu") if torch is not None else None
        self._duv_program = int(
            self._duv_cfg.get(
                "program",
                _DEFAULT_PROGRAMS.get(
                    self.vocab_instrument, _DEFAULT_PROGRAMS.get(self.instrument, -1)
                ),
            )
        )
        self._duv_duration_grid = _parse_fraction(self._duv_cfg.get("dur_quant"))
        self._duv_batch = int(self._duv_cfg.get("batch", 64))
        self._duv_model_name = None
        self._duv_enabled = False
        self._init_duv()

    # ------------------------------------------------------------------
    def annotate_plan(self, plan: Dict[str, Any]) -> None:
        events = plan.get("events") or []
        metadata = plan.get("metadata") or {}
        if not isinstance(events, list):
            return
        duv_annotations = self._run_duv(events, metadata) if self._duv_enabled else {}
        emotion_map = self._extract_emotion_map(metadata)
        for idx, event in enumerate(events):
            section_value = event.get("section") or event.get("section_label")
            if not section_value:
                section_value = metadata.get("default_section", "verse")
            section = str(section_value or "verse").lower()
            pattern_type = str(event.get("pattern") or event.get("event_type") or "root_quarter")
            rhythm_pattern_id = event.get("rhythm_pattern_id")
            manifest_entry = (
                self._manifest_index.get(str(rhythm_pattern_id)) if rhythm_pattern_id else None
            )
            section_profile = self._resolve_section_profile(section)
            duv_payload = duv_annotations.get(idx)
            bar_idx = self._infer_bar_idx(event)
            emotion_snapshot = emotion_map.get(bar_idx)
            humanize_payload = self._build_payload(
                event,
                section_profile=section_profile,
                pattern_type=pattern_type,
                manifest_entry=manifest_entry,
                bar_idx=bar_idx,
                emotion_data=emotion_snapshot,
                duv_data=duv_payload,
            )
            if duv_payload:
                humanize_payload["duv"] = duv_payload
            event["humanize"] = humanize_payload

    # ------------------------------------------------------------------
    def _resolve_section_profile(self, section: str) -> _SectionProfile:
        base = {
            "timing_std_ms": float(self.global_profile.get("timing_std_ms", 6.0)),
            "duration_scale": float(self.global_profile.get("duration_scale_mean", 0.95)),
            "duration_jitter": float(self.global_profile.get("duration_scale_jitter", 0.10)),
            "staccato_prob": float(self.global_profile.get("staccato_prob", 0.10)),
            "phrase_end_extend": float(self.global_profile.get("phrase_end_extend", 1.15)),
            "max_shift_ms": float(self.global_profile.get("max_shift_ms", 18.0)),
        }
        section_cfg = self.section_profiles.get(section.lower()) or {}
        base.update({k: float(v) for k, v in section_cfg.items() if isinstance(v, (int, float))})
        return _SectionProfile(
            timing_std_ms=max(0.0, base["timing_std_ms"]),
            duration_scale=max(0.01, base["duration_scale"]),
            duration_jitter=max(0.0, base["duration_jitter"]),
            staccato_prob=max(0.0, min(1.0, base["staccato_prob"])),
            phrase_end_extend=max(1.0, base["phrase_end_extend"]),
            max_shift_ms=max(0.0, base["max_shift_ms"]),
        )

    # ------------------------------------------------------------------
    def _build_payload(
        self,
        event: Mapping[str, Any],
        *,
        section_profile: _SectionProfile,
        pattern_type: str,
        manifest_entry: Optional[PatternEntry],
        bar_idx: int,
        emotion_data: Optional[Mapping[str, Any]] = None,
        duv_data: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        timing_ms = section_profile.timing_std_ms
        duration_scale = section_profile.duration_scale
        duration_jitter = section_profile.duration_jitter
        staccato_prob = section_profile.staccato_prob
        phrase_end_extend = section_profile.phrase_end_extend
        timing_push_ms = 0.0
        duration_override: float | None = None
        base_duration = float(event.get("duration_ql") or event.get("duration") or 0.0)
        emotion_payload: dict[str, Any] = {"bar_idx": int(bar_idx)}
        velocity_scale: float | None = None
        density_scale: float | None = None

        # Pattern heuristics
        if pattern_type == "root_eighth":
            timing_ms *= 0.85
            duration_scale *= 0.92
        elif pattern_type == "walking":
            timing_ms *= 0.8
            duration_scale *= 0.88
            staccato_prob = min(1.0, staccato_prob + 0.05)
        else:  # root_quarter or pads
            duration_scale *= 1.05

        # Manifest-driven adjustments
        if manifest_entry is not None:
            descriptors = {d.lower() for d in manifest_entry.descriptors}
            density_label = _normalize_density(manifest_entry.density)
            grid = manifest_entry.grid or 1.0
            if grid <= 0.25:
                timing_ms *= 0.7
                duration_scale *= 0.9
            elif grid >= 1.0:
                timing_ms *= 1.1
                duration_scale *= 1.1
            if density_label in {"dense", "wall"}:
                duration_scale *= 0.9
            elif density_label == "sparse":
                duration_scale *= 1.05
            if "pedal" in descriptors:
                timing_push_ms -= 3.0
                duration_scale *= 1.08
            if "drive" in descriptors or "lift" in descriptors:
                timing_push_ms += 4.0
                timing_ms *= 0.9
            bass_hook = (manifest_entry.ai_hooks or {}).get("bass_pattern_type")
            if bass_hook == "walking":
                staccato_prob = min(1.0, staccato_prob + 0.1)
            if bass_hook == "root_eighth":
                duration_scale *= 0.93

        # EmotionAI-driven adjustments
        if isinstance(emotion_data, Mapping):
            energy = _coerce_float(emotion_data.get("energy"))
            if energy is not None:
                energy = _clamp_float(energy, 0.0, 1.0)
                timing_ms *= _lerp(0.85, 1.2, energy)
                duration_jitter *= _lerp(0.85, 1.2, energy)
                timing_push_ms += (energy - 0.5) * 4.0
                emotion_payload["energy"] = round(energy, 4)

            tension = _coerce_float(emotion_data.get("tension"))
            if tension is not None:
                tension = _clamp_float(tension, 0.0, 1.0)
                timing_ms *= _lerp(1.12, 0.9, tension)
                duration_scale *= _lerp(1.06, 0.88, tension)
                timing_push_ms -= (tension - 0.5) * 5.5
                staccato_prob = _clamp_float(staccato_prob + (tension - 0.5) * 0.35, 0.0, 1.0)
                emotion_payload["tension"] = round(tension, 4)

            brightness = _coerce_float(emotion_data.get("brightness"))
            if brightness is not None:
                brightness = _clamp_float(brightness, 0.0, 1.0)
                timing_push_ms += (brightness - 0.5) * 3.0
                emotion_payload["brightness"] = round(brightness, 4)

            valence = _coerce_float(emotion_data.get("valence"))
            if valence is not None:
                valence = _clamp_float(valence, 0.0, 1.0)
                duration_scale *= _lerp(0.94, 1.06, valence)
                timing_ms *= _lerp(1.05, 0.95, valence)
                timing_push_ms += (valence - 0.5) * 2.5
                emotion_payload["valence"] = round(valence, 4)

            emotion_dur_scale = _coerce_float(emotion_data.get("duration_scale"))
            if emotion_dur_scale is not None and emotion_dur_scale > 0.0:
                blended = _lerp(1.0, _clamp_float(emotion_dur_scale, 0.6, 1.4), 0.75)
                duration_scale *= blended
                emotion_payload["duration_scale_hint"] = round(emotion_dur_scale, 4)

            velocity_raw = _coerce_float(emotion_data.get("velocity_scale"))
            if velocity_raw is None:
                velocity_raw = _derive_velocity_scale(energy, valence)
            if velocity_raw is not None:
                velocity_scale = _clamp_float(velocity_raw, 0.5, 1.5)
                emotion_payload["velocity_scale"] = round(velocity_scale, 4)

            density_raw = _coerce_float(emotion_data.get("density_scale"))
            if density_raw is None:
                density_raw = _derive_density_scale(energy)
            if density_raw is not None and density_raw > 0.0:
                density_scale = _clamp_float(density_raw, 0.5, 1.5)
                density_delta = density_scale - 1.0
                duration_jitter *= _clamp_float(1.0 - 0.35 * density_delta, 0.6, 1.4)
                staccato_prob = _clamp_float(staccato_prob + density_delta * 0.2, 0.0, 1.0)
                emotion_payload["density_scale"] = round(density_scale, 4)

            phrase_role = str(emotion_data.get("phrase_role") or "").lower()
            if phrase_role:
                emotion_payload["phrase_role"] = phrase_role
                if phrase_role == "start":
                    timing_push_ms -= 1.5
                elif phrase_role == "end":
                    duration_scale *= 1.05
                    staccato_prob *= 0.85
                    timing_push_ms -= 2.0

            tags = emotion_data.get("tags")
            if isinstance(tags, (list, tuple, set)):
                normalized = sorted({str(tag).lower() for tag in tags if tag})
                if normalized:
                    emotion_payload["tags"] = normalized
                if "climax" in normalized:
                    timing_ms *= 1.12
                    duration_scale *= 1.04
                    staccato_prob *= 0.9
                if "vocal_focus" in normalized:
                    timing_push_ms -= 2.0
                    staccato_prob = _clamp_float(staccato_prob * 0.9, 0.0, 1.0)

            section_label = emotion_data.get("section")
            if section_label:
                emotion_payload["section"] = str(section_label)
        else:
            emotion_payload = {}

        if velocity_scale is not None:
            velocity_delta = velocity_scale - 1.0
            timing_ms *= _clamp_float(1.0 - 0.2 * velocity_delta, 0.8, 1.2)
            timing_push_ms += velocity_delta * 2.0

        # Phrase ending boost when event is at tail of bar
        rel_time = event.get("time_ql")
        if rel_time is None:
            rel_time = event.get("start_ql")
        beat = float(rel_time or 0.0) - (bar_idx * 4.0)
        if beat >= 3.0:
            duration_scale *= phrase_end_extend

        # Final clamping to avoid extreme tails
        duration_scale = _clamp_float(duration_scale, 0.25, 1.75)
        duration_jitter = _clamp_float(duration_jitter, 0.0, 0.8)
        staccato_prob = _clamp_float(staccato_prob, 0.0, 1.0)
        timing_ms = max(0.0, timing_ms)
        timing_push_ms = _clamp_float(
            timing_push_ms,
            -section_profile.max_shift_ms,
            section_profile.max_shift_ms,
        )

        payload = {
            "timing_std_ms": round(timing_ms, 4),
            "timing_push_ms": round(timing_push_ms, 4),
            "duration_scale": round(duration_scale, 4),
            "duration_jitter": round(duration_jitter, 4),
            "staccato_prob": round(staccato_prob, 4),
            "max_shift_ms": round(section_profile.max_shift_ms, 4),
            "source": "DurationHumanizeAI",
        }
        if emotion_payload:
            payload["emotion"] = emotion_payload
        if duv_data:
            payload["source"] = "DurationHumanizeAI+DUV"
            duv_velocity = duv_data.get("velocity")
            if duv_velocity is not None:
                payload["velocity_target"] = int(duv_velocity)
            duv_duration = duv_data.get("duration_ql")
            if duv_duration is not None:
                duv_duration = float(duv_duration)
                if base_duration > 1e-4:
                    duration_scale = max(0.25, min(2.0, duv_duration / max(base_duration, 1e-4)))
                    payload["duration_scale"] = round(duration_scale, 4)
                else:
                    duration_override = max(0.01, duv_duration)
                payload["duration_source"] = "duv"
        if duration_override is not None:
            payload["duration_override_ql"] = round(duration_override, 5)
        rhythm_pattern_id = event.get("rhythm_pattern_id")
        if rhythm_pattern_id:
            payload["rhythm_pattern_id"] = rhythm_pattern_id
        payload["pattern_type"] = pattern_type
        payload["section"] = event.get("section") or event.get("section_label")
        if manifest_entry is not None:
            payload["manifest_density"] = manifest_entry.density
            payload["manifest_grid"] = manifest_entry.grid
        if velocity_scale is not None:
            payload["velocity_scale"] = round(velocity_scale, 4)
        if density_scale is not None:
            payload["density_scale"] = round(density_scale, 4)
        return payload

    # ------------------------------------------------------------------
    def _init_duv(self) -> None:
        if (
            not _DUV_AVAILABLE
            or MLVelocityModel is None
            or duv_sequence_predict is None
            or torch is None
        ):
            if self._duv_cfg.get("ckpt"):
                print(
                    "ℹ️  DurationHumanizeAI: DUV disabled (torch or utilities missing)",
                )
            return
        ckpt_path = self._duv_cfg.get("ckpt")
        if not ckpt_path:
            return
        path = Path(ckpt_path).expanduser()
        if not path.exists():
            return
        try:
            self._duv_device = _resolve_device(str(self._duv_cfg.get("device", "cpu")))
        except Exception:
            self._duv_device = torch.device("cpu")
        try:
            model = MLVelocityModel.load(str(path))
        except Exception:
            return
        model = model.to(self._duv_device).eval()
        if not getattr(model, "requires_duv_feats", False):
            # Sequence-aware checkpoints advertise this flag; bail out otherwise.
            return
        self._duv_model = model
        self._duv_model_name = path.name
        self._duv_enabled = True

    # ------------------------------------------------------------------
    def _run_duv(self, events: list[Any], metadata: Mapping[str, Any]) -> dict[int, Dict[str, Any]]:
        if (
            not events
            or self._duv_model is None
            or duv_sequence_predict is None
            or mask_any is None
        ):
            return {}
        prepared = self._prepare_duv_dataframe(events, metadata)
        if prepared is None:
            return {}
        df, event_indices = prepared
        if df.empty or not event_indices:
            return {}
        float_cols = {"velocity", "duration", "bar_phase", "beat_phase", "start", "onset"}
        int_cols = {"pitch", "position", "bar", "section", "track_id", "program"}
        df = coerce_columns(df, float32=float_cols, int32=int_cols)
        try:
            preds = duv_sequence_predict(df, self._duv_model, self._duv_device)
        except Exception:
            return {}
        has_vel = preds and mask_any(preds.get("velocity_mask"))
        has_dur = preds and mask_any(preds.get("duration_mask"))
        if not has_vel and not has_dur:
            return {}

        vel_pred = preds.get("velocity") if preds else None
        dur_pred = preds.get("duration") if has_dur and preds else None
        vel_mask = preds.get("velocity_mask") if preds else None
        dur_mask = preds.get("duration_mask") if preds else None

        if isinstance(vel_pred, torch.Tensor):
            vel_pred = vel_pred.detach().cpu().numpy()
        if isinstance(dur_pred, torch.Tensor):
            dur_pred = dur_pred.detach().cpu().numpy()

        if isinstance(vel_pred, np.ndarray):
            vel_pred = np.clip(vel_pred, 1, 127).astype(np.int16, copy=False)
        if isinstance(dur_pred, np.ndarray):
            dur_pred = np.clip(dur_pred, 0.0, 32.0)
            if self._duv_duration_grid > 0:
                step = self._duv_duration_grid
                dur_pred = np.maximum(step, np.round(dur_pred / step) * step)

        annotations: dict[int, Dict[str, Any]] = {}
        for row_idx, event_idx in enumerate(event_indices):
            payload: dict[str, Any] = {"model": self._duv_model_name}
            if has_vel and isinstance(vel_pred, np.ndarray):
                mask_val = bool(vel_mask[row_idx]) if vel_mask is not None else False
                payload["velocity_mask"] = mask_val
                if mask_val and row_idx < len(vel_pred):
                    payload["velocity"] = int(vel_pred[row_idx])
            if has_dur and isinstance(dur_pred, np.ndarray):
                mask_val = bool(dur_mask[row_idx]) if dur_mask is not None else False
                payload["duration_mask"] = mask_val
                if mask_val and row_idx < len(dur_pred):
                    payload["duration_ql"] = round(float(dur_pred[row_idx]), 5)
            payload = {k: v for k, v in payload.items() if v is not None}
            if payload.get("velocity") is not None or payload.get("duration_ql") is not None:
                annotations[event_idx] = payload
        return annotations

    # ------------------------------------------------------------------
    def _prepare_duv_dataframe(
        self, events: list[Any], metadata: Mapping[str, Any]
    ) -> tuple[pd.DataFrame, list[int]] | None:
        rows: list[dict[str, Any]] = []
        event_indices: list[int] = []
        default_section = str(metadata.get("default_section", "verse")).lower()
        section_ids: dict[str, int] = {}
        bar_positions: dict[int, int] = defaultdict(int)
        program = int(self._duv_program)
        track_id = int(self._duv_cfg.get("track_id", 0))
        for idx, event in enumerate(events):
            pitch = event.get("pitch")
            if pitch is None:
                pitch = event.get("note")
            if pitch is None:
                continue
            duration = float(event.get("duration_ql") or event.get("duration") or 0.0)
            if duration <= 0.0:
                continue
            velocity = event.get("velocity")
            if velocity is None:
                velocity = metadata.get("default_velocity", 80)
            velocity = int(max(1, min(127, int(velocity))))
            time_ql = event.get("time_ql")
            if time_ql is None:
                time_ql = event.get("start_ql")
            time_ql = float(time_ql or 0.0)
            bar_idx = int(
                event.get("bar_idx") if event.get("bar_idx") is not None else time_ql // 4.0
            )
            rel_beat = time_ql - (bar_idx * 4.0)
            if math.isfinite(rel_beat):
                position = int(max(0, min(63, round(rel_beat * 4.0))))
            else:
                position = bar_positions[bar_idx]
            bar_positions[bar_idx] = position + 1
            section_label = event.get("section") or event.get("section_label")
            if not section_label:
                section_label = default_section
            section_norm = str(section_label).lower()
            section_id = section_ids.setdefault(section_norm, len(section_ids))
            start_val = max(time_ql, 0.0)
            bar_phase = rel_beat / 4.0 if math.isfinite(rel_beat) else 0.0
            beat_phase = rel_beat - math.floor(rel_beat) if math.isfinite(rel_beat) else 0.0
            if not math.isfinite(beat_phase):
                beat_phase = 0.0
            beat_phase = (beat_phase + 1.0) % 1.0
            rows.append(
                {
                    "track_id": track_id,
                    "bar": bar_idx,
                    "position": position,
                    "pitch": int(pitch),
                    "velocity": float(velocity),
                    "duration": float(duration),
                    "section": section_id,
                    "program": program,
                    "start": start_val,
                    "onset": start_val,
                    "bar_phase": max(0.0, min(1.0, bar_phase)) if math.isfinite(bar_phase) else 0.0,
                    "beat_phase": beat_phase,
                }
            )
            event_indices.append(idx)
        if not rows:
            return None
        df = pd.DataFrame(rows)
        df["bar_phase"] = df["bar_phase"].clip(0.0, 1.0)
        df["beat_phase"] = df["beat_phase"].mod(1.0)
        return df, event_indices

    # ------------------------------------------------------------------
    def _extract_emotion_map(self, metadata: Mapping[str, Any]) -> dict[int, dict[str, Any]]:
        tracking = metadata.get("emotion_tracking") if isinstance(metadata, Mapping) else None
        if not isinstance(tracking, Mapping):
            return {}
        per_bar = tracking.get("per_bar")
        if not isinstance(per_bar, Mapping):
            return {}
        result: dict[int, dict[str, Any]] = {}
        for key, value in per_bar.items():
            if not isinstance(value, Mapping):
                continue
            try:
                bar_idx = int(key)
            except (TypeError, ValueError):
                continue
            snapshot = _enrich_emotion_snapshot(dict(value))
            result[bar_idx] = snapshot
        return result

    # ------------------------------------------------------------------
    def _infer_bar_idx(self, event: Mapping[str, Any]) -> int:
        bar_value = event.get("bar_idx")
        if bar_value is not None:
            try:
                return int(bar_value)
            except (TypeError, ValueError):
                pass
        time_ql = event.get("time_ql")
        if time_ql is None:
            time_ql = event.get("start_ql")
        try:
            time_val = float(time_ql or 0.0)
        except (TypeError, ValueError):
            time_val = 0.0
        return int(max(0, math.floor(time_val / 4.0)))


def _normalize_density(label: Optional[str]) -> str:
    if not label:
        return "medium"
    lowered = label.lower()
    if lowered in {"low", "lo"}:
        return "sparse"
    if lowered in {"mid", "mid_low"}:
        return "medium"
    if lowered in {"mid_high", "high", "hi"}:
        return "dense"
    return lowered


def _coerce_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _clamp_float(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _lerp(lo: float, hi: float, alpha: float) -> float:
    alpha = max(0.0, min(1.0, alpha))
    return lo + (hi - lo) * alpha

def _enrich_emotion_snapshot(snapshot: dict[str, Any]) -> dict[str, Any]:
    energy = _coerce_float(snapshot.get("energy"))
    valence = _coerce_float(snapshot.get("valence"))
    if "velocity_scale" not in snapshot:
        derived_vel = _derive_velocity_scale(energy, valence)
        if derived_vel is not None:
            snapshot["velocity_scale"] = derived_vel
    if "density_scale" not in snapshot:
        derived_density = _derive_density_scale(energy)
        if derived_density is not None:
            snapshot["density_scale"] = derived_density
    return snapshot


def _derive_velocity_scale(energy: float | None, valence: float | None) -> float | None:
    if energy is None and valence is None:
        return None
    energy_term = _clamp_float(energy, 0.0, 1.0) if energy is not None else 0.5
    valence_term = _clamp_float(valence, 0.0, 1.0) if valence is not None else 0.5
    scale = 1.0 + (energy_term - 0.5) * 0.5 + (valence_term - 0.5) * 0.2
    return _clamp_float(scale, 0.5, 1.5)


def _derive_density_scale(energy: float | None) -> float | None:
    if energy is None:
        return None
    energy_term = _clamp_float(energy, 0.0, 1.0)
    scale = 0.7 + energy_term * 0.6
    return _clamp_float(scale, 0.5, 1.5)

