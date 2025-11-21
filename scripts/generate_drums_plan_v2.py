#!/usr/bin/env python3
"""
Generate Drums Plan (V2) - Slot-Based Fill/Riff System with Collaborative Architecture
========================================================================================

Purpose:
    Read fill_slot from bars_with_slots.parquet and policy YAML to generate
    drums plan with GUARANTEED boundary fills and energy-responsive patterns.

Design Philosophy (2025-11-12):
    「位置決め（スロット）は bars/sections。表現の造形は楽器別レンダラ。」
    - bars.parquet: fill_slot (where to fire)
    - policy YAML: how to fire (density, length, accent patterns)
    - music21: NOT used for drums (harmony support only)

Collaborative Architecture (三段ロケット):
    1. recommend_drums (optional): Suggest fill/riff patterns based on context
    2. generate_drums_plan_v2 (core): Slot-based rendering with section density
    3. adapt_drums_to_plan (optional): Kit conversion, humanization, layer adjustment

    This design preserves existing assets while centralizing slot/tempo/density logic.

Integration:
    Called from make_song_package_from_sources.sh STEP 19 (or similar)

    Input:
        - bars_with_slots.parquet (fill_slot, section_label, drums_active, energy_curve)
        - sections.json (section boundaries)
        - policy/song_004.yaml (drums config)
        - --use-recommender (optional): Call recommend_drums for pattern suggestions
        - --post-adapt (optional): Call adapt_drums_to_plan for kit/humanization

    Output:
        - plans/drums_plan.json (events with time_ql, note, velocity, duration_ql)

Quality Gate:
    - Boundary fill rate ≥ 80% (checked by quality_gate_fill_riff.py)
    - Max 16th density < 75% (over-density prevention)
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional

import pandas as pd
import yaml
import numpy as np

# Import shared utilities
CURRENT_DIR = Path(__file__).parent
sys.path.insert(0, str(CURRENT_DIR))
sys.path.insert(0, str(CURRENT_DIR.parent))
from v2_common import (
    ensure_activity_floor,
    apply_humanize,
)
from ai_hook_utils import load_reference_layers

from otobonAI.rhythm_ai import RhythmAI, RhythmContext, RhythmCandidate

try:
    from otobonAI.lyric_index import LyricAnchorIndex
    from otobonAI.emotion_ai_v2 import EmotionAI as EmotionAIv2
    from otobonAI.guide_tone_ai_v2 import GuideToneAI as GuideToneAIv2
    from otobonAI.rulebook_engine import Rulebook
except ImportError as exc:  # pragma: no cover - optional dependency
    print(f"⚠️  OtobonAI Phase 2.0 modules unavailable for drums plan: {exc}")
    LyricAnchorIndex = None  # type: ignore
    EmotionAIv2 = None  # type: ignore
    GuideToneAIv2 = None  # type: ignore
    Rulebook = None  # type: ignore

# GM Drum Map (Standard Kit)
DRUM_MAP = {
    "kick": 36,  # Bass Drum 1
    "snare": 38,  # Acoustic Snare
    "closed_hh": 42,  # Closed Hi-Hat
    "open_hh": 46,  # Open Hi-Hat
    "floor_tom": 41,  # Low Floor Tom
    "mid_tom": 47,  # Low-Mid Tom
    "high_tom": 50,  # High Tom
    "crash": 49,  # Crash Cymbal 1
    "ride": 51,  # Ride Cymbal 1
}


def _safe_str(value: Any) -> Optional[str]:
    """Return None for NaN/empty values, otherwise a trimmed string."""

    if value is None:
        return None
    if isinstance(value, float) and np.isnan(value):
        return None
    text = str(value).strip()
    return text if text else None


def _attach_rhythmai_metadata(
    events: List[Dict[str, Any]], candidate: Optional[RhythmCandidate]
) -> None:
    """Annotate output events with RhythmAI provenance for downstream tooling."""

    if not candidate:
        return
    for ev in events:
        ev["rhythm_ai_pattern"] = candidate.pattern_family
        ev["rhythm_ai_source"] = candidate.source
        ev["rhythm_ai_swing"] = candidate.swing_class
        ev["rhythm_ai_density"] = candidate.density_bucket


def _adjust_fill_type(base_type: str, candidate: Optional[RhythmCandidate]) -> str:
    """Derive fill length heuristics using RhythmAI metadata when possible."""

    if not candidate:
        return base_type
    density = candidate.density_bucket
    if density in {"dense", "wall"}:
        return "long" if base_type != "short" else base_type
    if density == "sparse":
        return "short"
    return base_type


def load_bars(bars_path: Path) -> pd.DataFrame:
    """Load bars_with_slots.parquet with required columns."""
    bars = pd.read_parquet(bars_path)

    required_cols = ["bar_index", "fill_slot", "section_label", "start_sec", "end_sec"]
    optional_cols = ["drums_active", "energy_curve", "fill_likelihood"]

    missing = [c for c in required_cols if c not in bars.columns]
    if missing:
        raise ValueError(f"Missing required columns in bars.parquet: {missing}")

    # Fill missing optional columns
    for col in optional_cols:
        if col not in bars.columns:
            bars[col] = 0.5  # Default neutral value

    return bars


def load_policy(policy_path: Path) -> Dict[str, Any]:
    """Load policy YAML and extract drums config."""
    with open(policy_path, encoding="utf-8") as f:
        policy = yaml.safe_load(f)

    if "instruments" not in policy or "drums" not in policy["instruments"]:
        raise ValueError("policy YAML missing instruments.drums section")

    return policy


def load_sections(sections_path: Path) -> List[Dict[str, Any]]:
    """Load sections.json for section boundary detection."""
    with open(sections_path, encoding="utf-8") as f:
        data = json.load(f)

    # Handle both formats: list or {"sections": [...]}
    if isinstance(data, list):
        sections = data
    elif isinstance(data, dict) and "sections" in data:
        sections = data["sections"]
    else:
        raise ValueError("sections.json must be a list or have 'sections' key")

    return sections


def is_section_boundary(bar_idx: int, bars: pd.DataFrame, sections: List[Dict[str, Any]]) -> bool:
    """Check if bar_idx is at section end-1 (preparation for boundary)."""
    for sec in sections:
        # Section boundary is typically at end_bar-1 (e.g., bar 7 before bar 8 section start)
        if "end_bar" in sec:
            if bar_idx == sec["end_bar"] - 1:
                return True
        # Fallback: check if next bar has different section_label
        if bar_idx + 1 < len(bars):
            if bars.iloc[bar_idx]["section_label"] != bars.iloc[bar_idx + 1]["section_label"]:
                return True

    return False


def make_backbeat_pattern(
    bar_idx: int,
    bar_data: pd.Series,
    drums_cfg: Dict[str, Any],
    section_density: float,
    ai_choice: Optional[RhythmCandidate] = None,
    emotion_params=None,
    guide_params=None,
) -> List[Dict[str, Any]]:
    """
    Generate standard backbeat pattern (kick/snare/hh).

    Args:
        bar_idx: Bar index
        bar_data: Row from bars.parquet
        drums_cfg: policy['instruments']['drums']
        section_density: sections[section_label]['drums']
        ai_choice: RhythmAI candidate metadata (optional)
        emotion_params: EmotionAI parameters (optional)
        guide_params: GuideToneAI plan (optional)

    Returns:
        List of drum events (time_ql, note, velocity, duration_ql)
    """
    start_ql = bar_idx * 4.0  # Assume 4/4 time signature (4 quarter notes per bar)
    events = []
    pattern_family = ai_choice.pattern_family if ai_choice else "backbeat"
    swing_class = ai_choice.swing_class if ai_choice else "straight"
    density_bucket = ai_choice.density_bucket if ai_choice else None

    velocity_scale = 1.0
    density_scale = 1.0
    phrase_velocity_boost = 0

    if emotion_params is not None:
        velocity_scale = float(getattr(emotion_params, "velocity_scale", 1.0) or 1.0)
        density_attr = getattr(emotion_params, "density_scale", None)
        if density_attr is not None:
            try:
                density_scale *= float(density_attr)
            except (TypeError, ValueError):
                pass

    if guide_params is not None:
        notes_hint = getattr(guide_params, "notes_per_bar", None)
        if notes_hint is not None:
            try:
                density_scale *= float(np.clip(float(notes_hint) / 4.0, 0.5, 1.75))
            except (TypeError, ValueError):
                pass

        phrase_shape = getattr(guide_params, "phrase_shape", None)
        if phrase_shape == "uphill":
            phrase_velocity_boost = 8
        elif phrase_shape == "downhill":
            phrase_velocity_boost = -6

    # Effective density = section_density * drums_active * energy_curve
    drums_active = bar_data.get("drums_active", 0.5)
    energy = bar_data.get("energy_curve", 0.5)

    # Handle NaN values (replace with neutral 0.5)
    if pd.isna(drums_active):
        drums_active = 0.5
    if pd.isna(energy):
        energy = 0.5

    effective_density = section_density * drums_active * energy * density_scale
    if density_bucket == "sparse":
        effective_density *= 0.75
    elif density_bucket in {"dense", "wall"}:
        effective_density = min(1.0, effective_density * 1.2)

    # Base velocity (scaled by energy and emotion)
    base_velocity = int((80 + 25 * energy) * velocity_scale)
    base_velocity = int(np.clip(base_velocity + phrase_velocity_boost, 40, 127))

    # Humanization
    humanize_ms = drums_cfg.get("humanize_timing_ms", 12)
    humanize_vel = drums_cfg.get("humanize_velocity", 8)

    # Kick pattern (beats vary based on RhythmAI family)
    kick_template = [0, 2]
    if pattern_family == "four_on_floor" or pattern_family == "double_time":
        kick_template = [0, 1, 2, 3]
    elif pattern_family == "half_time":
        kick_template = [0, 3]
    elif pattern_family == "minimal":
        kick_template = [0]

    if effective_density > 0.2:
        for beat in kick_template:
            time_ql = start_ql + beat + np.random.uniform(-humanize_ms / 1000, humanize_ms / 1000)
            vel = base_velocity + np.random.randint(-humanize_vel, humanize_vel)
            events.append(
                {
                    "bar_idx": bar_idx,
                    "time_ql": float(time_ql),
                    "note": int(DRUM_MAP["kick"]),
                    "velocity": int(np.clip(vel, 40, 127)),
                    "duration_ql": 0.25,
                    "type": "backbeat_kick",
                    "pattern": "backbeat",
                    "event_type": "groove",
                }
            )

    # Snare pattern (half-time tweak when requested)
    snare_template = [1, 3]
    if pattern_family == "half_time":
        snare_template = [2]

    if effective_density > 0.2:
        for beat in snare_template:
            time_ql = start_ql + beat + np.random.uniform(-humanize_ms / 1000, humanize_ms / 1000)
            vel = base_velocity + 10 + np.random.randint(-humanize_vel, humanize_vel)
            events.append(
                {
                    "bar_idx": bar_idx,
                    "time_ql": float(time_ql),
                    "note": int(DRUM_MAP["snare"]),
                    "velocity": int(np.clip(vel, 50, 127)),
                    "duration_ql": 0.25,
                    "type": "backbeat_snare",
                    "pattern": "backbeat",
                    "event_type": "groove",
                }
            )

    # Hi-hat pattern (density + RhythmAI swing)
    hat_density_cfg = drums_cfg.get("hat_density_curve", "match_energy_curve")
    hat_prob = effective_density if hat_density_cfg == "match_energy_curve" else 0.5
    hat_prob = float(np.clip(hat_prob, 0.05, 1.0))
    if density_bucket == "sparse":
        hat_prob *= 0.6
    elif density_bucket in {"dense", "wall"}:
        hat_prob = min(1.0, hat_prob * 1.35)

    hat_step = 0.5
    if density_bucket == "sparse":
        hat_step = 1.0
    elif density_bucket in {"dense", "wall"}:
        hat_step = 0.25

    steps = max(1, int(round(4 / hat_step)))
    if hat_prob > 0.2:
        for step in range(steps):
            if np.random.random() >= hat_prob:
                continue
            swing_offset = 0.0
            if swing_class in {"shuffle", "swing"} and hat_step <= 0.5 and step % 2 == 1:
                swing_offset = hat_step * 0.3
            time_ql = start_ql + step * hat_step + swing_offset
            vel = base_velocity - 20 + np.random.randint(-humanize_vel, humanize_vel)
            events.append(
                {
                    "bar_idx": bar_idx,
                    "time_ql": float(time_ql),
                    "note": int(DRUM_MAP["closed_hh"]),
                    "velocity": int(np.clip(vel, 30, 100)),
                    "duration_ql": hat_step / 2,
                    "type": "backbeat_hihat",
                    "pattern": pattern_family,
                    "event_type": "groove",
                }
            )

    return events


def make_fill_pattern(
    bar_idx: int,
    bar_data: pd.Series,
    drums_cfg: Dict[str, Any],
    section_density: float,
    fill_type: str = "standard",
    emotion_params=None,
    guide_params=None,
) -> List[Dict[str, Any]]:
    """
    Generate fill pattern (buildup/uplifting/celebration).

    Args:
        bar_idx: Bar index
        bar_data: Row from bars.parquet
        drums_cfg: policy['instruments']['drums']
        section_density: sections[section_label]['drums']
        fill_type: "short" (2 beats), "standard" (4 beats), "long" (8 beats)
        emotion_params: EmotionAI parameters (optional)
        guide_params: GuideToneAI plan (optional)

    Returns:
        List of drum events
    """
    start_ql = bar_idx * 4.0
    events = []

    # Determine fill length
    if fill_type == "short":
        fill_len_beats = 2.0
        # Get pattern from YAML (handle both dict and list formats)
        short_fill_cfg = drums_cfg.get("accent_patterns", {}).get("short_fill", [])
        if isinstance(short_fill_cfg, list) and len(short_fill_cfg) > 0:
            accent_pattern = short_fill_cfg[0].get("pattern", [0.5, 0.6, 0.7, 0.9])
        else:
            accent_pattern = [0.5, 0.6, 0.7, 0.9]

    elif fill_type == "long":
        fill_len_beats = 8.0
        long_fill_cfg = drums_cfg.get("accent_patterns", {}).get("long_fill", [])
        if isinstance(long_fill_cfg, list) and len(long_fill_cfg) > 0:
            accent_pattern = long_fill_cfg[0].get(
                "pattern", [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0]
            )
        else:
            accent_pattern = [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0]

    else:  # standard (4 beats)
        fill_len_beats = 4.0
        standard_fill_cfg = drums_cfg.get("accent_patterns", {}).get("standard_fill", [])
        if isinstance(standard_fill_cfg, list) and len(standard_fill_cfg) > 0:
            accent_pattern = standard_fill_cfg[0].get(
                "pattern", [0.6, 0.7, 0.8, 0.9, 0.95, 1.0, 0.9, 0.8]
            )
        else:
            accent_pattern = [0.6, 0.7, 0.8, 0.9, 0.95, 1.0, 0.9, 0.8]

    velocity_scale = 1.0
    duration_scale = 1.0
    density_scale = 1.0
    phrase_velocity_boost = 0

    if emotion_params is not None:
        velocity_scale = float(getattr(emotion_params, "velocity_scale", 1.0) or 1.0)
        duration_scale = float(getattr(emotion_params, "duration_scale", 1.0) or 1.0)
        density_attr = getattr(emotion_params, "density_scale", None)
        if density_attr is not None:
            try:
                density_scale *= float(density_attr)
            except (TypeError, ValueError):
                pass

    if guide_params is not None:
        notes_hint = getattr(guide_params, "notes_per_bar", None)
        if notes_hint is not None:
            try:
                density_scale *= float(np.clip(float(notes_hint) / 4.0, 0.5, 2.0))
            except (TypeError, ValueError):
                pass
        phrase_shape = getattr(guide_params, "phrase_shape", None)
        if phrase_shape == "uphill":
            phrase_velocity_boost = 10
        elif phrase_shape == "downhill":
            phrase_velocity_boost = -8

    fill_len_beats = float(np.clip(fill_len_beats * density_scale, 2.0, 8.0))

    # Humanization
    humanize_ms = drums_cfg.get("humanize_timing_ms", 12)
    humanize_vel = drums_cfg.get("humanize_velocity", 8)

    # Generate 16th note fill (4 notes per beat)
    num_notes = int(fill_len_beats * 4)

    # Tom pattern (low → mid → high)
    tom_sequence = [DRUM_MAP["floor_tom"], DRUM_MAP["mid_tom"], DRUM_MAP["high_tom"]]

    for i in range(num_notes):
        time_offset = i * 0.25  # 16th notes
        time_ql = (
            start_ql + time_offset + np.random.uniform(-humanize_ms / 1000, humanize_ms / 1000)
        )

        # Accent curve (0.5 → 1.0)
        accent_idx = int(i / num_notes * len(accent_pattern))
        accent = accent_pattern[min(accent_idx, len(accent_pattern) - 1)]

        # Select drum (toms for most, crash at peak)
        if i == num_notes - 1:
            note = int(DRUM_MAP["crash"])  # Crash at climax
            vel = int(np.clip(127 * accent * velocity_scale + phrase_velocity_boost, 40, 127))
        elif i % 4 == 0:
            note = int(DRUM_MAP["kick"])  # Kick on quarter notes
            vel = int(np.clip(100 * accent * velocity_scale + phrase_velocity_boost, 35, 127))
            vel += np.random.randint(-humanize_vel, humanize_vel)
        else:
            # Cycle through toms
            tom_idx = (i // 2) % len(tom_sequence)
            note = int(tom_sequence[tom_idx])
            vel = int(np.clip(90 * accent * velocity_scale + phrase_velocity_boost, 35, 127))
            vel += np.random.randint(-humanize_vel, humanize_vel)

        events.append(
            {
                "bar_idx": bar_idx,
                "time_ql": float(time_ql),
                "note": note,
                "velocity": int(np.clip(vel, 40, 127)),
                "duration_ql": float(0.25 * duration_scale),
                "type": f"fill_{fill_type}",
                "pattern": fill_type,
                "is_fill": True,
                "event_type": "fill",
            }
        )

    return events


def generate_drums_plan(
    bars: pd.DataFrame,
    sections: List[Dict[str, Any]],
    policy: Dict[str, Any],
    rng_seed: int = 42,
    tempo_bpm: float = 120.0,
    rhythm_ai: Optional[RhythmAI] = None,
    lyric_index: Optional[LyricAnchorIndex] = None,
    emotion_ai: Optional[EmotionAIv2] = None,
    guidetone_ai: Optional[GuideToneAIv2] = None,
    reference_layers: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Main generation logic: Read fill_slot and generate patterns.

    Strategy (from policy.integration.slot_reading.drums):
        - if fill_slot=True → force_fill
        - elif fill_likelihood > min_fill_prob → optional fill
        - else → backbeat pattern

    Boundary fill guarantee (policy.instruments.drums.boundary_fill=always):
        - All section end-1 bars get fill (overrides fill_slot)

    Args:
        bars: bars_with_slots.parquet
        sections: sections.json
        policy: policy YAML

    Returns:
        List of events [{time_ql, note, velocity, duration_ql}, ...]
    """
    drums_cfg = policy["instruments"]["drums"]
    sections_cfg = policy.get("sections", {})
    global_cfg = policy.get("global", {})
    global_h = global_cfg.get("humanize", {})
    min_notes = drums_cfg.get("min_notes_per_bar", 1)

    # Initialize RNG
    rng = np.random.RandomState(rng_seed)

    # RhythmAI context defaults
    fallback_tempo = float(tempo_bpm or 120.0)
    emotion_cfg = global_cfg.get("emotion")
    if isinstance(emotion_cfg, dict):
        default_emotion = emotion_cfg.get("primary") or emotion_cfg.get("mood")
    else:
        default_emotion = emotion_cfg

    # Get global settings
    boundary_fill_mode = drums_cfg.get("boundary_fill", "auto")  # never/auto/always
    min_fill_prob = drums_cfg.get("min_fill_prob", 0.15)

    all_events = []
    reference_layers = reference_layers or {}

    for idx, bar in bars.iterrows():
        bar_idx = bar["bar_index"]
        section_label = bar["section_label"]
        bar_start_ql = float(bar.get("start_ql", bar_idx * 4.0))
        bar_end_ql = float(bar.get("end_ql", (bar_idx + 1) * 4.0))
        bar_tempo = bar.get("tempo_bpm", fallback_tempo)
        if pd.isna(bar_tempo):
            bar_tempo = fallback_tempo
        bar_tempo = float(bar_tempo)

        # Get section density (fallback to 0.5 if missing)
        section_density = sections_cfg.get(section_label, {}).get("drums", 0.5)

        lyric_info = None
        if lyric_index:
            lyric_info = lyric_index.get_bar_info(bar_idx)

        ai_context = {
            "bar_index": int(bar_idx),
            "bar": int(bar_idx),
            "section": section_label,
            "role": "drums",
            "chord_symbol": None,
            "slots": {"fill": bool(bar.get("fill_slot", False))},
        }
        if lyric_info and lyric_info.get("has_anchor"):
            ai_context["lyric"] = {
                "phrase_role": lyric_info["phrase_role"],
                "stress_level": lyric_info.get("stress_level", 0.0),
                "is_silent": lyric_info.get("is_silent", False),
            }
        if reference_layers:
            ai_context["reference_layers"] = reference_layers

        emotion_params = None
        if emotion_ai:
            try:
                emotion_params = emotion_ai.get_params(ai_context)
            except Exception as exc:
                print(f"⚠️  EmotionAI v2 error at bar {bar_idx}: {exc}")

        guide_params = None
        if guidetone_ai:
            try:
                guide_params = guidetone_ai.get_plan(ai_context)
            except Exception as exc:
                print(f"⚠️  GuideToneAI v2 error at bar {bar_idx}: {exc}")

        if emotion_params is not None:
            density_scale = getattr(emotion_params, "density_scale", None)
            if density_scale is not None:
                try:
                    section_density *= float(density_scale)
                except (TypeError, ValueError):
                    pass

        # Check if boundary bar
        is_boundary = is_section_boundary(bar_idx, bars, sections)

        # Determine if fill should fire
        should_fill = False
        fill_type = "standard"

        # 1. Boundary fill (highest priority)
        if boundary_fill_mode == "always" and is_boundary:
            should_fill = True
            # Outro gets long fill
            if section_label == "outro":
                fill_type = "long"

        # 2. Slot-based fill (from bars.parquet)
        elif bar.get("fill_slot", False):
            should_fill = True

        # 3. Fallback to fill_likelihood
        elif bar.get("fill_likelihood", 0) > min_fill_prob:
            # Energy jump check
            energy_jump_thresh = drums_cfg.get("energy_jump_thresh", 0.06)
            if idx > 0:
                prev_energy = bars.iloc[idx - 1].get("energy_curve", 0.5)
                curr_energy = bar.get("energy_curve", 0.5)
                if abs(curr_energy - prev_energy) > energy_jump_thresh:
                    should_fill = True

        # Generate pattern
        ai_candidate: Optional[RhythmCandidate] = None
        raw_energy = bar.get("energy_curve", 0.5)
        if pd.isna(raw_energy):
            raw_energy = 0.5
        energy_for_ai = float(np.clip(raw_energy, 0.0, 1.0) * 100.0)
        riff_slot = bool(bar.get("riff_slot", False))
        drum_label = _safe_str(bar.get("drum_label"))
        emotion_label = _safe_str(bar.get("emotion")) or _safe_str(bar.get("emotion_label"))
        emotive_context = emotion_label or _safe_str(default_emotion)
        groove_slot = bool(bar.get("groove_slot", False))
        style_hint = _safe_str(bar.get("drum_style_hint"))
        if style_hint == "neutral":
            style_hint = ""
        vocal_profile = _safe_str(bar.get("vocal_profile"))
        if not style_hint:
            style_hint = vocal_profile
        vocal_voiced_ratio = float(bar.get("vocal_voiced_ratio", 0.0) or 0.0)
        vocal_voiced_ratio = float(np.clip(vocal_voiced_ratio, 0.0, 1.2))
        vocal_profile_conf = float(bar.get("vocal_profile_confidence", 0.0) or 0.0)
        vocal_profile_conf = float(np.clip(vocal_profile_conf, 0.0, 1.0))
        if not style_hint:
            if should_fill:
                style_hint = "fill"
            elif groove_slot:
                style_hint = "groove"

        if rhythm_ai:
            ctx = RhythmContext(
                section_label=str(section_label),
                tempo_bpm=bar_tempo,
                energy=energy_for_ai,
                drum_label=drum_label,
                emotion=emotive_context,
                style_hint=style_hint,
                fill_slot=bool(should_fill),
                riff_slot=riff_slot,
                vocal_voiced_ratio=vocal_voiced_ratio,
                vocal_profile=vocal_profile,
                vocal_profile_confidence=vocal_profile_conf,
            )
            try:
                ai_candidate = rhythm_ai.choose_pattern(ctx)
            except Exception as exc:  # pragma: no cover - guard rail
                print(f"⚠️  RhythmAI failed on bar {bar_idx}: {exc}")

        if guide_params is not None:
            phrase_shape = getattr(guide_params, "phrase_shape", None)
            if should_fill:
                if phrase_shape == "uphill":
                    fill_type = "long"
                elif phrase_shape == "downhill":
                    fill_type = "short"
            elif phrase_shape == "burst":
                section_density = min(1.0, section_density * 1.15)

        if should_fill:
            fill_type = _adjust_fill_type(fill_type, ai_candidate)

        if should_fill:
            events = make_fill_pattern(
                bar_idx,
                bar,
                drums_cfg,
                section_density,
                fill_type,
                emotion_params=emotion_params,
                guide_params=guide_params,
            )
        else:
            events = make_backbeat_pattern(
                bar_idx,
                bar,
                drums_cfg,
                section_density,
                ai_candidate,
                emotion_params=emotion_params,
                guide_params=guide_params,
            )

        # Activity floor (for drums, use kick/snare MIDI notes)
        chordpad_pitches = [36, 38, 42]  # Kick, Snare, Closed HH
        events = ensure_activity_floor(
            events, bar_start_ql, bar_end_ql, min_notes, chordpad_pitches, velocity=70
        )
        _attach_rhythmai_metadata(events, ai_candidate)

        for ev in events:
            ev.setdefault("bar_idx", int(bar_idx))
            ev.setdefault("section_label", section_label)

        all_events.extend(events)

    # Apply humanization
    all_events = apply_humanize(
        all_events,
        timing_std_ms=global_h.get("timing_std_ms", 8),
        velocity_jitter=global_h.get("velocity_jitter", 6),
        legato_chance=global_h.get("legato_chance", 0.0),
        rng=rng,
    )

    return all_events


def main():
    parser = argparse.ArgumentParser(
        description="Generate drums plan from slots and policy (V2: collaborative)"
    )
    parser.add_argument("--bars", type=Path, required=True, help="Path to bars_with_slots.parquet")
    parser.add_argument("--sections", type=Path, required=True, help="Path to sections.json")
    parser.add_argument(
        "--policy", type=Path, required=True, help="Path to policy YAML (e.g., song_004.yaml)"
    )
    parser.add_argument("--out", type=Path, required=True, help="Output path for drums_plan.json")
    parser.add_argument("--lyric-anchors", help="Path to lyric_anchors.json (optional)")
    parser.add_argument("--emotion-profile", help="Path to emotion_profile.json (optional)")
    parser.add_argument("--guide-hints", help="Path to guide_tone_hints.json (optional)")
    parser.add_argument("--rulebook", help="Path to rulebook.yaml (optional)")
    parser.add_argument("--vocal-f0", help="Path to vocal_f0_crepe.parquet (optional)")
    parser.add_argument("--piano-oaf", help="Path to piano_onsets_and_frames.json (optional)")
    parser.add_argument(
        "--tempo-bpm",
        type=float,
        default=120.0,
        help="Fallback tempo used when bars data lacks tempo information",
    )
    parser.add_argument(
        "--groove-vocab",
        type=Path,
        default=Path("data/groove_vocab.parquet"),
        help="Path to groove vocab parquet for RhythmAI",
    )
    parser.add_argument(
        "--disable-rhythmai",
        action="store_true",
        help="Disable RhythmAI integration even if groove vocab exists",
    )

    # Collaborative architecture hooks
    parser.add_argument(
        "--use-recommender",
        action="store_true",
        help="Call recommend_drums.py for pattern suggestions (optional, not implemented yet)",
    )
    parser.add_argument(
        "--post-adapt",
        action="store_true",
        help="Call adapt_drums_to_plan.py for kit conversion/humanization (optional, not implemented yet)",
    )

    args = parser.parse_args()
    groove_vocab_path = args.groove_vocab.expanduser().resolve()

    rhythm_ai = None
    rhythm_meta: Dict[str, Any] = {"enabled": False}
    if not args.disable_rhythmai:
        try:
            rhythm_ai = RhythmAI(groove_vocab_path)
            rhythm_meta = {
                "enabled": True,
                "vocab_path": str(groove_vocab_path),
                "vocab_ready": rhythm_ai.is_ready(),
            }
        except Exception as exc:  # pragma: no cover - initialization guard
            print(f"⚠️  RhythmAI initialization failed: {exc}")
            rhythm_meta = {
                "enabled": False,
                "vocab_path": str(groove_vocab_path),
                "error": str(exc),
            }
    else:
        rhythm_meta["reason"] = "disabled_via_flag"

    # Load inputs
    print(f"📖 Loading bars from {args.bars}")
    bars = load_bars(args.bars)

    print(f"📖 Loading sections from {args.sections}")
    sections = load_sections(args.sections)

    print(f"📖 Loading policy from {args.policy}")
    policy = load_policy(args.policy)

    reference_layers = load_reference_layers(args.vocal_f0, args.piano_oaf)
    if reference_layers:
        print(
            "🔗 Reference layers loaded:",
            ", ".join(
                f"{k}={v.get('frames', v.get('notes', 0))}" for k, v in reference_layers.items()
            ),
        )

    rulebook = None
    lyric_index = None
    emotion_ai = None
    guidetone_ai = None

    if args.rulebook and Rulebook:
        print(f"📖 Loading Rulebook from {args.rulebook}")
        rulebook = Rulebook.load(Path(args.rulebook))

    if args.lyric_anchors and LyricAnchorIndex:
        print(f"📝 Loading LyricAnchorIndex from {args.lyric_anchors}")
        with open(args.lyric_anchors, "r", encoding="utf-8") as f:
            anchors_data = json.load(f)
        tempo_bpm = policy.get("global", {}).get("tempo_bpm", 120)
        lyric_index = LyricAnchorIndex(anchors=anchors_data, tempo_bpm=tempo_bpm)

    if args.emotion_profile and EmotionAIv2 and rulebook:
        print(f"🎭 Loading EmotionAI v2 from {args.emotion_profile}")
        with open(args.emotion_profile, "r", encoding="utf-8") as f:
            emotion_profile_data = json.load(f)
        emotion_ai = EmotionAIv2(emotion_profile=emotion_profile_data, rulebook=rulebook)

    if args.guide_hints and GuideToneAIv2 and rulebook:
        print(f"🎵 Loading GuideToneAI v2 from {args.guide_hints}")
        with open(args.guide_hints, "r", encoding="utf-8") as f:
            guide_tone_data = json.load(f)
        guidetone_ai = GuideToneAIv2(guide_tone_hints=guide_tone_data, rulebook=rulebook)

    # Generate plan
    print(f"🥁 Generating drums plan ({len(bars)} bars)")
    events = generate_drums_plan(
        bars,
        sections,
        policy,
        tempo_bpm=args.tempo_bpm,
        rhythm_ai=rhythm_ai,
        lyric_index=lyric_index,
        emotion_ai=emotion_ai,
        guidetone_ai=guidetone_ai,
        reference_layers=reference_layers,
    )

    # Sort by time
    events.sort(key=lambda e: e["time_ql"])
    rhythm_meta["events_tagged"] = int(sum("rhythm_ai_pattern" in ev for ev in events))

    # TODO: Collaborative hooks (future implementation)
    if args.use_recommender:
        print("   ℹ️  --use-recommender: Not implemented yet (recommend_drums.py integration)")
    if args.post_adapt:
        print("   ℹ️  --post-adapt: Not implemented yet (adapt_drums_to_plan.py integration)")

    # Wrap in plan structure
    plan = {
        "instrument": "drums",
        "events": events,
        "metadata": {
            "generator": "generate_drums_plan_v2.py",
            "num_bars": int(len(bars)),
            "num_events": len(events),
            "fill_slots_used": int(bars["fill_slot"].sum()),
            "boundary_fill_mode": policy["instruments"]["drums"].get("boundary_fill", "auto"),
            "rhythm_ai": rhythm_meta,
            "ai_hooks": {
                "lyric_anchor": bool(lyric_index),
                "emotion_ai": bool(emotion_ai),
                "guide_tone_ai": bool(guidetone_ai),
                "reference_layers": bool(reference_layers),
            },
        },
    }

    if reference_layers:
        plan["metadata"]["reference_layers"] = reference_layers

    # Write output
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(plan, f, indent=2, ensure_ascii=False)

    print(f"✅ Drums plan saved to {args.out}")
    print(f"   Events: {len(events)}")
    print(f"   Fill slots used: {plan['metadata']['fill_slots_used']}/{len(bars)}")
    print(f"   Boundary fill mode: {plan['metadata']['boundary_fill_mode']}")


if __name__ == "__main__":
    main()
