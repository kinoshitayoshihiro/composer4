#!/usr/bin/env python3
"""
generate_strings_plan_v2.py - Slot-based strings renderer for fill/riff system.

Architecture:
- Slot Planner: bars_with_slots.parquet (riff_slot: where to fire)
- Policy YAML: density/countermelody_styles/crescendo (how to fire)
- Chord Source: manual_chordmap.json (what notes to play)
- Melody Reference: vocal_f0_crepe.parquet (if available, for call-response)
- Output: plans/strings_plan.json

Design Philosophy:
"位置決めはbars/sections。造形は楽器別レンダラ。music21は和声支援のみ。"
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import yaml

# Import shared chordmap utilities
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))  # For otobonAI
from chordmap_utils import load_chordmap, get_chord_at_bar, parse_symbol, get_chord_tones
from ai_hook_utils import load_reference_layers
from melody_hint_utils import (
    MelodyHint,
    apply_melody_hint_filter,
    build_melody_hint_manifest_payload,
    build_melody_hint_table,
    summarize_melody_hints,
)

# Import OtobonAI (Phase 2.0)
try:
    from otobonAI.lyric_index import LyricAnchorIndex
    from otobonAI.emotion_ai_v2 import EmotionAI as EmotionAIv2
    from otobonAI.guide_tone_ai_v2 import GuideToneAI as GuideToneAIv2
    from otobonAI.rulebook_engine import Rulebook

    OTOBON_AI_AVAILABLE = True
except ImportError as e:
    OTOBON_AI_AVAILABLE = False
    print(f"⚠️  OtobonAI Phase 2.0 not available: {e}")
try:
    from otobonAI.duration_humanize_ai import DurationHumanizeAI
except Exception as exc:  # pragma: no cover - optional dependency
    print(f"⚠️  DurationHumanizeAI unavailable: {exc}")
    DurationHumanizeAI = None  # type: ignore
from v2_common import (
    ensure_activity_floor,
    choose_tension_enabled,
    ensure_register,
    spread_open_voicing,
    select_role,
    decide_tension_use,
    choose_tensions,
    pitch_of,
    resolve_tension_ratio,
    load_humanize_config,  # Phase 3: Humanize config
    apply_timing_humanize,  # Phase 3: Timing humanize
    apply_velocity_humanize,  # Phase 3: Velocity humanize
    apply_duration_humanize,  # Phase 3.5: Duration humanize
    extend_last_note_per_bar,  # Phase 3.6: Long note extension
    apply_rhythm_vocab_annotations,
    record_emotion_snapshot,
    summarize_emotion_log,
)
from v2_guide_tone import (  # Phase 3.7: Guide tone melodies
    load_guide_tone_config,
    generate_guide_tone_events,
    save_guide_tone_report,
)

# Strings range: G3 (MIDI 55) - E6 (MIDI 88)
STRINGS_MIN_PITCH = 55
STRINGS_MAX_PITCH = 88


def load_bars(bars_path: str) -> pd.DataFrame:
    """Load bars.parquet with riff_slot."""
    bars = pd.read_parquet(bars_path)
    required = ["section_label"]
    missing = [c for c in required if c not in bars.columns]
    if missing:
        raise ValueError(f"bars.parquet missing columns: {missing}")

    if "riff_slot" not in bars.columns:
        bars["riff_slot"] = 0
        print("ℹ️  bars file missing riff_slot column, defaulting to 0")

    # Ensure bar_idx exists
    if "bar_index" in bars.columns and "bar_idx" not in bars.columns:
        bars = bars.rename(columns={"bar_index": "bar_idx"})
    elif "bar_idx" not in bars.columns:
        bars["bar_idx"] = range(len(bars))

    return bars


def load_sections(sections_path: str) -> List[Dict[str, Any]]:
    """Load sections.json."""
    with open(sections_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    elif isinstance(data, dict) and "sections" in data:
        return data["sections"]
    else:
        raise ValueError('sections.json must be a list or {"sections": [...]}')


def load_chordmap(chordmap_path: str) -> List[Dict[str, Any]]:
    """Load chordmap_locked_extended.json."""
    with open(chordmap_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    elif isinstance(data, dict) and "events" in data:
        return data["events"]
    else:
        raise ValueError('chordmap must be a list or {"events": [...]}')


def load_vocal_f0(f0_path: Optional[str]) -> Optional[pd.DataFrame]:
    """Load vocal_f0_crepe.parquet (optional)."""
    if not f0_path or not Path(f0_path).exists():
        return None
    try:
        return pd.read_parquet(f0_path)
    except Exception:
        return None


def get_chord_at_bar(chordmap: List[Dict[str, Any]], bar_idx: int) -> Dict[str, Any]:
    """Find chord event overlapping with bar_idx."""
    bar_start_ql = bar_idx * 4.0
    for chord in chordmap:
        chord_start = chord.get("time_ql", 0.0)
        chord_end = chord_start + chord.get("duration_ql", 4.0)
        if chord_start <= bar_start_ql < chord_end:
            return chord
    return chordmap[0] if chordmap else {}


def make_countermelody(
    bar_idx: int,
    bar_data: pd.Series,
    chord: Dict[str, Any],
    strings_cfg: Dict[str, Any],
    policy: Dict[str, Any],
    section_label: str,
    countermelody_type: str = "call_response",
    vocal_f0: Optional[pd.DataFrame] = None,
    guide_params=None,  # Phase 2.0: GuideTonePlan dataclass
) -> List[Dict[str, Any]]:
    """
    Generate strings countermelody (Phase 2.0).

    Args:
        bar_idx: Bar index
        bar_data: Row from bars.parquet
        chord: Chord from chordmap
        strings_cfg: policy['instruments']['strings']
        policy: Full policy dictionary
        section_label: verse, chorus, etc.
        countermelody_type: "call_response", "ascending_line", "sustain_pad"
        vocal_f0: vocal_f0_crepe.parquet (optional)
        guide_params: GuideTonePlan (Phase 2.0, optional)

    Returns:
        List of strings events
    """
    start_ql = bar_idx * 4.0
    events = []
    emotion_log: Dict[int, Dict[str, Any]] = {}
    reference_layers = reference_layers or {}

    # Parse chord (keep tensions: 7, 9, sus)
    symbol = chord.get("symbol", "C")
    parsed = parse_symbol(symbol)
    chord_tones = get_chord_tones(parsed, bass_octave=4)
    if not chord_tones:
        chord_tones = [60, 64, 67]

    # === Phase 2: Pin-First Tension Adoption ===
    section_cfg = policy.get("sections", {}).get(section_label, {})
    use_extensions = section_cfg.get("use_extensions", True)
    tension_mode = section_cfg.get("tension_mode", "accent")

    # Choose tensions first (pin-first workflow)
    pinned_tensions = []
    if use_extensions and tension_mode != "none":
        tensions_allowed = chord.get("tensions", [9, 11, 13])
        tensions_to_add = choose_tensions(symbol, tensions_allowed, tension_mode)

        root_str = parsed.get("root", "C") if isinstance(parsed, dict) else "C"
        for t_num in tensions_to_add:
            # Calculate tension pitch: root + interval mod 12
            root_pitch = pitch_of(root_str, octave=4)
            interval_map = {9: 2, 11: 5, 13: 9}
            tension_note = root_pitch + 12 + interval_map.get(t_num, 0)
            pinned_tensions.append(tension_note)

    # Combine: tensions first, then base chord tones
    all_tones = pinned_tensions + chord_tones
    pinned_mask = [True] * len(pinned_tensions) + [False] * len(chord_tones)

    # === Phase 1: Register enforcement ===
    role = select_role(section_cfg, "strings", default_role="unison_top")

    # Enforce register (preserve pinned tensions)
    all_tones = ensure_register(
        all_tones, "strings", policy, section_label, pinned_mask=pinned_mask
    )

    # Apply open voicing if preferred (for thickness)
    if policy.get("global", {}).get("prefer_open_voicings", True) and role == "unison_top":
        reg_max = (
            policy.get("instruments", {}).get("strings", {}).get("register", {}).get("max", 81)
        )
        all_tones = spread_open_voicing(all_tones, prefer_open=True, top_max=reg_max)

    # Humanization
    humanize_ms = strings_cfg.get("humanize_timing_ms", 12)
    humanize_vel = strings_cfg.get("humanize_velocity", 7)
    base_velocity = strings_cfg.get("base_velocity", 70)

    # Phase 2.0: Apply GuideTonePlan (notes_per_bar, phrase_shape, register, motion)
    target_note_count = 4  # Default
    velocity_boost = 0
    duration_scale = 1.0
    register_override = None
    motion_override = None
    phrase_shape = None

    if guide_params:
        # notes_per_bar
        if hasattr(guide_params, "notes_per_bar") and guide_params.notes_per_bar is not None:
            target_note_count = int(guide_params.notes_per_bar)
            target_note_count = max(1, min(target_note_count, 8))  # Clamp to 1-8

        # phrase_shape (uphill/downhill/arch)
        if hasattr(guide_params, "phrase_shape") and guide_params.phrase_shape:
            phrase_shape = guide_params.phrase_shape
            if phrase_shape == "uphill":
                velocity_boost = 5  # Gradual crescendo
            elif phrase_shape == "downhill":
                velocity_boost = -5  # Gradual decrescendo
                duration_scale = 1.2  # Longer release

        # register (high/mid/low)
        if hasattr(guide_params, "register") and guide_params.register:
            register_override = guide_params.register

        # motion (step/leap_ok/chromatic)
        if hasattr(guide_params, "motion") and guide_params.motion:
            motion_override = guide_params.motion

    if countermelody_type == "call_response":
        # Call-response (reference vocal F0 if available)
        if vocal_f0 is not None:
            # TODO: Extract F0 contour from vocal_f0 (simplified)
            # For now, use chord tones directly
            melody_notes = all_tones[:target_note_count]
        else:
            # Default: use chord tones
            melody_notes = all_tones[:target_note_count]

        # Generate notes (quarter notes)
        for i, note in enumerate(melody_notes):
            # Check if this note is a tension (Phase 2 metadata)
            is_tension = i < len(pinned_tensions)

            time_ql = start_ql + i + np.random.uniform(-humanize_ms / 1000, humanize_ms / 1000)
            note_clamped = int(
                np.clip(note, STRINGS_MIN_PITCH, STRINGS_MAX_PITCH)
            )  # Already registered
            vel = base_velocity + velocity_boost + np.random.randint(-humanize_vel, humanize_vel)
            duration = 1.0 * duration_scale
            events.append(
                {
                    "bar_idx": bar_idx,
                    "time_ql": float(time_ql),
                    "note": note_clamped,
                    "velocity": int(np.clip(vel, 50, 100)),
                    "duration_ql": float(duration),
                    "is_tension": is_tension,
                    "type": f"countermelody_{countermelody_type}",
                    "pattern": "countermelody",
                    "is_riff": True,
                    "event_type": "countermelody",
                }
            )

    elif countermelody_type == "ascending_line":
        # Ascending line (chorus/bridge uplifting)
        # Use chord tones in ascending order
        ascending_notes = sorted(all_tones[:target_note_count])
        for i, note in enumerate(ascending_notes):
            # Check if this note is a tension (Phase 2 metadata)
            is_tension = i < len(pinned_tensions)

            time_ql = start_ql + i + np.random.uniform(-humanize_ms / 1000, humanize_ms / 1000)
            note_clamped = int(np.clip(note, STRINGS_MIN_PITCH, STRINGS_MAX_PITCH))
            vel = (
                base_velocity + 5 + velocity_boost + np.random.randint(-humanize_vel, humanize_vel)
            )
            duration = 1.0 * duration_scale
            events.append(
                {
                    "bar_idx": bar_idx,
                    "time_ql": float(time_ql),
                    "note": note_clamped,
                    "velocity": int(np.clip(vel, 55, 100)),
                    "duration_ql": float(duration),
                    "is_tension": is_tension,
                    "type": f"countermelody_{countermelody_type}",
                    "pattern": "countermelody",
                    "is_riff": True,
                    "event_type": "countermelody",
                }
            )

    elif countermelody_type == "sustain_pad":
        # Sustain pad (verse/intro long sustain)
        # Use fewer notes for sustain (typically 2-3)
        sustain_note_count = min(3, target_note_count)
        for i, note in enumerate(all_tones[:sustain_note_count]):
            # Check if this note is a tension (Phase 2 metadata)
            is_tension = i < len(pinned_tensions)

            time_ql = start_ql + np.random.uniform(-humanize_ms / 1000, humanize_ms / 1000)
            note_clamped = int(np.clip(note, STRINGS_MIN_PITCH, STRINGS_MAX_PITCH))
            vel = (
                base_velocity - 10 + velocity_boost + np.random.randint(-humanize_vel, humanize_vel)
            )
            duration = 4.0 * duration_scale  # Whole note
            events.append(
                {
                    "bar_idx": bar_idx,
                    "time_ql": float(time_ql),
                    "note": note_clamped,
                    "velocity": int(np.clip(vel, 45, 90)),
                    "duration_ql": float(duration),
                    "is_tension": is_tension,
                    "type": f"countermelody_{countermelody_type}",
                    "pattern": "countermelody",
                    "is_riff": True,
                    "event_type": "countermelody",
                }
            )

    return events


def make_sustain_pattern(
    bar_idx: int,
    bar_data: pd.Series,
    chord: Dict[str, Any],
    strings_cfg: Dict[str, Any],
    policy: Dict[str, Any],
    section_density: float,
    section_label: str,
    emotion_params=None,  # Phase 2.0: EmotionParams dataclass
) -> List[Dict[str, Any]]:
    """
    Generate strings sustain pattern (Phase 2.0).

    Args:
        bar_idx: Bar index
        bar_data: Row from bars.parquet
        chord: Chord from chordmap
        strings_cfg: policy['instruments']['strings']
        policy: Full policy dictionary
        section_density: sections[section_label]['strings']
        section_label: verse, chorus, etc.
        emotion_params: EmotionParams (Phase 2.0, optional)

    Returns:
        List of strings events
    """
    start_ql = bar_idx * 4.0
    events = []

    # Check if strings are active
    strings_active = bar_data.get("strings_activity", 1.0)
    if pd.isna(strings_active):
        strings_active = 1.0

    if strings_active < 0.3:
        return []

    # Effective density
    effective_density = section_density * strings_active

    if effective_density < 0.2:
        return []

    # Parse chord
    symbol = chord.get("symbol", "C")
    parsed = parse_symbol(symbol)
    chord_tones = get_chord_tones(parsed, bass_octave=4)
    if not chord_tones:
        chord_tones = [60, 64, 67]

    # === Phase 2: Pin-First Tension Adoption ===
    section_cfg = policy.get("sections", {}).get(section_label, {})
    use_extensions = section_cfg.get("use_extensions", True)
    tension_mode = section_cfg.get("tension_mode", "accent")

    # Choose tensions first (pin-first workflow)
    pinned_tensions = []
    if use_extensions and tension_mode != "none":
        tensions_allowed = chord.get("tensions", [9, 11, 13])
        tensions_to_add = choose_tensions(symbol, tensions_allowed, tension_mode)

        root_str = parsed.get("root", "C") if isinstance(parsed, dict) else "C"
        for t_num in tensions_to_add:
            # Calculate tension pitch: root + interval mod 12
            root_pitch = pitch_of(root_str, octave=4)
            interval_map = {9: 2, 11: 5, 13: 9}
            tension_note = root_pitch + 12 + interval_map.get(t_num, 0)
            pinned_tensions.append(tension_note)

    # Combine: tensions first, then base chord tones
    all_tones = pinned_tensions + chord_tones
    pinned_mask = [True] * len(pinned_tensions) + [False] * len(chord_tones)

    # === Phase 1: Register enforcement ===
    role = select_role(section_cfg, "strings", default_role="unison_top")

    # Enforce register (preserve pinned tensions)
    all_tones = ensure_register(
        all_tones, "strings", policy, section_label, pinned_mask=pinned_mask
    )

    # Apply open voicing if preferred (for thickness)
    if policy.get("global", {}).get("prefer_open_voicings", True) and role == "unison_top":
        reg_max = (
            policy.get("instruments", {}).get("strings", {}).get("register", {}).get("max", 81)
        )
        all_tones = spread_open_voicing(all_tones, prefer_open=True, top_max=reg_max)

    # Humanization
    humanize_ms = strings_cfg.get("humanize_timing_ms", 12)
    humanize_vel = strings_cfg.get("humanize_velocity", 7)
    base_velocity = strings_cfg.get("base_velocity", 70)

    # Phase 2.0: Apply EmotionParams to velocity/duration
    velocity_adjustment = 0
    duration_scale = 1.0
    if emotion_params:
        # Use velocity_scale from EmotionParams (if available)
        velocity_scale = getattr(emotion_params, "velocity_scale", None)
        if velocity_scale is not None:
            # Convert scale to adjustment: 1.0 → 0, 1.1 → +7, 0.9 → -7
            velocity_adjustment = int((velocity_scale - 1.0) * 70)

        # Use duration_scale from EmotionParams (if available)
        duration_scale_attr = getattr(emotion_params, "duration_scale", None)
        if duration_scale_attr is not None:
            duration_scale = duration_scale_attr

    # Sustain chord (whole note)
    if np.random.random() < effective_density:
        for i, note in enumerate(all_tones[:3]):
            # Check if this note is a tension (Phase 2 metadata)
            is_tension = i < len(pinned_tensions)

            time_ql = start_ql + np.random.uniform(-humanize_ms / 1000, humanize_ms / 1000)
            note_clamped = int(np.clip(note, STRINGS_MIN_PITCH, STRINGS_MAX_PITCH))
            vel = (
                base_velocity + velocity_adjustment + np.random.randint(-humanize_vel, humanize_vel)
            )
            duration = 4.0 * duration_scale
            events.append(
                {
                    "bar_idx": bar_idx,
                    "time_ql": float(time_ql),
                    "note": note_clamped,
                    "velocity": int(np.clip(vel, 45, 95)),
                    "duration_ql": float(duration),
                    "is_tension": is_tension,
                    "type": "sustain",
                    "pattern": "pad",
                    "event_type": "pad",
                }
            )

    return events


def enforce_max_notes_per_bar(
    bar_events: List[Dict[str, Any]], max_notes: Optional[int]
) -> List[Dict[str, Any]]:
    """Trim low-priority notes if a bar exceeds the policy max."""
    if not bar_events or not max_notes or max_notes <= 0:
        return bar_events

    if len(bar_events) <= max_notes:
        return bar_events

    priority: List[Dict[str, Any]] = []
    support: List[Dict[str, Any]] = []
    floor: List[Dict[str, Any]] = []

    for ev in sorted(bar_events, key=lambda e: e.get("time_ql", 0.0)):
        ev_type = str(ev.get("type", "")).lower()

        if ev.get("is_riff") or "countermelody" in ev_type or "guide" in ev_type:
            priority.append(ev)
        elif "floor" in ev_type:
            floor.append(ev)
        else:
            support.append(ev)

    kept: List[Dict[str, Any]] = []
    for bucket in (priority, support, floor):
        if len(kept) >= max_notes:
            break
        space = max_notes - len(kept)
        kept.extend(bucket[:space])

    return sorted(kept, key=lambda e: e.get("time_ql", 0.0))


def generate_strings_plan(
    bars: pd.DataFrame,
    sections: List[Dict[str, Any]],
    chordmap: List[Dict[str, Any]],
    policy: Dict[str, Any],
    vocal_f0: Optional[pd.DataFrame] = None,
    melody_hints: Optional[Dict[int, "MelodyHint"]] = None,
    rng_seed: int = 42,
    lyric_index=None,  # Phase 2.0: LyricAnchorIndex
    emotion_ai=None,  # Phase 2.0: EmotionAI v2
    guidetone_ai=None,  # Phase 2.0: GuideToneAI v2
    reference_layers: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Main logic: Generate slot-based strings plan (Phase 2.0).

    Args:
        bars: bars_with_slots.parquet
        sections: sections.json
        chordmap: chordmap_locked_extended.json
        policy: policy YAML
        vocal_f0: vocal_f0_crepe.parquet (optional)
        rng_seed: Random seed
        lyric_index: LyricAnchorIndex (Phase 2.0, optional)
        emotion_ai: EmotionAI v2 instance (Phase 2.0, optional)
        guidetone_ai: GuideToneAI v2 instance (Phase 2.0, optional)

    Returns:
        Strings plan JSON
    """
    strings_cfg = policy.get("instruments", {}).get("strings", {})
    sections_density = policy.get("sections", {})
    countermelody_on_sections = strings_cfg.get("countermelody_on_sections", ["chorus", "bridge"])
    min_notes = strings_cfg.get("min_notes_per_bar", 1)
    max_notes = strings_cfg.get("max_notes_per_bar")

    # Countermelody style distribution
    countermelody_styles_cfg = strings_cfg.get("countermelody_styles", [])
    if isinstance(countermelody_styles_cfg, list) and len(countermelody_styles_cfg) > 0:
        countermelody_styles = {s["type"]: s["probability"] for s in countermelody_styles_cfg}
    else:
        countermelody_styles = {"call_response": 0.4, "ascending_line": 0.3, "sustain_pad": 0.3}

    events = []
    emotion_log: Dict[int, Dict[str, Any]] = {}

    for _, bar_row in bars.iterrows():
        bar_idx = int(bar_row["bar_idx"])
        section_label = bar_row.get("section_label", "verse")
        riff_slot = bar_row.get("riff_slot", False)
        bar_start_ql = float(bar_row.get("start_ql", bar_idx * 4.0))
        bar_end_ql = float(bar_row.get("end_ql", (bar_idx + 1) * 4.0))

        # Get chord
        chord = get_chord_at_bar(chordmap, bar_idx)
        parsed = parse_symbol(chord.get("symbol", "C"))
        chordpad_pitches = get_chord_tones(parsed, bass_octave=4) or [60, 64, 67]

        # Get section density
        section_cfg = sections_density.get(section_label, {})
        strings_density = section_cfg.get("strings", 0.5)

        # Phase 2.0: Lyric info (phrase_role detection)
        lyric_info = None
        if lyric_index:
            lyric_info = lyric_index.get_bar_info(bar_idx)

        # Phase 2.0: Build unified context
        context = {
            "bar_index": bar_idx,  # GuideToneAI v2 expects "bar_index"
            "bar": bar_idx,  # EmotionAI v2 expects "bar"
            "section": section_label,
            "role": "strings",
            "chord_symbol": chord.get("symbol", "C"),
            "slots": {"riff": riff_slot},
        }

        if reference_layers:
            context["reference_layers"] = reference_layers

        # Add lyric info to context
        if lyric_info and lyric_info.get("has_anchor"):
            context["lyric"] = {
                "phrase_role": lyric_info["phrase_role"],
                "stress_level": lyric_info.get("stress_level", 0.0),
                "is_silent": lyric_info.get("is_silent", False),
            }

        # Phase 2.0: Get EmotionParams from EmotionAI v2
        emotion_params = None
        if emotion_ai:
            try:
                emotion_params = emotion_ai.get_params(context)
                record_emotion_snapshot(
                    emotion_log,
                    bar_idx=bar_idx,
                    section_label=section_label,
                    emotion_params=emotion_params,
                )
            except Exception as e:
                print(f"⚠️  EmotionAI v2 error at bar {bar_idx}: {e}")

        # Phase 2.0: Get GuideTonePlan from GuideToneAI v2
        guide_params = None
        if guidetone_ai:
            try:
                guide_params = guidetone_ai.get_plan(context)
            except Exception as e:
                print(f"⚠️  GuideToneAI v2 error at bar {bar_idx}: {e}")

        # Phase 2.0: Apply EmotionParams to density
        if emotion_params:
            # Use density_scale from EmotionParams (if available)
            density_scale = getattr(emotion_params, "density_scale", None)
            if density_scale is not None:
                strings_density *= density_scale

        # Decision: Countermelody or Sustain
        if riff_slot and section_label in countermelody_on_sections:
            # Fire countermelody
            countermelody_type = np.random.choice(
                list(countermelody_styles.keys()), p=list(countermelody_styles.values())
            )
            bar_events = make_countermelody(
                bar_idx,
                bar_row,
                chord,
                strings_cfg,
                policy,
                section_label,
                countermelody_type,
                vocal_f0,
                guide_params,  # Phase 2.0: GuideTonePlan
            )
        else:
            # Sustain pattern
            bar_events = make_sustain_pattern(
                bar_idx,
                bar_row,
                chord,
                strings_cfg,
                policy,
                strings_density,
                section_label,
                emotion_params,  # Phase 2.0: EmotionParams
            )

        # Activity floor
        # Phase 2: Generate tension candidates for floor padding
        tension_ratio = resolve_tension_ratio(policy, "strings", section_label, 0.25)
        tension_pitches = []

        if tension_ratio > 0.0:
            allow_tensions = strings_cfg.get("tensions", {}).get("allow", [9, 11, 13])
            tension_mode = strings_cfg.get("tensions", {}).get("mode", "accent")
            tension_pcs = choose_tensions(chord.get("symbol", "C"), allow_tensions, tension_mode)

            # Convert tension PCs to MIDI pitches in strings register
            reg_pref = strings_cfg.get("register", {}).get("octave_prefer", 69)  # A4
            for t in tension_pcs:
                tension_note = t + ((reg_pref // 12) * 12)
                tension_pitches.append(tension_note)

        # Ensure tensions are in register
        if tension_pitches:
            tension_pitches = ensure_register(tension_pitches, "strings", policy, section_label)

        bar_events = ensure_activity_floor(
            bar_events,
            bar_start_ql,
            bar_end_ql,
            min_notes,
            chordpad_pitches,
            velocity=65,
            tension_pitches=tension_pitches,
        )

        bar_events = enforce_max_notes_per_bar(bar_events, max_notes)

        for ev in bar_events:
            if "bar_idx" not in ev:
                ev["bar_idx"] = bar_idx
            if "section_label" not in ev:
                ev["section_label"] = section_label

        events.extend(bar_events)

    # Sort by time
    events = sorted(events, key=lambda e: e["time_ql"])

    # Phase 3: Apply humanization (section-aware)
    from itertools import groupby

    humanized_events = []
    tempo_bpm = policy.get("global", {}).get("tempo_bpm", 120.0)

    for section_label, section_events in groupby(
        events, key=lambda e: e.get("section_label", "verse")
    ):
        section_events = list(section_events)
        humanize_cfg = load_humanize_config(
            policy, section_label, "strings", song_id=policy.get("metadata", {}).get("song_id")
        )

        section_events = apply_timing_humanize(section_events, humanize_cfg, tempo_bpm)
        section_events = apply_velocity_humanize(section_events, humanize_cfg, base_velocity=75)
        section_events = apply_duration_humanize(
            section_events, humanize_cfg, bar_duration_ql=4.0
        )  # Phase 3.5

        humanized_events.extend(section_events)

    events = sorted(humanized_events, key=lambda e: e["time_ql"])

    # Phase 3.6: Long Note Extension (Strings → Pad土台化)
    inst_cfg = policy.get("instruments", {}).get("strings", {})
    sustain_cfg = inst_cfg.get("sustain", {})
    min_dur = sustain_cfg.get("min_duration_ql", 3.0)  # 3拍以上を「長音」とみなす
    bar_span = sustain_cfg.get("max_bar_span", 3)  # 最大3小節先まで伸ばす

    events = extend_last_note_per_bar(
        events=events,
        bars_df=bars,
        role="strings",
        min_duration_ql=min_dur,
        max_bar_span=bar_span,
    )

    # Phase 3.7: Guide Tone Melodies (NEW!)
    # Build section mapping for guide tone generator
    sections_map = {
        int(bar_row["bar_idx"]): bar_row.get("section_label", "verse")
        for _, bar_row in bars.iterrows()
    }

    gt_cfg = load_guide_tone_config(
        policy, "strings", default_low=STRINGS_MIN_PITCH, default_high=STRINGS_MAX_PITCH
    )

    guide_events = generate_guide_tone_events(
        bars_df=bars,
        chordmap_events=chordmap,
        sections=sections_map,
        cfg=gt_cfg,
        unit="bar",
    )

    # Merge guide tones with existing events
    if guide_events:
        print(f"   🎵 Generated {len(guide_events)} guide tone notes")
        events.extend(guide_events)
        events = sorted(events, key=lambda e: e["time_ql"])

    filter_stats = {"annotated": 0, "removed": 0}
    if melody_hints:
        events, filter_stats = apply_melody_hint_filter(
            events,
            melody_hints,
            instrument="strings",
            drop_tags=("melody_hint_long",),
            drop_threshold_beats=2.0,
            annotate=True,
        )

    # Clamp register to policy bounds (defensive)
    reg_cfg = policy.get("instruments", {}).get("strings", {}).get("register", {})
    reg_min = int(reg_cfg.get("min", STRINGS_MIN_PITCH))
    reg_max = int(reg_cfg.get("max", STRINGS_MAX_PITCH))
    for ev in events:
        if "note" in ev:
            ev["note"] = int(np.clip(ev["note"], reg_min, reg_max))

    # Final metadata update
    metadata = {
        "instrument": "strings",
        "num_bars": int(len(bars)),
        "num_events": len(events),
        "riff_slots_used": int(bars["riff_slot"].sum()),
        "generator": "generate_strings_plan_v2.py",
        "humanize_profile": policy.get("humanize", {}).get("profile", "pop_easy"),
        "guide_tones_enabled": gt_cfg.enabled,
        "guide_tone_count": len(guide_events) if guide_events else 0,
        "sustain_mode": "last_note_per_bar",  # Phase 3.6
        "sustain_config": {"min_duration_ql": min_dur, "max_bar_span": bar_span},
        "melody_hint": {
            "annotated": filter_stats.get("annotated", 0),
            "removed_for_strings": filter_stats.get("removed", 0),
            "bars_with_hints": len(melody_hints or {}),
        },
        "ai_hooks": {
            "lyric_anchor": bool(lyric_index),
            "emotion_ai": bool(emotion_ai),
            "guide_tone_ai": bool(guidetone_ai),
            "reference_layers": bool(reference_layers),
        },
    }

    if reference_layers:
        metadata["reference_layers"] = reference_layers

    emotion_tracking = summarize_emotion_log(emotion_log)
    if emotion_tracking:
        metadata["emotion_tracking"] = emotion_tracking

    return {"metadata": metadata, "events": events}


def main():
    parser = argparse.ArgumentParser(description="Generate strings plan (slot-based V2)")
    parser.add_argument("--bars", required=True, help="Path to bars_with_slots.parquet")
    parser.add_argument("--sections", required=True, help="Path to sections.json")
    parser.add_argument("--chordmap", required=True, help="Path to chordmap_locked_extended.json")
    parser.add_argument("--policy", required=True, help="Path to policy YAML")
    parser.add_argument("--vocal-f0", help="Path to vocal_f0_crepe.parquet (optional)")
    parser.add_argument("--piano-oaf", help="Path to piano_onsets_and_frames.json (optional)")
    parser.add_argument(
        "--emit-melody-manifest",
        action="store_true",
        help="Write melody_hint_manifest.json (requires --vocal-f0)",
    )
    parser.add_argument(
        "--melody-manifest-path",
        type=Path,
        help="Override melody hint manifest path (default: alongside --out)",
    )
    parser.add_argument("--lyric-anchors", help="Path to lyric_anchors.json (Phase 2.0, optional)")
    parser.add_argument(
        "--emotion-profile", help="Path to emotion_profile.json (Phase 2.0, optional)"
    )
    parser.add_argument("--guide-hints", help="Path to guide_tone_hints.json (Phase 2.0, optional)")
    parser.add_argument("--rulebook", help="Path to rulebook.yaml (Phase 2.0, optional)")
    parser.add_argument("--out", required=True, help="Output strings_plan.json")
    parser.add_argument("--seed", type=int, help="Random seed for reproducibility")
    parser.add_argument(
        "--rhythm-manifest",
        help="Override rhythm_vocab manifest path (default: data/rhythm_vocab.yaml)",
    )
    args = parser.parse_args()

    # Set random seed for reproducibility
    if args.seed is not None:
        np.random.seed(args.seed)
        print(f"🎲 Random seed set to {args.seed}")

    print(f"📖 Loading bars from {args.bars}")
    bars = load_bars(args.bars)

    print(f"📖 Loading sections from {args.sections}")
    sections = load_sections(args.sections)

    print(f"📖 Loading chordmap from {args.chordmap}")
    chordmap = load_chordmap(args.chordmap)

    print(f"📖 Loading policy from {args.policy}")
    with open(args.policy, "r", encoding="utf-8") as f:
        policy = yaml.safe_load(f)

    vocal_f0 = None
    if args.vocal_f0:
        print(f"📖 Loading vocal F0 from {args.vocal_f0}")
        vocal_f0 = load_vocal_f0(args.vocal_f0)

    melody_hints = build_melody_hint_table(bars, vocal_f0) if vocal_f0 is not None else {}
    if melody_hints:
        print("📊 Melody hint summary (CREPE):")
        for section, stats in summarize_melody_hints(melody_hints).items():
            print(
                f"   - {section}: bars={stats['bars']} long={stats['long']} phrase={stats['phrase']} gliss={stats['gliss']} avg_len={stats['avg_duration_beats']}"
            )
    elif args.emit_melody_manifest:
        print("Melody hint manifest requested but no vocal F0 provided; skipping manifest export.")

    reference_layers = load_reference_layers(args.vocal_f0, args.piano_oaf)
    if reference_layers:
        print(
            "📡 Reference layers loaded: "
            + ", ".join(
                f"{name}={data.get('frames', data.get('notes', 0))}"
                for name, data in reference_layers.items()
            )
        )

    manifest_out = args.melody_manifest_path
    if manifest_out and not manifest_out.is_absolute():
        manifest_out = Path(args.out).parent / manifest_out
    if manifest_out is None:
        manifest_out = Path(args.out).with_name("melody_hint_manifest.json")

    if args.emit_melody_manifest and melody_hints:
        manifest_payload = build_melody_hint_manifest_payload(
            melody_hints,
            bars_total=len(bars),
            song_id=policy.get("metadata", {}).get("song_id"),
            bars_path=Path(args.bars),
            vocal_f0_path=Path(args.vocal_f0) if args.vocal_f0 else None,
            out_path=manifest_out,
        )
        manifest_out.parent.mkdir(parents=True, exist_ok=True)
        manifest_out.write_text(
            json.dumps(manifest_payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        rel_path = (
            manifest_out.relative_to(Path(args.out).parent)
            if manifest_out.is_relative_to(Path(args.out).parent)
            else manifest_out
        )
        print(f"melody_hint_manifest: {rel_path} (hints={len(melody_hints)})")

    # Initialize OtobonAI (Phase 2.0)
    lyric_index = None
    emotion_ai = None
    guidetone_ai = None
    rulebook = None

    if OTOBON_AI_AVAILABLE:
        # Rulebook（共通）
        if args.rulebook and Path(args.rulebook).exists():
            print(f"📖 Loading Rulebook from {args.rulebook}")
            rulebook = Rulebook.load(Path(args.rulebook))

        # LyricAnchorIndex
        if args.lyric_anchors and Path(args.lyric_anchors).exists():
            print(f"📝 Loading LyricAnchorIndex from {args.lyric_anchors}")
            # tempo_bpm取得（policy.global.tempo_bpm、デフォルト120）
            tempo_bpm = policy.get("global", {}).get("tempo_bpm", 120.0)
            # Load lyric_anchors.json
            with open(args.lyric_anchors, "r", encoding="utf-8") as f:
                anchors_data = json.load(f)
            lyric_index = LyricAnchorIndex(anchors=anchors_data, tempo_bpm=tempo_bpm)

        # EmotionAI v2
        if args.emotion_profile and Path(args.emotion_profile).exists():
            print(f"🎭 Loading EmotionAI v2 from {args.emotion_profile}")
            # Load emotion_profile.json
            with open(args.emotion_profile, "r", encoding="utf-8") as f:
                emotion_profile_data = json.load(f)
            emotion_ai = EmotionAIv2(emotion_profile=emotion_profile_data, rulebook=rulebook)

        # GuideToneAI v2
        if args.guide_hints and Path(args.guide_hints).exists():
            print(f"🎵 Loading GuideToneAI v2 from {args.guide_hints}")
            # Load guide_tone_hints.json
            with open(args.guide_hints, "r", encoding="utf-8") as f:
                guide_hints_data = json.load(f)
            guidetone_ai = GuideToneAIv2(guide_tone_hints=guide_hints_data, rulebook=rulebook)

    print(f"🎻 Generating strings plan ({len(bars)} bars)")
    plan = generate_strings_plan(
        bars,
        sections,
        chordmap,
        policy,
        vocal_f0,
        melody_hints,
        rng_seed=args.seed if args.seed is not None else 42,
        lyric_index=lyric_index,
        emotion_ai=emotion_ai,
        guidetone_ai=guidetone_ai,
        reference_layers=reference_layers,
    )

    plan_meta = plan.setdefault("metadata", {})
    default_section = plan_meta.get("default_section") or policy.get("metadata", {}).get(
        "default_section"
    )
    song_id = plan_meta.get("song_id") or policy.get("metadata", {}).get("song_id")
    rhythm_stats = apply_rhythm_vocab_annotations(
        plan.get("events", []),
        instrument="strings",
        policy=policy,
        default_section=default_section,
        song_id=song_id,
    )
    if rhythm_stats.get("assigned", 0) > 0:
        plan_meta["rhythm_vocab_ids"] = rhythm_stats["used_ids"]
        plan_meta["rhythm_vocab_instrument"] = "strings"
        plan_meta.setdefault("ai_hooks", {}).update({"rhythm_vocab_policy": True})

    rhythm_manifest_path = Path(args.rhythm_manifest).expanduser() if args.rhythm_manifest else None
    if DurationHumanizeAI is not None:
        try:
            duration_ai = DurationHumanizeAI(
                instrument="strings",
                policy=policy,
                tempo_bpm=policy.get("global", {}).get("tempo_bpm", 120),
                rhythm_manifest_path=rhythm_manifest_path,
                vocab_instrument="strings",
            )
            duration_ai.annotate_plan(plan)
            plan_meta.setdefault("ai_hooks", {}).update({"duration_humanize_ai": True})
        except Exception as exc:
            print(f"⚠️  DurationHumanizeAI annotation skipped: {exc}")

    # Save
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(plan, f, indent=2, ensure_ascii=False)

    print(f"✅ Strings plan saved to {out_path}")
    print(f"   Events: {plan['metadata']['num_events']}")
    print(f"   Riff slots used: {plan['metadata']['riff_slots_used']}/{len(bars)}")


if __name__ == "__main__":
    main()
