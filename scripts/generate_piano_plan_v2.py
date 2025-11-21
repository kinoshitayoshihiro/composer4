#!/usr/bin/env python3
"""
generate_piano_plan_v2.py - Slot-based piano renderer for fill/riff system.

Architecture:
- Slot Planner: bars_with_slots.parquet (fill_slot: where to fire)
- Policy YAML: density/comping_styles/voicing (how to fire)
- Chord Source: manual_chordmap.json (what notes to play)
- Output: plans/piano_plan.json

Design Philosophy:
"位置決めはbars/sections。造形は楽器別レンダラ。music21は和声支援のみ。"
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Mapping

import numpy as np
import pandas as pd
import yaml

# Import shared chordmap utilities
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))  # Add project root
from chordmap_utils import (
    load_chordmap,
    get_chord_at_bar,
    parse_symbol,
    parse_chord_symbol,
    get_chord_tones,
)
from ai_hook_utils import load_reference_layers
from melody_hint_utils import (
    MelodyHint,
    apply_melody_hint_filter,
    build_melody_hint_manifest_payload,
    build_melody_hint_table,
    summarize_melody_hints,
)
from v2_common import (
    ensure_activity_floor,
    choose_tension_enabled,
    apply_humanize,
    ensure_register,
    spread_open_voicing,
    select_role,
    decide_tension_use,
    choose_tensions,  # Phase 2: Tension selection
    fold_to_register,  # Phase 2: Octave wrapping
    voice_lead_voicing,  # Phase 2: Voice-leading
    resolve_tension_ratio,  # Phase 2: Multi-tier tension_ratio resolution
    load_humanize_config,  # Phase 3: Humanize config
    apply_timing_humanize,  # Phase 3: Timing humanize
    apply_velocity_humanize,  # Phase 3: Velocity humanize
    apply_duration_humanize,  # Phase 3.5: Duration humanize
    extend_last_note_per_bar,  # Phase 3.6: Long note extension
    apply_rhythm_vocab_annotations,  # Rhythm vocab annotations
    record_emotion_snapshot,
    summarize_emotion_log,
)

# Phase 2.0: AI Integration
try:
    from otobonAI.lyric_index import LyricAnchorIndex
    from otobonAI.emotion_ai_v2 import EmotionAI as EmotionAIv2
    from otobonAI.guide_tone_ai_v2 import GuideToneAI as GuideToneAIv2
    from otobonAI.rulebook_engine import Rulebook
except ImportError as e:
    print(f"⚠️  Phase 2.0 AI modules not available: {e}")
    print("   Continuing with slot-based generation only.")
    LyricAnchorIndex = None
    EmotionAIv2 = None
    GuideToneAIv2 = None
    Rulebook = None

try:
    from otobonAI.duration_humanize_ai import DurationHumanizeAI
except Exception as exc:  # pragma: no cover - optional dependency
    print(f"⚠️  DurationHumanizeAI unavailable: {exc}")


def _prepare_emotion_profile(payload: Any) -> Dict[int, Dict[str, Any]]:
    """Normalize EmotionAI profile JSON into {bar_idx: metrics} mapping."""

    def _extract_entry(entry: Mapping[str, Any]) -> tuple[int, Dict[str, Any]] | None:
        if "bar" not in entry:
            return None
        try:
            bar_idx = int(entry["bar"])
        except (TypeError, ValueError):
            return None
        cleaned: Dict[str, Any] = {
            key: entry[key]
            for key in (
                "energy",
                "tension",
                "brightness",
                "valence",
                "velocity_scale",
                "duration_scale",
                "density_scale",
                "phrase_role",
                "tags",
            )
            if key in entry
        }
        return bar_idx, cleaned

    result: Dict[int, Dict[str, Any]] = {}
    if payload is None:
        return result

    # Case 1: explicit {"events": [...]} wrapper
    if isinstance(payload, Mapping) and isinstance(payload.get("events"), list):
        entries = payload["events"]
    # Case 2: already {bar_idx: {...}}
    elif isinstance(payload, Mapping):
        numeric_key_map = True
        for key, value in payload.items():
            try:
                bar_idx = int(key)
            except (TypeError, ValueError):
                numeric_key_map = False
                break
            if not isinstance(value, Mapping):
                numeric_key_map = False
                break
            result[bar_idx] = dict(value)
        if numeric_key_map:
            return result
        entries = [payload]
    # Case 3: simple list of entries
    elif isinstance(payload, list):
        entries = payload
    else:
        return result

    for entry in entries:
        if not isinstance(entry, Mapping):
            continue
        extracted = _extract_entry(entry)
        if extracted is None:
            continue
        bar_idx, cleaned = extracted
        result[bar_idx] = cleaned
    return result
    DurationHumanizeAI = None  # type: ignore


# Phase 1: Register enforcement handled by ensure_register() from v2_common
# No longer using hardcoded PIANO_MIN_PITCH/PIANO_MAX_PITCH


def load_bars(bars_path: str) -> pd.DataFrame:
    """Load bars.parquet with fill_slot."""
    bars = pd.read_parquet(bars_path)
    required = ["section_label"]
    missing = [c for c in required if c not in bars.columns]
    if missing:
        raise ValueError(f"bars.parquet missing columns: {missing}")

    if "fill_slot" not in bars.columns:
        bars["fill_slot"] = 0
        print("ℹ️  bars file missing fill_slot column, defaulting to inactive slots")

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


def get_chord_at_bar(chordmap: List[Dict[str, Any]], bar_idx: int) -> Dict[str, Any]:
    """Find chord event overlapping with bar_idx."""
    bar_start_ql = bar_idx * 4.0
    for chord in chordmap:
        chord_start = chord.get("time_ql", 0.0)
        chord_end = chord_start + chord.get("duration_ql", 4.0)
        if chord_start <= bar_start_ql < chord_end:
            return chord
    return chordmap[0] if chordmap else {}


def load_vocal_f0(f0_path: Optional[str]) -> Optional[pd.DataFrame]:
    """Load vocal_f0_crepe.parquet if supplied."""

    if not f0_path:
        return None
    path = Path(f0_path)
    if not path.exists():
        print(f"⚠️  vocal_f0 file not found: {f0_path}")
        return None
    try:
        return pd.read_parquet(path)
    except Exception as exc:
        print(f"⚠️  Failed to load vocal F0 ({f0_path}): {exc}")
        return None


def is_section_boundary(bar_idx: int, sections: List[Dict[str, Any]]) -> bool:
    """Check if bar is section boundary (end-1 bar)."""
    bar_start_ql = bar_idx * 4.0
    for section in sections:
        sec_end_ql = section.get("end_ql", 0.0)
        # Check if bar is last bar before section end (within 1 bar)
        if abs(bar_start_ql - (sec_end_ql - 4.0)) < 0.1:
            return True
    return False


def make_fill_decoration(
    bar_idx: int,
    bar_data: pd.Series,
    chord: Dict[str, Any],
    piano_cfg: Dict[str, Any],
    policy: Dict[str, Any],
    section_label: str,
    prev_voicing: List[int] = None,
    emotion_params=None,
    guide_params=None,
    reference_layers: Optional[Dict[str, Any]] = None,
) -> tuple:
    """
    Generate piano fill (2-4 note decoration at section boundary).

    Args:
        bar_idx: Bar index
        bar_data: Row from bars.parquet
        chord: Chord from chordmap
        piano_cfg: policy['instruments']['piano']
        policy: Full policy dictionary
        section_label: verse, chorus, etc.
        prev_voicing: Previous voicing for voice-leading (Phase 2)
        emotion_params: EmotionParams from EmotionAI v2 (Phase 2.0)
        guide_params: GuideTonePlan from GuideToneAI v2 (Phase 2.0)

    Returns:
        tuple: (events, final_voicing)
    """
    # === Phase 2: Chord tone selection with Pin-First tensions ===
    if isinstance(chord, dict):
        parsed = parse_symbol(chord.get("symbol", "C"))
    else:
        # Fallback: if chord is object with root/quality/tensions
        root = chord.get("root", "C")
        quality = chord.get("quality", "")
        tensions = chord.get("tensions", [])
        parsed = parse_chord_symbol(root, quality, tensions)

    # Phase 2: Pin-First Workflow
    base_tones = get_chord_tones(parsed, bass_octave=5)  # Piano octave 5 (C5=72)
    if not base_tones:
        base_tones = [60, 64, 67]  # C major fallback

    # 1) Check if tensions enabled for this section
    section_cfg = policy.get("sections", {}).get(section_label, {})
    tension_ratio = resolve_tension_ratio(policy, "piano", section_label, 0.35)
    use_extensions = tension_ratio >= 0.2 or (
        tension_ratio > 0.0 and np.random.random() < tension_ratio
    )

    start_ql = bar_idx * 4.0
    events = []
    emotion_log: Dict[int, Dict[str, Any]] = {}
    reference_layers = reference_layers or {}

    # Parse chord - handle both symbol and root+quality formats
    if "symbol" in chord and chord["symbol"]:
        parsed = parse_symbol(chord["symbol"])
    else:
        root = chord.get("root", "C")
        quality = chord.get("quality", "")
        tensions = chord.get("tensions", [])
        parsed = parse_chord_symbol(root, quality, tensions)

    # Phase 2: Pin-First Workflow
    base_tones = get_chord_tones(parsed, bass_octave=5)  # Piano octave 5 (C5=72)
    if not base_tones:
        base_tones = [60, 64, 67]  # C major fallback

    # 1) Check if tensions enabled for this section
    section_cfg = policy.get("sections", {}).get(section_label, {})
    tension_ratio = resolve_tension_ratio(policy, "piano", section_label, 0.35)
    use_extensions = tension_ratio >= 0.2 or (
        tension_ratio > 0.0 and np.random.random() < tension_ratio
    )

    # 2) Choose tensions if enabled
    pinned_tensions = []
    if use_extensions:
        allow_tensions = piano_cfg.get("tensions", {}).get("allow", [9, 11, 13])
        tension_mode = piano_cfg.get("tensions", {}).get("mode", "accent")
        tensions = choose_tensions(chord.get("symbol", "C"), allow_tensions, tension_mode)

        # Convert tension PC to MIDI (prefer upper register for piano)
        reg_pref = 72  # C5 - piano upper-mid
        for t in tensions:
            tension_pc = t
            tension_note = tension_pc + ((reg_pref // 12) * 12)
            pinned_tensions.append(tension_note)

    # 3) Priority merge (tensions first)
    all_tones = []
    seen = set()
    for t in pinned_tensions + base_tones:
        pc = t % 12
        if pc not in seen:
            all_tones.append(t)
            seen.add(pc)

    pinned_mask = [True] * len(pinned_tensions) + [False] * (len(all_tones) - len(pinned_tensions))

    # 4) Determine target note count (ensure room for tensions)
    target_n = 4  # Fill decoration: max 4 notes
    target_n = max(target_n, len(pinned_tensions))

    # 5) Select notes (tensions have priority)
    chord_tones = all_tones[:target_n]
    pinned_mask = pinned_mask[:target_n]

    # === Phase 2: Register enforcement with pinned protection ===
    role = select_role(section_cfg, "piano", default_role="pad")
    chord_tones = ensure_register(
        chord_tones, "piano", policy, section_label, pinned_mask=pinned_mask
    )

    # Apply open voicing if preferred (for thickness)
    if policy.get("global", {}).get("prefer_open_voicings", True) and role == "pad":
        reg_max = policy.get("instruments", {}).get("piano", {}).get("register", {}).get("max", 84)
        reg_min = policy.get("instruments", {}).get("piano", {}).get("register", {}).get("min", 43)
        chord_tones = spread_open_voicing(
            chord_tones, prefer_open=True, top_max=reg_max, low_min=reg_min
        )

    # Voice-leading (Phase 2) - NEW API
    if prev_voicing:
        reg_min = policy.get("instruments", {}).get("piano", {}).get("register", {}).get("min", 36)
        reg_max = policy.get("instruments", {}).get("piano", {}).get("register", {}).get("max", 84)
        chord_tones = voice_lead_voicing(
            chord_tones=chord_tones,
            prev_voicing=prev_voicing,
            reg_min=reg_min,
            reg_max=reg_max,
            max_step=7,  # Allow up to perfect 5th motion
        )

    # Use all chord_tones (no slicing - already limited to target_n)
    voicing = chord_tones

    # Humanization
    humanize_ms = piano_cfg.get("humanize_timing_ms", 12)
    humanize_vel = piano_cfg.get("humanize_velocity", 7)
    base_velocity = piano_cfg.get("base_velocity", 75)

    # Phase 2.0: Apply EmotionAI velocity/duration scaling
    velocity_scale = 1.0
    duration_scale = 1.0
    if emotion_params:
        velocity_scale = emotion_params.velocity_scale
        duration_scale = emotion_params.duration_scale
        # Adjust base_velocity with energy
        base_velocity = int(base_velocity * velocity_scale)

    # Phase 2.0: Adjust note count with GuideToneAI (optional)
    if guide_params and hasattr(guide_params, "notes_per_bar"):
        target_n = min(target_n, int(guide_params.notes_per_bar))

    # Fill decoration (last 2 beats of bar) - all voicing notes
    for i, note in enumerate(voicing):
        time_ql = (
            start_ql + 2.0 + i * 0.25 + np.random.uniform(-humanize_ms / 1000, humanize_ms / 1000)
        )
        # Octave up (bright) with register enforcement
        note_shifted = ensure_register([note + 12], "piano", policy, section_label)
        note_octave_up = note_shifted[0] if note_shifted else note

        vel = base_velocity + 10 + np.random.randint(-humanize_vel, humanize_vel)

        # Phase 2.0: Apply duration_scale
        note_duration = 0.25 * duration_scale

        event = {
            "bar_idx": bar_idx,
            "time_ql": float(time_ql),
            "note": int(note_octave_up),
            "velocity": int(np.clip(vel, 60, 100)),
            "duration_ql": float(note_duration),
            "is_fill": True,
            "type": "fill_decoration",
            "event_type": "fill",
        }
        # Phase 2: Mark tensions for QAgate
        if i < len(pinned_mask) and pinned_mask[i]:
            event["is_tension"] = True
        events.append(event)

    # Phase 2: Return tuple (events, final_voicing)
    final_voicing = sorted(chord_tones)
    return (events, final_voicing)


def make_comping_pattern(
    bar_idx: int,
    bar_data: pd.Series,
    chord: Dict[str, Any],
    piano_cfg: Dict[str, Any],
    policy: Dict[str, Any],
    section_density: float,
    section_label: str,
    prev_voicing: List[int] = None,
    emotion_params=None,
    guide_params=None,
) -> tuple:
    """
    Generate piano comping (offbeat/arpeggio/block chord) with role-based voicing.

    Args:
        bar_idx: Bar index
        bar_data: Row from bars.parquet
        chord: Chord from chordmap
        piano_cfg: policy['instruments']['piano']
        policy: Full policy dictionary
        section_density: sections[section_label]['piano']
        section_label: verse, chorus, etc.
        prev_voicing: Previous voicing for voice-leading (Phase 2)
        emotion_params: EmotionParams from EmotionAI v2 (Phase 2.0)
        guide_params: GuideTonePlan from GuideToneAI v2 (Phase 2.0)

    Returns:
        (events, final_voicing) tuple
    """
    start_ql = bar_idx * 4.0
    events = []

    # Check if piano is active
    piano_active = bar_data.get("piano_activity", 1.0)
    if pd.isna(piano_active):
        piano_active = 1.0

    if piano_active < 0.3:
        return ([], None)

    # Effective density
    effective_density = section_density * piano_active

    if effective_density < 0.2:
        return ([], None)

    # Parse chord - handle both symbol and root+quality formats
    if "symbol" in chord and chord["symbol"]:
        parsed = parse_symbol(chord["symbol"])
    else:
        root = chord.get("root", "C")
        quality = chord.get("quality", "")
        tensions = chord.get("tensions", [])
        parsed = parse_chord_symbol(root, quality, tensions)

    # Phase 2: Pin-First Workflow
    base_tones = get_chord_tones(parsed, bass_octave=5)  # Piano octave 5
    if not base_tones:
        base_tones = [60, 64, 67]  # C major fallback

    # 1) Check if tensions enabled for this section
    section_cfg = policy.get("sections", {}).get(section_label, {})
    tension_ratio = resolve_tension_ratio(policy, "piano", section_label, 0.35)
    use_extensions = tension_ratio >= 0.2 or (
        tension_ratio > 0.0 and np.random.random() < tension_ratio
    )

    # 2) Choose tensions if enabled
    pinned_tensions = []
    if use_extensions:
        allow_tensions = piano_cfg.get("tensions", {}).get("allow", [9, 11, 13])
        tension_mode = piano_cfg.get("tensions", {}).get("mode", "accent")
        tensions = choose_tensions(chord.get("symbol", "C"), allow_tensions, tension_mode)

        # Convert tension PC to MIDI (prefer upper register for piano)
        reg_pref = 72  # C5
        for t in tensions:
            tension_pc = t
            tension_note = tension_pc + ((reg_pref // 12) * 12)
            pinned_tensions.append(tension_note)

    # 3) Priority merge (tensions first)
    all_tones = []
    seen = set()
    for t in pinned_tensions + base_tones:
        pc = t % 12
        if pc not in seen:
            all_tones.append(t)
            seen.add(pc)

    pinned_mask = [True] * len(pinned_tensions) + [False] * (len(all_tones) - len(pinned_tensions))

    # 4) Determine target note count (ensure room for tensions)
    target_n = 4  # Comping: typically 3-4 notes
    target_n = max(target_n, len(pinned_tensions))

    # 5) Select notes (tensions have priority)
    chord_tones = all_tones[:target_n]
    pinned_mask = pinned_mask[:target_n]

    # === Phase 2: Register enforcement with pinned protection ===
    role = select_role(section_cfg, "piano", default_role="pad")
    chord_tones = ensure_register(
        chord_tones, "piano", policy, section_label, pinned_mask=pinned_mask
    )

    # Voice-leading (Phase 2) - NEW API
    if prev_voicing:
        reg_min = policy.get("instruments", {}).get("piano", {}).get("register", {}).get("min", 36)
        reg_max = policy.get("instruments", {}).get("piano", {}).get("register", {}).get("max", 84)
        chord_tones = voice_lead_voicing(
            chord_tones=chord_tones,
            prev_voicing=prev_voicing,
            reg_min=reg_min,
            reg_max=reg_max,
            max_step=7,  # Allow up to perfect 5th motion
        )

    # Apply open voicing if preferred (for thickness)
    if policy.get("global", {}).get("prefer_open_voicings", True) and role == "pad":
        reg_max = policy.get("instruments", {}).get("piano", {}).get("register", {}).get("max", 84)
        reg_min = policy.get("instruments", {}).get("piano", {}).get("register", {}).get("min", 43)
        chord_tones = spread_open_voicing(
            chord_tones, prefer_open=True, top_max=reg_max, low_min=reg_min
        )

    # Humanization
    humanize_ms = piano_cfg.get("humanize_timing_ms", 12)
    humanize_vel = piano_cfg.get("humanize_velocity", 7)
    base_velocity = piano_cfg.get("base_velocity", 75)

    # Phase 2.0: Apply EmotionAI velocity/duration scaling
    velocity_scale = 1.0
    duration_scale = 1.0
    density_scale = 1.0
    if emotion_params:
        velocity_scale = emotion_params.velocity_scale
        duration_scale = emotion_params.duration_scale
        density_scale = emotion_params.density_scale
        # Adjust base_velocity with energy
        base_velocity = int(base_velocity * velocity_scale)
        # Adjust effective_density
        effective_density *= density_scale

    # Phase 2.0: Adjust note count with GuideToneAI (optional)
    if guide_params and hasattr(guide_params, "notes_per_bar"):
        target_n = min(target_n, int(guide_params.notes_per_bar))

    # Get comping style from policy (section-based)
    comping_styles_cfg = piano_cfg.get("comping_styles", {}).get(section_label, [])
    if isinstance(comping_styles_cfg, list) and len(comping_styles_cfg) > 0:
        style_weights = {s["type"]: s["probability"] for s in comping_styles_cfg}
    else:
        # Default: offbeat_chord 50%, arpeggio 30%, block_chord 20%
        style_weights = {"offbeat_chord": 0.5, "arpeggio": 0.3, "block_chord": 0.2}

    # Select style
    style = np.random.choice(list(style_weights.keys()), p=list(style_weights.values()))

    if style == "arpeggio":
        # Ascending arpeggio - all chord_tones (no slicing)
        arp_prob = piano_cfg.get("arpeggio_probability", 0.15)
        if np.random.random() < arp_prob:
            for i, note in enumerate(chord_tones):
                time_ql = (
                    start_ql + i * 0.5 + np.random.uniform(-humanize_ms / 1000, humanize_ms / 1000)
                )
                vel = base_velocity + np.random.randint(-humanize_vel, humanize_vel)

                event = {
                    "bar_idx": bar_idx,
                    "time_ql": float(time_ql),
                    "note": int(note),
                    "velocity": int(np.clip(vel, 55, 90)),
                    "duration_ql": float(0.5 * duration_scale),  # Phase 2.0
                    "type": "comping",
                    "pattern": "arpeggio",
                    "event_type": "arpeggio",
                }
                # Phase 2: Mark tensions
                if i < len(pinned_mask) and pinned_mask[i]:
                    event["is_tension"] = True
                events.append(event)

    elif style == "block_chord":
        # Block chord on beats 1, 3 - all chord_tones
        for beat in [0, 2]:
            if np.random.random() < effective_density:
                time_ql = (
                    start_ql + beat + np.random.uniform(-humanize_ms / 1000, humanize_ms / 1000)
                )
                for i, note in enumerate(chord_tones):
                    vel = base_velocity + np.random.randint(-humanize_vel, humanize_vel)

                    event = {
                        "bar_idx": bar_idx,
                        "time_ql": float(time_ql),
                        "note": int(note),
                        "velocity": int(np.clip(vel, 60, 95)),
                        "duration_ql": float(1.0 * duration_scale),  # Phase 2.0
                        "type": "comping",
                        "pattern": "block_chord",
                        "event_type": "comping",
                    }
                    # Phase 2: Mark tensions
                    if i < len(pinned_mask) and pinned_mask[i]:
                        event["is_tension"] = True
                    events.append(event)

    elif style == "offbeat_chord":
        # Offbeat chord on beats 2, 4 - limit to 3 notes
        for beat in [1, 3]:
            if np.random.random() < effective_density:
                time_ql = (
                    start_ql + beat + np.random.uniform(-humanize_ms / 1000, humanize_ms / 1000)
                )
                for i, note in enumerate(chord_tones[:3]):
                    vel = base_velocity - 5 + np.random.randint(-humanize_vel, humanize_vel)

                    event = {
                        "bar_idx": bar_idx,
                        "time_ql": float(time_ql),
                        "note": int(note),
                        "velocity": int(np.clip(vel, 55, 85)),
                        "duration_ql": float(0.5 * duration_scale),  # Phase 2.0
                        "type": "comping",
                        "pattern": "offbeat_chord",
                        "event_type": "comping",
                    }
                    # Phase 2: Mark tensions
                    if i < len(pinned_mask) and pinned_mask[i]:
                        event["is_tension"] = True
                    events.append(event)

    # Phase 2: Return tuple (events, final_voicing)
    final_voicing = sorted(chord_tones)
    return (events, final_voicing)


def generate_piano_plan(
    bars: pd.DataFrame,
    sections: List[Dict[str, Any]],
    chordmap: List[Dict[str, Any]],
    policy: Dict[str, Any],
    melody_hints: Optional[Dict[int, MelodyHint]] = None,
    rulebook=None,
    lyric_index=None,
    emotion_ai=None,
    guidetone_ai=None,
    reference_layers: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Main logic: Generate slot-based piano plan with Phase 2.0 AI integration.

    Args:
        bars: bars_with_slots.parquet
        sections: sections.json
        chordmap: chordmap_locked_extended.json
        policy: policy YAML
        rulebook: Rulebook instance (optional, Phase 2.0)
        lyric_index: LyricAnchorIndex instance (optional, Phase 2.0)
        emotion_ai: EmotionAI v2 instance (optional, Phase 2.0)
        guidetone_ai: GuideToneAI v2 instance (optional, Phase 2.0)

    Returns:
        Piano plan JSON
    """
    piano_cfg = policy.get("instruments", {}).get("piano", {})
    sections_density = policy.get("sections", {})

    events = []
    prev_voicing = None  # Phase 2: Track voicing for voice-leading
    emotion_log: Dict[int, Dict[str, Any]] = {}

    for _, bar_row in bars.iterrows():
        bar_idx = int(bar_row["bar_idx"])
        section_label = bar_row.get("section_label", "verse")
        fill_slot = bar_row.get("fill_slot", False)

        # Get chord
        chord = get_chord_at_bar(chordmap, bar_idx)

        # Get section density
        section_cfg = sections_density.get(section_label, {})
        piano_density = section_cfg.get("piano", 0.5)

        # Phase 2.0: Build AI context
        context = {
            "bar_index": bar_idx,
            "bar": bar_idx,
            "section": section_label,
            "role": "piano",
            "chord_symbol": chord.get("symbol", "C"),
            "slots": {"fill": fill_slot},
        }

        if reference_layers:
            context["reference_layers"] = reference_layers

        # Phase 2.0: Get lyric info
        if lyric_index:
            lyric_info = lyric_index.get_bar_info(bar_idx)
            if lyric_info and lyric_info.get("has_anchor"):
                context["lyric"] = {
                    "phrase_role": lyric_info["phrase_role"],
                    "stress_level": lyric_info.get("stress_level", 0.0),
                    "is_silent": lyric_info.get("is_silent", False),
                }

        # Phase 2.0: Get EmotionParams
        emotion_params = None
        if emotion_ai:
            emotion_params = emotion_ai.get_params(context)
            record_emotion_snapshot(
                emotion_log,
                bar_idx=bar_idx,
                section_label=section_label,
                emotion_params=emotion_params,
            )

        # Phase 2.0: Get GuideTonePlan
        guide_params = None
        if guidetone_ai:
            guide_params = guidetone_ai.get_plan(context)

        # Decision: Fill or Comping (Phase 2: returns tuple)
        # Use fill_slot directly (don't require is_section_boundary)
        if fill_slot:
            # Fire fill decoration
            result = make_fill_decoration(
                bar_idx,
                bar_row,
                chord,
                piano_cfg,
                policy,
                section_label,
                prev_voicing,
                emotion_params=emotion_params,
                guide_params=guide_params,
                reference_layers=reference_layers,
            )
            bar_events, prev_voicing = result
        else:
            # Comping
            result = make_comping_pattern(
                bar_idx,
                bar_row,
                chord,
                piano_cfg,
                policy,
                piano_density,
                section_label,
                prev_voicing,
                emotion_params=emotion_params,
                guide_params=guide_params,
            )
            bar_events, new_voicing = result
            if new_voicing:  # Only update if comping returned valid voicing
                prev_voicing = new_voicing

        # Activity floor: ensure min_notes_per_bar
        bar_start_ql = bar_idx * 4.0
        bar_end_ql = (bar_idx + 1) * 4.0
        min_notes = piano_cfg.get("min_notes_per_bar", 2)

        # Get chord tones for chordpad - handle both symbol and root+quality formats
        if "symbol" in chord and chord["symbol"]:
            parsed = parse_symbol(chord["symbol"])
        else:
            # Use root+quality format (chordmap_locked.json)
            root = chord.get("root", "C")
            quality = chord.get("quality", "")
            tensions = chord.get("tensions", [])
            parsed = parse_chord_symbol(root, quality, tensions)

        chord_tones = get_chord_tones(parsed, bass_octave=5)

        # Ensure chordpad pitches are within register
        chord_tones_in_register = ensure_register(chord_tones, "piano", policy, section_label)
        chordpad_pitches = chord_tones_in_register[-3:] if chord_tones_in_register else [60, 64, 67]

        # Phase 2: Generate tension candidates for floor padding
        section_cfg = policy.get("sections", {}).get(section_label, {})
        tension_ratio = resolve_tension_ratio(policy, "piano", section_label, 0.35)
        tension_pitches = []

        if tension_ratio > 0.0:
            allow_tensions = piano_cfg.get("tensions", {}).get("allow", [9, 11, 13])
            tension_mode = piano_cfg.get("tensions", {}).get("mode", "accent")
            tension_pcs = choose_tensions(chord.get("symbol", "C"), allow_tensions, tension_mode)

            # Convert tension PCs to MIDI pitches (piano upper register)
            reg_pref = 72  # C5
            for t in tension_pcs:
                tension_note = t + ((reg_pref // 12) * 12)
                tension_pitches.append(tension_note)

        # Ensure tensions are in register
        if tension_pitches:
            tension_pitches = ensure_register(tension_pitches, "piano", policy, section_label)

        bar_events = ensure_activity_floor(
            bar_events,
            bar_start_ql,
            bar_end_ql,
            min_notes,
            chordpad_pitches,
            velocity=65,
            tension_pitches=tension_pitches,  # Phase 2: Pass tension candidates
        )

        # Phase 3: Add metadata for humanize (beat_in_bar, section_label)
        for ev in bar_events:
            if "beat_in_bar" not in ev:
                ev["beat_in_bar"] = (ev.get("time_ql", 0.0) - bar_start_ql) / 1.0
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
            policy, section_label, "piano", song_id=policy.get("metadata", {}).get("song_id")
        )

        section_events = apply_timing_humanize(section_events, humanize_cfg, tempo_bpm)
        section_events = apply_velocity_humanize(section_events, humanize_cfg, base_velocity=80)
        section_events = apply_duration_humanize(
            section_events, humanize_cfg, bar_duration_ql=4.0
        )  # Phase 3.5

        humanized_events.extend(section_events)

    events = sorted(humanized_events, key=lambda e: e["time_ql"])

    # Phase 3.6: Long Note Extension (Piano → bar内延長)
    inst_cfg = policy.get("instruments", {}).get("piano", {})
    sustain_cfg = inst_cfg.get("sustain", {})
    min_dur = sustain_cfg.get("min_duration_ql", 2.0)  # 2拍以上を長音扱い
    bar_span = sustain_cfg.get("max_bar_span", 1)  # bar内に収める

    events = extend_last_note_per_bar(
        events=events,
        bars_df=bars,
        role="piano",
        min_duration_ql=min_dur,
        max_bar_span=bar_span,
    )

    filter_stats = {"annotated": 0, "removed": 0}
    if melody_hints:
        events, filter_stats = apply_melody_hint_filter(
            events,
            melody_hints,
            instrument="piano",
            drop_tags=(),  # Piano keeps hints but still annotates
            annotate=True,
        )

    # Metadata
    metadata = {
        "instrument": "piano",
        "num_bars": int(len(bars)),
        "num_events": len(events),
        "fill_slots_used": int(bars["fill_slot"].sum()),
        "generator": "generate_piano_plan_v2.py",
        "humanize_profile": policy.get("humanize", {}).get("profile", "pop_easy"),
        "sustain_mode": "last_note_per_bar",  # Phase 3.6
        "sustain_config": {"min_duration_ql": min_dur, "max_bar_span": bar_span},
        "melody_hint": {
            "annotated": filter_stats.get("annotated", 0),
            "removed_for_strings": 0,
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
    parser = argparse.ArgumentParser(
        description="Generate piano plan (slot-based V2 + Phase 2.0 AI)"
    )
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
    parser.add_argument("--out", required=True, help="Output piano_plan.json")
    parser.add_argument("--seed", type=int, help="Random seed for reproducibility")

    # Phase 2.0: AI Integration arguments (optional)
    parser.add_argument("--lyric-anchors", help="Path to lyric_anchors.json")
    parser.add_argument("--emotion-profile", help="Path to emotion_profile.json")
    parser.add_argument("--guide-hints", help="Path to guide_tone_hints.json")
    parser.add_argument("--rulebook", help="Path to rulebook.yaml")
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

    vocal_f0 = load_vocal_f0(args.vocal_f0) if args.vocal_f0 else None
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

    # Phase 2.0: AI Integration (optional)
    rulebook = None
    lyric_index = None
    emotion_ai = None
    guidetone_ai = None

    if args.rulebook and Rulebook:
        print(f"📖 Loading Rulebook from {args.rulebook}")
        rulebook = Rulebook.load(Path(args.rulebook))

    if args.lyric_anchors and LyricAnchorIndex:
        print(f"📝 Loading LyricAnchorIndex from {args.lyric_anchors}")
        with open(args.lyric_anchors, "r") as f:
            anchors_data = json.load(f)
        # Get tempo from policy
        tempo_bpm = policy.get("global", {}).get("tempo_bpm", 120)
        lyric_index = LyricAnchorIndex(anchors=anchors_data, tempo_bpm=tempo_bpm)

    if args.emotion_profile and EmotionAIv2 and rulebook:
        print(f"🎭 Loading EmotionAI v2 from {args.emotion_profile}")
        with open(args.emotion_profile, "r", encoding="utf-8") as f:
            raw_profile = json.load(f)
        emotion_profile_data = _prepare_emotion_profile(raw_profile)
        if emotion_profile_data:
            emotion_ai = EmotionAIv2(emotion_profile=emotion_profile_data, rulebook=rulebook)
        else:
            print("⚠️  Emotion profile contained no usable entries; skipping EmotionAI integration.")

    if args.guide_hints and GuideToneAIv2 and rulebook:
        print(f"🎵 Loading GuideToneAI v2 from {args.guide_hints}")
        with open(args.guide_hints, "r") as f:
            guide_hints_data = json.load(f)
        guidetone_ai = GuideToneAIv2(guide_tone_hints=guide_hints_data, rulebook=rulebook)

    print(f"🎹 Generating piano plan ({len(bars)} bars)")
    plan = generate_piano_plan(
        bars,
        sections,
        chordmap,
        policy,
        melody_hints,
        rulebook=rulebook,
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
        instrument="piano",
        policy=policy,
        default_section=default_section,
        song_id=song_id,
    )
    if rhythm_stats.get("assigned", 0) > 0:
        plan_meta["rhythm_vocab_ids"] = rhythm_stats["used_ids"]
        plan_meta["rhythm_vocab_instrument"] = "piano"
        plan_meta.setdefault("ai_hooks", {}).update({"rhythm_vocab_policy": True})

    rhythm_manifest_path = Path(args.rhythm_manifest).expanduser() if args.rhythm_manifest else None
    if DurationHumanizeAI is not None:
        try:
            duration_ai = DurationHumanizeAI(
                instrument="piano",
                policy=policy,
                tempo_bpm=policy.get("global", {}).get("tempo_bpm", 120),
                rhythm_manifest_path=rhythm_manifest_path,
                vocab_instrument="piano",
            )
            duration_ai.annotate_plan(plan)
            plan_meta.setdefault("ai_hooks", {}).update({"duration_humanize_ai": True})
        except Exception as exc:  # pragma: no cover - diagnostics only
            print(f"⚠️  DurationHumanizeAI annotation skipped: {exc}")

    # Save
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(plan, f, indent=2, ensure_ascii=False)

    print(f"✅ Piano plan saved to {out_path}")
    print(f"   Events: {plan['metadata']['num_events']}")
    print(f"   Fill slots used: {plan['metadata']['fill_slots_used']}/{len(bars)}")


if __name__ == "__main__":
    main()
