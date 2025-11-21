#!/usr/bin/env python3
"""
v2_common.py - Shared utilities for V2 generators (activity_floor, tension, humanize, register).

Purpose:
    Provide policy-driven helper functions for:
    1. Activity floor (ensure min notes per bar with chordpad fallback)
    2. Tension adoption (section-based tension_ratio)
    3. Light humanize (timing jitter, velocity jitter)
    4. Register enforcement (clamp/drop notes to instrument range)
    5. Voicing spread (open voicings for thickness)

Usage:
    from v2_common import (
        ensure_activity_floor, choose_tension_enabled, apply_humanize,
        clamp_to_register, ensure_register, spread_open_voicing,
        select_role, decide_tension_use
    )
"""
from typing import Any, Dict, List, Mapping, Set, Tuple
import random
import numpy as np


def choose_tension_enabled(section_cfg: Dict[str, Any], rng: random.Random = None) -> bool:
    """
    Decide whether to enable tension based on section's tension_ratio.

    Args:
        section_cfg: policy['sections'][section_label]
        rng: Random generator (optional)

    Returns:
        True if tension should be used (probabilistic)
    """
    if rng is None:
        rng = random.Random()

    t_ratio = float(section_cfg.get("tension_ratio", 0.0))
    return rng.random() < max(0.0, min(1.0, t_ratio))


def ensure_activity_floor(
    events: List[Dict[str, Any]],
    bar_start_ql: float,
    bar_end_ql: float,
    min_notes: int,
    chordpad_pitches: List[int],
    velocity: int = 65,
    tension_pitches: List[int] | None = None,
) -> List[Dict[str, Any]]:
    """
    Ensure bar has at least min_notes events. If not, add sustained chordpad notes.

    Phase 2 Enhancement:
        - Prioritizes remaining tension notes before falling back to triad
        - Marks added tension notes with is_tension: true
        - This prevents "tension selected but lost during floor padding" issue

    Args:
        events: Existing events (will be modified in-place)
        bar_start_ql: Bar start time (quarter notes)
        bar_end_ql: Bar end time (quarter notes)
        min_notes: Minimum notes required
        chordpad_pitches: MIDI pitches for chordpad (typically top 2-3 chord tones)
        velocity: Velocity for added notes
        tension_pitches: Optional list of tension notes to prioritize (Phase 2)

    Returns:
        Modified events list
    """
    if min_notes <= 0 or not chordpad_pitches:
        return events

    # Count events in this bar
    existing_notes = sum(
        1 for e in events if bar_start_ql <= e.get("time_ql", e.get("time", 0.0)) < bar_end_ql
    )

    if existing_notes >= min_notes:
        return events

    # Add chordpad to fill gap
    missing = min_notes - existing_notes
    bar_duration = bar_end_ql - bar_start_ql

    # Phase 2: Prioritize tension notes, then fall back to triad
    tension_pool = list(tension_pitches) if tension_pitches else []
    triad_pool = list(chordpad_pitches)

    for i in range(missing):
        # Try to use tension first
        if tension_pool:
            note = tension_pool.pop(0)
            is_tension = True
        else:
            note = triad_pool[i % len(triad_pool)]
            is_tension = False

        event = {
            "time_ql": float(bar_start_ql + 0.001 * i),  # Slight offset to avoid collision
            "note": int(note),
            "velocity": int(velocity),
            "duration_ql": float(bar_duration * 0.95),  # Sustain for most of bar
            "type": "chordpad_floor",
            "event_type": "pad",
        }

        # Mark tension notes for QAgate
        if is_tension:
            event["is_tension"] = True

        events.append(event)

    return events


def apply_rhythm_vocab_annotations(
    events: List[Dict[str, Any]],
    *,
    instrument: str,
    policy: Mapping[str, Any],
    default_section: str | None = None,
    song_id: str | None = None,
) -> Dict[str, Any]:
    """Populate ``rhythm_pattern_id`` on events using policy ``rhythm_vocab`` hints."""

    instrument_cfg = (policy.get("instruments", {}) or {}).get(instrument, {})
    vocab_cfg = instrument_cfg.get("rhythm_vocab") or {}
    if not vocab_cfg:
        return {"instrument": instrument, "assigned": 0, "used_ids": []}

    default_ids = list(vocab_cfg.get("default_ids") or [])
    section_table = {
        str(name).lower(): dict(cfg) for name, cfg in (vocab_cfg.get("sections", {}) or {}).items()
    }
    descriptors_default = list(vocab_cfg.get("descriptors", []) or [])
    density_default = vocab_cfg.get("density")

    rng_seed = vocab_cfg.get("seed")
    if rng_seed is None and song_id:
        rng_seed = hash(f"{instrument}:{song_id}") & 0x7FFFFFFF
    rng = random.Random(rng_seed)

    assigned = 0
    used_ids: Set[str] = set()
    touched_sections: Set[str] = set()

    for event in events:
        if event.get("rhythm_pattern_id"):
            continue
        section_label = event.get("section") or event.get("section_label") or default_section
        section_norm = str(section_label or "verse").lower()
        section_cfg = section_table.get(section_norm, {})
        candidate_ids = list(section_cfg.get("preferred_ids") or default_ids)
        if not candidate_ids:
            continue
        pattern_id = rng.choice(candidate_ids) if len(candidate_ids) > 1 else candidate_ids[0]
        event["rhythm_pattern_id"] = pattern_id
        descriptors = section_cfg.get("descriptors") or descriptors_default
        if descriptors:
            event["rhythm_pattern_descriptors"] = list(descriptors)
        density = section_cfg.get("density") or density_default
        if density:
            event["rhythm_pattern_density"] = density
        assigned += 1
        used_ids.add(pattern_id)
        touched_sections.add(section_norm)

    return {
        "instrument": instrument,
        "assigned": assigned,
        "used_ids": sorted(used_ids),
        "sections": sorted(touched_sections),
    }


def record_emotion_snapshot(
    target: Dict[int, Dict[str, Any]],
    *,
    bar_idx: int,
    section_label: str,
    emotion_params: Any,
) -> None:
    """Capture bar-level EmotionAI outputs for downstream annotators."""

    if target is None or emotion_params is None:
        return

    def _coerce(value: Any) -> float | None:
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    payload: Dict[str, Any] = {"section": str(section_label or "unknown").lower()}
    numeric_fields = (
        "energy",
        "tension",
        "brightness",
        "valence",
        "velocity_scale",
        "duration_scale",
        "density_scale",
    )
    source = emotion_params
    for field in numeric_fields:
        value = getattr(source, field, None)
        if value is None and isinstance(source, Mapping):
            value = source.get(field)
        coerced = _coerce(value)
        if coerced is not None:
            payload[field] = coerced

    phrase_role = getattr(source, "phrase_role", None)
    if phrase_role is None and isinstance(source, Mapping):
        phrase_role = source.get("phrase_role")
    if phrase_role:
        payload["phrase_role"] = str(phrase_role)

    tags = getattr(source, "tags", None)
    if tags is None and isinstance(source, Mapping):
        tags = source.get("tags")
    if tags:
        payload["tags"] = sorted({str(tag) for tag in tags if tag})

    target[int(bar_idx)] = payload


def summarize_emotion_log(data: Dict[int, Dict[str, Any]]) -> Dict[str, Any]:
    """Return JSON-friendly per-bar and per-section emotion summaries."""

    if not data:
        return {}

    per_bar = {str(bar): dict(payload) for bar, payload in sorted(data.items())}
    numeric_fields = (
        "energy",
        "tension",
        "brightness",
        "valence",
        "velocity_scale",
        "duration_scale",
        "density_scale",
    )

    section_totals: Dict[str, Dict[str, Any]] = {}
    for payload in data.values():
        section = payload.get("section", "unknown")
        entry = section_totals.setdefault(
            section,
            {
                "count": 0,
                "tags": set(),
                **{field: 0.0 for field in numeric_fields},
            },
        )
        entry["count"] += 1
        tags = payload.get("tags") or []
        if isinstance(tags, (list, set, tuple)):
            entry["tags"].update({str(tag) for tag in tags if tag})
        for field in numeric_fields:
            value = payload.get(field)
            if isinstance(value, (int, float)):
                entry[field] += float(value)

    section_summary: Dict[str, Any] = {}
    for section, entry in section_totals.items():
        count = max(1, int(entry.pop("count", 1)))
        tags = sorted(entry.pop("tags", set()))
        averages = {
            field: round(entry[field] / count, 5) for field in numeric_fields if field in entry
        }
        if tags:
            averages["tags"] = tags
        section_summary[section] = averages

    return {"per_bar": per_bar, "section_avg": section_summary}


def apply_humanize(
    events: List[Dict[str, Any]],
    timing_std_ms: float = 8.0,
    velocity_jitter: int = 6,
    legato_chance: float = 0.15,
    rng: Any = None,
) -> List[Dict[str, Any]]:
    """
    Apply lightweight humanization (timing jitter, velocity jitter, legato).

    Args:
        events: Events to humanize (will be modified in-place)
        timing_std_ms: Standard deviation for timing jitter (milliseconds)
        velocity_jitter: Standard deviation for velocity jitter
        legato_chance: Probability of extending note duration (legato)
        rng: Random generator (random.Random or np.random.RandomState)

    Returns:
        Modified events list
    """
    if rng is None:
        rng = random.Random()

    if timing_std_ms <= 0 and velocity_jitter <= 0 and legato_chance <= 0:
        return events

    # Check if rng is numpy or stdlib random
    has_gauss = hasattr(rng, "gauss")
    has_normal = hasattr(rng, "normal")
    has_uniform = hasattr(rng, "uniform")

    for e in events:
        # Timing jitter (convert ms to beats; rough approximation at 120 BPM)
        if timing_std_ms > 0:
            if has_gauss:
                jitter_sec = rng.gauss(0.0, timing_std_ms / 1000.0)
            elif has_normal:
                jitter_sec = rng.normal(0.0, timing_std_ms / 1000.0)
            else:
                jitter_sec = random.gauss(0.0, timing_std_ms / 1000.0)

            jitter_beats = jitter_sec / 0.5  # Approximate: 120 BPM = 0.5s per beat

            if "time_ql" in e:
                e["time_ql"] = max(0.0, e["time_ql"] + jitter_beats)
            elif "time" in e:
                e["time"] = max(0.0, e["time"] + jitter_beats)

        # Velocity jitter
        if velocity_jitter > 0 and "velocity" in e:
            if has_gauss:
                vel_change = int(round(rng.gauss(0.0, velocity_jitter)))
            elif has_normal:
                vel_change = int(round(rng.normal(0.0, velocity_jitter)))
            else:
                vel_change = int(round(random.gauss(0.0, velocity_jitter)))
            e["velocity"] = max(1, min(127, e["velocity"] + vel_change))

        # Legato (extend duration slightly)
        if legato_chance > 0:
            rand_val = (
                rng.uniform(0, 1)
                if has_uniform
                else (rng.random() if hasattr(rng, "random") else random.random())
            )
            if rand_val < legato_chance:
                if "duration_ql" in e:
                    e["duration_ql"] = e["duration_ql"] * 1.15
                elif "duration" in e:
                    e["duration"] = e["duration"] * 1.15

    return events


def pitch_of(root: str, octave: int) -> int:
    """
    Convert note name + octave to MIDI pitch.

    Args:
        root: Note name (e.g., "C", "C#", "Db")
        octave: MIDI octave (C4 = 60)

    Returns:
        MIDI pitch number
    """
    NOTE_BASE = {
        "C": 0,
        "C#": 1,
        "Db": 1,
        "D": 2,
        "D#": 3,
        "Eb": 3,
        "E": 4,
        "F": 5,
        "F#": 6,
        "Gb": 6,
        "G": 7,
        "G#": 8,
        "Ab": 8,
        "A": 9,
        "A#": 10,
        "Bb": 10,
        "B": 11,
    }
    return 12 * (octave + 1) + NOTE_BASE.get(root, 0)


# ========== Phase 1: Register & Voicing ==========


def _get_inst_register(inst: str, policy: Dict[str, Any], section: str) -> Tuple[int, int]:
    """
    Get instrument register (min, max) from policy.

    Priority: sections.{section}.register > instruments.{inst}.register > (0, 127)

    Args:
        inst: Instrument name (guitar, piano, strings, bass)
        policy: Policy dictionary
        section: Section label (intro, verse, chorus, etc.)

    Returns:
        (min_pitch, max_pitch) tuple
    """
    # Section-specific register (if any)
    reg = policy.get("sections", {}).get(section, {}).get("register")
    if not reg:
        # Instrument-level register
        reg = policy.get("instruments", {}).get(inst, {}).get("register")
    if not reg:
        return (0, 127)  # No constraints

    return int(reg.get("min", 0)), int(reg.get("max", 127))


def fold_to_register(pitch: int, rmin: int, rmax: int, prefer: int | None = None) -> int:
    """
    折り返しで [rmin, rmax] に入れる。prefer があれば"最短距離"優先。

    Phase 2: テンション保持のため、境界で丸めず±12で折り返す。

    Args:
        pitch: MIDI pitch to fold
        rmin: Register minimum
        rmax: Register maximum
        prefer: Preferred octave center (for shortest distance calculation)

    Returns:
        Folded pitch within [rmin, rmax]
    """
    if rmin <= pitch <= rmax:
        return pitch

    if prefer is None:
        prefer = (rmin + rmax) // 2

    # ±12 ずつ動かして最短で入れる
    candidates = []
    for k in range(-6, 7):  # ±6oct あれば十分
        p = pitch + 12 * k
        if rmin <= p <= rmax:
            candidates.append(p)

    if not candidates:
        # 最後の砦：境界へ丸め
        return rmin if pitch < rmin else rmax

    return min(candidates, key=lambda p: abs(p - prefer))


def clamp_to_register(pitch: int, inst: str, policy: Dict[str, Any], section: str) -> int:
    """
    Clamp a single pitch to instrument register.

    Args:
        pitch: MIDI pitch
        inst: Instrument name
        policy: Policy dictionary
        section: Section label

    Returns:
        Clamped pitch
    """
    lo, hi = _get_inst_register(inst, policy, section)
    return max(lo, min(hi, pitch))


def ensure_register(
    notes: List[int],
    inst: str,
    policy: Dict[str, Any],
    section: str,
    pinned_mask: List[bool] | None = None,
) -> List[int]:
    """
    Enforce instrument register on list of notes with smart octave shifting.

    Phase 2 Enhancement:
        - ピン留めされた音（tension等）は極力残す
        - fallback='clamp' は折り返し（fold）に変更
        - pinned は drop モードでも折り返して保持

    Args:
        notes: List of MIDI pitches
        inst: Instrument name
        policy: Policy dictionary
        section: Section label
        pinned_mask: Optional mask indicating which notes are pinned (must be preserved)

    Returns:
        Filtered/folded notes (deduplicated, preserving pinned notes)
    """
    if not notes:
        return []

    mode = policy.get("global", {}).get("register_fallback", "clamp")
    lo, hi = _get_inst_register(inst, policy, section)

    # Get octave preference for smarter shifting
    reg_cfg = policy.get("instruments", {}).get(inst, {}).get("register", {})
    prefer = int(reg_cfg.get("octave_prefer", (lo + hi) // 2))

    if pinned_mask is None:
        pinned_mask = [False] * len(notes)

    out: List[int] = []
    for i, n in enumerate(notes):
        if lo <= n <= hi:
            out.append(n)
        elif mode == "clamp":
            # 折り返しで範囲内に入れる (Phase 2: preserve tensions)
            out.append(fold_to_register(n, lo, hi, prefer))
        else:
            # drop モードでも pinned は折り返して保持
            if pinned_mask[i]:
                out.append(fold_to_register(n, lo, hi, prefer))
            # 非 pinned なら捨て

    # Deduplicate while preserving order
    dedup = []
    for x in out:
        if x not in dedup:
            dedup.append(x)

    # Optional open-voicing re-spread (keeps chord color) - ChatGPT suggestion
    # Phase 2 DEBUG: Temporarily disabled to prevent tension loss
    # prefer_open = policy.get("global", {}).get("prefer_open_voicings", True)
    # if prefer_open and len(dedup) >= 3:
    #     dedup = spread_open_voicing(dedup, prefer_open=True, top_max=hi, low_min=lo)

    return dedup


def spread_open_voicing(
    tones: List[int],
    prefer_open: bool = True,
    top_max: int | None = None,
    low_min: int | None = None,
) -> List[int]:
    """
    Spread chord tones into open voicing for thickness.

    Strategy: Lower root by octave, raise thirds/fifths by octave.

    Phase 1 Fix: Respect low_min to avoid going below register minimum.

    Args:
        tones: Chord tones (ascending)
        prefer_open: Whether to apply spreading
        top_max: Maximum pitch for top note (clamp if exceeded)
        low_min: Minimum pitch for lowest note (don't lower root below this)

    Returns:
        Spread tones
    """
    if not tones or not prefer_open:
        return tones

    # Simple spread: root down (if safe), upper notes up
    root = tones[0]
    lowered_root = root - 12

    # Check if lowering root violates low_min
    if low_min is not None and lowered_root < low_min:
        # Don't lower root, keep original
        out = [root]
    else:
        out = [lowered_root]

    for i, t in enumerate(tones):
        if i == 0:
            continue  # Root already added
        # Raise thirds/fifths/sevenths by octave
        spread_note = t + 12
        out.append(spread_note)

    # Clamp top note if needed
    if top_max is not None:
        out = [min(t, top_max) for t in out]

    return sorted(out)  # Keep ascending order


def select_role(section_cfg: Dict[str, Any], inst: str, default_role: str) -> str:
    """
    Select instrument role from section configuration.

    Args:
        section_cfg: policy['sections'][section_label]
        inst: Instrument name
        default_role: Fallback role

    Returns:
        Role string (pad, riff, arp, counter, etc.)
    """
    return section_cfg.get("roles", {}).get(inst, default_role)


def decide_tension_use(section_cfg: Dict[str, Any]) -> bool:
    """
    Decide whether to use tension based on section's tension_ratio.

    Args:
        section_cfg: policy['sections'][section_label]

    Returns:
        True if tension should be used (probabilistic)
    """
    r = float(section_cfg.get("tension_ratio", 0.0))
    return random.random() < max(0.0, min(1.0, r))


# ============================================================
# Phase 2: Tension / Voice-Leading / Countermelody
# ============================================================


def resolve_tension_ratio(
    policy: Dict[str, Any],
    instrument: str,
    section_name: str,
    default: float = 0.0,
) -> float:
    """
    Resolve tension_ratio from policy with priority fallback.

    Priority (max value wins):
      1. sections.<section>.tension_ratio
      2. instruments.<inst>.tension_ratio
      3. global.tension_ratio
      4. default parameter

    Args:
        policy: Policy dictionary
        instrument: Instrument name (guitar, piano, strings, bass)
        section_name: Section label (intro, verse, chorus, etc.)
        default: Fallback value if no policy value found

    Returns:
        Resolved tension_ratio clamped to [0.0, 1.0]
    """
    if policy is None:
        return float(default)

    # global
    global_cfg = policy.get("global") or {}
    t_global = float(global_cfg.get("tension_ratio", 0.0))

    # instrument
    inst_cfg = (policy.get("instruments") or {}).get(instrument, {}) or {}
    t_inst = float(inst_cfg.get("tension_ratio", 0.0))

    # section
    sec_cfg = (policy.get("sections") or {}).get(section_name, {}) or {}
    t_sec = float(sec_cfg.get("tension_ratio", 0.0))

    # Take max (highest priority wins)
    t = max(t_global, t_inst, t_sec, float(default))

    # Clamp to [0.0, 1.0]
    return max(0.0, min(1.0, t))


def choose_tensions(symbol: str, allow: List[int], mode: str) -> List[int]:
    """
    Choose tension extensions (9/11/13) based on chord symbol and mode.

    Phase 2 Enhancement:
        - Respects chord flags (sus4, add9, maj7, etc.)
        - Mode controls number and emphasis:
          - 'lead': aggressive (2 tensions)
          - 'accent': moderate (1 tension)
          - 'pad': subtle (1 tension, first available)
          - 'none': no tensions

    Args:
        symbol: Chord symbol string (e.g., "Cmaj7") OR chord dict with "flags"
        allow: List of allowed tension numbers [9, 11, 13]
        mode: Tension mode from section config

    Returns:
        List of tension numbers to add
    """
    # Support both string symbol and chord dict
    if isinstance(symbol, dict):
        flags = symbol.get("flags", [])
    elif isinstance(symbol, str):
        try:
            from chordmap_utils import parse_symbol

            # parse_symbol returns (root, quality)
            # For now, skip parsing and just check for "sus4" in string
            flags = ["sus4"] if "sus4" in symbol else []
        except ImportError:
            flags = []
    else:
        flags = []

    # Respect sus4: avoid 11 (conflicts with sus4)
    if "sus4" in flags:
        cand = [9] if 9 in allow else []
    else:
        cand = [x for x in [9, 11, 13] if x in allow]

    if not cand:
        return []

    if mode == "lead":
        # Aggressive: up to 2 tensions
        k = min(2, len(cand))
        return random.sample(cand, k=k)
    elif mode == "accent":
        # Moderate: 1 random tension
        return [random.choice(cand)]
    elif mode == "pad":
        # Subtle: first available tension only
        return [cand[0]]
    else:  # 'none' or unknown
        return []


def voice_lead_voicing(
    chord_tones: List[int],
    prev_voicing: List[int] | None = None,
    reg_min: int | None = None,
    reg_max: int | None = None,
    max_step: int = 7,
) -> List[int]:
    """
    Voice-lead chord_tones to minimize motion from previous voicing.

    Phase 2 Redesign (ChatGPT):
        - NEVER drops triad + tension notes
        - Only wraps to register via ±12 (no deletion)
        - Minimal movement optimization WITHOUT destroying chord structure
        - If no prev_voicing, just folds to register

    Args:
        chord_tones: MIDI pitches to voice-lead (triad + tensions)
        prev_voicing: Previous voicing (optional)
        reg_min: Minimum MIDI pitch for register
        reg_max: Maximum MIDI pitch for register
        max_step: Maximum allowed semitone step per voice

    Returns:
        Adjusted voicing (same length as input, deduplicated)
    """
    if not chord_tones:
        return []

    tones = sorted(int(n) for n in chord_tones)

    # Auto-detect register if not specified
    if reg_min is None or reg_max is None:
        center = sum(tones) / len(tones)
        span = 24  # 2 octaves
        if reg_min is None:
            reg_min = int(center - span / 2)
        if reg_max is None:
            reg_max = int(center + span / 2)

    # No previous voicing: just fold to register
    if not prev_voicing:
        adjusted = []
        for n in tones:
            m = n
            while m < reg_min:
                m += 12
            while m > reg_max:
                m -= 12
            adjusted.append(m)
        return sorted(set(adjusted))

    prev = sorted(int(n) for n in prev_voicing)
    adjusted = []

    # For each note, find best octave transposition (±2 octaves)
    for n in tones:
        candidates = []
        for k in range(-2, 3):  # ±2 octaves
            m = n + 12 * k
            if reg_min <= m <= reg_max:
                candidates.append(m)

        if not candidates:
            # Last resort: use original note
            candidates = [n]

        # Cost: distance to closest prev note + penalty for large jumps
        best = None
        best_cost = None
        for m in candidates:
            dist_prev = min(abs(m - p) for p in prev) if prev else 0
            dist_orig = abs(m - n)

            # Weight: prev distance (heavier) + original distance (lighter)
            cost = dist_prev + 0.5 * dist_orig

            # Penalize jumps > max_step
            if dist_prev > max_step:
                cost += (dist_prev - max_step) * 2

            if best is None or cost < best_cost:
                best = m
                best_cost = cost

        adjusted.append(best)

    # Deduplicate while preserving order
    return sorted(set(adjusted))


def generate_countermelody(
    bars_df,
    sections: List[str],
    anchors: Dict,
    scale_pitches: List[int],
    register: Tuple[int, int],
    density: float,
    avoid_stress: bool,
    motif_span_beats: Tuple[float, float],
) -> List[Dict]:
    """
    Generate simple countermelody for specified sections.

    Phase 2 Enhancement:
        - Places scale-based melodic fragments in target sections
        - Avoids vocal stress points (if avoid_stress=True)
        - Uses motif_span_beats for phrase length control
        - Density controls note count per bar

    Args:
        bars_df: DataFrame with bar timing info
        sections: List of section labels to apply countermelody
        anchors: Lyric anchors dictionary (optional, for stress avoidance)
        scale_pitches: List of scale MIDI pitches
        register: (min, max) MIDI pitch range
        density: Target note density (notes per bar ≈ density * 2)
        avoid_stress: Whether to avoid vocal stress points
        motif_span_beats: (min, max) beats for motif duration

    Returns:
        List of event dicts: [{"time": float, "duration": float, "note": int, "vel": int}, ...]
    """
    events = []
    lo, hi = register

    for _, bar in bars_df.iterrows():
        sec = bar.get("section_label", "")
        if sections and sec not in sections:
            continue

        start = float(bar["start_sec"])
        end = float(bar["end_sec"])
        dur = end - start

        # Target note count (density-based)
        n_notes = max(1, int(density * 2))
        span = random.uniform(*motif_span_beats)
        step = dur / (n_notes + 1)

        t = start + step * 0.5
        for i in range(n_notes):
            # Simple stress avoidance (placeholder - can be enhanced)
            if avoid_stress and anchors:
                # Future: check anchors['by_time'] for stress points
                pass

            # Pick random scale pitch and transpose to register
            p = random.choice(scale_pitches)
            while p < lo:
                p += 12
            while p > hi:
                p -= 12

            events.append({"time": t, "duration": min(0.45, step * 0.9), "note": p, "vel": 80})
            t += step

    return events


# ============================================================
# Phase 3: Humanize Engine (Micro-Timing & Velocity Curves)
# ============================================================

from dataclasses import dataclass, field


@dataclass
class HumanizeConfig:
    """
    Configuration for Phase 3 Humanize engine.

    Attributes:
        timing_std_ms: Random timing jitter (±ms)
        swing_8th: 8th note swing ratio (0.12 = 12% delay for offbeat 8ths)
        max_shift_ms: Safety limit for timing shifts (prevent extreme drift)
        velocity_jitter: Random velocity variation (±)
        accent_pattern_4: 4-beat accent multipliers [beat0, beat1, beat2, beat3]
        energy_scale: Section-wide energy multiplier
        legato_chance: Probability of extending note duration
        duration_scale_mean: Base duration multiplier (0.9 = slightly shorter)
        duration_scale_jitter: Random duration variation (±)
        staccato_prob: Probability of shortening note to staccato
        phrase_end_extend: Duration multiplier for phrase-end notes
        seed: Random seed for reproducibility (None = fully random)
    """

    # Timing
    timing_std_ms: float = 0.0
    swing_8th: float = 0.0
    max_shift_ms: float = 20.0  # Safety limit

    # Velocity
    velocity_jitter: float = 0.0
    accent_pattern_4: List[float] = field(default_factory=lambda: [1.0, 1.0, 1.0, 1.0])
    energy_scale: float = 1.0

    # Duration (Phase 3.5: Note Length Humanize)
    duration_scale_mean: float = 1.0
    duration_scale_jitter: float = 0.0
    staccato_prob: float = 0.0
    phrase_end_extend: float = 1.0

    # Misc
    legato_chance: float = 0.0
    seed: int | None = None  # Reproducibility


def load_humanize_config(
    policy: Dict[str, Any], section_name: str, instrument: str = "", song_id: str | None = None
) -> HumanizeConfig:
    """
    Load humanize configuration from policy with section override.

    Priority:
      1. humanize.sections.<section>.*
      2. humanize.global.*
      3. HumanizeConfig defaults

    Args:
        policy: Policy dictionary
        section_name: Section label (verse, chorus, etc.)
        instrument: Instrument name (optional, for future per-instrument override)
        song_id: Song ID for reproducible seed generation

    Returns:
        HumanizeConfig instance
    """
    h_cfg = policy.get("humanize", {})
    h_global = h_cfg.get("global", {})
    h_section = h_cfg.get("sections", {}).get(section_name, {})

    # Merge: section > global > defaults
    def get_val(key, default):
        return h_section.get(key, h_global.get(key, default))

    # Generate reproducible seed from song_id + instrument + section
    seed = None
    if song_id is not None:
        profile = h_cfg.get("profile", "pop_easy")
        seed_source = f"{song_id}:{instrument}:{section_name}:{profile}"
        seed = hash(seed_source) & 0x7FFFFFFF  # Positive 32-bit int

    return HumanizeConfig(
        # Timing
        timing_std_ms=float(get_val("timing_std_ms", 0.0)),
        swing_8th=float(get_val("swing_8th", 0.0)),
        max_shift_ms=float(get_val("max_shift_ms", 20.0)),
        # Velocity
        velocity_jitter=float(get_val("velocity_jitter", 0.0)),
        accent_pattern_4=list(get_val("accent_pattern_4", [1.0, 1.0, 1.0, 1.0])),
        energy_scale=float(get_val("energy_scale", 1.0)),
        # Duration (Phase 3.5)
        duration_scale_mean=float(get_val("duration_scale_mean", 1.0)),
        duration_scale_jitter=float(get_val("duration_scale_jitter", 0.0)),
        staccato_prob=float(get_val("staccato_prob", 0.0)),
        phrase_end_extend=float(get_val("phrase_end_extend", 1.0)),
        # Misc
        legato_chance=float(get_val("legato_chance", 0.0)),
        seed=seed,
    )


def apply_timing_humanize(
    events: List[Dict[str, Any]],
    humanize_cfg: HumanizeConfig,
    tempo_bpm: float = 120.0,
) -> List[Dict[str, Any]]:
    """
    Apply micro-timing humanization (jitter + swing).

    Phase 3 Enhancement:
        - Random timing jitter (±timing_std_ms)
        - 8th note swing (swing_8th)
        - Works on events with time_ql and optional beat_in_bar

    Phase 3.5 Safety:
        - Respects max_shift_ms safety limit
        - Uses reproducible seed (if cfg.seed is set)
        - Clamps timing to bar boundaries (if bar_start/end available)

    Args:
        events: List of event dicts (will be modified in-place)
        humanize_cfg: HumanizeConfig instance
        tempo_bpm: Tempo in BPM for ms->ql conversion

    Returns:
        Modified events list
    """
    if not humanize_cfg:
        return events

    if humanize_cfg.timing_std_ms <= 0 and humanize_cfg.swing_8th <= 0:
        return events

    # Reproducible random generator (or fully random if seed=None)
    rng = random.Random(humanize_cfg.seed)

    # Convert ms to quarter-length based on tempo
    sec_per_beat = 60.0 / tempo_bpm
    ms_per_ql = sec_per_beat * 1000.0
    max_shift_ql = humanize_cfg.max_shift_ms / ms_per_ql

    for ev in events:
        time_ql = ev.get("time_ql")
        if time_ql is None:
            continue

        # Random timing jitter
        jitter_ql = 0.0
        if humanize_cfg.timing_std_ms > 0:
            jitter_ms = rng.gauss(0.0, humanize_cfg.timing_std_ms)
            jitter_ql = jitter_ms / ms_per_ql

        # 8th note swing (offbeat 8th notes are delayed)
        swing_ql = 0.0
        if humanize_cfg.swing_8th > 0:
            beat_in_bar = ev.get("beat_in_bar")
            if beat_in_bar is not None:
                # Check if this is an offbeat 8th (0.5, 1.5, 2.5, 3.5)
                frac = beat_in_bar - int(beat_in_bar)
                if abs(frac - 0.5) < 0.01:  # Offbeat 8th
                    swing_ql = humanize_cfg.swing_8th * 0.5  # Delay by swing% of 8th note

        # Total shift (clamped to safety limit)
        total_shift = jitter_ql + swing_ql
        total_shift = max(-max_shift_ql, min(max_shift_ql, total_shift))

        # Apply shift
        new_time_ql = time_ql + total_shift

        # Clamp to bar boundaries (if available)
        bar_start_ql = ev.get("bar_start_ql")
        bar_end_ql = ev.get("bar_end_ql")
        duration_ql = ev.get("duration_ql", 0.5)

        if bar_start_ql is not None:
            new_time_ql = max(bar_start_ql, new_time_ql)
        if bar_end_ql is not None:
            # Ensure note doesn't start too late (leave space for min duration)
            max_start = bar_end_ql - max(0.05, duration_ql * 0.5)
            new_time_ql = min(new_time_ql, max_start)

        ev["time_ql"] = max(0.0, new_time_ql)

    return events


def apply_velocity_humanize(
    events: List[Dict[str, Any]],
    humanize_cfg: HumanizeConfig,
    base_velocity: int = 80,
) -> List[Dict[str, Any]]:
    """
    Apply velocity humanization (accent pattern + energy + jitter).

    Phase 3 Enhancement:
        - Accent pattern (4-beat cycle)
        - Section energy scale
        - Random velocity jitter

    Phase 3.5 Safety:
        - Uses reproducible seed (if cfg.seed is set)
        - Guarantees velocity in [1, 127] range
        - Always returns int values

    Args:
        events: List of event dicts (will be modified in-place)
        humanize_cfg: HumanizeConfig instance
        base_velocity: Fallback velocity if not specified in event

    Returns:
        Modified events list
    """
    if not humanize_cfg:
        return events

    # Reproducible random generator (or fully random if seed=None)
    rng = random.Random(humanize_cfg.seed)

    pattern = humanize_cfg.accent_pattern_4
    n = len(pattern)

    for ev in events:
        v = ev.get("velocity", base_velocity)
        beat_in_bar = ev.get("beat_in_bar", 0.0)
        beat_index = int(beat_in_bar)  # 0, 1, 2, 3...

        # Apply accent pattern
        accent = pattern[beat_index % n]
        v = v * accent * humanize_cfg.energy_scale

        # Apply velocity jitter
        if humanize_cfg.velocity_jitter > 0:
            jitter = rng.gauss(0.0, humanize_cfg.velocity_jitter)
            v += jitter

        # Clamp to MIDI range [1, 127] and ensure int
        v = max(1, min(127, int(round(v))))
        ev["velocity"] = v

    return events


def apply_duration_humanize(
    events: List[Dict[str, Any]],
    humanize_cfg: HumanizeConfig,
    bar_duration_ql: float = 4.0,
) -> List[Dict[str, Any]]:
    """
    Apply duration humanization (note length variation).

    Phase 3.5 Enhancement:
        - Random duration scaling (duration_scale_mean ± duration_scale_jitter)
        - Staccato probability (shortens notes to 0.4-0.6x)
        - Phrase-end extension (extends last note of phrase)

    Strategy:
        - Normal notes: base_duration * N(duration_scale_mean, duration_scale_jitter)
        - Staccato: base_duration * uniform(0.4, 0.6)
        - Phrase-end: base_duration * phrase_end_extend
        - Minimum duration: max(0.05, original * 0.2)

    Args:
        events: List of event dicts (will be modified in-place)
        humanize_cfg: HumanizeConfig instance
        bar_duration_ql: Bar duration in quarter-lengths (default 4.0 for 4/4)

    Returns:
        Modified events list
    """
    if not humanize_cfg:
        return events

    # Skip if no duration humanization is configured
    if (
        abs(humanize_cfg.duration_scale_mean - 1.0) < 1e-6
        and humanize_cfg.duration_scale_jitter <= 0
        and humanize_cfg.staccato_prob <= 0
        and abs(humanize_cfg.phrase_end_extend - 1.0) < 1e-6
    ):
        return events

    # Reproducible random generator (or fully random if seed=None)
    rng = random.Random(humanize_cfg.seed)

    # Sort events by time to detect phrase endings
    events_sorted = sorted(events, key=lambda e: e.get("time_ql", 0.0))

    for i, ev in enumerate(events_sorted):
        duration_ql = ev.get("duration_ql")
        if duration_ql is None or duration_ql <= 0:
            continue

        base_duration = duration_ql

        # Detect phrase-end (last note in bar or large gap to next note)
        is_phrase_end = False
        if i == len(events_sorted) - 1:
            # Last note overall
            is_phrase_end = True
        else:
            next_time = events_sorted[i + 1].get("time_ql", 0.0)
            curr_time = ev.get("time_ql", 0.0)
            gap = next_time - (curr_time + duration_ql)
            if gap > 1.0:  # 1 quarter-note gap = phrase boundary
                is_phrase_end = True

        # Apply transformations
        if is_phrase_end and humanize_cfg.phrase_end_extend > 1.0:
            # Phrase-end: extend duration
            new_duration = base_duration * humanize_cfg.phrase_end_extend
        elif humanize_cfg.staccato_prob > 0 and rng.random() < humanize_cfg.staccato_prob:
            # Staccato: shorten to 40-60% of original
            staccato_factor = rng.uniform(0.4, 0.6)
            new_duration = base_duration * staccato_factor
        else:
            # Normal: apply scale + jitter
            scale = humanize_cfg.duration_scale_mean
            if humanize_cfg.duration_scale_jitter > 0:
                jitter = rng.gauss(0.0, humanize_cfg.duration_scale_jitter)
                scale += jitter
            new_duration = base_duration * scale

        # Safety: minimum duration (avoid too-short notes)
        min_duration = max(0.05, base_duration * 0.2)
        new_duration = max(min_duration, new_duration)

        # Safety: don't exceed bar boundary (if bar_end_ql available)
        bar_end_ql = ev.get("bar_end_ql")
        if bar_end_ql is not None:
            time_ql = ev.get("time_ql", 0.0)
            max_duration = bar_end_ql - time_ql
            new_duration = min(new_duration, max_duration)

        ev["duration_ql"] = max(0.05, new_duration)

    return events


# ============================================================
# Phase 3.6: Long Note Extension (Sustain Layer)
# ============================================================


def extend_last_note_per_bar(
    events: List[Dict[str, Any]],
    bars_df,
    role: str,
    min_duration_ql: float = 2.0,
    max_bar_span: int = 1,
) -> List[Dict[str, Any]]:
    """
    各barについて「一番最後に鳴るノート」を見つけ、bar終端まで延長する。

    Phase 3.6 Enhancement:
        - Strings: 2-3小節分の長音パッド (max_bar_span=3)
        - Piano: bar内フル延長 (max_bar_span=1)
        - Bass: 2拍以上のルート土台 (min_duration_ql=1.5)

    Strategy:
        - 各barの最後のノートを anchor とする
        - anchor.duration_ql を bar_end_ql (または複数bar先) まで延長
        - すでに min_duration_ql 以上なら触らない

    Args:
        events: V2が生成したevents (time_ql / duration_ql / note を含む)
        bars_df: bars_with_slots.parquet DataFrame (bar_index / start_ql / end_ql)
        role: "strings" / "piano" / "bass" (metadata用)
        min_duration_ql: 延長対象の最小duration (これより短ければ延長)
        max_bar_span: 何小節先まで伸ばすか (1=bar内、3=3小節先まで)

    Returns:
        Modified events list
    """
    if not events:
        return events

    # bars_df を bar_index -> (start_ql, end_ql) に変換
    # NOTE: 4/4前提で start_ql = bar_idx * 4.0 のfallback
    bars_data = []
    for _, row in bars_df.iterrows():
        bar_idx = int(row.get("bar_index", row.get("bar_idx", 0)))
        start_ql = float(row.get("start_ql", bar_idx * 4.0))
        end_ql = float(row.get("end_ql", (bar_idx + 1) * 4.0))
        bars_data.append((bar_idx, start_ql, end_ql))

    if not bars_data:
        return events

    bar_lookup = {b[0]: (b[1], b[2]) for b in bars_data}
    bar_indices_sorted = sorted([b[0] for b in bars_data])

    # events を time_ql でソート
    events_sorted = sorted(events, key=lambda e: float(e.get("time_ql", 0.0)))

    # bar ごとに events を分類
    events_by_bar: Dict[int, List[Dict[str, Any]]] = {idx: [] for idx in bar_indices_sorted}
    for ev in events_sorted:
        t = float(ev.get("time_ql", 0.0))
        # 所属 bar を探索
        for bar_idx, (start_ql, end_ql) in bar_lookup.items():
            if start_ql <= t < end_ql:
                events_by_bar[bar_idx].append(ev)
                break

    # 各 bar ごとに最後のノートを延長
    for i, bar_idx in enumerate(bar_indices_sorted):
        evs = events_by_bar.get(bar_idx, [])
        if not evs:
            continue

        # 一番後ろの note を anchor とする
        anchor = max(evs, key=lambda e: float(e.get("time_ql", 0.0)))

        start_ql = float(anchor.get("time_ql", 0.0))
        current_dur = float(anchor.get("duration_ql", 0.0))

        # 今いる bar の終端
        _, bar_end_ql = bar_lookup[bar_idx]
        target_end_ql = bar_end_ql

        # max_bar_span > 1 のとき、複数 bar 分まで伸ばす
        if max_bar_span > 1:
            max_idx = min(i + max_bar_span - 1, len(bar_indices_sorted) - 1)
            for j in range(i + 1, max_idx + 1):
                next_bar_idx = bar_indices_sorted[j]
                _, next_end = bar_lookup[next_bar_idx]
                target_end_ql = max(target_end_ql, float(next_end))

        new_dur = target_end_ql - start_ql

        # すでに min_duration_ql 以上なら触らない
        if current_dur >= min_duration_ql:
            continue

        anchor["duration_ql"] = max(current_dur, new_dur)

        # Metadata for debugging
        meta = anchor.setdefault("meta", {})
        meta["longified_by"] = role
        meta["original_duration_ql"] = current_dur

    return events
