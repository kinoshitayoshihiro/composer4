#!/usr/bin/env python3
"""
v2_guide_tone.py - Guide Tone Generator for V2 Pipeline

Purpose:
    Generate melodic guide tone lines (3rd/7th/9th based) for V2 generators.
    Creates smooth voice-leading melodies that bring "musical life" to arrangements.

Strategy:
    - Focus on chord guide tones (3rd, 7th, 9th)
    - Smooth voice leading (minimal motion from prev note)
    - Density control (notes_per_bar)
    - Register-aware pitch selection
    - Section-based targeting

Usage:
    from v2_guide_tone import (
        GuideToneConfig,
        load_guide_tone_config,
        generate_guide_tone_events,
    )
"""

from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Tuple, Sequence
import random
import math
import json


@dataclass
class GuideToneConfig:
    """
    Configuration for guide tone generation.

    Attributes:
        enabled: Whether guide tones are enabled
        mode: Generation mode ("3rd7th", "3rd7th9th", "scale")
        notes_per_bar: Average notes per bar (1.0 = 1 note, 1.5 = 1-2 notes)
        prefer_tension: Probability of using 9th/11th/13th (0.0-1.0)
        max_step: Maximum semitone jump from previous note
        voice_lead_weight: Preference for smooth voice leading (0.0-1.0)
        register_low: Minimum MIDI pitch
        register_high: Maximum MIDI pitch
        target_sections: Section labels to apply guide tones
        velocity_base: Base velocity for guide tone notes
        duration_scale: Duration multiplier (1.0 = normal, 1.5 = longer)
        seed: Random seed for reproducibility
    """

    enabled: bool = False
    mode: str = "3rd7th9th"
    notes_per_bar: float = 1.0
    prefer_tension: float = 0.4
    max_step: int = 5
    voice_lead_weight: float = 0.8
    register_low: int = 60
    register_high: int = 79
    target_sections: Tuple[str, ...] = ("verse", "pre_chorus", "chorus", "bridge")
    velocity_base: int = 85
    duration_scale: float = 0.8
    seed: Optional[int] = None


def load_guide_tone_config(
    policy: Dict[str, Any],
    instrument_name: str,
    default_low: int = 60,
    default_high: int = 79,
) -> GuideToneConfig:
    """
    Load guide tone configuration from policy YAML.

    Args:
        policy: Policy dictionary
        instrument_name: Instrument name (strings, piano, etc.)
        default_low: Default register minimum
        default_high: Default register maximum

    Returns:
        GuideToneConfig instance
    """
    inst_cfg = policy.get("instruments", {}).get(instrument_name, {})
    gt_cfg = inst_cfg.get("guide_tones", {})

    if not gt_cfg.get("enabled", False):
        return GuideToneConfig(enabled=False)

    # Get register from instrument config
    register = inst_cfg.get("register", {})
    low = int(register.get("low", default_low))
    high = int(register.get("high", default_high))

    # Generate reproducible seed from song_id + instrument
    seed = None
    song_id = policy.get("metadata", {}).get("song_id")
    if song_id:
        seed_str = f"{song_id}:{instrument_name}:guide_tone"
        seed = hash(seed_str) & 0x7FFFFFFF

    return GuideToneConfig(
        enabled=True,
        mode=str(gt_cfg.get("mode", "3rd7th9th")),
        notes_per_bar=float(gt_cfg.get("notes_per_bar", 1.0)),
        prefer_tension=float(gt_cfg.get("prefer_tension", 0.4)),
        max_step=int(gt_cfg.get("max_step", 5)),
        voice_lead_weight=float(gt_cfg.get("voice_lead_weight", 0.8)),
        register_low=low,
        register_high=high,
        target_sections=tuple(
            gt_cfg.get("target_sections", ["verse", "pre_chorus", "chorus", "bridge"])
        ),
        velocity_base=int(gt_cfg.get("velocity_base", 85)),
        duration_scale=float(gt_cfg.get("duration_scale", 0.8)),
        seed=seed,
    )


def _parse_chord_symbol(symbol: str) -> Dict[str, Any]:
    """
    Simple chord symbol parser for guide tone extraction.

    Args:
        symbol: Chord symbol (e.g., "C#m7", "Gmaj7", "Aadd9")

    Returns:
        Dict with root_pc (pitch class 0-11), quality, extensions
    """
    import re

    # Note to pitch class mapping
    NOTE_PC = {
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

    # Extract root
    root_match = re.match(r"^([A-G](?:#|b)?)", symbol)
    if not root_match:
        return {"root_pc": 0, "quality": "", "extensions": []}

    root = root_match.group(1)
    root_pc = NOTE_PC.get(root, 0)

    # Detect quality
    sym_lower = symbol.lower()
    if "m7" in sym_lower and "maj7" not in sym_lower:
        quality = "m7"
    elif "maj7" in sym_lower or "m7" in sym_lower:
        quality = "maj7"
    elif sym_lower.endswith("m") or "min" in sym_lower:
        quality = "m"
    elif "7" in sym_lower:
        quality = "7"
    elif "dim" in sym_lower:
        quality = "dim"
    elif "aug" in sym_lower:
        quality = "aug"
    elif "sus4" in sym_lower:
        quality = "sus4"
    elif "sus2" in sym_lower:
        quality = "sus2"
    else:
        quality = ""  # Major triad

    # Detect extensions
    extensions = []
    if "9" in symbol or "(9)" in symbol:
        extensions.append("9")
    if "11" in symbol or "(11)" in symbol:
        extensions.append("11")
    if "13" in symbol or "(13)" in symbol:
        extensions.append("13")
    if "add9" in sym_lower:
        extensions.append("9")

    return {
        "root_pc": root_pc,
        "quality": quality,
        "extensions": extensions,
    }


def _get_guide_tone_intervals(chord_info: Dict[str, Any], cfg: GuideToneConfig) -> List[int]:
    """
    Get guide tone intervals (semitones from root) based on chord quality.

    Args:
        chord_info: Parsed chord info from _parse_chord_symbol()
        cfg: GuideToneConfig

    Returns:
        List of semitone intervals (e.g., [3, 10] for m7 chord)
    """
    quality = chord_info.get("quality", "")
    extensions = chord_info.get("extensions", [])

    intervals = []

    # 3rd (major or minor)
    if quality in ("m", "m7", "dim"):
        intervals.append(3)  # minor 3rd
    else:
        intervals.append(4)  # major 3rd

    # 7th (if applicable)
    if quality == "m7":
        intervals.append(10)  # minor 7th
    elif quality == "maj7":
        intervals.append(11)  # major 7th
    elif quality == "7":
        intervals.append(10)  # dominant 7th
    elif quality == "dim":
        intervals.append(9)  # diminished 7th

    # 9th/11th/13th (tension extensions)
    if cfg.prefer_tension > 0 and random.random() < cfg.prefer_tension:
        if "9" in extensions:
            intervals.append(14)  # 9th
        elif "11" in extensions:
            intervals.append(17)  # 11th
        elif "13" in extensions:
            intervals.append(21)  # 13th

    # Special cases
    if quality == "sus4":
        intervals = [5, 7]  # 4th and 5th (no 3rd)
    elif quality == "sus2":
        intervals = [2, 7]  # 2nd and 5th

    return sorted(set(intervals))


def _choose_guide_pitch(
    prev_pitch: Optional[int],
    chord_info: Dict[str, Any],
    cfg: GuideToneConfig,
    rng: random.Random,
) -> Optional[int]:
    """
    Choose optimal guide tone pitch with voice leading.

    Strategy:
        - Get candidate pitch classes from chord
        - Transpose to register range
        - Prefer pitch closest to prev_pitch (voice leading)
        - Penalize large jumps (> max_step)

    Args:
        prev_pitch: Previous guide tone MIDI pitch (or None)
        chord_info: Parsed chord info
        cfg: GuideToneConfig
        rng: Random generator

    Returns:
        Selected MIDI pitch (or None if no valid candidate)
    """
    root_pc = chord_info["root_pc"]
    intervals = _get_guide_tone_intervals(chord_info, cfg)

    if not intervals:
        return None

    # Generate pitch class candidates
    candidate_pcs = [(root_pc + iv) % 12 for iv in intervals]

    # Expand to full MIDI pitches across register
    candidates = []
    for octave in range(3, 8):  # C3-B7 range
        base = octave * 12
        for pc in candidate_pcs:
            pitch = base + pc
            if cfg.register_low <= pitch <= cfg.register_high:
                candidates.append(pitch)

    if not candidates:
        return None

    # Choose pitch with voice leading optimization
    if prev_pitch is None:
        # First note: prefer middle of register
        mid = (cfg.register_low + cfg.register_high) / 2
        return min(candidates, key=lambda p: abs(p - mid))

    # Voice leading: minimize distance to prev_pitch
    best_pitch = None
    best_cost = float("inf")

    for pitch in candidates:
        step = abs(pitch - prev_pitch)

        # Cost = weighted sum of:
        #   - Distance from prev pitch (voice leading)
        #   - Penalty for large jumps
        voice_lead_cost = step * cfg.voice_lead_weight
        jump_penalty = 0
        if step > cfg.max_step:
            jump_penalty = (step - cfg.max_step) * 3.0

        total_cost = voice_lead_cost + jump_penalty

        if total_cost < best_cost:
            best_cost = total_cost
            best_pitch = pitch

    return best_pitch


def generate_guide_tone_events(
    bars_df,
    chordmap_events: Sequence[Dict[str, Any]],
    sections: Dict[int, str],
    cfg: GuideToneConfig,
    unit: str = "bar",
) -> List[Dict[str, Any]]:
    """
    Generate guide tone event list for V2 pipeline.

    Args:
        bars_df: DataFrame with bar timing (bar_index, start_ql, length_ql)
        chordmap_events: Chordmap events (bar, symbol)
        sections: {bar_index -> section_label} mapping
        cfg: GuideToneConfig
        unit: Chordmap unit ("bar" or "ql")

    Returns:
        List of event dicts with time_ql, duration_ql, note, velocity, metadata
    """
    if not cfg.enabled:
        return []

    # Initialize random generator
    rng = random.Random(cfg.seed)

    # Build bar -> symbol mapping
    symbol_by_bar = {}
    for ev in chordmap_events:
        bar = ev.get("bar")
        symbol = ev.get("symbol")
        if bar is not None and symbol:
            symbol_by_bar[bar] = symbol

    events = []
    prev_pitch: Optional[int] = None

    for _, row in bars_df.iterrows():
        bar = int(row.get("bar_index", row.get("bar_idx", 0)))
        section_label = sections.get(bar, "")

        # Skip if section not targeted
        if cfg.target_sections and section_label not in cfg.target_sections:
            continue

        # Get chord symbol
        symbol = symbol_by_bar.get(bar)
        if not symbol:
            continue

        # Parse chord
        chord_info = _parse_chord_symbol(symbol)

        # Determine note count for this bar
        base = cfg.notes_per_bar
        n_notes = 1 if rng.random() > (base - math.floor(base)) else math.ceil(base)

        # Bar timing
        bar_start = float(row.get("start_ql", bar * 4.0))
        bar_len = float(row.get("length_ql", 4.0))

        # Distribute notes across bar
        step = bar_len / (n_notes + 1)

        for i in range(n_notes):
            # Time position (avoid exact downbeat for breath)
            t = bar_start + step * (i + 1)

            # Choose pitch with voice leading
            pitch = _choose_guide_pitch(prev_pitch, chord_info, cfg, rng)
            if pitch is None:
                continue

            # Duration (scaled by config)
            base_dur = step * 0.7  # 70% of available space
            duration = base_dur * cfg.duration_scale

            # Create event
            ev = {
                "time_ql": t,
                "duration_ql": duration,
                "note": pitch,
                "velocity": cfg.velocity_base,
                "bar": bar,
                "section": section_label,
                "type": "guide_tone",
                "is_guide_tone": True,
                "chord_symbol": symbol,
            }
            events.append(ev)
            prev_pitch = pitch

    return events


def save_guide_tone_report(
    events: List[Dict[str, Any]],
    output_path: str,
) -> None:
    """
    Save guide tone analysis report (for debugging).

    Args:
        events: Guide tone events
        output_path: Output JSON path
    """
    if not events:
        report = {"total_events": 0, "message": "No guide tones generated"}
    else:
        pitches = [e["note"] for e in events]
        intervals = []
        for i in range(1, len(pitches)):
            intervals.append(abs(pitches[i] - pitches[i - 1]))

        report = {
            "total_events": len(events),
            "pitch_range": {
                "min": min(pitches),
                "max": max(pitches),
                "span": max(pitches) - min(pitches),
            },
            "intervals": {
                "mean": sum(intervals) / len(intervals) if intervals else 0,
                "max": max(intervals) if intervals else 0,
                "large_jumps": sum(1 for iv in intervals if iv > 7),
            },
            "sample_events": events[:5],
        }

    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)


__all__ = [
    "GuideToneConfig",
    "load_guide_tone_config",
    "generate_guide_tone_events",
    "save_guide_tone_report",
]
