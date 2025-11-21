#!/usr/bin/env python3
# scripts/generate_guidetone_and_emotion_from_rulebook.py

"""
Generate GuideTone & Emotion hints from configs/otobonAI/rulebook.yaml.

This script:
1. Loads rulebook.yaml (or .json)
2. Analyzes bars, chordmap, sections
3. Builds per-bar context (section, chord, function, etc.)
4. Matches rules to each bar
5. Outputs:
   - analysis/guide_tone_hints.json (for GuideToneAI)
   - analysis/emotion_profile.json (for EmotionAI)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from otobonAI.rulebook_engine import Rulebook


# ------------- Utility Functions -------------

NOTE_NAMES_SHARP = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def _parse_key_center(key_center: str) -> Tuple[str, bool]:
    """Parse key center string like 'C#m' → ('C#', True)."""
    key_center = key_center.strip()
    is_minor = key_center.endswith("m")
    if is_minor:
        root = key_center[:-1]
    else:
        root = key_center
    return root, is_minor


def _note_to_degree(root: str, key_root: str) -> int:
    """Convert note to scale degree (1-7) relative to key."""
    try:
        idx_note = NOTE_NAMES_SHARP.index(root)
        idx_key = NOTE_NAMES_SHARP.index(key_root)
    except ValueError:
        return 1
    
    diff = (idx_note - idx_key) % 12
    # Map chromatic intervals to scale degrees
    mapping = {
        0: 1,   # Tonic
        1: 1,   # ♭2 (treat as 1)
        2: 2,   # 2
        3: 3,   # ♭3
        4: 3,   # 3
        5: 4,   # 4
        6: 4,   # ♯4/♭5
        7: 5,   # 5
        8: 6,   # ♭6
        9: 6,   # 6
        10: 7,  # ♭7
        11: 7,  # 7
    }
    return mapping.get(diff, 1)


def _infer_function(degree: int, is_minor: bool = False) -> str:
    """Infer harmonic function from scale degree."""
    if is_minor:
        # Minor key functions
        if degree in (1, 3, 6):
            return "tonic"
        if degree in (2, 4):
            return "subdominant"
        if degree in (5, 7):
            return "dominant"
    else:
        # Major key functions
        if degree in (1, 3, 6):
            return "tonic"
        if degree in (2, 4):
            return "subdominant"
        if degree in (5, 7):
            return "dominant"
    return "unknown"


def _chord_root_from_symbol(symbol: str) -> str:
    """Extract root note from chord symbol like 'C#m7' → 'C#'."""
    symbol = symbol.strip()
    if not symbol:
        return "C"
    if len(symbol) >= 2 and symbol[1] in ("#", "b"):
        return symbol[:2]
    return symbol[0]


def _section_for_bar(sections: List[Dict], bar: int) -> str:
    """Find section name for given bar."""
    for s in sections:
        if s["start_bar"] <= bar <= s["end_bar"]:
            # Support both 'name' and 'label' keys
            return s.get("name") or s.get("label", "unknown")
    return "unknown"


def _load_lyric_anchors(path: Path, bar_duration: float = 2.0) -> Dict[int, Dict]:
    """
    Load lyric_anchors.json and aggregate by bar.
    
    Args:
        path: Path to lyric_anchors.json
        bar_duration: Assumed duration per bar in seconds (default: 2.0)
    
    Returns:
        {
            bar_idx: {
                "stress_count": int,
                "stress_level": float (0-1),
                "has_stress": bool,
                "phrase_boundaries": ["end", ...],
                "classes": ["stress", "sibilant", ...],
                "vowel_rich": bool
            }
        }
    """
    if not path.exists():
        return {}
    
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    
    bar_anchors = {}
    for anchor in data.get("anchors", []):
        time = anchor.get("time", 0.0)
        bar = int(time / bar_duration)
        
        if bar not in bar_anchors:
            bar_anchors[bar] = {
                "stress_count": 0,
                "has_stress": False,
                "phrase_boundaries": [],
                "classes": [],
                "vowel_rich": False
            }
        
        classes = anchor.get("classes", [])
        bar_anchors[bar]["classes"].extend(classes)
        
        if "stress" in classes:
            bar_anchors[bar]["stress_count"] += 1
            bar_anchors[bar]["has_stress"] = True
        
        if "phrase_end" in classes or "boundary" in classes:
            bar_anchors[bar]["phrase_boundaries"].append("end")
        
        # Vowel rich判定
        if any(c in classes for c in ["vowel", "sustained", "long"]):
            bar_anchors[bar]["vowel_rich"] = True
    
    # stress_level 計算 (0-1)
    for bar, info in bar_anchors.items():
        info["stress_level"] = min(1.0, info["stress_count"] * 0.3)
    
    return bar_anchors


def _position_in_section(section_info: Dict[str, Any], bar: int) -> str:
    """Determine position within section: start, middle, end."""
    start = section_info["start_bar"]
    end = section_info["end_bar"]
    length = max(1, end - start + 1)
    pos = (bar - start) / length
    
    if pos <= 0.2:
        return "start"
    if pos >= 0.8:
        return "end"
    return "middle"


def _load_sections(path: Path) -> List[Dict[str, Any]]:
    """Load sections from JSON file."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if "sections" in data:
        return data["sections"]
    if isinstance(data, list):
        return data
    raise ValueError(f"Unexpected sections format in {path}")


def _load_chordmap(path: Path) -> Dict[str, Any]:
    """Load chordmap and index events by bar."""
    data = json.loads(path.read_text(encoding="utf-8"))
    events = data.get("events", [])
    by_bar = {}
    for ev in events:
        bar_num = int(ev.get("bar", ev.get("bar_idx", 0)))
        by_bar[bar_num] = ev
    meta = data.get("meta", {})
    return {"events_by_bar": by_bar, "meta": meta}


def _infer_base_emotion_from_key(key_center: str) -> Dict[str, float]:
    """Infer base emotion values from key (major vs minor)."""
    root, is_minor = _parse_key_center(key_center)
    
    if is_minor:
        return {
            "energy": 0.45,
            "tension": 0.55,
            "brightness": 0.4,
            "valence": 0.35,
        }
    else:
        return {
            "energy": 0.6,
            "tension": 0.5,
            "brightness": 0.7,
            "valence": 0.7,
        }


def _safe_clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    """Clamp value to [lo, hi] range."""
    return max(lo, min(hi, x))


# ------------- Context Building -------------


def build_song_context_per_bar(
    bars_df: pd.DataFrame,
    chordmap_info: Dict[str, Any],
    sections: List[Dict[str, Any]],
    song_emotion_tags: List[str],
    tempo_bpm: float = 120.0,
) -> List[Dict[str, Any]]:
    """
    Build rulebook context for each bar.
    
    Returns list of dicts with keys:
    - bar: int
    - section: str
    - position_in_section: str
    - chord_symbol: str
    - chord_root: str
    - scale_degree: int
    - function: str
    - song_emotion_tags: List[str]
    - key_center: str
    - tempo_bpm: float
    """
    events_by_bar = chordmap_info["events_by_bar"]
    key_center = chordmap_info.get("meta", {}).get("key_center", "C")
    key_root, is_minor = _parse_key_center(key_center)

    # Index sections by name/label
    section_index = {}
    for s in sections:
        # Support both 'name' and 'label' keys
        key = s.get("name") or s.get("label", "unknown")
        section_index[key] = s

    contexts = []
    
    # Detect bar column name
    bar_col = "bar"
    if "bar" not in bars_df.columns and "bar_index" in bars_df.columns:
        bar_col = "bar_index"
    elif "bar" not in bars_df.columns and "bar_idx" in bars_df.columns:
        bar_col = "bar_idx"

    for _, row in bars_df.sort_values(bar_col).iterrows():
        bar = int(row[bar_col])
        
        # Get section
        section_name = row.get("section") or row.get("section_label")
        if not section_name:
            section_name = _section_for_bar(sections, bar)
        
        section_info = section_index.get(section_name)
        if section_info is None:
            section_info = {"name": section_name, "start_bar": bar, "end_bar": bar}

        position = _position_in_section(section_info, bar)

        # Get chord
        chord_ev = events_by_bar.get(bar, {})
        symbol = chord_ev.get("symbol", "C")
        
        chord_root = _chord_root_from_symbol(symbol)
        degree = _note_to_degree(chord_root, key_root)
        func = _infer_function(degree, is_minor)

        ctx = {
            "bar": bar,
            "section": section_name,
            "position_in_section": position,
            "chord_symbol": symbol,
            "chord_root": chord_root,
            "scale_degree": degree,
            "function": func,
            "song_emotion_tags": song_emotion_tags,
            "key_center": key_center,
            "tempo_bpm": tempo_bpm,
        }
        contexts.append(ctx)
    
    return contexts


# ------------- Emotion Profile Generation -------------


def generate_emotion_profile(
    rulebook: Rulebook,
    contexts: List[Dict[str, Any]],
    key_center: str,
    lyric_anchors: Optional[Dict[int, Dict]] = None,
) -> Dict[str, Any]:
    """
    Generate EmotionAI profile from rulebook.
    
    Args:
        rulebook: Loaded rulebook
        contexts: Bar contexts
        key_center: Key center string (e.g., "C#m")
        lyric_anchors: Bar-indexed anchor info (v1.5, optional)
    
    Returns dict with:
    - unit: "bar"
    - meta: {key_center, base}
    - events: [{bar, energy, tension, brightness, valence, rule_ids, tags,
                anchor_weight, has_lyric_stress, phrase_position, vocal_focus}]
    """
    if lyric_anchors is None:
        lyric_anchors = {}
    base = _infer_base_emotion_from_key(key_center)
    events = []
    
    for ctx in contexts:
        bar = ctx["bar"]
        energy = base["energy"]
        tension = base["tension"]
        brightness = base["brightness"]
        valence = base["valence"]
        density = 0.5  # Default density
        agg_tags: List[str] = []

        # Match emotion rules
        matched = rulebook.find_matching(ctx, "emotion", "harmony", "melody")
        
        for r in matched:
            emo = r.get_emotion_action()
            if emo is None:
                continue
            
            energy += emo.energy_delta
            tension += emo.tension_delta
            brightness += emo.brightness_delta
            valence += emo.valence_delta
            density += emo.density_delta
            agg_tags.extend(emo.tags_add)
        
        # v1.5: lyric_anchor 補正
        anchor_weight = 0.0
        has_lyric_stress = False
        phrase_position = "mid"
        vocal_focus = False
        
        if bar in lyric_anchors:
            anchor = lyric_anchors[bar]
            anchor_weight = anchor.get("stress_level", 0.0)
            has_lyric_stress = anchor.get("has_stress", False)
            
            # phrase_position 推定
            if "end" in anchor.get("phrase_boundaries", []):
                phrase_position = "end"
            elif anchor.get("stress_count", 0) > 0:
                phrase_position = "begin"
            
            vocal_focus = anchor_weight > 0.5
            
            # Energy/Tension 補正
            if has_lyric_stress:
                energy = min(1.0, energy + 0.1)
                tension = min(1.0, tension + 0.1)
            
            # phrase_end → tension ピーク
            if phrase_position == "end":
                tension = min(1.0, tension + 0.15)
                if "phrase_end" not in agg_tags:
                    agg_tags.append("phrase_end")
            
            # vocal_focus → tags 追加
            if vocal_focus and "vocal_focus" not in agg_tags:
                agg_tags.append("vocal_focus")

        events.append({
            "bar": bar,
            "energy": _safe_clamp(energy),
            "tension": _safe_clamp(tension),
            "brightness": _safe_clamp(brightness),
            "valence": _safe_clamp(valence),
            "density": _safe_clamp(density),
            "rule_ids": [r.id for r in matched],
            "tags": sorted(set(agg_tags)),
            # v1.5 fields
            "anchor_weight": round(anchor_weight, 2),
            "has_lyric_stress": has_lyric_stress,
            "phrase_position": phrase_position,
            "vocal_focus": vocal_focus,
        })

    return {
        "unit": "bar",
        "meta": {
            "key_center": key_center,
            "base": base,
            "description": "Emotion profile generated from rulebook v0.1",
        },
        "events": events,
    }


# ------------- Guide Tone Hints Generation -------------


def _pick_guide_tone_pitch(
    scale_degree: int,
    previous_pitch: Optional[int],
    target_register: str,
    key_root: str,
) -> int:
    """
    Pick MIDI pitch for guide tone based on scale degree and register.
    
    Uses voice leading (minimal motion from previous pitch).
    """
    # Map scale degree to semitone offset from key root
    degree_to_semitone = {
        1: 0,   # Tonic
        2: 2,   # 2nd
        3: 4,   # 3rd
        4: 5,   # 4th
        5: 7,   # 5th
        6: 9,   # 6th
        7: 11,  # 7th
    }
    
    # Get key root pitch class (0-11)
    try:
        root_pc = NOTE_NAMES_SHARP.index(key_root)
    except ValueError:
        root_pc = 0  # Default to C
    
    # Calculate pitch class for this degree
    offset = degree_to_semitone.get(scale_degree, 0)
    target_pc = (root_pc + offset) % 12
    
    # Determine center pitch based on register
    if target_register == "low":
        center_pitch = 55  # G3
    elif target_register == "high":
        center_pitch = 76  # E5
    else:
        center_pitch = 64  # E4
    
    # Generate candidates across octaves
    candidates = []
    for octave in range(2, 7):
        pitch = octave * 12 + target_pc
        if 36 <= pitch <= 84:  # Reasonable range (C2-C6)
            candidates.append(pitch)
    
    if not candidates:
        return center_pitch
    
    # Choose candidate closest to previous pitch (voice leading)
    if previous_pitch is not None:
        diffs = [abs(p - previous_pitch) for p in candidates]
    else:
        # No previous pitch: choose closest to center register
        diffs = [abs(p - center_pitch) for p in candidates]
    
    idx = int(np.argmin(diffs))
    return candidates[idx]


def _guide_tone_tag_to_degree(tag: str) -> Optional[int]:
    """Convert guide tone tag to scale degree."""
    mapping = {
        "root": 1,
        "3rd": 3,
        "5th": 5,
        "7th": 7,
        "9th": 2,   # 9 = 2
        "11th": 4,  # 11 = 4
        "13th": 6,  # 13 = 6
    }
    return mapping.get(tag)


def generate_guide_tone_hints(
    rulebook: Rulebook,
    contexts: List[Dict[str, Any]],
    lyric_anchors: Optional[Dict[int, Dict]] = None,
) -> Dict[str, Any]:
    """
    Generate GuideToneAI hints from rulebook.
    
    Args:
        rulebook: Loaded rulebook
        contexts: Bar contexts
        lyric_anchors: Bar-indexed anchor info (v1.5, optional)
    
    Returns dict with:
    - unit: "bar"
    - meta: {description}
    - events: [{bar, scale_degree, register, approx_pitch, rule_ids, motion, notes_per_bar,
                lyric_anchor_weight, phrase_role, stress_alignment, vowel_rich}]
    """
    if lyric_anchors is None:
        lyric_anchors = {}
    events = []
    prev_pitch: Optional[int] = None
    key_root = contexts[0]["key_center"].rstrip("m") if contexts else "C"

    for ctx in contexts:
        bar = ctx["bar"]
        
        # Match guidetone/harmony/melody rules
        matched = rulebook.find_matching(ctx, "guidetone", "harmony", "melody")

        # Default settings
        priority_degrees = [3, 7]  # 3rd and 7th
        target_register = "mid"
        motion = "step"
        notes_per_bar = 1.2
        rule_ids: List[str] = []

        # Apply matched rules
        for r in matched:
            rule_ids.append(r.id)
            g = r.get_guidetone_action()
            if g is None:
                continue

            # Extract priority degrees from tags
            if g.priority_tones:
                degs = []
                for tag in g.priority_tones:
                    deg = _guide_tone_tag_to_degree(tag)
                    if deg is not None:
                        degs.append(deg)
                if degs:
                    priority_degrees = degs

            if g.default_register:
                target_register = g.default_register
            
            if g.motion:
                motion = g.motion
            
            if g.notes_per_bar is not None:
                notes_per_bar = g.notes_per_bar

        # Use first priority degree
        degree = priority_degrees[0] if priority_degrees else 3
        
        # Pick pitch using voice leading
        use_prev = prev_pitch if motion != "hold" else None
        pitch = _pick_guide_tone_pitch(degree, use_prev, target_register, key_root)
        
        # Update previous pitch for voice leading
        if motion != "hold":
            prev_pitch = pitch
        
        # v1.5: lyric_anchor 補正
        anchor_weight = 0.0
        phrase_role = "mid"
        stress_alignment = False
        vowel_rich = False
        
        if bar in lyric_anchors:
            anchor = lyric_anchors[bar]
            anchor_weight = anchor.get("stress_level", 0.0)
            vowel_rich = anchor.get("vowel_rich", False)
            
            # 強勢音節 → stress_alignment
            if anchor.get("has_stress", False):
                stress_alignment = True
            
            # phrase_role 推定
            boundaries = anchor.get("phrase_boundaries", [])
            if "end" in boundaries:
                phrase_role = "release"
                motion = "leap_to_resolution"
            elif anchor.get("stress_count", 0) > 2:
                phrase_role = "climax"
            elif anchor.get("stress_count", 0) > 0:
                phrase_role = "build"
            
            # 母音豊か → 音数減
            if vowel_rich:
                notes_per_bar *= 0.8

        events.append({
            "bar": bar,
            "scale_degree": degree,
            "register": target_register,
            "approx_pitch": pitch,
            "rule_ids": rule_ids,
            "motion": motion,
            "notes_per_bar": round(notes_per_bar, 1),
            # v1.5 fields
            "lyric_anchor_weight": round(anchor_weight, 2),
            "phrase_role": phrase_role,
            "stress_alignment": stress_alignment,
            "vowel_rich": vowel_rich,
        })

    return {
        "unit": "bar",
        "meta": {
            "description": "Guide-tone hints derived from rulebook v0.1",
        },
        "events": events,
    }


# ------------- CLI -------------


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Generate GuideTone & Emotion hints from configs/otobonAI/rulebook.yaml"
    )
    ap.add_argument(
        "--song-dir",
        type=str,
        default=".",
        help="Song package root (default: current directory)",
    )
    ap.add_argument(
        "--rulebook",
        type=str,
        default="configs/otobonAI/rulebook.yaml",
        help="Path to rulebook YAML/JSON (relative to project root)",
    )
    ap.add_argument(
        "--bars",
        type=str,
        default="analysis/bars_with_slots.parquet",
        help="Path to bars_with_slots.parquet (relative to song-dir)",
    )
    ap.add_argument(
        "--chordmap",
        type=str,
        default="analysis/manual_chordmap.json",
        help="Path to chordmap JSON (relative to song-dir)",
    )
    ap.add_argument(
        "--sections",
        type=str,
        default="analysis/sections.json",
        help="Path to sections.json (relative to song-dir)",
    )
    ap.add_argument(
        "--emotion-tags",
        type=str,
        default="",
        help="Comma-separated song-level emotion tags (e.g. 'sad,hopeful')",
    )
    ap.add_argument(
        "--tempo-map",
        type=str,
        default="analysis/tempo_map.json",
        help="Path to tempo_map.json (relative to song-dir). If not found, uses --tempo-default",
    )
    ap.add_argument(
        "--tempo-default",
        type=float,
        default=120.0,
        help="Default tempo in BPM if tempo-map not found (default: 120)",
    )
    ap.add_argument(
        "--out-guide",
        type=str,
        default="analysis/guide_tone_hints.json",
        help="Output path for guide tone hints (relative to song-dir)",
    )
    ap.add_argument(
        "--out-emotion",
        type=str,
        default="analysis/emotion_profile.json",
        help="Output emotion_profile.json (relative to song-dir)",
    )
    
    ap.add_argument(
        "--lyric-anchors",
        type=str,
        default="analysis/lyric_anchors.json",
        help="Path to lyric_anchors.json (v1.5, optional)",
    )

    args = ap.parse_args()
    
    song_dir = Path(args.song_dir).resolve()
    project_root = Path(__file__).parent.parent

    def resolve_project(p: str) -> Path:
        """Resolve path relative to project root."""
        pth = Path(p)
        if not pth.is_absolute():
            pth = project_root / pth
        return pth

    def resolve_song(p: str) -> Path:
        """Resolve path relative to song dir."""
        pth = Path(p)
        if not pth.is_absolute():
            pth = song_dir / pth
        return pth

    # Load paths
    rulebook_path = resolve_project(args.rulebook)
    bars_path = resolve_song(args.bars)
    chordmap_path = resolve_song(args.chordmap)
    sections_path = resolve_song(args.sections)
    out_guide_path = resolve_song(args.out_guide)
    out_emotion_path = resolve_song(args.out_emotion)

    # Parse emotion tags
    emotion_tags = (
        [t.strip() for t in args.emotion_tags.split(",") if t.strip()]
        if args.emotion_tags
        else []
    )

    print(f"🎼 OtobonAI: Generating guide tones & emotion profile")
    print(f"   Rulebook: {rulebook_path}")
    print(f"   Song dir: {song_dir}")
    
    # Load rulebook
    print(f"📖 Loading rulebook...")
    rb = Rulebook.load(rulebook_path)
    print(f"   Loaded {len(rb.rules)} rules")

    # Load song data
    print(f"📊 Loading song analysis...")
    bars_df = pd.read_parquet(bars_path)
    chordmap_info = _load_chordmap(chordmap_path)
    sections = _load_sections(sections_path)
    
    # Load lyric anchors (v1.5)
    lyric_anchors_path = resolve_song(args.lyric_anchors)
    lyric_anchors = _load_lyric_anchors(lyric_anchors_path)
    if lyric_anchors:
        print(f"   Lyric anchors: {len(lyric_anchors)} bars with anchors")
    
    # Load tempo from tempo_map if available
    tempo_map_path = resolve_song(args.tempo_map)
    if tempo_map_path.exists():
        tempo_data = json.loads(tempo_map_path.read_text(encoding="utf-8"))
        # tempo_map.json has "tempo_points": [[time, bpm], ...]
        # Calculate average BPM from all tempo points
        if tempo_data.get("tempo_points"):
            tempo_points = tempo_data["tempo_points"]
            bpms = [point[1] for point in tempo_points if len(point) >= 2]
            if bpms:
                tempo_bpm = float(np.mean(bpms))
                tempo_min = float(np.min(bpms))
                tempo_max = float(np.max(bpms))
                print(f"   Tempo: {tempo_bpm:.1f} BPM (avg from tempo_map, range: {tempo_min:.1f}-{tempo_max:.1f})")
            else:
                tempo_bpm = args.tempo_default
                print(f"   Tempo: {tempo_bpm:.1f} BPM (default, no valid tempo points)")
        else:
            tempo_bpm = args.tempo_default
            print(f"   Tempo: {tempo_bpm:.1f} BPM (default, no tempo_points in map)")
    else:
        tempo_bpm = args.tempo_default
        print(f"   Tempo: {tempo_bpm:.1f} BPM (default, tempo_map not found)")
    
    print(f"   Bars: {len(bars_df)}")
    print(f"   Sections: {len(sections)}")
    print(f"   Chords: {len(chordmap_info['events_by_bar'])}")

    # Build contexts
    print(f"🔍 Building bar contexts...")
    contexts = build_song_context_per_bar(
        bars_df=bars_df,
        chordmap_info=chordmap_info,
        sections=sections,
        song_emotion_tags=emotion_tags,
        tempo_bpm=tempo_bpm,
    )
    print(f"   Created {len(contexts)} bar contexts")

    key_center = chordmap_info.get("meta", {}).get("key_center", "C")
    print(f"   Key: {key_center}")
    if emotion_tags:
        print(f"   Emotion tags: {', '.join(emotion_tags)}")

    # Generate emotion profile
    print(f"😊 Generating emotion profile...")
    emo_profile = generate_emotion_profile(
        rulebook=rb,
        contexts=contexts,
        key_center=key_center,
        lyric_anchors=lyric_anchors,
    )
    
    out_emotion_path.parent.mkdir(parents=True, exist_ok=True)
    out_emotion_path.write_text(
        json.dumps(emo_profile, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    print(f"   ✅ Written to: {out_emotion_path}")

    # Generate guide tone hints
    print(f"🎵 Generating guide tone hints...")
    guide_hints = generate_guide_tone_hints(
        rulebook=rb,
        contexts=contexts,
        lyric_anchors=lyric_anchors,
    )
    
    out_guide_path.parent.mkdir(parents=True, exist_ok=True)
    out_guide_path.write_text(
        json.dumps(guide_hints, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    print(f"   ✅ Written to: {out_guide_path}")

    # Summary
    print(f"\n📋 Summary:")
    print(f"   Emotion events: {len(emo_profile['events'])}")
    print(f"   Guide tone events: {len(guide_hints['events'])}")
    
    # Sample emotion
    if emo_profile['events']:
        sample = emo_profile['events'][0]
        print(f"\n   Sample emotion (bar {sample['bar']}):")
        print(f"      Energy: {sample['energy']:.2f}")
        print(f"      Tension: {sample['tension']:.2f}")
        print(f"      Brightness: {sample['brightness']:.2f}")
        print(f"      Valence: {sample['valence']:.2f}")
    
    # Sample guide tone
    if guide_hints['events']:
        sample = guide_hints['events'][0]
        print(f"\n   Sample guide tone (bar {sample['bar']}):")
        print(f"      Degree: {sample['scale_degree']}")
        print(f"      Pitch: {sample['approx_pitch']} (MIDI)")
        print(f"      Register: {sample['register']}")
        print(f"      Notes/bar: {sample['notes_per_bar']:.1f}")

    print(f"\n✨ Complete!")


if __name__ == "__main__":
    main()
