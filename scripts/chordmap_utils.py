"""Shared utilities for chordmap handling in V2 generators."""

from typing import Any, Dict, List, Set, Optional
import json
import re


# MIDI note mapping (octave 3 reference)
NOTE_TO_MIDI = {
    "C": 48,
    "C#": 49,
    "Db": 49,
    "D": 50,
    "D#": 51,
    "Eb": 51,
    "E": 52,
    "F": 53,
    "F#": 54,
    "Gb": 54,
    "G": 55,
    "G#": 56,
    "Ab": 56,
    "A": 57,
    "A#": 58,
    "Bb": 58,
    "B": 59,
}

# Interval to semitones mapping
INTERVAL_SEMITONES = {
    "b3": 3,
    "3": 4,
    "4": 5,
    "5": 7,
    "b5": 6,
    "#5": 8,
    "b7": 10,
    "7": 11,
    "maj7": 11,
    "9": 14,
    "11": 17,
    "13": 21,
    "add9": 14,
    "add11": 17,
    "add13": 21,
}


# Chord symbol regex parser
SYMBOL_RE = re.compile(
    r"^(?P<root>[A-G](?:#|b)?)"
    r"(?P<quality>maj7|M7|m7|m|7|dim|aug|sus4|sus2|sus)?"
    r"(?:\((?P<paren>[^\)]+)\))?"
    r"(?:add(?P<add>\d+))?$",
    re.IGNORECASE,
)


def parse_full_symbol(symbol: str) -> Dict[str, Any]:
    """
    Parse complete chord symbol (e.g., "C#m7(9)", "Gadd9", "Dsus4").

    Returns dict with root, quality, and extensions.
    """
    # Normalize
    sym = symbol.strip().replace("maj7", "M7")

    m = SYMBOL_RE.match(sym)
    if not m:
        # Fallback: try to extract at least root
        root_match = re.match(r"^([A-G](?:#|b)?)", symbol)
        if root_match:
            return {"root": root_match.group(1), "quality": "", "extensions": [], "sus": None}
        return {"root": "C", "quality": "", "extensions": [], "sus": None}

    root = m.group("root")
    quality = (m.group("quality") or "").replace("M7", "maj7")

    extensions = []
    sus = None

    # Parse parenthetical extensions
    paren = m.group("paren")
    if paren:
        for ext in paren.split(","):
            ext = ext.strip()
            if ext in {"9", "11", "13"}:
                extensions.append(ext)

    # Parse add extensions
    add = m.group("add")
    if add:
        extensions.append(f"add{add}")

    # Detect sus
    if quality and "sus" in quality.lower():
        if "sus2" in quality.lower():
            sus = "2"
        elif "sus4" in quality.lower():
            sus = "4"
        else:
            sus = "4"  # Default sus = sus4

    return {"root": root, "quality": quality, "extensions": extensions, "sus": sus}


def parse_symbol(symbol: str) -> Dict[str, Any]:
    """
    Parse chord symbol for V2 generators (converts symbol to root/quality/intervals).

    Args:
        symbol: Chord symbol (e.g., "C#m7(9)", "Gadd9", "Dsus4")

    Returns:
        {
            "root": "C#",
            "quality": "m7",
            "extensions": ["9"],
            "sus": None,
            "root_midi": 49,  # Added for compatibility
            "intervals": {0, 3, 7, 10, 2}  # Added for compatibility
        }
    """
    # First parse symbol structure
    parsed = parse_full_symbol(symbol)
    root = parsed["root"]
    quality = parsed["quality"]
    extensions = parsed["extensions"]

    # Convert to chord_info format with intervals
    tensions = [ext.replace("add", "") for ext in extensions]
    chord_info = parse_chord_symbol(root, quality, tensions)

    # Merge results
    return {
        "root": root,
        "quality": quality,
        "extensions": extensions,
        "sus": parsed["sus"],
        "root_midi": chord_info["root_midi"],
        "intervals": chord_info["intervals"],
        "has_tension": chord_info["has_tension"],
    }


def parse_chord_symbol(root: str, quality: str, tensions: List[str] = None) -> Dict[str, Any]:
    """
    Parse chord from root + quality + tensions format.

    Args:
        root: Root note (e.g., "E", "C#")
        quality: Quality (e.g., "m", "7", "maj7", "sus4", "")
        tensions: Optional list of tensions (e.g., ["9"], ["9", "13"])

    Returns:
        {
            "root_pc": pitch class (0-11),
            "root_midi": MIDI note (octave 3),
            "intervals": set of semitones from root,
            "quality": normalized quality string,
            "has_tension": bool
        }
    """
    root_pc = NOTE_TO_MIDI.get(root, 48) % 12
    root_midi = NOTE_TO_MIDI.get(root, 48)

    intervals: Set[int] = {0}  # Always include root
    has_tension = False

    # Parse quality for base triad/tetrad
    q = (quality or "").lower()

    if "m" in q and "maj" not in q:  # Minor
        intervals.update([3, 7])  # b3, 5
    elif "sus2" in q:
        intervals.update([2, 7])  # 2, 5
    elif "sus4" in q or q == "sus":
        intervals.update([5, 7])  # 4, 5
    elif "dim" in q:
        intervals.update([3, 6])  # b3, b5
    elif "aug" in q:
        intervals.update([4, 8])  # 3, #5
    else:  # Major (default)
        intervals.update([4, 7])  # 3, 5

    # Add 7ths
    if "maj7" in q or "M7" in q.upper():
        intervals.add(11)  # maj7
    elif "7" in q:
        intervals.add(10)  # b7

    # Add tensions from tensions list
    if tensions:
        has_tension = True
        for t in tensions:
            t_clean = str(t).strip()
            if t_clean == "9":
                intervals.add(14 % 12)  # 9 -> 2
            elif t_clean == "11":
                intervals.add(17 % 12)  # 11 -> 5
            elif t_clean == "13":
                intervals.add(21 % 12)  # 13 -> 9
            elif t_clean == "add9":
                intervals.add(14 % 12)

    # Handle add9 in quality
    if "add9" in q:
        intervals.add(14 % 12)
        has_tension = True

    return {
        "root_pc": root_pc,
        "root_midi": root_midi,
        "intervals": intervals,
        "quality": quality or "",
        "has_tension": has_tension,
    }


def get_chord_tones(
    chord_info: Dict[str, Any], bass_octave: int = 2, upper_octave: Optional[int] = None
) -> List[int]:
    """
    Generate MIDI note numbers for chord tones.

    Args:
        chord_info: Output from parse_symbol() or parse_chord_symbol()
                    Must contain "root_midi" and "intervals"
        bass_octave: Octave for bass/root notes (default 2 = C2, E2, etc)
        upper_octave: Octave for upper voices (optional, defaults to bass_octave)

    Returns:
        Sorted list of MIDI note numbers
    """
    root_midi = chord_info.get("root_midi", 48)
    intervals = chord_info.get("intervals", {0, 4, 7})

    # Default upper_octave to bass_octave if not specified
    if upper_octave is None:
        upper_octave = bass_octave

    notes = []
    root_pc = root_midi % 12

    # Generate notes for each interval
    for interval in sorted(intervals):
        # Calculate MIDI note at specified octave
        if interval == 0:
            # Root uses bass_octave
            note = root_pc + 12 * bass_octave
        else:
            # Other intervals use upper_octave
            note = root_pc + interval + 12 * upper_octave

        notes.append(note)

    return sorted(set(notes))


def load_chordmap(chordmap_path: str) -> List[Dict[str, Any]]:
    """
    Load chordmap (manual or locked format).

    Handles multiple formats:
    - Symbol-based: {"unit": "bar", "events": [{"bar": 0, "symbol": "C#m7"}, ...]}
    - Manual: {"unit": "ql", "events": [{"time": 0.0, "root": "E", "quality": "m", "tensions": ["9"]}, ...]}
    - Legacy: [{"time_ql": 0.0, "root_midi": 52, ...}, ...]

    Returns normalized events with root_midi, time_ql/bar, and parsed chord info.
    """
    with open(chordmap_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        events = data
        unit = "ql"
    elif isinstance(data, dict) and "events" in data:
        events = data["events"]
        unit = data.get("unit", "ql")
    else:
        raise ValueError('chordmap must be a list or {"events": [...]}')

    # Normalize events: add root_midi if missing, handle time/time_ql/bar
    for ev in events:
        # Handle symbol-based format
        if "symbol" in ev:
            parsed = parse_full_symbol(ev["symbol"])
            ev["root"] = parsed["root"]
            ev["quality"] = parsed["quality"]
            ev["tensions"] = parsed["extensions"]
            ev["sus"] = parsed["sus"]

        # Normalize time field
        if "bar" in ev:
            ev["bar_idx"] = ev["bar"]
            ev["time_ql"] = ev["bar"] * 4.0  # Assume 4/4 time
        elif "time" in ev and "time_ql" not in ev:
            ev["time_ql"] = ev["time"]
        elif "time_ql" not in ev:
            ev["time_ql"] = 0.0

        # Calculate root_midi from root string if missing
        if "root_midi" not in ev and "root" in ev:
            root = ev["root"]
            ev["root_midi"] = NOTE_TO_MIDI.get(root, 48)
        elif "root_midi" not in ev:
            ev["root_midi"] = 48  # Default C

        # Parse chord for voicing/tension support
        if "root" in ev:
            root = ev["root"]
            quality = ev.get("quality", "")
            tensions = ev.get("tensions", [])

            chord_info = parse_chord_symbol(root, quality, tensions)
            ev["chord_info"] = chord_info

    return events


def get_chord_at_bar(chordmap: List[Dict[str, Any]], bar_idx: int) -> Dict[str, Any]:
    """
    Find chord event overlapping with bar_idx.

    Handles both bar-indexed and time-based formats:
    - Bar-indexed: uses "bar" or "bar_idx" field for direct lookup
    - Time-based: converts bar_idx to time_ql and finds overlapping chord

    Assumes chords extend until the next chord (or indefinitely for last chord).
    """
    if not chordmap:
        return {}

    # Check if bar-indexed format (has "bar" or "bar_idx" field)
    has_bar_idx = "bar" in chordmap[0] or "bar_idx" in chordmap[0]

    if has_bar_idx:
        # Bar-indexed format: direct lookup
        active_chord = chordmap[0]
        for i, chord in enumerate(chordmap):
            chord_bar = chord.get("bar", chord.get("bar_idx", 0))

            # Check if this chord covers bar_idx
            if i + 1 < len(chordmap):
                next_chord_bar = chordmap[i + 1].get(
                    "bar", chordmap[i + 1].get("bar_idx", chord_bar + 1)
                )
            else:
                next_chord_bar = chord_bar + 999999  # Last chord extends to end

            if chord_bar <= bar_idx < next_chord_bar:
                return chord
            elif chord_bar <= bar_idx:
                active_chord = chord

        return active_chord

    else:
        # Time-based format: convert bar to time_ql
        bar_start_ql = bar_idx * 4.0

        active_chord = chordmap[0]
        for i, chord in enumerate(chordmap):
            chord_start = chord.get("time_ql", 0.0)
            # Duration: until next chord or default to extend to end
            if i + 1 < len(chordmap):
                chord_end = chordmap[i + 1].get("time_ql", chord_start + 4.0)
            else:
                chord_end = chord_start + chord.get("duration_ql", 999999.0)  # Last chord extends

            if chord_start <= bar_start_ql < chord_end:
                return chord
            elif chord_start <= bar_start_ql:
                active_chord = chord  # Update to latest chord before bar

        return active_chord
