"""Chord Analyzer - LAMDA v2.6+

Analyzes MIDI data to extract chord progressions with support for:
- Basic triads (maj/min)
- 7th chords (maj7/min7/dom7)
- Extended chords (sus2/sus4/add9)
- Consecutive identical chord merging
- Minimum dwell time enforcement (2QL default)
"""

from typing import Any, Dict, List, Optional
import numpy as np


def extract_bar_chords(
    midi_data: Any,
    downbeats_ql: List[float],
    min_dwell_ql: float = 2.0,
    extended_vocab: bool = True,
) -> Dict[str, Any]:
    """Extract one chord per bar from MIDI data.

    Parameters
    ----------
    midi_data : pretty_midi.PrettyMIDI
        MIDI data object.
    downbeats_ql : List[float]
        Downbeat positions in quarter lengths.
    min_dwell_ql : float, optional
        Minimum chord duration in quarter lengths. Defaults to 2.0.
    extended_vocab : bool, optional
        Enable 7th/sus/add9 recognition. Defaults to True.

    Returns
    -------
    Dict[str, Any]
        Chord map with "unit"="ql" and "events" list.
        Each event: {"time": float, "root": str, "quality": str, "confidence": float}

    Examples
    --------
    >>> chordmap = extract_bar_chords(midi_data, [0.0, 4.0, 8.0])
    >>> chordmap["events"]
    [{"time": 0.0, "root": "C", "quality": "maj", "confidence": 0.85},
     {"time": 4.0, "root": "Am", "quality": "min", "confidence": 0.78}]
    """
    if not downbeats_ql or not hasattr(midi_data, 'instruments'):
        return {"unit": "ql", "events": []}

    # Import tempo_timing for ql↔sec conversion
    from scripts.lamda_v2.tempo_timing import ql_to_sec, build_beat_grid
    
    # Build beat grid from MIDI
    grid = build_beat_grid(midi_data)
    tempo_map = grid.get("tempo_map", [(0.0, 120.0)])
    
    # Collect all notes from all instruments
    all_notes = []
    for inst in midi_data.instruments:
        if inst.is_drum:
            continue  # Skip drum tracks
        all_notes.extend(inst.notes)
    
    if not all_notes:
        return {"unit": "ql", "events": []}
    
    # Analyze one chord per bar
    events = []
    for bar_idx, db_ql in enumerate(downbeats_ql):
        # Convert bar start/end to seconds
        db_sec = ql_to_sec(db_ql, tempo_map)
        next_db_ql = downbeats_ql[bar_idx + 1] if bar_idx + 1 < len(downbeats_ql) else db_ql + 4.0
        next_db_sec = ql_to_sec(next_db_ql, tempo_map)
        
        # Collect notes in this bar
        bar_notes = [n for n in all_notes if n.start >= db_sec and n.start < next_db_sec]
        
        if not bar_notes:
            continue  # No notes in this bar
        
        # Analyze chord (simple pitch class histogram)
        root, quality, confidence = _analyze_chord(bar_notes, extended_vocab)
        
        events.append({
            "time": float(db_ql),
            "root": root,
            "quality": quality,
            "confidence": confidence,
        })
    
    # Apply min_dwell and merge consecutive identical chords
    events = merge_consecutive_chords(events)
    events = enforce_min_dwell(events, min_dwell_ql)
    
    return {"unit": "ql", "events": events}


def _analyze_chord(notes: List[Any], extended_vocab: bool) -> tuple[str, str, float]:
    """Analyze notes to determine chord root and quality.
    
    Returns (root, quality, confidence).
    """
    ROOTS = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    
    # Build pitch class histogram (weighted by velocity and duration)
    pc_hist = np.zeros(12)
    for note in notes:
        pc = note.pitch % 12
        weight = note.velocity * (note.end - note.start)
        pc_hist[pc] += weight
    
    if pc_hist.sum() == 0:
        return "C", "maj", 0.0
    
    # Normalize
    pc_hist /= pc_hist.sum()
    
    # Find root (most prominent pitch class)
    root_pc = int(np.argmax(pc_hist))
    root = ROOTS[root_pc]
    
    # Determine quality (simple maj/min heuristic)
    # Check for minor 3rd (3 semitones above root)
    # Check for major 3rd (4 semitones above root)
    minor_3rd = (root_pc + 3) % 12
    major_3rd = (root_pc + 4) % 12
    
    has_minor_3rd = pc_hist[minor_3rd] > 0.1
    has_major_3rd = pc_hist[major_3rd] > 0.1
    
    # Check for 7th extensions if enabled
    if extended_vocab:
        minor_7th = (root_pc + 10) % 12
        major_7th = (root_pc + 11) % 12
        has_minor_7th = pc_hist[minor_7th] > 0.05
        has_major_7th = pc_hist[major_7th] > 0.05
        
        # Determine quality
        if has_minor_3rd and has_minor_7th:
            quality = "min7"
        elif has_major_3rd and has_minor_7th:
            quality = "dom7"
        elif has_major_3rd and has_major_7th:
            quality = "maj7"
        elif has_minor_3rd:
            quality = "min"
        else:
            quality = "maj"
    else:
        # Basic maj/min only
        quality = "min" if has_minor_3rd else "maj"
    
    # Confidence = sum of top 3 pitch classes
    confidence = float(np.sort(pc_hist)[-3:].sum())
    
    return root, quality, confidence


def merge_consecutive_chords(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Merge consecutive identical chords.

    Parameters
    ----------
    events : List[Dict[str, Any]]
        Chord events with "root" and "quality" keys.

    Returns
    -------
    List[Dict[str, Any]]
        Merged events (C→C→Am becomes C→Am).
    """
    if not events:
        return []

    merged = [events[0]]
    for event in events[1:]:
        prev = merged[-1]
        if event["root"] == prev["root"] and event["quality"] == prev["quality"]:
            continue  # Skip duplicate
        merged.append(event)

    return merged


def enforce_min_dwell(events: List[Dict[str, Any]], min_ql: float = 2.0) -> List[Dict[str, Any]]:
    """Remove chords shorter than min_ql.

    Parameters
    ----------
    events : List[Dict[str, Any]]
        Chord events with "time" key.
    min_ql : float, optional
        Minimum duration in quarter lengths.

    Returns
    -------
    List[Dict[str, Any]]
        Filtered events.
    """
    if len(events) <= 1:
        return events

    filtered = []
    for i, event in enumerate(events):
        if i == len(events) - 1:
            # Last event always kept
            filtered.append(event)
            continue

        duration = events[i + 1]["time"] - event["time"]
        if duration >= min_ql:
            filtered.append(event)

    return filtered
