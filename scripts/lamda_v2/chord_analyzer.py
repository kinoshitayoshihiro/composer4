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
    out: Dict[str, Any] = {"unit": "ql", "events": []}

    # TODO: Implement chord recognition
    # Phase1: Basic maj/min
    # Phase2: 7th/sus/add9

    return out


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
