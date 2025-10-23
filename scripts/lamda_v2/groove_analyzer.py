#!/usr/bin/env python3
"""Groove Analyzer - LAMDA v2.6+

Analyzes rhythmic groove characteristics:
- Swing percentage (8th note timing deviation)
- Backbeat strength (emphasis on beats 2 & 4)
- Onset deviation histogram (for ML features)

Design: NO-OP safe for drum-less or simple MIDI clips.
"""

from __future__ import annotations
from typing import Any, Dict, List
import numpy as np


def _collect_onsets(pm: Any) -> List[float]:
    """Collect all note onsets from MIDI data.
    
    Parameters
    ----------
    pm : pretty_midi.PrettyMIDI
        MIDI data object.
    
    Returns
    -------
    List[float]
        Sorted list of onset times in seconds.
    """
    onsets = []
    for inst in pm.instruments:
        for note in inst.notes:
            onsets.append(float(note.start))
    return sorted(onsets)


def _bar_grid_sec(downbeats_sec: List[float]) -> List[float]:
    """Get bar grid in seconds.
    
    Parameters
    ----------
    downbeats_sec : List[float]
        Downbeat times in seconds.
    
    Returns
    -------
    List[float]
        Bar start times.
    """
    return list(downbeats_sec or [0.0])


def analyze_groove(
    pm: Any,
    downbeats_sec: List[float],
) -> Dict[str, Any]:
    """Rough swing/backbeat estimator. NO-OP safe for drums-less clips.
    
    Parameters
    ----------
    pm : pretty_midi.PrettyMIDI
        MIDI data object.
    downbeats_sec : List[float]
        Downbeat positions in seconds.
    
    Returns
    -------
    Dict[str, Any]
        {
            "swing_pct": float (0-100),
            "backbeat_strength": float (0-1),
            "onset_deviation_hist": List[int]
        }
    
    Examples
    --------
    >>> groove = analyze_groove(midi_data, [0.0, 0.5, 1.0])
    >>> groove["swing_pct"]
    52.3
    >>> groove["backbeat_strength"]
    0.65
    """
    onsets = _collect_onsets(pm)
    if not onsets or not downbeats_sec:
        return {
            "swing_pct": 0.0,
            "backbeat_strength": 0.5,
            "onset_deviation_hist": []
        }

    # Map each onset to its local beat (quarter grid inside the bar)
    bar_starts = _bar_grid_sec(downbeats_sec)
    q_onsets = []
    bi = 0
    for t in onsets:
        # Advance bar index
        while bi + 1 < len(bar_starts) and t >= bar_starts[bi + 1]:
            bi += 1
        bar_t = t - bar_starts[bi]
        q_onsets.append(bar_t)

    # Estimate 8th swing: compare offsets near 1/8 and 3/8 positions
    # Assume constant bar length from first two downbeats
    if len(downbeats_sec) >= 2:
        bar_len = downbeats_sec[1] - downbeats_sec[0]
    else:
        bar_len = 0.5
    eighth = bar_len / 2.0
    sixteenth = bar_len / 4.0

    # Collect deviations from straight 8th
    devs = []
    for bt in q_onsets:
        # Nearest 8th grid
        k = round(bt / (eighth / 2))  # 16th steps
        grid = k * (eighth / 2)
        devs.append(bt - grid)

    if devs:
        # Swing: positive mean near offbeats suggests delayed off-beat
        mean_dev = float(np.mean(devs))
        swing_pct = max(0.0, min(100.0, 50.0 + 800.0 * mean_dev / (eighth or 1e-6)))
    else:
        swing_pct = 0.0

    # Backbeat: energy near beats 2/4 (i.e., 0.5 and 1.5 quarters in 4/4)
    # Simple ratio using onsets histogram into 4 slots
    slots = [0, 0, 0, 0]
    for bt in q_onsets:
        q = bt / (sixteenth or 1e-6) / 4.0  # quarters
        idx = int(q) % 4
        slots[idx] += 1
    
    # backbeat_strength = (slots at 2 & 4) / total
    total = sum(slots) or 1
    backbeat = float((slots[1] + slots[3]) / total)

    # Tiny histogram of deviations for later ML features
    if devs:
        hist, edges = np.histogram(devs, bins=21, range=(-sixteenth, sixteenth))
        hist_list = list(map(int, hist))
    else:
        hist_list = []

    return {
        "swing_pct": float(round(swing_pct, 1)),
        "backbeat_strength": float(round(backbeat, 3)),
        "onset_deviation_hist": hist_list,
    }
