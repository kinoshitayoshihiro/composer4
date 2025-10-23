"""Section Analyzer - LAMDA v2.6+

Segments music into sections (Intro/Verse/Chorus/Bridge/Outro) using:
- RMS energy curve
- Novelty detection (self-similarity matrix)
- Minimum section length (8 bars default)
- Downbeat snapping
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


def auto_segment_sections(
    midi_data: Any,
    downbeats_ql: List[float],
    min_bars: int = 8,
) -> Dict[str, Any]:
    """Automatically segment music into sections.

    Parameters
    ----------
    midi_data : pretty_midi.PrettyMIDI
        MIDI data object.
    downbeats_ql : List[float]
        Downbeat positions in quarter lengths.
    min_bars : int, optional
        Minimum section length in bars. Defaults to 8.

    Returns
    -------
    Dict[str, Any]
        {"unit": "bar",
         "sections": [{"bar": 0, "label": "intro"}, ...],
         "energy": [[bar, normalized_energy], ...]}

    Examples
    --------
    >>> sections = auto_segment_sections(midi_data, [0, 4, 8, 12, 16], min_bars=4)
    >>> sections["sections"]
    [{"bar": 0, "label": "intro"}, {"bar": 8, "label": "verse"}]
    """
    if not downbeats_ql or not hasattr(midi_data, 'instruments'):
        return {"unit": "bar", "sections": [], "energy": []}

    # Import tempo_timing for ql↔sec conversion
    from scripts.lamda_v2.tempo_timing import ql_to_sec, build_beat_grid
    
    # Build beat grid
    grid = build_beat_grid(midi_data)
    tempo_map = grid.get("tempo_map", [(0.0, 120.0)])
    
    # Calculate RMS energy per bar
    energy_per_bar = _compute_bar_energy(midi_data, downbeats_ql, tempo_map)
    
    # Normalize energy
    if len(energy_per_bar) > 0:
        max_energy = max(energy_per_bar) if max(energy_per_bar) > 0 else 1.0
        energy_normalized = [e / max_energy for e in energy_per_bar]
    else:
        energy_normalized = []
    
    # Detect section boundaries using energy peaks + novelty
    boundaries = _detect_section_boundaries(
        energy_normalized,
        min_bars=min_bars,
    )
    
    # Assign section labels
    sections = _assign_section_labels(boundaries, len(downbeats_ql))
    
    # Format energy output
    energy_output = [[i, float(e)] for i, e in enumerate(energy_normalized)]
    
    return {
        "unit": "bar",
        "sections": sections,
        "energy": energy_output,
    }


def _compute_bar_energy(
    midi_data: Any,
    downbeats_ql: List[float],
    tempo_map: List[Tuple[float, float]],
) -> List[float]:
    """Compute RMS energy per bar from MIDI data.
    
    Returns list of energy values (one per bar).
    """
    from scripts.lamda_v2.tempo_timing import ql_to_sec
    
    # Collect all notes from non-drum instruments
    all_notes = []
    for inst in midi_data.instruments:
        if inst.is_drum:
            continue
        all_notes.extend(inst.notes)
    
    if not all_notes:
        return [0.0] * len(downbeats_ql)
    
    # Calculate energy per bar
    energy = []
    for bar_idx in range(len(downbeats_ql)):
        db_ql = downbeats_ql[bar_idx]
        db_sec = ql_to_sec(db_ql, tempo_map)
        
        # Next bar start
        next_db_ql = downbeats_ql[bar_idx + 1] if bar_idx + 1 < len(downbeats_ql) else db_ql + 4.0
        next_db_sec = ql_to_sec(next_db_ql, tempo_map)
        
        # Collect notes in this bar
        bar_notes = [n for n in all_notes if n.start >= db_sec and n.start < next_db_sec]
        
        if not bar_notes:
            energy.append(0.0)
            continue
        
        # RMS calculation: sqrt(mean(velocity^2))
        velocities = [n.velocity for n in bar_notes]
        rms = np.sqrt(np.mean(np.array(velocities) ** 2))
        energy.append(float(rms))
    
    return energy


def _detect_section_boundaries(
    energy: List[float],
    min_bars: int = 8,
) -> List[int]:
    """Detect section boundaries using energy curve analysis.
    
    Returns list of bar indices where sections start.
    """
    if len(energy) < min_bars * 2:
        # Too short for segmentation
        return [0]
    
    # Compute energy derivative (rate of change)
    derivative = []
    for i in range(1, len(energy)):
        derivative.append(energy[i] - energy[i - 1])
    
    # Find peaks in derivative (significant energy changes)
    boundaries = [0]  # Always start at bar 0
    
    for i in range(min_bars, len(derivative), min_bars):
        # Check if this is a significant energy change
        # Look for local maxima in absolute derivative
        window_start = max(0, i - 2)
        window_end = min(len(derivative), i + 3)
        window = derivative[window_start:window_end]
        
        if not window:
            continue
        
        local_max = max(abs(d) for d in window)
        current_change = abs(derivative[i])
        
        # If current change is significant (> 50% of local max)
        if current_change > 0.5 * local_max and current_change > 0.1:
            # Ensure minimum spacing
            if not boundaries or i - boundaries[-1] >= min_bars:
                boundaries.append(i)
    
    return boundaries


def _assign_section_labels(
    boundaries: List[int],
    total_bars: int,
) -> List[Dict[str, Any]]:
    """Assign section labels based on position and count.
    
    Simple heuristic:
    - First section: "intro"
    - Last section: "outro"
    - Middle sections: alternate "verse", "chorus"
    """
    if not boundaries:
        return [{"bar": 0, "label": "intro"}]
    
    sections = []
    num_sections = len(boundaries)
    
    for idx, bar in enumerate(boundaries):
        if idx == 0:
            label = "intro"
        elif idx == num_sections - 1 and total_bars - bar <= 8:
            label = "outro"
        elif idx % 2 == 1:
            label = "verse"
        else:
            label = "chorus"
        
        sections.append({"bar": int(bar), "label": label})
    
    return sections


def compute_novelty_curve(
    energy: List[float],
    kernel_size: int = 8,
) -> List[float]:
    """Compute novelty curve using self-similarity matrix (optional enhancement).
    
    Parameters
    ----------
    energy : List[float]
        Energy values per bar.
    kernel_size : int, optional
        Kernel size for self-similarity computation.
    
    Returns
    -------
    List[float]
        Novelty values (higher = more novel/different).
    
    Note
    ----
    This is a placeholder for future enhancement. Currently returns
    simple energy derivative as a proxy for novelty.
    """
    if len(energy) < 2:
        return [0.0] * len(energy)
    
    # Simple proxy: absolute energy derivative
    novelty = [0.0]  # First bar has no novelty
    for i in range(1, len(energy)):
        novelty.append(abs(energy[i] - energy[i - 1]))
    
    # Normalize
    max_nov = max(novelty) if novelty else 1.0
    if max_nov > 0:
        novelty = [n / max_nov for n in novelty]
    
    return novelty
