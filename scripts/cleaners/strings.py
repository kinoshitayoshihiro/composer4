#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Strings専用クリーニング処理
"""

from __future__ import annotations

import statistics
from typing import Any, Dict, List, Tuple

import pretty_midi


def clean_strings(
    pm: pretty_midi.PrettyMIDI,
) -> Tuple[pretty_midi.PrettyMIDI, Dict[str, Any], List[str]]:
    """
    Strings専用のクリーニング
    
    処理内容:
    - レガート接続検出
    - スタッカート洪水検出
    - 和音広がり検出
    
    Returns:
        (pm, metadata, reason_codes)
    """
    metadata: Dict[str, Any] = {}
    reason_codes: List[str] = []
    
    # Strings instruments (40-55: Strings range)
    string_instruments = [
        inst for inst in pm.instruments
        if not inst.is_drum and 40 <= inst.program <= 55
    ]
    
    if not string_instruments:
        return pm, metadata, reason_codes
    
    for inst in string_instruments:
        # 1. レガート接続率
        legato_rate = _detect_legato_connections(inst)
        metadata["legato_connection_rate"] = legato_rate
        
        # 2. スタッカート洪水
        staccato_warnings = _check_staccato_flood(inst)
        reason_codes.extend(staccato_warnings)
        
        # 3. 和音広がり
        chord_spread = _check_chord_spread(inst)
        metadata["chord_spread_semitones"] = chord_spread
        
        if chord_spread > 24:
            reason_codes.append("chord_spread_excess")
        
        # 4. 統計
        if inst.notes:
            velocities = [n.velocity for n in inst.notes]
            metadata["velocity_std"] = round(statistics.stdev(velocities), 2)
    
    return pm, metadata, reason_codes


def _detect_legato_connections(inst: pretty_midi.Instrument) -> float:
    """
    隣接ノートのギャップ < 20ms をレガートとみなす
    """
    if len(inst.notes) < 2:
        return 0.0
    
    notes = sorted(inst.notes, key=lambda n: n.start)
    legato_count = 0
    
    for i in range(len(notes) - 1):
        gap = notes[i + 1].start - notes[i].end
        if -0.020 <= gap <= 0.020:  # ±20ms
            legato_count += 1
    
    return round(legato_count / (len(notes) - 1), 3)


def _check_staccato_flood(inst: pretty_midi.Instrument) -> List[str]:
    """
    短ノート (≤120ms) の連打が規定密度超
    """
    warnings: List[str] = []
    
    if not inst.notes:
        return warnings
    
    short_notes = [
        n for n in inst.notes
        if (n.end - n.start) <= 0.120
    ]
    
    # 全体の70%以上がスタッカート
    if len(short_notes) / len(inst.notes) > 0.7:
        warnings.append("staccato_flood")
    
    return warnings


def _check_chord_spread(inst: pretty_midi.Instrument) -> float:
    """
    同時鳴りの音域広がりを計算
    """
    if not inst.notes:
        return 0.0
    
    notes = sorted(inst.notes, key=lambda n: n.start)
    max_spread = 0.0
    
    i = 0
    while i < len(notes):
        current_time = notes[i].start
        chord_notes = [notes[i]]
        
        j = i + 1
        while j < len(notes) and abs(notes[j].start - current_time) < 0.050:
            chord_notes.append(notes[j])
            j += 1
        
        if len(chord_notes) >= 2:
            pitches = [n.pitch for n in chord_notes]
            spread = max(pitches) - min(pitches)
            max_spread = max(max_spread, spread)
        
        i = j if j > i + 1 else i + 1
    
    return round(max_spread, 1)
