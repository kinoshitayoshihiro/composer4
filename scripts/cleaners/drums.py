#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Drums専用クリーニング処理
(既存のLAMDa Stage1ロジックを統合)
"""

from __future__ import annotations

import statistics
from typing import Any, Dict, List, Tuple

import pretty_midi


def clean_drums(
    pm: pretty_midi.PrettyMIDI,
) -> Tuple[pretty_midi.PrettyMIDI, Dict[str, Any], List[str]]:
    """
    Drums専用のクリーニング
    
    処理内容:
    - グリッド外れ検出
    - キックオンビート率
    - 既存LAMDaメトリクス統合
    
    Returns:
        (pm, metadata, reason_codes)
    """
    metadata: Dict[str, Any] = {}
    reason_codes: List[str] = []
    
    # Drum instruments
    drum_instruments = [inst for inst in pm.instruments if inst.is_drum]
    
    if not drum_instruments:
        return pm, metadata, reason_codes
    
    for inst in drum_instruments:
        # 1. グリッド外れ
        grid_stats = _check_drum_grid_alignment(inst, pm)
        metadata.update(grid_stats)
        
        # 2. キックオンビート率
        kick_rate = _check_kick_on_beat(inst, pm)
        metadata["kick_on_beat_rate"] = kick_rate
        
        # 3. 統計
        if inst.notes:
            velocities = [n.velocity for n in inst.notes]
            metadata["velocity_std"] = round(statistics.stdev(velocities), 2)
            metadata["velocity_mean"] = round(statistics.mean(velocities), 2)
    
    return pm, metadata, reason_codes


def _check_drum_grid_alignment(
    inst: pretty_midi.Instrument,
    pm: pretty_midi.PrettyMIDI,
) -> Dict[str, Any]:
    """
    ドラムノートのグリッド整合性
    """
    stats: Dict[str, Any] = {}
    
    if not inst.notes:
        return stats
    
    tempo_changes = pm.get_tempo_changes()
    tempo = tempo_changes[1][0] if len(tempo_changes[1]) > 0 else 120.0
    
    # 16分音符グリッド
    sixteenth_duration = (60.0 / tempo) / 4
    
    offsets = []
    for note in inst.notes:
        grid_position = note.start / sixteenth_duration
        nearest_grid = round(grid_position)
        offset_ms = abs(grid_position - nearest_grid) * sixteenth_duration * 1000
        offsets.append(offset_ms)
    
    if offsets:
        stats["grid_off_std_ms"] = round(statistics.stdev(offsets), 2)
        stats["grid_off_mean_ms"] = round(statistics.mean(offsets), 2)
    
    return stats


def _check_kick_on_beat(
    inst: pretty_midi.Instrument,
    pm: pretty_midi.PrettyMIDI,
) -> float:
    """
    キック (MIDI 35, 36) が拍頭にある割合
    """
    kick_pitches = {35, 36}
    kick_notes = [n for n in inst.notes if n.pitch in kick_pitches]
    
    if not kick_notes:
        return 0.0
    
    tempo_changes = pm.get_tempo_changes()
    tempo = tempo_changes[1][0] if len(tempo_changes[1]) > 0 else 120.0
    
    beat_duration = 60.0 / tempo
    
    on_beat_count = 0
    for note in kick_notes:
        beat_position = note.start / beat_duration
        nearest_beat = round(beat_position)
        offset = abs(beat_position - nearest_beat)
        
        # 拍頭から±5%以内
        if offset < 0.05:
            on_beat_count += 1
    
    return round(on_beat_count / len(kick_notes), 3)
