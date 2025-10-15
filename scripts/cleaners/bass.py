#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bass専用クリーニング処理
"""

from __future__ import annotations

import statistics
from typing import Any, Dict, List, Tuple

import pretty_midi


def clean_bass(
    pm: pretty_midi.PrettyMIDI,
) -> Tuple[pretty_midi.PrettyMIDI, Dict[str, Any], List[str]]:
    """
    Bass専用のクリーニング
    
    処理内容:
    - モノフォニック保証
    - 巨大跳躍フラグ
    - グリッド外れ検出
    
    Returns:
        (pm, metadata, reason_codes)
    """
    metadata: Dict[str, Any] = {}
    reason_codes: List[str] = []
    
    # Bass instruments (32-39: Bass range)
    bass_instruments = [
        inst for inst in pm.instruments
        if not inst.is_drum and 32 <= inst.program <= 39
    ]
    
    if not bass_instruments:
        return pm, metadata, reason_codes
    
    for inst in bass_instruments:
        # 1. モノフォニック保証
        poly_count = _enforce_monophonic(inst)
        if poly_count > 0:
            metadata["polyphonic_conflicts_resolved"] = poly_count
        
        # 2. 巨大跳躍検出
        leap_warnings = _check_large_leaps(inst)
        reason_codes.extend(leap_warnings)
        
        # 3. グリッド外れ
        grid_stats = _check_grid_alignment(inst, pm)
        metadata.update(grid_stats)
        
        if grid_stats.get("grid_off_ratio", 0) > 0.3:
            reason_codes.append("grid_off_outlier")
    
    return pm, metadata, reason_codes


def _enforce_monophonic(inst: pretty_midi.Instrument) -> int:
    """
    同一タイムレンジで2音以上ある場合、長い方を残す
    """
    if not inst.notes:
        return 0
    
    notes = sorted(inst.notes, key=lambda n: n.start)
    filtered_notes = []
    conflicts = 0
    
    i = 0
    while i < len(notes):
        current = notes[i]
        overlapping = [current]
        
        # 重複するノートを探す
        j = i + 1
        while j < len(notes) and notes[j].start < current.end:
            overlapping.append(notes[j])
            j += 1
        
        if len(overlapping) > 1:
            # 最も長いノートを残す
            longest = max(overlapping, key=lambda n: n.end - n.start)
            filtered_notes.append(longest)
            conflicts += len(overlapping) - 1
        else:
            filtered_notes.append(current)
        
        i = j if j > i + 1 else i + 1
    
    inst.notes = filtered_notes
    return conflicts


def _check_large_leaps(inst: pretty_midi.Instrument) -> List[str]:
    """
    >12半音の連発を検出
    """
    warnings: List[str] = []
    
    if len(inst.notes) < 2:
        return warnings
    
    notes = sorted(inst.notes, key=lambda n: n.start)
    large_leap_count = 0
    
    for i in range(len(notes) - 1):
        interval = abs(notes[i + 1].pitch - notes[i].pitch)
        if interval > 12:
            large_leap_count += 1
    
    # 全体の30%以上が大跳躍
    if large_leap_count / len(notes) > 0.3:
        warnings.append("leap_excess")
    
    return warnings


def _check_grid_alignment(
    inst: pretty_midi.Instrument,
    pm: pretty_midi.PrettyMIDI,
) -> Dict[str, Any]:
    """
    拍基準からのずれを計算
    """
    stats: Dict[str, Any] = {}
    
    if not inst.notes:
        return stats
    
    # テンポと拍子を取得
    tempo_changes = pm.get_tempo_changes()
    tempo = tempo_changes[1][0] if len(tempo_changes[1]) > 0 else 120.0
    
    time_sig = pm.time_signature_changes[0] if pm.time_signature_changes else None
    numerator = time_sig.numerator if time_sig else 4
    
    # 1拍の長さ (秒)
    beat_duration = 60.0 / tempo
    
    # 各ノートのグリッドからのずれ
    offsets = []
    for note in inst.notes:
        beat_position = note.start / beat_duration
        nearest_beat = round(beat_position)
        offset_ms = abs(beat_position - nearest_beat) * beat_duration * 1000
        offsets.append(offset_ms)
    
    if offsets:
        stats["grid_off_std_ms"] = round(statistics.stdev(offsets), 2)
        stats["grid_off_mean_ms"] = round(statistics.mean(offsets), 2)
        
        # 30ms以上外れた割合
        off_count = sum(1 for o in offsets if o > 30)
        stats["grid_off_ratio"] = round(off_count / len(offsets), 3)
    
    return stats
