#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Guitar専用クリーニング処理
"""

from __future__ import annotations

import statistics
from typing import Any, Dict, List, Tuple

import pretty_midi


def clean_guitar(
    pm: pretty_midi.PrettyMIDI,
) -> Tuple[pretty_midi.PrettyMIDI, Dict[str, Any], List[str]]:
    """
    Guitar専用のクリーニング
    
    処理内容:
    - ストラム検出
    - 12弦化ノイズ整理
    - 過密アルペジオ検出
    - 音域警告
    
    Returns:
        (pm, metadata, reason_codes)
    """
    metadata: Dict[str, Any] = {}
    reason_codes: List[str] = []
    
    # Guitar instruments (24-31: Guitar range)
    guitar_instruments = [
        inst for inst in pm.instruments
        if not inst.is_drum and 24 <= inst.program <= 31
    ]
    
    if not guitar_instruments:
        return pm, metadata, reason_codes
    
    for inst in guitar_instruments:
        # 1. ストラム検出
        strum_count = _detect_strums(inst)
        metadata["strum_count"] = strum_count
        
        # 2. 12弦ノイズ除去
        octave_count = _remove_octave_doubling(inst)
        if octave_count > 0:
            metadata["removed_octave_doublings"] = octave_count
        
        # 3. 過密アルペジオ
        arpeggio_warnings = _check_dense_arpeggios(inst)
        reason_codes.extend(arpeggio_warnings)
        
        # 4. 音域チェック
        pitch_warnings = _check_guitar_range(inst)
        reason_codes.extend(pitch_warnings)
        
        # 5. 統計
        if inst.notes:
            velocities = [n.velocity for n in inst.notes]
            metadata["velocity_std"] = round(statistics.stdev(velocities), 2)
            metadata["notes_per_bar"] = len(inst.notes) / max(1, metadata.get("bars", 1))
    
    return pm, metadata, reason_codes


def _detect_strums(inst: pretty_midi.Instrument) -> int:
    """
    ストラム検出 (0-60ms以内の和音群)
    """
    if not inst.notes:
        return 0
    
    notes = sorted(inst.notes, key=lambda n: n.start)
    strum_count = 0
    
    i = 0
    while i < len(notes):
        current_time = notes[i].start
        chord_notes = [notes[i]]
        
        j = i + 1
        while j < len(notes) and (notes[j].start - current_time) <= 0.060:
            chord_notes.append(notes[j])
            j += 1
        
        # 3音以上でストラムとみなす
        if len(chord_notes) >= 3:
            strum_count += 1
        
        i = j if j > i + 1 else i + 1
    
    return strum_count


def _remove_octave_doubling(inst: pretty_midi.Instrument) -> int:
    """
    完全8veのダブリング (±5ms以内) を1音に縮退
    """
    if not inst.notes:
        return 0
    
    notes = sorted(inst.notes, key=lambda n: (n.start, n.pitch))
    filtered_notes = []
    removed_count = 0
    
    i = 0
    while i < len(notes):
        current = notes[i]
        keep = True
        
        # 近接ノートをチェック
        for j in range(i + 1, min(i + 10, len(notes))):
            other = notes[j]
            
            # 時間が離れたら終了
            if abs(other.start - current.start) > 0.005:
                break
            
            # 完全8ve差
            if abs(other.pitch - current.pitch) == 12:
                # 短い方を削除
                if (current.end - current.start) < (other.end - other.start):
                    keep = False
                    removed_count += 1
                    break
        
        if keep:
            filtered_notes.append(current)
        
        i += 1
    
    inst.notes = filtered_notes
    return removed_count


def _check_dense_arpeggios(inst: pretty_midi.Instrument) -> List[str]:
    """
    過密アルペジオ (IOI < 15ms連続) を検出
    """
    warnings: List[str] = []
    
    if len(inst.notes) < 4:
        return warnings
    
    notes = sorted(inst.notes, key=lambda n: n.start)
    dense_count = 0
    
    for i in range(len(notes) - 1):
        ioi = notes[i + 1].start - notes[i].start
        if 0 < ioi < 0.015:  # 15ms以下
            dense_count += 1
    
    # 全体の20%以上が過密
    if dense_count / len(notes) > 0.2:
        warnings.append("arpeggio_glitch")
    
    return warnings


def _check_guitar_range(inst: pretty_midi.Instrument) -> List[str]:
    """
    ギター音域チェック (E2=40 ~ C7=84)
    """
    warnings: List[str] = []
    
    if not inst.notes:
        return warnings
    
    pitches = [n.pitch for n in inst.notes]
    min_pitch = min(pitches)
    max_pitch = max(pitches)
    
    # 人間不可域
    if min_pitch < 40:  # E2未満
        warnings.append("guitar_pitch_too_low")
    if max_pitch > 84:  # C7超過
        warnings.append("guitar_pitch_too_high")
    
    return warnings
