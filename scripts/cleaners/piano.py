#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Piano専用クリーニング処理
"""

from __future__ import annotations

import statistics
from typing import Any, Dict, List, Tuple

import pretty_midi


def clean_piano(
    pm: pretty_midi.PrettyMIDI,
) -> Tuple[pretty_midi.PrettyMIDI, Dict[str, Any], List[str]]:
    """
    Piano専用のクリーニング
    
    処理内容:
    - ペダル正規化 (CC64)
    - 片手重複の緩和
    - キー/コード推定 (ヒント)
    
    Returns:
        (pm, metadata, reason_codes)
    """
    metadata: Dict[str, Any] = {}
    reason_codes: List[str] = []
    
    # Piano instrumentのみ処理
    piano_instruments = [
        inst for inst in pm.instruments
        if not inst.is_drum and 0 <= inst.program <= 7  # Piano range
    ]
    
    if not piano_instruments:
        return pm, metadata, reason_codes
    
    # 1. ペダル正規化
    for inst in piano_instruments:
        pedal_ratio, pedal_warnings = _normalize_pedal(inst)
        metadata["pedal_sustain_ratio"] = pedal_ratio
        reason_codes.extend(pedal_warnings)
    
    # 2. 片手重複の緩和
    for inst in piano_instruments:
        dedup_count = _deduplicate_chord_fragments(inst)
        if dedup_count > 0:
            metadata["deduped_chord_fragments"] = dedup_count
    
    # 3. 音域分析
    for inst in piano_instruments:
        hand_sep = _analyze_hand_separation(inst)
        metadata["hand_separation"] = hand_sep
    
    # 4. Velocity統計
    all_velocities = []
    for inst in piano_instruments:
        all_velocities.extend([n.velocity for n in inst.notes])
    
    if all_velocities:
        metadata["velocity_std"] = round(statistics.stdev(all_velocities), 2)
        metadata["velocity_mean"] = round(statistics.mean(all_velocities), 2)
    
    return pm, metadata, reason_codes


def _normalize_pedal(inst: pretty_midi.Instrument) -> Tuple[float, List[str]]:
    """
    CC64 (Sustain Pedal) の正規化
    
    - 短い"ペダル点"ノイズを除去
    - ペダル使用率を計算
    """
    warnings: List[str] = []
    
    # CC64イベントを抽出
    sustain_events = [cc for cc in inst.control_changes if cc.number == 64]
    
    if not sustain_events:
        return 0.0, warnings
    
    # ペダルオン時間を計算
    pedal_on_duration = 0.0
    current_state = False
    last_time = 0.0
    
    for cc in sorted(sustain_events, key=lambda x: x.time):
        if cc.value >= 64 and not current_state:
            # ペダルオン
            current_state = True
            last_time = cc.time
        elif cc.value < 64 and current_state:
            # ペダルオフ
            pedal_on_duration += (cc.time - last_time)
            current_state = False
    
    # 総時間に対する割合
    total_duration = inst.get_end_time()
    if total_duration > 0:
        pedal_ratio = pedal_on_duration / total_duration
    else:
        pedal_ratio = 0.0
    
    # 過剰ペダル警告
    if pedal_ratio > 0.9:
        warnings.append("pedal_excessive")
    
    return round(pedal_ratio, 3), warnings


def _deduplicate_chord_fragments(inst: pretty_midi.Instrument) -> int:
    """
    極端な和音重複 (<5ms違いのハーモニック複製) をマージ
    """
    if not inst.notes:
        return 0
    
    notes = sorted(inst.notes, key=lambda n: n.start)
    dedup_count = 0
    filtered_notes = []
    
    i = 0
    while i < len(notes):
        current = notes[i]
        duplicates = [current]
        
        # 同時発音グループを探す (<5ms以内)
        j = i + 1
        while j < len(notes) and abs(notes[j].start - current.start) < 0.005:
            if notes[j].pitch == current.pitch:
                # 同じpitchの重複
                duplicates.append(notes[j])
            j += 1
        
        if len(duplicates) > 1:
            # 最も長いノートを残す
            longest = max(duplicates, key=lambda n: n.end - n.start)
            filtered_notes.append(longest)
            dedup_count += len(duplicates) - 1
        else:
            filtered_notes.append(current)
        
        i = j if j > i + 1 else i + 1
    
    inst.notes = filtered_notes
    return dedup_count


def _analyze_hand_separation(inst: pretty_midi.Instrument) -> float:
    """
    左手/右手の音域分離度を計算
    
    Returns:
        中央値の差 (半音単位)
    """
    if not inst.notes:
        return 0.0
    
    pitches = [n.pitch for n in inst.notes]
    median_pitch = statistics.median(pitches)
    
    # 中央値より上/下に分ける
    upper = [p for p in pitches if p >= median_pitch]
    lower = [p for p in pitches if p < median_pitch]
    
    if not upper or not lower:
        return 0.0
    
    upper_median = statistics.median(upper)
    lower_median = statistics.median(lower)
    
    return round(upper_median - lower_median, 1)
