#!/usr/bin/env python3
"""
Stage2 Instrument-Specific Metrics
楽器別Stage2メトリクス実装

Guitar: アルペジオ品質、コード検出、ストラムパターン
Bass: ルート音正確性、グルーヴ評価、音域適合性
Strings: ボウイング表現、ハーモニー評価、レガート品質
"""

import numpy as np
from collections import Counter, defaultdict
from typing import List, Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# Guitar Metrics
# ============================================================================

def calculate_arpeggio_quality(notes: List[Dict], config: Dict) -> float:
    """
    アルペジオパターン品質評価
    
    Args:
        notes: List of note dicts with 'pitch', 'start', 'velocity'
        config: arpeggio_quality config from YAML
    
    Returns:
        Score 0.0-1.0
    """
    if len(notes) < config.get('min_notes', 4):
        return 0.0
    
    # Sort by start time
    sorted_notes = sorted(notes, key=lambda n: n['start'])
    
    # Pattern consistency: 音高の変化方向を見る
    pitch_diffs = [sorted_notes[i+1]['pitch'] - sorted_notes[i]['pitch'] 
                   for i in range(len(sorted_notes)-1)]
    
    if not pitch_diffs:
        return 0.0
    
    # 上昇/下降/交互のパターン検出
    ascending = sum(1 for d in pitch_diffs if d > 0) / len(pitch_diffs)
    descending = sum(1 for d in pitch_diffs if d < 0) / len(pitch_diffs)
    
    pattern_consistency = max(ascending, descending)
    
    # Interval regularity: 音程間隔の規則性
    interval_std = np.std([abs(d) for d in pitch_diffs]) if len(pitch_diffs) > 1 else 0
    interval_regularity = max(0, 1.0 - interval_std / 12.0)  # 1オクターブで正規化
    
    # Timing precision: 時間間隔の規則性
    time_diffs = [sorted_notes[i+1]['start'] - sorted_notes[i]['start'] 
                  for i in range(len(sorted_notes)-1)]
    timing_std = np.std(time_diffs) if len(time_diffs) > 1 else 0
    timing_precision = max(0, 1.0 - timing_std / 0.5)  # 0.5秒で正規化
    
    # Weighted combination
    weights = config.get('weights', {})
    score = (
        pattern_consistency * weights.get('pattern_consistency', 0.4) +
        interval_regularity * weights.get('interval_regularity', 0.3) +
        timing_precision * weights.get('timing_precision', 0.3)
    )
    
    return min(1.0, max(0.0, score))


def calculate_chord_coherence(notes: List[Dict], config: Dict) -> float:
    """
    コード検出・和音評価
    
    Args:
        notes: List of note dicts
        config: chord_coherence config from YAML
    
    Returns:
        Score 0.0-1.0
    """
    if len(notes) < config.get('min_simultaneous_notes', 2):
        return 0.0
    
    # 同時発音ノートをグループ化(50ms以内)
    TIME_WINDOW = 0.05
    sorted_notes = sorted(notes, key=lambda n: n['start'])
    
    chords = []
    current_chord = [sorted_notes[0]]
    
    for note in sorted_notes[1:]:
        if note['start'] - current_chord[0]['start'] < TIME_WINDOW:
            current_chord.append(note)
        else:
            if len(current_chord) >= config.get('min_simultaneous_notes', 2):
                chords.append(current_chord)
            current_chord = [note]
    
    if len(current_chord) >= config.get('min_simultaneous_notes', 2):
        chords.append(current_chord)
    
    if not chords:
        return 0.0
    
    # 各コードの協和度を評価
    consonance_scores = []
    weights_by_interval = config.get('weights_by_interval', {})
    
    for chord in chords:
        pitches = sorted([n['pitch'] for n in chord])
        intervals = [(pitches[i+1] - pitches[i]) % 12 
                     for i in range(len(pitches)-1)]
        
        # Classify intervals
        consonance_intervals = config.get('consonance_intervals', {})
        perfect = consonance_intervals.get('perfect', [0, 7, 12])
        major = consonance_intervals.get('major', [4, 9])
        minor = consonance_intervals.get('minor', [3, 8])
        
        chord_score = 0
        for interval in intervals:
            if interval in perfect:
                chord_score += weights_by_interval.get('perfect', 1.0)
            elif interval in major:
                chord_score += weights_by_interval.get('major', 0.8)
            elif interval in minor:
                chord_score += weights_by_interval.get('minor', 0.7)
            else:
                chord_score += weights_by_interval.get('dissonant', 0.2)
        
        if intervals:
            consonance_scores.append(chord_score / len(intervals))
    
    return np.mean(consonance_scores) if consonance_scores else 0.0


def calculate_strumming_pattern(notes: List[Dict], config: Dict) -> float:
    """
    ストラムパターン評価
    
    Args:
        notes: List of note dicts
        config: strumming_pattern config from YAML
    
    Returns:
        Score 0.0-1.0
    """
    if len(notes) < config.get('min_notes_per_strum', 3):
        return 0.0
    
    # ストラム検出(短時間に複数音)
    sorted_notes = sorted(notes, key=lambda n: n['start'])
    max_strum_duration = config.get('max_strum_duration_ms', 100) / 1000.0
    
    strums = []
    current_strum = [sorted_notes[0]]
    
    for note in sorted_notes[1:]:
        if note['start'] - current_strum[0]['start'] < max_strum_duration:
            current_strum.append(note)
        else:
            if len(current_strum) >= config.get('min_notes_per_strum', 3):
                strums.append(current_strum)
            current_strum = [note]
    
    if len(current_strum) >= config.get('min_notes_per_strum', 3):
        strums.append(current_strum)
    
    if not strums:
        return 0.0
    
    # Pattern regularity: ストラム間隔の規則性
    strum_times = [s[0]['start'] for s in strums]
    if len(strum_times) < 2:
        return 0.5
    
    intervals = [strum_times[i+1] - strum_times[i] for i in range(len(strum_times)-1)]
    pattern_regularity = max(0, 1.0 - np.std(intervals) / np.mean(intervals)) if intervals else 0
    
    # Velocity consistency: 各ストラム内のベロシティ一貫性
    velocity_consistencies = []
    for strum in strums:
        velocities = [n['velocity'] for n in strum]
        if len(velocities) > 1:
            consistency = 1.0 - (np.std(velocities) / 64.0)  # MIDI range正規化
            velocity_consistencies.append(max(0, consistency))
    
    velocity_consistency = np.mean(velocity_consistencies) if velocity_consistencies else 0
    
    # Weighted combination
    weights = config.get('weights', {})
    score = (
        pattern_regularity * weights.get('pattern_regularity', 0.4) +
        velocity_consistency * weights.get('velocity_consistency', 0.3) +
        0.5 * weights.get('timing_tightness', 0.3)  # Placeholder
    )
    
    return min(1.0, max(0.0, score))


# ============================================================================
# Bass Metrics
# ============================================================================

def calculate_root_accuracy(notes: List[Dict], config: Dict) -> float:
    """
    ルート音正確性評価
    
    Args:
        notes: List of note dicts
        config: root_accuracy config from YAML
    
    Returns:
        Score 0.0-1.0
    """
    if not notes:
        return 0.0
    
    # ルート音検出(最低音)
    pitches = [n['pitch'] for n in notes]
    
    if config.get('octave_equivalence', True):
        pitch_classes = [p % 12 for p in pitches]
    else:
        pitch_classes = pitches
    
    # Stability: 最頻出音をルートとみなす
    pitch_counter = Counter(pitch_classes)
    most_common = pitch_counter.most_common(1)[0]
    root_pitch_class = most_common[0]
    root_frequency = most_common[1] / len(pitch_classes)
    
    # Frequency score
    frequency_score = min(1.0, root_frequency * 2.0)  # 50%以上で満点
    
    # Beat alignment bonus
    beat_aligned_roots = 0
    for note in notes:
        pitch_class = note['pitch'] % 12 if config.get('octave_equivalence', True) else note['pitch']
        if pitch_class == root_pitch_class:
            # ビート頭(0.25拍以内)ならボーナス
            beat_position = note['start'] % 1.0
            if beat_position < 0.25 or beat_position > 0.75:
                beat_aligned_roots += 1
    
    beat_alignment_bonus = config.get('beat_alignment_bonus', 0.15)
    beat_score = min(1.0, (beat_aligned_roots / len(notes)) * 2.0) * beat_alignment_bonus
    
    # Weighted combination
    weights = config.get('weights', {})
    score = (
        1.0 * weights.get('stability', 0.4) +  # Stability = root consistency
        frequency_score * weights.get('frequency', 0.3) +
        0.7 * weights.get('consonance', 0.3)  # Placeholder
    ) + beat_score
    
    return min(1.0, max(0.0, score))


def calculate_groove_quality(notes: List[Dict], config: Dict) -> float:
    """
    グルーヴ品質評価
    
    Args:
        notes: List of note dicts
        config: groove_quality config from YAML
    
    Returns:
        Score 0.0-1.0
    """
    if len(notes) < 4:
        return 0.0
    
    sorted_notes = sorted(notes, key=lambda n: n['start'])
    
    # Timing consistency: 音符間隔の規則性
    time_diffs = [sorted_notes[i+1]['start'] - sorted_notes[i]['start'] 
                  for i in range(len(sorted_notes)-1)]
    
    if not time_diffs:
        return 0.0
    
    # グリッド分割でquantize
    grid_divisions = config.get('grid_divisions', 16)
    grid_size = 1.0 / grid_divisions
    
    grid_deviations = []
    for diff in time_diffs:
        nearest_grid = round(diff / grid_size) * grid_size
        deviation = abs(diff - nearest_grid)
        grid_deviations.append(deviation)
    
    timing_consistency = max(0, 1.0 - np.mean(grid_deviations) / grid_size)
    
    # Swing detection
    if config.get('swing_detection', True) and len(time_diffs) >= 2:
        # 偶数・奇数番目の間隔比を見る
        even_diffs = [time_diffs[i] for i in range(0, len(time_diffs), 2)]
        odd_diffs = [time_diffs[i] for i in range(1, len(time_diffs), 2)]
        
        if even_diffs and odd_diffs:
            swing_ratio = np.mean(even_diffs) / (np.mean(even_diffs) + np.mean(odd_diffs))
            swing_range = config.get('swing_ratio_range', {})
            min_swing = swing_range.get('min', 0.52)
            max_swing = swing_range.get('max', 0.68)
            
            if min_swing <= swing_ratio <= max_swing:
                swing_feel = 1.0
            else:
                swing_feel = max(0, 1.0 - abs(swing_ratio - 0.6) / 0.2)
        else:
            swing_feel = 0.5
    else:
        swing_feel = 0.5
    
    # Density balance
    total_duration = sorted_notes[-1]['start'] - sorted_notes[0]['start']
    if total_duration > 0:
        density = len(notes) / total_duration
        # TODO: density_bandsでBPM適応評価
        density_balance = 0.7  # Placeholder
    else:
        density_balance = 0.0
    
    # Weighted combination
    weights = config.get('weights', {})
    score = (
        timing_consistency * weights.get('timing_consistency', 0.35) +
        swing_feel * weights.get('swing_feel', 0.25) +
        0.5 * weights.get('syncopation', 0.20) +  # Placeholder
        density_balance * weights.get('density_balance', 0.20)
    )
    
    return min(1.0, max(0.0, score))


def calculate_pitch_range_fit(notes: List[Dict], config: Dict) -> float:
    """
    音域適合性評価
    
    Args:
        notes: List of note dicts
        config: pitch_range_fit config from YAML
    
    Returns:
        Score 0.0-1.0
    """
    if not notes:
        return 0.0
    
    pitches = [n['pitch'] for n in notes]
    
    optimal_range = config.get('optimal_range', {})
    optimal_min = optimal_range.get('min', 28)  # E1
    optimal_max = optimal_range.get('max', 60)  # C4
    
    # Optimal coverage: 最適音域内の割合
    optimal_notes = sum(1 for p in pitches if optimal_min <= p <= optimal_max)
    optimal_coverage = optimal_notes / len(pitches)
    
    # Range concentration: 音域の集中度(分散が小さいほど良い)
    pitch_std = np.std(pitches)
    range_concentration = max(0, 1.0 - pitch_std / 12.0)  # 1オクターブで正規化
    
    # Outlier penalty: 極端な外れ値のペナルティ
    extended_range = config.get('extended_range', {})
    extended_min = extended_range.get('min', 24)
    extended_max = extended_range.get('max', 67)
    
    outliers = sum(1 for p in pitches if p < extended_min or p > extended_max)
    outlier_penalty = outliers / len(pitches)
    
    # Weighted combination
    weights = config.get('weights', {})
    score = (
        optimal_coverage * weights.get('optimal_coverage', 0.50) +
        range_concentration * weights.get('range_concentration', 0.30) +
        (1.0 - outlier_penalty) * weights.get('outlier_penalty', 0.20)
    )
    
    return min(1.0, max(0.0, score))


# ============================================================================
# Strings Metrics
# ============================================================================

def calculate_bowing_expression(notes: List[Dict], config: Dict) -> float:
    """
    ボウイング表現評価(velocity変化)
    
    Args:
        notes: List of note dicts
        config: bowing_expression config from YAML
    
    Returns:
        Score 0.0-1.0
    """
    if len(notes) < config.get('min_notes', 4):
        return 0.0
    
    velocities = [n['velocity'] for n in notes]
    
    # Dynamic range: ダイナミックレンジ
    vel_min = min(velocities)
    vel_max = max(velocities)
    dynamic_range = vel_max - vel_min
    
    optimal_min = config.get('dynamic_range', {}).get('optimal_min', 40)
    optimal_max = config.get('dynamic_range', {}).get('optimal_max', 110)
    optimal_range = optimal_max - optimal_min
    
    if optimal_range > 0:
        range_score = min(1.0, dynamic_range / optimal_range)
    else:
        range_score = 0.0
    
    # Crescendo/Decrescendo detection
    sorted_notes = sorted(notes, key=lambda n: n['start'])
    sorted_velocities = [n['velocity'] for n in sorted_notes]
    
    # 移動平均で平滑化
    window = 3
    if len(sorted_velocities) >= window:
        smoothed = np.convolve(sorted_velocities, np.ones(window)/window, mode='valid')
        
        # 傾向検出
        diffs = np.diff(smoothed)
        increasing = sum(1 for d in diffs if d > 0)
        decreasing = sum(1 for d in diffs if d < 0)
        
        crescendo_score = increasing / len(diffs) if diffs.size > 0 else 0
        decrescendo_score = decreasing / len(diffs) if diffs.size > 0 else 0
        transition_score = max(crescendo_score, decrescendo_score)
    else:
        transition_score = 0.0
    
    # Attack variation: ベロシティの多様性
    velocity_std = np.std(velocities)
    attack_variation = min(1.0, velocity_std / 32.0)  # 32で正規化
    
    # Smoothness: 急激な変化を避ける
    if len(sorted_velocities) > 1:
        velocity_diffs = np.diff(sorted_velocities)
        smoothness_threshold = config.get('smoothness_threshold_ms', 50) / 1000.0
        large_jumps = sum(1 for d in velocity_diffs if abs(d) > 30)
        smoothness = max(0, 1.0 - large_jumps / len(velocity_diffs))
    else:
        smoothness = 1.0
    
    # Weighted combination
    weights = config.get('weights', {})
    score = (
        range_score * weights.get('dynamic_range', 0.35) +
        transition_score * weights.get('crescendo_decrescendo', 0.30) +
        attack_variation * weights.get('attack_variation', 0.20) +
        smoothness * weights.get('smoothness', 0.15)
    )
    
    return min(1.0, max(0.0, score))


def calculate_harmony_quality(notes: List[Dict], config: Dict) -> float:
    """
    ハーモニー品質評価
    
    Args:
        notes: List of note dicts
        config: harmony_quality config from YAML
    
    Returns:
        Score 0.0-1.0
    """
    # Guitar用のchord_coherenceと類似だが、Strings用に調整
    return calculate_chord_coherence(notes, config)


def calculate_legato_quality(notes: List[Dict], config: Dict) -> float:
    """
    レガート品質評価
    
    Args:
        notes: List of note dicts
        config: legato_quality config from YAML
    
    Returns:
        Score 0.0-1.0
    """
    if len(notes) < 2:
        return 0.0
    
    sorted_notes = sorted(notes, key=lambda n: n['start'])
    
    max_gap_ms = config.get('max_gap_ms', 50) / 1000.0
    min_overlap_ms = config.get('min_overlap_ms', 10) / 1000.0
    
    connected = 0
    overlaps = []
    
    for i in range(len(sorted_notes) - 1):
        note1 = sorted_notes[i]
        note2 = sorted_notes[i + 1]
        
        # note1の終了時刻を計算(durationがあれば)
        if 'duration' in note1:
            note1_end = note1['start'] + note1['duration']
        else:
            # durationがない場合は次の音までとみなす
            note1_end = note2['start']
        
        gap = note2['start'] - note1_end
        
        if gap <= max_gap_ms:
            connected += 1
            if gap < 0:  # オーバーラップ
                overlaps.append(abs(gap))
    
    # Connection rate
    connection_rate = connected / (len(sorted_notes) - 1) if len(sorted_notes) > 1 else 0
    
    # Overlap consistency
    if overlaps:
        overlap_std = np.std(overlaps)
        overlap_consistency = max(0, 1.0 - overlap_std / min_overlap_ms)
    else:
        overlap_consistency = 0.5  # オーバーラップなしは中間評価
    
    # Duration balance: 音価のバランス
    if 'duration' in sorted_notes[0]:
        durations = [n.get('duration', 0.5) for n in sorted_notes]
        duration_std = np.std(durations)
        duration_balance = max(0, 1.0 - duration_std / np.mean(durations)) if durations else 0
    else:
        duration_balance = 0.5
    
    # Weighted combination
    weights = config.get('weights', {})
    score = (
        connection_rate * weights.get('connection_rate', 0.40) +
        overlap_consistency * weights.get('overlap_consistency', 0.30) +
        duration_balance * weights.get('duration_balance', 0.30)
    )
    
    return min(1.0, max(0.0, score))


# ============================================================================
# Main Interface
# ============================================================================

def calculate_instrument_metrics(
    instrument: str,
    notes: List[Dict],
    config: Dict
) -> Dict[str, float]:
    """
    楽器別メトリクス計算のメインインターフェース
    
    Args:
        instrument: 'guitar', 'bass', or 'strings'
        notes: List of note dicts
        config: Instrument-specific config from YAML
    
    Returns:
        Dict of metric scores
    """
    scores = {}
    
    if instrument == 'guitar':
        if 'arpeggio_quality' in config:
            scores['arpeggio_quality'] = calculate_arpeggio_quality(
                notes, config['arpeggio_quality']
            )
        if 'chord_coherence' in config:
            scores['chord_coherence'] = calculate_chord_coherence(
                notes, config['chord_coherence']
            )
        if 'strumming_pattern' in config:
            scores['strumming_pattern'] = calculate_strumming_pattern(
                notes, config['strumming_pattern']
            )
    
    elif instrument == 'bass':
        if 'root_accuracy' in config:
            scores['root_accuracy'] = calculate_root_accuracy(
                notes, config['root_accuracy']
            )
        if 'groove_quality' in config:
            scores['groove_quality'] = calculate_groove_quality(
                notes, config['groove_quality']
            )
        if 'pitch_range_fit' in config:
            scores['pitch_range_fit'] = calculate_pitch_range_fit(
                notes, config['pitch_range_fit']
            )
    
    elif instrument == 'strings':
        if 'bowing_expression' in config:
            scores['bowing_expression'] = calculate_bowing_expression(
                notes, config['bowing_expression']
            )
        if 'harmony_quality' in config:
            scores['harmony_quality'] = calculate_harmony_quality(
                notes, config['harmony_quality']
            )
        if 'legato_quality' in config:
            scores['legato_quality'] = calculate_legato_quality(
                notes, config['legato_quality']
            )
    
    else:
        logger.warning(f"Unknown instrument: {instrument}")
    
    return scores


if __name__ == "__main__":
    # テストコード
    import yaml
    
    # Guitar test
    guitar_notes = [
        {'pitch': 64, 'start': 0.0, 'velocity': 80},
        {'pitch': 67, 'start': 0.25, 'velocity': 82},
        {'pitch': 71, 'start': 0.5, 'velocity': 78},
        {'pitch': 76, 'start': 0.75, 'velocity': 85},
    ]
    
    guitar_config = {
        'arpeggio_quality': {
            'min_notes': 3,
            'weights': {
                'pattern_consistency': 0.4,
                'interval_regularity': 0.3,
                'timing_precision': 0.3
            }
        }
    }
    
    guitar_scores = calculate_instrument_metrics('guitar', guitar_notes, guitar_config)
    print(f"Guitar scores: {guitar_scores}")
    
    # Bass test
    bass_notes = [
        {'pitch': 40, 'start': 0.0, 'velocity': 90},  # E1
        {'pitch': 40, 'start': 0.5, 'velocity': 88},
        {'pitch': 45, 'start': 1.0, 'velocity': 85},
        {'pitch': 40, 'start': 1.5, 'velocity': 92},
    ]
    
    bass_config = {
        'root_accuracy': {
            'octave_equivalence': True,
            'beat_alignment_bonus': 0.15,
            'weights': {
                'stability': 0.4,
                'frequency': 0.3,
                'consonance': 0.3
            }
        }
    }
    
    bass_scores = calculate_instrument_metrics('bass', bass_notes, bass_config)
    print(f"Bass scores: {bass_scores}")
