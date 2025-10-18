#!/usr/bin/env python3
"""
ドラムパターン抽出スクリプト（Stage2 - Phase 2強化版）

SLAKH/LAMDA等のMIDIファイルからドラムパートを解析し、
高品質なリズムパターンを抽出してPickleファイルに保存します。

Phase 2強化:
- BPM帯で層化（60-90/90-120/120-150/150-180）
- 拍子別分類（4/4, 3/4, 6/8）
- 品質フィルタ強化（on-beat率、ゴースト比率、密度レンジ）
- 目標: 最低1,000件、理想3,000件

使用方法:
    # SLAKH から抽出
    python scripts/extract_drum_patterns.py \\
      --input-dir data/midi/slakh \\
      --output data/patterns/stage2_drums.pickle \\
      --min-bars 4 \\
      --max-bars 8 \\
      --min-quality 0.6
    
    # LAMDA から抽出
    python scripts/extract_drum_patterns.py \\
      --input-dir data/midi/lamda \\
      --output data/patterns/stage2_drums_lamda.pickle \\
      --min-bars 2 \\
      --max-bars 4
"""

import argparse
import music21
from music21 import stream, note, converter
from music21 import note as m21note, chord as m21chord
import numpy as np
import pickle
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import sys
sys.path.append(str(Path(__file__).parent.parent))

from generator.drums_generator_stage2 import DrumPattern, GM_DRUM_MAP

import logging
logger = logging.getLogger(__name__)


def _safe_velocity(el, default=96):
    """要素からvelocityを安全に取得"""
    v = getattr(getattr(el, "volume", None), "velocity", None)
    return int(v) if v is not None else default


def _unpitched_midi(el, fallback=35):
    """
    music21.note.Unpitched から MIDI ノート番号を安全に取得。
    取れるものを優先順に: .midi -> .pitch.ps -> fallback
    """
    midi_num = getattr(el, "midi", None)
    if midi_num is not None:
        return int(midi_num)
    # 一部バージョンでは el.pitch が無い個体がある
    p = getattr(el, "pitch", None)
    if p is not None:
        ps = getattr(p, "ps", None)
        if ps is not None:
            return int(ps)
    return int(fallback)


def iter_drum_midi_events_m21(s: stream.Stream):
    """
    music21 Stream からドラム系イベントを安全に列挙。
    戻り: (offset, quarterLength, midi, velocity)
    - Note           -> el.pitch.midi
    - Chord          -> el.pitches（無ければ el.notes）を展開
    - Unpitched      -> _unpitched_midi() で安全取得
    Rest は除外、qlen<=0 は後段で除外してください。
    """
    for el in s.flat.notesAndRests:
        if isinstance(el, m21note.Rest):
            continue

        if isinstance(el, m21note.Note):
            yield el.offset, el.duration.quarterLength, int(el.pitch.midi), _safe_velocity(el)
            continue

        if isinstance(el, m21chord.Chord):
            # 通常は .pitches に Pitch が入る。無ければ .notes 経由で各音を確認
            pitches = list(getattr(el, "pitches", []))
            if pitches:
                for p in pitches:
                    yield el.offset, el.duration.quarterLength, int(p.midi), _safe_velocity(el)
            else:
                for n in getattr(el, "notes", []):
                    if isinstance(n, m21note.Unpitched):
                        midi_num = _unpitched_midi(n)
                    elif isinstance(n, m21note.Note):
                        midi_num = int(n.pitch.midi)
                    else:
                        continue
                    yield el.offset, el.duration.quarterLength, midi_num, _safe_velocity(n)
            continue

        if isinstance(el, m21note.Unpitched):
            midi_num = _unpitched_midi(el)
            yield el.offset, el.duration.quarterLength, midi_num, _safe_velocity(el)
            continue

# Phase 2強化: 品質ゲート定数
MIN_KICK_ONBEAT_RATIO = 0.6  # キックの拍頭率（最低60%）
MAX_GHOST_NOTE_RATIO = 0.3   # ゴーストノート率（最大30%）
MIN_DENSITY = 4.0             # 最小密度（4 hits/bar）
MAX_DENSITY = 32.0            # 最大密度（32 hits/bar）

# BPM層化
BPM_RANGES = [
    (60, 90, "slow"),
    (90, 120, "medium"),
    (120, 150, "fast"),
    (150, 180, "very_fast")
]


def is_drum_note(pitch: int) -> bool:
    """MIDIピッチがドラム音か判定"""
    # GMドラムマップ範囲: 27-87
    return 27 <= pitch <= 87


def classify_drum_hit(pitch: int) -> Optional[str]:
    """ドラムヒットを分類"""
    for drum_type, pitches in GM_DRUM_MAP.items():
        if pitch in pitches:
            return drum_type
    
    # その他のドラム音
    if 27 <= pitch <= 87:
        return 'other'
    
    return None


def extract_drum_hits_from_part(
    part: music21.stream.Part,
    bars: int
) -> Dict[str, Tuple[List[float], List[int]]]:
    """
    ドラムパートからヒット位置とベロシティを抽出
    
    Returns:
        {drum_type: ([positions], [velocities])}
    """
    hits = {
        'kick': ([], []),
        'snare': ([], []),
        'hihat_closed': ([], []),
        'hihat_open': ([], []),
        'crash': ([], []),
        'ride': ([], [])
    }
    
    # 安全イテレータで全イベントを取得
    for offset, qlen, midi, velocity in iter_drum_midi_events_m21(part):
        # 負/ゼロ長は除外（極小音符の丸め誤差対応）
        if qlen < 1e-6:
            continue
        
        drum_type = classify_drum_hit(midi)
        
        if drum_type and drum_type in hits:
            # 小節内の相対位置（0.0-4.0）
            relative_offset = float(offset % 4.0)
            
            hits[drum_type][0].append(relative_offset)
            hits[drum_type][1].append(velocity)
    
    return hits


def calculate_pattern_metrics(hits: Dict[str, Tuple[List[float], List[int]]]) -> Dict[str, float]:
    """
    パターンのメトリクスを計算（Phase 2強化）
    
    Returns:
        - density: 1小節あたりヒット数
        - complexity: ヒットタイプ数 / 6
        - syncopation_rate: オフビート率
        - kick_onbeat_ratio: キックの拍頭率（品質指標）
        - ghost_note_ratio: ゴーストノート率（vel < 40）
        - quality_score: 総合品質スコア（0.0-1.0）
    """
    total_hits = sum(len(positions) for positions, _ in hits.values())
    
    # 密度（1小節あたりヒット数）
    density = total_hits / 4.0 if total_hits > 0 else 0.0
    
    # 複雑度（ヒットタイプ数 / 6）
    active_types = sum(1 for positions, _ in hits.values() if len(positions) > 0)
    complexity = active_types / 6.0
    
    # シンコペーション率 & キックの拍頭率（Phase 2強化）
    on_beat_hits = 0
    off_beat_hits = 0
    kick_onbeat_count = 0
    kick_total = 0
    
    for drum_type in ['kick', 'snare']:
        positions, _ = hits[drum_type]
        for pos in positions:
            # 拍ちょうど（0.0, 1.0, 2.0, 3.0）ならオンビート
            if abs(pos - round(pos)) < 0.1:
                on_beat_hits += 1
                if drum_type == 'kick':
                    kick_onbeat_count += 1
            else:
                off_beat_hits += 1
            
            if drum_type == 'kick':
                kick_total += 1
    
    total_beats = on_beat_hits + off_beat_hits
    syncopation_rate = off_beat_hits / total_beats if total_beats > 0 else 0.0
    kick_onbeat_ratio = kick_onbeat_count / kick_total if kick_total > 0 else 0.0
    
    # ゴーストノート率（Phase 2強化）
    ghost_count = 0
    total_notes = 0
    
    for drum_type in ['kick', 'snare', 'hihat_closed']:
        positions, velocities = hits[drum_type]
        for vel in velocities:
            total_notes += 1
            if vel < 40:
                ghost_count += 1
    
    ghost_note_ratio = ghost_count / total_notes if total_notes > 0 else 0.0
    
    # 品質スコア計算（Phase 2強化）
    quality_score = 0.0
    
    # キックの拍頭率（重要）
    if kick_onbeat_ratio >= MIN_KICK_ONBEAT_RATIO:
        quality_score += 0.4
    else:
        quality_score += 0.4 * (kick_onbeat_ratio / MIN_KICK_ONBEAT_RATIO)
    
    # 密度範囲（適切な範囲内か）
    if MIN_DENSITY <= density <= MAX_DENSITY:
        quality_score += 0.3
    elif density < MIN_DENSITY:
        quality_score += 0.3 * (density / MIN_DENSITY)
    else:  # density > MAX_DENSITY
        quality_score += 0.3 * (MAX_DENSITY / density)
    
    # ゴーストノート率（少なめが良い）
    if ghost_note_ratio <= MAX_GHOST_NOTE_RATIO:
        quality_score += 0.2
    else:
        quality_score += 0.2 * (MAX_GHOST_NOTE_RATIO / ghost_note_ratio)
    
    # 複雑度（多様性）
    quality_score += 0.1 * complexity
    
    return {
        'density': density,
        'complexity': complexity,
        'syncopation_rate': syncopation_rate,
        'kick_onbeat_ratio': kick_onbeat_ratio,
        'ghost_note_ratio': ghost_note_ratio,
        'quality_score': quality_score
    }


def estimate_tempo_from_score(score) -> float:
    """
    スコアからテンポを推定（Phase 2強化）
    
    Returns:
        推定BPM（デフォルト120.0）
    """
    # MetronomeMarkを探す
    for element in score.flatten():
        if isinstance(element, music21.tempo.MetronomeMark):
            return float(element.number)
    
    # デフォルト
    return 120.0


def classify_bpm_range(bpm: float) -> str:
    """BPMを層化分類（Phase 2強化）"""
    for min_bpm, max_bpm, label in BPM_RANGES:
        if min_bpm <= bpm < max_bpm:
            return label
    
    # 範囲外
    if bpm < 60:
        return "very_slow"
    else:
        return "extreme_fast"


def extract_patterns_from_midi(
    midi_path: Path,
    min_bars: int = 4,
    max_bars: int = 8,
    min_quality: float = 0.6
) -> List[DrumPattern]:
    """
    MIDIファイルからドラムパターンを抽出（Phase 2強化）
    
    Args:
        midi_path: MIDIファイルパス
        min_bars: 最小小節数
        max_bars: 最大小節数
        min_quality: 最小品質スコア（0.0-1.0）
    
    Returns:
        抽出されたパターンリスト（品質フィルタ済み）
    """
    try:
        score = converter.parse(midi_path)
    except Exception as e:
        print(f"⚠️  Failed to parse {midi_path}: {e}")
        return []
    
    # テンポ推定（Phase 2強化）
    estimated_tempo = estimate_tempo_from_score(score)
    bpm_category = classify_bpm_range(estimated_tempo)
    
    patterns = []
    
    # ドラムパートを探す
    for part in score.parts:
        # Channel 10 (GMドラムチャンネル) またはパーカッション楽器
        is_drum_part = False
        
        if hasattr(part, 'getInstrument'):
            inst = part.getInstrument()
            if isinstance(inst, music21.instrument.Percussion):
                is_drum_part = True
        
        # ノートチェック - 安全イテレータでドラム音を確認
        drum_events = []
        total_events = 0
        for offset, qlen, midi, velocity in iter_drum_midi_events_m21(part):
            if qlen < 1e-6:
                continue
            total_events += 1
            if is_drum_note(midi):
                drum_events.append((offset, qlen, midi, velocity))
        
        if total_events == 0:
            continue
        
        # ドラム音が80%以上ならドラムパート
        if len(drum_events) / total_events > 0.8:
            is_drum_part = True
        
        if not is_drum_part:
            continue
        
        # パターン抽出
        total_duration = part.duration.quarterLength
        total_bars = int(total_duration / 4.0)
        
        for start_bar in range(0, total_bars, min_bars):
            bars = min(max_bars, total_bars - start_bar)
            if bars < min_bars:
                break
            
            # 部分抽出
            start_offset = start_bar * 4.0
            end_offset = start_offset + bars * 4.0
            
            sub_part = part.measures(start_bar + 1, start_bar + bars)
            
            # ヒット抽出
            hits = extract_drum_hits_from_part(sub_part, bars)
            
            # メトリクス計算
            metrics = calculate_pattern_metrics(hits)
            
            # Phase 2強化: 品質フィルタ
            if metrics['quality_score'] < min_quality:
                continue  # 品質不足をスキップ
            
            # パターン作成（Phase 2強化：テンポ・品質スコア含む）
            pattern = DrumPattern(
                id=f"{midi_path.stem}_{bpm_category}_bar{start_bar}_{bars}bars",
                instrument='drums',
                technique=bpm_category,  # BPMカテゴリを技法として記録
                tempo=estimated_tempo,  # Phase 2: 推定テンポ
                bars=bars,
                emotion='neutral_medium',  # デフォルト
                
                kick_hits=hits['kick'][0],
                snare_hits=hits['snare'][0],
                hihat_hits=hits['hihat_closed'][0],
                crash_hits=hits['crash'][0],
                ride_hits=hits['ride'][0],
                
                kick_velocities=hits['kick'][1],
                snare_velocities=hits['snare'][1],
                hihat_velocities=hits['hihat_closed'][1],
                crash_velocities=hits['crash'][1],
                ride_velocities=hits['ride'][1],
                
                density=metrics['density'],
                complexity=metrics['complexity'],
                syncopation_rate=metrics['syncopation_rate'],
                
                quality_score=metrics['quality_score']  # Phase 2: 計算済み
            )
            
            patterns.append(pattern)
    
    return patterns


def main():
    parser = argparse.ArgumentParser(
        description='Extract drum patterns from MIDI files (Phase 2 Enhanced)'
    )
    parser.add_argument('--input-dir', type=Path, required=True,
                       help='Input directory containing MIDI files')
    parser.add_argument('--output', type=Path, required=True,
                       help='Output pickle file')
    parser.add_argument('--min-bars', type=int, default=4,
                       help='Minimum pattern length in bars')
    parser.add_argument('--max-bars', type=int, default=8,
                       help='Maximum pattern length in bars')
    parser.add_argument('--min-quality', type=float, default=0.6,
                       help='Minimum quality score (0.0-1.0, default: 0.6)')
    parser.add_argument('--limit', type=int, default=None,
                       help='Limit number of MIDI files to process')
    parser.add_argument('--target', type=int, default=1000,
                       help='Target number of patterns (minimum: 1000, ideal: 3000)')
    
    args = parser.parse_args()
    
    if not args.input_dir.exists():
        print(f"❌ Input directory not found: {args.input_dir}")
        return 1
    
    # Phase 2強化: ヘッダー
    print("\n" + "=" * 60)
    print("🥁 Drum Pattern Extraction (Phase 2 Enhanced)")
    print("=" * 60)
    print(f"Input dir:     {args.input_dir}")
    print(f"Output:        {args.output}")
    print(f"Pattern length: {args.min_bars}-{args.max_bars} bars")
    print(f"Min quality:   {args.min_quality:.2f}")
    print(f"Target:        {args.target} patterns")
    print()
    
    # MIDI ファイル収集
    midi_files = list(args.input_dir.rglob('*.mid')) + list(args.input_dir.rglob('*.midi'))
    
    if args.limit:
        midi_files = midi_files[:args.limit]
    
    print(f"📂 Found {len(midi_files)} MIDI files")
    print(f"🔧 Extracting patterns with quality filter...")
    print()
    
    all_patterns = []
    skipped_low_quality = 0
    
    for i, midi_path in enumerate(midi_files, 1):
        print(f"[{i}/{len(midi_files)}] {midi_path.name}...", end=' ')
        
        patterns = extract_patterns_from_midi(
            midi_path,
            min_bars=args.min_bars,
            max_bars=args.max_bars,
            min_quality=args.min_quality
        )
        
        all_patterns.extend(patterns)
        print(f"{len(patterns)} patterns")
        
        # 目標達成チェック
        if len(all_patterns) >= args.target:
            print(f"\n✅ Target reached: {len(all_patterns)} patterns")
            break
    
    print(f"\n" + "=" * 60)
    print(f"✅ Extracted {len(all_patterns)} total patterns")
    
    # Phase 2強化: BPM分布
    if all_patterns:
        bpm_dist = {}
        for p in all_patterns:
            category = classify_bpm_range(p.tempo)
            bpm_dist[category] = bpm_dist.get(category, 0) + 1
        
        print(f"\n📊 BPM Distribution:")
        for category, count in sorted(bpm_dist.items()):
            print(f"   {category:12s}: {count:4d} patterns")
    
    # 保存
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'wb') as f:
        pickle.dump(all_patterns, f)
    
    print(f"\n💾 Saved to: {args.output}")
    
    # Phase 2強化: 統計詳細
    if all_patterns:
        densities = [p.density for p in all_patterns]
        complexities = [p.complexity for p in all_patterns]
        quality_scores = [p.quality_score for p in all_patterns]
        
        print(f"\n📊 Quality Statistics:")
        print(f"   Density:    {np.mean(densities):.2f} ± {np.std(densities):.2f} hits/bar")
        print(f"   Complexity: {np.mean(complexities):.2f} ± {np.std(complexities):.2f}")
        print(f"   Quality:    {np.mean(quality_scores):.2f} ± {np.std(quality_scores):.2f}")
        
        # 目標達成状況
        if len(all_patterns) >= args.target:
            print(f"\n✅ Target achieved: {len(all_patterns)}/{args.target} patterns")
        else:
            shortage = args.target - len(all_patterns)
            print(f"\n⚠️  Below target: {len(all_patterns)}/{args.target} patterns ({shortage} short)")
            print(f"   Consider: --min-quality {args.min_quality - 0.1:.1f} or process more MIDI files")
    
    print("=" * 60)
    
    return 0


if __name__ == '__main__':
    exit(main())
