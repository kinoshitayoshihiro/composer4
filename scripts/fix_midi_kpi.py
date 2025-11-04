#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MIDI KPI Auto-Fix Script
=========================

KPI Gate失敗小節をMIDIレベルで自動修正します。

修正項目:
1. 過密（density/notes_per_bar超過）: Hat間引き（グリッドから遠いノートを削除）
2. backbeat低下: Snare Velocity +10%調整
3. テンポ外れ: tempo_bpm範囲外の警告（修正なし）

使用例:
    python3 scripts/fix_midi_kpi.py \
        --midi drums.mid \
        --gate-config configs/gate_prod.yaml \
        --output drums_fixed.mid \
        --tempo-bpm 120
"""

import argparse
import json
import yaml
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np


def load_gate_config(yaml_path: Path) -> dict:
    """gate_prod.yaml読み込み"""
    with open(yaml_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def fix_density_overage(bar_notes, max_density, tempo_bpm):
    """
    Hat密度超過を修正（グリッドから遠いノートを削除）
    
    Args:
        bar_notes: 小節内のノートリスト
        max_density: 最大Hat密度
        tempo_bpm: テンポ
    
    Returns:
        修正後のノートリスト
    """
    try:
        import pretty_midi
    except ImportError:
        raise ImportError("pretty_midi required. Install: pip install pretty_midi")
    
    # Hat抽出（42=Closed, 44=Pedal, 46=Open）
    hat_notes = [n for n in bar_notes if n.pitch in (42, 44, 46)]
    other_notes = [n for n in bar_notes if n.pitch not in (42, 44, 46)]
    
    if len(hat_notes) <= max_density:
        return bar_notes  # 修正不要
    
    # 削除数
    to_remove = len(hat_notes) - int(max_density)
    
    # グリッド（8分音符）から最も遠いノートを特定
    sec_per_beat = 60.0 / tempo_bpm
    eighth_grid = sec_per_beat / 2.0
    
    bar_start = min(n.start for n in bar_notes)
    
    # 各Hatのグリッド偏差計算
    deviations = []
    for n in hat_notes:
        relative_time = n.start - bar_start
        nearest_grid = round(relative_time / eighth_grid) * eighth_grid
        deviation = abs(relative_time - nearest_grid)
        deviations.append((deviation, n))
    
    # 偏差が大きい順にソート（グリッドから遠いノート）
    deviations.sort(key=lambda x: -x[0])
    
    # 削除対象
    to_remove_notes = set(d[1] for d in deviations[:to_remove])
    
    # 残すHat
    kept_hats = [n for n in hat_notes if n not in to_remove_notes]
    
    return other_notes + kept_hats


def fix_backbeat_weakness(bar_notes, min_backbeat, target_increase=1.1):
    """
    Snare Velocity調整でbackbeat強化
    
    Args:
        bar_notes: 小節内のノートリスト
        min_backbeat: 最小backbeat_strength
        target_increase: Velocity増幅率（デフォルト1.1 = +10%）
    
    Returns:
        修正後のノートリスト
    """
    try:
        import pretty_midi
    except ImportError:
        raise ImportError("pretty_midi required. Install: pip install pretty_midi")
    
    # Snare抽出（38=Acoustic Snare, 40=Electric Snare）
    snare_notes = [n for n in bar_notes if n.pitch in (38, 40)]
    other_notes = [n for n in bar_notes if n.pitch not in (38, 40)]
    
    if not snare_notes:
        return bar_notes
    
    # Velocity調整
    for n in snare_notes:
        new_vel = int(n.velocity * target_increase)
        n.velocity = min(new_vel, 127)
    
    return other_notes + snare_notes


def add_hat_notes_for_low_density(bar_notes, bar_start, bar_duration, min_density, tempo_bpm, max_change_ratio=0.20):
    """
    Hat不足（density < min_density）の小節にHatを追加生成
    
    Args:
        bar_notes: 小節内のノートリスト
        bar_start: 小節開始時刻（秒）
        bar_duration: 小節長（秒）
        min_density: 最小Hat密度
        tempo_bpm: テンポ
        max_change_ratio: 最大改変率（デフォルト20%）
    
    Returns:
        修正後のノートリスト
    """
    try:
        import pretty_midi
    except ImportError:
        raise ImportError("pretty_midi required. Install: pip install pretty_midi")
    
    # Hat抽出（42=Closed, 44=Pedal, 46=Open）
    hat_notes = [n for n in bar_notes if n.pitch in (42, 44, 46)]
    other_notes = [n for n in bar_notes if n.pitch not in (42, 44, 46)]
    
    current_density = len(hat_notes)
    
    if current_density >= min_density:
        return bar_notes  # 修正不要
    
    # 追加数（min_densityまで補填）
    to_add = int(min_density - current_density)
    
    # 改変率チェック（20%超えたら修正中断）
    total_notes = len(bar_notes)
    if total_notes > 0 and to_add / total_notes > max_change_ratio:
        return bar_notes  # 改変率過大
    
    # 8分音符グリッド生成
    sec_per_beat = 60.0 / tempo_bpm
    eighth_grid = sec_per_beat / 2.0  # 8分音符間隔
    
    # グリッド候補（4/4 → 8個の8分音符）
    grid_positions = [bar_start + i * eighth_grid for i in range(8)]
    
    # 既存Hatの近傍（±50ms）を除外
    existing_times = set(n.start for n in hat_notes)
    available_positions = []
    for pos in grid_positions:
        too_close = any(abs(pos - t) < 0.05 for t in existing_times)  # 50ms窓
        if not too_close:
            available_positions.append(pos)
    
    # 追加Hat生成（Closed Hat 42、Velocity 64-80のランダム）
    if len(available_positions) < to_add:
        to_add = len(available_positions)  # 利用可能な位置のみ
    
    np.random.seed(42)  # 再現性のため
    selected_positions = np.random.choice(available_positions, size=to_add, replace=False)
    
    new_hats = []
    for pos in selected_positions:
        velocity = int(np.random.uniform(64, 80))
        note = pretty_midi.Note(
            velocity=velocity,
            pitch=42,  # Closed Hat
            start=pos,
            end=pos + 0.1  # 100ms duration
        )
        new_hats.append(note)
    
    return other_notes + hat_notes + new_hats


def fix_midi_kpi(
    midi_path: Path,
    gate_config_path: Path,
    output_path: Path,
    tempo_bpm: Optional[float] = None,
    verbose: bool = True
):
    """
    MIDI KPI自動修正メイン処理
    
    Args:
        midi_path: 入力MIDIファイル
        gate_config_path: gate_prod.yaml
        output_path: 出力MIDIファイル
        tempo_bpm: テンポ（Noneの場合は自動検出）
        verbose: 詳細出力
    """
    try:
        import pretty_midi
    except ImportError:
        raise ImportError("pretty_midi required. Install: pip install pretty_midi")
    
    if verbose:
        print(f"📖 Loading MIDI: {midi_path}")
    
    midi = pretty_midi.PrettyMIDI(str(midi_path))
    gate_config = load_gate_config(gate_config_path)
    drums_config = gate_config.get('drums', {})
    
    # テンポ取得
    change_times, tempi = midi.get_tempo_changes()
    if tempo_bpm is None:
        tempo_bpm = float(tempi[0]) if len(tempi) > 0 else 120.0
    
    if verbose:
        print(f"   Tempo: {tempo_bpm} BPM")
        print(f"   Gate config: {gate_config_path}")
        print("")
    
    # ドラムトラック抽出
    drum_instruments = [inst for inst in midi.instruments if inst.is_drum]
    if not drum_instruments:
        raise ValueError(f"No drum track found in {midi_path}")
    
    # 修正統計
    density_fixes = 0
    backbeat_fixes = 0
    total_bars = 0
    
    # 小節分割（4/4前提）
    bar_duration = 60.0 / tempo_bpm * 4
    total_duration = midi.get_end_time()
    num_bars = int(np.ceil(total_duration / bar_duration))
    
    # KPI閾値取得
    max_density = drums_config.get('density', {}).get('max', 12.0)
    min_density = drums_config.get('density', {}).get('min', 2.0)  # 低密度閾値追加
    max_notes_per_bar = drums_config.get('notes_per_bar', {}).get('max', 240.0)
    min_backbeat = drums_config.get('backbeat_strength', {}).get('min', 0.3)
    
    if verbose:
        print(f"🔧 Fixing KPI violations...")
        print(f"   Density range: {min_density} - {max_density}")
        print(f"   Max notes/bar: {max_notes_per_bar}")
        print(f"   Min backbeat: {min_backbeat}")
        print("")
    
    # 修正統計（低密度用追加）
    low_density_fixes = 0
    
    # 各小節を修正
    for inst in drum_instruments:
        all_notes = sorted(inst.notes, key=lambda n: n.start)
        fixed_notes = []
        
        for bar_idx in range(num_bars):
            total_bars += 1
            bar_start = bar_idx * bar_duration
            bar_end = (bar_idx + 1) * bar_duration
            
            bar_notes = [n for n in all_notes if bar_start <= n.start < bar_end]
            if not bar_notes:
                continue
            
            # KPI計算
            hat_notes = [n for n in bar_notes if n.pitch in (42, 44, 46)]
            density = len(hat_notes)
            notes_per_bar = len(bar_notes)
            
            snare_notes = [n for n in bar_notes if n.pitch in (38, 40)]
            backbeat_strength = (np.mean([n.velocity for n in snare_notes]) / 127.0) if snare_notes else 0.0
            
            # 修正適用
            modified = False
            
            # 1. 過密修正
            if density > max_density or notes_per_bar > max_notes_per_bar:
                bar_notes = fix_density_overage(bar_notes, max_density, tempo_bpm)
                density_fixes += 1
                modified = True
            
            # 2. 低密度修正（新規追加）
            elif density < min_density:
                bar_notes = add_hat_notes_for_low_density(bar_notes, bar_start, bar_duration, min_density, tempo_bpm)
                low_density_fixes += 1
                modified = True
            
            # 3. backbeat修正
            if backbeat_strength < min_backbeat and snare_notes:
                bar_notes = fix_backbeat_weakness(bar_notes, min_backbeat)
                backbeat_fixes += 1
                modified = True
            
            if verbose and modified:
                print(f"  ✓ bar_{bar_idx}: ", end="")
                if density > max_density:
                    print(f"density {density:.1f} → {max_density:.1f} ", end="")
                elif density < min_density:
                    print(f"density {density:.1f} → {min_density:.1f} (add hats) ", end="")
                if notes_per_bar > max_notes_per_bar:
                    print(f"notes {notes_per_bar} → reduced ", end="")
                if backbeat_strength < min_backbeat:
                    print(f"backbeat {backbeat_strength:.2f} → boosted ", end="")
                print()
            
            fixed_notes.extend(bar_notes)
        
        # ノートを置き換え
        inst.notes = fixed_notes
    
    # 保存
    output_path.parent.mkdir(parents=True, exist_ok=True)
    midi.write(str(output_path))
    
    if verbose:
        print("")
        print(f"📊 Fix Statistics:")
        print(f"   Total bars: {total_bars}")
        print(f"   Density fixes (high): {density_fixes}")
        print(f"   Density fixes (low): {low_density_fixes}")
        print(f"   Backbeat fixes: {backbeat_fixes}")
        print("")
        print(f"✅ Saved fixed MIDI: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='MIDI KPI Auto-Fix')
    parser.add_argument('--midi', type=Path, required=True, help='Input MIDI file')
    parser.add_argument('--gate-config', type=Path, required=True, help='gate_prod.yaml')
    parser.add_argument('--output', type=Path, required=True, help='Output MIDI file')
    parser.add_argument('--tempo-bpm', type=float, default=None, help='Tempo in BPM (auto-detect if omitted)')
    parser.add_argument('--quiet', action='store_true', help='Suppress verbose output')
    args = parser.parse_args()
    
    fix_midi_kpi(
        midi_path=args.midi,
        gate_config_path=args.gate_config,
        output_path=args.output,
        tempo_bpm=args.tempo_bpm,
        verbose=not args.quiet
    )


if __name__ == '__main__':
    main()
