#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/analyze_key_difference.py

手動chordmapと自動生成のキー差分を分析し、転置後の精度を評価する。
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import List, Tuple
from collections import Counter

NOTE_CIRCLE = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

def normalize_note(note: str) -> str:
    """Db → C#, Eb → D# 等のエンハーモニック変換"""
    enharmonic = {
        'Db': 'C#', 'Eb': 'D#', 'Gb': 'F#', 'Ab': 'G#', 'Bb': 'A#'
    }
    return enharmonic.get(note, note)

def transpose_note(note: str, semitones: int) -> str:
    """ノートを半音単位で転置"""
    note = normalize_note(note.strip())
    if note not in NOTE_CIRCLE:
        return note
    idx = NOTE_CIRCLE.index(note)
    new_idx = (idx + semitones) % 12
    return NOTE_CIRCLE[new_idx]

def extract_roots(chords: List[Tuple[float, str, str]]) -> List[str]:
    """コードリストからルート音のみを抽出"""
    roots = []
    for _, root, _ in chords:
        # "Am" → "A" 等の変換
        if len(root) > 1 and root[-1].lower() == 'm':
            root = root[:-1]
        roots.append(normalize_note(root))
    return roots

def find_best_transposition(manual_roots: List[str], auto_roots: List[str]) -> Tuple[int, float]:
    """
    最も精度が高くなる転置量（半音）を検索
    Returns: (semitones, accuracy)
    """
    best_semitones = 0
    best_accuracy = 0.0
    
    for semitones in range(12):
        # auto_rootsを転置
        transposed = [transpose_note(r, semitones) for r in auto_roots]
        
        # マッチング（順序は考慮せず、頻度で比較）
        manual_counter = Counter(manual_roots)
        transposed_counter = Counter(transposed)
        
        # 共通するルート音の数
        matches = sum((manual_counter & transposed_counter).values())
        accuracy = matches / max(len(manual_roots), 1)
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_semitones = semitones
    
    return best_semitones, best_accuracy

def main():
    ap = argparse.ArgumentParser(description="Analyze key difference between manual and auto chordmaps")
    ap.add_argument("--manual", required=True, help="Path to manual sections.json")
    ap.add_argument("--auto", required=True, help="Path to auto-generated chordmap.json")
    args = ap.parse_args()
    
    # データ読み込み（scripts/compare_chordmaps.pyの関数を再利用）
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from compare_chordmaps import parse_manual_sections, parse_auto_chordmap
    
    manual_path = Path(args.manual)
    auto_path = Path(args.auto)
    
    manual_chords = parse_manual_sections(manual_path)
    auto_chords = parse_auto_chordmap(auto_path)
    
    manual_roots = extract_roots(manual_chords)
    auto_roots = extract_roots(auto_chords)
    
    # キー差分分析
    best_semitones, best_accuracy = find_best_transposition(manual_roots, auto_roots)
    
    print("\n" + "="*60)
    print("Key Difference Analysis")
    print("="*60)
    print(f"\nManual chords: {len(manual_chords)}")
    print(f"Auto chords:   {len(auto_chords)}")
    
    print(f"\nManual root distribution:")
    manual_counter = Counter(manual_roots)
    for root, count in manual_counter.most_common():
        print(f"  {root:3s}: {count:2d} ({'*' * count})")
    
    print(f"\nAuto root distribution:")
    auto_counter = Counter(auto_roots)
    for root, count in auto_counter.most_common():
        print(f"  {root:3s}: {count:2d} ({'*' * count})")
    
    print(f"\nBest transposition:")
    print(f"  Semitones: {best_semitones:+d} (auto → manual)")
    if best_semitones > 0:
        print(f"  Key shift: {NOTE_CIRCLE[0]} → {NOTE_CIRCLE[best_semitones]}")
    print(f"  Root match accuracy: {best_accuracy*100:.1f}%")
    
    # 転置後のルート分布
    transposed_roots = [transpose_note(r, best_semitones) for r in auto_roots]
    transposed_counter = Counter(transposed_roots)
    
    print(f"\nAuto root distribution (after transposition by {best_semitones:+d}):")
    for root, count in transposed_counter.most_common():
        print(f"  {root:3s}: {count:2d} ({'*' * count})")
    
    print("\n" + "="*60 + "\n")
    
    # 推奨事項
    if best_semitones != 0:
        print(f"🔧 Recommendation:")
        print(f"   The auto-generated chordmap is {abs(best_semitones)} semitone(s) {'sharp' if best_semitones > 0 else 'flat'}.")
        print(f"   This is likely due to librosa's tuning correction.")
        print(f"   Consider adding --force-key option to stem_harmony.py")
        print(f"   to fix the global key to the manual key.\n")

if __name__ == "__main__":
    main()
