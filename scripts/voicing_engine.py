#!/usr/bin/env python3
"""
voicing_engine.py
-----------------
コード記号（例: Cmaj7, Dm7）をピッチ配列に展開します。
music21が利用可能ならそれを使用、なければ簡易実装にフォールバック。
"""
from typing import List, Dict

# 依存が無ければ簡易展開で動く。music21 があれば高度化できるように設計
try:
    from music21 import chord as m21chord, pitch as m21pitch
    HAVE_M21 = True
except Exception:
    HAVE_M21 = False

SEMITONES = {
    "C":0, "C#":1, "Db":1, "D":2, "D#":3, "Eb":3, "E":4,
    "F":5, "F#":6, "Gb":6, "G":7, "G#":8, "Ab":8, "A":9,
    "A#":10, "Bb":10, "B":11
}

BASIC_QUAL = {
    "maj": [0,4,7], "": [0,4,7], "m": [0,3,7], "min": [0,3,7],
    "7": [0,4,7,10], "maj7":[0,4,7,11], "m7":[0,3,7,10],
    "add9": [0,4,7,14], "sus4": [0,5,7], "6": [0,4,7,9]
}

def parse_root_qual(sym: str):
    """コード記号からroot（根音）とquality（和音種類）を分離"""
    # 例: C, Cmaj7, Dm7, F#7
    root = ""
    qual = ""
    for i in range(len(sym)):
        cand = sym[:len(sym)-i]
        if cand in SEMITONES:
            root = cand
            qual = sym[len(cand):]
            break
    if not root:
        raise ValueError(f"Unknown chord symbol: {sym}")
    return root, qual

def chord_to_pitches(
    symbol: str,
    octave: int = 4,
    style: str = "close",
    inversion: int = 0
) -> List[int]:
    """
    コード記号をMIDIピッチ番号配列に変換
    
    Args:
        symbol: コード記号（例: "Cmaj7", "Dm7"）
        octave: オクターブ（4 = C4 = middle C）
        style: ボイシングスタイル（close/drop2/spread）
        inversion: 転回形（0=基本形、1=第1転回形...）
    
    Returns:
        MIDIピッチ番号のリスト
    """
    if HAVE_M21:
        # music21が利用可能な場合
        try:
            c = m21chord.Chord(symbol)
            c.closedPosition(forceOctave=octave)
            if inversion:
                c = c.inversion(inversion)
            return [p.midi for p in c.pitches]
        except Exception:
            # フォールバック
            pass
    
    # 簡易実装
    root, qual = parse_root_qual(symbol)
    base = BASIC_QUAL.get(qual, BASIC_QUAL.get("", [0,4,7]))
    root_midi = 12 * (octave + 1) + SEMITONES[root]
    notes = [root_midi + iv for iv in base]
    
    # 簡易 inversion
    for _ in range(max(0, inversion)):
        if len(notes) > 0:
            n = notes.pop(0)
            notes.append(n + 12)
    
    # 簡易 spread/drop2
    if style == "drop2" and len(notes) >= 3:
        notes[1] -= 12
    elif style == "spread":
        notes = [n + (i//2)*12 for i, n in enumerate(notes)]
    
    return notes
