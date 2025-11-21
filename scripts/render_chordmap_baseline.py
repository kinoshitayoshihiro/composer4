#!/usr/bin/env python3
"""
render_chordmap_baseline.py - Chordmapを"そのまま鳴らす"最低保証レンダラ

目的: manual_chordmap.jsonを確実に音にするベースライン生成
用途: V2ジェネレーターのフォールバック、または初期レイヤー
"""

import json
import argparse
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

try:
    import pandas as pd
except ImportError as e:
    raise SystemExit("pandas が必要です: pip install pandas") from e

# 既存ユーティリティを使用
sys.path.insert(0, str(Path(__file__).parent))
from chordmap_utils import load_chordmap, get_chord_at_bar, parse_symbol, get_chord_tones

# 楽器別の設定
INSTR_CFG = {
    "bass": {
        "range": (28, 43),  # E1-G2
        "notes_per_bar": 4,
        "pattern": "root-5th",
        "velocity": (85, 100),
    },
    "piano": {
        "range": (48, 72),  # C3-C5
        "notes_per_bar": 4,
        "pattern": "block-chord",
        "velocity": (75, 90),
    },
    "guitar": {
        "range": (48, 69),  # C3-A4
        "notes_per_bar": 4,
        "pattern": "arpeggio",
        "velocity": (80, 95),
    },
    "strings": {
        "range": (60, 79),  # C4-G5
        "notes_per_bar": 2,
        "pattern": "sustained-top",
        "velocity": (70, 85),
    },
}


def closest_in_range(pitch: int, lo: int, hi: int) -> int:
    """ピッチを指定範囲に収める(オクターブ移動)"""
    while pitch < lo:
        pitch += 12
    while pitch > hi:
        pitch -= 12
    return max(lo, min(pitch, hi))


def voice_lead(prev_top: Optional[int], tones: List[int], lo: int, hi: int) -> int:
    """最小移動で上声を選択(メロディックな動き)"""
    if not tones:
        return 60

    candidates = []
    for tone in tones:
        adjusted = closest_in_range(tone, lo, hi)
        distance = abs(adjusted - (prev_top if prev_top is not None else adjusted))
        candidates.append((distance, adjusted))

    candidates.sort()
    return candidates[0][1]


def bar_start_ql(bar_idx: int, beats_per_bar: int = 4) -> float:
    """小節番号から開始時刻(quarter notes)を計算"""
    return float(bar_idx * beats_per_bar)


def make_bass_events(
    bars_df: pd.DataFrame, chordmap: List[Dict[str, Any]], cfg: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """Bassパターン生成: root-5th交互"""
    lo, hi = cfg["range"]
    vel_lo, vel_hi = cfg["velocity"]
    events = []

    for bar_idx in range(len(bars_df)):
        chord = get_chord_at_bar(chordmap, bar_idx)
        symbol = chord.get("symbol", "C")
        parsed = parse_symbol(symbol)
        chord_tones = get_chord_tones(parsed, bass_octave=2)

        if not chord_tones:
            continue

        root = closest_in_range(chord_tones[0], lo, hi)
        fifth = closest_in_range(root + 7, lo, hi)

        start_ql = bar_start_ql(bar_idx)

        # riff_slotではウォーキングベース(4分4つ)
        bar_data = bars_df.iloc[bar_idx]
        if bar_data.get("riff_slot", False):
            for beat in range(4):
                note = root if beat % 2 == 0 else fifth
                events.append(
                    {
                        "time_ql": start_ql + beat,
                        "note": note,
                        "velocity": vel_hi,
                        "duration_ql": 0.9,
                    }
                )
        else:
            # 通常は全音符root
            events.append(
                {
                    "time_ql": start_ql,
                    "note": root,
                    "velocity": vel_lo,
                    "duration_ql": 4.0,
                }
            )

    return events


def make_piano_events(
    bars_df: pd.DataFrame, chordmap: List[Dict[str, Any]], cfg: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """Pianoパターン生成: block chord / arpeggio"""
    lo, hi = cfg["range"]
    vel_lo, vel_hi = cfg["velocity"]
    events = []

    for bar_idx in range(len(bars_df)):
        chord = get_chord_at_bar(chordmap, bar_idx)
        symbol = chord.get("symbol", "C")
        parsed = parse_symbol(symbol)
        chord_tones = get_chord_tones(parsed, bass_octave=4)

        if not chord_tones:
            continue

        # レンジ内に調整
        voicing = [closest_in_range(t, lo, hi) for t in chord_tones[:4]]
        voicing = sorted(set(voicing))[:4]  # 最大4声

        start_ql = bar_start_ql(bar_idx)
        bar_data = bars_df.iloc[bar_idx]
        section = bar_data.get("section_label", "verse")

        # Chorus/Bridgeまたはriff_slotではアルペジオ
        if section in ("chorus", "bridge") or bar_data.get("riff_slot", False):
            # 4分音符アルペジオ
            arp_seq = (voicing * 2)[:4]  # 4音確保
            for beat, note in enumerate(arp_seq):
                events.append(
                    {
                        "time_ql": start_ql + beat,
                        "note": note,
                        "velocity": vel_hi,
                        "duration_ql": 0.9,
                    }
                )
        else:
            # ブロックコード(全音符)
            for note in voicing:
                events.append(
                    {
                        "time_ql": start_ql,
                        "note": note,
                        "velocity": vel_lo,
                        "duration_ql": 4.0,
                    }
                )

    return events


def make_guitar_events(
    bars_df: pd.DataFrame, chordmap: List[Dict[str, Any]], cfg: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """Guitarパターン生成: arpeggio / strumming"""
    lo, hi = cfg["range"]
    vel_lo, vel_hi = cfg["velocity"]
    events = []

    for bar_idx in range(len(bars_df)):
        chord = get_chord_at_bar(chordmap, bar_idx)
        symbol = chord.get("symbol", "C")
        parsed = parse_symbol(symbol)
        chord_tones = get_chord_tones(parsed, bass_octave=4)

        if not chord_tones:
            continue

        voicing = [closest_in_range(t, lo, hi) for t in chord_tones[:4]]
        voicing = sorted(set(voicing))

        start_ql = bar_start_ql(bar_idx)
        bar_data = bars_df.iloc[bar_idx]

        # 常にアルペジオパターン
        arp_seq = (voicing * 2)[:4]
        for beat, note in enumerate(arp_seq):
            events.append(
                {
                    "time_ql": start_ql + beat,
                    "note": note,
                    "velocity": vel_hi if bar_data.get("riff_slot", False) else vel_lo,
                    "duration_ql": 0.8,
                }
            )

    return events


def make_strings_events(
    bars_df: pd.DataFrame, chordmap: List[Dict[str, Any]], cfg: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """Stringsパターン生成: sustained top voice (voice leading)"""
    lo, hi = cfg["range"]
    vel_lo, vel_hi = cfg["velocity"]
    events = []
    prev_top = None

    for bar_idx in range(len(bars_df)):
        chord = get_chord_at_bar(chordmap, bar_idx)
        symbol = chord.get("symbol", "C")
        parsed = parse_symbol(symbol)
        chord_tones = get_chord_tones(parsed, bass_octave=4)

        if not chord_tones:
            continue

        # Voice leading: 前の音に最も近い上声を選択
        top = voice_lead(prev_top, chord_tones, lo, hi)
        prev_top = top

        start_ql = bar_start_ql(bar_idx)
        bar_data = bars_df.iloc[bar_idx]
        section = bar_data.get("section_label", "verse")

        # 上声(メロディ)
        events.append(
            {
                "time_ql": start_ql,
                "note": top,
                "velocity": vel_hi if section in ("chorus", "bridge") else vel_lo,
                "duration_ql": 4.0,
            }
        )

        # Chorus/Bridgeでは下声を追加(厚み)
        if section in ("chorus", "bridge"):
            lower = max(lo, top - 7)  # 5度下
            events.append(
                {
                    "time_ql": start_ql,
                    "note": lower,
                    "velocity": vel_lo,
                    "duration_ql": 4.0,
                }
            )

    return events


def main():
    parser = argparse.ArgumentParser(
        description="Chordmapベースライン生成: 確実に鳴る最低保証レイヤー"
    )
    parser.add_argument("--bars", required=True, help="bars_with_slots.parquet path")
    parser.add_argument("--chordmap", required=True, help="chordmap JSON path")
    parser.add_argument("--outdir", default="plans", help="Output directory")
    parser.add_argument(
        "--instruments",
        nargs="+",
        default=["bass", "piano", "guitar", "strings"],
        help="Instruments to generate",
    )

    args = parser.parse_args()

    # Load data
    bars_df = pd.read_parquet(args.bars)
    chordmap = load_chordmap(args.chordmap)

    # Create output directory
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Generate plans for each instrument
    generators = {
        "bass": make_bass_events,
        "piano": make_piano_events,
        "guitar": make_guitar_events,
        "strings": make_strings_events,
    }

    for inst in args.instruments:
        if inst not in generators:
            print(f"Warning: Unknown instrument '{inst}', skipping")
            continue

        cfg = INSTR_CFG[inst]
        events = generators[inst](bars_df, chordmap, cfg)

        plan = {
            "metadata": {
                "instrument": inst,
                "num_bars": len(bars_df),
                "num_events": len(events),
                "generator": "render_chordmap_baseline.py",
                "mode": "chordmap-as-is",
            },
            "events": events,
        }

        out_path = outdir / f"{inst}_plan.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(plan, f, ensure_ascii=False, indent=2)

        print(f"✅ {inst}: {len(events)} events → {out_path}")

    print(f"\n🎵 Baseline plans generated in {outdir}")
    print("Next: Run Phase C to generate MIDI")


if __name__ == "__main__":
    main()
