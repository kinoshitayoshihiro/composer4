#!/usr/bin/env python3
"""
Bassline Plan Generator from Chordmap

chordmap.json + sections.jsonからベースライン指針を生成

Usage:
    python make_bassline_plan_from_chordmap.py \
        --chordmap analysis/chordmap.json \
        --sections analysis/sections.json \
        --out-csv analysis/bassline_plan.csv
"""

import argparse
import json
import pandas as pd
from pathlib import Path


def chord_root_to_midi(root: str) -> int:
    """コードルート → MIDI番号（C=36）"""
    notes = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}

    # シャープ・フラット処理
    name = root.replace("#", "").replace("b", "")
    base = notes.get(name[0].upper(), 0)

    if "#" in root:
        base += 1
    elif "b" in root:
        base -= 1

    return 36 + (base % 12)  # C2 (MIDI 36)をベース


def generate_bassline_plan(chordmap_path: Path, sections_path: Path) -> pd.DataFrame:
    """ベースライン指針生成"""

    # chordmap読み込み
    with open(chordmap_path) as f:
        chordmap = json.load(f)

    # sections読み込み
    with open(sections_path) as f:
        sections_data = json.load(f)
        sections = sections_data.get("sections", [])

    # セクション別運用方針
    section_policy = {
        "intro": {"pattern": "root", "octave": 2, "density": 0.5},
        "verse": {"pattern": "root_fifth", "octave": 2, "density": 0.6},
        "chorus": {"pattern": "walking", "octave": 2, "density": 0.9},
        "bridge": {"pattern": "chromatic", "octave": 2, "density": 0.7},
        "outro": {"pattern": "root", "octave": 2, "density": 0.4},
        "pre_chorus": {"pattern": "ascending", "octave": 2, "density": 0.75},
    }

    rows = []

    for event in chordmap:
        bar = event.get("bar", 0)
        root = event.get("root", "C")
        quality = event.get("quality", "major")

        # 該当セクション検索
        section_label = "verse"  # default
        for sec in sections:
            start = sec.get("start_bar", 0)
            end = sec.get("end_bar", 999)
            if start <= bar <= end:
                section_label = sec.get("label", "verse").lower()
                break

        # セクション運用方針取得
        policy = section_policy.get(section_label, section_policy["verse"])

        # MIDI番号計算
        root_midi = chord_root_to_midi(root)
        fifth_midi = root_midi + 7

        rows.append(
            {
                "bar": bar,
                "root": root,
                "quality": quality,
                "section_label": section_label,
                "pattern": policy["pattern"],
                "root_midi": root_midi,
                "fifth_midi": fifth_midi,
                "octave": policy["octave"],
                "density": policy["density"],
            }
        )

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Bassline plan generator")
    parser.add_argument("--chordmap", type=Path, required=True)
    parser.add_argument("--sections", type=Path, required=True)
    parser.add_argument("--out-csv", type=Path, required=True)

    args = parser.parse_args()

    # ベースラインplan生成
    bassline_df = generate_bassline_plan(args.chordmap, args.sections)

    # CSV出力
    bassline_df.to_csv(args.out_csv, index=False)

    print(f"✅ Bassline plan generated: {len(bassline_df)} bars")
    print(f"   Output: {args.out_csv}")
    print(f"   Patterns: {bassline_df['pattern'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
