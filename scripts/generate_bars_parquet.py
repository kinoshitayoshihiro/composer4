#!/usr/bin/env python3
"""
bars.parquet生成スクリプト
sections.json + chordmap.json → bars.parquet（小節単位の目標値）

生成カラム:
- bar_index: int (0..total_bars-1)
- section_label: str (intro/verse/chorus/pre_chorus/outro)
- energy_curve: float (0..1、sections.jsonのenergy参照)
- accent_score_target: float (0..1、エナジー曲線から自動計算）
- density_target: float (Hi-hat密度目標、セクション種別から推定）
- swing_target: float (0..1、chordmap.jsonのドラム設定から推定）

使用例:
    python3 scripts/generate_bars_parquet.py \\
        --sections sections.json \\
        --chordmap data/chordmap.json \\
        --output song_packages/sample_project/sample_song/bars.parquet
"""

import argparse
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional


def load_sections(sections_path: Path) -> dict:
    """sections.jsonロード"""
    with open(sections_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_chordmap(chordmap_path: Path) -> dict:
    """chordmap.jsonロード"""
    with open(chordmap_path, "r", encoding="utf-8") as f:
        return json.load(f)


def infer_density_target(section_label: str, energy: float) -> float:
    """セクション種別+エナジーからHi-hat密度目標推定

    Args:
        section_label: セクションラベル (intro/verse/chorus/pre_chorus/outro)
        energy: エナジー曲線 (0..1)

    Returns:
        Hi-hat密度目標 (onset/bar)
    """
    # セクション基準密度
    base_density = {
        "intro": 4.0,
        "verse": 5.0,
        "pre_chorus": 6.0,
        "chorus": 8.0,
        "outro": 3.0,
    }.get(section_label.lower(), 5.0)

    # エナジー補正 (±2.0)
    energy_factor = (energy - 0.5) * 4.0

    return max(2.0, min(12.0, base_density + energy_factor))


def infer_swing_target(section_label: str, chordmap_sections: dict) -> float:
    """セクション種別+chordmap.jsonドラム設定からスウィング目標推定

    Args:
        section_label: セクションラベル (intro/verse/chorus/pre_chorus/outro)
        chordmap_sections: chordmap.jsonのsections辞書

    Returns:
        スウィング目標 (0..1、0=STRAIGHT, 1=SWING)
    """
    # chordmap.jsonからドラムスタイル検索
    for section_name, section_data in chordmap_sections.items():
        if section_label.lower() in section_name.lower():
            drum_style = section_data.get("part_settings", {}).get("drum_style_key", "")

            # スウィング推定ルール
            if "swing" in drum_style.lower():
                return 0.8  # SWING
            elif "ballad" in drum_style.lower():
                return 0.3  # 軽微なスウィング
            elif "no_drums" in drum_style.lower():
                return 0.0  # ドラムなし（STRAIGHT）
            else:
                return 0.1  # STRAIGHT（デフォルト）

    return 0.1  # デフォルト（STRAIGHT）


def generate_bars_parquet(
    sections_path: Path,
    chordmap_path: Path,
    output_path: Path,
    drums_midi_path: Optional[Path] = None,
):
    """bars.parquet生成

    Args:
        sections_path: sections.jsonパス
        chordmap_path: chordmap.jsonパス
        output_path: bars.parquet出力パス
        drums_midi_path: drums.midパス（小節数推定用、省略時はenergy_curveベース）
    """
    # ロード
    sections_data = load_sections(sections_path)
    chordmap_data = load_chordmap(chordmap_path)

    # エナジー曲線（小節単位）
    energy_curve = {int(bar): float(val) for bar, val in sections_data.get("energy", [])}

    # セクション区切り
    section_boundaries = sections_data.get("sections", [])

    # 小節数推定
    if drums_midi_path and drums_midi_path.exists():
        # drums.midから小節数推定
        import pretty_midi

        mid = pretty_midi.PrettyMIDI(str(drums_midi_path))
        tempo = mid.estimate_tempo()
        duration_sec = mid.get_end_time()
        # 4/4拍子想定で小節数推定
        total_bars = int(duration_sec / (240.0 / tempo)) + 1
    else:
        # energy_curveベース（従来）
        total_bars = max(energy_curve.keys()) + 1 if energy_curve else 72

    # bars.parquet生成
    bars = []

    for bar_idx in range(total_bars):
        # セクションラベル特定
        section_label = "intro"
        for i, boundary in enumerate(section_boundaries):
            if bar_idx >= boundary["bar"]:
                section_label = boundary["label"]

        # エナジー曲線
        energy = energy_curve.get(bar_idx, 0.5)

        # アクセント目標（エナジー曲線そのまま使用）
        accent_score_target = energy

        # 密度目標
        density_target = infer_density_target(section_label, energy)

        # スウィング目標
        swing_target = infer_swing_target(section_label, chordmap_data.get("sections", {}))

        bars.append(
            {
                "bar_index": bar_idx,
                "section_label": section_label,
                "energy_curve": energy,
                "accent_score_target": accent_score_target,
                "density_target": density_target,
                "swing_target": swing_target,
            }
        )

    # DataFrame作成
    df = pd.DataFrame(bars)

    # Parquet保存
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, compression="snappy", index=False)

    print(f"✅ Generated bars.parquet: {output_path}")
    print(f"   Total bars: {len(df)}")
    print(f"   Columns: {list(df.columns)}")
    print(f"\n📊 Statistics:")
    print(
        f"   Energy: {df['energy_curve'].min():.2f} .. {df['energy_curve'].max():.2f} (mean: {df['energy_curve'].mean():.2f})"
    )
    print(
        f"   Density: {df['density_target'].min():.2f} .. {df['density_target'].max():.2f} (mean: {df['density_target'].mean():.2f})"
    )
    print(
        f"   Swing: {df['swing_target'].min():.2f} .. {df['swing_target'].max():.2f} (mean: {df['swing_target'].mean():.2f})"
    )
    print(f"\n🔍 Section distribution:")
    print(df["section_label"].value_counts().to_string())


def main():
    parser = argparse.ArgumentParser(
        description="Generate bars.parquet from sections.json + chordmap.json"
    )
    parser.add_argument("--sections", type=Path, required=True, help="Path to sections.json")
    parser.add_argument("--chordmap", type=Path, required=True, help="Path to chordmap.json")
    parser.add_argument("--output", type=Path, required=True, help="Path to output bars.parquet")
    parser.add_argument(
        "--drums-midi",
        type=Path,
        default=None,
        help="Path to drums.mid (optional, for total bars estimation)",
    )

    args = parser.parse_args()

    generate_bars_parquet(
        args.sections, args.chordmap, args.output, drums_midi_path=args.drums_midi
    )


if __name__ == "__main__":
    main()
