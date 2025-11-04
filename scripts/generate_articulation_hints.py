#!/usr/bin/env python3
"""
generate_articulation_hints.py - articulation_hints.json生成

drums_recommendations.json（または任意のMIDI解析結果）から
articulation_hints.json を生成します。

Usage:
    python3 scripts/generate_articulation_hints.py \
      --recommendations song_packages/<project>/<song>/drums_recommendations.json \
      --output song_packages/<project>/<song>/articulation_hints.json \
      --tempo-bpm 120
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict


def calculate_staccato_ratio(pattern: dict) -> float:
    """Staccato比率を計算（density高い + notes_per_bar多い → staccato傾向）"""
    density = pattern.get("density", 5.0)
    notes = pattern.get("notes_per_bar", 15.0)

    # densityが高い（8以上）かつnotes_per_barが多い（25以上）→ staccato
    staccato_score = 0.0
    if density >= 8.0:
        staccato_score += 0.4
    if notes >= 25:
        staccato_score += 0.4

    # swing低い（0.1未満）→ staccato傾向
    swing = pattern.get("swing", 0.2)
    if swing < 0.1:
        staccato_score += 0.2

    return min(1.0, staccato_score)


def calculate_legato_ratio(pattern: dict) -> float:
    """Legato比率を計算（swing高い + density低め → legato傾向）"""
    swing = pattern.get("swing", 0.2)
    density = pattern.get("density", 5.0)

    # swing高い（0.25以上）→ legato
    legato_score = 0.0
    if swing >= 0.25:
        legato_score += 0.5

    # density適度（4-7）→ legato
    if 4.0 <= density <= 7.0:
        legato_score += 0.3

    # backbeat強い（0.5以上）→ legato（グルーヴ感）
    backbeat = pattern.get("backbeat_strength", 0.5)
    if backbeat >= 0.5:
        legato_score += 0.2

    return min(1.0, legato_score)


def calculate_pizzicato_score(pattern: dict) -> float:
    """Pizzicato傾向を計算（notes少ない + density低い → pizzicato）"""
    notes = pattern.get("notes_per_bar", 15.0)
    density = pattern.get("density", 5.0)

    # notes少ない（10未満）+ density低い（3未満）→ pizzicato
    pizz_score = 0.0
    if notes < 10:
        pizz_score += 0.5
    if density < 3.0:
        pizz_score += 0.5

    return min(1.0, pizz_score)


def calculate_tremolo_ratio(pattern: dict) -> float:
    """Tremolo比率を計算（density非常に高い → tremolo）"""
    density = pattern.get("density", 5.0)

    # density非常に高い（10以上）→ tremolo
    if density >= 10.0:
        return min(1.0, (density - 10.0) / 5.0 + 0.5)

    return 0.0


def calculate_accent_score(pattern: dict) -> float:
    """Accent強度を計算（backbeat強い + kick_downbeat → accent）"""
    backbeat = pattern.get("backbeat_strength", 0.5)
    kick_down = pattern.get("kick_downbeat_rate", 0.0)

    # backbeat強い + kick_downbeat高い → accent
    accent_score = backbeat * 0.6 + kick_down * 0.4

    return min(1.0, accent_score)


def calculate_energy_curve(pattern: dict) -> float:
    """Energy曲線（densityベース、0.0-1.0正規化）"""
    density = pattern.get("density", 5.0)

    # density範囲: 2.0-12.0 → 0.0-1.0
    min_density = 2.0
    max_density = 12.0

    normalized = (density - min_density) / (max_density - min_density)
    return max(0.0, min(1.0, normalized))


def calculate_vibrato_depth(pattern: dict) -> float:
    """Vibrato深さ（swing/backbeat組み合わせ）"""
    swing = pattern.get("swing", 0.2)
    backbeat = pattern.get("backbeat_strength", 0.5)

    # swing高い + backbeat中程度 → vibrato
    vibrato = swing * 0.7 + (0.5 - abs(backbeat - 0.5)) * 0.3

    return min(1.0, vibrato)


def generate_hints_from_recommendations(recommendations_path: Path, tempo_bpm: float) -> List[Dict]:
    """drums_recommendations.jsonからarticulation_hints生成"""
    with open(recommendations_path) as f:
        recs = json.load(f)

    # bar_0, bar_1, ... という形式のキーを抽出
    bar_keys = [k for k in recs.keys() if k.startswith("bar_")]
    hints = []

    for bar_key in sorted(bar_keys, key=lambda x: int(x.split("_")[1])):
        bar_data = recs[bar_key]
        bar_idx = bar_data["bar_index"]
        pattern = bar_data["pattern"]

        # 小節開始時刻（秒）
        time_sec = bar_idx * (4.0 * 60.0 / tempo_bpm)  # 4/4拍子想定

        # articulation_hints計算
        hint = {
            "time": round(time_sec, 3),
            "bar_index": bar_idx,
            "staccato_ratio": round(calculate_staccato_ratio(pattern), 3),
            "legato_ratio": round(calculate_legato_ratio(pattern), 3),
            "pizzicato_score": round(calculate_pizzicato_score(pattern), 3),
            "tremolo_ratio": round(calculate_tremolo_ratio(pattern), 3),
            "accent_score": round(calculate_accent_score(pattern), 3),
            "energy_curve": round(calculate_energy_curve(pattern), 3),
            "vibrato_depth": round(calculate_vibrato_depth(pattern), 3),
            # 元のパターン情報（参考用）
            "source_density": round(pattern.get("density", 5.0), 2),
            "source_swing": round(pattern.get("swing", 0.2), 2),
            "source_backbeat": round(pattern.get("backbeat_strength", 0.5), 2),
        }

        hints.append(hint)

    return hints


def main():
    parser = argparse.ArgumentParser(description="articulation_hints.json生成")
    parser.add_argument(
        "--recommendations", type=Path, required=True, help="drums_recommendations.json path"
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="Output articulation_hints.json path"
    )
    parser.add_argument(
        "--tempo-bpm", type=float, default=120.0, help="Tempo in BPM (default: 120)"
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="詳細ログ出力")

    args = parser.parse_args()

    # articulation_hints生成
    print(f"📖 Loading recommendations: {args.recommendations}")
    hints = generate_hints_from_recommendations(args.recommendations, args.tempo_bpm)

    # JSON出力
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(hints, f, indent=2)

    print(f"✅ articulation_hints saved: {args.output}")
    print(f"   Total hints: {len(hints)}")
    print(f"   Tempo: {args.tempo_bpm} BPM")

    if args.verbose:
        print(f"\n📊 Sample hints (first 3):")
        for hint in hints[:3]:
            print(f"   Bar {hint['bar_index']} (t={hint['time']}s):")
            print(f"     staccato: {hint['staccato_ratio']}, legato: {hint['legato_ratio']}")
            print(f"     accent: {hint['accent_score']}, energy: {hint['energy_curve']}")


if __name__ == "__main__":
    main()
