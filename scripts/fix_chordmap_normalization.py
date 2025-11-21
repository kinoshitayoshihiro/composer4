#!/usr/bin/env python3
"""
chordmap_locked.json の正規化バグを修正

問題:
- root に "Em", "Am" など質を含む表記 → 正: "E", "A" + quality="m"
- 結果として symbol が "Emm", "Amm7" になる二重 m バグ

修正:
- root から quality を取り除く
- symbol を正しく再構築: {root}{quality}
"""

import json
import sys
from pathlib import Path


def normalize_root(root: str, quality: str) -> str:
    """
    root から quality suffix を除去

    Examples:
        ("Em", "m")   -> "E"
        ("Am", "m7")  -> "A"
        ("Gmaj", "maj") -> "G"
    """
    # Minor の場合: root が "m" で終わる → 除去
    if quality in ["m", "m7", "m9", "m11", "m13"] and root.endswith("m"):
        return root[:-1]

    # Major の場合: root が "maj" で終わる → 除去
    if quality in ["maj", "maj7", "maj9"] and root.endswith("maj"):
        return root[:-3]

    return root


def fix_chordmap_locked(input_path: Path, output_path: Path = None) -> dict:
    """chordmap_locked.json を正規化"""

    if output_path is None:
        output_path = input_path

    with open(input_path) as f:
        data = json.load(f)

    fixed_count = 0

    for event in data["events"]:
        root = event["root"]
        quality = event["quality"]

        # root を正規化
        normalized_root = normalize_root(root, quality)

        if normalized_root != root:
            print(
                f"Bar {event['bar']:2d}: {root:6s} + {quality:6s} → {normalized_root:6s} + {quality:6s}"
            )
            event["root"] = normalized_root
            fixed_count += 1

    # meta を更新
    if "meta" not in data:
        data["meta"] = {}
    data["meta"]["normalization_fixes"] = fixed_count

    # 保存
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Fixed {fixed_count} events")
    print(f"✅ Saved to: {output_path}")

    return data


def regenerate_m21_chordmap(locked_path: Path, output_path: Path):
    """
    chordmap_locked.json から chordmap_m21.json を再生成

    正しい symbol = root + quality を構築
    """
    with open(locked_path) as f:
        locked = json.load(f)

    m21_data = {"unit": "ql", "events": []}

    for event in locked["events"]:
        root = event["root"]
        quality = event["quality"]

        # symbol を正しく構築
        symbol = f"{root}{quality}"

        m21_event = {
            "symbol": symbol,
            "root": root,
            "quality": quality,
            "time_ql": event["time_ql"],
            "bar": event["bar"],
        }

        m21_data["events"].append(m21_event)

    with open(output_path, "w") as f:
        json.dump(m21_data, f, indent=2, ensure_ascii=False)

    print(f"✅ Regenerated chordmap_m21.json with {len(m21_data['events'])} events")

    # 和音分布を表示
    from collections import Counter

    cnt = Counter(e["symbol"] for e in m21_data["events"])
    print("\n=== 和音分布 (Top 10) ===")
    for sym, count in cnt.most_common(10):
        print(f"{sym:15s}: {count:3d}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fix chordmap normalization bugs")
    parser.add_argument(
        "song_dir", help="Song directory (e.g., data/suno_ai/suno_themesong/song_004)"
    )

    args = parser.parse_args()

    song_dir = Path(args.song_dir)
    analysis_dir = song_dir / "analysis"

    locked_path = analysis_dir / "chordmap_locked.json"
    m21_path = analysis_dir / "chordmap_m21.json"

    if not locked_path.exists():
        print(f"❌ Not found: {locked_path}")
        sys.exit(1)

    print("=" * 60)
    print("Phase 1: Fix chordmap_locked.json")
    print("=" * 60)
    fix_chordmap_locked(locked_path)

    print("\n" + "=" * 60)
    print("Phase 2: Regenerate chordmap_m21.json")
    print("=" * 60)
    regenerate_m21_chordmap(locked_path, m21_path)
