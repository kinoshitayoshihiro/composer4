"""
Lyric Anchor Indexer - Phase 2.0
Bar単位でlyric_anchorsをインデックス化し、phrase_roleとstress_levelを提供。
"""

from typing import Dict, Any, List
import json
from pathlib import Path


class LyricAnchorIndex:
    """
    lyric_anchors.jsonをbar単位でインデックス化。
    各barのphrase_role（start/mid/end）、stress_level、anchor存在を返す。
    """

    def __init__(self, anchors: Dict[str, Any], tempo_bpm: float = 120, beats_per_bar: int = 4):
        """
        Args:
            anchors: lyric_anchors.json内容
                {
                  "anchors": [
                    {"time": float, "classes": ["stress", "sibilant"], ...},
                    ...
                  ]
                }
            tempo_bpm: テンポ（BPM、デフォルト120）
            beats_per_bar: 1小節の拍数（デフォルト4）
        """
        self.by_bar = self._build_index(anchors, tempo_bpm, beats_per_bar)

    def _build_index(
        self, anchors: Dict[str, Any], tempo_bpm: float, beats_per_bar: int
    ) -> Dict[int, Dict[str, Any]]:
        """
        lyric_anchors（time-based）をbar単位で集約。

        Args:
            anchors: lyric_anchors.json内容
            tempo_bpm: BPM
            beats_per_bar: 1小節の拍数

        Returns:
            {
              bar_index: {
                "stress": max_stress,
                "phrase_role": "start"|"mid"|"end"|"none",
                "count": anchor_count,
                "has_vocal": True|False
              }
            }
        """
        result = {}
        anchor_list = anchors.get("anchors", [])

        # Time→Bar変換係数
        # 1 bar = beats_per_bar beats = (beats_per_bar / tempo_bpm) * 60 seconds
        bar_duration_sec = (beats_per_bar / tempo_bpm) * 60.0

        # Phrase boundary detection用
        last_stress_bar = -1

        for anchor in anchor_list:
            time = anchor.get("time", 0.0)
            bar = int(time / bar_duration_sec)

            classes = anchor.get("classes", [])

            info = result.setdefault(
                bar,
                {
                    "stress": 0.0,
                    "phrase_role": "mid",  # Default
                    "count": 0,
                    "has_vocal": False,
                    "stress_count": 0,
                },
            )

            # Stress detection
            if "stress" in classes:
                info["stress"] = max(info["stress"], 0.8)  # Stressあり→0.8
                info["stress_count"] += 1
                info["has_vocal"] = True

                # Phrase start detection（前回のstressから離れている）
                if last_stress_bar >= 0 and bar - last_stress_bar >= 2:
                    info["phrase_role"] = "start"

                last_stress_bar = bar

            # Sibilant detection（子音強調）
            if "sibilant" in classes:
                info["stress"] = max(info["stress"], 0.5)  # 軽いstress
                info["has_vocal"] = True

            info["count"] += 1

        # Phrase end detection（連続stressの最後）
        bars_sorted = sorted(result.keys())
        for i, bar in enumerate(bars_sorted):
            if i < len(bars_sorted) - 1:
                next_bar = bars_sorted[i + 1]
                if next_bar - bar >= 2:  # Gap検出
                    result[bar]["phrase_role"] = "end"

        return result

    def get_bar_info(self, bar: int) -> Dict[str, Any]:
        """
        指定barのlyric情報を取得。

        Args:
            bar: Bar index

        Returns:
            {
              "has_anchor": bool,
              "phrase_role": "start"|"mid"|"end"|"none",
              "stress_level": 0.0-1.0,
              "is_silent": bool  # Vocal無し（間奏など）
            }
        """
        base = self.by_bar.get(bar)
        if not base or base["count"] == 0:
            return {
                "has_anchor": False,
                "phrase_role": "none",
                "stress_level": 0.0,
                "is_silent": True,  # Anchor無し=vocal無し
            }

        return {
            "has_anchor": True,
            "phrase_role": base["phrase_role"],
            "stress_level": base["stress"],
            "is_silent": not base["has_vocal"],
        }

    @classmethod
    def from_file(
        cls, path: Path, tempo_bpm: float = 120, beats_per_bar: int = 4
    ) -> "LyricAnchorIndex":
        """
        lyric_anchors.jsonからインスタンス生成。

        Args:
            path: lyric_anchors.json path
            tempo_bpm: BPM（デフォルト120）
            beats_per_bar: 1小節の拍数（デフォルト4）

        Returns:
            LyricAnchorIndex instance
        """
        with open(path, "r", encoding="utf-8") as f:
            anchors = json.load(f)
        return cls(anchors, tempo_bpm, beats_per_bar)


def main():
    """テスト実行"""
    import sys

    if len(sys.argv) < 2:
        print("Usage: python3 lyric_index.py <lyric_anchors.json> [tempo_bpm]")
        sys.exit(1)

    tempo = int(sys.argv[2]) if len(sys.argv) > 2 else 120
    idx = LyricAnchorIndex.from_file(Path(sys.argv[1]), tempo_bpm=tempo)

    print("🎵 Lyric Anchor Index Test")
    print(f"Tempo: {tempo} BPM")
    print(f"Total bars with anchors: {len(idx.by_bar)}")
    print()

    if not idx.by_bar:
        print("⚠️  No anchors found (check lyric_anchors.json format)")
        return

    # 最初の10 barを表示
    max_bar = max(idx.by_bar.keys())
    for bar in range(min(10, max_bar + 1)):
        info = idx.get_bar_info(bar)
        if info["has_anchor"]:
            print(
                f"Bar {bar:2d}: role={info['phrase_role']:5s} stress={info['stress_level']:.2f} vocal={not info['is_silent']}"
            )
        else:
            print(f"Bar {bar:2d}: (no anchor)")


if __name__ == "__main__":
    main()
