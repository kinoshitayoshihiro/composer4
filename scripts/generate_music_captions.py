#!/usr/bin/env python3
"""
Stage3 Step2: MetaScore Music Caption Generation

Stage2メトリクス + 感情/ジャンルラベルから、
簡潔な日本語音楽キャプションを生成。

Usage:
    python scripts/generate_music_captions.py \
        --labeled-summary outputs/stage3/loop_summary_labeled.csv \
        --output outputs/stage3/loop_summary_with_captions.csv \
        --mode template  # or 'llm' for GPT-based generation
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


class CaptionGenerator:
    """音楽キャプション生成器"""

    # テンプレートベース生成用のパターン
    EMOTION_PHRASES = {
        "calm": ["落ち着いた", "穏やかな", "静かな", "やさしい"],
        "warm": ["温かい", "心地よい", "柔らかな", "優しい"],
        "sad": ["悲しげな", "物憂げな", "切ない", "哀愁の"],
        "happy": ["明るい", "楽しい", "陽気な", "ハッピーな"],
        "tense": ["緊張感のある", "張り詰めた", "緊迫した", "ドキドキする"],
        "intense": ["激しい", "パワフルな", "エネルギッシュな", "力強い"],
        "dark": ["暗い", "重厚な", "荘厳な", "ダークな"],
        "bright": ["明るい", "輝く", "華やかな", "光り輝く"],
    }

    GENRE_PHRASES = {
        "rock": ["ロック", "ロックビート", "ロックサウンド"],
        "pop": ["ポップ", "ポップビート", "キャッチー"],
        "jazz": ["ジャズ", "スウィング", "ジャジー"],
        "funk": ["ファンク", "グルーヴィー", "ファンキー"],
        "soul": ["ソウル", "ソウルフル", "深みのある"],
        "edm": ["EDM", "エレクトロ", "ダンス"],
        "hiphop": ["ヒップホップ", "アーバン", "ビート"],
        "ballad": ["バラード", "しっとりした", "叙情的"],
        "orchestral": ["オーケストラ", "壮大な", "シンフォニック"],
    }

    TEMPO_DESCRIPTORS = {
        "slow": "ゆったりとした",
        "mid": "ミディアムテンポの",
        "fast": "速いテンポの",
        "very_fast": "超高速な",
    }

    STRUCTURE_DESCRIPTORS = {
        "simple": "シンプルな",
        "moderate": "程よく複雑な",
        "complex": "複雑な",
    }

    def __init__(self, mode: str = "template"):
        self.mode = mode

    def _classify_tempo(self, bpm: float) -> str:
        """BPMをテンポ記述子に分類"""
        if bpm < 90:
            return "slow"
        elif bpm < 120:
            return "mid"
        elif bpm < 150:
            return "fast"
        else:
            return "very_fast"

    def _classify_structure(self, metrics: Dict[str, Any]) -> str:
        """構造の複雑さを分類"""
        periodicity = float(metrics.get("axes_raw.structure", 0.5))
        if periodicity < 0.3:
            return "complex"
        elif periodicity < 0.6:
            return "moderate"
        else:
            return "simple"

    def generate_template_caption(self, row: Dict[str, Any]) -> str:
        """テンプレートベースでキャプション生成"""

        emotion = row.get("label.emotion", "calm")
        genre = row.get("label.genre", "pop")
        bpm = float(row.get("bpm", 120.0))

        # フレーズ選択
        emotion_phrase = random.choice(self.EMOTION_PHRASES.get(emotion, [""])) or ""
        genre_phrase = random.choice(self.GENRE_PHRASES.get(genre, [""])) or genre
        tempo_class = self._classify_tempo(bpm)
        tempo_phrase = self.TEMPO_DESCRIPTORS[tempo_class]

        # 特徴的なメトリクス
        swing_ratio = float(row.get("metrics.swing_ratio", 0.5))
        ghost_rate = float(row.get("metrics.ghost_rate", 0.0))

        # キャプション構築パターン
        patterns = [
            f"{emotion_phrase}{tempo_phrase}{genre_phrase}ビート",
            f"{tempo_phrase}{emotion_phrase}{genre_phrase}グルーヴ",
            f"{genre_phrase}スタイルの{emotion_phrase}ドラムパターン",
        ]

        caption = random.choice(patterns)

        # スウィング・ゴーストノート特徴を追加
        if swing_ratio > 0.55:
            caption += "、スウィング感あり"
        if ghost_rate > 0.3:
            caption += "、ゴーストノート多め"

        return caption

    def generate_llm_caption(self, row: Dict[str, Any]) -> str:
        """LLMベースでキャプション生成 (プレースホルダー)"""
        # TODO: OpenAI API統合
        # このバージョンではテンプレートにフォールバック
        return self.generate_template_caption(row)

    def generate(self, row: Dict[str, Any]) -> str:
        """キャプション生成 (モード自動選択)"""
        if self.mode == "llm":
            return self.generate_llm_caption(row)
        else:
            return self.generate_template_caption(row)


def main():
    parser = argparse.ArgumentParser(description="Generate music captions from labeled loops")
    parser.add_argument(
        "--labeled-summary", type=Path, required=True, help="Input CSV with emotion/genre labels"
    )
    parser.add_argument("--output", type=Path, required=True, help="Output CSV with captions")
    parser.add_argument(
        "--mode", choices=["template", "llm"], default="template", help="Caption generation mode"
    )
    parser.add_argument(
        "--sample", type=int, help="Generate captions for only N samples (for testing)"
    )

    args = parser.parse_args()

    print(f"Generating captions in '{args.mode}' mode")
    generator = CaptionGenerator(mode=args.mode)

    print(f"Reading labeled summary from {args.labeled_summary}")
    rows = []
    with args.labeled_summary.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames)

        for i, row in enumerate(reader):
            if args.sample and i >= args.sample:
                break

            # キャプション生成
            caption = generator.generate(row)
            row["label.caption"] = caption
            rows.append(row)

            if (i + 1) % 100 == 0:
                print(f"  Generated {i+1} captions...")

    # 新しいカラム追加
    if "label.caption" not in fieldnames:
        fieldnames.append("label.caption")

    # 出力
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"✅ Generated captions for {len(rows)} loops")
    print(f"   Output: {args.output}")

    # サンプル表示
    if rows:
        print("\n📝 Sample captions:")
        for row in rows[:5]:
            emotion = row.get("label.emotion", "")
            genre = row.get("label.genre", "")
            caption = row.get("label.caption", "")
            print(f"  [{emotion}/{genre}] {caption}")


if __name__ == "__main__":
    main()
