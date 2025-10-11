#!/usr/bin/env python3
"""
Stage3 Step1: XMIDI Emotion/Genre Assignment

XMIDIベースの感情・ジャンル教師を使い、Stage2メトリクスから
valence/arousal, emotion, genreを自動推定してloop_summaryに追加。

Usage:
    python scripts/assign_labels.py \
        --loop-summary output/drumloops_stage2/loop_summary.csv \
        --schema configs/labels/labels_schema.yaml \
        --output output/drumloops_stage2/loop_summary_labeled.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml
import numpy as np
from dataclasses import dataclass


@dataclass
class EmotionProfile:
    """感情プロファイル (Valence/Arousal)"""

    valence: float  # -1.0 to 1.0
    arousal: float  # 0.0 to 1.0
    confidence: float = 0.0

    def to_emotion_label(self) -> str:
        """V/A座標から感情ラベルを決定"""
        # XMIDI準拠の8分類マッピング
        if self.valence > 0.3:
            if self.arousal > 0.7:
                return "intense"  # 高V高A
            elif self.arousal > 0.4:
                return "happy"  # 高V中A
            else:
                return "warm"  # 高V低A
        elif self.valence > -0.3:
            if self.arousal > 0.6:
                return "tense"  # 中V高A
            else:
                return "calm"  # 中V低A
        else:
            if self.arousal > 0.5:
                return "dark"  # 低V高A
            else:
                return "sad"  # 低V低A

    def to_dict(self) -> Dict[str, Any]:
        return {
            "valence": round(self.valence, 3),
            "arousal": round(self.arousal, 3),
            "confidence": round(self.confidence, 3),
            "label": self.to_emotion_label(),
        }


class LabelAssigner:
    """メトリクスからラベルを推定"""

    def __init__(self, schema_path: Path):
        with schema_path.open("r", encoding="utf-8") as f:
            self.schema = yaml.safe_load(f)

        self.emotion_classes = self.schema["schema"]["emotion"]["classes"]
        self.genre_classes = self.schema["schema"]["genre"]["classes"]

    def estimate_emotion(self, metrics: Dict[str, Any]) -> EmotionProfile:
        """メトリクスからvalence/arousalを推定"""

        # Arousal推定 (エネルギー・緊張度)
        velocity_range = float(metrics.get("metrics.velocity_range", 0.5))
        microtiming_rms = float(metrics.get("metrics.microtiming_rms", 10.0))
        accent_rate = float(metrics.get("metrics.accent_rate", 0.1))

        # 正規化: velocity_range [0-1], microtiming_rms [0-30], accent_rate [0-0.5]
        arousal_velocity = min(1.0, velocity_range)
        arousal_timing = min(1.0, microtiming_rms / 30.0)
        arousal_accent = min(1.0, accent_rate / 0.3)

        arousal = arousal_velocity * 0.5 + arousal_timing * 0.3 + arousal_accent * 0.2
        arousal = max(0.0, min(1.0, arousal))

        # Valence推定 (ポジティブ/ネガティブ)
        swing_ratio = float(metrics.get("metrics.swing_ratio", 0.5))
        dynamics_range = float(metrics.get("metrics.dynamics_range", 0.5))

        # スウィング率が高い→ポジティブ傾向
        # ダイナミクスレンジが広い→感情豊か
        valence_swing = (swing_ratio - 0.5) * 1.5  # 0.5を中心に±0.75
        valence_dynamics = (dynamics_range - 0.5) * 0.8

        valence = valence_swing + valence_dynamics
        valence = max(-1.0, min(1.0, valence))

        # 信頼度: メトリクス欠損が少ないほど高い
        available_metrics = sum(
            [
                1 if "velocity_range" in str(metrics) else 0,
                1 if "microtiming_rms" in str(metrics) else 0,
                1 if "swing_ratio" in str(metrics) else 0,
                1 if "accent_rate" in str(metrics) else 0,
            ]
        )
        confidence = available_metrics / 4.0

        return EmotionProfile(valence=valence, arousal=arousal, confidence=confidence)

    def estimate_genre(self, metrics: Dict[str, Any]) -> Tuple[str, float]:
        """メトリクスからジャンルを推定"""

        bpm = float(metrics.get("bpm", 120.0))
        swing_ratio = float(metrics.get("metrics.swing_ratio", 0.5))
        ride_rate = float(metrics.get("metrics.ride_usage_ratio", 0.0))
        hat_ratio = float(metrics.get("metrics.hihat_ratio", 0.0))

        scores: Dict[str, float] = {}

        # Jazz: swing高い、ride使用多い
        if swing_ratio > 0.55 and ride_rate > 0.10:
            scores["jazz"] = 0.7 + (swing_ratio - 0.55) + ride_rate

        # Rock: BPM 90-160、hihat多い、even寄り
        if 90 <= bpm <= 160 and hat_ratio > 0.20 and swing_ratio < 0.55:
            scores["rock"] = 0.6 + (hat_ratio * 0.5)

        # Funk: BPM 100-120、swing中程度
        if 100 <= bpm <= 120 and 0.50 <= swing_ratio <= 0.60:
            scores["funk"] = 0.5 + (0.55 - abs(swing_ratio - 0.55)) * 2

        # EDM: BPM高い、evenビート
        if bpm > 120 and swing_ratio < 0.52:
            scores["edm"] = 0.4 + ((bpm - 120) / 60.0)

        # Hiphop: BPM 80-100、swing低め
        if 80 <= bpm <= 100 and swing_ratio < 0.54:
            scores["hiphop"] = 0.5 + (100 - bpm) / 40.0

        # デフォルト: pop
        if not scores:
            scores["pop"] = 0.3

        # 最高スコアのジャンルを選択
        best_genre = max(scores.items(), key=lambda x: x[1])
        confidence = min(1.0, best_genre[1])

        return best_genre[0], confidence

    def assign_labels(self, row: Dict[str, Any]) -> Dict[str, Any]:
        """1行のloop_summaryにラベルを追加"""

        # 感情推定
        emotion = self.estimate_emotion(row)
        row["label.emotion"] = emotion.to_emotion_label()
        row["emotion.valence"] = emotion.valence
        row["emotion.arousal"] = emotion.arousal
        row["emotion.confidence"] = emotion.confidence

        # ジャンル推定
        genre, genre_conf = self.estimate_genre(row)
        row["label.genre"] = genre
        row["genre.confidence"] = genre_conf

        return row


def main():
    parser = argparse.ArgumentParser(description="Assign XMIDI-based emotion/genre labels")
    parser.add_argument(
        "--loop-summary", type=Path, required=True, help="Input loop_summary.csv from Stage2"
    )
    parser.add_argument(
        "--schema",
        type=Path,
        default=Path("configs/labels/labels_schema.yaml"),
        help="Label schema YAML",
    )
    parser.add_argument("--output", type=Path, required=True, help="Output CSV with labels")
    parser.add_argument("--stats-output", type=Path, help="Optional JSON stats output")

    args = parser.parse_args()

    print(f"Loading schema from {args.schema}")
    assigner = LabelAssigner(args.schema)

    print(f"Reading loop summary from {args.loop_summary}")
    rows = []
    with args.loop_summary.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames

        for row in reader:
            labeled_row = assigner.assign_labels(row)
            rows.append(labeled_row)

    # 新しいカラム追加
    new_fieldnames = list(fieldnames) + [
        "label.emotion",
        "emotion.valence",
        "emotion.arousal",
        "emotion.confidence",
        "label.genre",
        "genre.confidence",
    ]

    # 出力
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=new_fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"✅ Wrote {len(rows)} labeled loops to {args.output}")

    # 統計出力
    if args.stats_output:
        from collections import Counter

        emotion_dist = Counter(row["label.emotion"] for row in rows)
        genre_dist = Counter(row["label.genre"] for row in rows)

        avg_valence = np.mean([float(row["emotion.valence"]) for row in rows])
        avg_arousal = np.mean([float(row["emotion.arousal"]) for row in rows])

        stats = {
            "total_loops": len(rows),
            "emotion_distribution": dict(emotion_dist),
            "genre_distribution": dict(genre_dist),
            "average_valence": round(avg_valence, 3),
            "average_arousal": round(avg_arousal, 3),
        }

        with args.stats_output.open("w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)

        print(f"📊 Stats saved to {args.stats_output}")
        print(f"\nEmotion distribution:")
        for emo, count in emotion_dist.most_common():
            print(f"  {emo:12s}: {count:4d} ({count/len(rows)*100:.1f}%)")
        print(f"\nGenre distribution:")
        for gen, count in genre_dist.most_common():
            print(f"  {gen:12s}: {count:4d} ({count/len(rows)*100:.1f}%)")


if __name__ == "__main__":
    main()
