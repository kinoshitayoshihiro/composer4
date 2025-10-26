#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A/B Test: Guitar Stage2 v1 (Rule-based) vs v2 (XGBoost)

評価指標:
- パターン一致率: pattern_id_v1 == pattern_id_v2
- 演奏密度差: |notes_v2 - notes_v1| / bar
- アクセント整合: accent_grid との一致率
- 和声整合: root/quality に対する禁則チェック

合格ライン:
- パターン一致率 >= 65%
- アクセント整合 +5% 以上 (v1比)
- 演奏密度差中央値 <= 1 ノート/小節
"""

import os
import sys
import logging
import random
import pickle
import pandas as pd
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Any
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def load_song_packages(
    limit: int = 50, index_csv: str = "song_packages_index.csv"
) -> List[Dict[str, Any]]:
    """ランダム50曲抽出 (Gold/Silver優先)"""
    if not Path(index_csv).exists():
        logger.error(f"Index CSV not found: {index_csv}")
        return []

    # CSVから読み込み
    df = pd.read_csv(index_csv)
    logger.info(f"Found {len(df)} packages in index")

    # label_strength を quality にリネーム
    if "label_strength" in df.columns:
        df["quality"] = df["label_strength"]

    # Quality優先ソート（Gold > Silver > Bronze）
    quality_order = {"gold": 0, "silver": 1, "bronze": 2}
    df["quality_rank"] = df["quality"].map(lambda q: quality_order.get(str(q).lower(), 3))
    df = df.sort_values("quality_rank")

    # 上位limit曲をランダムサンプル（Gold/Silver中心）
    gold_silver = df[df["quality"].isin(["gold", "silver"])]
    if len(gold_silver) >= limit:
        sampled = gold_silver.sample(n=limit, random_state=42)
    else:
        sampled = df.sample(n=min(len(df), limit), random_state=42)

    packages = []
    for _, row in sampled.iterrows():
        packages.append(
            {"song_id": row["song_id"], "path": row["package_path"], "quality": row["quality"]}
        )

    logger.info(f"Sampled {len(packages)} songs for A/B test")
    quality_counts = sampled["quality"].value_counts()
    for q, cnt in quality_counts.items():
        logger.info(f"  {q}: {cnt}")

    return packages


def get_pattern_from_recommender(
    recommender,
    section: str,
    chord_root: str,
    chord_quality: str,
    tempo: float,
    confidence: float,
    time_sig: str = "4/4",
) -> Dict[str, Any]:
    """Recommender からパターン取得"""
    try:
        pattern = recommender.get_pattern(
            section=section,
            chord_root=chord_root,
            chord_quality=chord_quality,
            tempo=tempo,
            confidence=confidence,
            time_sig=time_sig,
        )
        return pattern or {}
    except Exception as e:
        logger.warning(f"Pattern fetch failed: {e}")
        return {}


def compute_note_density(pattern: Dict[str, Any]) -> float:
    """演奏密度（ノート数/小節）"""
    rhythm = pattern.get("rhythm", "standard_quarter")
    density_map = {
        "standard_quarter": 4.0,
        "standard_8ths": 8.0,
        "standard_16ths": 16.0,
        "arp_8ths": 8.0,
        "arp_16ths": 16.0,
        "strum_down_8ths": 8.0,
        "strum_alt_16ths": 16.0,
        "sparse_half": 2.0,
        "sparse_whole": 1.0,
        "syncopated_8ths": 6.0,
    }
    return density_map.get(rhythm, 4.0)


def compute_accent_match(pattern: Dict[str, Any], accent_grid: List[int]) -> float:
    """アクセント整合（簡易版: rhythm と accent_grid の一致）"""
    rhythm = pattern.get("rhythm", "standard_quarter")

    # accent_grid: [0, 1, 2, 3, ...] (16分音符単位、0=強拍、4/8/12=中拍)
    strong_beats = {0, 4, 8, 12}  # 4/4拍子の強拍

    # rhythm がアクセント位置で発音しているか（ヒューリスティック）
    if "quarter" in rhythm or "half" in rhythm:
        # 強拍重視 → 良好
        return 0.9
    elif "16ths" in rhythm:
        # 過密 → 裏拍も多い → やや低下
        return 0.7
    elif "syncopated" in rhythm:
        # シンコペ → 強拍外し → 低下
        return 0.6
    else:
        return 0.8


def check_harmonic_rules(pattern: Dict[str, Any], chord_quality: str) -> bool:
    """和声禁則チェック（簡易版）"""
    voicing = pattern.get("voicing", [0, 4, 7])

    # Minor系でM3禁止
    if chord_quality in ["min", "min7", "min9"]:
        if 4 in voicing:  # M3
            return False

    # Maj系でm3禁止
    if chord_quality in ["maj", "maj7", "maj9"]:
        if 3 in voicing:  # m3
            return False

    return True


def run_ab_test(
    v1_pickle_path: str,
    v2_pickle_path: str,
    songs: List[Dict[str, Any]],
    output_csv: str = "data/ab_test_guitar_results.csv",
) -> pd.DataFrame:
    """A/Bテスト実行"""

    # v1 / v2 Recommender 初期化
    logger.info(f"Loading v1 pickle: {v1_pickle_path}")
    logger.info(f"Loading v2 pickle: {v2_pickle_path}")

    from ml.simple_pattern_recommender import SimplePatternRecommender

    recommender_v1 = SimplePatternRecommender(instrument="guitar", patterns_path=v1_pickle_path)

    recommender_v2 = SimplePatternRecommender(instrument="guitar", patterns_path=v2_pickle_path)

    logger.info(f"v1 selector type: {recommender_v1.selector.get('type')}")
    logger.info(f"v2 selector type: {recommender_v2.selector.get('type')}")

    # テストケース生成
    test_cases = []
    for song in songs:
        # ダミーコード進行（Chorus, C:maj7, 120bpm）
        # 実際は song_package.yaml から chordmap を読み込む
        for section in ["Intro", "Verse", "Chorus", "Bridge"]:
            for chord_root in ["C", "G", "Am", "F"]:
                for chord_quality in ["maj", "maj7", "min", "min7"]:
                    test_cases.append(
                        {
                            "song_id": song["song_id"],
                            "section": section,
                            "chord_root": chord_root,
                            "chord_quality": chord_quality,
                            "tempo": 120.0,
                            "confidence": 0.8,
                            "time_sig": "4/4",
                        }
                    )

    logger.info(f"Generated {len(test_cases)} test cases")

    # A/B評価
    results = []
    for tc in test_cases:
        pattern_v1 = get_pattern_from_recommender(
            recommender_v1,
            tc["section"],
            tc["chord_root"],
            tc["chord_quality"],
            tc["tempo"],
            tc["confidence"],
            tc["time_sig"],
        )

        pattern_v2 = get_pattern_from_recommender(
            recommender_v2,
            tc["section"],
            tc["chord_root"],
            tc["chord_quality"],
            tc["tempo"],
            tc["confidence"],
            tc["time_sig"],
        )

        # 評価指標
        pattern_id_v1 = pattern_v1.get("pattern_id", "unknown")
        pattern_id_v2 = pattern_v2.get("pattern_id", "unknown")
        pattern_match = int(pattern_id_v1 == pattern_id_v2)

        density_v1 = compute_note_density(pattern_v1)
        density_v2 = compute_note_density(pattern_v2)
        density_diff = abs(density_v2 - density_v1)

        accent_grid = list(range(16))  # ダミー
        accent_match_v1 = compute_accent_match(pattern_v1, accent_grid)
        accent_match_v2 = compute_accent_match(pattern_v2, accent_grid)

        harmonic_ok_v1 = check_harmonic_rules(pattern_v1, tc["chord_quality"])
        harmonic_ok_v2 = check_harmonic_rules(pattern_v2, tc["chord_quality"])

        results.append(
            {
                "song_id": tc["song_id"],
                "section": tc["section"],
                "chord_root": tc["chord_root"],
                "chord_quality": tc["chord_quality"],
                "tempo": tc["tempo"],
                "pattern_id_v1": pattern_id_v1,
                "pattern_id_v2": pattern_id_v2,
                "pattern_match": pattern_match,
                "density_v1": density_v1,
                "density_v2": density_v2,
                "density_diff": density_diff,
                "accent_match_v1": accent_match_v1,
                "accent_match_v2": accent_match_v2,
                "accent_delta": accent_match_v2 - accent_match_v1,
                "harmonic_ok_v1": int(harmonic_ok_v1),
                "harmonic_ok_v2": int(harmonic_ok_v2),
            }
        )

    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    logger.info(f"Results saved to {output_csv}")

    # 集計
    logger.info("\n" + "=" * 60)
    logger.info("A/B Test Results Summary")
    logger.info("=" * 60)

    pattern_match_rate = df["pattern_match"].mean() * 100
    logger.info(f"Pattern Match Rate: {pattern_match_rate:.2f}%")

    density_diff_median = df["density_diff"].median()
    logger.info(f"Density Diff (median): {density_diff_median:.2f} notes/bar")

    accent_delta_mean = df["accent_delta"].mean() * 100
    logger.info(f"Accent Match Delta (v2 - v1): {accent_delta_mean:+.2f}%")

    harmonic_ok_v1_rate = df["harmonic_ok_v1"].mean() * 100
    harmonic_ok_v2_rate = df["harmonic_ok_v2"].mean() * 100
    logger.info(f"Harmonic Rules OK (v1): {harmonic_ok_v1_rate:.2f}%")
    logger.info(f"Harmonic Rules OK (v2): {harmonic_ok_v2_rate:.2f}%")

    # 合格ライン判定
    logger.info("\n" + "-" * 60)
    logger.info("Pass/Fail Criteria")
    logger.info("-" * 60)

    pass_pattern_match = pattern_match_rate >= 65.0
    pass_accent_delta = accent_delta_mean >= 5.0
    pass_density_diff = density_diff_median <= 1.0

    logger.info(
        f"Pattern Match Rate >= 65%: {'PASS' if pass_pattern_match else 'FAIL'} ({pattern_match_rate:.2f}%)"
    )
    logger.info(
        f"Accent Delta >= +5%: {'PASS' if pass_accent_delta else 'FAIL'} ({accent_delta_mean:+.2f}%)"
    )
    logger.info(
        f"Density Diff <= 1 note/bar: {'PASS' if pass_density_diff else 'FAIL'} ({density_diff_median:.2f})"
    )

    all_pass = pass_pattern_match and pass_accent_delta and pass_density_diff
    logger.info("\n" + "=" * 60)
    logger.info(
        f"Overall: {'✓ PASS (v2 is ready for rollout)' if all_pass else '✗ FAIL (needs tuning or fallback)'}"
    )
    logger.info("=" * 60 + "\n")

    return df


def main():
    import argparse

    parser = argparse.ArgumentParser(description="A/B Test for Guitar Stage2 v1 vs v2")
    parser.add_argument(
        "--v1-pickle",
        type=str,
        default="data/patterns/stage2_guitar.pickle",
        help="Path to v1 pickle (rule-based)",
    )
    parser.add_argument(
        "--v2-pickle",
        type=str,
        default="data/patterns/stage2_guitar_v2.pickle",
        help="Path to v2 pickle (XGBoost)",
    )
    parser.add_argument("--num-songs", type=int, default=50, help="Number of songs to test")
    parser.add_argument(
        "--output", type=str, default="data/ab_test_guitar_results.csv", help="Output CSV path"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    # 曲リスト取得
    songs = load_song_packages(limit=args.num_songs)
    if not songs:
        logger.error("No songs found for testing")
        sys.exit(1)

    # A/Bテスト実行
    df = run_ab_test(
        v1_pickle_path=args.v1_pickle,
        v2_pickle_path=args.v2_pickle,
        songs=songs,
        output_csv=args.output,
    )

    logger.info(f"A/B test complete. Results: {args.output}")


if __name__ == "__main__":
    main()
