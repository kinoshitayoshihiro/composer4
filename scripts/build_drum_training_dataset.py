#!/usr/bin/env python3
"""Drum Training Dataset Construction (Task 5)

drum_patterns_labeled.parquetから学習用データセットを構築。

Processing:
1. Parquet読み込み（song_id, family, features）
2. 特徴量エンジニアリング:
   - tempo_bpm, slots, section_encoded
   - density_k/s/h, syncopation
   - kick_downbeat_rate, snare_backbeat_rate (計算)
   - swing_hint (triplet detection)
3. 曲単位でTrain/Val/Test分割（リーク防止）
4. 3つのParquet出力

Output:
- train.parquet (70%)
- val.parquet (15%)
- test.parquet (15%)
- dataset_info.json (統計・メタデータ)

Usage:
    python build_drum_training_dataset.py \\
        --input drum_patterns_labeled.parquet \\
        --output-dir data/drums_training/ \\
        --split-ratio 0.7 0.15 0.15
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ===== Feature Engineering =====

def calculate_kick_downbeat_rate(kick_vec: list[float], slots: int) -> float:
    """キックのダウンビート命中率
    
    slots=16: 0, 4, 8, 12がダウンビート
    slots=24: 0, 6, 12, 18がダウンビート
    """
    if not kick_vec:
        return 0.0
    
    if slots == 16:
        downbeats = [0, 4, 8, 12]
    elif slots == 24:
        downbeats = [0, 6, 12, 18]
    else:
        return 0.0
    
    hits = sum(1 for i in downbeats if i < len(kick_vec) and kick_vec[i] > 0.0)
    return hits / len(downbeats)


def calculate_snare_backbeat_rate(snare_vec: list[float], slots: int) -> float:
    """スネアのバックビート命中率
    
    slots=16: 4, 12がバックビート（2拍目・4拍目）
    slots=24: 6, 18がバックビート
    """
    if not snare_vec:
        return 0.0
    
    if slots == 16:
        backbeats = [4, 12]
    elif slots == 24:
        backbeats = [6, 18]
    else:
        return 0.0
    
    hits = sum(1 for i in backbeats if i < len(snare_vec) and snare_vec[i] > 0.0)
    return hits / len(backbeats)


def detect_swing_hint(hat_vec: list[float], slots: int) -> float:
    """Swing/Triplet検出
    
    slots=24: 明確な3連符 → 0.33
    slots=16で不均等: シャッフル → 0.20
    straight: 0.0
    
    Returns:
        0.0 (straight) ~ 0.33 (triplet)
    """
    if slots == 24:
        # 3連符拍子
        return 0.33
    
    if slots == 16:
        # 偶数位置 vs 奇数位置の密度差
        if not hat_vec:
            return 0.0
        even_density = sum(1 for i in range(0, len(hat_vec), 2) if hat_vec[i] > 0.0)
        odd_density = sum(1 for i in range(1, len(hat_vec), 2) if hat_vec[i] > 0.0)
        total = even_density + odd_density
        if total == 0:
            return 0.0
        imbalance = abs(even_density - odd_density) / total
        # 不均等が大きい場合はシャッフル感
        if imbalance > 0.3:
            return 0.20
    
    return 0.0


def encode_section(section: str) -> int:
    """セクション名→数値エンコード"""
    mapping = {
        "Chorus": 0,
        "Verse": 1,
        "Bridge": 2,
        "Intro": 3,
        "Outro": 4,
        "Solo": 5,
        "Unknown": 6,
    }
    return mapping.get(section, 6)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """特徴量エンジニアリング
    
    Input columns:
        song_id, bar_index, slots, tempo_bpm, time_sig,
        kick_vec, snare_vec, hat_vec (JSON),
        density_k/s/h, syncopation,
        pattern_id_normalized, family
    
    Added columns:
        section_encoded, kick_downbeat_rate, snare_backbeat_rate, swing_hint
    """
    logger.info("Engineering features for %d patterns...", len(df))
    
    # JSON→list変換
    kick_vecs = df["kick_vec"].apply(
        lambda x: json.loads(x) if isinstance(x, str) else x
    )
    snare_vecs = df["snare_vec"].apply(
        lambda x: json.loads(x) if isinstance(x, str) else x
    )
    hat_vecs = df["hat_vec"].apply(
        lambda x: json.loads(x) if isinstance(x, str) else x
    )
    
    # 新規特徴量計算
    df["kick_downbeat_rate"] = [
        calculate_kick_downbeat_rate(kv, s)
        for kv, s in zip(kick_vecs, df["slots"])
    ]
    df["snare_backbeat_rate"] = [
        calculate_snare_backbeat_rate(sv, s)
        for sv, s in zip(snare_vecs, df["slots"])
    ]
    df["swing_hint"] = [
        detect_swing_hint(hv, s)
        for hv, s in zip(hat_vecs, df["slots"])
    ]
    
    # セクションエンコード
    df["section_encoded"] = df.get("section", "Unknown").apply(encode_section)
    
    logger.info("Feature engineering complete.")
    return df


# ===== Train/Val/Test Split =====

def split_by_song(
    df: pd.DataFrame,
    split_ratio: tuple[float, float, float] = (0.7, 0.15, 0.15),
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """曲単位でTrain/Val/Test分割（データリーク防止）
    
    Args:
        df: 全パターンDataFrame
        split_ratio: (train, val, test)比率
        seed: 乱数シード
    
    Returns:
        (train_df, val_df, test_df)
    """
    np.random.seed(seed)
    
    # 曲IDリスト取得
    song_ids = df["song_id"].unique()
    np.random.shuffle(song_ids)
    
    # 分割点計算
    n_songs = len(song_ids)
    train_end = int(n_songs * split_ratio[0])
    val_end = train_end + int(n_songs * split_ratio[1])
    
    train_songs = set(song_ids[:train_end])
    val_songs = set(song_ids[train_end:val_end])
    test_songs = set(song_ids[val_end:])
    
    logger.info(
        "Split songs: Train=%d, Val=%d, Test=%d",
        len(train_songs),
        len(val_songs),
        len(test_songs),
    )
    
    # DataFrame分割
    train_df = df[df["song_id"].isin(train_songs)].copy()
    val_df = df[df["song_id"].isin(val_songs)].copy()
    test_df = df[df["song_id"].isin(test_songs)].copy()
    
    logger.info(
        "Split patterns: Train=%d, Val=%d, Test=%d",
        len(train_df),
        len(val_df),
        len(test_df),
    )
    
    return train_df, val_df, test_df


# ===== Dataset Info Generation =====

def generate_dataset_info(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> dict[str, Any]:
    """データセット統計・メタデータ生成
    
    Returns:
        {
            "total_patterns": int,
            "total_songs": int,
            "split": {"train": {...}, "val": {...}, "test": {...}},
            "family_distribution": {...},
            "feature_ranges": {...},
        }
    """
    all_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    
    total_patterns = len(all_df)
    total_songs = all_df["song_id"].nunique()
    
    split_info = {}
    for name, df_split in [("train", train_df), ("val", val_df), ("test", test_df)]:
        split_info[name] = {
            "num_patterns": len(df_split),
            "num_songs": df_split["song_id"].nunique(),
            "family_counts": df_split["family"].value_counts().to_dict(),
        }
    
    # Family分布（全体）
    family_dist = all_df["family"].value_counts().to_dict()
    
    # 特徴量範囲
    feature_cols = [
        "tempo_bpm",
        "slots",
        "density_k",
        "density_s",
        "density_h",
        "syncopation",
        "kick_downbeat_rate",
        "snare_backbeat_rate",
        "swing_hint",
    ]
    feature_ranges = {}
    for col in feature_cols:
        if col in all_df.columns:
            feature_ranges[col] = {
                "min": float(all_df[col].min()),
                "max": float(all_df[col].max()),
                "mean": float(all_df[col].mean()),
                "std": float(all_df[col].std()),
            }
    
    return {
        "total_patterns": total_patterns,
        "total_songs": total_songs,
        "split": split_info,
        "family_distribution": family_dist,
        "feature_ranges": feature_ranges,
    }


# ===== Main Pipeline =====

def build_dataset(
    input_parquet: Path,
    output_dir: Path,
    split_ratio: tuple[float, float, float] = (0.7, 0.15, 0.15),
    seed: int = 42,
) -> None:
    """メイン処理: 学習用データセット構築
    
    Args:
        input_parquet: drum_patterns_labeled.parquet
        output_dir: 出力ディレクトリ
        split_ratio: (train, val, test)比率
        seed: 乱数シード
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. 読み込み
    logger.info("Loading labeled patterns from %s...", input_parquet)
    df = pd.read_parquet(input_parquet)
    logger.info("Loaded %d patterns from %d songs.", len(df), df["song_id"].nunique())
    
    # 2. 特徴量エンジニアリング
    df = engineer_features(df)
    
    # 3. Train/Val/Test分割
    train_df, val_df, test_df = split_by_song(df, split_ratio, seed)
    
    # 4. 保存
    train_path = output_dir / "train.parquet"
    val_path = output_dir / "val.parquet"
    test_path = output_dir / "test.parquet"
    info_path = output_dir / "dataset_info.json"
    
    logger.info("Saving train dataset to %s...", train_path)
    train_df.to_parquet(train_path, index=False)
    
    logger.info("Saving val dataset to %s...", val_path)
    val_df.to_parquet(val_path, index=False)
    
    logger.info("Saving test dataset to %s...", test_path)
    test_df.to_parquet(test_path, index=False)
    
    # 5. メタデータ
    dataset_info = generate_dataset_info(train_df, val_df, test_df)
    logger.info("Saving dataset info to %s...", info_path)
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(dataset_info, f, indent=2, ensure_ascii=False)
    
    logger.info("Dataset construction complete.")
    logger.info("  Train: %d patterns (%d songs)", len(train_df), train_df["song_id"].nunique())
    logger.info("  Val: %d patterns (%d songs)", len(val_df), val_df["song_id"].nunique())
    logger.info("  Test: %d patterns (%d songs)", len(test_df), test_df["song_id"].nunique())


# ===== CLI =====

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build drum training dataset with Train/Val/Test split (Task 5)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input parquet: drum_patterns_labeled.parquet",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for train/val/test parquet files",
    )
    parser.add_argument(
        "--split-ratio",
        type=float,
        nargs=3,
        default=[0.7, 0.15, 0.15],
        help="Train/Val/Test split ratio (default: 0.7 0.15 0.15)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for split (default: 42)",
    )
    
    args = parser.parse_args()
    
    if not args.input.exists():
        logger.error("Input file not found: %s", args.input)
        return 1
    
    # 比率検証
    split_ratio = tuple(args.split_ratio)
    if abs(sum(split_ratio) - 1.0) > 1e-6:
        logger.error("Split ratios must sum to 1.0 (got %.3f)", sum(split_ratio))
        return 1
    
    try:
        build_dataset(
            input_parquet=args.input,
            output_dir=args.output_dir,
            split_ratio=split_ratio,
            seed=args.seed,
        )
        return 0
    except Exception as exc:
        logger.exception("Failed to build dataset: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
