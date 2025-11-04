#!/usr/bin/env python3
"""Drum Pattern Family Labeling (Task 4)

groovesamplerパターンから教師ラベルを抽出するのではなく、
Safe-Kitベースの手動定義ルールを使用してパターン分類を実施。

Processing:
1. drum_patterns_normalized.parquet読み込み
2. 各パターンをルールベースで分類:
   - STRAIGHT_8: 8分ハット主体（straight feel）
   - STRAIGHT_16: 16分ハット主体
   - HALF_TIME: ハーフタイム感
   - TRIPLET_DRIVE: 3連符系（6/8拍子）
   - SHUFFLE: シャッフル（swing > 0.3）
   - FILL: フィル判定（タム/シンバル密度高）
3. family列追加してparquet出力

Output:
- drum_patterns_labeled.parquet
- family_distribution.json (統計)

Usage:
    python label_drum_families.py \\
        --input drum_patterns_normalized.parquet \\
        --output drum_patterns_labeled.parquet
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ===== Drum Family Classification Rules =====

def classify_drum_family(
    slots: int,
    tempo_bpm: float,
    density_k: float,
    density_s: float,
    density_h: float,
    syncopation: float,
    kick_vec: list[float],
    snare_vec: list[float],
    hat_vec: list[float],
    section: str,
) -> str:
    """ルールベースでドラムパターン分類
    
    Family Types:
    - STRAIGHT_8: 8分ハット主体、シンプルなビート
    - STRAIGHT_16: 16分ハット主体、エネルギッシュ
    - HALF_TIME: ハーフタイム感（スネア密度低）
    - TRIPLET_DRIVE: 3連符系（slots=24）
    - SHUFFLE: シャッフル（swing > 0.3）
    - FILL: フィル（タム/シンバル、高シンコペーション）
    - OTHER: その他
    
    Args:
        slots: 16 (4/4) or 24 (6/8)
        tempo_bpm: BPM
        density_k/s/h: キック/スネア/ハット密度
        syncopation: シンコペーション度 (0.0-1.0)
        kick_vec/snare_vec/hat_vec: アクセント配列
        section: Chorus/Verse/Bridge等
    
    Returns:
        Family name string
    """
    # Fill判定（高シンコペーション or 高ハット密度）
    if syncopation > 0.5 and density_h > 1.5:
        return "FILL"
    
    # 3連符系（6/8拍子）
    if slots == 24:
        if density_h > 0.8:
            return "TRIPLET_DRIVE"
        else:
            return "TRIPLET_SIMPLE"
    
    # 4/4拍子（slots=16）
    if slots == 16:
        # ハーフタイム判定（スネア密度低 < 0.3）
        if density_s < 0.3:
            return "HALF_TIME"
        
        # 16分ハット主体（密度 > 1.2）
        if density_h > 1.2:
            return "STRAIGHT_16"
        
        # 8分ハット主体（0.6 < 密度 <= 1.2）
        if 0.6 < density_h <= 1.2:
            return "STRAIGHT_8"
        
        # シンプル8ビート（低ハット密度 <= 0.6）
        if density_h <= 0.6:
            return "STRAIGHT_8_SIMPLE"
    
    return "OTHER"


def add_family_labels(df: pd.DataFrame) -> pd.DataFrame:
    """パターンDataFrameにfamily列追加
    
    Args:
        df: drum_patterns_normalized.parquet
    
    Returns:
        family列追加済みDataFrame
    """
    logger.info("Adding family labels to %d patterns...", len(df))
    
    families = []
    for idx, row in df.iterrows():
        # JSON文字列→リスト変換
        kick_vec = json.loads(row["kick_vec"]) if isinstance(row["kick_vec"], str) else row["kick_vec"]
        snare_vec = json.loads(row["snare_vec"]) if isinstance(row["snare_vec"], str) else row["snare_vec"]
        hat_vec = json.loads(row["hat_vec"]) if isinstance(row["hat_vec"], str) else row["hat_vec"]
        
        family = classify_drum_family(
            slots=row["slots"],
            tempo_bpm=row["tempo_bpm"],
            density_k=row["density_k"],
            density_s=row["density_s"],
            density_h=row["density_h"],
            syncopation=row["syncopation"],
            kick_vec=kick_vec,
            snare_vec=snare_vec,
            hat_vec=hat_vec,
            section=row.get("section", "Unknown"),
        )
        families.append(family)
    
    df["family"] = families
    logger.info("Family labeling complete.")
    return df


def compute_family_statistics(df: pd.DataFrame) -> dict[str, Any]:
    """Family分布統計計算
    
    Returns:
        {
            "total_patterns": int,
            "family_counts": {family: count},
            "family_ratios": {family: ratio},
            "avg_quality_by_family": {family: avg_quality},
        }
    """
    total = len(df)
    family_counts = df["family"].value_counts().to_dict()
    family_ratios = {k: v / total for k, v in family_counts.items()}
    
    # 平均品質（avg_qualityカラムがある場合）
    avg_quality_by_family = {}
    if "avg_quality" in df.columns:
        for family in family_counts.keys():
            subset = df[df["family"] == family]
            avg_quality_by_family[family] = float(subset["avg_quality"].mean())
    
    return {
        "total_patterns": total,
        "family_counts": family_counts,
        "family_ratios": family_ratios,
        "avg_quality_by_family": avg_quality_by_family,
    }


# ===== Main Pipeline =====

def label_patterns(
    input_parquet: Path,
    output_parquet: Path,
    output_stats: Path | None = None,
) -> None:
    """メイン処理: Family Labeling
    
    Args:
        input_parquet: drum_patterns_normalized.parquet
        output_parquet: drum_patterns_labeled.parquet
        output_stats: family_distribution.json (optional)
    """
    logger.info("Loading patterns from %s...", input_parquet)
    df = pd.read_parquet(input_parquet)
    logger.info("Loaded %d patterns.", len(df))
    
    # Family追加
    df = add_family_labels(df)
    
    # 統計計算
    stats = compute_family_statistics(df)
    logger.info("Family distribution:")
    for family, count in stats["family_counts"].items():
        ratio = stats["family_ratios"][family]
        logger.info("  %s: %d (%.2f%%)", family, count, ratio * 100)
    
    # 保存
    logger.info("Saving labeled patterns to %s...", output_parquet)
    df.to_parquet(output_parquet, index=False)
    
    if output_stats:
        logger.info("Saving statistics to %s...", output_stats)
        with open(output_stats, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
    
    logger.info("Family labeling complete.")


# ===== CLI =====

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Add drum family labels to normalized patterns (Task 4)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input parquet: drum_patterns_normalized.parquet",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output parquet: drum_patterns_labeled.parquet",
    )
    parser.add_argument(
        "--stats",
        type=Path,
        default=None,
        help="Output statistics JSON (optional)",
    )
    
    args = parser.parse_args()
    
    if not args.input.exists():
        logger.error("Input file not found: %s", args.input)
        return 1
    
    try:
        label_patterns(
            input_parquet=args.input,
            output_parquet=args.output,
            output_stats=args.stats,
        )
        return 0
    except Exception as exc:
        logger.exception("Failed to label patterns: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
