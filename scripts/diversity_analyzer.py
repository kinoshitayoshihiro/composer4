#!/usr/bin/env python3
"""
Diversity Analyzer for Chord Progressions
==========================================

コード進行の多様性を分析し、同質化ペナルティを計算するツール

主要機能:
- N-gram ベースの多様性スコア算出
- コード進行の類似度計算
- Top-K 推薦での多様性フィルタリング
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple
import yaml


@dataclass
class DiversityConfig:
    """多様性計算の設定"""
    ngram_size: int = 3
    window_bars: int = 4
    similarity_threshold: float = 0.8
    penalty_weight: float = 0.15


def normalize_chord_name(chord: str) -> str:
    """コード名を正規化（異名同音を統一）"""
    # 基本的な正規化（拡張可能）
    chord = chord.strip()
    
    # 異名同音の統一
    replacements = {
        "C#": "Db", "D#": "Eb", "F#": "Gb",
        "G#": "Ab", "A#": "Bb"
    }
    
    for old, new in replacements.items():
        if chord.startswith(old):
            chord = new + chord[2:]
    
    return chord


def calculate_ngram_diversity(
    chords: List[str],
    n: int = 3
) -> float:
    """
    N-gram ベースの多様性スコア
    
    Args:
        chords: コードリスト ["C", "G", "Am", "F", ...]
        n: N-gram のサイズ（デフォルト3）
    
    Returns:
        多様性スコア (0.0-1.0)
        - 1.0 = 完全に多様（重複なし）
        - 0.0 = 完全に同質（すべて同じ）
    
    Examples:
        >>> calculate_ngram_diversity(["C", "G", "Am", "F"] * 4, n=3)
        0.0  # 完全な繰り返し
        
        >>> calculate_ngram_diversity(["C", "Dm", "Em", "F", "G", "Am", "Bdim", "C"], n=3)
        1.0  # すべてユニーク
    """
    if len(chords) < n:
        return 0.0
    
    # N-gramを抽出
    ngrams = []
    for i in range(len(chords) - n + 1):
        gram = tuple(normalize_chord_name(ch) for ch in chords[i:i+n])
        ngrams.append(gram)
    
    if not ngrams:
        return 0.0
    
    # ユニーク率を計算
    unique_count = len(set(ngrams))
    total_count = len(ngrams)
    
    diversity = unique_count / total_count
    return float(min(1.0, max(0.0, diversity)))


def calculate_chord_similarity(
    chords1: List[str],
    chords2: List[str]
) -> float:
    """
    2つのコード進行の類似度を計算
    
    Args:
        chords1: コード進行1
        chords2: コード進行2
    
    Returns:
        類似度 (0.0-1.0)
        - 1.0 = 完全一致
        - 0.0 = 完全に異なる
    
    計算方法:
    1. Jaccard係数（共通コード数 / 全コード数）
    2. 位置一致度（同じ位置に同じコードが出現する割合）
    3. 長さ類似度（コード進行の長さの類似性）
    """
    if not chords1 or not chords2:
        return 0.0
    
    # 正規化
    norm1 = [normalize_chord_name(ch) for ch in chords1]
    norm2 = [normalize_chord_name(ch) for ch in chords2]
    
    # 1. Jaccard係数
    set1, set2 = set(norm1), set(norm2)
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    jaccard = intersection / union if union > 0 else 0.0
    
    # 2. 位置一致度
    min_len = min(len(norm1), len(norm2))
    if min_len > 0:
        positional_matches = sum(
            1 for i in range(min_len) if norm1[i] == norm2[i]
        )
        positional_sim = positional_matches / min_len
    else:
        positional_sim = 0.0
    
    # 3. 長さ類似度
    len1, len2 = len(norm1), len(norm2)
    length_sim = min(len1, len2) / max(len1, len2) if max(len1, len2) > 0 else 0.0
    
    # 重み付き平均
    similarity = (
        jaccard * 0.4 +
        positional_sim * 0.5 +
        length_sim * 0.1
    )
    
    return float(min(1.0, max(0.0, similarity)))


def calculate_homogeneity_score(
    progressions: List[List[str]],
    config: DiversityConfig | None = None
) -> float:
    """
    複数のコード進行の同質化スコアを計算
    
    Args:
        progressions: コード進行のリスト [["C", "G", "Am", "F"], ...]
        config: 多様性計算の設定
    
    Returns:
        同質化スコア (0.0-1.0)
        - 1.0 = 完全に同質（すべて同じ）
        - 0.0 = 完全に多様（すべて異なる）
    """
    if not progressions or len(progressions) < 2:
        return 0.0
    
    config = config or DiversityConfig()
    
    # ペアごとの類似度を計算
    similarities = []
    for i in range(len(progressions)):
        for j in range(i + 1, len(progressions)):
            sim = calculate_chord_similarity(progressions[i], progressions[j])
            similarities.append(sim)
    
    if not similarities:
        return 0.0
    
    # 平均類似度 = 同質化スコア
    homogeneity = sum(similarities) / len(similarities)
    return float(min(1.0, max(0.0, homogeneity)))


def filter_diverse_progressions(
    progressions: List[Tuple[List[str], float]],
    top_k: int = 5,
    config: DiversityConfig | None = None
) -> List[Tuple[List[str], float]]:
    """
    Top-K 推薦で多様性を強制
    
    Args:
        progressions: (コード進行, 品質スコア) のリスト
        top_k: 返却する候補数
        config: 多様性計算の設定
    
    Returns:
        多様性を考慮してフィルタされた候補リスト
    
    アルゴリズム:
    1. 品質スコアでソート
    2. 上位候補を順次選択
    3. 各候補が既選択候補と類似度 > threshold なら除外
    4. top_k個選択するまで継続
    """
    config = config or DiversityConfig()
    
    if not progressions or top_k <= 0:
        return []
    
    # 品質スコアでソート（降順）
    sorted_progs = sorted(progressions, key=lambda x: x[1], reverse=True)
    
    selected = []
    for chords, score in sorted_progs:
        if len(selected) >= top_k:
            break
        
        # 既選択候補との類似度チェック
        is_similar = False
        for selected_chords, _ in selected:
            sim = calculate_chord_similarity(chords, selected_chords)
            if sim >= config.similarity_threshold:
                is_similar = True
                break
        
        # 類似していなければ選択
        if not is_similar:
            selected.append((chords, score))
    
    return selected


def apply_diversity_penalty(
    base_score: float,
    progression: List[str],
    reference_progressions: List[List[str]],
    config: DiversityConfig | None = None
) -> float:
    """
    多様性ペナルティを適用
    
    Args:
        base_score: 基本品質スコア
        progression: 評価対象のコード進行
        reference_progressions: 参照コード進行リスト
        config: 多様性計算の設定
    
    Returns:
        ペナルティ適用後のスコア
    """
    config = config or DiversityConfig()
    
    if not reference_progressions:
        return base_score
    
    # 参照進行との平均類似度を計算
    similarities = [
        calculate_chord_similarity(progression, ref)
        for ref in reference_progressions
    ]
    avg_similarity = sum(similarities) / len(similarities)
    
    # 類似度が高いほどペナルティ
    penalty = avg_similarity * config.penalty_weight
    
    # スコアから減算
    penalized_score = base_score * (1.0 - penalty)
    return float(max(0.0, min(1.0, penalized_score)))


def load_config_from_yaml(yaml_path: str | Path) -> DiversityConfig:
    """YAML から DiversityConfig を読み込み"""
    with open(yaml_path) as f:
        data = yaml.safe_load(f)
    
    # quality_gates.strings.diversity_penalty を探す
    strings_gates = data.get("quality_gates", {}).get("strings", {})
    div_penalty = strings_gates.get("diversity_penalty", {})
    
    if not div_penalty.get("enabled", False):
        return DiversityConfig()
    
    return DiversityConfig(
        ngram_size=div_penalty.get("ngram_size", 3),
        window_bars=div_penalty.get("window_bars", 4),
        similarity_threshold=div_penalty.get("similarity_threshold", 0.8),
        penalty_weight=div_penalty.get("penalty_weight", 0.15)
    )


# ============================================================================
# CLI
# ============================================================================

def cli_main():
    """CLI エントリポイント"""
    parser = argparse.ArgumentParser(
        description="Analyze chord progression diversity"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default="configs/structure_template.yaml",
        help="YAML config file"
    )
    parser.add_argument(
        "--progressions",
        nargs="+",
        help="Chord progressions (e.g., 'C G Am F' 'C Dm G C')"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose output"
    )
    
    args = parser.parse_args()
    
    # Config 読み込み
    config = load_config_from_yaml(args.config)
    
    if args.verbose:
        print(f"=== Diversity Configuration ===")
        print(f"N-gram size: {config.ngram_size}")
        print(f"Similarity threshold: {config.similarity_threshold}")
        print(f"Penalty weight: {config.penalty_weight}")
        print()
    
    # コード進行を解析
    if args.progressions:
        progs = [prog.split() for prog in args.progressions]
        
        print(f"=== Analyzing {len(progs)} Chord Progressions ===\n")
        
        for i, prog in enumerate(progs, 1):
            diversity = calculate_ngram_diversity(prog, n=config.ngram_size)
            print(f"{i}. {' - '.join(prog)}")
            print(f"   Diversity: {diversity:.3f}")
        
        print(f"\n=== Homogeneity Score ===")
        homogeneity = calculate_homogeneity_score(progs, config)
        print(f"Homogeneity: {homogeneity:.3f}")
        print(f"Overall Diversity: {1.0 - homogeneity:.3f}")
        
        if homogeneity >= config.similarity_threshold:
            print("\n⚠️  WARNING: High homogeneity detected!")
            print("   Consider using more diverse chord progressions.")
    else:
        print("No progressions provided. Use --progressions option.")


if __name__ == "__main__":
    cli_main()
