#!/usr/bin/env python3
"""
Normalize Drum Phases - Phase 25.1 Task 3

コサイン類似度による位相正規化で同形パターンを集約。

Features:
- 円環シフト（0..N-1）で最適位相探索
- コサイン類似度計算
- Downbeat一致の正規化Pattern ID生成
- 同形パターン集約・統計

Pipeline:
1. drum_patterns.parquetを読み込み
2. 各パターンをコサイン類似度で位相最適化
3. 正規化済みPattern ID生成（SHA1先頭12桁）
4. 同形パターン集約
5. drum_patterns_normalized.parquet出力

Usage:
    python scripts/normalize_drum_phases.py \
        --input data/drum_patterns.parquet \
        --output data/drum_patterns_normalized.parquet \
        --reference-role kick
"""

import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """コサイン類似度計算
    
    Args:
        a, b: ベクトル
    
    Returns:
        類似度（-1.0 ~ 1.0）
    """
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    
    if norm_a == 0 or norm_b == 0:
        return 0.0
    
    return float(np.dot(a, b) / (norm_a * norm_b))


def find_best_phase(
    pattern_vec: np.ndarray,
    reference_vec: np.ndarray
) -> Tuple[int, float]:
    """最適位相探索（円環シフト）
    
    Args:
        pattern_vec: 位相合わせ対象ベクトル
        reference_vec: 参照ベクトル（通常はキック）
    
    Returns:
        (最適シフト量, 最大類似度)
    """
    N = len(pattern_vec)
    
    best_shift = 0
    best_sim = -1.0
    
    for shift in range(N):
        # 円環シフト
        shifted = np.roll(pattern_vec, shift)
        
        # コサイン類似度
        sim = cosine_similarity(shifted, reference_vec)
        
        if sim > best_sim:
            best_sim = sim
            best_shift = shift
    
    return best_shift, best_sim


def normalize_pattern_phase(
    kick_vec: List[int],
    snare_vec: List[int],
    hat_vec: List[int],
    reference_role: str = 'kick'
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int, float]:
    """パターン位相正規化
    
    Args:
        kick_vec, snare_vec, hat_vec: アクセントベクトル
        reference_role: 参照役割（kick/snare/hat）
    
    Returns:
        (正規化kick, 正規化snare, 正規化hat, シフト量, 類似度)
    """
    # numpy配列化
    kick_arr = np.array(kick_vec, dtype=float)
    snare_arr = np.array(snare_vec, dtype=float)
    hat_arr = np.array(hat_vec, dtype=float)
    
    # 参照ベクトル選択
    if reference_role == 'kick':
        reference = kick_arr
    elif reference_role == 'snare':
        reference = snare_arr
    elif reference_role == 'hat':
        reference = hat_arr
    else:
        reference = kick_arr  # デフォルト
    
    # ダウンビート基準ベクトル（1拍目に強アクセント）
    # 例: 16スロット → [1,0,0,0, 0,0,0,0, 1,0,0,0, 0,0,0,0]
    N = len(kick_vec)
    downbeat_pattern = np.zeros(N, dtype=float)
    
    # 4分音符ごとにアクセント（4/4拍子想定）
    if N == 16:
        downbeat_pattern[[0, 4, 8, 12]] = 1.0
    elif N == 24:
        # 6/8拍子: 6スロットごと
        downbeat_pattern[[0, 6, 12, 18]] = 1.0
    else:
        # その他: 均等分割
        step = N // 4
        for i in range(0, N, step):
            downbeat_pattern[i] = 1.0
    
    # 最適位相探索
    best_shift, best_sim = find_best_phase(reference, downbeat_pattern)
    
    # 全ベクトルをシフト
    kick_normalized = np.roll(kick_arr, best_shift)
    snare_normalized = np.roll(snare_arr, best_shift)
    hat_normalized = np.roll(hat_arr, best_shift)
    
    return kick_normalized, snare_normalized, hat_normalized, best_shift, best_sim


def generate_normalized_pattern_id(
    kick_vec: np.ndarray,
    snare_vec: np.ndarray,
    hat_vec: np.ndarray,
    slots: int
) -> str:
    """正規化済みPattern ID生成
    
    Args:
        kick_vec, snare_vec, hat_vec: 正規化済みアクセントベクトル
        slots: スロット数
    
    Returns:
        Pattern ID (e.g., "norm_a3f7e2b9c1d4_s16")
    """
    # ベクトル結合
    combined = np.concatenate([kick_vec, snare_vec, hat_vec])
    
    # int化（0/1）
    combined_int = combined.astype(int)
    
    # bytes化
    data_bytes = combined_int.tobytes()
    
    # SHA1ハッシュ
    hash_obj = hashlib.sha1(data_bytes)
    hash_hex = hash_obj.hexdigest()
    
    # 正規化済み識別子 + 先頭12桁 + スロット情報
    pattern_id = f"norm_{hash_hex[:12]}_s{slots}"
    
    return pattern_id


def normalize_dataset(
    df: pd.DataFrame,
    reference_role: str = 'kick'
) -> pd.DataFrame:
    """データセット一括位相正規化
    
    Args:
        df: drum_patterns.parquet DataFrame
        reference_role: 参照役割
    
    Returns:
        正規化済みDataFrame
    """
    logger.info(f"Normalizing {len(df)} patterns (reference: {reference_role})")
    
    normalized_data = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Normalizing phases"):
        # JSONからリスト復元
        kick_vec = json.loads(row['kick_vec']) if isinstance(row['kick_vec'], str) else row['kick_vec']
        snare_vec = json.loads(row['snare_vec']) if isinstance(row['snare_vec'], str) else row['snare_vec']
        hat_vec = json.loads(row['hat_vec']) if isinstance(row['hat_vec'], str) else row['hat_vec']
        
        # 位相正規化
        kick_norm, snare_norm, hat_norm, shift, similarity = normalize_pattern_phase(
            kick_vec, snare_vec, hat_vec, reference_role
        )
        
        # 正規化Pattern ID生成
        norm_pattern_id = generate_normalized_pattern_id(
            kick_norm, snare_norm, hat_norm, row['slots']
        )
        
        # 元データコピー
        norm_row = row.to_dict()
        
        # 正規化データ追加
        norm_row['kick_vec_normalized'] = kick_norm.astype(int).tolist()
        norm_row['snare_vec_normalized'] = snare_norm.astype(int).tolist()
        norm_row['hat_vec_normalized'] = hat_norm.astype(int).tolist()
        norm_row['phase_shift'] = shift
        norm_row['phase_similarity'] = similarity
        norm_row['pattern_id_normalized'] = norm_pattern_id
        
        normalized_data.append(norm_row)
    
    # DataFrame化
    df_normalized = pd.DataFrame(normalized_data)
    
    # 統計出力
    logger.info(f"✅ Normalized {len(df_normalized)} patterns")
    logger.info(f"   Original unique IDs: {df['pattern_id'].nunique()}")
    logger.info(f"   Normalized unique IDs: {df_normalized['pattern_id_normalized'].nunique()}")
    logger.info(f"   Reduction: {(1 - df_normalized['pattern_id_normalized'].nunique() / df['pattern_id'].nunique()) * 100:.1f}%")
    
    return df_normalized


def analyze_pattern_clusters(df_normalized: pd.DataFrame) -> pd.DataFrame:
    """正規化後パターンクラスタ分析
    
    Args:
        df_normalized: 正規化済みDataFrame
    
    Returns:
        クラスタ統計DataFrame
    """
    clusters = []
    
    for pattern_id, group in df_normalized.groupby('pattern_id_normalized'):
        # クラスタ統計
        cluster_stat = {
            'pattern_id_normalized': pattern_id,
            'count': len(group),
            'unique_songs': group['song_id'].nunique(),
            'avg_tempo': group['tempo_bpm'].mean(),
            'tempo_std': group['tempo_bpm'].std(),
            'avg_density_k': group['density_k'].mean(),
            'avg_density_s': group['density_s'].mean(),
            'avg_density_h': group['density_h'].mean(),
            'avg_syncopation': group['syncopation'].mean(),
            'slots': group['slots'].iloc[0],
            # 代表例（最初の出現）
            'example_song_id': group['song_id'].iloc[0],
            'example_bar_index': group['bar_index'].iloc[0]
        }
        
        clusters.append(cluster_stat)
    
    df_clusters = pd.DataFrame(clusters)
    df_clusters = df_clusters.sort_values('count', ascending=False)
    
    logger.info(f"\n{'='*70}")
    logger.info(f"Pattern Cluster Analysis")
    logger.info(f"{'='*70}")
    logger.info(f"Total clusters: {len(df_clusters)}")
    logger.info(f"Top 10 most common patterns:")
    
    for idx, row in df_clusters.head(10).iterrows():
        logger.info(f"  {row['pattern_id_normalized']}: "
                   f"{row['count']} occurrences, "
                   f"{row['unique_songs']} songs, "
                   f"tempo={row['avg_tempo']:.1f}")
    
    logger.info(f"{'='*70}\n")
    
    return df_clusters


def main():
    """メインエントリーポイント"""
    parser = argparse.ArgumentParser(
        description="Normalize Drum Phases - 位相正規化・Pattern集約"
    )
    parser.add_argument(
        '--input',
        type=Path,
        required=True,
        help='Input parquet (drum_patterns.parquet)'
    )
    parser.add_argument(
        '--output',
        type=Path,
        required=True,
        help='Output parquet (drum_patterns_normalized.parquet)'
    )
    parser.add_argument(
        '--reference-role',
        type=str,
        default='kick',
        choices=['kick', 'snare', 'hat'],
        help='Reference role for phase alignment (default: kick)'
    )
    parser.add_argument(
        '--cluster-stats',
        type=Path,
        default=None,
        help='Output cluster statistics CSV (optional)'
    )
    
    args = parser.parse_args()
    
    # データ読み込み
    logger.info(f"Loading {args.input}")
    df = pd.read_parquet(args.input)
    logger.info(f"Loaded {len(df)} patterns from {df['song_id'].nunique()} songs")
    
    # 位相正規化
    df_normalized = normalize_dataset(df, reference_role=args.reference_role)
    
    # JSON列をstr化（parquet保存用）
    for col in ['kick_vec_normalized', 'snare_vec_normalized', 'hat_vec_normalized']:
        if col in df_normalized.columns:
            df_normalized[col] = df_normalized[col].apply(json.dumps)
    
    # 出力
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df_normalized.to_parquet(args.output, index=False)
    logger.info(f"✅ Saved to {args.output}")
    
    # クラスタ分析
    df_clusters = analyze_pattern_clusters(df_normalized)
    
    if args.cluster_stats:
        df_clusters.to_csv(args.cluster_stats, index=False)
        logger.info(f"📊 Cluster stats saved to {args.cluster_stats}")


if __name__ == '__main__':
    main()
