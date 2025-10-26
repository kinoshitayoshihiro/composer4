#!/usr/bin/env python3
"""
Dataset Splitter - Train/Val/Test分割

学習データを Train/Val/Test に分割（デフォルト 8:1:1）。
Source stratification（pop909/slakh均等分割）をサポート。

Usage:
    python scripts/split_dataset.py \\
      --input harmony_dataset/training_sequences.parquet \\
      --output-dir harmony_dataset/splits \\
      --train 0.8 --val 0.1 --test 0.1 \\
      --stratify-by source \\
      --random-seed 42

Features:
    - Source stratification（pop909/slakh/その他を均等分割）
    - 再現性保証（random seed固定）
    - 統計サマリー出力（各split・各source別の曲数・シーケンス数）
    - validation: split比率の合計が1.0になることを確認
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import numpy as np

# Logging設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Split training sequences into Train/Val/Test'
    )
    
    parser.add_argument(
        '--input',
        type=Path,
        required=True,
        help='Input parquet file (training_sequences.parquet)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=Path,
        required=True,
        help='Output directory for splits'
    )
    
    parser.add_argument(
        '--train',
        type=float,
        default=0.8,
        help='Train split ratio (default: 0.8)'
    )
    
    parser.add_argument(
        '--val',
        type=float,
        default=0.1,
        help='Validation split ratio (default: 0.1)'
    )
    
    parser.add_argument(
        '--test',
        type=float,
        default=0.1,
        help='Test split ratio (default: 0.1)'
    )
    
    parser.add_argument(
        '--stratify-by',
        type=str,
        default='none',
        choices=['source', 'quality', 'none'],
        help='Stratification strategy (default: none)'
    )
    
    parser.add_argument(
        '--random-seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    
    return parser.parse_args()


def validate_split_ratios(train: float, val: float, test: float):
    """Validate split ratios sum to 1.0"""
    total = train + val + test
    if not np.isclose(total, 1.0, atol=1e-6):
        raise ValueError(
            f"Split ratios must sum to 1.0, got {total:.6f} "
            f"(train={train}, val={val}, test={test})"
        )


def extract_source_from_song_id(song_id: str) -> str:
    """
    Extract source dataset from song_id
    
    Examples:
        - 'pop909_001' -> 'pop909'
        - 'slakh_Track00001' -> 'slakh'
        - 'other_abc123' -> 'other'
        - 'abc123def456' -> 'unknown'
    
    Args:
        song_id: Song ID
    
    Returns:
        Source name (pop909/slakh/other/unknown)
    """
    if song_id.startswith('pop909_'):
        return 'pop909'
    elif song_id.startswith('slakh_'):
        return 'slakh'
    elif '_' in song_id:
        # Prefix_XXX形式
        return song_id.split('_')[0]
    else:
        return 'unknown'


def split_by_source(
    df: pd.DataFrame,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    random_seed: int
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split dataset with source stratification
    
    Args:
        df: Input DataFrame
        train_ratio: Train ratio
        val_ratio: Val ratio
        test_ratio: Test ratio
        random_seed: Random seed
    
    Returns:
        (train_df, val_df, test_df)
    """
    # Extract source from song_id
    df['source'] = df['song_id'].apply(extract_source_from_song_id)
    
    # Group by source
    sources = df['source'].unique()
    logger.info(f"Found {len(sources)} sources: {sorted(sources)}")
    
    train_dfs = []
    val_dfs = []
    test_dfs = []
    
    np.random.seed(random_seed)
    
    for source in sources:
        source_df = df[df['source'] == source].copy()
        song_ids = source_df['song_id'].unique()
        
        logger.info(f"  {source}: {len(song_ids)} songs, {len(source_df)} sequences")
        
        # Shuffle song_ids
        np.random.shuffle(song_ids)
        
        # Split indices
        n_songs = len(song_ids)
        n_train = int(n_songs * train_ratio)
        n_val = int(n_songs * val_ratio)
        
        train_songs = song_ids[:n_train]
        val_songs = song_ids[n_train:n_train + n_val]
        test_songs = song_ids[n_train + n_val:]
        
        # Split DataFrames
        train_dfs.append(source_df[source_df['song_id'].isin(train_songs)])
        val_dfs.append(source_df[source_df['song_id'].isin(val_songs)])
        test_dfs.append(source_df[source_df['song_id'].isin(test_songs)])
        
        logger.info(f"    Train: {len(train_songs)} songs, Val: {len(val_songs)} songs, Test: {len(test_songs)} songs")
    
    # Concatenate
    train_df = pd.concat(train_dfs, ignore_index=True)
    val_df = pd.concat(val_dfs, ignore_index=True)
    test_df = pd.concat(test_dfs, ignore_index=True)
    
    # Drop temporary 'source' column
    train_df = train_df.drop(columns=['source'])
    val_df = val_df.drop(columns=['source'])
    test_df = test_df.drop(columns=['source'])
    
    return train_df, val_df, test_df


def split_no_stratification(
    df: pd.DataFrame,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    random_seed: int
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split dataset without stratification (random shuffle)
    
    Args:
        df: Input DataFrame
        train_ratio: Train ratio
        val_ratio: Val ratio
        test_ratio: Test ratio
        random_seed: Random seed
    
    Returns:
        (train_df, val_df, test_df)
    """
    song_ids = df['song_id'].unique()
    
    np.random.seed(random_seed)
    np.random.shuffle(song_ids)
    
    n_songs = len(song_ids)
    n_train = int(n_songs * train_ratio)
    n_val = int(n_songs * val_ratio)
    
    train_songs = song_ids[:n_train]
    val_songs = song_ids[n_train:n_train + n_val]
    test_songs = song_ids[n_train + n_val:]
    
    train_df = df[df['song_id'].isin(train_songs)]
    val_df = df[df['song_id'].isin(val_songs)]
    test_df = df[df['song_id'].isin(test_songs)]
    
    logger.info(f"Split: Train {len(train_songs)} songs, Val {len(val_songs)} songs, Test {len(test_songs)} songs")
    
    return train_df, val_df, test_df


def determine_quality_label(row: pd.Series) -> str:
    """
    Determine quality label from ratios
    
    Args:
        row: DataFrame row with gold_ratio, silver_ratio, bronze_ratio
    
    Returns:
        'gold', 'silver', or 'bronze'
    """
    if row['gold_ratio'] > 0.5:
        return 'gold'
    elif row['silver_ratio'] > 0.5:
        return 'silver'
    else:
        return 'bronze'


def split_by_quality(
    df: pd.DataFrame,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    random_seed: int
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split dataset with quality stratification (Gold/Silver/Bronze均等分割)
    
    Args:
        df: Input DataFrame
        train_ratio: Train ratio
        val_ratio: Val ratio
        test_ratio: Test ratio
        random_seed: Random seed
    
    Returns:
        (train_df, val_df, test_df)
    """
    # Determine quality label for each song (majority vote across sequences)
    song_quality = df.groupby('song_id').apply(
        lambda g: determine_quality_label(g.iloc[0])
    ).reset_index()
    song_quality.columns = ['song_id', 'quality']
    
    qualities = song_quality['quality'].unique()
    logger.info(f"Found {len(qualities)} quality levels: {sorted(qualities)}")
    
    train_dfs = []
    val_dfs = []
    test_dfs = []
    
    np.random.seed(random_seed)
    
    for quality in qualities:
        quality_songs = song_quality[song_quality['quality'] == quality]['song_id'].values
        quality_df = df[df['song_id'].isin(quality_songs)].copy()
        
        logger.info(f"  {quality}: {len(quality_songs)} songs, {len(quality_df)} sequences")
        
        # Shuffle song_ids
        np.random.shuffle(quality_songs)
        
        # Split indices
        n_songs = len(quality_songs)
        n_train = int(n_songs * train_ratio)
        n_val = int(n_songs * val_ratio)
        
        train_songs = quality_songs[:n_train]
        val_songs = quality_songs[n_train:n_train + n_val]
        test_songs = quality_songs[n_train + n_val:]
        
        # Split DataFrames
        train_dfs.append(quality_df[quality_df['song_id'].isin(train_songs)])
        val_dfs.append(quality_df[quality_df['song_id'].isin(val_songs)])
        test_dfs.append(quality_df[quality_df['song_id'].isin(test_songs)])
        
        logger.info(f"    Train: {len(train_songs)} songs, Val: {len(val_songs)} songs, Test: {len(test_songs)} songs")
    
    # Concatenate
    train_df = pd.concat(train_dfs, ignore_index=True)
    val_df = pd.concat(val_dfs, ignore_index=True)
    test_df = pd.concat(test_dfs, ignore_index=True)
    
    return train_df, val_df, test_df


def print_split_summary(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame
):
    """Print split summary statistics"""
    logger.info("")
    logger.info("=" * 60)
    logger.info("Split Summary")
    logger.info("=" * 60)
    
    for name, df in [('Train', train_df), ('Val', val_df), ('Test', test_df)]:
        n_songs = df['song_id'].nunique()
        n_sequences = len(df)
        
        # Source distribution
        df['source'] = df['song_id'].apply(extract_source_from_song_id)
        source_counts = df.groupby('source')['song_id'].nunique().to_dict()
        
        # Quality distribution
        gold_seq = len(df[df['gold_ratio'] > 0.5])
        silver_seq = len(df[(df['silver_ratio'] > 0.5) & (df['gold_ratio'] <= 0.5)])
        bronze_seq = len(df) - gold_seq - silver_seq
        
        logger.info(f"{name}:")
        logger.info(f"  Songs: {n_songs}")
        logger.info(f"  Sequences: {n_sequences}")
        logger.info(f"  Quality: Gold {gold_seq} ({gold_seq/n_sequences*100:.1f}%), "
                   f"Silver {silver_seq} ({silver_seq/n_sequences*100:.1f}%), "
                   f"Bronze {bronze_seq} ({bronze_seq/n_sequences*100:.1f}%)")
        
        if source_counts:
            source_str = ", ".join([f"{src}: {cnt}" for src, cnt in sorted(source_counts.items())])
            logger.info(f"  Sources: {source_str}")
        
        logger.info("")


def main():
    args = parse_args()
    
    # Validate split ratios
    validate_split_ratios(args.train, args.val, args.test)
    
    # Load input
    logger.info(f"Loading training sequences from {args.input}")
    df = pd.read_parquet(args.input)
    
    n_songs = df['song_id'].nunique()
    n_sequences = len(df)
    logger.info(f"Loaded {n_sequences} sequences from {n_songs} songs")
    
    # Split
    logger.info(f"Splitting with strategy: {args.stratify_by}")
    logger.info(f"Ratios: Train={args.train}, Val={args.val}, Test={args.test}")
    logger.info(f"Random seed: {args.random_seed}")
    logger.info("")
    
    if args.stratify_by == 'source':
        train_df, val_df, test_df = split_by_source(
            df, args.train, args.val, args.test, args.random_seed
        )
    elif args.stratify_by == 'quality':
        train_df, val_df, test_df = split_by_quality(
            df, args.train, args.val, args.test, args.random_seed
        )
    else:
        train_df, val_df, test_df = split_no_stratification(
            df, args.train, args.val, args.test, args.random_seed
        )
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save splits
    train_path = args.output_dir / 'train.parquet'
    val_path = args.output_dir / 'val.parquet'
    test_path = args.output_dir / 'test.parquet'
    
    logger.info("Saving splits...")
    train_df.to_parquet(train_path, index=False)
    logger.info(f"  ✓ Train: {train_path}")
    
    val_df.to_parquet(val_path, index=False)
    logger.info(f"  ✓ Val: {val_path}")
    
    test_df.to_parquet(test_path, index=False)
    logger.info(f"  ✓ Test: {test_path}")
    
    # Print summary
    print_split_summary(train_df, val_df, test_df)
    
    # Save metadata
    metadata = {
        'input': str(args.input),
        'output_dir': str(args.output_dir),
        'train_ratio': args.train,
        'val_ratio': args.val,
        'test_ratio': args.test,
        'stratify_by': args.stratify_by,
        'random_seed': args.random_seed,
        'total_songs': n_songs,
        'total_sequences': n_sequences,
        'splits': {
            'train': {
                'songs': train_df['song_id'].nunique(),
                'sequences': len(train_df)
            },
            'val': {
                'songs': val_df['song_id'].nunique(),
                'sequences': len(val_df)
            },
            'test': {
                'songs': test_df['song_id'].nunique(),
                'sequences': len(test_df)
            }
        }
    }
    
    import json
    metadata_path = args.output_dir / 'split_metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"✓ Saved metadata to {metadata_path}")
    logger.info("")
    logger.info("=" * 60)
    logger.info("Split complete!")
    logger.info("=" * 60)


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        sys.exit(1)
