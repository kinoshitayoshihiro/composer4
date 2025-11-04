#!/usr/bin/env python3
"""
パターンPC集合事前計算スクリプト

pickleファイル内の全パターンにvoicing→PC集合を事前計算して保存。
推論時のmod12計算を削減し、Chord Fit計算を高速化。

使用方法:
    python scripts/precompute_pattern_pc_sets.py \
        --input data/patterns/stage2_guitar_v3_fixed.pickle \
        --output data/patterns/stage2_guitar_v3_fixed_pc.pickle
"""

import sys
import os
import pickle
import logging
from pathlib import Path
from typing import Set, Any, Dict, List

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def compute_pc_set(voicing: List[int]) -> Set[int]:
    """
    voicingからピッチクラス集合を計算
    
    Args:
        voicing: MIDI note番号のリスト（例: [60, 64, 67]）
    
    Returns:
        PC集合（0-11の整数set、例: {0, 4, 7}）
    """
    return {(pitch % 12) for pitch in voicing if pitch > 0}


def precompute_pc_sets(input_path: str, output_path: str):
    """
    pickleファイル内の全パターンにPC集合を追加
    
    Args:
        input_path: 入力pickleファイルパス
        output_path: 出力pickleファイルパス
    """
    logger.info(f"Loading pickle: {input_path}")
    
    # Pickleロード
    try:
        with open(input_path, 'rb') as f:
            data = pickle.load(f)
    except Exception as e:
        logger.error(f"Failed to load pickle: {e}")
        return
    
    # データ構造確認
    if isinstance(data, dict) and 'patterns' in data:
        patterns = data['patterns']
        is_dict_of_patterns = isinstance(patterns, dict)
        if is_dict_of_patterns:
            pattern_items = list(patterns.items())
            logger.info(f"Found {len(pattern_items)} patterns in dict['patterns'] (as dict)")
        else:
            pattern_items = [(i, p) for i, p in enumerate(patterns)]
            logger.info(f"Found {len(pattern_items)} patterns in dict['patterns'] (as list)")
    elif isinstance(data, list):
        patterns = data
        pattern_items = [(i, p) for i, p in enumerate(patterns)]
        logger.info(f"Found {len(pattern_items)} patterns in list")
    else:
        logger.error(f"Unknown data structure: {type(data)}")
        return
    
    # PC集合事前計算
    modified_count = 0
    error_count = 0
    
    for i, (key, pattern) in enumerate(pattern_items):
        try:
            if isinstance(pattern, dict):
                # dict構造（v3形式）
                voicing = pattern.get('voicing', [])
                
                if voicing:
                    pc_set = compute_pc_set(voicing)
                    pattern['pc_set'] = list(pc_set)  # setはpickle可能だが、listの方が安全
                    modified_count += 1
                    
                    # 進捗ログ（1000件ごと）
                    if (i + 1) % 1000 == 0:
                        logger.info(f"Processed {i + 1}/{len(pattern_items)} patterns...")
            else:
                # ExtractedPattern構造（v1形式）
                if hasattr(pattern, 'voicing'):
                    voicing = pattern.voicing or []
                    pc_set = compute_pc_set(voicing)
                    pattern.pc_set = list(pc_set)
                    modified_count += 1
                    
                    if (i + 1) % 1000 == 0:
                        logger.info(f"Processed {i + 1}/{len(pattern_items)} patterns...")
        
        except Exception as e:
            logger.warning(f"Error processing pattern {i}: {e}")
            error_count += 1
    
    logger.info(f"PC sets computed: {modified_count} patterns")
    logger.info(f"Errors: {error_count} patterns")
    
    # 保存
    logger.info(f"Saving to: {output_path}")
    
    try:
        # 出力ディレクトリ作成
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        logger.info(f"✓ Successfully saved {output_path}")
        
        # ファイルサイズ比較
        input_size = Path(input_path).stat().st_size / (1024 * 1024)  # MB
        output_size = Path(output_path).stat().st_size / (1024 * 1024)  # MB
        logger.info(f"Input size: {input_size:.2f} MB")
        logger.info(f"Output size: {output_size:.2f} MB")
        logger.info(f"Size increase: {((output_size - input_size) / input_size * 100):.2f}%")
    
    except Exception as e:
        logger.error(f"Failed to save pickle: {e}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Precompute PC sets for pattern pickle")
    parser.add_argument(
        '--input',
        required=True,
        help='Input pickle file path'
    )
    parser.add_argument(
        '--output',
        required=True,
        help='Output pickle file path (with precomputed PC sets)'
    )
    
    args = parser.parse_args()
    
    logger.info("========================================")
    logger.info("Pattern PC Set Precomputation Started")
    logger.info("========================================")
    
    precompute_pc_sets(args.input, args.output)
    
    logger.info("========================================")
    logger.info("Precomputation Completed")
    logger.info("========================================")


if __name__ == '__main__':
    main()
