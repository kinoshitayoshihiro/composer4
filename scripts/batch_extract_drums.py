"""
scripts/batch_extract_drums.py

SLAKH/LAMDAから大量のドラムパターンをバッチ抽出
Todo #4: ドラムパターンバンク充実
"""

import sys
import pickle
import argparse
from pathlib import Path
from typing import List, Dict, Any
import logging
from tqdm import tqdm
import random

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scripts.extract_drum_patterns import (
    extract_patterns_from_midi,
    estimate_tempo_from_score,
    classify_bpm_range,
    iter_drum_midi_events_m21
)
from generator.drums_generator_stage2 import DrumPattern

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)


def find_midi_files(root_dir: Path, max_files: int = None) -> List[Path]:
    """
    指定ディレクトリから再帰的にMIDIファイルを検索
    
    Args:
        root_dir: 検索ルートディレクトリ
        max_files: 最大ファイル数（Noneで全件）
    
    Returns:
        MIDIファイルパスのリスト
    """
    logger.info(f"Searching MIDI files in: {root_dir}")
    
    midi_files = list(root_dir.rglob("*.mid")) + list(root_dir.rglob("*.midi"))
    
    if max_files:
        random.shuffle(midi_files)
        midi_files = midi_files[:max_files]
    
    logger.info(f"Found {len(midi_files)} MIDI files")
    return midi_files


def extract_from_file(
    midi_path: Path,
    min_bars: int = 4,
    max_bars: int = 8,
    min_quality: float = 0.6
) -> List[DrumPattern]:
    """
    1つのMIDIファイルからドラムパターンを抽出
    
    Args:
        midi_path: MIDIファイルパス
        min_bars: 最小小節数
        max_bars: 最大小節数
        min_quality: 最小品質スコア
    
    Returns:
        抽出されたパターンのリスト (既に品質フィルタ済み)
    """
    try:
        # パターン抽出 (extract_patterns_from_midiが品質フィルタリング済み)
        patterns = extract_patterns_from_midi(
            midi_path,
            min_bars=min_bars,
            max_bars=max_bars,
            min_quality=min_quality
        )
        
        return patterns
    
    except Exception as e:
        logger.warning(f"Failed to extract from {midi_path}: {e}")
        return []


def stratify_patterns(
    patterns: List[DrumPattern],
    target_per_bin: int = 500
) -> Dict[str, List[DrumPattern]]:
    """
    パターンをBPM範囲で層化
    
    Args:
        patterns: パターンリスト
        target_per_bin: 各BPM範囲の目標数
    
    Returns:
        {bpm_range: [patterns]} の辞書
    """
    logger.info("Stratifying patterns by BPM range...")
    
    # BPM範囲ごとに分類
    bins = {}
    for pattern in patterns:
        bpm_range = classify_bpm_range(pattern.tempo)
        if bpm_range not in bins:
            bins[bpm_range] = []
        bins[bpm_range].append(pattern)
    
    # 各binから目標数をサンプリング
    stratified = {}
    for bpm_range, bin_patterns in bins.items():
        # テンポでソート（ばらつき確保）
        sorted_patterns = sorted(
            bin_patterns,
            key=lambda p: p.tempo
        )
        
        # 目標数まで取得（不足していれば全部）
        sampled = sorted_patterns[:target_per_bin]
        stratified[bpm_range] = sampled
        
        logger.info(
            f"  {bpm_range:12s}: {len(bin_patterns):4d} → {len(sampled):4d} patterns "
            f"(tempo: {sampled[0].tempo:.1f}~{sampled[-1].tempo:.1f} BPM)"
        )
    
    return stratified


def save_patterns(
    patterns: Dict[str, List[DrumPattern]],
    output_path: Path
):
    """
    パターンをpickleファイルに保存
    
    Args:
        patterns: 層化されたパターン辞書
        output_path: 出力ファイルパス
    """
    logger.info(f"Saving patterns to: {output_path}")
    
    # 統計情報計算
    total_count = sum(len(p) for p in patterns.values())
    tempos = [
        pattern.tempo
        for bin_patterns in patterns.values()
        for pattern in bin_patterns
    ]
    avg_tempo = sum(tempos) / len(tempos) if tempos else 0.0
    
    # メタデータ追加
    data = {
        "patterns": patterns,
        "metadata": {
            "total_patterns": total_count,
            "bins": {k: len(v) for k, v in patterns.items()},
            "avg_tempo": avg_tempo,
            "min_tempo": min(tempos) if tempos else 0.0,
            "max_tempo": max(tempos) if tempos else 0.0,
        }
    }
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    logger.info(f"✓ Saved {total_count} patterns (avg tempo: {avg_tempo:.1f} BPM)")


def main():
    parser = argparse.ArgumentParser(
        description="Batch extract drum patterns from SLAKH/LAMDA datasets"
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input directory containing MIDI files"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/patterns/stage2_drums.pkl"),
        help="Output pickle file"
    )
    parser.add_argument(
        "--max-files",
        type=int,
        help="Maximum number of MIDI files to process"
    )
    parser.add_argument(
        "--min-bars",
        type=int,
        default=4,
        help="Minimum number of bars per pattern"
    )
    parser.add_argument(
        "--max-bars",
        type=int,
        default=8,
        help="Maximum number of bars per pattern"
    )
    parser.add_argument(
        "--min-quality",
        type=float,
        default=0.6,
        help="Minimum quality score (0.0~1.0)"
    )
    parser.add_argument(
        "--target-per-bin",
        type=int,
        default=500,
        help="Target number of patterns per BPM bin"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    
    args = parser.parse_args()
    
    # Seed設定
    random.seed(args.seed)
    
    logger.info("=" * 60)
    logger.info("Drum Pattern Batch Extraction - Todo #4")
    logger.info("=" * 60)
    logger.info(f"Input:           {args.input}")
    logger.info(f"Output:          {args.output}")
    logger.info(f"Max files:       {args.max_files or 'All'}")
    logger.info(f"Min bars:        {args.min_bars}")
    logger.info(f"Max bars:        {args.max_bars}")
    logger.info(f"Min quality:     {args.min_quality}")
    logger.info(f"Target/bin:      {args.target_per_bin}")
    logger.info(f"Seed:            {args.seed}")
    logger.info("=" * 60)
    
    # 1. MIDIファイル検索
    midi_files = find_midi_files(args.input, args.max_files)
    
    if not midi_files:
        logger.error("No MIDI files found!")
        return 1
    
    # 2. パターン抽出
    logger.info("\nExtracting patterns...")
    all_patterns = []
    
    for midi_path in tqdm(midi_files, desc="Processing", unit="file"):
        patterns = extract_from_file(
            midi_path,
            min_bars=args.min_bars,
            max_bars=args.max_bars,
            min_quality=args.min_quality
        )
        all_patterns.extend(patterns)
    
    logger.info(f"\n✓ Extracted {len(all_patterns)} valid patterns from {len(midi_files)} files")
    
    if not all_patterns:
        logger.error("No valid patterns extracted!")
        return 1
    
    # 3. BPM層化
    stratified_patterns = stratify_patterns(
        all_patterns,
        target_per_bin=args.target_per_bin
    )
    
    # 4. 保存
    save_patterns(stratified_patterns, args.output)
    
    logger.info("=" * 60)
    logger.info("✅ Batch extraction completed successfully!")
    logger.info("=" * 60)
    
    return 0


if __name__ == "__main__":
    exit(main())
