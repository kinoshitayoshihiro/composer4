#!/usr/bin/env python3
"""
Test Instrument Metrics
楽器別メトリクスのテストスクリプト

少量データでメトリクス計算をテストし、スコア分布を確認する
"""

import sys
import os
from pathlib import Path
import argparse
import yaml
import json
import numpy as np
from collections import defaultdict
from typing import List, Dict
import logging

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scripts.stage2_instrument_metrics import calculate_instrument_metrics

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_midi_to_notes(midi_path: Path) -> List[Dict]:
    """
    MIDIファイルを読み込んでノート情報を抽出
    
    Args:
        midi_path: MIDI file path
    
    Returns:
        List of note dicts with 'pitch', 'start', 'velocity', 'duration'
    """
    try:
        import pretty_midi
    except ImportError:
        logger.error("pretty_midi not installed. Run: pip install pretty_midi")
        return []
    
    try:
        pm = pretty_midi.PrettyMIDI(str(midi_path))
        notes = []
        
        for instrument in pm.instruments:
            if instrument.is_drum:
                continue  # Skip drum tracks for guitar/bass/strings
            
            for note in instrument.notes:
                notes.append({
                    'pitch': note.pitch,
                    'start': note.start,
                    'velocity': note.velocity,
                    'duration': note.end - note.start
                })
        
        return notes
    
    except Exception as e:
        logger.warning(f"Failed to load {midi_path.name}: {e}")
        return []


def test_instrument_metrics(
    instrument: str,
    input_dir: Path,
    config_path: Path,
    max_files: int = 100,
    output_json: Path = None
):
    """
    楽器別メトリクスをテスト
    
    Args:
        instrument: 'guitar', 'bass', or 'strings'
        input_dir: Input directory with clean MIDI files
        config_path: YAML config file path
        max_files: Maximum files to test
        output_json: Output JSON path for results
    """
    logger.info(f"╔══════════════════════════════════════════════════════════════╗")
    logger.info(f"║  Testing {instrument.upper()} Metrics")
    logger.info(f"╚══════════════════════════════════════════════════════════════╝")
    logger.info(f"Input:  {input_dir}")
    logger.info(f"Config: {config_path}")
    logger.info(f"Limit:  {max_files} files")
    logger.info("")
    
    # Load config
    if not config_path.exists():
        logger.error(f"Config not found: {config_path}")
        return
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Extract score config
    score_config = config.get('score', {})
    
    # Get MIDI files
    midi_files = list(input_dir.glob("*.mid")) + list(input_dir.glob("*.midi"))
    
    if not midi_files:
        logger.error(f"No MIDI files found in {input_dir}")
        return
    
    # Limit to max_files
    midi_files = midi_files[:max_files]
    logger.info(f"Testing {len(midi_files)} files...")
    logger.info("")
    
    # Calculate metrics
    results = []
    score_distributions = defaultdict(list)
    
    for i, midi_path in enumerate(midi_files, 1):
        if i % 10 == 0:
            logger.info(f"  Progress: {i}/{len(midi_files)}")
        
        # Load notes
        notes = load_midi_to_notes(midi_path)
        
        if not notes:
            logger.debug(f"  Skipping {midi_path.name} (no notes)")
            continue
        
        # Calculate metrics
        try:
            scores = calculate_instrument_metrics(instrument, notes, score_config)
            
            if not scores:
                logger.debug(f"  Skipping {midi_path.name} (no scores)")
                continue
            
            # Store results
            result = {
                'file': midi_path.name,
                'num_notes': len(notes),
                'scores': scores
            }
            results.append(result)
            
            # Collect distributions
            for metric, score in scores.items():
                score_distributions[metric].append(score)
        
        except Exception as e:
            logger.warning(f"  Error processing {midi_path.name}: {e}")
            continue
    
    logger.info("")
    logger.info(f"✅ Processed {len(results)} files successfully")
    logger.info("")
    
    # Print statistics
    logger.info("═" * 70)
    logger.info("SCORE DISTRIBUTIONS")
    logger.info("═" * 70)
    
    for metric, scores in sorted(score_distributions.items()):
        if not scores:
            continue
        
        scores_array = np.array(scores)
        
        logger.info(f"\n{metric}:")
        logger.info(f"  Count:  {len(scores)}")
        logger.info(f"  Mean:   {np.mean(scores_array):.4f}")
        logger.info(f"  Std:    {np.std(scores_array):.4f}")
        logger.info(f"  Min:    {np.min(scores_array):.4f}")
        logger.info(f"  25%:    {np.percentile(scores_array, 25):.4f}")
        logger.info(f"  Median: {np.median(scores_array):.4f}")
        logger.info(f"  75%:    {np.percentile(scores_array, 75):.4f}")
        logger.info(f"  Max:    {np.max(scores_array):.4f}")
    
    logger.info("")
    logger.info("═" * 70)
    
    # Calculate weighted total score
    axes = score_config.get('axes', {})
    total_scores = []
    
    for result in results:
        scores = result['scores']
        weighted_sum = 0
        weight_sum = 0
        
        for metric, score in scores.items():
            weight = axes.get(metric, 1.0)
            weighted_sum += score * weight
            weight_sum += weight
        
        if weight_sum > 0:
            total_score = weighted_sum / weight_sum
            total_scores.append(total_score)
            result['total_score'] = total_score
    
    if total_scores:
        logger.info("WEIGHTED TOTAL SCORE:")
        logger.info(f"  Mean:   {np.mean(total_scores):.4f}")
        logger.info(f"  Median: {np.median(total_scores):.4f}")
        logger.info(f"  Min:    {np.min(total_scores):.4f}")
        logger.info(f"  Max:    {np.max(total_scores):.4f}")
        logger.info("")
    
    # Threshold analysis
    pipeline_threshold = config.get('pipeline', {}).get('threshold', 40.0)
    
    if total_scores:
        total_scores_array = np.array(total_scores) * 100  # Convert to 0-100 scale
        passed = np.sum(total_scores_array >= pipeline_threshold)
        pass_rate = (passed / len(total_scores_array)) * 100
        
        logger.info(f"THRESHOLD ANALYSIS (threshold={pipeline_threshold}):")
        logger.info(f"  Total:       {len(total_scores_array)}")
        logger.info(f"  Passed:      {passed} ({pass_rate:.1f}%)")
        logger.info(f"  Failed:      {len(total_scores_array) - passed} ({100-pass_rate:.1f}%)")
        logger.info("")
    
    # Save results
    if output_json:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        
        summary = {
            'instrument': instrument,
            'input_dir': str(input_dir),
            'config': str(config_path),
            'files_tested': len(results),
            'files_total': len(midi_files),
            'score_distributions': {
                metric: {
                    'mean': float(np.mean(scores)),
                    'std': float(np.std(scores)),
                    'min': float(np.min(scores)),
                    'max': float(np.max(scores)),
                    'median': float(np.median(scores)),
                    'percentile_25': float(np.percentile(scores, 25)),
                    'percentile_75': float(np.percentile(scores, 75)),
                }
                for metric, scores in score_distributions.items()
            },
            'total_score': {
                'mean': float(np.mean(total_scores)) if total_scores else 0,
                'median': float(np.median(total_scores)) if total_scores else 0,
                'min': float(np.min(total_scores)) if total_scores else 0,
                'max': float(np.max(total_scores)) if total_scores else 0,
            } if total_scores else {},
            'threshold_analysis': {
                'threshold': pipeline_threshold,
                'passed': int(passed) if total_scores else 0,
                'total': len(total_scores),
                'pass_rate': float(pass_rate) if total_scores else 0,
            } if total_scores else {},
            'results': results
        }
        
        with open(output_json, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"📊 Results saved to: {output_json}")
        logger.info("")
    
    # Top 5 and Bottom 5
    if total_scores and len(results) >= 10:
        sorted_results = sorted(results, key=lambda r: r.get('total_score', 0), reverse=True)
        
        logger.info("🏆 TOP 5 FILES:")
        for i, result in enumerate(sorted_results[:5], 1):
            logger.info(f"  {i}. {result['file']}: {result.get('total_score', 0)*100:.2f}")
        
        logger.info("")
        logger.info("⚠️  BOTTOM 5 FILES:")
        for i, result in enumerate(sorted_results[-5:], 1):
            logger.info(f"  {i}. {result['file']}: {result.get('total_score', 0)*100:.2f}")
        logger.info("")


def main():
    parser = argparse.ArgumentParser(description='Test instrument-specific metrics')
    parser.add_argument('--instrument', required=True, choices=['guitar', 'bass', 'strings'],
                        help='Instrument to test')
    parser.add_argument('--input-dir', required=True, type=Path,
                        help='Input directory with clean MIDI files')
    parser.add_argument('--config', required=True, type=Path,
                        help='YAML config file path')
    parser.add_argument('--max-files', type=int, default=100,
                        help='Maximum files to test (default: 100)')
    parser.add_argument('--output-json', type=Path,
                        help='Output JSON path for results')
    
    args = parser.parse_args()
    
    test_instrument_metrics(
        instrument=args.instrument,
        input_dir=args.input_dir,
        config_path=args.config,
        max_files=args.max_files,
        output_json=args.output_json
    )


if __name__ == "__main__":
    main()
