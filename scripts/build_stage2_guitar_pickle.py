#!/usr/bin/env python3
"""
Build stage2_guitar.pickle for AI-powered pattern recommendation.

互換pickle生成：
- 入力：harmony_dataset/training_sequences.parquet（Gold/Silver 152,237シーケンス）
- 出力：data/patterns/stage2_guitar.pickle
  - selector: ルールベース（section/chord/tempo→pattern_id）
  - patterns: パターン辞書（pattern_id→voicing/rhythm）
  
将来的に selector を学習済みモデル（XGB/RandomForest）に差し替え可能。
"""
import argparse
import hashlib
import json
import logging
import pickle
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def load_training_data(parquet_path: Path) -> pd.DataFrame:
    """Load training sequences"""
    logger.info(f"Loading training data from {parquet_path}")
    df = pd.read_parquet(parquet_path)
    logger.info(f"Loaded {len(df)} sequences from {df['song_id'].nunique()} songs")
    return df


def extract_patterns(df: pd.DataFrame) -> Dict[str, Dict]:
    """
    Extract guitar patterns from training data.
    
    Pattern ID = hash(chord_root + chord_quality + section + tempo_bin)
    Pattern content = {
        'voicing': simplified voicing rules,
        'rhythm': rhythm templates,
        'metadata': usage stats
    }
    """
    logger.info("Extracting guitar patterns...")
    
    # Expand chord_sequence into individual rows
    expanded_rows = []
    for _, row in df.iterrows():
        for chord in row['chord_sequence']:
            expanded_rows.append({
                'song_id': row['song_id'],
                'section': row['section'],
                'tempo': row['tempo'],
                'time_sig': row['time_sig'],
                'chord_root': chord['root'],
                'chord_quality': chord['quality'],
                'confidence': chord['confidence'],
                'label_strength': chord['label_strength']
            })
    
    expanded_df = pd.DataFrame(expanded_rows)
    
    # Tempo binning
    expanded_df['tempo_bin'] = pd.cut(
        expanded_df['tempo'], 
        bins=[0, 90, 120, 150, 200],
        labels=['slow', 'mid', 'fast', 'very_fast']
    )
    
    # Group by pattern key
    pattern_groups = expanded_df.groupby([
        'section', 'chord_root', 'chord_quality', 'tempo_bin'
    ])
    
    patterns = {}
    for (section, root, quality, tempo_bin), group in pattern_groups:
        # Create pattern ID
        key = f"{section}_{root}_{quality}_{tempo_bin}"
        pattern_id = hashlib.md5(key.encode()).hexdigest()[:12]
        
        # Extract pattern content
        avg_confidence = group['confidence'].mean()
        usage_count = len(group)
        
        # Simplified voicing (基本形)
        if quality in ('maj', 'M', ''):
            voicing = [0, 4, 7]  # Major triad
        elif quality in ('min', 'm'):
            voicing = [0, 3, 7]  # Minor triad
        elif quality in ('7', 'dom7'):
            voicing = [0, 4, 7, 10]  # Dominant 7th
        elif quality in ('maj7', 'M7'):
            voicing = [0, 4, 7, 11]  # Major 7th
        elif quality in ('min7', 'm7'):
            voicing = [0, 3, 7, 10]  # Minor 7th
        elif quality in ('dim', 'dim7'):
            voicing = [0, 3, 6]  # Diminished
        elif quality in ('aug'):
            voicing = [0, 4, 8]  # Augmented
        else:
            voicing = [0, 4, 7]  # Fallback to major
        
        # Rhythm template (section-dependent)
        if section == 'Verse':
            rhythm = 'sparse_quarter'  # 控えめ
        elif section == 'Chorus':
            rhythm = 'full_eighth'  # 力強く
        elif section == 'Bridge':
            rhythm = 'arpeggio'  # アルペジオ
        elif section == 'Intro':
            rhythm = 'pickup'  # ピックアップ
        else:
            rhythm = 'standard_quarter'
        
        patterns[pattern_id] = {
            'key': key,
            'voicing': voicing,
            'rhythm': rhythm,
            'metadata': {
                'section': section,
                'chord_root': root,
                'chord_quality': quality,
                'tempo_bin': str(tempo_bin),
                'usage_count': usage_count,
                'avg_confidence': float(avg_confidence),
                'label_strength': group['label_strength'].mode()[0] if len(group) > 0 else 'bronze'
            }
        }
    
    logger.info(f"Extracted {len(patterns)} unique patterns")
    return patterns, expanded_df


def build_rule_selector(df: pd.DataFrame, patterns: Dict) -> Dict:
    """
    Build rule-based selector.
    
    Input: section, chord_root, chord_quality, tempo, time_sig
    Output: pattern_id (with confidence score)
    
    将来的にXGB/RandomForestで差し替え可能。
    """
    logger.info("Building rule-based selector...")
    
    # Build lookup table: (section, root, quality, tempo_bin) -> pattern_id
    lookup = {}
    
    for pattern_id, pattern in patterns.items():
        meta = pattern['metadata']
        key = (
            meta['section'],
            meta['chord_root'],
            meta['chord_quality'],
            meta['tempo_bin']
        )
        lookup[key] = {
            'pattern_id': pattern_id,
            'confidence': meta['avg_confidence']
        }
    
    selector = {
        'type': 'rule_based',
        'version': '1.0',
        'lookup_table': lookup,
        'fallback': {
            'pattern_id': 'default_major',
            'confidence': 0.5
        },
        'notes': 'Rule-based selector. Can be replaced with trained model (XGB/RF).'
    }
    
    logger.info(f"Built selector with {len(lookup)} lookup entries")
    return selector


def create_default_patterns() -> Dict:
    """Create minimal default patterns for fallback"""
    return {
        'default_major': {
            'key': 'fallback_major',
            'voicing': [0, 4, 7],
            'rhythm': 'standard_quarter',
            'metadata': {
                'section': 'any',
                'chord_root': 'any',
                'chord_quality': 'maj',
                'tempo_bin': 'any',
                'usage_count': 0,
                'avg_confidence': 0.5,
                'label_strength': 'bronze'
            }
        },
        'default_minor': {
            'key': 'fallback_minor',
            'voicing': [0, 3, 7],
            'rhythm': 'standard_quarter',
            'metadata': {
                'section': 'any',
                'chord_root': 'any',
                'chord_quality': 'min',
                'tempo_bin': 'any',
                'usage_count': 0,
                'avg_confidence': 0.5,
                'label_strength': 'bronze'
            }
        }
    }


def main():
    parser = argparse.ArgumentParser(
        description='Build stage2_guitar.pickle for AI pattern recommendation'
    )
    parser.add_argument(
        '--training-data',
        type=Path,
        default=Path('harmony_dataset/training_sequences.parquet'),
        help='Path to training sequences parquet'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('data/patterns/stage2_guitar.pickle'),
        help='Output pickle path'
    )
    parser.add_argument(
        '--min-usage',
        type=int,
        default=5,
        help='Minimum usage count for pattern inclusion'
    )
    
    args = parser.parse_args()
    
    # Load training data
    df = load_training_data(args.training_data)
    
    # Extract patterns (returns patterns dict and expanded_df)
    patterns, expanded_df = extract_patterns(df)
    
    # Filter by usage
    patterns = {
        pid: p for pid, p in patterns.items()
        if p['metadata']['usage_count'] >= args.min_usage
    }
    logger.info(f"Filtered to {len(patterns)} patterns (usage >= {args.min_usage})")
    
    # Add default patterns
    defaults = create_default_patterns()
    patterns.update(defaults)
    
    # Build selector
    selector = build_rule_selector(df, patterns)
    
    # Build final pickle structure
    stage2_data = {
        'version': '1.0',
        'created_utc': pd.Timestamp.utcnow().isoformat(),
        'data_source': {
            'training_sequences': str(args.training_data),
            'total_sequences': len(df),
            'total_songs': df['song_id'].nunique(),
            'gold_sequences': len(expanded_df[expanded_df['label_strength'] == 'gold']),
            'silver_sequences': len(expanded_df[expanded_df['label_strength'] == 'silver'])
        },
        'selector': selector,
        'patterns': patterns,
        'stats': {
            'total_patterns': len(patterns),
            'default_patterns': len(defaults),
            'avg_pattern_usage': sum(p['metadata']['usage_count'] for p in patterns.values()) / len(patterns),
            'coverage': {
                section: len([p for p in patterns.values() if p['metadata']['section'] == section])
                for section in ['Intro', 'Verse', 'Pre-Chorus', 'Chorus', 'Bridge', 'Outro']
            }
        }
    }
    
    # Write pickle
    args.output.parent.mkdir(parents=True, exist_ok=True)
    
    with open(args.output, 'wb') as f:
        pickle.dump(stage2_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    logger.info(f"✓ Wrote pickle to {args.output}")
    logger.info(f"\n{'='*60}")
    logger.info("Stage2 Guitar Pickle Summary")
    logger.info(f"{'='*60}")
    logger.info(f"Total patterns: {stage2_data['stats']['total_patterns']}")
    logger.info(f"Selector type: {selector['type']}")
    logger.info(f"Selector entries: {len(selector['lookup_table'])}")
    logger.info(f"\nSection coverage:")
    for section, count in stage2_data['stats']['coverage'].items():
        logger.info(f"  {section}: {count} patterns")
    logger.info(f"\nTo use:")
    logger.info(f"  export STAGE2_GUITAR_PATTERNS={args.output.absolute()}")
    logger.info(f"{'='*60}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
