#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build Stage2 Bass Pickle (Rule-based Selector)

Bass パターン辞書の構築。ルールベースselectorでAI経路をONにする。
後からXGBモデルに差し替え可能な構造。

Patterns:
- ROOT_8ths: ルート音8分音符刻み
- ROOT_5TH_ALT: ルート→5度→ルート（交互）
- APPROACH: アプローチノート（半音下→目的音）
- WALKING_4: ウォーキングベース（4分音符、スケール音）

Features for ML (将来):
- section, chord_root, chord_quality, bar_pos, tempo_bin, accent_level
"""

import sys
import logging
import pickle
from pathlib import Path
from typing import Dict, Any, List, Tuple
from collections import defaultdict
from datetime import datetime

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# ========== Pattern Definitions ==========

BASS_PATTERNS = {
    # ROOT_8ths: ルート音8分音符刻み（密度高）
    "root_8ths_standard": {
        "key": "root_8ths",
        "voicing": [0],  # Root only
        "rhythm": "standard_8ths",
        "metadata": {
            "section": "any",
            "chord_quality": "any",
            "tempo_bin": "medium",
            "usage_count": 100,
            "avg_confidence": 0.85,
            "label_strength": "gold",
            "description": "Root note, 8th notes",
        },
    },
    # ROOT_5TH_ALT: ルート→5度交互（アクセント強）
    "root_5th_alt_chorus": {
        "key": "root_5th_alt",
        "voicing": [0, 7],  # Root + P5
        "rhythm": "alt_root_5th_8ths",
        "metadata": {
            "section": "Chorus",
            "chord_quality": "any",
            "tempo_bin": "medium",
            "usage_count": 80,
            "avg_confidence": 0.88,
            "label_strength": "gold",
            "description": "Root-5th alternating, 8th notes",
        },
    },
    "root_5th_alt_verse": {
        "key": "root_5th_alt",
        "voicing": [0, 7],
        "rhythm": "alt_root_5th_quarter",
        "metadata": {
            "section": "Verse",
            "chord_quality": "any",
            "tempo_bin": "medium",
            "usage_count": 70,
            "avg_confidence": 0.82,
            "label_strength": "gold",
            "description": "Root-5th alternating, quarter notes",
        },
    },
    # APPROACH: アプローチノート（半音下→目的音）
    "approach_chorus_major": {
        "key": "approach",
        "voicing": [-1, 0],  # Chromatic approach from below
        "rhythm": "approach_16ths",
        "metadata": {
            "section": "Chorus",
            "chord_quality": "maj",
            "tempo_bin": "medium",
            "usage_count": 60,
            "avg_confidence": 0.80,
            "label_strength": "silver",
            "description": "Chromatic approach, 16th notes",
        },
    },
    "approach_bridge_minor": {
        "key": "approach",
        "voicing": [-1, 0],
        "rhythm": "approach_8ths",
        "metadata": {
            "section": "Bridge",
            "chord_quality": "min",
            "tempo_bin": "medium",
            "usage_count": 50,
            "avg_confidence": 0.78,
            "label_strength": "silver",
            "description": "Chromatic approach, 8th notes",
        },
    },
    # WALKING_4: ウォーキングベース（4分音符、スケール音）
    "walking_chorus_major": {
        "key": "walking_4",
        "voicing": [0, 2, 4, 7],  # Root, M2, M3, P5 (scale tones)
        "rhythm": "walking_quarter",
        "metadata": {
            "section": "Chorus",
            "chord_quality": "maj",
            "tempo_bin": "medium",
            "usage_count": 90,
            "avg_confidence": 0.87,
            "label_strength": "gold",
            "description": "Walking bass, quarter notes, scale tones",
        },
    },
    "walking_verse_minor": {
        "key": "walking_4",
        "voicing": [0, 2, 3, 7],  # Root, M2, m3, P5 (minor scale tones)
        "rhythm": "walking_quarter",
        "metadata": {
            "section": "Verse",
            "chord_quality": "min",
            "tempo_bin": "medium",
            "usage_count": 75,
            "avg_confidence": 0.84,
            "label_strength": "gold",
            "description": "Walking bass, quarter notes, minor scale",
        },
    },
    # Sparse patterns (Intro/Bridge)
    "sparse_whole_intro": {
        "key": "sparse",
        "voicing": [0],
        "rhythm": "sparse_whole",
        "metadata": {
            "section": "Intro",
            "chord_quality": "any",
            "tempo_bin": "any",
            "usage_count": 40,
            "avg_confidence": 0.75,
            "label_strength": "silver",
            "description": "Sparse, whole notes",
        },
    },
    "sparse_half_bridge": {
        "key": "sparse",
        "voicing": [0],
        "rhythm": "sparse_half",
        "metadata": {
            "section": "Bridge",
            "chord_quality": "any",
            "tempo_bin": "slow",
            "usage_count": 45,
            "avg_confidence": 0.77,
            "label_strength": "silver",
            "description": "Sparse, half notes",
        },
    },
    # Default fallback
    "default_major": {
        "key": "default",
        "voicing": [0, 7],
        "rhythm": "standard_quarter",
        "metadata": {
            "section": "any",
            "chord_quality": "maj",
            "tempo_bin": "any",
            "usage_count": 120,
            "avg_confidence": 0.90,
            "label_strength": "gold",
            "description": "Default major, quarter notes",
        },
    },
    "default_minor": {
        "key": "default",
        "voicing": [0, 7],
        "rhythm": "standard_quarter",
        "metadata": {
            "section": "any",
            "chord_quality": "min",
            "tempo_bin": "any",
            "usage_count": 110,
            "avg_confidence": 0.89,
            "label_strength": "gold",
            "description": "Default minor, quarter notes",
        },
    },
}


def get_tempo_bin(tempo: float) -> str:
    """Tempo binning"""
    if tempo < 90:
        return "slow"
    elif tempo < 130:
        return "medium"
    else:
        return "fast"


def build_rule_selector(patterns: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build rule-based selector

    Lookup table: (section, chord_quality, tempo_bin) -> pattern_id

    Args:
        patterns: Pattern dictionary

    Returns:
        Selector configuration
    """
    logger.info("Building rule-based selector...")

    lookup_table = {}

    for pattern_id, pattern in patterns.items():
        meta = pattern["metadata"]
        section = meta.get("section", "any")
        chord_quality = meta.get("chord_quality", "any")
        tempo_bin = meta.get("tempo_bin", "any")

        # Exact match
        key = (section, chord_quality, tempo_bin)
        if key not in lookup_table:
            lookup_table[key] = pattern_id

        # Wildcard variants
        if section != "any":
            lookup_table[("any", chord_quality, tempo_bin)] = pattern_id
        if chord_quality != "any":
            lookup_table[(section, "any", tempo_bin)] = pattern_id
        if tempo_bin != "any":
            lookup_table[(section, chord_quality, "any")] = pattern_id

    # Convert tuple keys to string keys for pickle serialization
    lookup_table_str = {f"{s}|{q}|{t}": pid for (s, q, t), pid in lookup_table.items()}

    selector = {"type": "rule_based", "lookup_table": lookup_table_str, "fallback": "default_major"}

    logger.info(f"  Rule selector built: {len(lookup_table_str)} rules")

    return selector


def compute_stats(patterns: Dict[str, Any]) -> Dict[str, Any]:
    """Compute pattern statistics"""
    total = len(patterns)

    quality_dist = defaultdict(int)
    section_dist = defaultdict(int)
    rhythm_dist = defaultdict(int)

    for pattern in patterns.values():
        meta = pattern["metadata"]
        quality_dist[meta.get("label_strength", "unknown")] += 1
        section_dist[meta.get("section", "unknown")] += 1
        rhythm_dist[pattern.get("rhythm", "unknown")] += 1

    stats = {
        "total_patterns": total,
        "quality_distribution": dict(quality_dist),
        "section_distribution": dict(section_dist),
        "rhythm_distribution": dict(rhythm_dist),
    }

    return stats


def build_pickle(patterns: Dict[str, Any], output_path: str) -> None:
    """
    Build Stage2 Bass pickle

    Args:
        patterns: Pattern dictionary
        output_path: Output pickle path
    """
    logger.info("Building Stage2 Bass pickle...")

    # Build selector
    selector = build_rule_selector(patterns)

    # Compute stats
    stats = compute_stats(patterns)

    # Meta
    meta = {
        "version": "1.0",
        "instrument": "bass",
        "provider": "rule_based",
        "created_utc": datetime.utcnow().isoformat(),
        "description": "Stage2 Bass patterns with rule-based selector (ML-ready structure)",
    }

    # Pickle structure
    data = {
        "version": "1.0",
        "patterns": patterns,
        "selector": selector,
        "stats": stats,
        "meta": meta,
    }

    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    logger.info(f"Pickle saved: {output_path}")
    logger.info(f"  Version: {data['version']}")
    logger.info(f"  Patterns: {len(patterns)}")
    logger.info(f"  Selector type: {selector['type']}")
    logger.info(f"  Stats:")
    for key, value in stats.items():
        logger.info(f"    {key}: {value}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Build Stage2 Bass Pickle")
    parser.add_argument(
        "--output", type=str, default="data/patterns/stage2_bass.pickle", help="Output pickle path"
    )

    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("Building Stage2 Bass Pickle (Rule-based)")
    logger.info("=" * 60)

    # Build pickle
    build_pickle(patterns=BASS_PATTERNS, output_path=args.output)

    logger.info("\n" + "=" * 60)
    logger.info("Bass pickle build complete!")
    logger.info("=" * 60)
    logger.info(f"\nUsage:")
    logger.info(f"  export STAGE2_BASS_PATTERNS={Path(args.output).resolve()}")
    logger.info(f"  # Then initialize BassGeneratorStage2")


if __name__ == "__main__":
    main()
