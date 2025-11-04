#!/usr/bin/env python3
"""Build DrumPatternRecommender Pickle from LAMDA Index

LAMDA Index Pickle → DrumPatternRecommender Pickle変換

Input:
- LAMDA v2 index pickle (drums_index.pkl)
- Sharded pickles (drums_00000.pkl, ...)

Output:
- DrumPatternRecommender互換Pickle
  - pattern_dict: {pattern_id: DrumPattern}
  - xgb_model: None (未学習)
  - lr_model: None (未学習)
  - metadata: データセット情報

Usage:
    python scripts/build_drum_recommender_pickle.py \\
        --input output/rhythm_ai/drumclean_metadata/drums_index.pkl \\
        --output data/patterns/stage2_drums_rhythm_ai.pickle
"""

from __future__ import annotations

import argparse
import logging
import pickle
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s:%(name)s:%(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


class DrumPattern:
    """Drum pattern data class (compatible with DrumPatternRecommender)"""
    
    def __init__(
        self,
        pattern_id: str,
        filename: str,
        tempo_bpm: float,
        time_signature: str,
        note_count: int,
        duration_ms: float,
        pitches: dict,
        avg_velocity: int,
        family: str = "unknown",
        swing_ratio: float = 0.0,
        accent_profile: dict | None = None,
    ):
        self.pattern_id = pattern_id
        self.filename = filename
        self.tempo_bpm = tempo_bpm
        self.time_signature = time_signature
        self.note_count = note_count
        self.duration_ms = duration_ms
        self.pitches = pitches
        self.avg_velocity = avg_velocity
        self.family = family
        self.swing_ratio = swing_ratio
        self.accent_profile = accent_profile or {}
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary format"""
        return {
            "pattern_id": self.pattern_id,
            "filename": self.filename,
            "tempo_bin": self.tempo_bpm,  # For compatibility
            "time_sig_slots": 16 if self.time_signature == "4/4" else 12,
            "note_count": self.note_count,
            "duration_ms": self.duration_ms,
            "pitches": self.pitches,
            "avg_velocity": self.avg_velocity,
            "family": self.family,
            "swing_ratio": self.swing_ratio,
            "accent_profile": self.accent_profile,
        }


def extract_family_from_filename(filename: str) -> str:
    """Extract family from filename (heuristic)
    
    Examples:
        100_pop_142_beat_4-4_10 -> pop
        rock_120_beat -> rock
        jazz_swing_140 -> jazz
    """
    lower = filename.lower()
    
    # Common genre patterns
    genres = [
        "pop", "rock", "jazz", "funk", "soul", "blues",
        "latin", "reggae", "metal", "punk", "edm", "hip-hop",
        "country", "folk", "disco", "rnb", "swing", "shuffle"
    ]
    
    for genre in genres:
        if genre in lower:
            return genre
    
    return "unknown"


def calculate_swing_ratio(pitches: dict) -> float:
    """Calculate swing ratio from pitches distribution (heuristic)
    
    Swing is characterized by uneven eighth notes.
    This is a simplified heuristic based on hi-hat distribution.
    """
    # Hi-hat pitches: 42 (closed), 46 (open), 44 (pedal)
    hihat_pitches = [42, 44, 46]
    
    counts = pitches.get("counts", {})
    hihat_total = sum(counts.get(p, 0) for p in hihat_pitches)
    
    if hihat_total < 8:
        return 0.0  # Not enough hi-hat hits to determine
    
    # Simple heuristic: if hi-hat count is odd, assume some swing
    # Real swing detection would require timing analysis
    return 0.3 if hihat_total % 2 == 1 else 0.0


def build_accent_profile(pitches: dict, note_count: int) -> dict[str, list[int]]:
    """Build accent profile from pitches distribution
    
    Returns:
        {
            "kick": [1, 0, 0, 0, 1, 0, 0, 0, ...],  # 16 steps
            "snare": [0, 0, 1, 0, 0, 0, 1, 0, ...],
            "hat": [1, 1, 1, 1, 1, 1, 1, 1, ...]
        }
    """
    counts = pitches.get("counts", {})
    
    # Standard GM drum mapping
    kick_pitches = [35, 36]
    snare_pitches = [38, 40]
    hihat_pitches = [42, 44, 46]
    
    # Simplified 16-step profile
    # This is a heuristic - real accent profile would require timing data
    profile = {
        "kick": [0] * 16,
        "snare": [0] * 16,
        "hat": [0] * 16,
    }
    
    # Distribute counts across 16 steps (simplified)
    kick_count = sum(counts.get(p, 0) for p in kick_pitches)
    snare_count = sum(counts.get(p, 0) for p in snare_pitches)
    hat_count = sum(counts.get(p, 0) for p in hihat_pitches)
    
    # Basic pattern: distribute evenly
    if kick_count > 0:
        profile["kick"] = [1 if i % 4 == 0 else 0 for i in range(16)]
    if snare_count > 0:
        profile["snare"] = [1 if i % 4 == 2 else 0 for i in range(16)]
    if hat_count > 0:
        profile["hat"] = [1 if i % 2 == 0 else 0 for i in range(16)]
    
    return profile


def load_lamda_index(index_path: Path) -> dict:
    """Load LAMDA v2 index pickle"""
    with open(index_path, "rb") as f:
        index = pickle.load(f)
    
    logger.info(f"Loaded index: {index.get('version')}")
    logger.info(f"Total files: {index.get('total_files')}")
    logger.info(f"Shards: {len(index.get('shards', []))}")
    
    return index


def build_pattern_dict(index_path: Path, max_patterns: int | None = None) -> dict[str, dict]:
    """Build pattern dictionary from LAMDA index
    
    Args:
        index_path: Path to drums_index.pkl
        max_patterns: Maximum patterns to load (None = all)
    
    Returns:
        {pattern_id: pattern_data_dict}
    """
    index = load_lamda_index(index_path)
    index_dir = index_path.parent
    
    pattern_dict = {}
    total_loaded = 0
    
    for shard_info in index["shards"]:
        shard_path = index_dir / shard_info["path"]
        
        logger.info(f"Loading shard: {shard_path.name}")
        
        with open(shard_path, "rb") as f:
            shard = pickle.load(f)
        
        loops = shard["loops"]
        
        for loop in loops:
            pattern_id = loop["md5"]
            filename = loop["filename"]
            
            # Extract features
            family = extract_family_from_filename(filename)
            swing_ratio = calculate_swing_ratio(loop["pitches"])
            accent_profile = build_accent_profile(loop["pitches"], loop["note_count"])
            
            # Create DrumPattern
            pattern = DrumPattern(
                pattern_id=pattern_id,
                filename=filename,
                tempo_bpm=loop["bpm"],
                time_signature=loop.get("time_signature", "4/4"),
                note_count=loop["note_count"],
                duration_ms=loop["duration_ms"],
                pitches=loop["pitches"],
                avg_velocity=loop.get("avg_velocity", 64),
                family=family,
                swing_ratio=swing_ratio,
                accent_profile=accent_profile,
            )
            
            pattern_dict[pattern_id] = pattern.to_dict()
            total_loaded += 1
            
            if max_patterns and total_loaded >= max_patterns:
                logger.info(f"Reached max_patterns limit: {max_patterns}")
                return pattern_dict
        
        logger.info(f"  Loaded {len(loops)} patterns from {shard_path.name} (total: {total_loaded})")
    
    logger.info(f"Total patterns loaded: {total_loaded}")
    
    # Log family distribution
    family_counts = defaultdict(int)
    for pattern in pattern_dict.values():
        family_counts[pattern["family"]] += 1
    
    logger.info("Family distribution:")
    for family, count in sorted(family_counts.items(), key=lambda x: -x[1]):
        logger.info(f"  {family}: {count} ({count / total_loaded * 100:.1f}%)")
    
    return pattern_dict


def build_recommender_pickle(
    index_path: Path,
    output_path: Path,
    max_patterns: int | None = None,
) -> None:
    """Build DrumPatternRecommender pickle from LAMDA index
    
    Args:
        index_path: Path to drums_index.pkl
        output_path: Output pickle path
        max_patterns: Maximum patterns to load (None = all)
    """
    logger.info("=" * 70)
    logger.info("Building DrumPatternRecommender Pickle")
    logger.info("=" * 70)
    
    # Load pattern dictionary
    pattern_dict = build_pattern_dict(index_path, max_patterns=max_patterns)
    
    # Build recommender pickle structure
    recommender_data = {
        "pattern_dict": pattern_dict,
        "xgb_model": None,  # Not trained yet
        "lr_model": None,   # Not trained yet
        "class_labels": None,
        "feature_names": None,
        "scaler": None,
        "metadata": {
            "source": str(index_path),
            "total_patterns": len(pattern_dict),
            "version": "rhythm_ai_v1",
            "ml_trained": False,
        },
    }
    
    # Save pickle
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(recommender_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    logger.info(f"✅ Pickle saved: {output_path}")
    logger.info(f"   Total patterns: {len(pattern_dict)}")
    logger.info(f"   ML models: Not trained (rule-based only)")
    logger.info("=" * 70)


def main() -> int:
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Build DrumPatternRecommender pickle from LAMDA index",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="LAMDA index pickle (drums_index.pkl)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/patterns/stage2_drums_rhythm_ai.pickle"),
        help="Output pickle path",
    )
    parser.add_argument(
        "--max-patterns",
        type=int,
        default=None,
        help="Maximum patterns to load (for testing)",
    )
    
    args = parser.parse_args()
    
    if not args.input.exists():
        logger.error(f"Input index not found: {args.input}")
        return 1
    
    try:
        build_recommender_pickle(
            index_path=args.input,
            output_path=args.output,
            max_patterns=args.max_patterns,
        )
        return 0
    except Exception as e:
        logger.exception(f"Failed to build pickle: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
