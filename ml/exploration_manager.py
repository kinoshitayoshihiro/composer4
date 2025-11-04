"""
Exploration Manager for Pattern Discovery

Implements epsilon-greedy exploration strategy to discover new high-quality patterns.
After v3_ratio reaches 1.00, allocates 10% of traffic for exploration.

Phase 24.4: セクション別探索上限（cap_by_section）対応

Usage:
    manager = ExplorationManager(epsilon=0.10)
    
    # セクション別探索判定（Chorusは抑制）
    if manager.should_explore_section(section='Chorus'):
        pattern = manager.select_exploration_pattern(exploration_pool, section='Chorus')
    else:
        pattern = v3_candidates[0]  # Exploit best pattern
    
    manager.record_exploration_result(pattern_id, quality_score, section='Chorus')
"""

import json
import random
import logging
import yaml
from typing import Dict, List, Optional, Set
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np

logger = logging.getLogger(__name__)


class ExplorationManager:
    """Manages epsilon-greedy exploration for pattern discovery with section-specific caps"""
    
    def __init__(
        self,
        epsilon: float = 0.10,
        exploration_log_path: str = "data/exploration_log.json",
        discovered_patterns_path: str = "data/discovered_patterns.json",
        min_exploration_samples: int = 10,
        quality_threshold: float = 0.70,
        config_path: Optional[str] = None
    ):
        """
        Initialize exploration manager
        
        Args:
            epsilon: Exploration rate (0.0-1.0). Default 0.10 = 10% exploration
            exploration_log_path: Path to exploration results log
            discovered_patterns_path: Path to discovered patterns database
            min_exploration_samples: Minimum samples before pattern evaluation
            quality_threshold: Quality score threshold for pattern promotion (0-1)
            config_path: Path to exploration_config.yaml (optional)
        """
        self.epsilon = epsilon
        self.exploration_log_path = Path(exploration_log_path)
        self.discovered_patterns_path = Path(discovered_patterns_path)
        self.min_exploration_samples = min_exploration_samples
        self.quality_threshold = quality_threshold
        
        # Load exploration config (cap_by_section等)
        self.config = self._load_exploration_config(config_path)
        self.cap_by_section = self.config.get('exploration', {}).get('cap_by_section', {})
        
        # Ensure data directories exist
        self.exploration_log_path.parent.mkdir(parents=True, exist_ok=True)
        self.discovered_patterns_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Load existing exploration data
        self.exploration_log = self._load_exploration_log()
        self.discovered_patterns = self._load_discovered_patterns()
        
        logger.info(f"ExplorationManager initialized: epsilon={epsilon}, "
                   f"quality_threshold={quality_threshold}, "
                   f"cap_by_section={self.cap_by_section}")
    
    def _load_exploration_config(self, config_path: Optional[str] = None) -> Dict:
        """Load exploration_config.yaml"""
        if config_path is None:
            config_path = Path(__file__).parent.parent / "config" / "exploration_config.yaml"
        else:
            config_path = Path(config_path)
        
        if not config_path.exists():
            logger.warning(f"exploration_config.yaml not found: {config_path}")
            return {}
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            logger.info(f"Exploration config loaded from {config_path}")
            return config
        except Exception as e:
            logger.error(f"Failed to load exploration config: {e}")
            return {}
    
    def should_explore(self) -> bool:
        """
        Decide whether to explore (new pattern) or exploit (best known pattern)
        
        Returns:
            True if should explore, False if should exploit
        """
        explore = random.random() < self.epsilon
        logger.debug(f"Exploration decision: {'EXPLORE' if explore else 'EXPLOIT'}")
        return explore
    
    def should_explore_section(self, section: Optional[str] = None) -> bool:
        """
        Decide whether to explore with section-specific cap
        
        Phase 24.4: セクション別上限対応
        Chorusは3%上限、Verseは12%許容など
        
        Args:
            section: Song section (Chorus, Verse, etc.)
        
        Returns:
            True if should explore, False if should exploit
        """
        if section and section in self.cap_by_section:
            section_epsilon = self.cap_by_section[section]
            explore = random.random() < section_epsilon
            logger.debug(f"Exploration decision ({section}): {'EXPLORE' if explore else 'EXPLOIT'} "
                        f"(cap={section_epsilon:.2%})")
        else:
            # セクション未指定 or 未定義 → グローバルepsilon
            explore = self.should_explore()
        
        return explore
    
    def select_exploration_pattern(
        self,
        exploration_pool: List[str],
        section: Optional[str] = None
    ) -> Optional[str]:
        """
        Select a pattern from the exploration pool
        
        Prioritizes less-explored patterns using UCB (Upper Confidence Bound) strategy.
        
        Args:
            exploration_pool: List of pattern IDs available for exploration
            section: Optional section filter (Chorus, Verse, etc.)
        
        Returns:
            Selected pattern ID, or None if pool is empty
        """
        if not exploration_pool:
            logger.warning("Exploration pool is empty")
            return None
        
        # Filter by section if specified
        if section:
            section_pool = [
                p for p in exploration_pool 
                if self._pattern_matches_section(p, section)
            ]
            if section_pool:
                exploration_pool = section_pool
        
        # Calculate exploration scores using UCB
        scores = []
        total_explorations = sum(
            len(self.exploration_log.get(p, []))
            for p in exploration_pool
        )
        
        for pattern_id in exploration_pool:
            exploration_count = len(self.exploration_log.get(pattern_id, []))
            
            if exploration_count == 0:
                # Prioritize unexplored patterns (infinite score)
                scores.append(float('inf'))
            else:
                # UCB formula: mean_quality + sqrt(2 * ln(total) / count)
                mean_quality = self._get_pattern_mean_quality(pattern_id)
                exploration_bonus = np.sqrt(
                    2 * np.log(total_explorations + 1) / exploration_count
                )
                ucb_score = mean_quality + exploration_bonus
                scores.append(ucb_score)
        
        # Select pattern with highest UCB score
        selected_idx = np.argmax(scores)
        selected_pattern = exploration_pool[selected_idx]
        
        logger.info(f"Selected exploration pattern: {selected_pattern} "
                   f"(score={scores[selected_idx]:.3f})")
        return selected_pattern
    
    def record_exploration_result(
        self,
        pattern_id: str,
        quality_score: float,
        section: Optional[str] = None,
        metadata: Optional[Dict] = None
    ):
        """
        Record the result of an exploration attempt
        
        Args:
            pattern_id: Explored pattern ID
            quality_score: Quality metric (0-1, higher is better)
            section: Song section (Chorus, Verse, etc.)
            metadata: Additional metadata (user feedback, metrics, etc.)
        """
        timestamp = datetime.now().isoformat()
        
        result = {
            "timestamp": timestamp,
            "pattern_id": pattern_id,
            "quality_score": quality_score,
            "section": section,
            "metadata": metadata or {}
        }
        
        # Add to exploration log
        if pattern_id not in self.exploration_log:
            self.exploration_log[pattern_id] = []
        
        self.exploration_log[pattern_id].append(result)
        
        # Save updated log
        self._save_exploration_log()
        
        # Check if pattern should be promoted to discovered patterns
        self._evaluate_pattern_promotion(pattern_id)
        
        logger.info(f"Recorded exploration result: {pattern_id} "
                   f"quality={quality_score:.3f} section={section}")
    
    def _evaluate_pattern_promotion(self, pattern_id: str):
        """
        Evaluate if a pattern should be promoted to discovered patterns
        
        Promotion criteria:
        - At least min_exploration_samples samples
        - Mean quality score >= quality_threshold
        """
        results = self.exploration_log.get(pattern_id, [])
        
        if len(results) < self.min_exploration_samples:
            logger.debug(f"Pattern {pattern_id} needs more samples "
                        f"({len(results)}/{self.min_exploration_samples})")
            return
        
        mean_quality = self._get_pattern_mean_quality(pattern_id)
        
        if mean_quality >= self.quality_threshold:
            # Promote to discovered patterns
            self.discovered_patterns[pattern_id] = {
                "discovered_at": datetime.now().isoformat(),
                "mean_quality": mean_quality,
                "sample_count": len(results),
                "quality_std": float(np.std([r["quality_score"] for r in results]))
            }
            self._save_discovered_patterns()
            logger.info(f"Pattern {pattern_id} PROMOTED to discovered patterns "
                       f"(quality={mean_quality:.3f}, samples={len(results)})")
        else:
            logger.debug(f"Pattern {pattern_id} quality below threshold "
                        f"({mean_quality:.3f} < {self.quality_threshold})")
    
    def get_discovered_patterns(
        self,
        min_quality: Optional[float] = None,
        section: Optional[str] = None
    ) -> List[Dict]:
        """
        Get list of discovered high-quality patterns
        
        Args:
            min_quality: Optional minimum quality filter
            section: Optional section filter
        
        Returns:
            List of discovered patterns with metadata
        """
        patterns = []
        
        for pattern_id, metadata in self.discovered_patterns.items():
            # Apply quality filter
            if min_quality and metadata["mean_quality"] < min_quality:
                continue
            
            # Apply section filter
            if section and not self._pattern_matches_section(pattern_id, section):
                continue
            
            patterns.append({
                "pattern_id": pattern_id,
                **metadata
            })
        
        # Sort by mean quality (descending)
        patterns.sort(key=lambda x: x["mean_quality"], reverse=True)
        
        return patterns
    
    def get_exploration_summary(self, days: int = 7) -> Dict:
        """
        Get exploration activity summary for last N days
        
        Args:
            days: Number of days to analyze
        
        Returns:
            Summary statistics
        """
        cutoff_time = datetime.now() - timedelta(days=days)
        
        total_explorations = 0
        patterns_explored = set()
        patterns_promoted = 0
        quality_scores = []
        
        for pattern_id, results in self.exploration_log.items():
            for result in results:
                result_time = datetime.fromisoformat(result["timestamp"])
                
                if result_time >= cutoff_time:
                    total_explorations += 1
                    patterns_explored.add(pattern_id)
                    quality_scores.append(result["quality_score"])
        
        # Count promotions in the time window
        for pattern_id, metadata in self.discovered_patterns.items():
            discovered_time = datetime.fromisoformat(metadata["discovered_at"])
            if discovered_time >= cutoff_time:
                patterns_promoted += 1
        
        summary = {
            "period_days": days,
            "total_explorations": total_explorations,
            "unique_patterns_explored": len(patterns_explored),
            "patterns_promoted": patterns_promoted,
            "mean_quality": float(np.mean(quality_scores)) if quality_scores else 0.0,
            "quality_std": float(np.std(quality_scores)) if quality_scores else 0.0,
            "exploration_rate": self.epsilon,
            "total_discovered_patterns": len(self.discovered_patterns)
        }
        
        logger.info(f"Exploration summary (last {days}d): {summary}")
        return summary
    
    def _get_pattern_mean_quality(self, pattern_id: str) -> float:
        """Calculate mean quality score for a pattern"""
        results = self.exploration_log.get(pattern_id, [])
        if not results:
            return 0.0
        
        scores = [r["quality_score"] for r in results]
        return float(np.mean(scores))
    
    def _pattern_matches_section(self, pattern_id: str, section: str) -> bool:
        """
        Check if pattern matches section
        
        Note: This is a placeholder. In production, should query pattern metadata.
        """
        # TODO: Integrate with actual pattern metadata system
        return True
    
    def _load_exploration_log(self) -> Dict[str, List[Dict]]:
        """Load exploration log from JSON file"""
        if not self.exploration_log_path.exists():
            logger.info("No existing exploration log found, starting fresh")
            return {}
        
        try:
            with open(self.exploration_log_path, 'r') as f:
                log = json.load(f)
            logger.info(f"Loaded exploration log: {len(log)} patterns explored")
            return log
        except Exception as e:
            logger.error(f"Error loading exploration log: {e}")
            return {}
    
    def _save_exploration_log(self):
        """Save exploration log to JSON file"""
        try:
            with open(self.exploration_log_path, 'w') as f:
                json.dump(self.exploration_log, f, indent=2)
            logger.debug("Exploration log saved")
        except Exception as e:
            logger.error(f"Error saving exploration log: {e}")
    
    def _load_discovered_patterns(self) -> Dict[str, Dict]:
        """Load discovered patterns from JSON file"""
        if not self.discovered_patterns_path.exists():
            logger.info("No discovered patterns found, starting fresh")
            return {}
        
        try:
            with open(self.discovered_patterns_path, 'r') as f:
                patterns = json.load(f)
            logger.info(f"Loaded discovered patterns: {len(patterns)} patterns")
            return patterns
        except Exception as e:
            logger.error(f"Error loading discovered patterns: {e}")
            return {}
    
    def _save_discovered_patterns(self):
        """Save discovered patterns to JSON file"""
        try:
            with open(self.discovered_patterns_path, 'w') as f:
                json.dump(self.discovered_patterns, f, indent=2)
            logger.debug("Discovered patterns saved")
        except Exception as e:
            logger.error(f"Error saving discovered patterns: {e}")


def demo_exploration_workflow():
    """Demo: Typical exploration workflow"""
    
    # Initialize manager
    manager = ExplorationManager(epsilon=0.10, quality_threshold=0.70)
    
    # Exploration pool (new untested patterns)
    exploration_pool = [
        "PATTERN_NEW_001",
        "PATTERN_NEW_002",
        "PATTERN_NEW_003",
        "PATTERN_EXPERIMENTAL_A"
    ]
    
    # Simulate 100 requests
    for i in range(100):
        if manager.should_explore():
            # Exploration: select from pool
            pattern = manager.select_exploration_pattern(exploration_pool)
            
            # Simulate quality evaluation (random for demo)
            quality = random.uniform(0.5, 0.9)
            
            # Record result
            manager.record_exploration_result(
                pattern_id=pattern,
                quality_score=quality,
                section="Chorus"
            )
            
            print(f"[EXPLORE] Request {i}: {pattern} quality={quality:.3f}")
        else:
            # Exploitation: use best known pattern
            print(f"[EXPLOIT] Request {i}: Using v3 best pattern")
    
    # Get summary
    summary = manager.get_exploration_summary(days=7)
    print("\nExploration Summary:")
    print(f"  Total explorations: {summary['total_explorations']}")
    print(f"  Patterns explored: {summary['unique_patterns_explored']}")
    print(f"  Patterns promoted: {summary['patterns_promoted']}")
    print(f"  Mean quality: {summary['mean_quality']:.3f}")
    
    # Get discovered patterns
    discovered = manager.get_discovered_patterns()
    print(f"\nDiscovered Patterns: {len(discovered)}")
    for pattern in discovered:
        print(f"  {pattern['pattern_id']}: quality={pattern['mean_quality']:.3f} "
              f"samples={pattern['sample_count']}")


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    demo_exploration_workflow()
