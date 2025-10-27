"""
Pattern Quality Config Loader
Phase 24.3: Load blacklist/whitelist for Pattern Recommender

Usage:
    from ml.pattern_quality_config import get_blacklist, get_whitelist
    
    blacklist = get_blacklist()
    if pattern_id in blacklist:
        # Skip this pattern
        pass
"""

import yaml
from pathlib import Path
from typing import Set, Dict, Optional
import logging

logger = logging.getLogger(__name__)


def load_pattern_quality_config(
    config_path: Optional[Path] = None
) -> dict:
    """
    Load pattern_quality.yaml configuration.
    
    Args:
        config_path: Path to pattern_quality.yaml. If None, use default.
        
    Returns:
        Parsed YAML config dict
    """
    if config_path is None:
        base_dir = Path(__file__).parent.parent
        config_path = base_dir / "config" / "pattern_quality.yaml"
    
    if not config_path.exists():
        logger.warning(
            f"Pattern quality config not found: {config_path}. "
            f"Using empty blacklist/whitelist."
        )
        return {
            'blacklist': [],
            'whitelist': [],
            'quality_rules': {},
            'recommender': {},
        }
    
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def get_blacklist(config_path: Optional[Path] = None) -> Set[str]:
    """
    Get blacklist pattern IDs.
    
    Args:
        config_path: Path to pattern_quality.yaml
        
    Returns:
        Set of blacklisted pattern IDs
    """
    cfg = load_pattern_quality_config(config_path)
    blacklist = cfg.get('blacklist') or []  # Handle None case
    
    if blacklist:
        logger.info(f"Loaded {len(blacklist)} blacklisted patterns")
    
    return set(blacklist)


def get_whitelist(config_path: Optional[Path] = None) -> Set[str]:
    """
    Get whitelist pattern IDs.
    
    Args:
        config_path: Path to pattern_quality.yaml
        
    Returns:
        Set of whitelisted pattern IDs
    """
    cfg = load_pattern_quality_config(config_path)
    whitelist = cfg.get('whitelist') or []  # Handle None case
    
    if whitelist:
        logger.info(f"Loaded {len(whitelist)} whitelisted patterns")
    
    return set(whitelist)


def get_quality_rules(config_path: Optional[Path] = None) -> Dict:
    """
    Get quality rules for Pattern Quality Learner.
    
    Args:
        config_path: Path to pattern_quality.yaml
        
    Returns:
        Dict with quality rules (blacklist_threshold, etc.)
    """
    cfg = load_pattern_quality_config(config_path)
    return cfg.get('quality_rules', {
        'blacklist_threshold': 0.05,
        'evaluation_window_days': 7,
        'min_samples': 10,
        'auto_update_enabled': False,
    })


def get_recommender_config(config_path: Optional[Path] = None) -> Dict:
    """
    Get recommender integration config.
    
    Args:
        config_path: Path to pattern_quality.yaml
        
    Returns:
        Dict with recommender config (apply_blacklist, etc.)
    """
    cfg = load_pattern_quality_config(config_path)
    return cfg.get('recommender', {
        'apply_blacklist': True,
        'apply_whitelist': False,
        'fallback_on_blacklist': True,
    })


def add_to_blacklist(
    pattern_id: str,
    config_path: Optional[Path] = None,
    reason: Optional[str] = None
) -> bool:
    """
    Add pattern to blacklist (manual or automated).
    
    Args:
        pattern_id: Pattern ID to blacklist
        config_path: Path to pattern_quality.yaml
        reason: Optional reason for blacklisting
        
    Returns:
        True if added, False if already exists
    """
    if config_path is None:
        base_dir = Path(__file__).parent.parent
        config_path = base_dir / "config" / "pattern_quality.yaml"
    
    cfg = load_pattern_quality_config(config_path)
    blacklist = cfg.get('blacklist', [])
    
    if pattern_id in blacklist:
        logger.info(f"Pattern already blacklisted: {pattern_id}")
        return False
    
    blacklist.append(pattern_id)
    cfg['blacklist'] = blacklist
    
    # Add comment if reason provided
    if reason:
        logger.info(f"Blacklisting {pattern_id}: {reason}")
    
    # Save updated config
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)
    
    logger.info(f"✅ Added to blacklist: {pattern_id}")
    return True


def remove_from_blacklist(
    pattern_id: str,
    config_path: Optional[Path] = None
) -> bool:
    """
    Remove pattern from blacklist.
    
    Args:
        pattern_id: Pattern ID to remove
        config_path: Path to pattern_quality.yaml
        
    Returns:
        True if removed, False if not found
    """
    if config_path is None:
        base_dir = Path(__file__).parent.parent
        config_path = base_dir / "config" / "pattern_quality.yaml"
    
    cfg = load_pattern_quality_config(config_path)
    blacklist = cfg.get('blacklist', [])
    
    if pattern_id not in blacklist:
        logger.warning(f"Pattern not in blacklist: {pattern_id}")
        return False
    
    blacklist.remove(pattern_id)
    cfg['blacklist'] = blacklist
    
    # Save updated config
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)
    
    logger.info(f"✅ Removed from blacklist: {pattern_id}")
    return True


if __name__ == '__main__':
    # Test
    print("=== Pattern Quality Config Test ===")
    
    blacklist = get_blacklist()
    print(f"Blacklist: {blacklist if blacklist else '(empty)'}")
    
    whitelist = get_whitelist()
    print(f"Whitelist: {whitelist if whitelist else '(empty)'}")
    
    rules = get_quality_rules()
    print(f"Quality Rules: {rules}")
    
    recommender_cfg = get_recommender_config()
    print(f"Recommender Config: {recommender_cfg}")
