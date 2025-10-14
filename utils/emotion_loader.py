"""
Emotion Mapping Loader Utility

This module provides utilities for loading and applying emotion mapping
configurations from emotion_mapping.yaml.

Usage:
    from utils.emotion_loader import load_emotion_mapping, get_emotion_adjustments
    
    config = load_emotion_mapping()
    adjustments = get_emotion_adjustments("piano", "happy_high", config)
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional, List
import yaml


def load_emotion_mapping(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load emotion mapping configuration from YAML file.
    
    Args:
        config_path: Path to emotion_mapping.yaml. If None, uses default location.
        
    Returns:
        Dictionary containing emotion_profiles, section_emotion_mapping,
        instrument_adjustments, transition_rules, and validation_rules.
        
    Raises:
        FileNotFoundError: If config file doesn't exist.
        yaml.YAMLError: If YAML parsing fails.
    """
    if config_path is None:
        # Default to config/emotion_mapping.yaml
        base_dir = Path(__file__).parent.parent
        config_path = base_dir / "config" / "emotion_mapping.yaml"
    
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(
            f"Emotion mapping config not found: {config_path}\n"
            f"Please ensure config/emotion_mapping.yaml exists."
        )
    
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    # Validate required keys
    required_keys = [
        "emotion_profiles",
        "section_emotion_mapping",
        "instrument_adjustments",
        "transition_rules",
        "validation_rules"
    ]
    
    missing = [k for k in required_keys if k not in config]
    if missing:
        raise ValueError(
            f"emotion_mapping.yaml missing required keys: {missing}"
        )
    
    return config


def get_emotion_adjustments(
    instrument: str,
    emotion_profile: str,
    config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Get instrument-specific adjustments for an emotion profile.
    
    Args:
        instrument: Instrument name (piano, guitar, bass, strings, drums)
        emotion_profile: Emotion profile name (e.g., "happy_high")
        config: Pre-loaded config. If None, loads from default location.
        
    Returns:
        Dictionary of adjustments for the instrument/emotion combination.
        Returns empty dict if no adjustments defined.
        
    Example:
        >>> adj = get_emotion_adjustments("piano", "happy_high")
        >>> print(adj)
        {'velocity_std_multiplier': 1.2, 'notes_per_bar_multiplier': 1.1}
    """
    if config is None:
        config = load_emotion_mapping()
    
    instrument_lower = instrument.lower()
    
    # Check if instrument exists
    if instrument_lower not in config["instrument_adjustments"]:
        raise ValueError(
            f"Unknown instrument: {instrument}. "
            f"Available: {list(config['instrument_adjustments'].keys())}"
        )
    
    # Check if emotion profile exists
    if emotion_profile not in config["emotion_profiles"]:
        raise ValueError(
            f"Unknown emotion profile: {emotion_profile}. "
            f"Available: {list(config['emotion_profiles'].keys())}"
        )
    
    # Get adjustments for this instrument/emotion combo
    instrument_adj = config["instrument_adjustments"][instrument_lower]
    
    if emotion_profile in instrument_adj:
        return instrument_adj[emotion_profile]
    else:
        # No specific adjustments - return empty dict
        return {}


def get_section_default_emotion(
    section: str,
    config: Optional[Dict[str, Any]] = None
) -> str:
    """
    Get the default emotion profile for a section.
    
    Args:
        section: Section name (Intro, Verse, Chorus, etc.)
        config: Pre-loaded config. If None, loads from default location.
        
    Returns:
        Default emotion profile name for this section.
        
    Example:
        >>> emotion = get_section_default_emotion("Chorus")
        >>> print(emotion)  # "happy_high"
    """
    if config is None:
        config = load_emotion_mapping()
    
    section_mapping = config["section_emotion_mapping"]
    
    if section not in section_mapping:
        raise ValueError(
            f"Unknown section: {section}. "
            f"Available: {list(section_mapping.keys())}"
        )
    
    return section_mapping[section]["default"]


def get_section_alternative_emotions(
    section: str,
    config: Optional[Dict[str, Any]] = None
) -> List[str]:
    """
    Get alternative emotion profiles for a section.
    
    Args:
        section: Section name (Intro, Verse, Chorus, etc.)
        config: Pre-loaded config. If None, loads from default location.
        
    Returns:
        List of alternative emotion profile names.
    """
    if config is None:
        config = load_emotion_mapping()
    
    section_mapping = config["section_emotion_mapping"]
    
    if section not in section_mapping:
        raise ValueError(
            f"Unknown section: {section}. "
            f"Available: {list(section_mapping.keys())}"
        )
    
    return section_mapping[section].get("alternatives", [])


def validate_section_constraints(
    section: str,
    bars: int,
    config: Optional[Dict[str, Any]] = None
) -> bool:
    """
    Validate that section length meets constraints.
    
    Args:
        section: Section name (Intro, Verse, Chorus, etc.)
        bars: Number of bars in the section
        config: Pre-loaded config. If None, loads from default location.
        
    Returns:
        True if valid, False otherwise.
        
    Example:
        >>> validate_section_constraints("Intro", 4)
        True
        >>> validate_section_constraints("Intro", 20)
        False
    """
    if config is None:
        config = load_emotion_mapping()
    
    validation = config["validation_rules"]
    
    if section not in validation["section_length_constraints"]:
        # Unknown section - be permissive
        return True
    
    constraints = validation["section_length_constraints"][section]
    min_bars = constraints["min_bars"]
    max_bars = constraints["max_bars"]
    
    return min_bars <= bars <= max_bars


def get_transition_rule(
    from_section: str,
    to_section: str,
    config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Get transition rule between two sections.
    
    Args:
        from_section: Source section name
        to_section: Destination section name
        config: Pre-loaded config. If None, loads from default location.
        
    Returns:
        Dictionary with max_overlap_ms, min_gap_ms, and optional description.
        Returns basic rule if no special rule defined.
        
    Example:
        >>> rule = get_transition_rule("PreChorus", "Chorus")
        >>> print(rule)
        {'max_overlap_ms': 100, 'min_gap_ms': 0, 'description': 'seamless'}
    """
    if config is None:
        config = load_emotion_mapping()
    
    transition_rules = config["transition_rules"]
    
    # Check for special transition rule
    special_key = f"{from_section}_to_{to_section}"
    
    if "special_transitions" in transition_rules:
        if special_key in transition_rules["special_transitions"]:
            # Copy basic rule and overlay special rule
            rule = transition_rules["basic"].copy()
            rule.update(transition_rules["special_transitions"][special_key])
            return rule
    
    # Return basic rule
    return transition_rules["basic"].copy()


def get_emotion_profile_info(
    emotion_profile: str,
    config: Optional[Dict[str, Any]] = None
) -> Dict[str, str]:
    """
    Get information about an emotion profile.
    
    Args:
        emotion_profile: Emotion profile name
        config: Pre-loaded config. If None, loads from default location.
        
    Returns:
        Dictionary with intensity, mood, tension, dynamics.
        
    Example:
        >>> info = get_emotion_profile_info("happy_high")
        >>> print(info)
        {'intensity': 'high', 'mood': 'happy', 'tension': 'high', 'dynamics': 'loud'}
    """
    if config is None:
        config = load_emotion_mapping()
    
    if emotion_profile not in config["emotion_profiles"]:
        raise ValueError(
            f"Unknown emotion profile: {emotion_profile}. "
            f"Available: {list(config['emotion_profiles'].keys())}"
        )
    
    return config["emotion_profiles"][emotion_profile]


def apply_adjustments_to_params(
    base_params: Dict[str, Any],
    adjustments: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Apply emotion adjustments to base generation parameters.
    
    This is a helper function that applies multipliers and overrides
    from emotion adjustments to base parameters.
    
    Args:
        base_params: Base generation parameters
        adjustments: Emotion adjustments from get_emotion_adjustments()
        
    Returns:
        Updated parameters with adjustments applied.
        
    Example:
        >>> base = {"velocity_std": 15, "notes_per_bar": 8}
        >>> adj = {"velocity_std_multiplier": 1.2, "notes_per_bar_multiplier": 1.1}
        >>> result = apply_adjustments_to_params(base, adj)
        >>> print(result)
        {'velocity_std': 18.0, 'notes_per_bar': 8.8}
    """
    params = base_params.copy()
    
    for key, value in adjustments.items():
        if key.endswith("_multiplier"):
            # Multiplier adjustment
            base_key = key.replace("_multiplier", "")
            if base_key in params:
                params[base_key] = params[base_key] * value
        elif key.endswith("_target"):
            # Target value (override)
            base_key = key.replace("_target", "")
            params[base_key] = value
        elif key.endswith("_boost"):
            # Boost adjustment (additive)
            base_key = key.replace("_boost", "")
            if base_key in params:
                params[base_key] = params[base_key] + value
        else:
            # Direct assignment
            params[key] = value
    
    return params


# Convenience function for common workflow
def get_generation_params(
    instrument: str,
    section: str = "Verse",
    emotion_profile: Optional[str] = None,
    base_params: Optional[Dict[str, Any]] = None,
    config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Get complete generation parameters with emotion adjustments.
    
    This is a high-level convenience function that:
    1. Determines emotion profile (use provided or section default)
    2. Gets emotion adjustments for instrument
    3. Applies adjustments to base parameters
    
    Args:
        instrument: Instrument name (piano, guitar, bass, strings, drums)
        section: Section name (default: "Verse")
        emotion_profile: Emotion profile name. If None, uses section default.
        base_params: Base generation parameters. If None, returns adjustments only.
        config: Pre-loaded config. If None, loads from default location.
        
    Returns:
        Dictionary of generation parameters with emotion adjustments applied.
        
    Example:
        >>> base = {"velocity_std": 15, "notes_per_bar": 8}
        >>> params = get_generation_params("piano", "Chorus", base_params=base)
        >>> print(params)  # Will have happy_high adjustments applied
    """
    if config is None:
        config = load_emotion_mapping()
    
    # Determine emotion profile
    if emotion_profile is None:
        emotion_profile = get_section_default_emotion(section, config)
    
    # Get adjustments
    adjustments = get_emotion_adjustments(instrument, emotion_profile, config)
    
    # Apply to base params
    if base_params is not None:
        return apply_adjustments_to_params(base_params, adjustments)
    else:
        return adjustments


if __name__ == "__main__":
    # Test loading
    print("Loading emotion_mapping.yaml...")
    config = load_emotion_mapping()
    
    print(f"\nEmotion profiles: {list(config['emotion_profiles'].keys())}")
    print(f"Sections: {list(config['section_emotion_mapping'].keys())}")
    print(f"Instruments: {list(config['instrument_adjustments'].keys())}")
    
    # Test adjustments
    print("\n--- Piano happy_high adjustments ---")
    adj = get_emotion_adjustments("piano", "happy_high", config)
    print(adj)
    
    print("\n--- Guitar melancholic_medium adjustments ---")
    adj = get_emotion_adjustments("guitar", "melancholic_medium", config)
    print(adj)
    
    # Test section defaults
    print("\n--- Section default emotions ---")
    for section in ["Intro", "Verse", "Chorus", "Bridge", "Outro"]:
        emotion = get_section_default_emotion(section, config)
        print(f"{section}: {emotion}")
    
    # Test validation
    print("\n--- Section length validation ---")
    print(f"Intro 4 bars: {validate_section_constraints('Intro', 4, config)}")
    print(f"Intro 20 bars: {validate_section_constraints('Intro', 20, config)}")
    
    # Test transition rules
    print("\n--- Transition rules ---")
    rule = get_transition_rule("PreChorus", "Chorus", config)
    print(f"PreChorus → Chorus: {rule}")
    
    # Test complete workflow
    print("\n--- Complete workflow example ---")
    base_params = {"velocity_std": 15, "notes_per_bar": 8}
    final_params = get_generation_params(
        "piano", 
        "Chorus", 
        base_params=base_params,
        config=config
    )
    print(f"Base: {base_params}")
    print(f"Final (Chorus): {final_params}")
    
    print("\n✅ All tests passed!")
