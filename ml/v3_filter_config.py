"""
V3 Filter Configuration Loader
Phase 24.3: YAML-based V3 filter settings for all instruments

Usage:
    from ml.v3_filter_config import get_v3_filter_params
    
    # Guitar, Chorus section
    params = get_v3_filter_params(instrument='guitar', section='chorus')
    results = recommender.recommend(
        query,
        filter_v3_only=params['enabled'],
        min_proba=params['min_proba'],
        min_margin=params['min_margin']
    )
"""

import yaml
from pathlib import Path
from typing import Dict, Optional


def load_gate_config(config_path: Optional[Path] = None) -> dict:
    """
    Load gate_prod.yaml configuration.
    
    Args:
        config_path: Path to gate_prod.yaml. If None, use default path.
        
    Returns:
        Parsed YAML configuration dict
    """
    if config_path is None:
        # Default path
        base_dir = Path(__file__).parent.parent
        config_path = base_dir / "config" / "gate_prod.yaml"
    
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def get_v3_filter_params(
    instrument: str = 'guitar',
    section: Optional[str] = None,
    config_path: Optional[Path] = None
) -> Dict[str, float]:
    """
    Get V3 filter parameters with instrument/section-specific overrides.
    
    Args:
        instrument: Instrument name (guitar, bass, piano, strings)
        section: Section name (chorus, verse, intro, outro, bridge, pre-chorus)
        config_path: Path to gate_prod.yaml
        
    Returns:
        Dict with keys: enabled, min_proba, min_margin
        
    Example:
        >>> params = get_v3_filter_params('guitar', 'chorus')
        >>> params
        {'enabled': True, 'min_proba': 0.15, 'min_margin': 0.12}
    """
    cfg = load_gate_config(config_path)
    v3_filter = cfg.get('v3_filter', {})
    
    # Default values
    enabled = bool(v3_filter.get('enabled', True))
    min_proba = float(v3_filter.get('min_proba', 0.15))
    min_margin = float(v3_filter.get('min_margin', 0.10))
    
    # Apply instrument-specific overrides
    per_instrument = v3_filter.get('per_instrument', {})
    if instrument in per_instrument:
        inst_cfg = per_instrument[instrument]
        min_proba = float(inst_cfg.get('min_proba', min_proba))
        min_margin = float(inst_cfg.get('min_margin', min_margin))
    
    # Apply section-specific overrides (highest priority)
    if section:
        per_section = v3_filter.get('per_section_override', {})
        section_lower = section.lower()
        if section_lower in per_section:
            sect_cfg = per_section[section_lower]
            min_proba = float(sect_cfg.get('min_proba', min_proba))
            min_margin = float(sect_cfg.get('min_margin', min_margin))
    
    return {
        'enabled': enabled,
        'min_proba': min_proba,
        'min_margin': min_margin
    }


def validate_v3_filter_config(config_path: Optional[Path] = None) -> bool:
    """
    Validate v3_filter section in gate_prod.yaml.
    
    Args:
        config_path: Path to gate_prod.yaml
        
    Returns:
        True if valid, raises ValueError otherwise
    """
    cfg = load_gate_config(config_path)
    v3_filter = cfg.get('v3_filter')
    
    if v3_filter is None:
        raise ValueError("v3_filter section not found in gate_prod.yaml")
    
    # Check required fields
    if 'enabled' not in v3_filter:
        raise ValueError("v3_filter.enabled is required")
    
    if 'min_proba' not in v3_filter:
        raise ValueError("v3_filter.min_proba is required")
    
    if 'min_margin' not in v3_filter:
        raise ValueError("v3_filter.min_margin is required")
    
    # Check value ranges
    min_proba = float(v3_filter['min_proba'])
    min_margin = float(v3_filter['min_margin'])
    
    if not (0.0 <= min_proba <= 1.0):
        raise ValueError(f"min_proba must be in [0, 1], got {min_proba}")
    
    if not (0.0 <= min_margin <= 1.0):
        raise ValueError(f"min_margin must be in [0, 1], got {min_margin}")
    
    # Validate per_section_override
    per_section = v3_filter.get('per_section_override', {})
    for section, sect_cfg in per_section.items():
        if 'min_proba' in sect_cfg:
            p = float(sect_cfg['min_proba'])
            if not (0.0 <= p <= 1.0):
                raise ValueError(
                    f"Section {section}: min_proba must be in [0, 1], got {p}"
                )
        
        if 'min_margin' in sect_cfg:
            m = float(sect_cfg['min_margin'])
            if not (0.0 <= m <= 1.0):
                raise ValueError(
                    f"Section {section}: min_margin must be in [0, 1], got {m}"
                )
    
    # Validate per_instrument
    per_instrument = v3_filter.get('per_instrument', {})
    valid_instruments = {'guitar', 'bass', 'piano', 'strings', 'drums'}
    for instrument, inst_cfg in per_instrument.items():
        if instrument not in valid_instruments:
            raise ValueError(
                f"Unknown instrument: {instrument}. "
                f"Valid: {valid_instruments}"
            )
        
        if 'min_proba' in inst_cfg:
            p = float(inst_cfg['min_proba'])
            if not (0.0 <= p <= 1.0):
                raise ValueError(
                    f"Instrument {instrument}: min_proba must be in [0, 1], got {p}"
                )
        
        if 'min_margin' in inst_cfg:
            m = float(inst_cfg['min_margin'])
            if not (0.0 <= m <= 1.0):
                raise ValueError(
                    f"Instrument {instrument}: min_margin must be in [0, 1], got {m}"
                )
    
    print("✅ V3 filter config validation passed")
    return True


if __name__ == '__main__':
    # Validation test
    validate_v3_filter_config()
    
    # Example usage
    print("\n=== V3 Filter Params Examples ===")
    
    for instrument in ['guitar', 'bass', 'piano', 'strings']:
        for section in ['chorus', 'verse', 'intro']:
            params = get_v3_filter_params(instrument, section)
            print(
                f"{instrument:8s} {section:8s} → "
                f"proba≥{params['min_proba']:.2f}, "
                f"margin≥{params['min_margin']:.2f}"
            )
