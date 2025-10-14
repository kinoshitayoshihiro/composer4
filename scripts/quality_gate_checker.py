#!/usr/bin/env python3
"""
Quality Gate Checker Utility

品質ゲートチェック用のユーティリティ関数。
quality_gates.yaml を読み込んで、メトリクスが基準を満たしているか判定。

Usage:
    from scripts.quality_gate_checker import check_gates, load_quality_gates
    
    gates = load_quality_gates("config/quality_gates.yaml")["piano"]
    metrics = {"chord_tone_rate": 0.72, "velocity_std": 18.0}
    fails = check_gates(metrics, gates)
    
    if fails:
        print("FAIL:", fails)
    else:
        print("PASS")
"""

import yaml
from pathlib import Path
from typing import Dict, List, Any, Optional


def load_quality_gates(yaml_path: str | Path) -> Dict[str, Any]:
    """
    Load quality gates from YAML file.
    
    Args:
        yaml_path: Path to quality_gates.yaml
    
    Returns:
        Dictionary with instrument names as keys
    
    Example:
        gates = load_quality_gates("config/quality_gates.yaml")
        piano_gates = gates["piano"]["defaults"]
    """
    with open(yaml_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    return data


def resolve_gates(
    instrument: str,
    gates_config: Dict[str, Any],
    style: Optional[str] = None,
    tempo: Optional[str] = None,
    section: Optional[str] = None
) -> Dict[str, Dict[str, float]]:
    """
    Resolve quality gates with style/tempo/section overrides.
    
    Args:
        instrument: Instrument name (e.g., "piano", "guitar")
        gates_config: Full quality_gates.yaml content
        style: Style name (e.g., "ballad", "shuffle")
        tempo: Tempo category (e.g., "slow", "fast")
        section: Section name (e.g., "Chorus", "Verse")
    
    Returns:
        Resolved gates dictionary
    
    Priority:
        section_overrides > style_overrides > tempo_overrides > defaults
    """
    inst_config = gates_config.get(instrument, {})
    gates = dict(inst_config.get("defaults", {}))
    
    # Apply tempo overrides
    if tempo and "tempo_overrides" in inst_config:
        tempo_ovr = inst_config["tempo_overrides"].get(tempo, {})
        gates.update(tempo_ovr)
    
    # Apply style overrides
    if style and "style_overrides" in inst_config:
        style_ovr = inst_config["style_overrides"].get(style, {})
        gates.update(style_ovr)
    
    # Apply section overrides (highest priority)
    if section and "section_overrides" in inst_config:
        section_ovr = inst_config["section_overrides"].get(section, {})
        gates.update(section_ovr)
    
    return gates


def check_gates(metrics: Dict[str, Any], gates: Dict[str, Dict[str, float]]) -> List[str]:
    """
    Check metrics against quality gates.
    
    Args:
        metrics: Dictionary of metric values (e.g., {"velocity_std": 18.0})
        gates: Dictionary of gate rules (e.g., {"velocity_std": {"range": [15, 25]}})
    
    Returns:
        List of failure messages. Empty list means PASS.
    
    Gate operators:
        - min: metric >= min
        - max: metric <= max
        - range: [lo, hi]: lo <= metric <= hi
    
    Example:
        >>> metrics = {"velocity_std": 10.0}
        >>> gates = {"velocity_std": {"range": [15, 25]}}
        >>> check_gates(metrics, gates)
        ['velocity_std:out_of_range(10.0∉[15,25])']
    """
    fails = []
    
    for name, rule in gates.items():
        if name not in metrics or metrics[name] is None:
            continue
        
        try:
            x = float(metrics[name])
        except (ValueError, TypeError):
            continue
        
        # Check min threshold
        if "min" in rule:
            min_val = float(rule["min"])
            if x < min_val:
                fails.append(f"{name}:low({x:.2f}<{min_val:.2f})")
        
        # Check max threshold
        if "max" in rule:
            max_val = float(rule["max"])
            if x > max_val:
                fails.append(f"{name}:high({x:.2f}>{max_val:.2f})")
        
        # Check range
        if "range" in rule:
            lo, hi = rule["range"]
            if not (lo <= x <= hi):
                fails.append(f"{name}:out_of_range({x:.2f}∉[{lo:.2f},{hi:.2f}])")
    
    return fails


def check_quality_gate(
    instrument: str,
    metrics: Dict[str, Any],
    gates_yaml: str | Path = "config/quality_gates.yaml",
    style: Optional[str] = None,
    tempo: Optional[str] = None,
    section: Optional[str] = None,
    verbose: bool = False
) -> tuple[bool, List[str]]:
    """
    High-level quality gate check function.
    
    Args:
        instrument: Instrument name
        metrics: Metrics dictionary
        gates_yaml: Path to quality_gates.yaml
        style: Style override
        tempo: Tempo override
        section: Section override
        verbose: Print detailed info
    
    Returns:
        (passed, failures) tuple
    
    Example:
        >>> passed, fails = check_quality_gate(
        ...     "piano",
        ...     {"chord_tone_rate": 0.72, "velocity_std": 18.0},
        ...     style="ballad"
        ... )
        >>> print("PASS" if passed else f"FAIL: {fails}")
    """
    gates_config = load_quality_gates(gates_yaml)
    gates = resolve_gates(instrument, gates_config, style, tempo, section)
    fails = check_gates(metrics, gates)
    
    if verbose:
        print(f"[Quality Gate] {instrument.upper()}")
        if style:
            print(f"  Style: {style}")
        if tempo:
            print(f"  Tempo: {tempo}")
        if section:
            print(f"  Section: {section}")
        print(f"  Metrics: {len(metrics)}")
        print(f"  Gates: {len(gates)}")
        print(f"  Result: {'✅ PASS' if not fails else '❌ FAIL'}")
        if fails:
            for fail in fails:
                print(f"    - {fail}")
    
    return (len(fails) == 0, fails)


def main():
    """Example usage and smoke test."""
    import sys
    
    print("=" * 60)
    print("Quality Gate Checker - Smoke Test")
    print("=" * 60)
    print()
    
    # Test 1: Piano (should PASS)
    print("[Test 1] Piano - Expected PASS")
    metrics_pass = {
        "chord_tone_rate": 0.75,
        "hand_separation": 0.65,
        "velocity_std": 18.0,
        "bar_violation_rate": 0.01,
        "notes_per_bar": 12.0
    }
    passed, fails = check_quality_gate("piano", metrics_pass, verbose=True)
    print()
    
    # Test 2: Piano with violations (should FAIL)
    print("[Test 2] Piano - Expected FAIL")
    metrics_fail = {
        "chord_tone_rate": 0.60,  # Too low (<0.70)
        "velocity_std": 10.0,      # Too low (<15)
        "bar_violation_rate": 0.05 # Too high (>0.02)
    }
    passed, fails = check_quality_gate("piano", metrics_fail, verbose=True)
    print()
    
    # Test 3: Guitar with style override
    print("[Test 3] Guitar - Shuffle style")
    guitar_metrics = {
        "strum_consistency": 0.73,  # OK for shuffle (min: 0.72)
        "bar_violation_rate": 0.02
    }
    passed, fails = check_quality_gate("guitar", guitar_metrics, style="shuffle", verbose=True)
    print()
    
    # Test 4: Load and display all instruments
    print("[Test 4] Available Instruments")
    gates = load_quality_gates("config/quality_gates.yaml")
    instruments = [k for k in gates.keys() if k not in ("schema_version", "metadata")]
    print(f"  Instruments: {', '.join(instruments)}")
    print()
    
    print("=" * 60)
    print("Smoke Test Complete")
    print("=" * 60)


if __name__ == "__main__":
    main()
