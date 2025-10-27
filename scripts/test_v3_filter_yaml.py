#!/usr/bin/env python3
"""
Test V3 Filter YAML Integration
Phase 24.3: Verify YAML-based V3 filter params are working

Usage:
    python scripts/test_v3_filter_yaml.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ml.v3_filter_config import get_v3_filter_params, validate_v3_filter_config

def main():
    print("="*60)
    print("V3 Filter YAML Integration Test")
    print("="*60)
    
    # Validate config
    print("\n1. Validating gate_prod.yaml v3_filter section...")
    try:
        validate_v3_filter_config()
        print("   ✅ Validation passed")
    except Exception as e:
        print(f"   ❌ Validation failed: {e}")
        return 1
    
    # Test parameter resolution
    print("\n2. Testing parameter resolution...")
    
    test_cases = [
        ('guitar', 'chorus'),
        ('guitar', 'intro'),
        ('bass', 'chorus'),
        ('bass', 'verse'),
        ('piano', 'chorus'),
        ('strings', 'intro'),
    ]
    
    for instrument, section in test_cases:
        params = get_v3_filter_params(instrument, section)
        print(
            f"   {instrument:8s} {section:8s} → "
            f"enabled={params['enabled']}, "
            f"proba≥{params['min_proba']:.2f}, "
            f"margin≥{params['min_margin']:.2f}"
        )
    
    print("\n3. Testing Bass Generator Stage2 integration...")
    try:
        from generator.bass_generator_stage2 import BassGeneratorStage2
        
        gen = BassGeneratorStage2(use_stage2=True, global_tempo=120.0)
        print(f"   ✅ BassGeneratorStage2 initialized (use_stage2={gen.use_stage2})")
        
        # Check if recommender loaded
        if gen.recommender:
            print(f"   ✅ Recommender loaded: {len(gen.recommender.patterns)} patterns")
        else:
            print("   ⚠️  Recommender not loaded (Stage2 patterns unavailable)")
        
    except Exception as e:
        print(f"   ❌ Bass Generator failed: {e}")
        return 1
    
    print("\n" + "="*60)
    print("✅ All tests passed")
    print("="*60)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
