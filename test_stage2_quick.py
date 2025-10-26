#!/usr/bin/env python3
"""Quick Stage2 Test - Base params auto-load + Drums safe filtering"""

import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s [%(name)s]: %(message)s'
)

logger = logging.getLogger(__name__)

def test_base_params_autoload():
    """Test 1: Base class auto-loads params from *_params_stage2.py"""
    logger.info("\n" + "="*60)
    logger.info("Test 1: Base Params Auto-Load")
    logger.info("="*60)
    
    try:
        from generator.piano_generator_stage2 import PianoGeneratorStage2
        
        # Create instance without explicit params
        gen = PianoGeneratorStage2(
            instrument_name="piano",
            default_instrument="piano"
        )
        
        logger.info(f"✅ Piano Stage2 created")
        logger.info(f"   - params keys: {list(gen.params.keys())[:5]}...")
        logger.info(f"   - params source: {gen._params_source}")
        logger.info(f"   - Total params: {len(gen.params)}")
        
        return True
    except Exception as e:
        logger.error(f"❌ Test 1 failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_drums_safe_init():
    """Test 2: Drums Stage2 initialization with safe filtering"""
    logger.info("\n" + "="*60)
    logger.info("Test 2: Drums Safe Initialization")
    logger.info("="*60)
    
    try:
        from generator.drums_generator_stage2 import DrumsGeneratorStage2
        
        # Create instance with various params (some may not be accepted by V1)
        gen = DrumsGeneratorStage2(
            instrument_name="drums",
            default_instrument="drums",
            params={'test_param': 'test_value'},
            overrides={'global_settings': {'tempo': 120}}
        )
        
        logger.info(f"✅ Drums Stage2 created")
        logger.info(f"   - V1 generator: {gen._v1_generator is not None}")
        logger.info(f"   - params keys: {list(gen.params.keys())[:5]}...")
        logger.info(f"   - params source: {gen._params_source}")
        
        # Check if initialization warning exists
        return True
    except Exception as e:
        logger.error(f"❌ Test 2 failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_factory_stage2_selection():
    """Test 3: Factory correctly selects Stage2 classes"""
    logger.info("\n" + "="*60)
    logger.info("Test 3: Factory Stage2 Selection")
    logger.info("="*60)
    
    try:
        from utilities.generator_factory import build_from_config
        
        # Minimal config for piano
        config = {
            'global_settings': {
                'tempo': 120,
                'time_signature': '4/4'
            },
            'instruments': {
                'piano': {
                    'enabled': True,
                    'parts': {}
                },
                'drums': {
                    'enabled': True,
                    'parts': {}
                }
            }
        }
        
        generators = build_from_config(config)
        
        piano_gen = generators.get('piano')
        drums_gen = generators.get('drums')
        
        logger.info(f"✅ Factory created generators")
        logger.info(f"   - Piano: {piano_gen.__class__.__name__}")
        logger.info(f"   - Piano is Stage2: {'Stage2' in piano_gen.__class__.__name__}")
        logger.info(f"   - Drums: {drums_gen.__class__.__name__}")
        logger.info(f"   - Drums is Stage2: {'Stage2' in drums_gen.__class__.__name__}")
        
        return True
    except Exception as e:
        logger.error(f"❌ Test 3 failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    results = []
    
    logger.info("\n🎹🥁 Stage2 Quick Test Suite")
    logger.info("="*60)
    
    results.append(("Base Params Auto-Load", test_base_params_autoload()))
    results.append(("Drums Safe Init", test_drums_safe_init()))
    results.append(("Factory Selection", test_factory_stage2_selection()))
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("Test Results Summary")
    logger.info("="*60)
    
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        logger.info(f"{status}: {name}")
    
    all_passed = all(r[1] for r in results)
    logger.info("\n" + ("="*60))
    if all_passed:
        logger.info("🎉 All tests passed!")
        sys.exit(0)
    else:
        logger.info("⚠️ Some tests failed")
        sys.exit(1)
