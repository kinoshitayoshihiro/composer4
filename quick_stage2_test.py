#!/usr/bin/env python3
"""Quick Stage2 Patch Test"""

import sys
sys.path.insert(0, '/Volumes/SSD-SCTU3A/ラジオ用/music_21/composer2-3')

import logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s [%(name)s]: %(message)s')
logger = logging.getLogger(__name__)

print("\n🎹🥁 Stage2 Patch Test\n" + "="*60)

# Test 1: Base params auto-load
print("\nTest 1: Base Auto-Load Params")
print("="*60)
try:
    from generator.piano_generator_stage2 import PianoGeneratorStage2
    gen = PianoGeneratorStage2(instrument_name="piano", default_instrument="piano")
    print(f"✅ Piano Stage2: {len(gen.params)} params loaded")
    print(f"   Source: {gen._params_source}")
    print(f"   Sample keys: {list(gen.params.keys())[:3]}")
except Exception as e:
    print(f"❌ Failed: {e}")
    import traceback
    traceback.print_exc()

# Test 2: Drums safe initialization
print("\nTest 2: Drums Safe Initialization")
print("="*60)
try:
    from generator.drums_generator_stage2 import DrumsGeneratorStage2
    gen = DrumsGeneratorStage2(
        instrument_name="drums",
        default_instrument="drums",
        params={'test': 'value'},
        overrides={'global_settings': {}}
    )
    print(f"✅ Drums Stage2: V1={gen._v1_generator is not None}")
    print(f"   Params: {len(gen.params)}")
    print(f"   Source: {gen._params_source}")
except Exception as e:
    print(f"❌ Failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*60)
print("✅ Tests completed!")
