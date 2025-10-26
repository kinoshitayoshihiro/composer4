#!/usr/bin/env python3
"""
LAMDA統合テスト（Stage2 with NO-OP safety）

**目的**:
- LAMDA オプション無し → 既存 v2.6 動作（100%互換）
- LAMDA オプション有り → chordmap_external/signatures/outliers 追加
"""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.lamda_v2.stage2_extractor import extract_to_json
from scripts.lamda_v2.lamda_sources import LamdaSources
import json

BASE_DIR = Path("/Volumes/SSD-SCTU3A/ラジオ用/music_21/composer2-3")
TEST_MIDI = BASE_DIR / "tmp_tempo.mid"
TEST_OUTPUT = BASE_DIR / "test_output"

def main():
    print("🧪 LAMDA Integration Test")
    print(f"   Test MIDI: {TEST_MIDI}")
    print(f"   Output: {TEST_OUTPUT}")
    print()
    
    # Ensure output directory
    TEST_OUTPUT.mkdir(parents=True, exist_ok=True)
    
    # ========================================
    # Test 1: Baseline (No LAMDA)
    # ========================================
    print("📋 Test 1: Baseline (No LAMDA)")
    baseline_json = TEST_OUTPUT / "lamda_baseline.json"
    
    try:
        extract_to_json(TEST_MIDI, baseline_json)
        print(f"   ✅ Success: {baseline_json}")
        
        # Show keys
        with open(baseline_json) as f:
            data = json.load(f)
        print(f"   📄 Keys: {list(data.keys())[:10]}...")
        print(f"   🏷️  Schema: {data.get('schema_version')}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return 1
    
    print()
    
    # ========================================
    # Test 2: With LAMDA (NO-OP fallback)
    # ========================================
    print("📋 Test 2: With LAMDA (NO-OP fallback)")
    
    # Create dummy LAMDA sources (all None → NO-OP)
    lamda_sources = LamdaSources(
        kilo=None,
        meta_dir=None,
        signatures=None,
        totals=None,
        id_map_csv=None
    )
    
    print(f"   📊 LAMDA Sources: {lamda_sources.summary()}")
    
    lamda_json = TEST_OUTPUT / "lamda_with_fusion.json"
    
    try:
        extract_to_json(TEST_MIDI, lamda_json, lamda_sources)
        print(f"   ✅ Success: {lamda_json}")
        
        # Show keys
        with open(lamda_json) as f:
            data = json.load(f)
        print(f"   📄 Keys: {list(data.keys())[:12]}...")
        print(f"   🏷️  Schema: {data.get('schema_version')}")
        
        # Check new fields
        print("   🆕 New fields:")
        print(f"      chordmap_external: {data.get('chordmap_external') is not None}")
        print(f"      signatures: {data.get('signatures') is not None}")
        print(f"      outliers: {data.get('outliers') is not None}")
        print(f"      lamda_meta_present: {data.get('lamda_meta_present', False)}")
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return 1
    
    print()
    print("✅ LAMDA Integration Test Complete!")
    print(f"   Baseline: {baseline_json}")
    print(f"   With LAMDA: {lamda_json}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
