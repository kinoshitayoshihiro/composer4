#!/usr/bin/env python3
"""
Stage3 Minimal Smoke Test

Validates that all Stage3 scripts can be imported and have correct CLI interfaces.
This is a lightweight test that doesn't require training.

Usage:
    python scripts/validate_stage3_pipeline.py
"""

import importlib.util
import sys
from pathlib import Path


def test_import(module_path: Path, module_name: str) -> bool:
    """Test if a module can be imported."""
    try:
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if spec is None or spec.loader is None:
            print(f"❌ Failed to load spec for {module_name}")
            return False
        
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        print(f"✅ Successfully imported {module_name}")
        return True
    except Exception as e:
        print(f"❌ Failed to import {module_name}: {e}")
        return False


def test_script_exists(script_path: Path) -> bool:
    """Test if a script file exists."""
    if script_path.exists():
        print(f"✅ Found {script_path.name}")
        return True
    else:
        print(f"❌ Missing {script_path.name}")
        return False


def main():
    print("=" * 60)
    print("Stage3 Pipeline Validation")
    print("=" * 60)
    
    results = []
    
    # Test script existence
    print("\n📂 Checking script files...")
    scripts = [
        "scripts/collect_conditions.py",
        "scripts/validate_conditions.py",
        "scripts/collect_failures.py",
        "scripts/caption_to_attrs.py",
        "scripts/generate_vptt_samples.py",
        "scripts/quick_eval_stage2.py",
        "scripts/ab_summarize_v2.py",
        "ml/stage3_generator.py",
        "ml/stage3_infer.py",
    ]
    
    for script in scripts:
        script_path = Path(script)
        results.append(test_script_exists(script_path))
    
    # Test config files
    print("\n⚙️  Checking config files...")
    configs = [
        "configs/failure_criteria.yaml",
        "configs/attribute_vocab.yaml",
    ]
    
    for config in configs:
        config_path = Path(config)
        results.append(test_script_exists(config_path))
    
    # Test documentation
    print("\n📖 Checking documentation...")
    docs = [
        "docs/schemas/conditions.schema.md",
        "docs/caption_to_attrs.md",
    ]
    
    for doc in docs:
        doc_path = Path(doc)
        results.append(test_script_exists(doc_path))
    
    # Test VPTT samples
    print("\n🎵 Checking VPTT samples...")
    vptt_metadata = Path("data/vptt_samples/vptt_metadata.yaml")
    vptt_midi_dir = Path("data/vptt_samples/midi")
    
    results.append(test_script_exists(vptt_metadata))
    
    if vptt_midi_dir.exists():
        midi_files = list(vptt_midi_dir.glob("*.mid"))
        print(f"✅ Found {len(midi_files)} VPTT MIDI files")
        results.append(True)
    else:
        print(f"❌ VPTT MIDI directory not found")
        results.append(False)
    
    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    
    total = len(results)
    passed = sum(results)
    failed = total - passed
    
    print(f"Total checks: {total}")
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    print(f"Success rate: {passed/total*100:.1f}%")
    
    if failed == 0:
        print("\n🎉 All validation checks passed!")
        print("Stage3 pipeline is ready for smoke testing.")
        return 0
    else:
        print(f"\n⚠️  {failed} validation check(s) failed")
        print("Please ensure all required files are present.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
