#!/usr/bin/env python3
"""
VioPTT基盤テストスクリプト
technique_map.yaml、labels_schema.yaml、DAWDreamerの統合動作確認

テスト項目:
1. technique_map.yaml読み込み
2. labels_schema.yaml読み込み
3. DAWDreamer基本動作
4. 技法マッピング検証
5. Stage2特徴量との整合性
"""

import sys
from pathlib import Path
import yaml

# プロジェクトルートをパスに追加
BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR))


def test_technique_map():
    """technique_map.yaml読み込みテスト"""
    print("=" * 60)
    print("Test 1: technique_map.yaml")
    print("=" * 60)
    
    yaml_path = BASE_DIR / "configs" / "labels" / "technique_map.yaml"
    
    if not yaml_path.exists():
        print(f"❌ FAIL: {yaml_path} not found")
        return False
    
    try:
        with open(yaml_path) as f:
            config = yaml.safe_load(f)
        
        print(f"✅ Loaded: {yaml_path}")
        print(f"   Top-level keys: {list(config.keys())}")
        
        # DAWDreamer設定確認
        if 'dawdreamer' in config:
            dd_config = config['dawdreamer']
            print(f"\n📌 DAWDreamer Config:")
            print(f"   Enabled: {dd_config.get('enabled', False)}")
            print(f"   Sample Rate: {dd_config.get('sample_rate', 'N/A')}")
            
            if 'vst_paths' in dd_config:
                print(f"   VST Paths:")
                for name, path in dd_config['vst_paths'].items():
                    print(f"     - {name}: {path}")
            
            if 'technique_presets' in dd_config:
                print(f"   Technique Presets: {len(dd_config['technique_presets'])} techniques")
                for tech, params in list(dd_config['technique_presets'].items())[:3]:
                    print(f"     - {tech}: {params}")
        else:
            print("⚠️  WARNING: No 'dawdreamer' section found")
        
        # Stage2特徴量設定確認
        if 'stage2_features' in config:
            s2_config = config['stage2_features']
            print(f"\n📌 Stage2 Features:")
            print(f"   Keys: {list(s2_config.keys())}")
            
            if 'articulation_hints' in s2_config:
                hints = s2_config['articulation_hints']
                print(f"   Articulation Hints: {list(hints.keys())}")
        else:
            print("⚠️  WARNING: No 'stage2_features' section found")
        
        return True
    
    except Exception as e:
        print(f"❌ FAIL: {e}")
        return False


def test_labels_schema():
    """labels_schema.yaml読み込みテスト"""
    print("\n" + "=" * 60)
    print("Test 2: labels_schema.yaml")
    print("=" * 60)
    
    yaml_path = BASE_DIR / "configs" / "labels" / "labels_schema.yaml"
    
    if not yaml_path.exists():
        print(f"❌ FAIL: {yaml_path} not found")
        return False
    
    try:
        with open(yaml_path) as f:
            schema = yaml.safe_load(f)
        
        print(f"✅ Loaded: {yaml_path}")
        
        # 技法セクション確認（schemaネスト対応）
        tech_section = None
        if 'schema' in schema and 'technique' in schema['schema']:
            tech_section = schema['schema']['technique']
        elif 'technique' in schema:
            tech_section = schema['technique']
        
        if tech_section:
            print(f"\n📌 Technique Section:")
            
            instruments = [k for k in tech_section.keys() 
                          if k not in ['confidence_thresholds', 'articulation_features'] 
                          and isinstance(tech_section[k], (list, dict))]
            print(f"   Instruments: {instruments}")
            
            # VioPTT 4技法確認
            if 'strings' in tech_section:
                strings_tech = tech_section['strings']
                if isinstance(strings_tech, dict) and 'vioptt_core' in strings_tech:
                    vioptt_core = strings_tech['vioptt_core']
                    print(f"\n   VioPTT Core (4 techniques): {vioptt_core}")
                    
                    # 4技法確認
                    expected = {'detache', 'spiccato', 'pizzicato', 'flageolet'}
                    actual = set(vioptt_core) if isinstance(vioptt_core, list) else set()
                    
                    if expected == actual:
                        print(f"   ✅ VioPTT 4技法完全一致")
                    else:
                        missing = expected - actual
                        extra = actual - expected
                        if missing:
                            print(f"   ⚠️  Missing: {missing}")
                        if extra:
                            print(f"   ⚠️  Extra: {extra}")
                else:
                    print("   ⚠️  WARNING: 'vioptt_core' not found in strings section")
            
            # 全楽器の技法数カウント
            print(f"\n   Techniques by instrument:")
            for inst in instruments:
                if isinstance(tech_section[inst], list):
                    count = len(tech_section[inst])
                    print(f"     - {inst}: {count} techniques")
                elif isinstance(tech_section[inst], dict):
                    total = sum(len(v) for v in tech_section[inst].values() if isinstance(v, list))
                    print(f"     - {inst}: {total} techniques (nested)")
        else:
            print("⚠️  WARNING: No 'technique' section found")
        
        return True
    
    except Exception as e:
        print(f"❌ FAIL: {e}")
        return False


def test_dawdreamer():
    """DAWDreamer基本動作テスト"""
    print("\n" + "=" * 60)
    print("Test 3: DAWDreamer")
    print("=" * 60)
    
    try:
        import dawdreamer as daw
        print("✅ DAWDreamer imported successfully")
        
        # 基本クラス確認
        available_processors = [
            'PluginProcessor',
            'PlaybackProcessor',
            'RenderEngine'
        ]
        
        print("\n📌 Available Processors:")
        for proc in available_processors:
            if hasattr(daw, proc):
                print(f"   ✅ {proc}")
            else:
                print(f"   ❌ {proc} (not found)")
        
        # RenderEngine作成テスト
        try:
            engine = daw.RenderEngine(sample_rate=44100, block_size=512)
            print(f"\n✅ RenderEngine created (SR: 44100, Block: 512)")
            return True
        except Exception as e:
            print(f"❌ RenderEngine creation failed: {e}")
            return False
    
    except ImportError as e:
        print(f"❌ FAIL: DAWDreamer import failed ({e})")
        return False


def test_technique_mapping():
    """技法マッピング整合性テスト"""
    print("\n" + "=" * 60)
    print("Test 4: Technique Mapping Consistency")
    print("=" * 60)
    
    tech_map_path = BASE_DIR / "configs" / "labels" / "technique_map.yaml"
    labels_path = BASE_DIR / "configs" / "labels" / "labels_schema.yaml"
    
    try:
        with open(tech_map_path) as f:
            tech_map = yaml.safe_load(f)
        
        with open(labels_path) as f:
            labels = yaml.safe_load(f)
        
        # schemaネスト対応
        tech_section = None
        if 'schema' in labels and 'technique' in labels['schema']:
            tech_section = labels['schema']['technique']
        elif 'technique' in labels:
            tech_section = labels['technique']
        
        # DAWDreamer技法とlabels_schema技法の整合性確認
        if 'dawdreamer' in tech_map and 'technique_presets' in tech_map['dawdreamer']:
            dd_techniques = set(tech_map['dawdreamer']['technique_presets'].keys())
            print(f"📌 DAWDreamer techniques: {dd_techniques}")
        else:
            dd_techniques = set()
            print("⚠️  No DAWDreamer techniques found")
        
        if tech_section and 'strings' in tech_section:
            strings_tech = tech_section['strings']
            if isinstance(strings_tech, dict) and 'vioptt_core' in strings_tech:
                vioptt_techniques = set(strings_tech['vioptt_core'])
                print(f"📌 VioPTT techniques: {vioptt_techniques}")
            else:
                vioptt_techniques = set()
        else:
            vioptt_techniques = set()
        
        if dd_techniques and vioptt_techniques:
            intersection = dd_techniques & vioptt_techniques
            dd_only = dd_techniques - vioptt_techniques
            vioptt_only = vioptt_techniques - dd_techniques
            
            print(f"\n📊 Mapping Consistency:")
            print(f"   Common: {intersection}")
            print(f"   DAWDreamer only: {dd_only}")
            print(f"   VioPTT only: {vioptt_only}")
            
            if dd_techniques == vioptt_techniques:
                print(f"   ✅ Perfect match!")
                return True
            elif len(intersection) >= 3:
                print(f"   ⚠️  Partial match ({len(intersection)}/4 techniques)")
                return True
            else:
                print(f"   ❌ Poor match")
                return False
        else:
            print("⚠️  Cannot compare (missing data)")
            return False
    
    except Exception as e:
        print(f"❌ FAIL: {e}")
        return False


def test_stage2_integration():
    """Stage2特徴量との統合テスト"""
    print("\n" + "=" * 60)
    print("Test 5: Stage2 Feature Integration")
    print("=" * 60)
    
    tech_map_path = BASE_DIR / "configs" / "labels" / "technique_map.yaml"
    
    try:
        with open(tech_map_path) as f:
            tech_map = yaml.safe_load(f)
        
        if 'stage2_features' not in tech_map:
            print("⚠️  No 'stage2_features' section found")
            return False
        
        s2_features = tech_map['stage2_features']
        
        # articulation_hints検証
        if 'articulation_hints' in s2_features:
            hints = s2_features['articulation_hints']
            print(f"📌 Articulation Hints:")
            
            required_hints = ['staccato_ratio', 'legato_ratio', 'pizzicato_score']
            found_hints = [h for h in required_hints if h in hints]
            
            print(f"   Required ({len(found_hints)}/{len(required_hints)}): {found_hints}")
            
            # 各hintのパラメータ確認
            for hint in found_hints:
                params = hints[hint]
                print(f"   - {hint}: {list(params.keys())}")
            
            if len(found_hints) == len(required_hints):
                print(f"   ✅ All required hints present")
                return True
            else:
                missing = set(required_hints) - set(found_hints)
                print(f"   ⚠️  Missing hints: {missing}")
                return False
        else:
            print("❌ 'articulation_hints' not found")
            return False
    
    except Exception as e:
        print(f"❌ FAIL: {e}")
        return False


def main():
    """全テスト実行"""
    print("\n" + "🎻" * 30)
    print("VioPTT Foundation Test Suite")
    print("🎻" * 30 + "\n")
    
    results = {
        "technique_map.yaml": test_technique_map(),
        "labels_schema.yaml": test_labels_schema(),
        "DAWDreamer": test_dawdreamer(),
        "Technique Mapping": test_technique_mapping(),
        "Stage2 Integration": test_stage2_integration()
    }
    
    # 結果サマリー
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status}  {test_name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! VioPTT foundation is ready.")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please review the issues above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
