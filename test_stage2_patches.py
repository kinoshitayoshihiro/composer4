#!/usr/bin/env python3
"""
Stage2 Patches 動作確認スクリプト

監査パッチの動作確認:
1. YAMLローダ両対応（presets: 有無）
2. density表記ゆれ正規化
"""

import sys
from pathlib import Path
import yaml

# パス設定
BASE_DIR = Path(__file__).parent


def load_yaml_presets_test(yaml_path: Path):
    """YAMLローダ（テスト用・独立実装）"""
    if not yaml_path.exists():
        return {}
    
    with open(yaml_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f) or {}
    
    # presets: キーがあればそれを使い、なければルート直下を返す（両対応）
    return data.get("presets", data)


def normalize_density_test(density_cfg):
    """Density正規化（テスト用・独立実装）"""
    if not density_cfg:
        return {}
    
    _DENSITY_ALIASES = {
        "strums_per_bar_range": ("notes_per_bar", "range"),
        "notes_per_bar_range":  ("notes_per_bar", "range"),
        "chords_per_bar":       ("events_per_bar", "obj"),
    }
    
    out = dict(density_cfg)
    
    for alias_key, (canonical_key, kind) in _DENSITY_ALIASES.items():
        if alias_key not in out:
            continue
        
        val = out[alias_key]
        
        if kind == "range" and isinstance(val, (list, tuple)) and len(val) == 2:
            out[canonical_key] = {"min": val[0], "max": val[1]}
            del out[alias_key]
        
        elif kind == "obj" and isinstance(val, dict):
            out[canonical_key] = {"min": val.get("min"), "max": val.get("max")}
            del out[alias_key]
    
    return out


def test_yaml_loader():
    """YAMLローダ両対応テスト"""
    print("\n" + "="*60)
    print("TEST 1: YAMLローダ両対応（presets: 有無）")
    print("="*60)
    
    # Bass/Piano: presets: あり
    bass_path = BASE_DIR / "data/presets/bass_style_presets.yaml"
    piano_path = BASE_DIR / "data/presets/piano_style_presets.yaml"
    
    # Guitar/Strings: 直置き
    guitar_path = BASE_DIR / "data/presets/guitar_style_presets.yaml"
    strings_path = BASE_DIR / "data/presets/strings_style_presets.yaml"
    
    for name, path in [
        ("Bass", bass_path),
        ("Piano", piano_path),
        ("Guitar", guitar_path),
        ("Strings", strings_path)
    ]:
        if not path.exists():
            print(f"⚠️  {name}: {path} NOT FOUND")
            continue
        
        presets = load_yaml_presets_test(path)
        style_names = list(presets.keys())
        print(f"✅ {name:8s}: {len(style_names)} presets - {', '.join(style_names[:2])}...")


def test_density_normalization():
    """Density表記ゆれ正規化テスト"""
    print("\n" + "="*60)
    print("TEST 2: Density表記ゆれ正規化")
    print("="*60)
    
    test_cases = [
        # Case 1: strums_per_bar_range (Guitar形式)
        {
            "input": {"strums_per_bar_range": [4, 8], "arpeggio_ratio": 0.1},
            "expected_key": "notes_per_bar",
            "name": "Guitar (strums_per_bar_range)"
        },
        # Case 2: notes_per_bar_range (Strings形式)
        {
            "input": {"notes_per_bar_range": [2, 6], "mode": "pad"},
            "expected_key": "notes_per_bar",
            "name": "Strings (notes_per_bar_range)"
        },
        # Case 3: chords_per_bar (Piano形式)
        {
            "input": {"chords_per_bar": {"min": 3, "max": 6}},
            "expected_key": "events_per_bar",
            "name": "Piano (chords_per_bar)"
        },
        # Case 4: notes_per_bar (Bass既存形式)
        {
            "input": {"notes_per_bar": {"min": 2, "max": 8}},
            "expected_key": "notes_per_bar",
            "name": "Bass (notes_per_bar)"
        },
    ]
    
    for case in test_cases:
        result = normalize_density_test(case["input"])
        
        if case["expected_key"] in result:
            val = result[case["expected_key"]]
            if isinstance(val, dict) and "min" in val and "max" in val:
                print(f"✅ {case['name']:30s}: {val}")
            else:
                print(f"⚠️  {case['name']:30s}: 形式不正 {val}")
        else:
            print(f"❌ {case['name']:30s}: キー不在 (got {list(result.keys())})")


def test_no_op_default():
    """NO-OP既定テスト"""
    print("\n" + "="*60)
    print("TEST 3: NO-OP既定（空設定 = 何もしない）")
    print("="*60)
    
    empty_cases = [
        {},
        None,
        {"other_key": "value"},
    ]
    
    for i, case in enumerate(empty_cases, 1):
        result = normalize_density_test(case)
        
        if not result or ("notes_per_bar" not in result and "events_per_bar" not in result):
            print(f"✅ Case {i}: NO-OP確認 (input={case})")
        else:
            print(f"⚠️  Case {i}: 予期しない変換 (input={case}, output={result})")


if __name__ == "__main__":
    print("\n🔧 Stage2 Patches 動作確認")
    print("監査パッチ① YAMLローダ両対応")
    print("監査パッチ② density表記ゆれ正規化")
    
    try:
        test_yaml_loader()
        test_density_normalization()
        test_no_op_default()
        
        print("\n" + "="*60)
        print("✅ 全テスト完了")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ エラー発生: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
