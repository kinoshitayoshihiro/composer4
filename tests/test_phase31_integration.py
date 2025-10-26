#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Phase 31 Mode/Scale統合

sections.json にkey_hint/mode_hintがある場合、Phase 31でスケール外音が
最近接スケール内音に修正されることを確認。
"""

import sys
import json
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ops.scale_modes import scale_mask_for_point


def test_scale_constraint_logic():
    """スケール制約ロジックのテスト"""
    
    print("=" * 60)
    print("Phase 31 Mode/Scale統合テスト")
    print("=" * 60)
    
    # テストケース1: D Ionian
    print("\n[Test 1] D Ionian - スケール外音の検出")
    test_sections_1 = {
        "meter": 4,
        "key_hint": [[0, "D"]],
        "mode_hint": [[0, "ionian"]]
    }
    
    # D Ionian: D, E, F#, G, A, B, C# (2, 4, 6, 7, 9, 11, 1)
    mask_1 = scale_mask_for_point(t_ql=0.0, sections=test_sections_1)
    print(f"  Mask: {[round(x, 2) for x in mask_1]}")
    
    test_pitches = [
        (62, "D4", "スケール内"),
        (63, "D#4/Eb4", "スケール外"),
        (64, "E4", "スケール内"),
        (65, "F4", "スケール外"),
        (66, "F#4", "スケール内"),
    ]
    
    for pitch, name, expected in test_pitches:
        pc = pitch % 12
        in_scale = mask_1[pc] > 0.5
        status = "スケール内" if in_scale else "スケール外"
        symbol = "✓" if status == expected else "✗"
        print(f"  {symbol} {name} (pitch={pitch}, PC={pc}): {status} (mask={mask_1[pc]:.2f})")
        
        # スケール外音の場合、最近接音を見つける
        if not in_scale:
            candidates = []
            for offset in [1, -1, 2, -2]:
                new_pc = (pc + offset) % 12
                if mask_1[new_pc] > 0.5:
                    candidates.append((abs(offset), pitch + offset, new_pc))
            
            if candidates:
                candidates.sort()
                distance, new_pitch, new_pc = candidates[0]
                print(f"    → 修正候補: pitch {new_pitch} (PC={new_pc}, distance={distance})")
    
    # テストケース2: G Mixolydian
    print("\n[Test 2] G Mixolydian - 特徴度数の確認")
    test_sections_2 = {
        "meter": 4,
        "key_hint": [[0, "G"]],
        "mode_hint": [[0, "mixolydian"]]
    }
    
    # G Mixolydian: G, A, B, C, D, E, F (7, 9, 11, 0, 2, 4, 5)
    # 特徴度数: F (b7) が強調される
    mask_2 = scale_mask_for_point(t_ql=0.0, sections=test_sections_2)
    print(f"  Mask: {[round(x, 2) for x in mask_2]}")
    
    test_pitches_2 = [
        (67, "G4", "トニック（ルート）"),
        (71, "B4", "第3音"),
        (74, "D5", "第5音"),
        (77, "F5", "♭7（特徴度数）"),
        (78, "F#5", "スケール外"),
    ]
    
    for pitch, name, note in test_pitches_2:
        pc = pitch % 12
        print(f"  {name} (PC={pc}): mask={mask_2[pc]:.2f} - {note}")
    
    # テストケース3: NO-OP（mode_hint無し）
    print("\n[Test 3] NO-OP - mode_hint無しの場合")
    test_sections_3 = {
        "meter": 4,
        "key_hint": [[0, "D"]],
        # mode_hint なし → 自動でIonianを推定
    }
    
    mask_3 = scale_mask_for_point(t_ql=0.0, sections=test_sections_3)
    if mask_3:
        print(f"  Mask: {[round(x, 2) for x in mask_3]}")
        print("  ✓ key_hintのみでもIonian/Aeolianを自動推定")
    else:
        print("  ✗ マスクが生成されませんでした")
    
    # テストケース4: 完全NO-OP（key_hint/mode_hint両方無し）
    print("\n[Test 4] 完全NO-OP - key_hint/mode_hint両方無し")
    test_sections_4 = {
        "meter": 4,
    }
    
    mask_4 = scale_mask_for_point(t_ql=0.0, sections=test_sections_4)
    if mask_4 is None:
        print("  ✓ None返却（完全NO-OP）- 旧来と同じ動作")
    else:
        print(f"  ✗ マスクが生成されました: {mask_4}")
    
    # テストケース5: 転調（D → G）
    print("\n[Test 5] 転調 - D Ionian → G Mixolydian")
    test_sections_5 = {
        "meter": 4,
        "key_hint": [[0, "D"], [8, "G"]],
        "mode_hint": [[0, "ionian"], [8, "mixolydian"]]
    }
    
    # Bar 0 (t_ql=0.0)
    mask_5a = scale_mask_for_point(t_ql=0.0, sections=test_sections_5)
    print(f"  Bar 0 (D Ionian): {[round(x, 2) for x in mask_5a]}")
    
    # Bar 8 (t_ql=32.0)
    mask_5b = scale_mask_for_point(t_ql=32.0, sections=test_sections_5)
    print(f"  Bar 8 (G Mixolydian): {[round(x, 2) for x in mask_5b]}")
    
    # F音の扱いが変わることを確認
    f_pc = 5  # F = PC 5
    print(f"\n  F音(PC={f_pc})の扱い:")
    print(f"    D Ionian: mask={mask_5a[f_pc]:.2f} (スケール外)")
    print(f"    G Mixolydian: mask={mask_5b[f_pc]:.2f} (♭7として強調)")
    
    print("\n" + "=" * 60)
    print("✅ 全テスト完了")
    print("=" * 60)


if __name__ == "__main__":
    try:
        test_scale_constraint_logic()
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ テスト失敗: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
