#!/usr/bin/env python3
"""
build_forbidden_mask の多拍子(3/4, 6/8)テスト。

前提:
- ml.stage3_infer.build_forbidden_mask が実装済み
- 拍子に応じた BEAT 上限制御と逆行防止を検証
"""
from __future__ import annotations

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_34_time_sig_progression():
    """3/4拍子: BEAT_1~3のみ有効、逆行防止確認"""
    print("=" * 70)
    print("Test 1: 3/4 Time Signature - Beat Progression")
    print("=" * 70)
    
    from ml.stage3_infer import build_forbidden_mask
    
    class MockTokenizer:
        max_bars = 16
        token_to_id = {}
        # BAR tokens
        for i in range(16):
            token_to_id[f"BAR_{i}"] = 100 + i
        # BEAT tokens (1-8 to cover various time sigs)
        for i in range(1, 9):
            token_to_id[f"BEAT_{i}"] = 200 + i
    
    tokenizer = MockTokenizer()
    
    # Case 1: last_beat=1 in 3/4
    print("\nCase 1: last_beat=1 (3/4)")
    forbid = build_forbidden_mask(
        tokenizer=tokenizer,
        current_bar=1,
        max_bars=8,
        last_beat=1,
        time_signature_beats=3,
    )
    
    # BEAT_2, BEAT_3 should be allowed
    # BEAT_4+ should be forbidden (out of range)
    beat_1_id = tokenizer.token_to_id.get("BEAT_1")
    beat_2_id = tokenizer.token_to_id.get("BEAT_2")
    beat_3_id = tokenizer.token_to_id.get("BEAT_3")
    beat_4_id = tokenizer.token_to_id.get("BEAT_4")
    
    assert beat_1_id in forbid, "BEAT_1 should be forbidden (backward jump)"
    assert beat_2_id not in forbid, "BEAT_2 should be allowed"
    assert beat_3_id not in forbid, "BEAT_3 should be allowed"
    assert beat_4_id in forbid, "BEAT_4 should be forbidden (out of 3/4 range)"
    
    print(f"  ✅ BEAT_1 forbidden (backward): {beat_1_id in forbid}")
    print(f"  ✅ BEAT_2 allowed: {beat_2_id not in forbid}")
    print(f"  ✅ BEAT_3 allowed: {beat_3_id not in forbid}")
    print(f"  ✅ BEAT_4 forbidden (range): {beat_4_id in forbid}")
    
    # Case 2: last_beat=3 (final beat of 3/4)
    print("\nCase 2: last_beat=3 (final beat, should force new BAR)")
    forbid = build_forbidden_mask(
        tokenizer=tokenizer,
        current_bar=2,
        max_bars=8,
        last_beat=3,
        time_signature_beats=3,
    )
    
    # All BEAT tokens should be forbidden (must advance to new BAR)
    assert beat_1_id in forbid, "BEAT_1 should be forbidden"
    assert beat_2_id in forbid, "BEAT_2 should be forbidden"
    assert beat_3_id in forbid, "BEAT_3 should be forbidden"
    
    print(f"  ✅ All BEATs forbidden (force new BAR): {all(tokenizer.token_to_id[f'BEAT_{i}'] in forbid for i in [1,2,3])}")


def test_68_time_sig_six_eighths():
    """6/8拍子: BEAT_1~6のみ有効、逆行防止確認"""
    print("\n" + "=" * 70)
    print("Test 2: 6/8 Time Signature - Six Eighths")
    print("=" * 70)
    
    from ml.stage3_infer import build_forbidden_mask
    
    class MockTokenizer:
        max_bars = 16
        token_to_id = {}
        for i in range(16):
            token_to_id[f"BAR_{i}"] = 100 + i
        for i in range(1, 9):
            token_to_id[f"BEAT_{i}"] = 200 + i
    
    tokenizer = MockTokenizer()
    
    # Case: last_beat=4 in 6/8
    print("\nCase: last_beat=4 (6/8)")
    forbid = build_forbidden_mask(
        tokenizer=tokenizer,
        current_bar=0,
        max_bars=16,
        last_beat=4,
        time_signature_beats=6,
    )
    
    beat_ids = {i: tokenizer.token_to_id[f"BEAT_{i}"] for i in range(1, 9)}
    
    # BEAT_1~3 should be forbidden (backward)
    # BEAT_5~6 should be allowed
    # BEAT_7+ should be forbidden (out of range)
    for i in [1, 2, 3]:
        assert beat_ids[i] in forbid, f"BEAT_{i} should be forbidden (backward)"
        print(f"  ✅ BEAT_{i} forbidden (backward)")
    
    for i in [5, 6]:
        assert beat_ids[i] not in forbid, f"BEAT_{i} should be allowed"
        print(f"  ✅ BEAT_{i} allowed")
    
    for i in [7, 8]:
        assert beat_ids[i] in forbid, f"BEAT_{i} should be forbidden (out of range)"
        print(f"  ✅ BEAT_{i} forbidden (range)")


def test_bar_overflow_cap():
    """BAR上限到達時のオーバーフロー防止"""
    print("\n" + "=" * 70)
    print("Test 3: BAR Overflow Prevention")
    print("=" * 70)
    
    from ml.stage3_infer import build_forbidden_mask
    
    class MockTokenizer:
        max_bars = 16
        token_to_id = {}
        for i in range(16):
            token_to_id[f"BAR_{i}"] = 100 + i
        for i in range(1, 5):
            token_to_id[f"BEAT_{i}"] = 200 + i
    
    tokenizer = MockTokenizer()
    
    # current_bar=8, max_bars=8 (at limit)
    print("\nCase: current_bar=8, max_bars=8")
    forbid = build_forbidden_mask(
        tokenizer=tokenizer,
        current_bar=8,
        max_bars=8,
        last_beat=1,
        time_signature_beats=4,
    )
    
    # BAR_8 and higher should be forbidden
    for i in range(8, 16):
        bar_id = tokenizer.token_to_id.get(f"BAR_{i}")
        if bar_id is not None:
            assert bar_id in forbid, f"BAR_{i} should be forbidden (overflow)"
    
    print(f"  ✅ BAR_8~15 all forbidden (overflow prevention)")
    
    # current_bar=7, max_bars=8 (not yet at limit)
    print("\nCase: current_bar=7, max_bars=8")
    forbid = build_forbidden_mask(
        tokenizer=tokenizer,
        current_bar=7,
        max_bars=8,
        last_beat=1,
        time_signature_beats=4,
    )
    
    # BAR_7 should be allowed (still under limit)
    bar_7_id = tokenizer.token_to_id.get("BAR_7")
    assert bar_7_id not in forbid, "BAR_7 should be allowed (under limit)"
    
    print(f"  ✅ BAR_7 allowed (under limit)")


def test_combined_constraints_34():
    """複合制約: 3/4拍子でBAR上限+BEAT境界"""
    print("\n" + "=" * 70)
    print("Test 4: Combined Constraints (3/4 at BAR limit)")
    print("=" * 70)
    
    from ml.stage3_infer import build_forbidden_mask
    
    class MockTokenizer:
        max_bars = 16
        token_to_id = {}
        for i in range(16):
            token_to_id[f"BAR_{i}"] = 100 + i
        for i in range(1, 9):
            token_to_id[f"BEAT_{i}"] = 200 + i
    
    tokenizer = MockTokenizer()
    
    # current_bar=8 (at limit), last_beat=2 (in 3/4)
    print("\nCase: current_bar=8, last_beat=2 (3/4)")
    forbid = build_forbidden_mask(
        tokenizer=tokenizer,
        current_bar=8,
        max_bars=8,
        last_beat=2,
        time_signature_beats=3,
    )
    
    # BAR_8+ forbidden
    bar_forbidden = sum(1 for i in range(8, 16) if tokenizer.token_to_id[f"BAR_{i}"] in forbid)
    assert bar_forbidden == 8, f"Expected 8 BAR tokens forbidden, got {bar_forbidden}"
    
    # BEAT_1~2 forbidden, BEAT_3 allowed
    beat_1_id = tokenizer.token_to_id["BEAT_1"]
    beat_2_id = tokenizer.token_to_id["BEAT_2"]
    beat_3_id = tokenizer.token_to_id["BEAT_3"]
    
    assert beat_1_id in forbid, "BEAT_1 should be forbidden (backward)"
    assert beat_2_id in forbid, "BEAT_2 should be forbidden (backward)"
    assert beat_3_id not in forbid, "BEAT_3 should be allowed"
    
    print(f"  ✅ {bar_forbidden} BAR tokens forbidden (overflow)")
    print(f"  ✅ BEAT_1,2 forbidden (backward), BEAT_3 allowed")


def main():
    """Run all multi-time-signature tests"""
    print("\n" + "=" * 70)
    print("Music Guards: Multi-Time-Signature Test Suite")
    print("=" * 70)
    
    try:
        test_34_time_sig_progression()
        test_68_time_sig_six_eighths()
        test_bar_overflow_cap()
        test_combined_constraints_34()
        
        print("\n" + "=" * 70)
        print("✅ All Multi-Time-Signature Tests Passed!")
        print("=" * 70)
        print("\nValidated:")
        print("  ✅ 3/4 time signature (BEAT_1~3)")
        print("  ✅ 6/8 time signature (BEAT_1~6)")
        print("  ✅ BAR overflow prevention")
        print("  ✅ BEAT backward jump prevention")
        print("  ✅ BEAT boundary enforcement (force new BAR)")
        print("  ✅ Combined constraints (BAR+BEAT)")
        print()
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
