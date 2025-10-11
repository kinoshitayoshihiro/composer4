#!/usr/bin/env python3
"""Manual test for music theory guards (no dependencies)."""


def test_forbidden_mask():
    """Test music theory constraint logic."""
    print("=" * 70)
    print("Music Theory Guards - Forbidden Token Logic")
    print("=" * 70)

    # Mock tokenizer
    class MockTokenizer:
        def __init__(self):
            self.max_bars = 16
            self.token_to_id = {}
            # Register BAR tokens
            for i in range(16):
                self.token_to_id[f"BAR_{i}"] = 100 + i
            # Register BEAT tokens
            for i in range(1, 5):
                self.token_to_id[f"BEAT_{i}"] = 200 + i

    def build_forbidden_mask(tokenizer, current_bar, max_bars, last_beat=0, time_signature_beats=4):
        """Simplified version of the guard function."""
        forbidden_ids = set()

        # Rule 1: BAR overflow prevention
        if current_bar >= max_bars:
            for bar_num in range(max_bars, tokenizer.max_bars):
                bar_tok_id = tokenizer.token_to_id.get(f"BAR_{bar_num}")
                if bar_tok_id is not None:
                    forbidden_ids.add(bar_tok_id)

        # Rule 2: BEAT order enforcement
        if last_beat > 0:
            for beat_num in range(1, last_beat):
                beat_tok_id = tokenizer.token_to_id.get(f"BEAT_{beat_num}")
                if beat_tok_id is not None:
                    forbidden_ids.add(beat_tok_id)

            if last_beat >= time_signature_beats:
                for beat_num in range(1, time_signature_beats + 1):
                    beat_tok_id = tokenizer.token_to_id.get(f"BEAT_{beat_num}")
                    if beat_tok_id is not None:
                        forbidden_ids.add(beat_tok_id)

        return forbidden_ids

    tokenizer = MockTokenizer()

    # Test Case 1: BAR overflow prevention
    print("\n" + "=" * 70)
    print("Test 1: BAR Overflow Prevention")
    print("=" * 70)

    forbidden = build_forbidden_mask(
        tokenizer=tokenizer,
        current_bar=8,
        max_bars=8,
        last_beat=0,
    )

    expected_bars = {100 + i for i in range(8, 16)}  # BAR_8 ~ BAR_15
    actual_bars = {tid for tid in forbidden if tid >= 100 and tid < 200}

    print(f"Current bar: 8, Max bars: 8")
    print(f"Expected forbidden BAR tokens: {len(expected_bars)}")
    print(f"Actual forbidden BAR tokens: {len(actual_bars)}")
    assert actual_bars == expected_bars, "BAR overflow guard failed"
    print("✅ BAR overflow prevention working")

    # Test Case 2: BEAT order enforcement
    print("\n" + "=" * 70)
    print("Test 2: BEAT Order Enforcement")
    print("=" * 70)

    # Case 2a: last_beat=2, should forbid BEAT_1
    forbidden = build_forbidden_mask(
        tokenizer=tokenizer,
        current_bar=2,
        max_bars=8,
        last_beat=2,
        time_signature_beats=4,
    )

    expected_beats = {201}  # BEAT_1
    actual_beats = {tid for tid in forbidden if tid >= 200 and tid < 300}

    print(f"\nCase 2a: last_beat=2")
    print(f"Expected forbidden: BEAT_1 (ID 201)")
    print(f"Actual forbidden: {actual_beats}")
    assert actual_beats == expected_beats, "BEAT order guard failed (case 2a)"
    print("✅ BEAT backward jump prevention working")

    # Case 2b: last_beat=4, should forbid all BEAT tokens
    forbidden = build_forbidden_mask(
        tokenizer=tokenizer,
        current_bar=2,
        max_bars=8,
        last_beat=4,
        time_signature_beats=4,
    )

    expected_beats = {201, 202, 203, 204}  # BEAT_1~4
    actual_beats = {tid for tid in forbidden if tid >= 200 and tid < 300}

    print(f"\nCase 2b: last_beat=4 (max for 4/4)")
    print(f"Expected forbidden: BEAT_1~4 (IDs 201-204)")
    print(f"Actual forbidden: {actual_beats}")
    assert actual_beats == expected_beats, "BEAT max guard failed (case 2b)"
    print("✅ BEAT max enforcement working (must advance to new BAR)")

    # Test Case 3: Combined constraints
    print("\n" + "=" * 70)
    print("Test 3: Combined Constraints")
    print("=" * 70)

    forbidden = build_forbidden_mask(
        tokenizer=tokenizer,
        current_bar=8,
        max_bars=8,
        last_beat=3,
        time_signature_beats=4,
    )

    expected_bars = {100 + i for i in range(8, 16)}
    expected_beats = {201, 202}  # BEAT_1, BEAT_2 (backward from 3)
    expected_all = expected_bars | expected_beats

    print(f"Current bar: 8 (at max), last_beat: 3")
    print(f"Expected forbidden: {len(expected_bars)} BARs + 2 BEATs")
    print(f"Actual forbidden: {len(forbidden)} tokens")
    assert forbidden == expected_all, "Combined constraints failed"
    print("✅ Combined BAR+BEAT constraints working")

    print("\n" + "=" * 70)
    print("Summary: All Music Theory Guards Validated")
    print("=" * 70)
    print("✅ Rule 1: BAR overflow prevention (blocks BAR_N when current_bar >= max_bars)")
    print("✅ Rule 2a: BEAT backward jump prevention (forbids BEAT_1~k-1 when last_beat=k)")
    print("✅ Rule 2b: BEAT max enforcement (forbids all BEATs at time signature boundary)")
    print("✅ Rule 3: Combined constraints work correctly\n")


if __name__ == "__main__":
    test_forbidden_mask()
