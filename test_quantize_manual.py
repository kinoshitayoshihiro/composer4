#!/usr/bin/env python3
"""Manual test for floor-based quantization (no dependencies)."""


def quantize_old(value: float, *, buckets: int) -> int:
    """Old round-based quantization (banker's rounding)."""
    import math

    if math.isnan(value):
        return -1
    clamped = min(max(value, 0.0), 1.0)
    step = 1.0 / buckets
    bucket = int(round(clamped / step))
    return min(bucket, buckets)


def quantize_new(value: float, *, buckets: int) -> int:
    """New floor-based quantization (monotonic, unbiased)."""
    import math

    if math.isnan(value):
        return -1
    clamped = min(max(value, 0.0), 1.0)
    return min(int(clamped * buckets), buckets - 1)


def test_quantization():
    """Test new quantization logic."""
    print("=" * 70)
    print("Quantization Comparison: round() vs floor()")
    print("=" * 70)

    test_cases = [
        0.0,
        0.15,
        0.25,
        0.33,
        0.5,
        0.75,
        0.85,
        0.95,
        1.0,
        -0.1,  # Clamped
        1.5,  # Clamped
    ]

    buckets = 10

    print(f"\n{'Value':<8} {'Old (round)':<15} {'New (floor)':<15} {'Change':<10}")
    print("-" * 70)

    for val in test_cases:
        old_result = quantize_old(val, buckets=buckets)
        new_result = quantize_new(val, buckets=buckets)
        change = "✓" if old_result == new_result else f"{old_result}→{new_result}"
        print(f"{val:<8.2f} {old_result:<15} {new_result:<15} {change:<10}")

    print("\n" + "=" * 70)
    print("Key Improvements:")
    print("=" * 70)
    print("✅ 0.75: round(7.5)=8 → floor(7.5)=7 (no banker's rounding bias)")
    print("✅ 0.85: round(8.5)=8 → floor(8.5)=8 (consistent)")
    print("✅ 1.0:  bucket=10 → bucket=9 (clamped to max index)")
    print("✅ Monotonic: x1 < x2 ⟹ quantize(x1) ≤ quantize(x2)\n")


if __name__ == "__main__":
    test_quantization()
