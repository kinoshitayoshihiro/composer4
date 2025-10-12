#!/usr/bin/env python3
"""Unit tests for multi time-signature music guards."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict

import pytest

# Ensure project root on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ml.stage3_infer import build_forbidden_mask


class MockTokenizer:
    """Minimal tokenizer stub exposing the fields build_forbidden_mask expects."""

    def __init__(self, max_bars: int = 16, max_beats: int = 8) -> None:
        self.max_bars = max_bars
        self.token_to_id: Dict[str, int] = {}
        for i in range(max_bars):
            self.token_to_id[f"BAR_{i}"] = 100 + i
        for i in range(1, max_beats + 1):
            self.token_to_id[f"BEAT_{i}"] = 200 + i


def _forbidden_names(tokenizer: MockTokenizer, mask: set[int]) -> set[str]:
    return {
        name
        for name, idx in tokenizer.token_to_id.items()
        if idx in mask and name.startswith(("BAR_", "BEAT_"))
    }


@pytest.mark.parametrize("max_bars", [2, 4, 8])
def test_timesig_34_order_and_overflow(max_bars: int) -> None:
    tokenizer = MockTokenizer(max_bars=max_bars)

    forbid = build_forbidden_mask(
        tokenizer=tokenizer,
        current_bar=max_bars,
        max_bars=max_bars,
        last_beat=2,
        time_signature_beats=3,
    )

    names = _forbidden_names(tokenizer, forbid)

    assert "BEAT_1" in names and "BEAT_2" in names, "Backward beats must be blocked"
    assert "BEAT_3" not in names, "Forward beat should remain legal"

    assert all(
        f"BAR_{b}" in names for b in range(max_bars, tokenizer.max_bars)
    ), "BAR overflow is not guarded"


def test_timesig_68_sequence_and_bar_end() -> None:
    tokenizer = MockTokenizer(max_bars=8, max_beats=8)

    forbid_mid = build_forbidden_mask(
        tokenizer=tokenizer,
        current_bar=2,
        max_bars=8,
        last_beat=4,
        time_signature_beats=6,
    )

    names_mid = _forbidden_names(tokenizer, forbid_mid)
    assert {"BEAT_1", "BEAT_2", "BEAT_3"}.issubset(names_mid)
    assert {"BEAT_5", "BEAT_6"}.isdisjoint(names_mid)
    assert {"BEAT_7", "BEAT_8"}.issubset(names_mid)

    forbid_end = build_forbidden_mask(
        tokenizer=tokenizer,
        current_bar=2,
        max_bars=8,
        last_beat=6,
        time_signature_beats=6,
    )
    names_end = _forbidden_names(tokenizer, forbid_end)
    assert all(f"BEAT_{i}" in names_end for i in range(1, 7)), "Bar boundary should force new BAR"


def test_time_signature_change_resets_rules() -> None:
    tokenizer = MockTokenizer(max_bars=8, max_beats=6)

    # Previous state in 3/4 reaching bar end
    _ = build_forbidden_mask(
        tokenizer=tokenizer,
        current_bar=3,
        max_bars=8,
        last_beat=3,
        time_signature_beats=3,
    )

    # Switch to 4/4: should allow BEAT_1 and forbid non-existent BEAT_5
    forbid = build_forbidden_mask(
        tokenizer=tokenizer,
        current_bar=3,
        max_bars=8,
        last_beat=0,
        time_signature_beats=4,
    )

    names = _forbidden_names(tokenizer, forbid)
    assert "BEAT_1" not in names, "Beat order did not reset after time signature change"
    assert "BEAT_5" in names, "Non-existent beat should remain forbidden"


def test_time_signature_transition_sequence() -> None:
    """Test consecutive time signature changes: 4/4 → 3/4 → 6/8."""
    tokenizer = MockTokenizer(max_bars=8, max_beats=8)

    # Phase 1: 4/4, last_beat=2
    forbid_44 = build_forbidden_mask(
        tokenizer=tokenizer,
        current_bar=1,
        max_bars=8,
        last_beat=2,
        time_signature_beats=4,
    )
    names_44 = _forbidden_names(tokenizer, forbid_44)
    assert {"BEAT_1", "BEAT_2"}.issubset(names_44), "4/4: backward beats must be blocked"
    assert {"BEAT_3", "BEAT_4"}.isdisjoint(names_44), "4/4: forward beats should be allowed"
    assert {"BEAT_5", "BEAT_6", "BEAT_7", "BEAT_8"}.issubset(
        names_44
    ), "4/4: beats beyond 4 should be blocked"

    # Phase 2: Transition to 3/4, reset with last_beat=0
    forbid_34 = build_forbidden_mask(
        tokenizer=tokenizer,
        current_bar=2,
        max_bars=8,
        last_beat=0,
        time_signature_beats=3,
    )
    names_34 = _forbidden_names(tokenizer, forbid_34)
    assert "BEAT_1" not in names_34, "3/4: BEAT_1 should be allowed after reset"
    assert {"BEAT_4", "BEAT_5", "BEAT_6", "BEAT_7", "BEAT_8"}.issubset(
        names_34
    ), "3/4: beats beyond 3 should be blocked"

    # Phase 3: Continue in 3/4 with last_beat=2
    forbid_34_mid = build_forbidden_mask(
        tokenizer=tokenizer,
        current_bar=2,
        max_bars=8,
        last_beat=2,
        time_signature_beats=3,
    )
    names_34_mid = _forbidden_names(tokenizer, forbid_34_mid)
    assert {"BEAT_1", "BEAT_2"}.issubset(names_34_mid), "3/4: backward beats blocked"
    assert "BEAT_3" not in names_34_mid, "3/4: BEAT_3 should be allowed"

    # Phase 4: Transition to 6/8, reset with last_beat=0
    forbid_68 = build_forbidden_mask(
        tokenizer=tokenizer,
        current_bar=3,
        max_bars=8,
        last_beat=0,
        time_signature_beats=6,
    )
    names_68 = _forbidden_names(tokenizer, forbid_68)
    assert "BEAT_1" not in names_68, "6/8: BEAT_1 should be allowed after reset"
    assert {"BEAT_7", "BEAT_8"}.issubset(names_68), "6/8: beats beyond 6 should be blocked"

    # Phase 5: Continue in 6/8 with last_beat=5
    forbid_68_mid = build_forbidden_mask(
        tokenizer=tokenizer,
        current_bar=3,
        max_bars=8,
        last_beat=5,
        time_signature_beats=6,
    )
    names_68_mid = _forbidden_names(tokenizer, forbid_68_mid)
    assert {"BEAT_1", "BEAT_2", "BEAT_3", "BEAT_4", "BEAT_5"}.issubset(
        names_68_mid
    ), "6/8: backward beats blocked"
    assert "BEAT_6" not in names_68_mid, "6/8: BEAT_6 should be allowed"
    assert {"BEAT_7", "BEAT_8"}.issubset(names_68_mid), "6/8: out-of-range beats blocked"
