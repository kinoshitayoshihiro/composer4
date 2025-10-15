"""
Emotion Parameter Fallback Robustness Tests (Phase 5 Brush-up #6)

Purpose:
- Verify that unknown emotion profiles fall back to neutral_medium
- Ensure that missing part_params still produces required emotion adjustments
- Prevent regression of Issue #2 (empty _emotion_adjustments)

Test Coverage:
1. Unknown emotion → neutral_medium equivalence (keys + values)
2. Missing part_params → REQUIRED keys always populated
"""

import numpy as np
import pytest
from music21 import instrument

# Generators (adjust paths as needed)
from generator.guitar_generator import GuitarGenerator
from generator.bass_generator import BassGenerator
from generator.strings_generator import StringsGenerator
from generator.drum_generator import DrumGenerator

# Minimum required keys per instrument (from Phase 5 architecture)
REQUIRED = {
    "guitar": {"velocity_boost", "strum_consistency_target"},
    "bass": {"velocity_boost", "sustain_control", "velocity_std_multiplier"},
    "strings": {"velocity_boost", "bow_pressure_factor", "articulation_legato_bias", "velocity_std_multiplier"},
    "drums": {"velocity_boost", "attack_sharpness", "groove_tightness", "velocity_std_multiplier"},
}

# Generator factory (common tempo/time_signature with required parameters)
GENS = {
    "guitar": lambda: GuitarGenerator(
        part_name="guitar",
        default_instrument=instrument.AcousticGuitar(),
        global_tempo=120,
        global_time_signature="4/4",
        global_key_signature_tonic="C",
        global_key_signature_mode="major"
    ),
    "bass": lambda: BassGenerator(
        part_name="bass",
        default_instrument=instrument.ElectricBass(),
        global_tempo=120,
        global_time_signature="4/4",
        global_key_signature_tonic="C",
        global_key_signature_mode="major"
    ),
    "strings": lambda: StringsGenerator(
        part_name="strings",
        default_instrument=instrument.StringInstrument(),
        global_tempo=120,
        global_time_signature="4/4",
        global_key_signature_tonic="C",
        global_key_signature_mode="major"
    ),
    "drums": lambda: DrumGenerator(
        main_cfg={
            "global_settings": {
                "tempo_bpm": 120,
                "time_signature": "4/4",
            }
        },
        default_instrument=instrument.Percussion(),
        global_tempo=120,
        global_time_signature="4/4"
    ),
}


def _base_section():
    """Minimal section data for compose() calls."""
    return {
        "chord_symbol_for_voicing": "C",
        "q_length": 8.0,
        "section_name": "Verse",
        "label": "Verse",
    }


# 6-1) Unknown emotion → neutral_medium equivalence
@pytest.mark.parametrize("name", list(GENS.keys()))
def test_emotion_unknown_falls_back_to_neutral(name):
    """
    Verify that unknown emotion profiles produce identical adjustments
    to neutral_medium (preventing silent failures or empty dicts).
    """
    gen = GENS[name]()

    # Unknown emotion
    data_u = _base_section()
    _ = gen.compose(section_data=data_u, section="Verse", emotion_profile="__unknown_profile__")
    adj_u = data_u.get("_emotion_adjustments", {}).get(name, {})

    # Neutral emotion
    data_n = _base_section()
    _ = gen.compose(section_data=data_n, section="Verse", emotion_profile="neutral_medium")
    adj_n = data_n.get("_emotion_adjustments", {}).get(name, {})

    # Adjustments must not be empty
    assert adj_u, f"{name}: adjustments empty on unknown emotion"
    assert adj_n, f"{name}: adjustments empty on neutral"

    # Key sets must match (same fallback parameter set)
    assert set(adj_u.keys()) == set(adj_n.keys()), (
        f"{name}: key mismatch - "
        f"unknown only: {set(adj_u) - set(adj_n)}, "
        f"neutral only: {set(adj_n) - set(adj_u)}"
    )

    # Values must be equivalent (allowing small float tolerance)
    for k in adj_n:
        v_u, v_n = adj_u[k], adj_n[k]
        if isinstance(v_n, (int, float)) and isinstance(v_u, (int, float)):
            assert np.isclose(v_u, v_n, rtol=0.0, atol=1e-6), (
                f"{name}.{k}: unknown={v_u} != neutral={v_n}"
            )
        else:
            assert v_u == v_n, f"{name}.{k}: unknown={v_u} != neutral={v_n}"


# 6-2) Missing part_params → REQUIRED keys still populated
@pytest.mark.parametrize("name", list(GENS.keys()))
def test_missing_part_params_still_produces_required_adjustments(name):
    """
    Verify that even when part_params is empty/missing, the triple-fallback
    pattern ensures all REQUIRED emotion adjustment keys are populated.
    (Regression test for Issue #2: empty _emotion_adjustments)
    """
    gen = GENS[name]()
    data = _base_section()
    data["part_params"] = {}  # Intentionally empty

    _ = gen.compose(section_data=data, section="Verse", emotion_profile="happy_high")

    adj = data.get("_emotion_adjustments", {}).get(name, {})
    assert adj, f"{name}: adjustments empty when part_params is missing"

    missing = REQUIRED[name] - set(adj.keys())
    assert not missing, f"{name}: missing fallback keys {missing}"


class TestEmotionFallbackRobustness:
    """
    Comprehensive test suite for emotion parameter fallback mechanisms.
    Ensures system resilience against incomplete inputs and unknown profiles.
    """

    def test_all_instruments_handle_unknown_emotion(self):
        """
        Integration test: all 4 instruments simultaneously process
        an unknown emotion profile without errors.
        """
        data = _base_section()
        for name, gen_fn in GENS.items():
            gen = gen_fn()
            try:
                result = gen.compose(
                    section_data=data,
                    section="Verse",
                    emotion_profile="definitely_not_a_real_emotion"
                )
                # Should produce valid result (Part or dict of Parts for Strings)
                assert result is not None
                # Strings returns dict, others return Part
                if isinstance(result, dict):
                    assert len(result) > 0, f"{name}: empty result dict"
                    for part in result.values():
                        assert hasattr(part, "flatten"), f"{name}: invalid Part in dict"
                else:
                    assert hasattr(result, "flatten"), f"{name}: invalid Part"
            except Exception as e:
                pytest.fail(f"{name} failed on unknown emotion: {e}")

    def test_partial_part_params_coverage(self):
        """
        Test that instruments can handle partially populated part_params
        (some keys present, some missing) and fill gaps via fallback.
        
        Note: Current implementation uses loader-based emotion parameters,
        not direct part_params overrides. This test verifies that even with
        custom part_params, emotion fallback still works.
        """
        for name, gen_fn in GENS.items():
            gen = gen_fn()
            data = _base_section()

            # Partial params (custom structure that doesn't override emotion params)
            data["part_params"] = {
                name: {
                    "some_custom_param": {"foo": "bar"}
                    # Not emotion_params - testing robustness
                }
            }

            _ = gen.compose(section_data=data, section="Verse", emotion_profile="happy_high")
            adj = data.get("_emotion_adjustments", {}).get(name, {})

            # All REQUIRED keys should still be populated by fallback
            # (since part_params doesn't contain emotion_params override)
            missing = REQUIRED[name] - set(adj.keys())
            assert not missing, f"{name}: missing keys even after fallback: {missing}"
            
            # velocity_boost should use default happy_high value (10)
            assert "velocity_boost" in adj
            assert adj["velocity_boost"] == 10, f"{name}: expected default happy_high velocity_boost=10"

    def test_emotion_profile_case_sensitivity(self):
        """
        Verify that emotion profile names are handled case-insensitively
        or with clear error messages (depends on implementation).
        """
        for name, gen_fn in GENS.items():
            gen = gen_fn()

            # Lowercase variant
            data_lower = _base_section()
            _ = gen.compose(section_data=data_lower, section="Verse", emotion_profile="happy_high")

            # Uppercase variant (should fallback or normalize)
            data_upper = _base_section()
            _ = gen.compose(section_data=data_upper, section="Verse", emotion_profile="HAPPY_HIGH")

            adj_lower = data_lower.get("_emotion_adjustments", {}).get(name, {})
            adj_upper = data_upper.get("_emotion_adjustments", {}).get(name, {})

            # Both should produce non-empty adjustments (either matched or fell back to neutral)
            assert adj_lower, f"{name}: empty on lowercase emotion"
            assert adj_upper, f"{name}: empty on uppercase emotion"

    @pytest.mark.parametrize("emotion", ["happy_high", "neutral_medium", "calm_low"])
    def test_emotion_adjustments_dual_storage(self, emotion):
        """
        Verify that emotion adjustments are stored in BOTH locations:
        1. section_data["_emotion_adjustments"][instrument_name]
        2. generator.current_emotion_adjustments (internal state)
        """
        for name, gen_fn in GENS.items():
            gen = gen_fn()
            data = _base_section()

            _ = gen.compose(section_data=data, section="Verse", emotion_profile=emotion)

            # Check section_data storage
            adj_section = data.get("_emotion_adjustments", {}).get(name, {})
            assert adj_section, f"{name}: section storage empty for {emotion}"

            # Check internal generator storage (if accessible)
            if hasattr(gen, "current_emotion_adjustments"):
                adj_internal = gen.current_emotion_adjustments
                assert adj_internal, f"{name}: internal storage empty for {emotion}"

                # Values should match
                for k in adj_section:
                    if k in adj_internal:
                        assert adj_section[k] == adj_internal[k], (
                            f"{name}.{k}: section={adj_section[k]} != internal={adj_internal[k]}"
                        )
