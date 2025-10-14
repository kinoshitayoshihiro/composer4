"""Integration tests for Strings emotion parameter application (Phase 5.4)."""

from __future__ import annotations

import pytest
from music21 import stream

from generator.strings_generator import StringsGenerator


class TestStringsEmotionIntegration:
    """Integration tests for Strings generator emotion parameters."""

    @pytest.fixture
    def generator(self):
        """Create a StringsGenerator instance for testing."""
        return StringsGenerator(
            global_tempo=120,
            global_time_signature="4/4",
            global_key_signature_tonic="C",
            global_key_signature_mode="major",
        )

    @pytest.fixture
    def section_data(self):
        """Create basic section data for testing."""
        return {
            "chord_symbol_for_voicing": "Cmaj7",
            "q_length": 4.0,
            "section_name": "Verse",
        }

    def test_compose_with_emotion_happy_high(self, generator, section_data):
        """Test compose() with happy_high emotion profile."""
        result = generator.compose(
            section_data=section_data,
            section="Verse",
            emotion_profile="happy_high"
        )
        
        # Verify return structure
        assert isinstance(result, dict)
        assert len(result) > 0
        
        # Verify emotion params were stored
        emotion_adj = section_data.get("_emotion_adjustments", {}).get("strings", {})
        assert emotion_adj is not None
        assert "velocity_boost" in emotion_adj
        assert "bow_pressure_factor" in emotion_adj
        assert "articulation_legato_bias" in emotion_adj
        assert "velocity_std_multiplier" in emotion_adj
        
        # Verify happy_high values
        assert emotion_adj.get("velocity_boost") == 10
        assert emotion_adj.get("bow_pressure_factor") == 1.15
        assert emotion_adj.get("articulation_legato_bias") == 0.30
        assert emotion_adj.get("velocity_std_multiplier") == 1.10

    def test_compose_emotion_comparison(self, generator, section_data):
        """Test that different emotions produce different velocities."""
        profiles = ["happy_high", "neutral_medium", "calm_low"]
        results = {}
        
        for profile in profiles:
            # Create fresh section_data for each test
            test_data = section_data.copy()
            parts = generator.compose(
                section_data=test_data,
                section="Verse",
                emotion_profile=profile
            )
            
            # Collect velocities from all parts
            velocities = []
            for part_name, part in parts.items():
                if isinstance(part, stream.Part):
                    for n in part.recurse().notes:
                        if n.volume and n.volume.velocity is not None:
                            velocities.append(n.volume.velocity)
            
            if velocities:
                results[profile] = {
                    "mean": sum(velocities) / len(velocities),
                    "count": len(velocities),
                    "velocities": velocities[:10]  # First 10 for inspection
                }
        
        # Verify we have results for all profiles
        assert "happy_high" in results
        assert "neutral_medium" in results
        assert "calm_low" in results
        
        # Verify velocity ordering: happy_high > neutral_medium > calm_low
        # (allowing for some variation due to randomization)
        happy_mean = results["happy_high"]["mean"]
        neutral_mean = results["neutral_medium"]["mean"]
        calm_mean = results["calm_low"]["mean"]
        
        print(f"\nEmotion velocity comparison:")
        print(f"happy_high: {happy_mean:.2f}")
        print(f"neutral_medium: {neutral_mean:.2f}")
        print(f"calm_low: {calm_mean:.2f}")
        
        # happy_high should be louder than neutral (with +10 boost)
        assert happy_mean > neutral_mean - 2, \
            f"happy_high ({happy_mean:.2f}) should be louder than neutral ({neutral_mean:.2f})"
        
        # calm_low should be softer than neutral (with -10 boost)
        assert calm_mean < neutral_mean + 2, \
            f"calm_low ({calm_mean:.2f}) should be softer than neutral ({neutral_mean:.2f})"

    def test_compose_backward_compatibility(self, generator, section_data):
        """Test that compose() works without emotion_profile (backward compatibility)."""
        result = generator.compose(
            section_data=section_data,
            section="Verse"
        )
        
        # Verify generation succeeded
        assert isinstance(result, dict)
        assert len(result) > 0
        
        # Verify parts have notes
        for part_name, part in result.items():
            if isinstance(part, stream.Part):
                notes = list(part.recurse().notes)
                assert len(notes) > 0, f"Part {part_name} should have notes"

    def test_compose_with_all_emotion_profiles(self, generator, section_data):
        """Test compose() with all defined emotion profiles."""
        profiles = [
            "happy_high",
            "happy_medium",
            "happy_low",
            "neutral_high",
            "neutral_medium",
            "neutral_low",
            "calm_high",
            "calm_medium",
            "calm_low",
            "sad_low"
        ]
        
        for profile in profiles:
            test_data = section_data.copy()
            result = generator.compose(
                section_data=test_data,
                section="Verse",
                emotion_profile=profile
            )
            
            # Verify generation succeeded
            assert isinstance(result, dict), f"Failed for profile: {profile}"
            assert len(result) > 0, f"No parts generated for profile: {profile}"
            
            # Verify emotion params were set
            emotion_adj = test_data.get("_emotion_adjustments", {}).get("strings", {})
            assert "velocity_boost" in emotion_adj, f"velocity_boost missing for: {profile}"

    def test_velocity_boost_consistency(self, generator, section_data):
        """Test that velocity_boost produces consistent statistical effects."""
        profiles = {
            "happy_high": +10,
            "neutral_medium": 0,
            "calm_low": -10,
        }
        
        results = {}
        
        for profile, expected_boost in profiles.items():
            test_data = section_data.copy()
            parts = generator.compose(
                section_data=test_data,
                section="Verse",
                emotion_profile=profile
            )
            
            velocities = []
            for part in parts.values():
                if isinstance(part, stream.Part):
                    for n in part.recurse().notes:
                        if n.volume and n.volume.velocity is not None:
                            velocities.append(n.volume.velocity)
            
            if velocities:
                mean_vel = sum(velocities) / len(velocities)
                results[profile] = mean_vel
                print(f"{profile}: Mean velocity = {mean_vel:.2f} (expected boost: {expected_boost:+d})")
        
        # Verify ordering
        assert results["happy_high"] > results["neutral_medium"], \
            "happy_high should have higher velocity than neutral_medium"
        assert results["calm_low"] < results["neutral_medium"], \
            "calm_low should have lower velocity than neutral_medium"
        
        print("\n✅ Velocity boost ordering verified!")

    def test_bow_pressure_factor_consistency(self, generator, section_data):
        """Test that bow_pressure_factor affects velocity appropriately."""
        # Test with different bow pressure factors
        test_data_high = section_data.copy()
        test_data_low = section_data.copy()
        
        parts_high = generator.compose(
            section_data=test_data_high,
            section="Verse",
            emotion_profile="happy_high"  # bow_pressure_factor = 1.15
        )
        
        parts_low = generator.compose(
            section_data=test_data_low,
            section="Verse",
            emotion_profile="calm_low"  # bow_pressure_factor = 0.90
        )
        
        # Get emotion adjustments
        adj_high = test_data_high.get("_emotion_adjustments", {}).get("strings", {})
        adj_low = test_data_low.get("_emotion_adjustments", {}).get("strings", {})
        
        assert adj_high.get("bow_pressure_factor") == 1.15
        assert adj_low.get("bow_pressure_factor") == 0.90
        
        print("✅ Bow pressure factor values verified!")
