"""
Integration tests for Bass emotion parameter application (Phase 5.3).

Tests verify that emotion parameters are correctly applied to Bass generation:
- sustain_control: Note duration control (0.60-0.90)
- velocity_boost: Velocity adjustment (-10 to +10)
"""

import pytest
from music21 import instrument, stream
import numpy as np

from generator.bass_generator import BassGenerator


class TestBassEmotionIntegration:
    """Test Bass emotion parameter integration."""

    def test_compose_with_emotion_happy_high(self):
        """Test basic emotion application with happy_high profile."""
        gen = BassGenerator(
            part_name="bass",
            default_instrument=instrument.ElectricBass(),
            global_tempo=120,
        )
        
        section_data = {
            "chord_symbol_for_voicing": "C",
            "q_length": 4.0,
            "section_name": "Test",
            "musical_intent": {},
            "part_params": {
                "bass": {
                    "velocity": 70,
                    "rhythm_key": "basic_chord_tone_quarters"
                }
            }
        }
        
        result = gen.compose(
            section_data=section_data,
            section="Verse",
            emotion_profile="happy_high"
        )
        
        assert result is not None
        assert isinstance(result, stream.Part)
        
        # Check that emotion adjustments were stored
        assert "_emotion_adjustments" in section_data
        assert "bass" in section_data["_emotion_adjustments"]
        params = section_data["_emotion_adjustments"]["bass"]
        assert params.get("velocity_boost") == 10
        assert params.get("sustain_control") == 0.70  # happy_high uses 0.70 (short/staccato)
    
    def test_compose_emotion_comparison(self):
        """Compare different emotion profiles for Bass."""
        gen = BassGenerator(
            part_name="bass",
            default_instrument=instrument.ElectricBass(),
            global_tempo=120,
        )
        
        emotions = ["happy_high", "neutral_medium", "calm_low"]
        results = {}
        
        for emotion in emotions:
            section_data = {
                "chord_symbol_for_voicing": "C",
                "q_length": 4.0,
                "section_name": "Test",
                "musical_intent": {},
                "part_params": {
                    "bass": {
                        "velocity": 70,
                        "rhythm_key": "basic_chord_tone_quarters"
                    }
                }
            }
            
            # Generate multiple samples for statistics
            velocities = []
            durations = []
            
            for _ in range(10):
                result = gen.compose(
                    section_data=section_data,
                    section="Verse",
                    emotion_profile=emotion
                )
                
                # Extract velocities and durations from Part
                notes = list(result.flatten().notes)
                velocities.extend([n.volume.velocity for n in notes])
                durations.extend([float(n.duration.quarterLength) for n in notes])
            
            results[emotion] = {
                "velocity_mean": np.mean(velocities) if velocities else 0,
                "velocity_std": np.std(velocities) if velocities else 0,
                "duration_mean": np.mean(durations) if durations else 0,
                "duration_std": np.std(durations) if durations else 0,
            }
        
        print("\n=== Bass Emotion Profile Comparison ===")
        for emotion, metrics in results.items():
            print(f"\n{emotion}:")
            print(f"  Velocity Mean: {metrics['velocity_mean']:.2f}")
            print(f"  Velocity STD: {metrics['velocity_std']:.2f}")
            print(f"  Duration Mean: {metrics['duration_mean']:.3f}")
            print(f"  Duration STD: {metrics['duration_std']:.3f}")
        
        # Verify happy_high has higher velocity than calm_low (velocity_boost effect)
        assert results["happy_high"]["velocity_mean"] > results["calm_low"]["velocity_mean"], \
            "happy_high should have higher velocity than calm_low"
        
        # Verify calm_low has longer durations than happy_high (sustain_control effect)
        # calm_low (0.85) has higher sustain than happy_high (0.70)
        assert results["calm_low"]["duration_mean"] > results["happy_high"]["duration_mean"], \
            f"calm_low should have longer notes than happy_high: calm={results['calm_low']['duration_mean']:.3f}, happy={results['happy_high']['duration_mean']:.3f}"
        
        print("\n✅ Bass emotion comparison successful!")
    
    def test_compose_backward_compatibility(self):
        """Test compose without emotion profile (backward compatibility)."""
        gen = BassGenerator(
            part_name="bass",
            default_instrument=instrument.ElectricBass(),
            global_tempo=120,
        )
        
        section_data = {
            "chord_symbol_for_voicing": "C",
            "q_length": 4.0,
            "section_name": "Test",
            "musical_intent": {},
            "part_params": {
                "bass": {
                    "velocity": 70,
                    "rhythm_key": "basic_chord_tone_quarters"
                }
            }
        }
        
        # No emotion_profile specified
        result = gen.compose(
            section_data=section_data,
            section="Verse"
        )
        
        assert result is not None
        assert isinstance(result, stream.Part)
        
        # Should work without emotion parameters
        notes = list(result.flatten().notes)
        assert len(notes) > 0
    
    def test_compose_with_all_emotion_profiles(self):
        """Test generation with all emotion profiles."""
        gen = BassGenerator(
            part_name="bass",
            default_instrument=instrument.ElectricBass(),
            global_tempo=120,
        )
        
        emotion_profiles = [
            "happy_high", "happy_medium", "happy_low",
            "neutral_high", "neutral_medium", "neutral_low",
            "calm_high", "calm_medium", "calm_low",
            "sad_low"
        ]
        
        section_data = {
            "chord_symbol_for_voicing": "C",
            "q_length": 4.0,
            "section_name": "Test",
            "musical_intent": {},
            "part_params": {
                "bass": {
                    "velocity": 70,
                    "rhythm_key": "basic_chord_tone_quarters"
                }
            }
        }
        
        for emotion in emotion_profiles:
            result = gen.compose(
                section_data=section_data.copy(),
                section="Verse",
                emotion_profile=emotion
            )
            assert result is not None, f"Failed to generate with emotion: {emotion}"
            notes = list(result.flatten().notes)
            assert len(notes) > 0, f"No notes generated for emotion: {emotion}"
    
    def test_velocity_boost_consistency(self):
        """Test that velocity_boost is consistently applied."""
        gen = BassGenerator(
            part_name="bass",
            default_instrument=instrument.ElectricBass(),
            global_tempo=120,
        )
        
        emotions_with_boost = {
            "happy_high": +10,
            "neutral_medium": 0,
            "calm_low": -10,
        }
        
        results = {}
        
        for emotion, expected_boost in emotions_with_boost.items():
            section_data = {
                "chord_symbol_for_voicing": "C",
                "q_length": 4.0,
                "section_name": "Test",
                "musical_intent": {},
                "part_params": {
                    "bass": {
                        "velocity": 48,  # Base velocity
                        "rhythm_key": "basic_chord_tone_quarters"
                    }
                }
            }
            
            result = gen.compose(
                section_data=section_data,
                section="Verse",
                emotion_profile=emotion
            )
            
            notes = list(result.flatten().notes)
            velocities = [n.volume.velocity for n in notes]
            mean_velocity = np.mean(velocities) if velocities else 0
            results[emotion] = mean_velocity
        
        print("\n=== Velocity Boost Consistency Test ===")
        for emotion, mean_vel in results.items():
            expected_boost = emotions_with_boost[emotion]
            print(f"{emotion}: Mean velocity = {mean_vel:.2f} (expected boost: {expected_boost:+d})")
        
        # Verify ordering
        assert results["happy_high"] > results["neutral_medium"] > results["calm_low"], \
            "Velocity ordering should be happy_high > neutral_medium > calm_low"
        
        print("\n✅ Velocity boost ordering verified!")
    
    def test_sustain_control_consistency(self):
        """Test that sustain_control is consistently applied."""
        gen = BassGenerator(
            part_name="bass",
            default_instrument=instrument.ElectricBass(),
            global_tempo=120,
        )
        
        emotions_with_sustain = {
            "happy_high": 0.70,     # Shorter notes
            "neutral_medium": 0.75,  # Medium
            "calm_low": 0.85,        # Longer notes
        }
        
        results = {}
        
        for emotion, expected_sustain in emotions_with_sustain.items():
            section_data = {
                "chord_symbol_for_voicing": "C",
                "q_length": 4.0,
                "section_name": "Test",
                "musical_intent": {},
                "part_params": {
                    "bass": {
                        "velocity": 70,
                        "rhythm_key": "basic_chord_tone_quarters"
                    }
                }
            }
            
            result = gen.compose(
                section_data=section_data,
                section="Verse",
                emotion_profile=emotion
            )
            
            notes = list(result.flatten().notes)
            durations = [float(n.duration.quarterLength) for n in notes]
            mean_duration = np.mean(durations) if durations else 0
            results[emotion] = mean_duration
        
        print("\n=== Sustain Control Consistency Test ===")
        for emotion, mean_dur in results.items():
            expected_sustain = emotions_with_sustain[emotion]
            print(f"{emotion}: Mean duration = {mean_dur:.3f} (sustain control: {expected_sustain})")
        
        # Verify ordering: calm_low (0.85) > neutral (0.75) > happy_high (0.70)
        assert results["calm_low"] > results["neutral_medium"] > results["happy_high"], \
            "Duration ordering should be calm_low > neutral_medium > happy_high"
        
        print("\n✅ Sustain control ordering verified!")
