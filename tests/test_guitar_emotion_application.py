"""
Phase 5.2: Guitar Emotion Application Tests
strum_consistency_target and velocity_boost application tests
"""

import pytest
import numpy as np
from generator.guitar_generator import GuitarGenerator
from music21 import harmony, instrument


class TestGuitarEmotionApplication:
    """Guitar emotion parameter application tests"""
    
    def test_strum_consistency_target_applied(self):
        """strum_consistency_targetがtiming_variationに正しく適用されることを確認"""
        # Setup generator
        gen = GuitarGenerator(
            part_name="guitar",
            default_instrument=instrument.AcousticGuitar(),
            global_tempo=120,
            global_time_signature="4/4",
            part_parameters={
                "guitar_folk_strum_simple": {
                    "pattern": [
                        {"offset": 0.0, "duration": 1.0, "pattern_type": "strum"},
                        {"offset": 1.0, "duration": 1.0, "pattern_type": "strum"},
                        {"offset": 2.0, "duration": 1.0, "pattern_type": "strum"},
                        {"offset": 3.0, "duration": 1.0, "pattern_type": "strum"},
                    ],
                    "length_beats": 4.0,
                    "execution_style": "block_chord"
                }
            }
        )
        
        # Test with happy_high (strum_consistency_target = 0.80)
        section_data_high_consistency = {
            "chord_symbol_for_voicing": "C",
            "q_length": 4.0,
            "section_name": "Test",
            "part_params": {
                "guitar": {
                    "velocity": 70,
                    "rhythm_key": "guitar_folk_strum_simple"
                }
            },
            "_emotion_adjustments": {
                "guitar": {
                    "strum_consistency_target": 0.80,  # High consistency
                    "velocity_boost": 0
                }
            }
        }
        
        # Test with calm_low (strum_consistency_target = 0.70)
        section_data_low_consistency = {
            "chord_symbol_for_voicing": "C",
            "q_length": 4.0,
            "section_name": "Test",
            "part_params": {
                "guitar": {
                    "velocity": 70,
                    "rhythm_key": "guitar_folk_strum_simple"
                }
            },
            "_emotion_adjustments": {
                "guitar": {
                    "strum_consistency_target": 0.70,  # Low consistency
                    "velocity_boost": 0
                }
            }
        }
        
        # Generate multiple samples for statistical comparison
        high_consistency_offsets = []
        low_consistency_offsets = []
        
        for _ in range(10):
            # Generate with high consistency
            result_high = gen._render_part(section_data_high_consistency)
            notes_high = list(result_high.flatten().notes)
            if len(notes_high) > 1:
                # Calculate offset std deviation
                offsets = [float(n.offset) for n in notes_high]
                expected_offsets = [i * 1.0 for i in range(len(offsets))]
                deviations = [abs(actual - expected) for actual, expected in zip(offsets, expected_offsets)]
                high_consistency_offsets.extend(deviations)
            
            # Generate with low consistency
            result_low = gen._render_part(section_data_low_consistency)
            notes_low = list(result_low.flatten().notes)
            if len(notes_low) > 1:
                offsets = [float(n.offset) for n in notes_low]
                expected_offsets = [i * 1.0 for i in range(len(offsets))]
                deviations = [abs(actual - expected) for actual, expected in zip(offsets, expected_offsets)]
                low_consistency_offsets.extend(deviations)
        
        # Calculate statistics
        high_consistency_mean_dev = np.mean(high_consistency_offsets)
        low_consistency_mean_dev = np.mean(low_consistency_offsets)
        
        print(f"\nStrum Consistency Test:")
        print(f"  High consistency (0.80): Mean deviation = {high_consistency_mean_dev:.4f}")
        print(f"  Low consistency (0.70): Mean deviation = {low_consistency_mean_dev:.4f}")
        print(f"  Ratio (low/high): {low_consistency_mean_dev / high_consistency_mean_dev:.2f}")
        
        # Verify low consistency has higher deviation than high consistency
        assert low_consistency_mean_dev > high_consistency_mean_dev, \
            "Low consistency (0.70) should have higher timing deviation than high consistency (0.80)"
        
        # Verify ratio is reasonable (low should be ~2-3x higher)
        ratio = low_consistency_mean_dev / high_consistency_mean_dev
        assert 1.5 < ratio < 4.0, f"Deviation ratio should be between 1.5-4.0, got {ratio:.2f}"
    
    def test_velocity_boost_applied(self):
        """velocity_boostが正しく適用されることを確認"""
        gen = GuitarGenerator(
            part_name="guitar",
            default_instrument=instrument.AcousticGuitar(),
            global_tempo=120,
            part_parameters={
                "guitar_folk_strum_simple": {
                    "pattern": [
                        {"offset": 0.0, "duration": 1.0, "pattern_type": "strum"},
                        {"offset": 1.0, "duration": 1.0, "pattern_type": "strum"},
                        {"offset": 2.0, "duration": 1.0, "pattern_type": "strum"},
                        {"offset": 3.0, "duration": 1.0, "pattern_type": "strum"},
                    ],
                    "length_beats": 4.0
                }
            }
        )
        
        base_velocity = 70
        
        # Test with +10 boost (happy_high)
        section_data_boost_pos = {
            "chord_symbol_for_voicing": "C",
            "q_length": 4.0,
            "section_name": "Test",
            "part_params": {
                "guitar": {
                    "velocity": base_velocity,
                    "rhythm_key": "guitar_folk_strum_simple"
                }
            },
            "_emotion_adjustments": {
                "guitar": {
                    "strum_consistency_target": 0.75,
                    "velocity_boost": 10  # Positive boost
                }
            }
        }
        
        # Test with -10 boost (calm_low)
        section_data_boost_neg = {
            "chord_symbol_for_voicing": "C",
            "q_length": 4.0,
            "section_name": "Test",
            "part_params": {
                "guitar": {
                    "velocity": base_velocity,
                    "rhythm_key": "guitar_folk_strum_simple"
                }
            },
            "_emotion_adjustments": {
                "guitar": {
                    "strum_consistency_target": 0.75,
                    "velocity_boost": -10  # Negative boost
                }
            }
        }
        
        # Test with no boost (neutral)
        section_data_no_boost = {
            "chord_symbol_for_voicing": "C",
            "q_length": 4.0,
            "section_name": "Test",
            "part_params": {
                "guitar": {
                    "velocity": base_velocity,
                    "rhythm_key": "guitar_folk_strum_simple"
                }
            },
            "_emotion_adjustments": {
                "guitar": {
                    "strum_consistency_target": 0.75,
                    "velocity_boost": 0
                }
            }
        }
        
        # Generate and collect velocities
        velocities_boost_pos = []
        velocities_boost_neg = []
        velocities_no_boost = []
        
        for _ in range(10):
            result_pos = gen._render_part(section_data_boost_pos)
            notes_pos = list(result_pos.flatten().notes)
            velocities_boost_pos.extend([n.volume.velocity for n in notes_pos])
            
            result_neg = gen._render_part(section_data_boost_neg)
            notes_neg = list(result_neg.flatten().notes)
            velocities_boost_neg.extend([n.volume.velocity for n in notes_neg])
            
            result_no = gen._render_part(section_data_no_boost)
            notes_no = list(result_no.flatten().notes)
            velocities_no_boost.extend([n.volume.velocity for n in notes_no])
        
        # Calculate means
        mean_boost_pos = np.mean(velocities_boost_pos)
        mean_boost_neg = np.mean(velocities_boost_neg)
        mean_no_boost = np.mean(velocities_no_boost)
        
        print(f"\nVelocity Boost Test:")
        print(f"  Positive boost (+10): Mean velocity = {mean_boost_pos:.2f}")
        print(f"  No boost (0): Mean velocity = {mean_no_boost:.2f}")
        print(f"  Negative boost (-10): Mean velocity = {mean_boost_neg:.2f}")
        print(f"  Diff (pos - neutral): {mean_boost_pos - mean_no_boost:.2f}")
        print(f"  Diff (neutral - neg): {mean_no_boost - mean_boost_neg:.2f}")
        
        # Verify boost is applied correctly
        assert mean_boost_pos > mean_no_boost, "Positive boost should increase velocity"
        assert mean_no_boost > mean_boost_neg, "Negative boost should decrease velocity"
        
        # Verify boost magnitude is approximately correct
        # Allow some tolerance due to clamping and other factors
        assert 5 < (mean_boost_pos - mean_no_boost) < 15, \
            f"Positive boost effect should be ~10, got {mean_boost_pos - mean_no_boost:.2f}"
        assert 5 < (mean_no_boost - mean_boost_neg) < 15, \
            f"Negative boost effect should be ~10, got {mean_no_boost - mean_boost_neg:.2f}"
        
        # Verify all velocities are in valid range
        all_velocities = velocities_boost_pos + velocities_boost_neg + velocities_no_boost
        assert all(1 <= v <= 127 for v in all_velocities), \
            "All velocities should be in MIDI range [1-127]"
    
    def test_guitar_emotion_no_emotion(self):
        """感情指定なしでも正常に動作することを確認"""
        gen = GuitarGenerator(
            part_name="guitar",
            default_instrument=instrument.AcousticGuitar(),
            global_tempo=120,
            part_parameters={
                "guitar_folk_strum_simple": {
                    "pattern": [
                        {"offset": 0.0, "duration": 1.0, "pattern_type": "strum"},
                        {"offset": 1.0, "duration": 1.0, "pattern_type": "strum"},
                    ],
                    "length_beats": 4.0
                }
            }
        )
        
        section_data_no_emotion = {
            "chord_symbol_for_voicing": "C",
            "q_length": 4.0,
            "section_name": "Test",
            "part_params": {
                "guitar": {
                    "velocity": 70,
                    "rhythm_key": "guitar_folk_strum_simple"
                }
            }
            # No _emotion_adjustments
        }
        
        # Should work without error
        result = gen._render_part(section_data_no_emotion)
        notes = list(result.flatten().notes)
        
        assert len(notes) > 0, "Should generate notes without emotion adjustments"
        
        # Verify velocities are in range
        velocities = [n.volume.velocity for n in notes]
        assert all(1 <= v <= 127 for v in velocities), \
            "All velocities should be in valid MIDI range"
        
        print(f"\nNo Emotion Test: Generated {len(notes)} notes successfully")
    
    def test_guitar_emotion_extreme_values(self):
        """極端なパラメータ値でも正常に動作することを確認"""
        gen = GuitarGenerator(
            part_name="guitar",
            default_instrument=instrument.AcousticGuitar(),
            global_tempo=120,
            part_parameters={
                "guitar_folk_strum_simple": {
                    "pattern": [
                        {"offset": 0.0, "duration": 1.0, "pattern_type": "strum"},
                        {"offset": 1.0, "duration": 1.0, "pattern_type": "strum"},
                        {"offset": 2.0, "duration": 1.0, "pattern_type": "strum"},
                    ],
                    "length_beats": 4.0
                }
            }
        )
        
        # Test extreme consistency values
        section_data_extreme_high = {
            "chord_symbol_for_voicing": "C",
            "q_length": 4.0,
            "section_name": "Test",
            "part_params": {
                "guitar": {
                    "velocity": 80,
                    "rhythm_key": "guitar_folk_strum_simple"
                }
            },
            "_emotion_adjustments": {
                "guitar": {
                    "strum_consistency_target": 0.90,  # Very high
                    "velocity_boost": 20  # Very high boost
                }
            }
        }
        
        section_data_extreme_low = {
            "chord_symbol_for_voicing": "C",
            "q_length": 4.0,
            "section_name": "Test",
            "part_params": {
                "guitar": {
                    "velocity": 40,
                    "rhythm_key": "guitar_folk_strum_simple"
                }
            },
            "_emotion_adjustments": {
                "guitar": {
                    "strum_consistency_target": 0.60,  # Very low
                    "velocity_boost": -30  # Very low boost
                }
            }
        }
        
        # Should work without error
        result_high = gen._render_part(section_data_extreme_high)
        result_low = gen._render_part(section_data_extreme_low)
        
        notes_high = list(result_high.flatten().notes)
        notes_low = list(result_low.flatten().notes)
        
        assert len(notes_high) > 0, "Should generate notes with extreme high values"
        assert len(notes_low) > 0, "Should generate notes with extreme low values"
        
        # Verify velocities are clamped to valid range
        velocities_high = [n.volume.velocity for n in notes_high]
        velocities_low = [n.volume.velocity for n in notes_low]
        
        assert all(1 <= v <= 127 for v in velocities_high), \
            "High velocity values should be clamped to MIDI range"
        assert all(1 <= v <= 127 for v in velocities_low), \
            "Low velocity values should be clamped to MIDI range"
        
        print(f"\nExtreme Values Test:")
        print(f"  Extreme high: {len(notes_high)} notes, velocity range {min(velocities_high)}-{max(velocities_high)}")
        print(f"  Extreme low: {len(notes_low)} notes, velocity range {min(velocities_low)}-{max(velocities_low)}")


if __name__ == "__main__":
    # Run unit tests
    pytest.main([__file__, "-v", "-s"])
