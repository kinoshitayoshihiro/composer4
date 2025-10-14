"""
Phase 5.1: Piano Emotion Application Tests
velocity_std_multiplier適用のテスト
"""

import pytest
import numpy as np
from generator.piano_generator import PianoGenerator
from music21 import harmony


class TestPianoEmotionApplication:
    """Piano emotion parameter application tests"""
    
    def test_velocity_std_multiplier_applied(self):
        """velocity_std_multiplierが実際に適用されることを確認"""
        # Setup generator
        gen = PianoGenerator(
            default_instrument="Piano",
            global_tempo=120,
            global_time_signature="4/4",
            part_parameters={
                "piano_rh_block_chords_quarters": {
                    "pattern": [
                        {"offset": 0.0, "duration": 1.0, "type": "chord", "velocity_factor": 1.0},
                        {"offset": 1.0, "duration": 1.0, "type": "chord", "velocity_factor": 1.0},
                        {"offset": 2.0, "duration": 1.0, "type": "chord", "velocity_factor": 1.0},
                        {"offset": 3.0, "duration": 1.0, "type": "chord", "velocity_factor": 1.0},
                    ],
                    "length_beats": 4.0
                }
            }
        )
        
        # Test with happy_high (velocity_std_multiplier = 1.3)
        section_data_happy = {
            "chord_symbol_for_voicing": "C",
            "q_length": 4.0,
            "part_params": {"piano": {"velocity": 70}},
            "_emotion_adjustments": {
                "piano": {
                    "velocity_std_multiplier": 1.3
                }
            }
        }
        
        # Test with neutral_medium (velocity_std_multiplier = 1.0)
        section_data_neutral = {
            "chord_symbol_for_voicing": "C",
            "q_length": 4.0,
            "part_params": {"piano": {"velocity": 70}},
            "_emotion_adjustments": {
                "piano": {
                    "velocity_std_multiplier": 1.0
                }
            }
        }
        
        # Generate multiple samples for statistical comparison
        happy_velocities = []
        neutral_velocities = []
        
        for _ in range(20):
            # Generate with happy_high
            cs = harmony.ChordSymbol("C")
            result_happy = gen._render_hand_part(
                "RH",
                cs,
                4.0,
                "piano_rh_block_chords_quarters",
                section_data_happy.get("part_params", {}).get("piano", {}),
                section_data=section_data_happy
            )
            
            # Generate with neutral_medium
            result_neutral = gen._render_hand_part(
                "RH",
                cs,
                4.0,
                "piano_rh_block_chords_quarters",
                section_data_neutral.get("part_params", {}).get("piano", {}),
                section_data=section_data_neutral
            )
            
            # Extract velocities
            for n in result_happy.flatten().notes:
                happy_velocities.append(n.volume.velocity)
            
            for n in result_neutral.flatten().notes:
                neutral_velocities.append(n.volume.velocity)
        
        # Calculate standard deviations
        happy_std = np.std(happy_velocities)
        neutral_std = np.std(neutral_velocities)
        
        # Verify: happy_high should have ~30% more variation
        # Expected: neutral_std ≈ 15, happy_std ≈ 19.5
        print(f"\nVelocity STD - Neutral: {neutral_std:.2f}, Happy: {happy_std:.2f}")
        print(f"Ratio: {happy_std/neutral_std:.2f} (expected: ~1.3)")
        
        # Allow some tolerance due to random variation
        assert happy_std > neutral_std, "happy_high should have more velocity variation"
        assert 1.15 < happy_std / neutral_std < 1.50, f"Ratio should be ~1.3, got {happy_std/neutral_std:.2f}"
    
    def test_velocity_std_multiplier_no_emotion(self):
        """emotion_adjustmentsがない場合、デフォルト動作を確認"""
        gen = PianoGenerator(
            default_instrument="Piano",
            global_tempo=120,
            part_parameters={
                "piano_rh_block_chords_quarters": {
                    "pattern": [
                        {"offset": 0.0, "duration": 1.0, "type": "chord", "velocity_factor": 1.0},
                        {"offset": 1.0, "duration": 1.0, "type": "chord", "velocity_factor": 1.0},
                    ],
                    "length_beats": 4.0
                }
            }
        )
        
        # No emotion adjustments
        section_data = {
            "chord_symbol_for_voicing": "C",
            "q_length": 4.0,
            "part_params": {"piano": {"velocity": 70}}
        }
        
        cs = harmony.ChordSymbol("C")
        result = gen._render_hand_part(
            "RH",
            cs,
            4.0,
            "piano_rh_block_chords_quarters",
            section_data.get("part_params", {}).get("piano", {}),
            section_data=section_data
        )
        
        # Should generate without error
        notes = list(result.flatten().notes)
        assert len(notes) > 0, "Should generate notes"
        
        # Velocities should be within reasonable range
        velocities = [n.volume.velocity for n in notes]
        assert all(1 <= v <= 127 for v in velocities), "Velocities should be in valid range"
    
    def test_velocity_std_extreme_multipliers(self):
        """極端なmultiplierでも正常に動作することを確認"""
        gen = PianoGenerator(
            default_instrument="Piano",
            global_tempo=120,
            part_parameters={
                "piano_rh_block_chords_quarters": {
                    "pattern": [
                        {"offset": 0.0, "duration": 1.0, "type": "chord", "velocity_factor": 1.0},
                    ],
                    "length_beats": 4.0
                }
            }
        )
        
        # Test calm_low (0.7x)
        section_data_calm = {
            "chord_symbol_for_voicing": "C",
            "q_length": 4.0,
            "part_params": {"piano": {"velocity": 70}},
            "_emotion_adjustments": {
                "piano": {
                    "velocity_std_multiplier": 0.7
                }
            }
        }
        
        # Test energetic_high (1.5x)
        section_data_energetic = {
            "chord_symbol_for_voicing": "C",
            "q_length": 4.0,
            "part_params": {"piano": {"velocity": 70}},
            "_emotion_adjustments": {
                "piano": {
                    "velocity_std_multiplier": 1.5
                }
            }
        }
        
        cs = harmony.ChordSymbol("C")
        
        # Generate samples
        for _ in range(10):
            result_calm = gen._render_hand_part(
                "RH", cs, 4.0, "piano_rh_block_chords_quarters",
                section_data_calm.get("part_params", {}).get("piano", {}),
                section_data=section_data_calm
            )
            
            result_energetic = gen._render_hand_part(
                "RH", cs, 4.0, "piano_rh_block_chords_quarters",
                section_data_energetic.get("part_params", {}).get("piano", {}),
                section_data=section_data_energetic
            )
            
            # Check all velocities are in valid range
            for result in [result_calm, result_energetic]:
                velocities = [n.volume.velocity for n in result.flatten().notes]
                assert all(1 <= v <= 127 for v in velocities), "All velocities must be in range 1-127"


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "-s"])
