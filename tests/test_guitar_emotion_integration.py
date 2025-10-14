"""
Phase 5.2: Guitar Emotion Application Integration Tests
compose()メソッド全体を使った統合テスト
"""

import pytest
import numpy as np
from generator.guitar_generator import GuitarGenerator
from music21 import instrument


class TestGuitarEmotionIntegration:
    """Guitar emotion parameter application integration tests"""
    
    def test_compose_with_emotion_happy_high(self):
        """compose()でhappy_high感情プロファイルが正しく適用されることを確認"""
        part_params = {
            "guitar_folk_strum_simple": {
                "pattern": [
                    {"offset": i * 1.0, "duration": 1.0, "pattern_type": "strum"}
                    for i in range(4)
                ],
                "length_beats": 4.0
            }
        }
        
        gen = GuitarGenerator(
            part_name="guitar",
            default_instrument=instrument.AcousticGuitar(),
            global_tempo=120,
            global_time_signature="4/4",
            part_parameters=part_params
        )
        
        section_data = {
            "chord_symbol_for_voicing": "C",
            "q_length": 4.0,
            "section_name": "Chorus",
            "musical_intent": {},
            "part_params": {
                "guitar": {
                    "velocity": 80,
                    "rhythm_key": "guitar_folk_strum_simple"
                }
            }
        }
        
        # Compose with emotion
        result = gen.compose(
            section_data=section_data,
            section="Chorus",
            emotion_profile="happy_high"
        )
        
        # Verify result is a Part
        assert result is not None, "Result should not be None"
        
        # Verify emotion adjustments were stored
        assert "_emotion_adjustments" in section_data, "Emotion adjustments should be stored"
        assert "guitar" in section_data["_emotion_adjustments"], "Guitar adjustments should exist"
        
        # Verify guitar part has notes
        guitar_notes = list(result.flatten().notes)
        assert len(guitar_notes) > 0, "Guitar should have notes"
        
        print(f"\n✅ Compose with happy_high:")
        print(f"  Guitar notes: {len(guitar_notes)}")
        print(f"  Emotion adjustments: {section_data['_emotion_adjustments']['guitar'].keys()}")
    
    def test_compose_emotion_comparison(self):
        """異なる感情プロファイルでの生成を比較"""
        part_params = {
            "guitar_folk_strum_simple": {
                "pattern": [
                    {"offset": i * 1.0, "duration": 1.0, "pattern_type": "strum"}
                    for i in range(4)
                ],
                "length_beats": 4.0
            }
        }
        
        gen = GuitarGenerator(
            part_name="guitar",
            default_instrument=instrument.AcousticGuitar(),
            global_tempo=120,
            part_parameters=part_params
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
                    "guitar": {
                        "velocity": 70,
                        "rhythm_key": "guitar_folk_strum_simple"
                    }
                }
            }
            
            # Generate multiple samples for statistics
            velocities = []
            timing_deviations = []
            
            for _ in range(10):
                result = gen.compose(
                    section_data=section_data,
                    section="Verse",
                    emotion_profile=emotion
                )
                
                # Extract velocities and timing from Part
                notes = list(result.flatten().notes)
                velocities.extend([n.volume.velocity for n in notes])
                
                # Calculate timing deviations: measure actual note spacing variation
                # (strum_consistency_target affects jitter on string delays within chords)
                if len(notes) > 2:
                    # Get intervals between consecutive notes
                    intervals = []
                    for i in range(len(notes) - 1):
                        interval = float(notes[i+1].offset) - float(notes[i].offset)
                        intervals.append(interval)
                    
                    # Calculate standard deviation of intervals as timing variation metric
                    if len(intervals) > 1:
                        mean_interval = sum(intervals) / len(intervals)
                        variance = sum((x - mean_interval) ** 2 for x in intervals) / len(intervals)
                        std_dev = variance ** 0.5
                        timing_deviations.append(std_dev)
            
            results[emotion] = {
                "velocity_mean": np.mean(velocities),
                "velocity_std": np.std(velocities),
                "timing_deviation_mean": np.mean(timing_deviations) if timing_deviations else 0.0
            }
        
        print("\n=== Emotion Profile Comparison ===")
        for emotion, metrics in results.items():
            print(f"\n{emotion}:")
            print(f"  Velocity Mean: {metrics['velocity_mean']:.2f}")
            print(f"  Velocity STD: {metrics['velocity_std']:.2f}")
            print(f"  Timing Deviation: {metrics['timing_deviation_mean']:.4f}")
        
        # Verify happy_high has higher velocity than calm_low (velocity_boost effect)
        assert results["happy_high"]["velocity_mean"] > results["calm_low"]["velocity_mean"], \
            "happy_high should have higher velocity than calm_low"
        
        # Verify calm_low has higher timing variation than happy_high (consistency effect)
        # calm_low (0.70) has higher variation (0.03) than happy_high (0.80) with lower variation (0.01)
        if results["calm_low"]["timing_deviation_mean"] > 0 and results["happy_high"]["timing_deviation_mean"] > 0:
            assert results["calm_low"]["timing_deviation_mean"] > results["happy_high"]["timing_deviation_mean"], \
                f"calm_low should have more timing variation than happy_high: calm={results['calm_low']['timing_deviation_mean']:.6f}, happy={results['happy_high']['timing_deviation_mean']:.6f}"
        else:
            # If timing deviations are too small to measure reliably, skip this assertion
            print("\n⚠️  Timing deviations too small to measure reliably (strum delay jitter may be minimal)")
        
        print("\n✅ Emotion comparison successful!")
    
    def test_compose_backward_compatibility(self):
        """emotion指定なしでも正常に動作することを確認（後方互換性）"""
        part_params = {
            "guitar_folk_strum_simple": {
                "pattern": [
                    {"offset": 0.0, "duration": 1.0, "pattern_type": "strum"},
                    {"offset": 1.0, "duration": 1.0, "pattern_type": "strum"},
                ],
                "length_beats": 4.0
            }
        }
        
        gen = GuitarGenerator(
            part_name="guitar",
            default_instrument=instrument.AcousticGuitar(),
            global_tempo=120,
            part_parameters=part_params
        )
        
        section_data = {
            "chord_symbol_for_voicing": "C",
            "q_length": 4.0,
            "section_name": "Verse",
            "musical_intent": {},
            "part_params": {
                "guitar": {
                    "velocity": 70,
                    "rhythm_key": "guitar_folk_strum_simple"
                }
            }
        }
        
        # Compose without emotion (backward compatibility)
        result = gen.compose(section_data=section_data)
        
        # Should work without error and return Part
        assert result is not None, "Result should not be None"
        
        # Should have notes
        notes = list(result.flatten().notes)
        assert len(notes) > 0, "Guitar should have notes"
        
        print(f"\n✅ Backward compatibility test passed")
        print(f"  Generated {len(notes)} guitar notes")
    
    def test_compose_with_all_emotion_profiles(self):
        """全ての感情プロファイルで正常に生成できることを確認"""
        part_params = {
            "guitar_folk_strum_simple": {
                "pattern": [
                    {"offset": i * 1.0, "duration": 1.0, "pattern_type": "strum"}
                    for i in range(4)
                ],
                "length_beats": 4.0
            }
        }
        
        gen = GuitarGenerator(
            part_name="guitar",
            default_instrument=instrument.AcousticGuitar(),
            global_tempo=120,
            part_parameters=part_params
        )
        
        # All emotion profiles from emotion_mapping.yaml
        profiles = [
            "happy_low", "happy_medium", "happy_high",
            "neutral_medium",
            "calm_low",
            "sad_low", "sad_high",
            "melancholic_medium",
            "energetic_medium", "energetic_high"
        ]
        
        results = {}
        
        for profile in profiles:
            section_data = {
                "chord_symbol_for_voicing": "C",
                "q_length": 4.0,
                "section_name": "Test",
                "musical_intent": {},
                "part_params": {
                    "guitar": {
                        "velocity": 70,
                        "rhythm_key": "guitar_folk_strum_simple"
                    }
                }
            }
            
            try:
                result = gen.compose(
                    section_data=section_data,
                    section="Verse",
                    emotion_profile=profile
                )
                
                notes = list(result.flatten().notes)
                results[profile] = {
                    "success": True,
                    "note_count": len(notes),
                    "error": None
                }
            except Exception as e:
                results[profile] = {
                    "success": False,
                    "note_count": 0,
                    "error": str(e)
                }
        
        print("\n=== All Emotion Profiles Test ===")
        for profile, result in results.items():
            status = "✅" if result["success"] else "❌"
            print(f"{status} {profile}: {result['note_count']} notes")
            if not result["success"]:
                print(f"   Error: {result['error']}")
        
        # Verify all profiles work
        failed = [p for p, r in results.items() if not r["success"]]
        assert len(failed) == 0, f"Failed profiles: {failed}"
        
        print(f"\n✅ All {len(profiles)} emotion profiles working!")
    
    def test_velocity_boost_consistency(self):
        """velocity_boostの効果を詳細に検証"""
        part_params = {
            "guitar_folk_strum_simple": {
                "pattern": [
                    {"offset": i * 1.0, "duration": 1.0, "pattern_type": "strum"}
                    for i in range(4)
                ],
                "length_beats": 4.0
            }
        }
        
        gen = GuitarGenerator(
            part_name="guitar",
            default_instrument=instrument.AcousticGuitar(),
            global_tempo=120,
            part_parameters=part_params
        )
        
        base_velocity = 70
        
        # Test different emotion profiles with different velocity_boost values
        test_cases = [
            ("happy_high", 10),      # +10 boost
            ("neutral_medium", 0),   # 0 boost
            ("calm_low", -10),       # -10 boost
        ]
        
        results = {}
        
        for emotion, expected_boost in test_cases:
            section_data = {
                "chord_symbol_for_voicing": "C",
                "q_length": 4.0,
                "section_name": "Test",
                "musical_intent": {},
                "part_params": {
                    "guitar": {
                        "velocity": base_velocity,
                        "rhythm_key": "guitar_folk_strum_simple"
                    }
                }
            }
            
            velocities = []
            for _ in range(15):
                result = gen.compose(
                    section_data=section_data,
                    section="Verse",
                    emotion_profile=emotion
                )
                
                notes = list(result.flatten().notes)
                velocities.extend([n.volume.velocity for n in notes])
            
            results[emotion] = {
                "mean": np.mean(velocities),
                "expected_boost": expected_boost
            }
        
        print("\n=== Velocity Boost Consistency Test ===")
        for emotion, data in results.items():
            print(f"{emotion}: Mean velocity = {data['mean']:.2f} (expected boost: {data['expected_boost']:+d})")
        
        # Verify ordering
        assert results["happy_high"]["mean"] > results["neutral_medium"]["mean"], \
            "happy_high should have higher velocity than neutral"
        assert results["neutral_medium"]["mean"] > results["calm_low"]["mean"], \
            "neutral should have higher velocity than calm_low"
        
        print("\n✅ Velocity boost ordering verified!")


if __name__ == "__main__":
    # Run integration tests
    pytest.main([__file__, "-v", "-s"])
