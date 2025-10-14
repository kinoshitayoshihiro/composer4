"""
Phase 5.1: Piano Emotion Application Integration Tests
compose()メソッド全体を使った統合テスト
"""

import pytest
import numpy as np
from generator.piano_generator import PianoGenerator
from music21 import stream, instrument


class TestPianoEmotionIntegration:
    """Piano emotion parameter application integration tests"""
    
    def test_compose_with_emotion_happy_high(self):
        """compose()でhappy_high感情プロファイルが正しく適用されることを確認"""
        # Load part parameters from file or use minimal config
        part_params = {
            "piano_rh_block_chords_quarters": {
                "pattern": [
                    {"offset": 0.0, "duration": 1.0, "type": "chord", "velocity_factor": 1.0},
                    {"offset": 1.0, "duration": 1.0, "type": "chord", "velocity_factor": 1.0},
                    {"offset": 2.0, "duration": 1.0, "type": "chord", "velocity_factor": 1.0},
                    {"offset": 3.0, "duration": 1.0, "type": "chord", "velocity_factor": 1.0},
                ],
                "length_beats": 4.0
            },
            "piano_lh_roots_half": {
                "pattern": [
                    {"offset": 0.0, "duration": 2.0, "type": "root", "velocity_factor": 0.8},
                    {"offset": 2.0, "duration": 2.0, "type": "root", "velocity_factor": 0.8},
                ],
                "length_beats": 4.0
            }
        }
        
        gen = PianoGenerator(
            default_instrument=instrument.Piano(),
            global_tempo=120,
            global_time_signature="4/4",
            part_parameters=part_params
        )
        
        # Section data with happy_high emotion
        section_data = {
            "chord_symbol_for_voicing": "C",
            "q_length": 4.0,
            "section_name": "Chorus",
            "musical_intent": {},
            "part_params": {
                "piano": {
                    "velocity": 80,
                    "rhythm_key_rh": "piano_rh_block_chords_quarters",
                    "rhythm_key_lh": "piano_lh_roots_half"
                }
            }
        }
        
        # Compose with emotion
        result = gen.compose(
            section_data=section_data,
            section="Chorus",
            emotion_profile="happy_high"
        )
        
        # Verify result structure
        assert isinstance(result, dict), "Result should be a dictionary"
        assert "piano_rh" in result, "Result should contain piano_rh"
        assert "piano_lh" in result, "Result should contain piano_lh"
        
        # Verify emotion adjustments were stored
        assert "_emotion_adjustments" in section_data, "Emotion adjustments should be stored"
        assert "piano" in section_data["_emotion_adjustments"], "Piano adjustments should exist"
        
        # Verify both parts have notes
        rh_notes = list(result["piano_rh"].flatten().notes)
        lh_notes = list(result["piano_lh"].flatten().notes)
        
        assert len(rh_notes) > 0, "RH should have notes"
        assert len(lh_notes) > 0, "LH should have notes"
        
        print(f"\n✅ Compose with happy_high:")
        print(f"  RH notes: {len(rh_notes)}")
        print(f"  LH notes: {len(lh_notes)}")
        print(f"  Emotion adjustments stored: {section_data['_emotion_adjustments']['piano'].keys()}")
    
    def test_compose_emotion_comparison(self):
        """異なる感情プロファイルでの生成を比較"""
        part_params = {
            "piano_rh_block_chords_quarters": {
                "pattern": [
                    {"offset": i * 1.0, "duration": 1.0, "type": "chord", "velocity_factor": 1.0}
                    for i in range(4)
                ],
                "length_beats": 4.0
            },
            "piano_lh_roots_half": {
                "pattern": [
                    {"offset": 0.0, "duration": 2.0, "type": "root", "velocity_factor": 0.8},
                    {"offset": 2.0, "duration": 2.0, "type": "root", "velocity_factor": 0.8},
                ],
                "length_beats": 4.0
            }
        }
        
        gen = PianoGenerator(
            default_instrument=instrument.Piano(),
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
                    "piano": {
                        "velocity": 70,
                        "rhythm_key_rh": "piano_rh_block_chords_quarters",
                        "rhythm_key_lh": "piano_lh_roots_half"
                    }
                }
            }
            
            # Generate multiple samples for statistics
            velocities = []
            note_counts = []
            
            for _ in range(10):
                result = gen.compose(
                    section_data=section_data,
                    section="Verse",
                    emotion_profile=emotion
                )
                
                # Extract RH velocities and note counts
                rh_notes = list(result["piano_rh"].flatten().notes)
                velocities.extend([n.volume.velocity for n in rh_notes])
                note_counts.append(len(rh_notes))
            
            results[emotion] = {
                "velocity_mean": np.mean(velocities),
                "velocity_std": np.std(velocities),
                "note_count_mean": np.mean(note_counts)
            }
        
        print("\n=== Emotion Profile Comparison ===")
        for emotion, metrics in results.items():
            print(f"\n{emotion}:")
            print(f"  Velocity Mean: {metrics['velocity_mean']:.2f}")
            print(f"  Velocity STD: {metrics['velocity_std']:.2f}")
            print(f"  Note Count: {metrics['note_count_mean']:.2f}")
        
        # Verify happy_high has higher velocity variation than calm_low
        assert results["happy_high"]["velocity_std"] > results["calm_low"]["velocity_std"], \
            "happy_high should have more velocity variation than calm_low"
        
        # Verify calm_low has fewer notes than happy_high
        assert results["calm_low"]["note_count_mean"] < results["happy_high"]["note_count_mean"], \
            "calm_low should have fewer notes than happy_high"
        
        print("\n✅ Emotion comparison successful!")
    
    def test_compose_backward_compatibility(self):
        """emotion指定なしでも正常に動作することを確認（後方互換性）"""
        part_params = {
            "piano_rh_block_chords_quarters": {
                "pattern": [
                    {"offset": 0.0, "duration": 1.0, "type": "chord"},
                    {"offset": 1.0, "duration": 1.0, "type": "chord"},
                ],
                "length_beats": 4.0
            },
            "piano_lh_roots_half": {
                "pattern": [
                    {"offset": 0.0, "duration": 2.0, "type": "root"},
                ],
                "length_beats": 4.0
            }
        }
        
        gen = PianoGenerator(
            default_instrument=instrument.Piano(),
            global_tempo=120,
            part_parameters=part_params
        )
        
        section_data = {
            "chord_symbol_for_voicing": "C",
            "q_length": 4.0,
            "section_name": "Verse",
            "musical_intent": {},
            "part_params": {
                "piano": {
                    "velocity": 70,
                    "rhythm_key_rh": "piano_rh_block_chords_quarters",
                    "rhythm_key_lh": "piano_lh_roots_half"
                }
            }
        }
        
        # Compose without emotion (backward compatibility)
        result = gen.compose(section_data=section_data)
        
        # Should work without error
        assert isinstance(result, dict), "Result should be a dictionary"
        assert "piano_rh" in result, "Result should contain piano_rh"
        assert "piano_lh" in result, "Result should contain piano_lh"
        
        # Should have notes
        rh_notes = list(result["piano_rh"].flatten().notes)
        lh_notes = list(result["piano_lh"].flatten().notes)
        
        assert len(rh_notes) > 0, "RH should have notes"
        assert len(lh_notes) > 0, "LH should have notes"
        
        print(f"\n✅ Backward compatibility test passed")
        print(f"  Generated {len(rh_notes)} RH notes, {len(lh_notes)} LH notes")
    
    def test_compose_with_all_emotion_profiles(self):
        """全ての感情プロファイルで正常に生成できることを確認"""
        part_params = {
            "piano_rh_block_chords_quarters": {
                "pattern": [
                    {"offset": i * 1.0, "duration": 1.0, "type": "chord"}
                    for i in range(4)
                ],
                "length_beats": 4.0
            },
            "piano_lh_roots_half": {
                "pattern": [
                    {"offset": 0.0, "duration": 2.0, "type": "root"},
                    {"offset": 2.0, "duration": 2.0, "type": "root"},
                ],
                "length_beats": 4.0
            }
        }
        
        gen = PianoGenerator(
            default_instrument=instrument.Piano(),
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
                    "piano": {
                        "velocity": 70,
                        "rhythm_key_rh": "piano_rh_block_chords_quarters",
                        "rhythm_key_lh": "piano_lh_roots_half"
                    }
                }
            }
            
            try:
                result = gen.compose(
                    section_data=section_data,
                    section="Verse",
                    emotion_profile=profile
                )
                
                rh_notes = list(result["piano_rh"].flatten().notes)
                results[profile] = {
                    "success": True,
                    "note_count": len(rh_notes),
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


if __name__ == "__main__":
    # Run integration tests
    pytest.main([__file__, "-v", "-s"])
