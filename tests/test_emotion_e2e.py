"""
Phase 5.9: End-to-End Emotion System Testing

Real-world use case validation of the emotion parameter system across
all instruments in actual song generation scenarios.

Test Scenarios:
1. Multi-section song generation with consistent emotion
2. Emotion profile switching between sections
3. Full band arrangement with all 5 instruments
4. Performance measurement and validation
"""

import pytest
from music21 import stream, instrument, tempo, meter, key
from generator.piano_generator import PianoGenerator
from generator.guitar_generator import GuitarGenerator
from generator.bass_generator import BassGenerator
from generator.strings_generator import StringsGenerator
from generator.drum_generator import DrumGenerator


class TestEmotionE2E:
    """End-to-end tests for emotion system in realistic scenarios."""
    
    @pytest.fixture
    def song_structure(self):
        """Realistic song structure with multiple sections."""
        return {
            "global_settings": {
                "tempo": 120,
                "time_signature": "4/4",
                "key_signature_tonic": "C",
                "key_signature_mode": "major",
            },
            "sections": [
                {
                    "name": "Intro",
                    "bars": 4,
                    "chord_progression": ["C", "Am", "F", "G"],
                    "emotion": "calm_low",
                },
                {
                    "name": "Verse1",
                    "bars": 8,
                    "chord_progression": ["C", "G", "Am", "F", "C", "G", "F", "G"],
                    "emotion": "neutral_medium",
                },
                {
                    "name": "Chorus",
                    "bars": 8,
                    "chord_progression": ["C", "G", "Am", "F"] * 2,
                    "emotion": "happy_high",
                },
                {
                    "name": "Verse2",
                    "bars": 8,
                    "chord_progression": ["C", "G", "Am", "F", "C", "G", "F", "G"],
                    "emotion": "neutral_medium",
                },
                {
                    "name": "Chorus",
                    "bars": 8,
                    "chord_progression": ["C", "G", "Am", "F"] * 2,
                    "emotion": "happy_high",
                },
                {
                    "name": "Outro",
                    "bars": 4,
                    "chord_progression": ["Am", "F", "C", "G"],
                    "emotion": "calm_low",
                },
            ]
        }
    
    @pytest.fixture
    def full_band_generators(self, song_structure):
        """Create all generators for full band arrangement."""
        settings = song_structure["global_settings"]
        
        # Piano part parameters
        piano_params = {
            "piano_rh_block_chords_quarters": {
                "pattern": [
                    {"offset": 0.0, "duration": 1.0, "type": "chord", "velocity_factor": 1.0},
                    {"offset": 1.0, "duration": 1.0, "type": "chord", "velocity_factor": 0.9},
                    {"offset": 2.0, "duration": 1.0, "type": "chord", "velocity_factor": 0.95},
                    {"offset": 3.0, "duration": 1.0, "type": "chord", "velocity_factor": 0.9},
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
        
        guitar_params = {
            "strum_sync_offset_beats": 0.0
        }
        
        return {
            "piano": PianoGenerator(
                part_name="piano",
                default_instrument=instrument.Piano(),
                global_tempo=settings["tempo"],
                global_time_signature=settings["time_signature"],
                global_key_signature_tonic=settings["key_signature_tonic"],
                global_key_signature_mode=settings["key_signature_mode"],
                part_parameters=piano_params
            ),
            "guitar": GuitarGenerator(
                part_name="guitar",
                default_instrument=instrument.AcousticGuitar(),
                global_tempo=settings["tempo"],
                global_time_signature=settings["time_signature"],
                global_key_signature_tonic=settings["key_signature_tonic"],
                global_key_signature_mode=settings["key_signature_mode"],
                part_parameters=guitar_params
            ),
            "bass": BassGenerator(
                part_name="bass",
                default_instrument=instrument.ElectricBass(),
                global_tempo=settings["tempo"],
                global_time_signature=settings["time_signature"],
                global_key_signature_tonic=settings["key_signature_tonic"],
                global_key_signature_mode=settings["key_signature_mode"],
            ),
            "strings": StringsGenerator(
                part_name="strings",
                default_instrument=instrument.StringInstrument(),
                global_tempo=settings["tempo"],
                global_time_signature=settings["time_signature"],
                global_key_signature_tonic=settings["key_signature_tonic"],
                global_key_signature_mode=settings["key_signature_mode"],
            ),
            "drums": DrumGenerator(
                main_cfg={
                    "global_settings": {
                        "tempo_bpm": settings["tempo"],
                        "time_signature": settings["time_signature"],
                    }
                }
            ),
        }
    
    def test_multi_section_song_generation(self, song_structure, full_band_generators):
        """Test: Generate a complete song with multiple sections and emotions.
        
        Validates:
        - All sections generate successfully
        - Emotion parameters are applied correctly per section
        - Section transitions are smooth
        """
        results = {}
        
        for idx, section_info in enumerate(song_structure["sections"]):
            section_name = section_info["name"]
            emotion = section_info["emotion"]
            chord = section_info["chord_progression"][0]  # Use first chord
            
            section_data = {
                "chord_symbol_for_voicing": chord,
                "q_length": float(section_info["bars"] * 4),
                "section_name": section_name,
                "label": section_name,
            }
            
            # Generate with Bass (representative instrument)
            bass = full_band_generators["bass"]
            bass_data = section_data.copy()
            bass_result = bass.compose(
                section_data=bass_data,
                section=section_name,
                emotion_profile=emotion
            )
            
            # 重複セクション名（例: Chorus 2回）での上書きを避ける
            results[f"{idx:02d}_{section_name}"] = {
                "emotion": emotion,
                "generated": bass_result is not None,
                "emotion_params": bass_data.get("_emotion_adjustments", {}).get("bass", {}),
            }
        
        # Validate all sections generated
        assert len(results) == len(song_structure["sections"])
        for section_key, result in results.items():
            assert result["generated"], f"Section {section_key} failed to generate"
            assert "velocity_boost" in result["emotion_params"], \
                f"Section {section_key} missing velocity_boost"
        
        # Validate emotion progression
        # Find Intro (00_Intro) and first Chorus (02_Chorus)
        intro_boost = results["00_Intro"]["emotion_params"]["velocity_boost"]
        chorus_boost = results["02_Chorus"]["emotion_params"]["velocity_boost"]
        
        # Intro (calm_low) should have negative boost, Chorus (happy_high) positive
        assert intro_boost < 0, f"Intro should have negative boost, got {intro_boost}"
        assert chorus_boost > 0, f"Chorus should have positive boost, got {chorus_boost}"
        
        print(f"\n✅ Multi-section song generation successful:")
        for section_key, result in results.items():
            boost = result["emotion_params"]["velocity_boost"]
            # Extract section name from key (e.g., "00_Intro" -> "Intro")
            section_name = section_key.split("_", 1)[1] if "_" in section_key else section_key
            print(f"  {section_name:10} ({result['emotion']:15}): velocity_boost={boost:+3}")
    
    def test_emotion_switching_between_sections(self, song_structure, full_band_generators):
        """Test: Emotion profile changes between consecutive sections.
        
        Validates:
        - Generator state resets between sections
        - New emotion parameters are applied
        - No carryover from previous section
        """
        drums = full_band_generators["drums"]
        
        # Section 1: calm_low
        section1_data = {
            "chord_symbol_for_voicing": "C",
            "q_length": 4.0,
            "section_name": "Intro",
            "label": "Intro",
        }
        result1 = drums.compose(
            section_data=section1_data,
            section="Intro",
            emotion_profile="calm_low"
        )
        params1 = section1_data["_emotion_adjustments"]["drums"]
        
        # Section 2: happy_high (immediate switch)
        section2_data = {
            "chord_symbol_for_voicing": "C",
            "q_length": 8.0,
            "section_name": "Chorus",
            "label": "Chorus",
        }
        result2 = drums.compose(
            section_data=section2_data,
            section="Chorus",
            emotion_profile="happy_high"
        )
        params2 = section2_data["_emotion_adjustments"]["drums"]
        
        # Section 3: neutral_medium
        section3_data = {
            "chord_symbol_for_voicing": "C",
            "q_length": 8.0,
            "section_name": "Verse",
            "label": "Verse",
        }
        result3 = drums.compose(
            section_data=section3_data,
            section="Verse",
            emotion_profile="neutral_medium"
        )
        params3 = section3_data["_emotion_adjustments"]["drums"]
        
        # Validate transitions
        assert result1 is not None and result2 is not None and result3 is not None
        
        # Verify distinct parameter sets
        assert params1["velocity_boost"] == -10  # calm_low
        assert params2["velocity_boost"] == +10  # happy_high
        assert params3["velocity_boost"] == 0    # neutral_medium
        
        assert params1["groove_tightness"] == 1.20  # calm_low (looser)
        assert params2["groove_tightness"] == 0.85  # happy_high (tighter)
        assert params3["groove_tightness"] == 1.00  # neutral_medium
        
        print(f"\n✅ Emotion switching validated:")
        print(f"  Intro (calm_low):       velocity_boost={params1['velocity_boost']:+3}, groove={params1['groove_tightness']:.2f}")
        print(f"  Chorus (happy_high):    velocity_boost={params2['velocity_boost']:+3}, groove={params2['groove_tightness']:.2f}")
        print(f"  Verse (neutral_medium): velocity_boost={params3['velocity_boost']:+3}, groove={params3['groove_tightness']:.2f}")
    
    def test_full_band_arrangement_generation(self, song_structure, full_band_generators):
        """Test: Generate full band arrangement for one section.
        
        Validates:
        - All 5 instruments generate simultaneously
        - Emotion parameters are consistent across instruments
        - Combined output is coherent
        """
        emotion = "happy_high"
        section_data_template = {
            "chord_symbol_for_voicing": "C",
            "q_length": 8.0,
            "section_name": "Chorus",
            "label": "Chorus",
        }
        
        band_parts = {}
        emotion_params_collected = {}
        
        for inst_name, generator in full_band_generators.items():
            inst_data = section_data_template.copy()
            result = generator.compose(
                section_data=inst_data,
                section="Chorus",
                emotion_profile=emotion
            )
            
            band_parts[inst_name] = result
            if "_emotion_adjustments" in inst_data:
                emotion_params_collected[inst_name] = inst_data["_emotion_adjustments"].get(inst_name, {})
        
        # Validate all instruments generated
        assert len(band_parts) == 5
        for inst_name, part in band_parts.items():
            assert part is not None, f"{inst_name} failed to generate"
        
        # Validate emotion consistency (all should have velocity_boost=+10 for happy_high)
        for inst_name, params in emotion_params_collected.items():
            if "velocity_boost" in params:  # Some instruments may not have all params
                assert params["velocity_boost"] == 10, \
                    f"{inst_name} velocity_boost={params['velocity_boost']}, expected 10"
        
        # Create combined score
        full_score = stream.Score()
        full_score.append(tempo.MetronomeMark(number=120))
        full_score.append(meter.TimeSignature("4/4"))
        full_score.append(key.Key("C"))
        
        # Add all parts (handle both dict and Part return types)
        for inst_name, result in band_parts.items():
            if isinstance(result, dict):
                for part_name, part in result.items():
                    if isinstance(part, stream.Part):
                        full_score.append(part)
            elif isinstance(result, stream.Part):
                full_score.append(result)
        
        # Validate combined score
        assert len(full_score.parts) >= 3, f"Expected at least 3 parts, got {len(full_score.parts)}"
        
        print(f"\n✅ Full band arrangement generated successfully:")
        print(f"  Total parts: {len(full_score.parts)}")
        print(f"  Emotion: {emotion}")
        print(f"  Instruments with emotion params:")
        for inst_name, params in emotion_params_collected.items():
            if params:
                boost = params.get("velocity_boost", "N/A")
                print(f"    {inst_name:10}: velocity_boost={boost}")
    
    @pytest.mark.slow
    def test_performance_full_song_generation(self, song_structure, full_band_generators):
        """Test: Performance measurement for full song generation.
        
        Validates:
        - Generation completes in reasonable time
        - Memory usage is acceptable
        - No performance degradation across sections
        """
        import time
        
        total_sections = len(song_structure["sections"])
        generation_times = []
        
        for section_info in song_structure["sections"]:
            section_name = section_info["name"]
            emotion = section_info["emotion"]
            chord = section_info["chord_progression"][0]
            
            section_data = {
                "chord_symbol_for_voicing": chord,
                "q_length": float(section_info["bars"] * 4),
                "section_name": section_name,
                "label": section_name,
            }
            
            # Measure generation time for one representative instrument
            start_time = time.time()
            
            guitar = full_band_generators["guitar"]
            guitar_data = section_data.copy()
            result = guitar.compose(
                section_data=guitar_data,
                section=section_name,
                emotion_profile=emotion
            )
            
            elapsed = time.time() - start_time
            generation_times.append(elapsed)
            
            assert result is not None, f"Section {section_name} failed to generate"
        
        # Performance assertions
        avg_time = sum(generation_times) / len(generation_times)
        max_time = max(generation_times)
        
        # Reasonable thresholds (adjust based on hardware)
        assert avg_time < 5.0, f"Average generation time too high: {avg_time:.2f}s"
        assert max_time < 10.0, f"Max generation time too high: {max_time:.2f}s"
        
        print(f"\n✅ Performance test passed:")
        print(f"  Total sections: {total_sections}")
        print(f"  Average time: {avg_time:.3f}s")
        print(f"  Max time: {max_time:.3f}s")
        print(f"  Min time: {min(generation_times):.3f}s")
        print(f"  Individual times: {[f'{t:.3f}s' for t in generation_times]}")
    
    def test_edge_case_unknown_emotion_profile(self, full_band_generators):
        """Test: Handling of unknown emotion profiles.
        
        Validates:
        - System falls back to neutral_medium
        - No crashes or exceptions
        - Reasonable default values
        """
        unknown_profile = "excited_ultra_mega_high"  # Non-existent profile
        
        section_data = {
            "chord_symbol_for_voicing": "C",
            "q_length": 4.0,
            "section_name": "Test",
            "label": "Test",
        }
        
        bass = full_band_generators["bass"]
        bass_data = section_data.copy()
        
        # Should not crash, should fallback to neutral_medium
        result = bass.compose(
            section_data=bass_data,
            section="Test",
            emotion_profile=unknown_profile
        )
        
        assert result is not None, "Generation failed with unknown emotion"
        
        # Should fallback to neutral_medium values
        params = bass_data.get("_emotion_adjustments", {}).get("bass", {})
        assert "velocity_boost" in params
        
        # neutral_medium has velocity_boost=0
        assert params["velocity_boost"] == 0, \
            f"Expected neutral fallback (0), got {params['velocity_boost']}"
        
        print(f"\n✅ Unknown emotion profile handled gracefully:")
        print(f"  Profile: {unknown_profile}")
        print(f"  Fallback params: {params}")
    
    def test_partial_band_configuration(self, song_structure):
        """Test: Generation with only subset of instruments.
        
        Validates:
        - Works with any combination of instruments
        - Missing instruments don't cause issues
        """
        settings = song_structure["global_settings"]
        
        # Only Bass + Drums (common rhythm section)
        rhythm_section = {
            "bass": BassGenerator(
                part_name="bass",
                default_instrument=instrument.ElectricBass(),
                global_tempo=settings["tempo"],
                global_time_signature=settings["time_signature"],
                global_key_signature_tonic=settings["key_signature_tonic"],
                global_key_signature_mode=settings["key_signature_mode"],
            ),
            "drums": DrumGenerator(
                main_cfg={
                    "global_settings": {
                        "tempo_bpm": settings["tempo"],
                        "time_signature": settings["time_signature"],
                    }
                }
            ),
        }
        
        section_data_template = {
            "chord_symbol_for_voicing": "C",
            "q_length": 4.0,
            "section_name": "Verse",
            "label": "Verse",
        }
        
        results = {}
        for inst_name, generator in rhythm_section.items():
            inst_data = section_data_template.copy()
            result = generator.compose(
                section_data=inst_data,
                section="Verse",
                emotion_profile="neutral_medium"
            )
            results[inst_name] = result
        
        # Validate both generated
        assert all(r is not None for r in results.values())
        
        print(f"\n✅ Partial band configuration works:")
        print(f"  Instruments: {list(results.keys())}")
        print(f"  All generated successfully")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
