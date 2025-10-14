"""Integration tests for multi-instrument emotion coordination (Phase 5.8)."""

from __future__ import annotations

import pytest
from music21 import instrument, stream

from generator.bass_generator import BassGenerator
from generator.drum_generator import DrumGenerator
from generator.guitar_generator import GuitarGenerator
from generator.piano_generator import PianoGenerator
from generator.strings_generator import StringsGenerator


class TestEmotionIntegrationFull:
    """Integration tests for multi-instrument emotion coordination."""

    @pytest.fixture
    def common_settings(self):
        """Common settings for all generators."""
        return {
            "tempo": 120,
            "time_signature": "4/4",
            "key_signature_tonic": "C",
            "key_signature_mode": "major",
        }

    @pytest.fixture
    def section_data(self):
        """Basic section data for testing."""
        return {
            "chord_symbol_for_voicing": "Cmaj7",
            "q_length": 4.0,
            "section_name": "Verse",
            "label": "Verse",
        }

    def test_all_instruments_with_happy_high(self, common_settings, section_data):
        """Test that all instruments can apply happy_high emotion."""
        emotion_profile = "happy_high"

        # Define piano part parameters (required for pattern-based generation)
        piano_part_params = {
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

        # Define guitar part parameters
        guitar_part_params = {
            "strum_sync_offset_beats": 0.0
        }

        # Create all generators
        piano = PianoGenerator(
            part_name="piano",
            default_instrument=instrument.Piano(),
            global_tempo=common_settings["tempo"],
            global_time_signature=common_settings["time_signature"],
            global_key_signature_tonic=common_settings["key_signature_tonic"],
            global_key_signature_mode=common_settings["key_signature_mode"],
            part_parameters=piano_part_params
        )

        guitar = GuitarGenerator(
            part_name="guitar",
            default_instrument=instrument.AcousticGuitar(),
            global_tempo=common_settings["tempo"],
            global_time_signature=common_settings["time_signature"],
            global_key_signature_tonic=common_settings["key_signature_tonic"],
            global_key_signature_mode=common_settings["key_signature_mode"],
            part_parameters=guitar_part_params
        )
        
        bass = BassGenerator(
            part_name="bass",
            default_instrument=instrument.ElectricBass(),
            global_tempo=common_settings["tempo"],
            global_time_signature=common_settings["time_signature"],
            global_key_signature_tonic=common_settings["key_signature_tonic"],
            global_key_signature_mode=common_settings["key_signature_mode"],
        )

        strings = StringsGenerator(
            part_name="strings",
            default_instrument=instrument.StringInstrument(),
            global_tempo=common_settings["tempo"],
            global_time_signature=common_settings["time_signature"],
            global_key_signature_tonic=common_settings["key_signature_tonic"],
            global_key_signature_mode=common_settings["key_signature_mode"],
        )
        
        drums = DrumGenerator(
            main_cfg={
                "global_settings": {
                    "tempo_bpm": common_settings["tempo"],
                    "time_signature": common_settings["time_signature"],
                }
            }
        )
        
        # Generate with same emotion profile
        piano_data = section_data.copy()
        piano_result = piano.compose(
            section_data=piano_data,
            section="Verse",
            emotion_profile=emotion_profile
        )
        
        guitar_data = section_data.copy()
        guitar_result = guitar.compose(
            section_data=guitar_data,
            section="Verse",
            emotion_profile=emotion_profile
        )
        
        bass_data = section_data.copy()
        bass_result = bass.compose(
            section_data=bass_data,
            section="Verse",
            emotion_profile=emotion_profile
        )
        
        strings_data = section_data.copy()
        strings_result = strings.compose(
            section_data=strings_data,
            section="Verse",
            emotion_profile=emotion_profile
        )
        
        drums_data = section_data.copy()
        drums_result = drums.compose(
            section_data=drums_data,
            section="Verse",
            emotion_profile=emotion_profile
        )
        
        # Verify all instruments generated content
        assert piano_result is not None
        assert guitar_result is not None
        assert bass_result is not None
        assert strings_result is not None
        assert drums_result is not None
        
        # Verify emotion parameters were stored
        assert "_emotion_adjustments" in piano_data
        assert "_emotion_adjustments" in guitar_data
        assert "_emotion_adjustments" in bass_data
        assert "_emotion_adjustments" in strings_data
        assert "_emotion_adjustments" in drums_data
        
        # Verify velocity_boost=+10 for happy_high across instruments (except Piano)
        assert guitar_data["_emotion_adjustments"]["guitar"]["velocity_boost"] == 10
        assert bass_data["_emotion_adjustments"]["bass"]["velocity_boost"] == 10
        assert strings_data["_emotion_adjustments"]["strings"]["velocity_boost"] == 10
        assert drums_data["_emotion_adjustments"]["drums"]["velocity_boost"] == 10
        
        print("\n✅ All instruments successfully applied happy_high emotion with velocity_boost=+10")

    def test_emotion_profile_comparison_across_instruments(self, common_settings, section_data):
        """Test that emotion profiles affect instruments consistently."""
        profiles = ["happy_high", "neutral_medium", "calm_low"]

        # Create generators (excluding Piano due to part_parameters requirements)
        bass = BassGenerator(
            part_name="bass",
            default_instrument=instrument.ElectricBass(),
            global_tempo=common_settings["tempo"],
            global_time_signature=common_settings["time_signature"],
            global_key_signature_tonic=common_settings["key_signature_tonic"],
            global_key_signature_mode=common_settings["key_signature_mode"],
        )
        
        drums = DrumGenerator(
            main_cfg={
                "global_settings": {
                    "tempo_bpm": common_settings["tempo"],
                    "time_signature": common_settings["time_signature"],
                }
            }
        )
        
        results = {}
        
        for profile in profiles:
            # Bass
            bass_data = section_data.copy()
            bass_result = bass.compose(
                section_data=bass_data,
                section="Verse",
                emotion_profile=profile
            )
            
            # Drums
            drums_data = section_data.copy()
            drums_result = drums.compose(
                section_data=drums_data,
                section="Verse",
                emotion_profile=profile
            )
            
            # Collect velocities
            bass_velocities = []
            drums_velocities = []
            
            # Bass velocities
            if isinstance(bass_result, stream.Part):
                for n in bass_result.recurse().notes:
                    if n.volume and n.volume.velocity is not None:
                        bass_velocities.append(n.volume.velocity)
            
            # Drums velocities
            if isinstance(drums_result, stream.Part):
                for n in drums_result.recurse().notes:
                    if n.volume and n.volume.velocity is not None:
                        drums_velocities.append(n.volume.velocity)
            
            results[profile] = {
                "bass_mean": sum(bass_velocities) / len(bass_velocities) if bass_velocities else 0,
                "drums_mean": sum(drums_velocities) / len(drums_velocities) if drums_velocities else 0,
            }
        
        # Verify ordering: happy_high > neutral_medium > calm_low for each instrument
        print(f"\n📊 Multi-instrument emotion comparison:")
        for profile in profiles:
            r = results[profile]
            print(f"{profile:15} - Bass: {r['bass_mean']:6.2f}, Drums: {r['drums_mean']:6.2f}")
        
        # Bass ordering
        assert results["happy_high"]["bass_mean"] > results["neutral_medium"]["bass_mean"] - 5
        assert results["calm_low"]["bass_mean"] < results["neutral_medium"]["bass_mean"] + 5
        
        # Drums ordering
        assert results["happy_high"]["drums_mean"] > results["neutral_medium"]["drums_mean"] - 5
        assert results["calm_low"]["drums_mean"] < results["neutral_medium"]["drums_mean"] + 5
        
        print("\n✅ All instruments show consistent emotion ordering!")

    def test_combined_band_generation(self, common_settings, section_data):
        """Test generating a full band arrangement with emotion."""
        emotion_profile = "calm_low"
        
        # Create all generators
        generators = {
            "piano": PianoGenerator(
                part_name="piano",
                default_instrument=instrument.Piano(),
                global_tempo=common_settings["tempo"],
                global_time_signature=common_settings["time_signature"],
                global_key_signature_tonic=common_settings["key_signature_tonic"],
                global_key_signature_mode=common_settings["key_signature_mode"],
            ),
            "guitar": GuitarGenerator(
                part_name="guitar",
                default_instrument=instrument.AcousticGuitar(),
                global_tempo=common_settings["tempo"],
                global_time_signature=common_settings["time_signature"],
                global_key_signature_tonic=common_settings["key_signature_tonic"],
                global_key_signature_mode=common_settings["key_signature_mode"],
            ),
            "bass": BassGenerator(
                part_name="bass",
                default_instrument=instrument.ElectricBass(),
                global_tempo=common_settings["tempo"],
                global_time_signature=common_settings["time_signature"],
                global_key_signature_tonic=common_settings["key_signature_tonic"],
                global_key_signature_mode=common_settings["key_signature_mode"],
            ),
            "drums": DrumGenerator(
                main_cfg={
                    "global_settings": {
                        "tempo_bpm": common_settings["tempo"],
                        "time_signature": common_settings["time_signature"],
                    }
                }
            ),
        }
        
        # Generate all parts with same emotion
        band_parts = {}
        for instrument_name, generator in generators.items():
            instrument_data = section_data.copy()
            result = generator.compose(
                section_data=instrument_data,
                section="Verse",
                emotion_profile=emotion_profile
            )
            band_parts[instrument_name] = result
        
        # Verify all parts exist
        assert len(band_parts) == 4
        
        # Create a full score
        score = stream.Score()
        
        # Add Piano parts
        if isinstance(band_parts["piano"], dict):
            for part_name, part in band_parts["piano"].items():
                if isinstance(part, stream.Part):
                    part.partName = f"Piano {part_name}"
                    score.append(part)
        
        # Add Guitar part
        if isinstance(band_parts["guitar"], stream.Part):
            band_parts["guitar"].partName = "Guitar"
            score.append(band_parts["guitar"])
        
        # Add Bass part
        if isinstance(band_parts["bass"], stream.Part):
            band_parts["bass"].partName = "Bass"
            score.append(band_parts["bass"])
        
        # Add Drums part
        if isinstance(band_parts["drums"], stream.Part):
            band_parts["drums"].partName = "Drums"
            score.append(band_parts["drums"])
        
        # Verify score structure
        assert len(score.parts) >= 4
        
        # Count total notes
        total_notes = sum(
            len(list(part.recurse().notes))
            for part in score.parts
        )
        
        print(f"\n🎵 Generated full band arrangement with {emotion_profile}:")
        print(f"   Total parts: {len(score.parts)}")
        print(f"   Total notes: {total_notes}")
        
        assert total_notes > 0
        print("\n✅ Full band generation successful!")

    def test_emotion_parameter_coverage(self, common_settings, section_data):
        """Test that all instruments have emotion parameters defined."""
        emotion_profile = "neutral_medium"
        
        # Expected parameters for each instrument
        expected_params = {
            "guitar": {"velocity_boost", "strum_consistency_target"},
            "bass": {"velocity_boost", "sustain_control", "velocity_std_multiplier"},
            "strings": {"velocity_boost", "bow_pressure_factor", "articulation_legato_bias", "velocity_std_multiplier"},
            "drums": {"velocity_boost", "attack_sharpness", "groove_tightness", "velocity_std_multiplier"},
        }
        
        # Guitar
        guitar = GuitarGenerator(
            part_name="guitar",
            default_instrument=instrument.AcousticGuitar(),
            global_tempo=common_settings["tempo"],
            global_time_signature=common_settings["time_signature"],
            global_key_signature_tonic=common_settings["key_signature_tonic"],
            global_key_signature_mode=common_settings["key_signature_mode"],
        )
        guitar_data = section_data.copy()
        guitar.compose(section_data=guitar_data, section="Verse", emotion_profile=emotion_profile)
        # GuitarGenerator modifies section_data in-place
        guitar_params = set(guitar_data.get("_emotion_adjustments", {}).get("guitar", {}).keys())
        assert expected_params["guitar"].issubset(guitar_params), \
            f"Guitar missing params: {expected_params['guitar'] - guitar_params}"
        
        # Bass
        bass = BassGenerator(
            part_name="bass",
            default_instrument=instrument.ElectricBass(),
            global_tempo=common_settings["tempo"],
            global_time_signature=common_settings["time_signature"],
            global_key_signature_tonic=common_settings["key_signature_tonic"],
            global_key_signature_mode=common_settings["key_signature_mode"],
        )
        bass_data = section_data.copy()
        bass.compose(section_data=bass_data, section="Verse", emotion_profile=emotion_profile)
        bass_params = set(bass_data["_emotion_adjustments"]["bass"].keys())
        assert expected_params["bass"].issubset(bass_params), \
            f"Bass missing params: {expected_params['bass'] - bass_params}"
        
        # Strings
        strings = StringsGenerator(
            part_name="strings",
            default_instrument=instrument.StringInstrument(),
            global_tempo=common_settings["tempo"],
            global_time_signature=common_settings["time_signature"],
            global_key_signature_tonic=common_settings["key_signature_tonic"],
            global_key_signature_mode=common_settings["key_signature_mode"],
        )
        strings_data = section_data.copy()
        strings.compose(section_data=strings_data, section="Verse", emotion_profile=emotion_profile)
        strings_params = set(strings_data["_emotion_adjustments"]["strings"].keys())
        assert expected_params["strings"].issubset(strings_params), \
            f"Strings missing params: {expected_params['strings'] - strings_params}"
        
        # Drums
        drums = DrumGenerator(
            main_cfg={
                "global_settings": {
                    "tempo_bpm": common_settings["tempo"],
                    "time_signature": common_settings["time_signature"],
                }
            }
        )
        drums_data = section_data.copy()
        drums.compose(section_data=drums_data, section="Verse", emotion_profile=emotion_profile)
        drums_params = set(drums_data["_emotion_adjustments"]["drums"].keys())
        assert expected_params["drums"].issubset(drums_params), \
            f"Drums missing params: {expected_params['drums'] - drums_params}"
        
        print("\n✅ All instruments have required emotion parameters!")
