"""Tests for VPTT sample generation script."""

import tempfile
from pathlib import Path

import pytest
import yaml

from scripts.generate_vptt_samples import VPTTSampleGenerator


class TestVPTTSampleGenerator:
    """Test VPTTSampleGenerator class."""

    def test_generator_initialization(self):
        """Test generator initializes with correct seed."""
        gen = VPTTSampleGenerator(seed=42)
        assert gen.combinations is not None
        assert len(gen.combinations) == 54

    def test_generate_combinations(self):
        """Test that all combinations are generated correctly."""
        gen = VPTTSampleGenerator()
        combinations = gen._generate_combinations()
        
        # Should have 2 instruments × 3 techniques × 3 tempos × 3 dynamics = 54
        assert len(combinations) == 54
        
        # Check structure
        for combo in combinations:
            assert "instrument" in combo
            assert "technique" in combo
            assert "tempo" in combo
            assert "dynamic" in combo
            assert combo["instrument"] in ["piano", "violin"]
            assert combo["tempo"] in ["slow", "medium", "fast"]
            assert combo["dynamic"] in ["soft", "medium", "loud"]

    def test_sample_combinations(self):
        """Test that sampling returns correct number of samples."""
        gen = VPTTSampleGenerator(seed=42)
        samples = gen.sample_combinations(n=10)
        
        assert len(samples) == 10
        
        # Check that samples are unique
        sample_tuples = [
            (s["instrument"], s["technique"], s["tempo"], s["dynamic"])
            for s in samples
        ]
        assert len(set(sample_tuples)) == 10

    def test_sample_combinations_reproducible(self):
        """Test that same seed produces same samples."""
        gen1 = VPTTSampleGenerator(seed=42)
        samples1 = gen1.sample_combinations(n=10)
        
        gen2 = VPTTSampleGenerator(seed=42)
        samples2 = gen2.sample_combinations(n=10)
        
        assert samples1 == samples2

    def test_generate_midi_piano_staccato(self):
        """Test MIDI generation for piano staccato."""
        gen = VPTTSampleGenerator()
        spec = {
            "id": "vptt_test",
            "instrument": "piano",
            "technique": "staccato",
            "tempo": "medium",
            "tempo_bpm": 120,
            "dynamic": "medium",
            "velocity": 80,
            "marking": "mf",
        }
        
        # Generate MIDI object
        mid = gen.generate_midi(spec)
        
        # Verify MIDI file
        assert len(mid.tracks) == 1
        
        # Check tempo (500000 μs/beat = 120 BPM)
        tempo_msg = next(msg for msg in mid.tracks[0] if msg.type == "set_tempo")
        assert tempo_msg.tempo == 500000
        
        # Check program (piano = 0)
        program_msg = next(msg for msg in mid.tracks[0] if msg.type == "program_change")
        assert program_msg.program == 0
        
        # Check velocity
        note_on_msgs = [msg for msg in mid.tracks[0] if msg.type == "note_on"]
        assert all(msg.velocity == 80 for msg in note_on_msgs)

    def test_generate_midi_violin_pizzicato(self):
        """Test MIDI generation for violin pizzicato."""
        gen = VPTTSampleGenerator()
        spec = {
            "id": "vptt_test2",
            "instrument": "violin",
            "technique": "pizzicato",
            "tempo": "fast",
            "tempo_bpm": 180,
            "dynamic": "soft",
            "velocity": 45,
            "marking": "pp",
        }
        
        # Generate MIDI object
        mid = gen.generate_midi(spec)
        
        # Verify MIDI file
        assert len(mid.tracks) == 1
        
        # Check tempo (333333 μs/beat ≈ 180 BPM)
        tempo_msg = next(msg for msg in mid.tracks[0] if msg.type == "set_tempo")
        expected_tempo = int(60_000_000 / 180)
        assert tempo_msg.tempo == expected_tempo
        
        # Check program (violin = 40)
        program_msg = next(msg for msg in mid.tracks[0] if msg.type == "program_change")
        assert program_msg.program == 40
        
        # Check velocity (pizzicato adds +10)
        note_on_msgs = [msg for msg in mid.tracks[0] if msg.type == "note_on"]
        assert all(msg.velocity == 55 for msg in note_on_msgs)  # 45 + 10

    def test_generate_phrase_staccato(self):
        """Test phrase generation for staccato technique."""
        gen = VPTTSampleGenerator()
        spec = {
            "instrument": "piano",
            "technique": "staccato",
            "dynamic": "medium",
            "velocity": 80,
        }
        
        phrase = gen._generate_phrase(spec, ticks_per_beat=480)
        
        # Staccato should have short notes (1/8 note = 240 ticks)
        assert all(duration == 240 for _, _, duration in phrase)

    def test_generate_phrase_legato(self):
        """Test phrase generation for legato technique."""
        gen = VPTTSampleGenerator()
        spec = {
            "instrument": "piano",
            "technique": "legato",
            "dynamic": "medium",
            "velocity": 80,
        }
        
        phrase = gen._generate_phrase(spec, ticks_per_beat=480)
        
        # Legato should have quarter notes (480 ticks)
        assert all(duration == 480 for _, _, duration in phrase)

    def test_generate_phrase_sustain(self):
        """Test phrase generation for sustain technique."""
        gen = VPTTSampleGenerator()
        spec = {
            "instrument": "piano",
            "technique": "sustain",
            "dynamic": "medium",
            "velocity": 80,
        }
        
        phrase = gen._generate_phrase(spec, ticks_per_beat=480)
        
        # Sustain should have half notes (960 ticks)
        assert all(duration == 960 for _, _, duration in phrase)

    def test_generate_phrase_pizzicato(self):
        """Test phrase generation for pizzicato technique."""
        gen = VPTTSampleGenerator()
        spec = {
            "instrument": "violin",
            "technique": "pizzicato",
            "dynamic": "soft",
            "velocity": 45,
        }
        
        phrase = gen._generate_phrase(spec, ticks_per_beat=480)
        
        # Pizzicato should have short notes (1/16 note = 120 ticks)
        assert all(duration == 120 for _, _, duration in phrase)

    def test_generate_metadata(self):
        """Test metadata generation."""
        gen = VPTTSampleGenerator(seed=42)
        specs = gen.sample_combinations(n=5)
        
        # Generate metadata dict
        metadata = gen.generate_metadata(specs)
        
        # Verify structure
        assert metadata["dataset"] == "VPTT-50"
        assert metadata["total_samples"] == 5
        assert len(metadata["samples"]) == 5
        
        # Check sample structure
        for sample in metadata["samples"]:
            assert "id" in sample
            assert "file" in sample
            assert "instrument" in sample
            assert "technique" in sample
            assert "tempo" in sample
            assert "tempo_bpm" in sample
            assert "dynamic" in sample
            assert "velocity" in sample
            assert "marking" in sample

    def test_full_generation_workflow(self):
        """Test complete generation workflow."""
        gen = VPTTSampleGenerator(seed=42)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            midi_dir = tmpdir / "midi"
            metadata_path = tmpdir / "metadata.yaml"
            
            # Generate samples
            specs = gen.sample_combinations(n=5)
            
            # Generate MIDI files
            midi_dir.mkdir(exist_ok=True)
            for i, spec in enumerate(specs):
                midi_path = midi_dir / f"vptt_{i:03d}.mid"
                mid = gen.generate_midi(spec)
                mid.save(str(midi_path))
            
            # Generate metadata and save to YAML
            metadata = gen.generate_metadata(specs)
            with open(metadata_path, "w") as f:
                yaml.dump(metadata, f, default_flow_style=False, allow_unicode=True)
            
            # Verify all files exist
            assert len(list(midi_dir.glob("*.mid"))) == 5
            assert metadata_path.exists()
            
            # Verify metadata content
            with open(metadata_path) as f:
                loaded_metadata = yaml.safe_load(f)
            assert loaded_metadata["total_samples"] == 5
