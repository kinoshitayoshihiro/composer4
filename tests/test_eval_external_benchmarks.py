#!/usr/bin/env python3
"""
Tests for External Benchmark Evaluation (Day 7-8)
"""

import json
import tempfile
from pathlib import Path

import pretty_midi
import pytest

from scripts.eval_external_benchmarks import (
    BenchmarkMetrics,
    ExternalBenchmarkEvaluator,
)


class TestExternalBenchmarkEvaluator:
    """Test external benchmark evaluator."""
    
    @pytest.fixture
    def sample_midi_dir(self) -> Path:
        """Create sample MIDI files for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            
            # Create 5 sample MIDI files
            for i in range(5):
                midi = pretty_midi.PrettyMIDI(initial_tempo=120)
                inst = pretty_midi.Instrument(program=0, is_drum=False)
                
                # Add some notes
                for j in range(10):
                    note = pretty_midi.Note(
                        velocity=60 + j * 5,
                        pitch=60 + j,
                        start=j * 0.5,
                        end=(j + 1) * 0.5,
                    )
                    inst.notes.append(note)
                
                midi.instruments.append(inst)
                midi.write(str(tmpdir_path / f"sample_{i}.mid"))
            
            yield tmpdir_path
    
    def test_initialization(self, sample_midi_dir):
        """Test evaluator initialization."""
        evaluator = ExternalBenchmarkEvaluator(
            dataset_dir=sample_midi_dir,
            output_dir=Path("outputs/test"),
            dataset_name="groove",
            subset_size=5,
        )
        
        assert evaluator.dataset_name == "groove"
        assert evaluator.subset_size == 5
    
    def test_find_midi_files(self, sample_midi_dir):
        """Test MIDI file discovery."""
        evaluator = ExternalBenchmarkEvaluator(
            dataset_dir=sample_midi_dir,
            output_dir=Path("outputs/test"),
        )
        
        midi_files = evaluator._find_midi_files()
        assert len(midi_files) == 5
    
    def test_bar_violation_detection(self, sample_midi_dir):
        """Test bar violation detection."""
        evaluator = ExternalBenchmarkEvaluator(
            dataset_dir=sample_midi_dir,
            output_dir=Path("outputs/test"),
        )
        
        midi_files = evaluator._find_midi_files()
        midi = pretty_midi.PrettyMIDI(str(midi_files[0]))
        
        violations = evaluator._detect_bar_violations(midi)
        
        assert "violation_count" in violations
        assert "violation_rate" in violations
        assert "total_bars" in violations
        assert violations["violation_rate"] >= 0.0
        assert violations["violation_rate"] <= 1.0
    
    def test_ci_acceptance_criteria(self):
        """Test CI acceptance criteria checking (寸評推奨)."""
        evaluator = ExternalBenchmarkEvaluator(
            dataset_dir=Path("dummy"),
            output_dir=Path("outputs/test"),
        )
        
        # Passing metrics
        passing_metrics = BenchmarkMetrics(
            bar_violation_rate=0.015,  # < 2.0%
            beat_violation_count=0,
            total_bars=100,
            harmonic_validity=88.0,  # ≥ 87.3%
            chord_transition_score=0.0,
            avg_sequence_length=1000.0,
            p95_sequence_length=1040.0,  # +4% (≤ 5%)
            p99_sequence_length=1080.0,
            velocity_std=12.0,
            velocity_range=60,
            unique_velocity_steps=10,
            timing_jitter_std=15.0,
            microtiming_rms=20.0,
            kick_snare_consistency=0.8,
            drum_role_separation=0.7,
        )
        
        result = evaluator._check_ci_acceptance(passing_metrics)
        assert result["bar_violation_rate"] is True
        assert result["harmonic_validity"] is True
        assert result["sequence_length_p95"] is True
        assert result["all_passed"] is True
        
        # Failing metrics
        failing_metrics = BenchmarkMetrics(
            bar_violation_rate=0.025,  # > 2.0% (FAIL)
            beat_violation_count=0,
            total_bars=100,
            harmonic_validity=85.0,  # < 87.3% (FAIL)
            chord_transition_score=0.0,
            avg_sequence_length=1000.0,
            p95_sequence_length=1070.0,  # +7% (FAIL)
            p99_sequence_length=1100.0,
            velocity_std=12.0,
            velocity_range=60,
            unique_velocity_steps=10,
            timing_jitter_std=15.0,
            microtiming_rms=20.0,
            kick_snare_consistency=0.8,
            drum_role_separation=0.7,
        )
        
        result = evaluator._check_ci_acceptance(failing_metrics)
        assert result["bar_violation_rate"] is False
        assert result["harmonic_validity"] is False
        assert result["sequence_length_p95"] is False
        assert result["all_passed"] is False
    
    def test_velocity_metrics(self, sample_midi_dir):
        """Test velocity diversity metrics."""
        evaluator = ExternalBenchmarkEvaluator(
            dataset_dir=sample_midi_dir,
            output_dir=Path("outputs/test"),
        )
        
        midi_files = evaluator._find_midi_files()
        midi = pretty_midi.PrettyMIDI(str(midi_files[0]))
        
        metrics = evaluator._compute_velocity_metrics(midi)
        
        assert "std" in metrics
        assert "range" in metrics
        assert "unique_steps" in metrics
        assert metrics["std"] >= 0.0
        assert metrics["range"] >= 0
    
    def test_timing_metrics(self, sample_midi_dir):
        """Test timing humanness metrics."""
        evaluator = ExternalBenchmarkEvaluator(
            dataset_dir=sample_midi_dir,
            output_dir=Path("outputs/test"),
        )
        
        midi_files = evaluator._find_midi_files()
        midi = pretty_midi.PrettyMIDI(str(midi_files[0]))
        
        metrics = evaluator._compute_timing_metrics(midi)
        
        assert "jitter_std" in metrics
        assert "microtiming_rms" in metrics
        assert metrics["jitter_std"] >= 0.0
        assert metrics["microtiming_rms"] >= 0.0
    
    def test_drum_coherence_metrics(self, sample_midi_dir):
        """Test drum coherence metrics."""
        evaluator = ExternalBenchmarkEvaluator(
            dataset_dir=sample_midi_dir,
            output_dir=Path("outputs/test"),
        )
        
        # Create drum MIDI
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            
            midi = pretty_midi.PrettyMIDI(initial_tempo=120)
            drum_inst = pretty_midi.Instrument(program=0, is_drum=True)
            
            # Add kick and snare pattern
            for i in range(8):
                # Kick on beats 1, 3
                if i % 2 == 0:
                    drum_inst.notes.append(
                        pretty_midi.Note(velocity=100, pitch=36, start=i * 0.5, end=i * 0.5 + 0.1)
                    )
                # Snare on beats 2, 4
                else:
                    drum_inst.notes.append(
                        pretty_midi.Note(velocity=90, pitch=38, start=i * 0.5, end=i * 0.5 + 0.1)
                    )
            
            midi.instruments.append(drum_inst)
            
            metrics = evaluator._compute_drum_coherence(midi)
            
            assert "kick_snare_consistency" in metrics
            assert "role_separation" in metrics
            assert metrics["kick_snare_consistency"] >= 0.0
            assert metrics["role_separation"] >= 0.0
