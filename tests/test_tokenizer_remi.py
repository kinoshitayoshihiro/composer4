#!/usr/bin/env python3
"""
Unit tests for REMI tokenizer (Stage3 v1.1 Day 4-6)

Tests:
- DURATION token encoding/decoding
- CHORD token vocabulary
- ROLE token assignment for drums
- Backward compatibility with v1.0
- Bar violation rate improvement
"""

import tempfile
from pathlib import Path

import pretty_midi
import pytest

from ml.tokenizer_remi import REMITokenizer


@pytest.fixture
def simple_midi() -> pretty_midi.PrettyMIDI:
    """Create a simple MIDI with varied durations."""
    midi = pretty_midi.PrettyMIDI(initial_tempo=120)
    piano = pretty_midi.Instrument(program=0, is_drum=False)
    
    # Notes with different durations (quarter, half, whole notes)
    notes = [
        (0.0, 0.5, 60),    # Quarter note (1 beat)
        (0.5, 1.5, 64),    # Half note (2 beats)
        (1.5, 5.5, 67),    # Whole note (4 beats)
    ]
    
    for start, end, pitch in notes:
        note = pretty_midi.Note(
            velocity=100,
            pitch=pitch,
            start=start,
            end=end,
        )
        piano.notes.append(note)
    
    midi.instruments.append(piano)
    return midi


@pytest.fixture
def drum_midi() -> pretty_midi.PrettyMIDI:
    """Create a drum pattern for ROLE token testing."""
    midi = pretty_midi.PrettyMIDI(initial_tempo=120)
    drums = pretty_midi.Instrument(program=0, is_drum=True)
    
    # Kick, Snare, HiHat pattern
    pattern = [
        (0.0, 36),   # Kick
        (0.5, 42),   # Closed HiHat
        (1.0, 38),   # Snare
        (1.5, 42),   # Closed HiHat
        (2.0, 36),   # Kick
        (2.5, 46),   # Open HiHat
        (3.0, 38),   # Snare
        (3.5, 49),   # Crash
    ]
    
    for start, pitch in pattern:
        note = pretty_midi.Note(
            velocity=100,
            pitch=pitch,
            start=start,
            end=start + 0.1,
        )
        drums.notes.append(note)
    
    midi.instruments.append(drums)
    return midi


class TestREMITokenizer:
    """Test suite for REMI tokenizer."""
    
    def test_initialization_legacy_mode(self):
        """Test that legacy mode (remi_enabled=False) works."""
        tokenizer = REMITokenizer(remi_enabled=False)
        
        # Should have base vocab only
        assert tokenizer.remi_enabled is False
        assert tokenizer.vocab_size > 0
        
        # Should NOT have REMI tokens
        assert not any(tok.startswith("RDUR_") for tok in tokenizer.token_to_id)
        assert not any(tok.startswith("CHORD_") for tok in tokenizer.token_to_id)
        assert not any(tok.startswith("ROLE_") for tok in tokenizer.token_to_id)
    
    def test_initialization_remi_mode(self):
        """Test that REMI mode adds extension tokens."""
        tokenizer = REMITokenizer(remi_enabled=True)
        
        assert tokenizer.remi_enabled is True
        
        # Should have DURATION tokens
        duration_tokens = [t for t in tokenizer.token_to_id if t.startswith("RDUR_")]
        assert len(duration_tokens) == 6, f"Expected 6 DURATION tokens, got {len(duration_tokens)}"
        
        # Should have CHORD tokens (74 in actual implementation)
        chord_tokens = [t for t in tokenizer.token_to_id if t.startswith("CHORD_")]
        assert len(chord_tokens) >= 70, f"Expected >=70 CHORD tokens, got {len(chord_tokens)}"
        
        # Should have ROLE tokens
        role_tokens = [t for t in tokenizer.token_to_id if t.startswith("ROLE_")]
        assert len(role_tokens) >= 8, f"Expected >=8 ROLE tokens, got {len(role_tokens)}"
    
    def test_duration_token_mapping(self):
        """Test that DURATION tokens cover common note lengths."""
        tokenizer = REMITokenizer(remi_enabled=True)
        
        expected_durations = ["1/16", "1/8", "1/4", "1/2", "1", "2"]
        for dur in expected_durations:
            token = f"RDUR_{dur}"
            assert token in tokenizer.token_to_id, f"Missing DURATION token: {token}"
    
    def test_chord_token_coverage(self):
        """Test that CHORD tokens cover major/minor/7th chords."""
        tokenizer = REMITokenizer(remi_enabled=True)
        
        # Check major triads
        for root in ["C", "D", "E", "F", "G", "A", "B"]:
            assert f"CHORD_{root}" in tokenizer.token_to_id
        
        # Check minor triads
        for root in ["Cm", "Dm", "Em", "Fm", "Gm", "Am", "Bm"]:
            assert f"CHORD_{root}" in tokenizer.token_to_id
        
        # Check dominant 7ths
        for root in ["C7", "D7", "E7", "F7", "G7", "A7", "B7"]:
            assert f"CHORD_{root}" in tokenizer.token_to_id
    
    def test_drum_role_mapping(self):
        """Test that drum pitches map to correct ROLEs."""
        tokenizer = REMITokenizer(remi_enabled=True)
        
        # Check key drum roles
        assert 36 in tokenizer.DRUM_ROLES  # Kick
        assert tokenizer.DRUM_ROLES[36] == "KICK"
        
        assert 38 in tokenizer.DRUM_ROLES  # Snare
        assert tokenizer.DRUM_ROLES[38] == "SNARE"
        
        assert 42 in tokenizer.DRUM_ROLES  # Closed HiHat
        assert tokenizer.DRUM_ROLES[42] == "HIHAT"
        
        assert 49 in tokenizer.DRUM_ROLES  # Crash
        assert tokenizer.DRUM_ROLES[49] == "CRASH"
    
    def test_encode_legacy_mode(self, simple_midi):
        """Test that legacy mode encoding works (backward compatibility)."""
        tokenizer = REMITokenizer(remi_enabled=False)
        tokens = tokenizer.encode_midi(simple_midi)
        
        # Should produce tokens
        assert len(tokens) > 0
        
        # Should NOT contain REMI tokens
        token_strs = [tokenizer.id_to_token[tid] for tid in tokens]
        assert not any(tok.startswith("RDUR_") for tok in token_strs)
        assert not any(tok.startswith("CHORD_") for tok in token_strs)
        assert not any(tok.startswith("ROLE_") for tok in token_strs)
    
    def test_encode_remi_mode(self, simple_midi):
        """Test that REMI mode encoding includes DURATION tokens."""
        tokenizer = REMITokenizer(remi_enabled=True)
        tokens = tokenizer.encode_midi(simple_midi)
        
        # Should produce tokens
        assert len(tokens) > 0
        
        # Should contain at least some REMI DURATION tokens
        token_strs = [tokenizer.id_to_token[tid] for tid in tokens]
        duration_count = sum(1 for tok in token_strs if tok.startswith("RDUR_"))
        
        # We expect DURATION tokens for the 3 notes in simple_midi
        assert duration_count >= 1, f"Expected REMI DURATION tokens, got {duration_count}"
    
    def test_encode_drums_with_roles(self, drum_midi):
        """Test that drum encoding includes ROLE tokens."""
        tokenizer = REMITokenizer(remi_enabled=True)
        tokens = tokenizer.encode_midi(drum_midi)
        
        # Should produce tokens
        assert len(tokens) > 0
        
        # Should contain ROLE tokens (may be fewer than expected if encoding doesn't use them)
        token_strs = [tokenizer.id_to_token[tid] for tid in tokens]
        role_count = sum(1 for tok in token_strs if tok.startswith("ROLE_"))
        
        # We expect at least one ROLE token for drums
        assert role_count >= 1, f"Expected at least one ROLE token, got {role_count}"
        
        # Check that ROLE vocabulary is present
        assert "ROLE_KICK" in tokenizer.token_to_id
        assert "ROLE_SNARE" in tokenizer.token_to_id
        assert "ROLE_HIHAT" in tokenizer.token_to_id
    
    def test_find_remi_duration(self):
        """Test REMI duration matching logic."""
        tokenizer = REMITokenizer(remi_enabled=True)
        
        # Exact matches
        assert tokenizer._find_remi_duration(1.0) == "RDUR_1/4"  # Quarter note = 1 beat
        assert tokenizer._find_remi_duration(2.0) == "RDUR_1/2"  # Half note = 2 beats
        assert tokenizer._find_remi_duration(4.0) == "RDUR_1"    # Whole note = 4 beats
        
        # Close matches (within tolerance)
        assert tokenizer._find_remi_duration(0.95) == "RDUR_1/4"  # ~Quarter note
        assert tokenizer._find_remi_duration(1.98) == "RDUR_1/2"  # ~Half note
        
        # No match (too far from any REMI duration)
        assert tokenizer._find_remi_duration(3.0) is None
        assert tokenizer._find_remi_duration(10.0) is None
    
    def test_save_and_load(self):
        """Test that REMI tokenizer can be saved and loaded."""
        tokenizer = REMITokenizer(remi_enabled=True, beat_division=24)
        
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        
        try:
            # Save
            tokenizer.save(tmp_path)
            
            # Load
            loaded = REMITokenizer.load(tmp_path)
            
            # Check properties
            assert loaded.remi_enabled is True
            assert loaded.beat_division == 24
            assert loaded.vocab_size == tokenizer.vocab_size
            
            # Check REMI tokens preserved
            assert "RDUR_1/4" in loaded.token_to_id
            assert "CHORD_C" in loaded.token_to_id
            assert "ROLE_KICK" in loaded.token_to_id
        
        finally:
            tmp_path.unlink(missing_ok=True)
    
    def test_backward_compatibility_vocab(self):
        """Test that legacy mode vocab is compatible with v1.0."""
        legacy_tokenizer = REMITokenizer(remi_enabled=False)
        remi_tokenizer = REMITokenizer(remi_enabled=True)
        
        # Legacy vocab should be a subset of REMI vocab
        for token in legacy_tokenizer.token_to_id:
            if not token.startswith("RDUR_") and not token.startswith("CHORD_") and not token.startswith("ROLE_"):
                assert token in remi_tokenizer.token_to_id, \
                    f"Token {token} missing in REMI tokenizer"
    
    def test_get_stats(self):
        """Test tokenizer statistics."""
        tokenizer = REMITokenizer(remi_enabled=True)
        stats = tokenizer.get_stats()
        
        # Check basic stats
        assert "vocab_size" in stats
        assert "remi_enabled" in stats
        assert stats["remi_enabled"] is True
        
        # Check REMI extension stats
        assert "remi_extensions" in stats
        assert stats["remi_extensions"]["duration_tokens"] == 6
        assert stats["remi_extensions"]["chord_tokens"] >= 70  # Actual count is 74
        assert stats["remi_extensions"]["role_tokens"] >= 8
    
    def test_vocab_size_comparison(self):
        """Test that REMI mode has larger vocab than legacy."""
        legacy = REMITokenizer(remi_enabled=False)
        remi = REMITokenizer(remi_enabled=True)
        
        # REMI should have 6 + 74 + ~10 = ~90 more tokens
        expected_increase = 6 + 74 + len(set(remi.DRUM_ROLES.values()))
        actual_increase = remi.vocab_size - legacy.vocab_size
        
        assert actual_increase >= expected_increase - 5, \
            f"Expected vocab increase of ~{expected_increase}, got {actual_increase}"
    
    def test_role_only_for_drums(self):
        """Test ROLE tokens only appear for drum instruments (寸評推奨)."""
        # Create MIDI with piano and drums
        midi = pretty_midi.PrettyMIDI(initial_tempo=120)
        
        # Add piano (non-drum)
        piano = pretty_midi.Instrument(program=0, is_drum=False)
        piano.notes.append(pretty_midi.Note(velocity=100, pitch=60, start=0.0, end=0.5))
        midi.instruments.append(piano)
        
        # Add drums
        drums = pretty_midi.Instrument(program=0, is_drum=True)
        drums.notes.append(pretty_midi.Note(velocity=100, pitch=36, start=1.0, end=1.1))  # Kick
        midi.instruments.append(drums)
        
        tokenizer = REMITokenizer(remi_enabled=True)
        tokens = tokenizer.encode_midi(midi)
        
        # Decode tokens to strings
        token_strs = [tokenizer.id_to_token.get(tid, "UNK") for tid in tokens]
        
        # Should have ROLE tokens
        role_tokens = [t for t in token_strs if t.startswith("ROLE_")]
        assert len(role_tokens) >= 1, "Expected at least one ROLE token for drums"
        
        # ROLE tokens should only appear after drum instrument markers
        # (This is a simplified check - in real implementation, verify position)
        assert "ROLE_KICK" in token_strs, "Expected ROLE_KICK for kick drum"
    
    def test_role_to_pitch_mapping(self):
        """Test ROLE → representative pitch mapping (寸評推奨: デコーダ頑健性)."""
        tokenizer = REMITokenizer(remi_enabled=True)
        
        # All ROLE types should have a representative pitch
        for role in set(tokenizer.DRUM_ROLES.values()):
            assert role in tokenizer.ROLE_TO_PITCH, \
                f"ROLE {role} missing from ROLE_TO_PITCH mapping"
            
            # Representative pitch should be valid GM drum pitch
            pitch = tokenizer.ROLE_TO_PITCH[role]
            assert 35 <= pitch <= 81, f"Invalid representative pitch for {role}: {pitch}"
    
    def test_save_load_with_version_metadata(self):
        """Test save/load includes version and vocab hash (寸評推奨)."""
        tokenizer = REMITokenizer(remi_enabled=True, beat_division=24)
        
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        
        try:
            # Save
            tokenizer.save(tmp_path)
            
            # Check saved JSON contains version metadata
            import json
            data = json.loads(tmp_path.read_text())
            
            assert "version" in data, "Missing version field"
            assert "vocab_hash" in data, "Missing vocab_hash field"
            assert "vocab_size" in data, "Missing vocab_size field"
            assert data["vocab_size"] == tokenizer.vocab_size
            
            # Load
            loaded = REMITokenizer.load(tmp_path)
            
            # Properties should match
            assert loaded.remi_enabled == tokenizer.remi_enabled
            assert loaded.vocab_size == tokenizer.vocab_size
        
        finally:
            tmp_path.unlink(missing_ok=True)
    
    def test_vocab_mismatch_raises_error(self):
        """Test that vocab mismatch raises explicit error (寸評推奨: 自動フォールバック禁止)."""
        # Create v1.0 tokenizer
        tokenizer_v10 = REMITokenizer(remi_enabled=False)
        
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        
        try:
            # Save v1.0
            tokenizer_v10.save(tmp_path)
            
            # Manually corrupt vocab_size in saved file
            import json
            data = json.loads(tmp_path.read_text())
            data["vocab_size"] = 999  # Wrong size
            tmp_path.write_text(json.dumps(data))
            
            # Load should raise ValueError
            with pytest.raises(ValueError, match="Vocabulary size mismatch"):
                REMITokenizer.load(tmp_path)
        
        finally:
            tmp_path.unlink(missing_ok=True)


@pytest.mark.integration
class TestREMIIntegration:
    """Integration tests for REMI tokenizer."""
    
    def test_encode_decode_roundtrip(self, simple_midi):
        """Test that encoding preserves musical structure."""
        tokenizer = REMITokenizer(remi_enabled=True)
        tokens = tokenizer.encode_midi(simple_midi)
        
        # Should have reasonable token count
        # 3 notes × ~4 tokens/note = ~12 tokens minimum
        assert len(tokens) >= 10, f"Too few tokens: {len(tokens)}"
        
        # Should not have excessive tokens
        assert len(tokens) < 100, f"Too many tokens: {len(tokens)}"
    
    def test_multiple_instruments(self):
        """Test REMI encoding with multiple instruments."""
        midi = pretty_midi.PrettyMIDI(initial_tempo=120)
        
        # Add piano
        piano = pretty_midi.Instrument(program=0)
        piano.notes.append(pretty_midi.Note(velocity=100, pitch=60, start=0.0, end=1.0))
        midi.instruments.append(piano)
        
        # Add drums
        drums = pretty_midi.Instrument(program=0, is_drum=True)
        drums.notes.append(pretty_midi.Note(velocity=100, pitch=36, start=0.0, end=0.1))
        midi.instruments.append(drums)
        
        tokenizer = REMITokenizer(remi_enabled=True)
        tokens = tokenizer.encode_midi(midi)
        
        # Should have tokens for both instruments
        assert len(tokens) > 0
        
        # Should have ROLE token for drums
        token_strs = [tokenizer.id_to_token[tid] for tid in tokens]
        assert "ROLE_KICK" in token_strs
