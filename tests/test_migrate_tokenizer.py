#!/usr/bin/env python3
"""
Tests for migrate_tokenizer.py (寸評推奨: 冪等性テスト)

Tests:
- Migration idempotency (same input → same output, even if run twice)
- Dry-run consistency
"""

import json
import tempfile
from pathlib import Path

import pytest

from scripts.migrate_tokenizer import TokenizerMigrator


class TestMigrateTokenizerIdempotency:
    """Test migration idempotency (寸評推奨)."""
    
    @pytest.fixture
    def sample_jsonl(self) -> Path:
        """Create sample JSONL file for testing."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            # Write sample data
            for i in range(5):
                data = {
                    "midi_path": f"sample_{i}.mid",
                    "tokens": [1, 2, 3, 4, 5 + i],
                    "tokenizer_version": "v1.0",
                }
                f.write(json.dumps(data) + "\n")
            return Path(f.name)
    
    def test_migration_idempotency(self, sample_jsonl):
        """Test that applying migration twice produces identical results (寸評推奨)."""
        # First migration
        output1 = tempfile.NamedTemporaryFile(suffix='.jsonl', delete=False)
        output1_path = Path(output1.name)
        output1.close()
        
        migrator1 = TokenizerMigrator(dry_run=False)
        migrator1.migrate_file(sample_jsonl, output1_path)
        
        # Read first output
        with open(output1_path, 'r') as f:
            lines1 = f.readlines()
        
        # Second migration (on already-migrated data)
        output2 = tempfile.NamedTemporaryFile(suffix='.jsonl', delete=False)
        output2_path = Path(output2.name)
        output2.close()
        
        migrator2 = TokenizerMigrator(dry_run=False)
        migrator2.migrate_file(output1_path, output2_path)
        
        # Read second output
        with open(output2_path, 'r') as f:
            lines2 = f.readlines()
        
        try:
            # Should be identical (idempotency)
            assert len(lines1) == len(lines2), "Line count differs after second migration"
            
            for i, (line1, line2) in enumerate(zip(lines1, lines2)):
                data1 = json.loads(line1)
                data2 = json.loads(line2)
                
                # Tokens should be identical
                assert data1.get("tokens") == data2.get("tokens"), \
                    f"Line {i}: tokens differ between migrations"
                
                # Version should be consistent
                assert data1.get("tokenizer_version") == data2.get("tokenizer_version"), \
                    f"Line {i}: tokenizer_version differs"
        
        finally:
            # Cleanup
            sample_jsonl.unlink(missing_ok=True)
            output1_path.unlink(missing_ok=True)
            output2_path.unlink(missing_ok=True)
    
    def test_dry_run_consistency(self, sample_jsonl):
        """Test that dry-run produces same stats as actual run (寸評推奨)."""
        # Dry-run
        migrator_dry = TokenizerMigrator(dry_run=True)
        migrator_dry.migrate_file(sample_jsonl, Path("dummy_output.jsonl"))
        stats_dry = migrator_dry.stats.copy()
        
        # Actual run
        output_actual = tempfile.NamedTemporaryFile(suffix='.jsonl', delete=False)
        output_actual_path = Path(output_actual.name)
        output_actual.close()
        
        migrator_actual = TokenizerMigrator(dry_run=False)
        migrator_actual.migrate_file(sample_jsonl, output_actual_path)
        stats_actual = migrator_actual.stats.copy()
        
        try:
            # Stats should match (except for the actual file writing)
            assert stats_dry["total_files"] == stats_actual["total_files"], \
                "File count differs between dry-run and actual run"
            assert stats_dry["total_samples"] == stats_actual["total_samples"], \
                "Sample count differs between dry-run and actual run"
        
        finally:
            sample_jsonl.unlink(missing_ok=True)
            output_actual_path.unlink(missing_ok=True)
    
    def test_no_data_corruption(self, sample_jsonl):
        """Test that migration preserves all non-token fields."""
        output = tempfile.NamedTemporaryFile(suffix='.jsonl', delete=False)
        output_path = Path(output.name)
        output.close()
        
        # Add extra fields to test data
        with open(sample_jsonl, 'w') as f:
            data = {
                "midi_path": "test.mid",
                "tokens": [1, 2, 3],
                "tokenizer_version": "v1.0",
                "extra_field": "should_be_preserved",
                "metadata": {"key": "value"},
            }
            f.write(json.dumps(data) + "\n")
        
        migrator = TokenizerMigrator(dry_run=False)
        migrator.migrate_file(sample_jsonl, output_path)
        
        try:
            # Read migrated data
            with open(output_path, 'r') as f:
                migrated = json.loads(f.readline())
            
            # Extra fields should be preserved
            assert "extra_field" in migrated, "extra_field was lost"
            assert migrated["extra_field"] == "should_be_preserved"
            assert "metadata" in migrated, "metadata was lost"
            assert migrated["metadata"]["key"] == "value"
        
        finally:
            sample_jsonl.unlink(missing_ok=True)
            output_path.unlink(missing_ok=True)
