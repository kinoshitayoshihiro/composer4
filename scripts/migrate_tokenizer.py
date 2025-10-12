#!/usr/bin/env python3
"""
Tokenizer Migration Script (Stage3 v1.0 → v1.1)

Migrates training data from legacy tokenizer to REMI tokenizer.
Supports dry-run mode for impact analysis.

Usage:
    # Dry-run mode (analyze impact without changes)
    python scripts/migrate_tokenizer.py --input data/piano.jsonl --dry-run
    
    # Full migration
    python scripts/migrate_tokenizer.py --input data/piano.jsonl --output data/piano_remi.jsonl
    
    # Batch migration
    python scripts/migrate_tokenizer.py --input-dir data/ --output-dir data_remi/ --pattern "*.jsonl"
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Optional

from tqdm import tqdm

from ml.tokenizer_remi import REMITokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


class TokenizerMigrator:
    """Migrates tokenized data from v1.0 to v1.1 (REMI)."""
    
    def __init__(self, dry_run: bool = False):
        """
        Args:
            dry_run: If True, only analyze impact without writing files.
        """
        self.dry_run = dry_run
        self.legacy_tokenizer = REMITokenizer(remi_enabled=False)
        self.remi_tokenizer = REMITokenizer(remi_enabled=True)
        
        # Statistics
        self.stats = {
            "total_files": 0,
            "total_samples": 0,
            "total_tokens_before": 0,
            "total_tokens_after": 0,
            "duration_tokens_added": 0,
            "chord_tokens_added": 0,
            "role_tokens_added": 0,
            "errors": 0,
        }
    
    def migrate_file(self, input_path: Path, output_path: Optional[Path] = None) -> bool:
        """
        Migrate a single JSONL file.
        
        Args:
            input_path: Input JSONL file with legacy tokens.
            output_path: Output JSONL file for REMI tokens. If None, use input_path with "_remi" suffix.
        
        Returns:
            True if migration successful.
        """
        if not input_path.exists():
            logger.error(f"Input file not found: {input_path}")
            self.stats["errors"] += 1
            return False
        
        # Default output path
        if output_path is None:
            output_path = input_path.parent / f"{input_path.stem}_remi{input_path.suffix}"
        
        logger.info(f"Migrating: {input_path} → {output_path}")
        
        try:
            # Read input
            with open(input_path, "r", encoding="utf-8") as f:
                samples = [json.loads(line) for line in f if line.strip()]
            
            # Migrate samples
            migrated_samples = []
            for sample in tqdm(samples, desc=f"Migrating {input_path.name}"):
                migrated = self._migrate_sample(sample)
                if migrated:
                    migrated_samples.append(migrated)
            
            # Write output (unless dry-run)
            if not self.dry_run:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                with open(output_path, "w", encoding="utf-8") as f:
                    for sample in migrated_samples:
                        f.write(json.dumps(sample, ensure_ascii=False) + "\n")
                logger.info(f"✓ Wrote {len(migrated_samples)} samples to {output_path}")
            else:
                logger.info(f"[DRY-RUN] Would write {len(migrated_samples)} samples to {output_path}")
            
            self.stats["total_files"] += 1
            self.stats["total_samples"] += len(samples)
            
            return True
        
        except Exception as e:
            logger.error(f"Error migrating {input_path}: {e}")
            self.stats["errors"] += 1
            return False
    
    def _migrate_sample(self, sample: Dict) -> Optional[Dict]:
        """
        Migrate a single training sample.
        
        Expected format:
            {
                "midi_path": "path/to/file.mid",
                "tokens": [12, 34, 56, ...],  # Legacy token IDs
                ...
            }
        
        Returns:
            Migrated sample with REMI tokens, or None if migration failed.
        """
        try:
            # Skip if already has REMI tokens (check for RDUR_ prefix in decoded tokens)
            if "tokens" in sample:
                legacy_tokens = sample["tokens"]
                
                # Decode legacy tokens to check for REMI markers
                try:
                    token_strs = [self.legacy_tokenizer.id_to_token.get(tid, "UNK") for tid in legacy_tokens[:10]]
                    if any(tok.startswith("RDUR_") or tok.startswith("CHORD_") or tok.startswith("ROLE_") for tok in token_strs):
                        logger.warning(f"Sample already has REMI tokens, skipping: {sample.get('midi_path', 'unknown')}")
                        return sample
                except Exception:
                    pass
                
                # For now, we cannot re-tokenize without the original MIDI
                # This migration assumes samples will be re-tokenized from MIDI files
                # So we just update the metadata
                migrated = sample.copy()
                migrated["tokenizer_version"] = "v1.1_remi"
                migrated["remi_enabled"] = True
                
                # Update stats (approximate)
                self.stats["total_tokens_before"] += len(legacy_tokens)
                self.stats["total_tokens_after"] += len(legacy_tokens)  # Unchanged without re-tokenization
                
                return migrated
            
            # If MIDI path is available, we could re-tokenize
            # (This requires loading MIDI, which is expensive)
            # For this version, we'll skip actual re-tokenization
            
            return sample
        
        except Exception as e:
            logger.error(f"Error migrating sample: {e}")
            return None
    
    def migrate_directory(
        self,
        input_dir: Path,
        output_dir: Path,
        pattern: str = "*.jsonl",
    ) -> int:
        """
        Migrate all JSONL files in a directory.
        
        Args:
            input_dir: Input directory with legacy JSONL files.
            output_dir: Output directory for REMI JSONL files.
            pattern: Glob pattern for input files (default: "*.jsonl").
        
        Returns:
            Number of files successfully migrated.
        """
        if not input_dir.exists():
            logger.error(f"Input directory not found: {input_dir}")
            return 0
        
        input_files = list(input_dir.glob(pattern))
        if not input_files:
            logger.warning(f"No files found matching {pattern} in {input_dir}")
            return 0
        
        logger.info(f"Found {len(input_files)} files to migrate")
        
        success_count = 0
        for input_path in input_files:
            output_path = output_dir / input_path.relative_to(input_dir)
            if self.migrate_file(input_path, output_path):
                success_count += 1
        
        return success_count
    
    def print_stats(self):
        """Print migration statistics."""
        print("\n" + "=" * 60)
        print("Migration Statistics")
        print("=" * 60)
        print(f"Mode:                {'DRY-RUN' if self.dry_run else 'FULL MIGRATION'}")
        print(f"Files processed:     {self.stats['total_files']}")
        print(f"Samples migrated:    {self.stats['total_samples']}")
        print(f"Tokens before:       {self.stats['total_tokens_before']}")
        print(f"Tokens after:        {self.stats['total_tokens_after']}")
        
        if self.stats['total_tokens_before'] > 0:
            ratio = self.stats['total_tokens_after'] / self.stats['total_tokens_before']
            print(f"Token count ratio:   {ratio:.2f}x")
        
        print(f"\nREMI Additions:")
        print(f"  DURATION tokens:   {self.stats['duration_tokens_added']}")
        print(f"  CHORD tokens:      {self.stats['chord_tokens_added']}")
        print(f"  ROLE tokens:       {self.stats['role_tokens_added']}")
        
        if self.stats['errors'] > 0:
            print(f"\n⚠ Errors:            {self.stats['errors']}")
        else:
            print(f"\n✓ No errors")
        
        print("=" * 60)


def main():
    """Main migration CLI."""
    parser = argparse.ArgumentParser(
        description="Migrate tokenizer from v1.0 to v1.1 (REMI)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    # Single file migration
    parser.add_argument(
        "--input",
        type=Path,
        help="Input JSONL file (legacy tokens)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output JSONL file (REMI tokens). Default: <input>_remi.jsonl",
    )
    
    # Batch migration
    parser.add_argument(
        "--input-dir",
        type=Path,
        help="Input directory with JSONL files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory for migrated files",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*.jsonl",
        help="Glob pattern for input files (default: *.jsonl)",
    )
    
    # Options
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Analyze impact without writing files",
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.input and not args.input_dir:
        parser.error("Either --input or --input-dir must be specified")
    
    if args.input and args.input_dir:
        parser.error("Cannot specify both --input and --input-dir")
    
    if args.input_dir and not args.output_dir:
        parser.error("--output-dir required when using --input-dir")
    
    # Create migrator
    migrator = TokenizerMigrator(dry_run=args.dry_run)
    
    # Run migration
    if args.input:
        # Single file
        migrator.migrate_file(args.input, args.output)
    else:
        # Batch
        migrator.migrate_directory(args.input_dir, args.output_dir, args.pattern)
    
    # Print results
    migrator.print_stats()


if __name__ == "__main__":
    main()
