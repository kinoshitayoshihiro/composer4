#!/usr/bin/env python3
"""Validate Stage3 conditions parquet schema and data quality.

Usage:
    PYTHONPATH=. python scripts/validate_conditions.py \
        conditions/stage3_conditions.parquet
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = {
    "loop_id": "object",
    "file_digest": "object",
}

OPTIONAL_COLUMNS = {
    "emotion": "object",
    "genre": "object",
    "valence": "float64",
    "arousal": "float64",
    "caption": "object",
    "technique": "object",
}

MAX_NULL_RATE = 0.10  # 10% null rate threshold


def validate_schema(df: pd.DataFrame) -> list[str]:
    """Validate required columns and types."""
    errors = []
    
    for col, expected_type in REQUIRED_COLUMNS.items():
        if col not in df.columns:
            errors.append(f"Missing required column: {col}")
        elif not pd.api.types.is_dtype_equal(df[col].dtype, expected_type):
            errors.append(
                f"Column {col} has type {df[col].dtype}, expected {expected_type}"
            )
    
    return errors


def validate_null_rates(df: pd.DataFrame) -> list[str]:
    """Validate null rates are within acceptable limits."""
    errors = []
    
    for col in ["emotion", "genre", "caption"]:
        if col not in df.columns:
            continue
        
        null_rate = df[col].isna().sum() / len(df)
        if null_rate > MAX_NULL_RATE:
            errors.append(
                f"Column {col} has high null rate: {null_rate:.1%} (max: {MAX_NULL_RATE:.1%})"
            )
    
    return errors


def validate_value_ranges(df: pd.DataFrame) -> list[str]:
    """Validate value ranges for numeric columns."""
    errors = []
    
    for col in ["valence", "arousal"]:
        if col not in df.columns:
            continue
        
        valid_mask = df[col].notna()
        if valid_mask.any():
            min_val = df.loc[valid_mask, col].min()
            max_val = df.loc[valid_mask, col].max()
            
            if min_val < 0 or max_val > 1:
                errors.append(
                    f"Column {col} has values outside [0, 1]: min={min_val:.3f}, max={max_val:.3f}"
                )
    
    return errors


def validate_uniqueness(df: pd.DataFrame) -> list[str]:
    """Validate key uniqueness."""
    errors = []
    
    if "loop_id" in df.columns:
        duplicates = df["loop_id"].duplicated().sum()
        if duplicates > 0:
            errors.append(f"Found {duplicates} duplicate loop_id values")
    
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate conditions parquet")
    parser.add_argument("parquet_file", type=Path, help="Parquet file to validate")
    parser.add_argument("--strict", action="store_true", help="Exit with error on warnings")
    
    args = parser.parse_args()
    
    if not args.parquet_file.exists():
        print(f"ERROR: File not found: {args.parquet_file}", file=sys.stderr)
        return 1
    
    print(f"Validating {args.parquet_file}...")
    df = pd.read_parquet(args.parquet_file)
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    
    all_errors = []
    
    # Run validations
    all_errors.extend(validate_schema(df))
    all_errors.extend(validate_null_rates(df))
    all_errors.extend(validate_value_ranges(df))
    all_errors.extend(validate_uniqueness(df))
    
    if all_errors:
        print(f"\n❌ Found {len(all_errors)} validation error(s):")
        for error in all_errors:
            print(f"  - {error}")
        return 1 if args.strict else 0
    else:
        print("\n✅ All validations passed")
        return 0


if __name__ == "__main__":
    sys.exit(main())
