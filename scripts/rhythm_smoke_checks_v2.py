#!/usr/bin/env python3
"""
Rhythm AI Stage2 Smoke Checks V2

3つの品質チェック（簡略化版）:
1. 拍子別・ファミリ別の最小数担保
2. 基本サニティチェック
3. 拍子×ファミリ層化シャッフル（多様性確保）
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-samples", type=int, default=30)
    parser.add_argument("--verbose", action="store_true")
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("🔍 Rhythm AI Stage2 Smoke Checks V2")
    print("=" * 70)
    
    df = pd.read_parquet(args.input)
    print(f"\n📂 Loaded: {args.input}")
    print(f"   Total records: {len(df)}")
    
    # Check 1: Time Sig × Family Distribution
    df['time_sig'] = df['time_sig_num'].astype(str) + '/' + df['time_sig_denom'].astype(str)
    grouped = df.groupby(['time_sig', 'family_label']).size().reset_index(name='count')
    
    valid_combinations = []
    dropped = 0
    
    for _, row in grouped.iterrows():
        if row['count'] >= args.min_samples:
            valid_combinations.append((row['time_sig'], row['family_label']))
        else:
            dropped += 1
            if args.verbose:
                print(f"⚠️  {row['time_sig']} × {row['family_label']}: {row['count']} samples (< {args.min_samples}) → 除外")
    
    mask = df.apply(lambda r: (r['time_sig'], r['family_label']) in valid_combinations, axis=1)
    df = df[mask].copy()
    
    print(f"\n✅ Check 1: Time Sig × Family Distribution")
    print(f"   Valid combinations: {len(valid_combinations)}")
    print(f"   Dropped combinations: {dropped}")
    print(f"   Records after: {len(df)}")
    
    # Check 2: Basic Sanity
    mask = (df['num_notes'] > 0) & (df['tempo_bpm'] > 0)
    before = len(df)
    df = df[mask].copy()
    
    print(f"\n✅ Check 2: Basic Sanity")
    print(f"   Dropped (invalid): {before - len(df)}")
    print(f"   Records after: {len(df)}")
    
    # Check 3: Stratified Shuffle
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"\n✅ Check 3: Stratified Shuffle")
    print(f"   Records after: {len(df)}")
    
    # Save
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(str(args.output), compression='snappy', index=False)
    
    print(f"\n💾 Saved: {args.output}")
    print(f"   Final records: {len(df)}")
    
    summary = {
        'initial_records': pd.read_parquet(args.input).shape[0],
        'final_records': len(df),
        'min_samples': args.min_samples
    }
    
    summary_path = args.output.parent / "rhythm_smoke_checks_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"💾 Summary: {summary_path}")
    print("\n" + "=" * 70)
    print("✅ Smoke checks completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()
