#!/usr/bin/env python3
"""
CSV重複除去ツール

Stage2のCSV集計ファイルから重複行を削除します。
（初回実行とResume実行で同じファイルが2回記録された場合の対処）

Usage:
    python scripts/dedupe_stage2_csv.py output/stage2_production/stage2_aggregate.csv
    
    # 出力先指定
    python scripts/dedupe_stage2_csv.py input.csv -o output_clean.csv
    
    # バックアップ作成
    python scripts/dedupe_stage2_csv.py input.csv --backup
"""
import argparse
import pandas as pd
from pathlib import Path


def dedupe_csv(input_path: Path, output_path: Path = None, backup: bool = False) -> None:
    """
    CSVファイルの重複を削除
    
    Parameters
    ----------
    input_path : Path
        入力CSVファイル
    output_path : Path, optional
        出力CSVファイル。Noneの場合は上書き
    backup : bool
        バックアップを作成するか
    """
    print(f"📂 Loading: {input_path}")
    df = pd.read_csv(input_path)
    
    original_count = len(df)
    print(f"   Original: {original_count:,} rows")
    
    # 'file'列で重複削除（最初の出現を保持）
    df_clean = df.drop_duplicates(subset=['file'], keep='first')
    clean_count = len(df_clean)
    dup_count = original_count - clean_count
    
    print(f"   Unique:   {clean_count:,} rows")
    print(f"   Removed:  {dup_count:,} duplicates")
    
    # バックアップ作成
    if backup and output_path is None:
        backup_path = input_path.with_suffix('.csv.backup')
        print(f"\n💾 Creating backup: {backup_path.name}")
        df.to_csv(backup_path, index=False)
    
    # 出力
    if output_path is None:
        output_path = input_path
    
    print(f"\n✅ Saving to: {output_path}")
    df_clean.to_csv(output_path, index=False)
    
    # 統計サマリー
    print(f"\n📊 Dataset Summary:")
    dataset_counts = df_clean.groupby('dataset').size().sort_values(ascending=False)
    for dataset, count in dataset_counts.items():
        print(f"   {dataset:20s} {count:>6,} files")
    print(f"   {'─' * 28}")
    print(f"   {'TOTAL':20s} {len(df_clean):>6,} files")


def main():
    ap = argparse.ArgumentParser(
        description="Remove duplicate rows from Stage2 CSV aggregate",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("input_csv", type=Path, help="Input CSV file")
    ap.add_argument("-o", "--output", type=Path, help="Output CSV file (default: overwrite input)")
    ap.add_argument("--backup", action="store_true", help="Create backup before overwrite")
    
    args = ap.parse_args()
    
    if not args.input_csv.exists():
        print(f"❌ File not found: {args.input_csv}")
        return 1
    
    dedupe_csv(args.input_csv, args.output, args.backup)
    return 0


if __name__ == "__main__":
    exit(main())
