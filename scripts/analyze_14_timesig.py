#!/usr/bin/env python3
"""1/4拍子問題の詳細分析"""
import pandas as pd

df = pd.read_csv("output/stage2_production/stage2_aggregate.csv")
bad = df[df["timesig0"] == "1/4"]

print("📋 1/4拍子が検出されたデータセット:")
print(bad.groupby("dataset").size().sort_values(ascending=False))

print(f"\n📝 サンプル（最初の10件）:")
print(bad[["file", "dataset", "timesig0", "bpm0", "n_chords"]].head(10).to_string())

# CSVの重複チェック
print(f"\n🔍 CSV重複問題:")
print(f"  総行数: {len(df):,}")
print(f"  ユニークファイル: {df['file'].nunique():,}")
print(f"  重複: {len(df) - df['file'].nunique():,}")

# 重複ファイルのサンプル
duplicates = df[df.duplicated(subset=["file"], keep=False)].sort_values("file")
if len(duplicates) > 0:
    print(f"\n📋 重複ファイルサンプル（最初の10件）:")
    print(duplicates[["file", "dataset", "bpm0", "timesig0"]].head(10).to_string())
