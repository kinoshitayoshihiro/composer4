#!/usr/bin/env python3
"""Stage2データ健全性チェック - 多角的統計分析"""
import pandas as pd
import json
from pathlib import Path
from collections import Counter

csv_path = Path("output/stage2_production/stage2_aggregate.csv")
json_dir = Path("output/stage2_production/json")

print("=" * 70)
print("📊 Stage2データ健全性チェック（完全版）")
print("=" * 70)

# 1. CSV基本統計
df = pd.read_csv(csv_path)
print(f"\n✅ CSV統計")
print(f"  総行数: {len(df):,}")
print(f"  ユニークファイル数: {df['file'].nunique():,}")
print(f"  重複数: {len(df) - df['file'].nunique():,}")

# 2. JSONファイル数
json_files = list(json_dir.glob("*.stage2.json"))
print(f"\n✅ JSONファイル")
print(f"  ファイル数: {len(json_files):,}")
print(f"  CSV専用エントリ: {len(df) - len(json_files):,}曲")

# 3. データセット別統計
print(f"\n📊 データセット別統計")
dataset_stats = df.groupby("dataset").agg({
    "file": "count",
    "bpm0": ["mean", "min", "max"],
    "n_chords": ["mean", "min", "max"],
    "swing_pct": "mean",
    "backbeat_strength": "mean"
}).round(2)
print(dataset_stats)

# 4. timesig（拍子）健全性チェック
print(f"\n🎵 拍子（timesig）分布")
timesig_counts = df["timesig0"].value_counts()
print(timesig_counts.head(10))
bad_timesig = df[df["timesig0"] == "1/4"]
if len(bad_timesig) > 0:
    print(f"\n⚠️  '1/4' 拍子が {len(bad_timesig)} 件検出（本来は 4/4 のはず）")
else:
    print(f"\n✅ '1/4' 拍子エラーなし")

# 5. BPM分布チェック
print(f"\n🎼 BPM統計")
print(f"  平均: {df['bpm0'].mean():.1f}")
print(f"  中央値: {df['bpm0'].median():.1f}")
print(f"  最小: {df['bpm0'].min():.1f}")
print(f"  最大: {df['bpm0'].max():.1f}")
print(f"  標準偏差: {df['bpm0'].std():.1f}")

# 6. コード数分布（和音進行の複雑さ）
print(f"\n🎹 コード数統計")
print(f"  平均: {df['n_chords'].mean():.1f}")
print(f"  中央値: {df['n_chords'].median():.1f}")
print(f"  最大: {df['n_chords'].max():.0f}")
no_chords = df[df['n_chords'] == 0]
print(f"  コードなし（ドラム等）: {len(no_chords):,}曲 ({len(no_chords)/len(df)*100:.1f}%)")

# 7. グルーヴ分析
print(f"\n🥁 グルーヴ統計")
print(f"  平均Swing: {df['swing_pct'].mean():.1f}%")
print(f"  平均Backbeat: {df['backbeat_strength'].mean():.2f}")
high_swing = df[df['swing_pct'] > 60]
print(f"  高スウィング(>60%): {len(high_swing):,}曲")

# 8. コントロール整合性
print(f"\n🎛️  コントロール整合性")
print(f"  平均integrity: {df['controls_integrity'].mean():.3f}")
low_integrity = df[df['controls_integrity'] < 0.9]
print(f"  低整合性(<0.9): {len(low_integrity):,}曲")

# 9. 処理速度分析
print(f"\n⚡ 処理速度")
print(f"  平均処理時間: {df['processing_time_sec'].mean():.3f}秒/曲")
print(f"  最速: {df['processing_time_sec'].min():.3f}秒")
print(f"  最遅: {df['processing_time_sec'].max():.3f}秒")

# 10. データセット別JSON出力率
print(f"\n📁 データセット別JSON出力率")
csv_files = set(df['file'])
json_stems = {f.stem.replace('.stage2', '') for f in json_files}
for dataset in df['dataset'].unique():
    ds_files = set(df[df['dataset'] == dataset]['file'])
    ds_json = ds_files & json_stems
    rate = len(ds_json) / len(ds_files) * 100 if len(ds_files) > 0 else 0
    print(f"  {dataset:20s}: {len(ds_json):6,} / {len(ds_files):6,} ({rate:5.1f}%)")

print("\n" + "=" * 70)
print("✅ 健全性チェック完了")
print("=" * 70)
