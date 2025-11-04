#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
verify_activity_correlation.py - Activity列と出音（密度/velocity）の相関検証

Usage:
    python3 scripts/verify_activity_correlation.py song_packages/suno_project/song_001
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def verify_activity_correlation(song_dir: str):
    """
    楽器別activityと出音の相関を検証

    期待値: 0.3〜0.6の正の相関（activityが効いている証拠）
    """
    song_path = Path(song_dir)

    # stem_features読み込み（activity列含む）
    stem_feat_path = song_path / "stem_features.parquet"
    if not stem_feat_path.exists():
        print(f"❌ stem_features.parquet not found: {stem_feat_path}")
        return False

    stem_feat = pd.read_parquet(stem_feat_path)
    if "bar" in stem_feat.columns:
        stem_feat = stem_feat.set_index("bar")

    # activity列検出
    act_cols = [c for c in stem_feat.columns if c.endswith("_activity")]
    if not act_cols:
        print(
            "ℹ️  No activity columns found in stem_features.parquet (expected after --inst-activity)"
        )
        return True  # エラーではない

    print(f"✅ Found activity columns: {', '.join(act_cols)}")
    print()

    # 楽器別検証
    instruments = [
        ("guitar", "guitar_activity"),
        ("piano", "piano_activity"),
        ("strings", "strings_activity"),
    ]

    results = []
    for role, col in instruments:
        plan_path = song_path / f"{role}_plan.json"
        if not plan_path.exists():
            print(f"⏭️  [{role}] plan not found, skipping")
            continue

        if col not in stem_feat.columns:
            print(f"⏭️  [{role}] {col} column not found, skipping")
            continue

        # plan読み込み
        with open(plan_path, "r", encoding="utf-8") as f:
            plan = json.load(f)

        events = plan.get("tracks", [{}])[0].get("events", [])
        if not events:
            print(f"⚠️  [{role}] No events in plan")
            continue

        # barごと密度/平均Vel集計
        df = pd.DataFrame(events)
        grouped = df.groupby("bar").agg(n=("pitch", "count"), vel=("velocity", "mean"))

        # activity列とマージ
        merged = grouped.join(stem_feat[[col]], how="left")
        merged = merged.dropna()

        if len(merged) < 2:
            print(f"⚠️  [{role}] Not enough data for correlation")
            continue

        # 相関計算
        corr_n = merged[[col, "n"]].corr().iloc[0, 1]
        corr_v = merged[[col, "vel"]].corr().iloc[0, 1]

        # 統計
        act_mean = merged[col].mean()
        act_std = merged[col].std()
        n_mean = merged["n"].mean()
        vel_mean = merged["vel"].mean()

        print(f"📊 [{role.upper()}]")
        print(f"   Activity: mean={act_mean:.3f}, std={act_std:.3f}")
        print(f"   Density:  mean={n_mean:.1f} notes/bar")
        print(f"   Velocity: mean={vel_mean:.1f}")
        print(f"   Correlation(activity, density):  {corr_n:+.3f}")
        print(f"   Correlation(activity, velocity): {corr_v:+.3f}")

        # 判定（0.3〜0.6が目安）
        if corr_n >= 0.3 and corr_v >= 0.2:
            print(f"   ✅ Activity control is EFFECTIVE")
        elif corr_n >= 0.15 or corr_v >= 0.10:
            print(f"   ⚠️  Activity control is WEAK (consider tuning)")
        else:
            print(f"   ❌ Activity control is NOT working (check implementation)")

        print()

        results.append(
            {
                "role": role,
                "corr_density": corr_n,
                "corr_velocity": corr_v,
                "effective": corr_n >= 0.3 and corr_v >= 0.2,
                "merged_df": merged,  # ワースト20バー抽出用
            }
        )

    # B-5: ワースト20バー出力（低相関バーの可視化）
    for res in results:
        role = res["role"]
        df = res["merged_df"]
        col = f"{role}_activity"

        # 相関の低いバーを抽出（activity高×density低 or activity低×density高）
        df["corr_residual"] = abs(df[col] - df["n"] / df["n"].max())
        worst20 = df.nsmallest(20, "corr_residual")[[col, "n", "vel", "corr_residual"]]

        csv_path = song_path / f"qa_activity_worst20_{role}.csv"
        worst20.to_csv(csv_path, index=True)
        print(f"   💾 Saved worst 20 bars to: {csv_path.name}")

    # サマリー
    if results:
        effective_count = sum(1 for r in results if r["effective"])
        print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print(
            f"Summary: {effective_count}/{len(results)} instruments show effective activity control"
        )
        print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        return effective_count == len(results)
    else:
        print("ℹ️  No instruments to verify (plans not generated yet?)")
        return True


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 scripts/verify_activity_correlation.py <song-dir>")
        print(
            "Example: python3 scripts/verify_activity_correlation.py song_packages/suno_project/song_001"
        )
        sys.exit(1)

    song_dir = sys.argv[1]
    success = verify_activity_correlation(song_dir)
    sys.exit(0 if success else 1)
