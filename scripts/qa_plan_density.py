#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
qa_plan_density.py

Usage:
  python qa_plan_density.py \
    --bars analysis/bars.parquet \
    --role-bars analysis/role_bars/guitar.parquet \
    --plan guitar_plan.json \
    --out qa_density_guitar.csv
"""
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd


def load_bars(p):
    df = pd.read_parquet(p).sort_values("bar_index").reset_index(drop=True)
    for c in ("bar_index", "start_beat", "end_beat", "start_sec", "end_sec", "tempo_bpm"):
        if c not in df.columns:
            raise ValueError(f"bars missing column: {c}")
    return df


def load_role_bars(p):
    return pd.read_parquet(p).sort_values("bar_index").reset_index(drop=True)


def load_plan(p):
    data = json.loads(Path(p).read_text(encoding="utf-8"))
    if isinstance(data, dict) and "events" in data:
        ev = data["events"]
    elif isinstance(data, dict) and "tracks" in data:
        # v4.1 format: tracks[0].events
        ev = data["tracks"][0]["events"]
    elif isinstance(data, list):
        ev = data
    else:
        raise ValueError("Unsupported plan JSON shape")
    return ev


def assign_bar_index(events, bars):
    starts = bars["start_beat"].values
    ends = bars["end_beat"].values
    out = []
    for e in events:
        sb = float(e.get("start_beats", e.get("start_beat", np.nan)))
        if np.isnan(sb):
            continue
        idx = np.where((sb >= starts) & (sb < ends))[0]
        if len(idx):
            out.append(int(bars.iloc[idx[0]]["bar_index"]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bars", required=True)
    ap.add_argument("--role-bars")
    ap.add_argument("--plan", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    bars = load_bars(args.bars)
    role = load_role_bars(args.role_bars) if args.role_bars else None
    events = load_plan(args.plan)

    bi = assign_bar_index(events, bars)
    counts = pd.Series(bi).value_counts().rename("events_per_bar")
    df = bars[["bar_index", "tempo_bpm", "start_sec", "end_sec"]].copy()
    df = df.merge(counts, left_on="bar_index", right_index=True, how="left")
    df["events_per_bar"] = df["events_per_bar"].fillna(0)
    df["bar_dur_sec"] = (df["end_sec"] - df["start_sec"]).clip(lower=1e-6)
    df["notes_per_sec"] = df["events_per_bar"] / df["bar_dur_sec"]

    if role is not None:
        df = df.merge(role, on="bar_index", how="left")

    # Simple expected band from density_target if present
    cols = df.columns
    base16 = (df["tempo_bpm"] / 60.0) * 4.0  # 16th grid baseline
    base8 = (df["tempo_bpm"] / 60.0) * 2.0  # 8th grid baseline

    if "guitar_density_target" in cols:
        dt = df["guitar_density_target"].astype(float).fillna(0.5)
        df["exp_notes_sec_min"] = (0.25 + 0.5 * dt) * base16 * 0.5
        df["exp_notes_sec_max"] = (0.50 + 1.5 * dt) * base16 * 0.8
    elif "piano_density_target" in cols:
        dt = df["piano_density_target"].astype(float).fillna(0.5)
        df["exp_notes_sec_min"] = (0.30 + 0.6 * dt) * base16 * 0.5
        df["exp_notes_sec_max"] = (0.60 + 1.4 * dt) * base16 * 0.8
    elif "strings_density_target" in cols:
        dt = df["strings_density_target"].astype(float).fillna(0.3)
        df["exp_notes_sec_min"] = (0.10 + 0.5 * dt) * base8 * 0.4
        df["exp_notes_sec_max"] = (0.30 + 1.0 * dt) * base8 * 0.7
    elif "bass_density_target" in cols:
        dt = df["bass_density_target"].astype(float).fillna(0.5)
        df["exp_notes_sec_min"] = (0.20 + 0.6 * dt) * base8 * 0.5
        df["exp_notes_sec_max"] = (0.40 + 1.2 * dt) * base8 * 0.8
    else:
        dt = 0.5
        df["exp_notes_sec_min"] = (0.25 + 0.5 * dt) * base16 * 0.5
        df["exp_notes_sec_max"] = (0.50 + 1.5 * dt) * base16 * 0.8

    df.to_csv(args.out, index=False)
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
