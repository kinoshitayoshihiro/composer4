#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
extract_stem_features.py

Compute per-bar features for stems in a directory and emit:
  - a wide parquet (stems_features.parquet)
  - per-role bar parquets under role_bars/{role}.parquet with
    {role}_activity, {role}_density_target, {role}_rms_db, {role}_onset_rate

Role inference: filename contains one of ["vocals","vocal","vox","drums","bass","guitar","piano","keys","strings","str"] unless a --role-map JSON is provided.
"""

import argparse, sys, re
from pathlib import Path
import numpy as np
import pandas as pd

def _lazy_import_librosa():
    try:
        import librosa
    except Exception as e:
        print("ERROR: librosa is required. pip install librosa", file=sys.stderr)
        raise
    return librosa

ROLE_KEYS = {
    "vocals":["vocals","vocal","vox"],
    "drums":["drums","drum"],
    "bass":["bass"],
    "guitar":["guitar","gtr"],
    "piano":["piano","keys","key"],
    "strings":["strings","string","str"]
}

def infer_role(name: str):
    low = name.lower()
    for role, keys in ROLE_KEYS.items():
        if any(k in low for k in keys):
            return role
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stems-dir", required=True)
    ap.add_argument("--bars", required=True)
    ap.add_argument("--out-wide", required=True, help="analysis/stems_features.parquet")
    ap.add_argument("--out-role-dir", required=True, help="analysis/role_bars/")
    ap.add_argument("--role-map", help="JSON mapping filename->role (optional)")
    ap.add_argument("--sr", type=int, default=22050)
    ap.add_argument("--hop", type=int, default=512)
    ap.add_argument("--rms_thresh_db", type=float, default=-40.0, help="activity threshold in dB")
    args = ap.parse_args()

    librosa = _lazy_import_librosa()
    stems_dir = Path(args.stems_dir)
    out_role_dir = Path(args.out_role_dir)
    out_role_dir.mkdir(parents=True, exist_ok=True)

    bars = pd.read_parquet(args.bars).sort_values("bar_index").reset_index(drop=True)
    for col in ("bar_index","start_sec","end_sec"):
        if col not in bars.columns:
            raise ValueError(f"Bars parquet missing column: {col}")

    rows = []
    role_feature_frames = {}

    files = [p for p in stems_dir.iterdir() if p.suffix.lower() in (".wav",".flac",".mp3",".m4a")]
    if not files:
        raise FileNotFoundError(f"No audio stems found under {stems_dir}")

    for f in files:
        role = infer_role(f.name) if args.role_map is None else None
        if args.role_map:
            import json
            rm = json.loads(Path(args.role_map).read_text(encoding="utf-8"))
            role = rm.get(f.name, role)
        if role is None:
            # skip unknown
            continue

        y, sr = librosa.load(str(f), sr=args.sr, mono=True)
        hop = args.hop
        S = np.abs(librosa.stft(y, n_fft=2048, hop_length=hop))
        times = librosa.frames_to_time(np.arange(S.shape[1]), sr=sr, hop_length=hop)

        # Features
        rms = librosa.feature.rms(S=S).reshape(-1)
        rms_db = librosa.power_to_db(rms**2 + 1e-12)
        onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
        onset_frames = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr, hop_length=hop, units="frames")
        onset_binary = np.zeros_like(onset_env)
        onset_binary[onset_frames] = 1.0

        # Per-bar aggregates
        def agg(values, agg="mean", count=False):
            out = []
            for bs, be in zip(bars["start_sec"].values, bars["end_sec"].values):
                mask = (times >= bs) & (times < be)
                v = values[mask]
                if v.size == 0:
                    out.append(np.nan if not count else 0.0)
                    continue
                if count:
                    dur = max(1e-6, be-bs)
                    out.append(float(np.sum(v))/dur)
                else:
                    if agg=="mean":
                        out.append(float(np.nanmean(v)))
                    elif agg=="max":
                        out.append(float(np.nanmax(v)))
                    else:
                        out.append(float(np.nanmean(v)))
            return np.array(out, dtype=float)

        bar_rms_db = agg(rms_db, agg="mean")
        bar_onset_rate = agg(onset_binary, count=True)
        activity = (bar_rms_db > args.rms_thresh_db).astype(int)
        density_target = np.clip((bar_onset_rate / (np.nanmax(bar_onset_rate)+1e-6)), 0.0, 1.0)

        # Wide columns for stems_features
        rows.append(pd.DataFrame({
            "bar_index": bars["bar_index"].values,
            f"{role}_rms_db": np.nan_to_num(bar_rms_db, nan=-80.0),
            f"{role}_onset_rate": np.nan_to_num(bar_onset_rate, nan=0.0),
        }))

        # Role-specific parquet
        role_df = pd.DataFrame({
            "bar_index": bars["bar_index"].values,
            f"{role}_activity": activity,
            f"{role}_rms_db": np.nan_to_num(bar_rms_db, nan=-80.0),
            f"{role}_onset_rate": np.nan_to_num(bar_onset_rate, nan=0.0),
            f"{role}_density_target": np.nan_to_num(density_target, nan=0.0)
        })
        role_df.to_parquet(out_role_dir / f"{role}.parquet", index=False)
        role_feature_frames[role] = role_df

    # Merge wide
    wide = bars[["bar_index"]].copy()
    for df in rows:
        wide = wide.merge(df, on="bar_index", how="left")
    wide.to_parquet(args.out_wide, index=False)

if __name__ == "__main__":
    main()