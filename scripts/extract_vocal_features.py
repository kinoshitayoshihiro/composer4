#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
extract_vocal_features.py

Compute per-bar vocal features from a vocal WAV and bars.parquet.
Outputs a parquet with columns merged by bar_index:
  bar_index, vocal_onsets, vocal_onset_rate, vocal_rms_db,
  f0_confidence_mean, vocal_plosive, phrase_boundary

Optional: --merge-into-bars will augment bars.parquet in-place with
these columns and, if missing, create naive 'energy'/'valence'
(energy from loudness_db min-max, valence=0.5).
"""

import argparse, sys, math, warnings, json
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

def _maybe_crepe():
    try:
        import crepe
        return crepe
    except Exception:
        return None

def load_audio(path, sr=22050):
    librosa = _lazy_import_librosa()
    y, _sr = librosa.load(path, sr=sr, mono=True)
    return y, sr

def per_bar_stats(times, values, bars_df, agg="mean", count_thresh=None):
    """
    Aggregate frame-level values into bar-level using times (sec) and bars_df [start_sec,end_sec].
    If count_thresh is not None, interpret 'values' as boolean-like and count occurrences per bar,
    divide by bar duration (sec) to form a rate.
    """
    out = []
    # Precompute index per frame to speed up
    starts = bars_df['start_sec'].values
    ends   = bars_df['end_sec'].values
    for i, (bs, be) in enumerate(zip(starts, ends)):
        mask = (times >= bs) & (times < be)
        v = values[mask]
        if v.size == 0:
            if count_thresh is None:
                out.append(np.nan)
            else:
                out.append(0.0)
            continue
        if count_thresh is None:
            if agg == "mean":
                out.append(float(np.nanmean(v)))
            elif agg == "max":
                out.append(float(np.nanmax(v)))
            elif agg == "sum":
                out.append(float(np.nansum(v)))
            else:
                out.append(float(np.nanmean(v)))
        else:
            # count events above threshold as hits per second
            hits = float(np.sum(v >= count_thresh))
            dur = max(1e-6, be - bs)
            out.append(hits / dur)
    return np.array(out, dtype=float)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", required=True, help="vocal WAV path")
    ap.add_argument("--bars", required=True, help="bars.parquet with bar_index/start_sec/end_sec")
    ap.add_argument("--out", required=True, help="output parquet (vocal_features.parquet)")
    ap.add_argument("--merge-into-bars", help="If set, merge features into this bars.parquet in-place")
    ap.add_argument("--sr", type=int, default=22050)
    ap.add_argument("--hop", type=int, default=512)
    ap.add_argument("--plosive_thresh", type=float, default=0.65, help="threshold for plosive proxy")
    args = ap.parse_args()

    # Load bars
    bars_df = pd.read_parquet(args.bars)
    for col in ("bar_index","start_sec","end_sec"):
        if col not in bars_df.columns:
            raise ValueError(f"Bars parquet missing required column: {col}")
    bars_df = bars_df.sort_values("bar_index").reset_index(drop=True)

    # Load audio
    librosa = _lazy_import_librosa()
    y, sr = load_audio(args.audio, sr=args.sr)
    hop = args.hop

    # Frame times
    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=hop))
    times = librosa.frames_to_time(np.arange(S.shape[1]), sr=sr, hop_length=hop)

    # RMS (dB)
    rms = librosa.feature.rms(S=S).reshape(-1)
    rms_db = librosa.power_to_db(rms**2 + 1e-12)
    vocal_rms_db = per_bar_stats(times, rms_db, bars_df, agg="mean")

    # Onset envelope & rate
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
    # Count onsets using a simple peak-pick
    onset_frames = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr, hop_length=hop, units="frames")
    onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=hop)
    # Build a binary vector aligned to frame times for counting proxy
    onset_binary = np.zeros_like(onset_env)
    onset_binary[onset_frames] = 1.0
    vocal_onset_rate = per_bar_stats(times, onset_binary, bars_df, count_thresh=0.5)  # per-second rate
    vocal_onsets = per_bar_stats(times, onset_binary, bars_df, agg="sum")  # raw counts

    # Spectral rolloff as high-frequency energy proxy (for plosives)
    roll = librosa.feature.spectral_rolloff(S=S, sr=sr, roll_percent=0.95).reshape(-1)
    roll_n = (roll - np.nanmin(roll)) / max(1e-6, (np.nanmax(roll)-np.nanmin(roll)))
    onset_n = (onset_env - np.nanmin(onset_env)) / max(1e-6, (np.nanmax(onset_env)-np.nanmin(onset_env)))
    plosive_proxy = np.sqrt(np.clip(roll_n,0,1) * np.clip(onset_n,0,1))
    vocal_plosive = per_bar_stats(times, plosive_proxy, bars_df, agg="mean")

    # Pitch & confidence
    crepe = _maybe_crepe()
    if crepe is not None:
        # CREPE expects 16k
        y16 = librosa.resample(y, orig_sr=sr, target_sr=16000)
        import crepe as _crepe
        f0_t, f0_f, f0_c, _ = _crepe.predict(y16, 16000, viterbi=True, step_size=1000*hop/sr)  # step_ms ~ hop length
        # Align confidence via interpolation onto 'times'
        conf_times = np.linspace(0, len(y16)/16000, num=len(f0_c))
        f0_conf = np.interp(times, conf_times, f0_c)
    else:
        # Fallback: compute f0 via yin and synthesize a pseudo confidence
        try:
            f0 = librosa.yin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'), sr=sr, frame_length=2048, hop_length=hop)
            f0 = np.where(np.isfinite(f0), f0, np.nan)
            # pseudo confidence: voiced ratio over a small window
            voiced = (~np.isnan(f0)).astype(float)
            # smooth
            kernel = np.ones(9)/9
            f0_conf = np.convolve(voiced, kernel, mode="same")
        except Exception:
            f0_conf = np.zeros_like(times) + 0.0

    f0_conf_mean = per_bar_stats(times, f0_conf, bars_df, agg="mean")

    # Phrase boundary detection (very simple valley-based heuristic)
    # boundary if local RMS valley and onset rate drops
    pb = np.zeros(len(bars_df), dtype=int)
    for i in range(1, len(bars_df)-1):
        if (vocal_rms_db[i] + 2.0 < (vocal_rms_db[i-1] + vocal_rms_db[i+1]) / 2.0) and (vocal_onset_rate[i] < 0.5 * max(1e-6, (vocal_onset_rate[i-1]+vocal_onset_rate[i+1])/2.0)):
            pb[i] = 1

    out = pd.DataFrame({
        "bar_index": bars_df["bar_index"].values,
        "vocal_onsets": np.nan_to_num(vocal_onsets, nan=0.0),
        "vocal_onset_rate": np.nan_to_num(vocal_onset_rate, nan=0.0),
        "vocal_rms_db": np.nan_to_num(vocal_rms_db, nan=-80.0),
        "f0_confidence_mean": np.nan_to_num(f0_conf_mean, nan=0.0),
        "vocal_plosive": np.nan_to_num(vocal_plosive, nan=0.0),
        "phrase_boundary": pb.astype(int)
    })
    out.to_parquet(args.out, index=False)

    if args.merge_into_bars:
        bars = pd.read_parquet(args.merge_into_bars)
        merged = bars.merge(out, on="bar_index", how="left")
        # Ensure energy/valence exist
        if "energy" not in merged.columns:
            # naive energy from loudness_db min-max normalization
            if "loudness_db" in merged.columns:
                ld = merged["loudness_db"].values.astype(float)
                e = (ld - np.nanmin(ld)) / max(1e-6, (np.nanmax(ld)-np.nanmin(ld)))
                merged["energy"] = np.clip(e, 0.0, 1.0)
            else:
                merged["energy"] = 0.5
        if "valence" not in merged.columns:
            merged["valence"] = 0.5
        merged.to_parquet(args.merge_into_bars, index=False)

if __name__ == "__main__":
    main()