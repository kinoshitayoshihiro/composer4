#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ops/ddsp_timbre_curves.py
------------------------------------------------------------
シンセ/パッド等のための「音色カーブ」を抽出（DDSPが無くても近似指標を生成）
- brightness: スペクトル重心（正規化）
- roughness:   スペクトルフラットネス
- am_env:      振幅エンベロープ（RMS）
- vibrato_rate: F0または明るさの揺れから近似
- noise_high_ratio: 高域(>6kHz)割合
入力: --audio, --bars
出力: --out timbre_curves.parquet （bar集計）
------------------------------------------------------------
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import soundfile as sf


def norm01(x):
    x = np.asarray(x, float)
    lo, hi = np.nanpercentile(x, 5), np.nanpercentile(x, 95)
    y = (x - lo) / max(1e-9, hi - lo)
    return np.clip(y, 0.0, 1.0)


def moving_mean(x, win):
    if win <= 1:
        return x
    s = pd.Series(x)
    return s.rolling(win, center=True, min_periods=1).mean().values


def vibrato_from_series(x, sr_hz, min_hz=3, max_hz=9):
    x = x - np.nanmean(x)
    x = np.nan_to_num(x)
    ac = np.correlate(x, x, mode="full")[len(x) - 1 :]
    min_lag = int(sr_hz / max_hz)
    max_lag = int(sr_hz / min_hz)
    if max_lag <= min_lag + 1:
        return 0.0
    lag = np.argmax(ac[min_lag:max_lag]) + min_lag
    return float(sr_hz / lag) if lag > 0 else 0.0


def main():
    ap = argparse.ArgumentParser(description="Timbre curves extraction (DDSP-free)")
    ap.add_argument("--audio", required=True, help="Input audio file (WAV)")
    ap.add_argument("--bars", required=True, help="bars.parquet path")
    ap.add_argument("--out", required=True, help="Output timbre curves (JSON or parquet)")
    ap.add_argument("--hop-ms", type=float, default=20.0, help="Hop size in ms")
    ap.add_argument("--smooth-ms", type=float, default=200.0, help="Smoothing window in ms")
    ap.add_argument("--hi-cut-hz", type=float, default=6000.0, help="High frequency cutoff")
    ap.add_argument(
        "--format",
        choices=["json", "parquet"],
        default="json",
        help="Output format (json for DAW, parquet for analysis)",
    )
    args = ap.parse_args()

    print(f"🎨 Timbre Curves Extraction: {args.audio}")
    import librosa

    y, sr = sf.read(args.audio, always_2d=False)
    if y.ndim > 1:
        y = np.mean(y, axis=1)

    hop_length = max(1, int(sr * (args.hop_ms / 1000.0)))
    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=hop_length))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)

    # 特徴
    centroid = librosa.feature.spectral_centroid(S=S, sr=sr)[0]  # brightness proxy
    flatness = librosa.feature.spectral_flatness(S=S)[0]  # roughness proxy
    rms = librosa.feature.rms(S=S)[0]  # amplitude env
    hi_mask = freqs >= args.hi_cut_hz
    hi_energy = (S[hi_mask, :] ** 2).sum(axis=0)
    total_energy = (S**2).sum(axis=0) + 1e-9
    hi_ratio = hi_energy / total_energy

    # 正規化 & 平滑
    win = max(1, int(args.smooth_ms / args.hop_ms))
    br = moving_mean(norm01(centroid), win)
    rf = moving_mean(norm01(flatness), win)
    am = moving_mean(norm01(rms), win)
    hn = moving_mean(norm01(hi_ratio), win)

    # vibrato: brightness揺れから近似（F0があれば差し替え可）
    sr_like = 1000.0 / args.hop_ms
    vib = vibrato_from_series(br - np.mean(br), sr_like)

    # フレーム時間とバー集計
    times = librosa.frames_to_time(np.arange(S.shape[1]), sr=sr, hop_length=hop_length)
    bars = pd.read_parquet(args.bars)
    if "start_sec" not in bars.columns or "end_sec" not in bars.columns:
        tempo = 120.0
        if "tempo_bpm" in bars.columns and np.isfinite(bars["tempo_bpm"].iloc[0]):
            tempo = float(bars["tempo_bpm"].iloc[0])
        bar_dur = 240.0 / tempo
        bars["start_sec"] = bars["bar_index"] * bar_dur
        bars["end_sec"] = (bars["bar_index"] + 1) * bar_dur

    rows = []
    for _, row in bars.iterrows():
        b = int(row["bar_index"])
        t0, t1 = float(row["start_sec"]), float(row["end_sec"])
        section = str(row.get("section", "verse"))  # セクション情報取得

        sel = (times >= t0) & (times < t1)
        if not np.any(sel):
            rows.append(
                dict(
                    bar_index=b,
                    section=section,
                    brightness=0.0,
                    roughness=0.0,
                    am_env=0.0,
                    noise_high_ratio=0.0,
                    vibrato_rate_hz=vib,
                )
            )
            continue

        # セクション係数（chorus 1.2倍、bridge 0.9倍等）
        section_factor = {"chorus": 1.2, "bridge": 0.9}.get(section, 1.0)

        rows.append(
            dict(
                bar_index=b,
                section=section,
                brightness=float(np.median(br[sel])) * section_factor,
                roughness=float(np.median(rf[sel])) * section_factor,
                am_env=float(np.median(am[sel])) * section_factor,
                noise_high_ratio=float(np.median(hn[sel])),
                vibrato_rate_hz=float(vib),
            )
        )

    df = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    if args.format == "json":
        # 0-127正規化してJSON出力（DAW CC用）
        import json

        curves = {
            "expression": [int(np.clip(row["am_env"] * 127, 0, 127)) for _, row in df.iterrows()],
            "cutoff": [int(np.clip(row["brightness"] * 127, 0, 127)) for _, row in df.iterrows()],
            "resonance": [int(np.clip(row["roughness"] * 127, 0, 127)) for _, row in df.iterrows()],
            "meta": {
                "bars": len(df),
                "vibrato_rate_hz": float(vib),
                "section_factors_applied": True,
            },
        }
        with open(args.out, "w") as f:
            json.dump(curves, f, ensure_ascii=False, indent=2)
        print(f"✅ Saved timbre curves (JSON 0-127) → {args.out} (bars={len(df)})")
    else:
        # Parquet出力（分析用、0-1正規化維持）
        df.to_parquet(args.out, index=False)
        print(f"✅ Saved timbre curves (parquet) → {args.out} (rows={len(df)})")


if __name__ == "__main__":
    main()
