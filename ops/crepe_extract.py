#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ops/crepe_extract.py
------------------------------------------------------------
F0曲線（CREPE優先, 未導入ならlibrosa YIN/pyinに自動フォールバック）
- 入力: --audio WAV, --bars bars.parquet
- 出力: --out bass_f0.parquet 例
 出力列: bar_index, f0_median_hz, f0_median_midi, f0_voiced_ratio, vibrato_rate_hz, slide_activity
------------------------------------------------------------
"""
import argparse
import sys
import json
import math
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import soundfile as sf

warnings.filterwarnings("ignore")


def hz_to_midi(hz):
    hz = np.asarray(hz, float)
    safe = np.where(hz > 0, hz, np.nan)
    midi = 69 + 12 * np.log2(safe / 440.0)
    return midi


def midi_to_hz(midi):
    return 440.0 * (2.0 ** ((midi - 69) / 12.0))


def moving_median(x, win):
    if win <= 1:
        return x
    s = pd.Series(x)
    return s.rolling(win, center=True, min_periods=1).median().values


def autocorr_peak_hz(x, sr, min_hz=3.0, max_hz=10.0):
    # x: cents系列（平均0付近）からビブラート周波数を推定
    if len(x) < 8:
        return 0.0
    x = x - np.nanmean(x)
    x = np.nan_to_num(x, nan=0.0)
    ac = np.correlate(x, x, mode="full")
    ac = ac[len(ac) // 2 :]
    # 探索ラグ範囲
    min_lag = int(sr / max_hz) if max_hz > 0 else 1
    max_lag = int(sr / min_hz) if min_hz > 0 else min_lag + 1
    if max_lag <= min_lag + 1:
        return 0.0
    idx = np.argmax(ac[min_lag:max_lag]) + min_lag
    if idx <= 0:
        return 0.0
    return float(sr / idx)


def extract_f0_times(
    y, sr, hop_s=0.01, min_hz=27.5, max_hz=880.0, smooth_ms=120, model_size="full", vuv_thresh=0.6
):
    """
    戻り値: times(sec), f0_hz(0=unvoiced), conf(0-1)

    Args:
        model_size: CREPE model size (tiny, full, large)
        vuv_thresh: V/UV threshold (0.0-1.0)
    """
    try:
        import crepe

        step_ms = hop_s * 1000.0
        # CREPEは16k固定推奨だが内部でリサンプルしてくれる
        time, frequency, confidence, _ = crepe.predict(
            y, sr, step_size=step_ms, model_capacity=model_size, viterbi=True, verbose=0
        )
        f0 = frequency.astype(float)
        conf = confidence.astype(float)
        times = time.astype(float)

        # V/UV閾値適用（信頼度が低い場合はunvoiced扱い）
        f0 = np.where(conf >= vuv_thresh, f0, 0.0)

        print(f"🎵 Using CREPE (model={model_size}, vuv_thresh={vuv_thresh}) for F0 extraction")
    except Exception as e:
        # librosaフォールバック: YIN/pyin
        print(f"⚠️  CREPE not available ({e}), falling back to librosa YIN")
        import librosa

        y_mono = librosa.to_mono(y) if y.ndim > 1 else y
        hop_length = max(1, int(sr * hop_s))
        f0 = librosa.yin(
            y_mono,
            fmin=min_hz,
            fmax=max_hz,
            sr=sr,
            frame_length=2048,
            hop_length=hop_length,
        )
        times = librosa.frames_to_time(np.arange(len(f0)), sr=sr, hop_length=hop_length)
        conf = np.ones_like(f0) * 0.6
        f0 = np.where(np.isfinite(f0), f0, 0.0)

    # 平滑（メディアン）
    win = max(1, int((smooth_ms / 1000.0) / hop_s))
    f0_smooth = moving_median(f0, win)
    return times, f0_smooth, conf


def per_bar_stats(times, f0_hz, bars_df):
    rows = []
    midi = hz_to_midi(f0_hz)
    median_f0 = np.nanmedian(f0_hz[f0_hz > 0]) if np.any(f0_hz > 0) else 1.0
    cents = 1200.0 * np.log2(np.maximum(f0_hz, 1e-6) / np.maximum(median_f0, 1e-6))
    dt = np.diff(times, prepend=(times[0] if len(times) > 0 else 0))
    sr_like = 1.0 / np.median(dt) if np.all(dt > 0) else 100.0

    for _, row in bars_df.iterrows():
        b = int(row["bar_index"])
        t0 = float(row.get("start_sec", np.nan))
        t1 = float(row.get("end_sec", np.nan))
        if not np.isfinite(t0) or not np.isfinite(t1):
            continue
        sel = (times >= t0) & (times < t1)
        if not np.any(sel):
            rows.append(
                dict(
                    bar_index=b,
                    f0_median_hz=0.0,
                    f0_median_midi=np.nan,
                    f0_voiced_ratio=0.0,
                    vibrato_rate_hz=0.0,
                    slide_activity=0.0,
                )
            )
            continue
        f0_seg = f0_hz[sel]
        midi_seg = midi[sel]
        cents_seg = cents[sel]
        voiced = f0_seg > 0
        voiced_ratio = float(np.mean(voiced)) if len(voiced) > 0 else 0.0
        med_hz = float(np.median(f0_seg[voiced])) if np.any(voiced) else 0.0
        med_midi = float(np.median(midi_seg[voiced])) if np.any(voiced) else np.nan

        # vibrato: centsの自己相関ピークから
        vib = (
            autocorr_peak_hz(cents_seg[np.isfinite(cents_seg)], sr_like, 3.0, 9.0)
            if np.any(voiced)
            else 0.0
        )

        # slide: |Δmidi|の90パーセンタイル
        dm = (
            np.abs(np.diff(midi_seg[np.isfinite(midi_seg)]))
            if np.any(np.isfinite(midi_seg))
            else np.array([])
        )
        slide = float(np.percentile(dm, 90)) if dm.size > 0 else 0.0

        rows.append(
            dict(
                bar_index=b,
                f0_median_hz=med_hz,
                f0_median_midi=med_midi,
                f0_voiced_ratio=voiced_ratio,
                vibrato_rate_hz=vib,
                slide_activity=slide,
            )
        )
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description="F0 extraction (CREPE or librosa fallback)")
    ap.add_argument("--audio", required=True, help="Input audio file (WAV)")
    ap.add_argument("--bars", required=True, help="bars.parquet path")
    ap.add_argument("--out", required=True, help="Output F0 parquet file")
    ap.add_argument("--hop-ms", type=float, default=10.0, help="Hop size in ms")
    ap.add_argument(
        "--smooth-ms", type=float, default=120.0, help="Smoothing window in ms (median filter)"
    )
    ap.add_argument("--median-ms", type=float, default=50.0, help="Median filter window in ms")
    ap.add_argument("--min-hz", type=float, default=27.5, help="Minimum F0 in Hz")
    ap.add_argument("--max-hz", type=float, default=880.0, help="Maximum F0 in Hz")
    ap.add_argument(
        "--model-size",
        default="full",
        choices=["tiny", "full", "large"],
        help="CREPE model size (tiny, full, large)",
    )
    ap.add_argument("--vuv-thresh", type=float, default=0.6, help="V/UV threshold (0.0-1.0)")
    args = ap.parse_args()

    print(f"🎵 F0 Extraction: {args.audio}")
    y, sr = sf.read(args.audio, always_2d=False)
    if y.ndim > 1:
        y = np.mean(y, axis=1)

    bars = pd.read_parquet(args.bars)
    if "start_sec" not in bars.columns or "end_sec" not in bars.columns:
        # tempo一定前提で計算（bar_index * bar_dur）
        tempo = 120.0
        if "tempo_bpm" in bars.columns and np.isfinite(bars["tempo_bpm"].iloc[0]):
            tempo = float(bars["tempo_bpm"].iloc[0])
        bar_dur = 240.0 / tempo  # 4/4前提
        bars["start_sec"] = bars["bar_index"] * bar_dur
        bars["end_sec"] = (bars["bar_index"] + 1) * bar_dur

    times, f0, conf = extract_f0_times(
        y,
        sr,
        hop_s=args.hop_ms / 1000.0,
        min_hz=args.min_hz,
        max_hz=args.max_hz,
        smooth_ms=args.median_ms,  # median filterのウィンドウサイズ
        model_size=args.model_size,
        vuv_thresh=args.vuv_thresh,
    )
    df = per_bar_stats(times, f0, bars)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out, index=False)
    print(f"✅ Saved F0 features → {args.out}  (rows={len(df)})")

    # Save metadata for sanity checking
    duration_sec = float(len(y) / sr)
    hop_ms = args.hop_ms
    frames = len(times)
    expected_min_frames = int(0.8 * (duration_sec * 1000 / hop_ms))
    ok = frames >= expected_min_frames

    meta = {
        "duration_sec": duration_sec,
        "hop_ms": hop_ms,
        "frames": frames,
        "expected_min_frames": expected_min_frames,
        "ok": ok,
        "model_size": args.model_size,
        "vuv_thresh": args.vuv_thresh,
    }
    meta_path = Path(args.out).with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    print(f"✅ Saved meta → {meta_path}")
    if not ok:
        print(f"⚠️  WARNING: Frame count ({frames}) below expected minimum ({expected_min_frames})")


if __name__ == "__main__":
    main()
