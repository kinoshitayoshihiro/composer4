#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ops/tempo_map_cli.py
可変テンポ推定 → bars.parquetへ start_sec/end_sec 付与、tempo_map.json を出力
- 優先: madmom の Downbeat 推定（高精度）
- 代替: librosa の beat_track（madmom不在時）

使い方:
python ops/tempo_map_cli.py \
  --audio data/.../mix.wav \
  --bars  song_packages/.../bars.parquet \
  --out-bars song_packages/.../bars.parquet \
  --out-tempo song_packages/.../tempo_map.json \
  --bpb 4 --bpm-hint 74.677
"""
import argparse, json, sys, os, math
from pathlib import Path

import numpy as np
import pandas as pd

# フォールバックでのみ使う
try:
    import librosa
except Exception:
    librosa = None


def _madmom_available():
    try:
        import madmom  # noqa
        from madmom.features.downbeats import (
            RNNDownBeatProcessor,
            DBNDownBeatTrackingProcessor,
        )  # noqa

        return True
    except Exception:
        return False


def estimate_downbeats_madmom(audio_path, fps=100.0):
    """madmomで downbeat を推定。戻り値: [(time_sec, is_downbeat_bool), ...]"""
    from madmom.features.downbeats import RNNDownBeatProcessor, DBNDownBeatTrackingProcessor

    proc = RNNDownBeatProcessor()
    act = proc(str(audio_path))
    tracker = DBNDownBeatTrackingProcessor(beats_per_bar=[3, 4], fps=fps)
    beats = tracker(act)  # shape (N, 2): [time_sec, beat_position(1..beats_per_bar)]
    out = [(float(t), int(b) == 1) for (t, b) in beats]
    return out


def estimate_beats_librosa(audio_path, sr=22050):
    """librosaで beats を推定（downbeatは不明）。戻り: [(time_sec, is_downbeat=False), ...]"""
    if librosa is None:
        raise RuntimeError("librosa が見つかりません。pip install librosa を実行してください。")
    y, sr = librosa.load(str(audio_path), sr=sr, mono=True)
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr, units="time")
    return [(float(t), False) for t in beats]


def beats_to_bars(beats, bpb=4):
    """ビート列からバー開始時刻を生成（downbeat情報があればそれを優先）"""
    if not beats:
        return []

    # downbeat付きならそれでバー境界を確定
    has_down = any(is_down for _, is_down in beats)
    if has_down:
        bar_starts = [t for (t, is_down) in beats if is_down]
    else:
        # ダウンビート無し: 最初のビートをバー頭と仮定して bpb ごとに切る
        times = [t for (t, _) in beats]
        bar_starts = times[::bpb]
    return sorted(bar_starts)


def make_tempo_map_from_beats(beats):
    """ビート間隔から瞬間テンポ系列を生成 → [(time_sec, tempo_bpm), ...]"""
    if len(beats) < 2:
        return []
    # beats: list[(time_sec, is_down)]
    times = np.array([t for (t, _) in beats], dtype=float)
    dt = np.diff(times)  # 秒
    # 60 / 秒 = BPM（1拍=四分音符想定）
    # 外れ値にロバストなため中間値でクリップ
    bpm = np.clip(60.0 / np.maximum(dt, 1e-6), 30.0, 240.0)
    # 区間代表点として前のビート時刻にBPMをアサイン
    points = [(float(times[i]), float(bpm[i])) for i in range(len(bpm))]
    # 先頭に初期点（同値）を1つ置くと扱いやすい
    if points:
        points = [(points[0][0], points[0][1])] + points
    return points


def write_bars_with_times(bars_parquet, out_parquet, bar_starts, song_end_time=None):
    """bars.parquet に start_sec/end_sec を埋める（可変テンポに基づく）"""
    df = pd.read_parquet(bars_parquet)
    if "bar" not in df.columns and "bar_index" not in df.columns:
        raise ValueError("bars.parquet に bar または bar_index 列が必要です。")

    # bar または bar_index を正規化
    if "bar_index" not in df.columns and "bar" in df.columns:
        df["bar_index"] = df["bar"]

    bar_starts = sorted(bar_starts)
    if not bar_starts:
        raise ValueError("bar_starts が空です。ビート推定に失敗している可能性があります。")

    # バー数と整合
    n_bars = len(df)
    # bar_starts が不足なら延長、過剰なら切り詰め
    if len(bar_starts) < n_bars:
        # 最終バーの長さを平均で推定して補完
        if len(bar_starts) >= 2:
            avg_len = float(np.median(np.diff(bar_starts)))
        else:
            avg_len = 2.0  # とりあえず2秒
        last = bar_starts[-1]
        while len(bar_starts) < n_bars + 1:
            last += avg_len
            bar_starts.append(last)
    elif len(bar_starts) > n_bars + 1:
        bar_starts = bar_starts[: n_bars + 1]

    starts = bar_starts[:n_bars]
    ends = bar_starts[1 : n_bars + 1]

    if song_end_time is not None and len(ends) == n_bars:
        ends[-1] = max(ends[-1], float(song_end_time))

    df["start_sec"] = starts
    df["end_sec"] = ends
    df.to_parquet(out_parquet, index=False)
    return df


def main():
    ap = argparse.ArgumentParser(description="Variable tempo map -> bars.parquet (start/end)")
    ap.add_argument("--audio", type=Path, required=True, help="mix/master WAV (テンポ推定用)")
    ap.add_argument("--bars", type=Path, required=True, help="入力 bars.parquet（bar_index必須）")
    ap.add_argument("--out-bars", type=Path, required=True, help="出力 bars.parquet（上書きOK）")
    ap.add_argument("--out-tempo", type=Path, required=True, help="tempo_map.json 出力先")
    ap.add_argument("--bpb", type=int, default=4, help="beats per bar（通常4）")
    ap.add_argument("--bpm-hint", type=float, default=None, help="初期BPMのヒント（任意）")
    ap.add_argument("--prefer-madmom", action="store_true", help="madmom を優先して使う")
    args = ap.parse_args()

    # 推定
    use_madmom = args.prefer_madmom and _madmom_available()
    if use_madmom:
        beats = estimate_downbeats_madmom(args.audio)
    else:
        if _madmom_available():
            try:
                beats = estimate_downbeats_madmom(args.audio)
            except Exception:
                if librosa is None:
                    raise
                beats = estimate_beats_librosa(args.audio)
        else:
            beats = estimate_beats_librosa(args.audio)

    # バー境界
    bar_starts = beats_to_bars(beats, bpb=args.bpb)

    # tempo map（可視化・デバッグ用）
    tempo_map = make_tempo_map_from_beats(beats)
    args.out_tempo.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_tempo, "w", encoding="utf-8") as f:
        json.dump({"tempo_points": tempo_map}, f, ensure_ascii=False, indent=2)

    # bars へ反映
    args.out_bars.parent.mkdir(parents=True, exist_ok=True)
    df = write_bars_with_times(args.bars, args.out_bars, bar_starts)

    print(f"✅ Wrote tempo_map: {args.out_tempo}")
    print(f"✅ Wrote bars (start_sec/end_sec): {args.out_bars}")
    print(
        f"   bars: {len(df)} rows | start[{df['start_sec'].iloc[0]:.2f}] -> end[{df['end_sec'].iloc[-1]:.2f}]"
    )


if __name__ == "__main__":
    main()
