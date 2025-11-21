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
import argparse, json, sys, os, math, warnings
from pathlib import Path

import numpy as np
import pandas as pd

# フォールバックでのみ使う
try:
    import librosa
except Exception:
    librosa = None


def _get_duration_sec(audio_path: str) -> float:
    """
    Robust duration detector with multi-backend fallback.
    """
    # 1) soundfile (fast, accurate)
    try:
        import soundfile as sf

        info = sf.info(audio_path)
        if info.duration and info.duration > 0:
            return float(info.duration)
    except Exception:
        pass

    # 2) librosa (universal fallback)
    try:
        import librosa

        y, sr = librosa.load(audio_path, sr=None, mono=True)
        if sr and len(y) > 0:
            return float(len(y) / sr)
    except Exception:
        pass

    raise RuntimeError(f"Failed to detect duration for: {audio_path}")


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


def _extend_bars_to_duration(bar_starts, est_bar_sec, duration_sec, tail_guard=1e-3):
    """
    Ensure bar grid reaches the audio duration.
    bar_starts: list[float] of bar start times (seconds), strictly increasing.
    est_bar_sec: estimated seconds per bar (from tempo map median or beats).
    """
    bar_starts = list(bar_starts)  # Copy to avoid modifying original
    while len(bar_starts) == 0 or (bar_starts[-1] + est_bar_sec) < (duration_sec - tail_guard):
        bar_starts.append((bar_starts[-1] if bar_starts else 0.0) + est_bar_sec)
    return bar_starts


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
    ap.add_argument("--out-bars", type=Path, required=True, help="出力 bars.parquet")
    ap.add_argument("--out-tempo", type=Path, required=True, help="tempo_map.json 出力先")

    # duration-sec をオプション化（自動検出）
    ap.add_argument("--duration-sec", type=float, default=0.0, help="曲の長さ（秒）- 0なら自動検出")
    ap.add_argument(
        "--num-bars", type=int, help="小節数 - duration-sec の代わりに使用可（新規生成時）"
    )

    # 既存マージモード: bars-in を指定すると既存を保持してマージ
    ap.add_argument("--bars-in", type=Path, help="既存 bars.parquet（マージモード）")

    # 拡張パラメータ
    ap.add_argument("--bpb", type=int, default=4, help="beats per bar（通常4）")
    ap.add_argument("--timesig", default="4/4", help="拍子記号（例: 4/4, 3/4）")
    ap.add_argument("--ppq", type=int, default=480, help="Pulses per quarter note（参照値）")
    ap.add_argument("--bpm-hint", type=float, default=None, help="初期BPMのヒント（任意）")
    ap.add_argument("--prefer-madmom", action="store_true", help="madmom を優先して使う")
    args = ap.parse_args()

    # ---- Robust duration detection & logging ----
    if args.duration_sec and args.duration_sec > 0:
        duration_sec = float(args.duration_sec)
        print(f"[INFO] Using provided duration: {duration_sec:.2f} sec")
    else:
        duration_sec = _get_duration_sec(str(args.audio))
        print(f"[INFO] Detected duration from audio: {duration_sec:.2f} sec")

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

    # tempo map（バー延長前に生成）
    tempo_map = make_tempo_map_from_beats(beats)

    # est_bar_sec: テンポ点/ビート間隔から堅牢に見積もる
    if tempo_map:
        tempo_median_bpm = np.median([bpm for _, bpm in tempo_map])
    else:
        tempo_median_bpm = 120.0  # fallback
    est_bar_sec = max(60.0 / max(1e-6, tempo_median_bpm) * args.bpb, 1e-3)

    # 重要修正：楽曲末尾までバーを延長
    if (len(bar_starts) < 2) and (len(beats) >= args.bpb):
        # downbeat が信頼できない場合、beats から bpb ごとに bar を復元
        beat_times = [t for (t, _) in beats]
        bar_starts = [beat_times[0]]
        for i in range(args.bpb, len(beat_times), args.bpb):
            bar_starts.append(beat_times[i])
        if len(bar_starts) < 2:
            # それでも足りなければ median テンポからグリッド生成
            bar_starts = [0.0]

    # duration まで不足分を補完
    bar_starts = _extend_bars_to_duration(bar_starts, est_bar_sec, duration_sec)

    # tempo map保存（可視化・デバッグ用）
    args.out_tempo.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_tempo, "w", encoding="utf-8") as f:
        json.dump({"tempo_points": tempo_map}, f, ensure_ascii=False, indent=2)

    # bars へ反映（新規生成 or マージ）
    args.out_bars.parent.mkdir(parents=True, exist_ok=True)

    if args.bars_in and args.bars_in.exists():
        # === マージモード: 既存 bars の編集列を保持して時間列のみ更新 ===
        print(f"📋 マージモード: 既存 bars を読み込み中... ({args.bars_in})")
        df_existing = pd.read_parquet(args.bars_in)

        # bar_index 正規化
        if "bar_index" not in df_existing.columns and "bar" in df_existing.columns:
            df_existing["bar_index"] = df_existing["bar"]

        n_bars = len(df_existing)

        # bar_starts を既存の小節数に合わせる
        if len(bar_starts) < n_bars:
            # 不足分を補完
            if len(bar_starts) >= 2:
                avg_len = float(np.median(np.diff(bar_starts)))
            else:
                avg_len = 2.0
            last = bar_starts[-1]
            while len(bar_starts) < n_bars + 1:
                last += avg_len
                bar_starts.append(last)
        elif len(bar_starts) > n_bars + 1:
            bar_starts = bar_starts[: n_bars + 1]

        starts = bar_starts[:n_bars]
        ends = bar_starts[1 : n_bars + 1]

        # 時間列のみ更新（既存の編集列は保持）
        df_existing["start_sec"] = starts
        df_existing["end_sec"] = ends

        # BPM も計算して追加
        bar_durations = np.array(ends) - np.array(starts)
        bpms = 60.0 * args.bpb / np.maximum(bar_durations, 0.1)  # bpb拍 / 秒
        df_existing["bpm"] = bpms

        df_existing.to_parquet(args.out_bars, index=False)
        df = df_existing

        print(f"✅ マージ完了: 既存列を保持して start_sec/end_sec/bpm を更新")

    else:
        # === 新規生成モード: ゼロから bars を作成 ===
        print("🆕 新規生成モード: ダミーなしで bars を作成中...")

        # bar_starts は既に duration_sec まで延長済み
        n_bars = len(bar_starts) - 1
        print(f"   曲の長さ: {duration_sec:.2f} 秒 → 生成小節数: {n_bars}")

        # bar_index を生成
        bar_indices = list(range(n_bars))
        starts = bar_starts[:n_bars]
        ends = bar_starts[1 : n_bars + 1]

        # end_sec を duration_sec でクランプ
        ends = [min(e, duration_sec) for e in ends]

        # BPM 計算
        bar_durations = np.array(ends) - np.array(starts)
        bpms = 60.0 * args.bpb / np.maximum(bar_durations, 0.1)

        # DataFrame 作成
        df = pd.DataFrame(
            {
                "bar_index": bar_indices,
                "start_sec": starts,
                "end_sec": ends,
                "bpm": bpms,
                "beats_per_bar": args.bpb,
                "time_sig": args.timesig,
            }
        )

        df.to_parquet(args.out_bars, index=False)
        print(f"✅ 新規生成完了: {n_bars} 小節")

    print(f"✅ Wrote tempo_map: {args.out_tempo}")
    print(f"✅ Wrote bars (start_sec/end_sec): {args.out_bars}")
    print(
        f"   bars: {len(df)} rows | start[{df['start_sec'].iloc[0]:.2f}] -> end[{df['end_sec'].iloc[-1]:.2f}]"
    )


if __name__ == "__main__":
    main()
