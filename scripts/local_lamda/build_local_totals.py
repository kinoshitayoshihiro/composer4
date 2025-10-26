#!/usr/bin/env python3
"""
LOCAL_TOTALS.pickle ビルダー

ステムMIDIから pitch/dur/vel の256-binヒストグラムを生成。
外れ値スコア計算（χ²距離）の先験として利用。

出力形式:
{
    "format": "local_totals_v1",
    "pitch_hist_256": [int, ...],  # 0..255 (pitch*2)
    "dur_hist_256": [int, ...],     # 1ms..10s を log2 で分割
    "vel_hist_256": [int, ...]      # 0..255 (vel*2)
}
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import pretty_midi as pm

# 除外ディレクトリ（temp/quarantine/shards等）
EXCLUDE_DIRS = {"temp", "quarantine", "shards", ".git", "__pycache__"}


def iter_midis(root: Path):
    """除外ディレクトリをスキップしてMIDIファイルを収集"""
    for p in root.rglob("*"):
        if p.suffix.lower() not in (".mid", ".midi"):
            continue
        if any(excl in p.parts for excl in EXCLUDE_DIRS):
            continue
        yield p


def bin256_pitch(p: int) -> int:
    """Pitch (0..127) を 256-bin へ"""
    return min(255, max(0, int(p * 2)))


def bin256_vel(v: int) -> int:
    """Velocity (0..127) を 256-bin へ"""
    return min(255, max(0, int(v * 2)))


def bin256_dur_ms(ms: float) -> int:
    """Duration (1ms..10s) を log2 で 256-bin へ"""
    ms = max(1.0, min(ms, 10000.0))
    # log2(1) = 0, log2(10000) ≈ 13.3 → 0..255 へ線形マッピング
    x = (np.log2(ms) - 0.0) / (np.log2(10000.0) - 0.0)
    return min(255, max(0, int(round(x * 255))))


def main():
    ap = argparse.ArgumentParser(
        description="LOCAL_TOTALS builder",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--midi-root",
        required=True,
        help="MIDI root directory (recursive search)",
    )
    ap.add_argument(
        "--out-pickle",
        required=True,
        help="Output pickle path (LOCAL_TOTALS.pickle)",
    )
    args = ap.parse_args()

    # ヒストグラム初期化
    pitch_hist = np.zeros(256, dtype=np.int64)
    dur_hist = np.zeros(256, dtype=np.int64)
    vel_hist = np.zeros(256, dtype=np.int64)

    # MIDI収集（除外ディレクトリをスキップ）
    midi_root = Path(args.midi_root)
    midi_files = sorted(set(iter_midis(midi_root)))
    print(f"📂 Found {len(midi_files)} MIDI files (excluding temp/quarantine/shards)")

    # ヒストグラム蓄積
    for i, mp in enumerate(midi_files, 1):
        if i % 100 == 0:
            print(f"  Processing {i}/{len(midi_files)}...")

        try:
            m = pm.PrettyMIDI(str(mp))
            for ins in m.instruments:
                for n in ins.notes:
                    pitch_hist[bin256_pitch(n.pitch)] += 1
                    vel_hist[bin256_vel(n.velocity)] += 1

                    dur_ms = (n.end - n.start) * 1000.0
                    dur_hist[bin256_dur_ms(dur_ms)] += 1

        except Exception as e:
            print(f"[skip] {mp}: {e}")

    # 保存
    total = {
        "format": "local_totals_v1",
        "pitch_hist_256": pitch_hist.tolist(),
        "dur_hist_256": dur_hist.tolist(),
        "vel_hist_256": vel_hist.tolist(),
    }

    with open(args.out_pickle, "wb") as f:
        pickle.dump(total, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"\n✅ Wrote totals -> {args.out_pickle}")
    print(f"   Total notes: {pitch_hist.sum():,}")


if __name__ == "__main__":
    main()
