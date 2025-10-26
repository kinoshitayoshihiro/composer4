#!/usr/bin/env python3
"""
LOCAL_META_DATA.pickle ビルダー

ステムMIDIからメタデータ（notes/CC/PB/tempo/patches/統計）を抽出。
公式METAと同じAPI（lamda_sources）で読める。

出力形式:
[file_id, {
    "total_number_of_tracks": int,
    "total_number_of_notes": int,
    "tempo_change_count": int,
    "pb_range": [min, max],
    "cc_summary": {cc_num: {"count", "min", "max"}},
    "midi_patches": [int, ...],
    "avg_velocity": float,
    "avg_dur_ms": float
}]
"""
from __future__ import annotations

import argparse
import hashlib
import os
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


def make_id(path: str) -> str:
    """安定したLOCAL IDを生成（4資源共通）"""
    h = hashlib.sha1()
    h.update((path + "|" + str(os.path.getsize(path))).encode("utf-8"))
    return "LOCAL:" + h.hexdigest()[:20]


def summarize_meta(m: pm.PrettyMIDI) -> dict:
    """MIDIファイルからメタデータを抽出"""
    notes = 0
    vels = []
    durs = []
    patches = set()
    tempo_changes = len(m.get_tempo_changes()[0])
    pb_min = 0
    pb_max = 0
    cc_summary = {}

    for inst in m.instruments:
        # パッチ収集
        if inst.program is not None and not inst.is_drum:
            patches.add(int(inst.program))

        # ノート統計
        for n in inst.notes:
            notes += 1
            vels.append(n.velocity)
            durs.append(n.end - n.start)

        # ピッチベンド範囲
        for pb in inst.pitch_bends:
            pb_min = min(pb_min, pb.pitch)
            pb_max = max(pb_max, pb.pitch)

        # CC統計
        for cc in inst.control_changes:
            d = cc_summary.setdefault(int(cc.number), {"count": 0, "min": 127, "max": 0})
            d["count"] += 1
            d["min"] = min(d["min"], int(cc.value))
            d["max"] = max(d["max"], int(cc.value))

    vels = np.array(vels) if vels else np.array([0])
    durs = np.array(durs) if durs else np.array([0.0])

    return {
        "total_number_of_tracks": len(m.instruments),
        "total_number_of_notes": int(notes),
        "tempo_change_count": int(tempo_changes),
        "pb_range": [int(pb_min), int(pb_max)],
        "cc_summary": cc_summary,
        "midi_patches": sorted(list(patches)),
        "avg_velocity": float(vels.mean()),
        "avg_dur_ms": float(durs.mean() * 1000.0),
    }


def main():
    ap = argparse.ArgumentParser(
        description="LOCAL_META_DATA builder",
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
        help="Output pickle path (LOCAL_META_DATA_000001.pickle)",
    )
    args = ap.parse_args()

    # MIDI収集（除外ディレクトリをスキップ）
    midi_root = Path(args.midi_root)
    midi_files = sorted(set(iter_midis(midi_root)))
    print(f"📂 Found {len(midi_files)} MIDI files (excluding temp/quarantine/shards)")

    out = []
    for i, mp in enumerate(midi_files, 1):
        if i % 100 == 0:
            print(f"  Processing {i}/{len(midi_files)}...")

        try:
            m = pm.PrettyMIDI(str(mp))
            out.append([make_id(str(mp)), summarize_meta(m)])
        except Exception as e:
            print(f"[skip] {mp}: {e}")

    with open(args.out_pickle, "wb") as f:
        pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"\n✅ Wrote {len(out)} entries -> {args.out_pickle}")


if __name__ == "__main__":
    main()
