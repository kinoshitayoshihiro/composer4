#!/usr/bin/env python3
"""
LOCAL_ID_MAP.csv ビルダー

全MIDIファイルのLOCAL ID -> 相対パスのマッピングを生成。
Stage2でlocal_id -> 元ファイルパスを逆引きするために使用。

出力形式（CSV）:
local_id,relative_path
LOCAL:abc123...,stem1/track1.mid
LOCAL:def456...,stem2/track2.mid
...

ID生成ロジック: make_local_id()と完全一致させる必要がある
- KILO: path + size + bars + bpm0
- META/SIGNATURES/TOTALS: path + size（シンプル版）

ここでは共通性を保つため、全ビルダーで同一のID生成ロジックを使用。
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import os
from pathlib import Path

import pretty_midi as pm

# 既存v2.6ユーティリティ（KILO用）
from scripts.lamda_v2.tempo_timing import build_beat_grid

# 除外ディレクトリ
EXCLUDE_DIRS = {"temp", "quarantine", "shards", ".git", "__pycache__"}


def iter_midis(root: Path):
    """除外ディレクトリをスキップしてMIDIファイルを収集"""
    for p in root.rglob("*"):
        if p.suffix.lower() not in (".mid", ".midi"):
            continue
        if any(excl in p.parts for excl in EXCLUDE_DIRS):
            continue
        yield p


def make_local_id_kilo(midi_path: str, grid) -> str:
    """
    KILO互換のLOCAL ID生成（build_local_kilo.pyと同一ロジック）
    
    キー: path + size + bars + bpm0
    """
    h = hashlib.sha1()
    try:
        sz = os.path.getsize(midi_path)
    except Exception:
        sz = 0
    
    downbeats = grid.get("downbeats_ql", [])
    tempo_map = grid.get("tempo_map", [])
    bpm0 = int(tempo_map[0][1]) if tempo_map and len(tempo_map[0]) > 1 else 120
    key = f"{midi_path}|{sz}|{len(downbeats)}|{bpm0}"
    h.update(key.encode("utf-8"))
    return "LOCAL:" + h.hexdigest()[:20]


def make_local_id_simple(midi_path: str) -> str:
    """
    META/SIGNATURES/TOTALS互換のLOCAL ID生成（シンプル版）
    
    キー: path + size
    """
    h = hashlib.sha1()
    try:
        sz = os.path.getsize(midi_path)
    except Exception:
        sz = 0
    key = f"{midi_path}|{sz}"
    h.update(key.encode("utf-8"))
    return "LOCAL:" + h.hexdigest()[:20]


def main():
    ap = argparse.ArgumentParser(
        description="LOCAL_ID_MAP.csv builder",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--midi-root",
        required=True,
        help="MIDI root directory (recursive search)",
    )
    ap.add_argument(
        "--out-csv",
        required=True,
        help="Output CSV path (LOCAL_ID_MAP.csv)",
    )
    ap.add_argument(
        "--id-type",
        default="kilo",
        choices=["kilo", "simple"],
        help="ID generation type: 'kilo' (with bars/bpm) or 'simple' (path+size only)",
    )
    args = ap.parse_args()

    midi_root = Path(args.midi_root)
    midi_files = sorted(set(iter_midis(midi_root)))
    print(f"📂 Found {len(midi_files)} MIDI files (excluding temp/quarantine/shards)")

    rows = []
    for i, mp in enumerate(midi_files, 1):
        if i % 100 == 0:
            print(f"  Processing {i}/{len(midi_files)}...")

        try:
            rel_path = mp.relative_to(midi_root)
            
            if args.id_type == "kilo":
                # KILOスタイル（bars/bpm含む）
                m = pm.PrettyMIDI(str(mp))
                grid = build_beat_grid(m)
                local_id = make_local_id_kilo(str(mp), grid)
            else:
                # シンプル版（path+sizeのみ）
                local_id = make_local_id_simple(str(mp))
            
            rows.append([local_id, str(rel_path)])

        except Exception as e:
            print(f"[skip] {mp}: {e}")

    # CSV書き込み
    with open(args.out_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["local_id", "relative_path"])
        writer.writerows(rows)

    print(f"\n✅ Wrote {len(rows)} entries -> {args.out_csv}")


if __name__ == "__main__":
    main()
