#!/usr/bin/env python3
"""
LOCAL_KILO_CHORDS_DATA.pickle ビルダー

ステムMIDIから小節単位コード進行をKILO互換形式で生成。
公式KILOと同じAPI（lamda_sources）で読める。

出力形式:
[file_id, {
    "format": "local_kilo_v1",
    "tokens": [[bar_idx, token_id], ...],
    "bars": int,
    "unit": "bar"
}]

ID形式: LOCAL:<sha1>[:20]
"""
from __future__ import annotations

import argparse
import hashlib
import os
import pickle
from pathlib import Path

import pretty_midi as pm
import yaml

# 既存v2.6ユーティリティ
from scripts.lamda_v2.tempo_timing import build_beat_grid
from scripts.lamda_v2.chord_analyzer import extract_bar_chords

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


def make_local_id(midi_path: str, grid) -> str:
    """安定したLOCAL IDを生成"""
    h = hashlib.sha1()
    try:
        sz = os.path.getsize(midi_path)
    except Exception:
        sz = 0
    # path + size + bars + bpm0 で安定キー
    downbeats = grid.get("downbeats_ql", [])
    tempo_map = grid.get("tempo_map", [])
    bpm0 = int(tempo_map[0][1]) if tempo_map and len(tempo_map[0]) > 1 else 120
    key = f"{midi_path}|{sz}|{len(downbeats)}|{bpm0}"
    h.update(key.encode("utf-8"))
    return "LOCAL:" + h.hexdigest()[:20]


def load_token_map(path):
    """
    lamda_chords_token_map.yaml を読み込み、
    quality code → int のマップを返す。
    
    期待形式:
      qualities:
        code_map:
          0: "maj"
          1: "m"
          ...
    """
    if not path or not os.path.exists(path):
        return {}, -1
    
    with open(path, "r", encoding="utf-8") as f:
        y = yaml.safe_load(f)
    
    # qualities.code_map から逆マップ作成
    code_map = y.get("qualities", {}).get("code_map", {})
    # key: quality文字列 -> value: トークンID
    token_map = {v: int(k) for k, v in code_map.items()}
    unk = max(token_map.values()) + 1 if token_map else -1
    
    return token_map, unk


def encode_event_to_token(root: str, quality: str, token_map: dict, unk: int) -> int:
    """コードイベントをトークンIDにエンコード"""
    key = f"{root}:{quality or ''}".replace("::", ":")
    return token_map.get(key, unk)


def main():
    ap = argparse.ArgumentParser(
        description="LOCAL_KILO_CHORDS_DATA builder",
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
        help="Output pickle path (LOCAL_KILO_CHORDS_DATA.pickle)",
    )
    ap.add_argument(
        "--token-map",
        default=None,
        help="Token map YAML (lamda_chords_token_map.yaml互換、任意)",
    )
    ap.add_argument(
        "--shard-size",
        type=int,
        default=50000,
        help="Entries per shard (default: 50000)",
    )
    args = ap.parse_args()

    token_map, unk = load_token_map(args.token_map)
    print(f"📚 Token map loaded: {len(token_map)} tokens, UNK={unk}")

    # MIDI収集（除外ディレクトリをスキップ）
    midi_root = Path(args.midi_root)
    midi_files = sorted(set(iter_midis(midi_root)))
    print(f"📂 Found {len(midi_files)} MIDI files (excluding temp/quarantine/shards)")

    entries = []
    shard_idx = 0
    out_base = Path(args.out_pickle)

    def flush():
        nonlocal entries, shard_idx
        if not entries:
            return

        # シャーディング
        if shard_idx == 0 and not out_base.name.endswith("_%03d.pickle"):
            path = out_base
        else:
            path = out_base.with_name(out_base.stem + f"_{shard_idx:03d}.pickle")

        with open(path, "wb") as f:
            pickle.dump(entries, f, protocol=pickle.HIGHEST_PROTOCOL)

        print(f"✓ Wrote {len(entries)} entries -> {path}")
        shard_idx += 1
        entries = []

    # 処理
    for i, mp in enumerate(midi_files, 1):
        if i % 100 == 0:
            print(f"  Processing {i}/{len(midi_files)}...")

        try:
            m = pm.PrettyMIDI(str(mp))
            grid = build_beat_grid(m)
            chords = extract_bar_chords(m, grid.get("downbeats_ql", []))

            # トークン化
            tokens = []
            for bar_idx, ev in enumerate(chords.get("events", [])):
                tok = encode_event_to_token(
                    ev.get("root", "N"), ev.get("quality", ""), token_map, unk
                )
                tokens.append([bar_idx, int(tok)])

            fid = make_local_id(str(mp), grid)
            payload = {
                "format": "local_kilo_v1",
                "tokens": tokens,
                "bars": len(chords.get("events", [])),
                "unit": "bar",
            }
            entries.append([fid, payload])

        except Exception as e:
            print(f"[skip] {mp}: {e}")

        # シャード分割
        if len(entries) >= args.shard_size:
            flush()

    # 最終フラッシュ
    flush()
    print(f"\n✅ LOCAL_KILO build complete!")


if __name__ == "__main__":
    main()
