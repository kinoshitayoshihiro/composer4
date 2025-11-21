#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CREPE F0 -> 度数推定（簡易HMMスムージング付）
入力: analysis/crepe_f0.parquet, analysis/sections.json, analysis/chordmap_locked.json
出力: analysis/f0_degrees.parquet, analysis/melody_hotspots.json
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import math
import numpy as np
import pandas as pd

PC_NAMES = np.array(["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"])


def hz_to_midi(hz):
    return 69 + 12 * np.log2(np.maximum(hz, 1e-6) / 440.0)


def nearest_pc(midi):
    return np.mod(np.round(midi).astype(int), 12)


DEGREE_TABLE = {
    # 12pc -> 12度数候補 (Cを主音とした場合) ※後でkey_pcで回す
    0: "1",
    1: "b2",
    2: "2",
    3: "#2",
    4: "3",
    5: "4",
    6: "#4",
    7: "5",
    8: "b6",
    9: "6",
    10: "b7",
    11: "7",
}


def pc_to_degree(pc: int, key_pc: int) -> str:
    rel = (pc - key_pc) % 12
    return DEGREE_TABLE[rel]


def hmm_smooth(degrees: np.ndarray, stay=0.92):
    # 度数シンボルを離散ラベルに映してHMM的スムージング（同値優先）
    uniq, inv = np.unique(degrees, return_inverse=True)
    n = len(inv)
    out = inv.copy()
    for i in range(1, n):
        if out[i] != out[i - 1]:
            # ランダムでなく確率 stay を優先（単純化）
            if np.random.rand() < stay:
                out[i] = out[i - 1]
    return uniq[out]


def find_hotspots(df, key_pc: int):
    # 9/#11/13近傍のヒット率をバー集計
    tens = {"9": {"d": ["2"]}, "#11": {"d": ["#4"]}, "13": {"d": ["6"]}}
    hot = {}
    for name, rule in tens.items():
        mask = df["degree"].isin(rule["d"])
        if "bar" in df.columns:
            rate = mask.groupby(df["bar"]).mean().fillna(0.0)
            hot[name] = {int(k): float(v) for k, v in rate.to_dict().items()}
        else:
            hot[name] = {}
    return hot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--f0-parquet", required=True)
    ap.add_argument("--sections", required=True)
    ap.add_argument("--locked-chordmap", required=True)
    ap.add_argument("--out-parquet", required=True)
    ap.add_argument("--out-hotspots", required=True)
    args = ap.parse_args()

    # 読み込み
    f0 = pd.read_parquet(args.f0_parquet)  # columns: time, hz, bar(任意)
    sections = json.loads(Path(args.sections).read_text())
    locked = json.loads(Path(args.locked_chordmap).read_text())

    # tonic推定（sectionsのkey_hintがあれば優先）
    # sections.jsonの構造に対応（{"sections": [...]} または [...] 形式）
    sections_list = sections.get("sections", sections) if isinstance(sections, dict) else sections
    key_pc = 0
    for s in sections_list:
        r = s.get("key_hint_root")
        if r:
            # 超簡易: C=0, C#=1 ...
            names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
            if r in names:
                key_pc = names.index(r)
                break

    # hz列名の正規化（"hz" or "f0_hz"）
    hz_col = "hz" if "hz" in f0.columns else ("f0_hz" if "f0_hz" in f0.columns else None)
    if hz_col is None:
        print("⚠️  No 'hz' or 'f0_hz' column found in F0 parquet")
        # ダミー出力
        f0["midi"] = 0
        f0["pc"] = 0
        f0["degree"] = "1"
        f0.to_parquet(args.out_parquet, index=False)
        Path(args.out_hotspots).write_text(
            json.dumps({"9": {}, "#11": {}, "13": {}}, ensure_ascii=False, indent=2)
        )
        return

    midi = hz_to_midi(f0[hz_col].to_numpy())
    pcs = nearest_pc(midi)
    degs = np.array([pc_to_degree(int(p), key_pc) for p in pcs])
    degs_s = hmm_smooth(degs, stay=0.92)

    out = f0.copy()
    out["midi"] = midi
    out["pc"] = pcs
    out["degree"] = degs_s
    out.to_parquet(args.out_parquet, index=False)

    # ホットスポット（bar単位）抽出
    if "bar" not in out.columns:
        # 10ms刻み→bars情報が無い場合は擬似bar=0
        out["bar"] = 0
    hot = find_hotspots(out, key_pc)
    Path(args.out_hotspots).write_text(json.dumps(hot, ensure_ascii=False, indent=2))

    print(f"✅ Generated F0 degrees: {args.out_parquet}")
    print(f"   Total frames: {len(out)}")
    print(f"   Unique degrees: {sorted(out['degree'].unique())}")
    print(f"✅ Generated melody hotspots: {args.out_hotspots}")
    for t, bars in hot.items():
        if bars:
            print(f"   {t}: {len(bars)} bars with hits")


if __name__ == "__main__":
    main()
