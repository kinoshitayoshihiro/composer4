#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
role_bars_index.py — bars_with_slots.parquet と sections/policy を合成して、
各楽器の bar 単位の活動目標を出す（activity_floor, density_target, register_pref）。

出力: role_bars/{instrument}_role_bars.parquet
  columns: [bar, section, active_prob, density_target, register_pref]
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Dict, Any

import pandas as pd
import yaml

DEFAULT_INSTRS = ["guitar", "piano", "strings", "bass", "drums"]


def load_sections(path: Path) -> pd.DataFrame:
    with open(path, "r", encoding="utf-8") as f:
        sec = json.load(f)
    # sections.json 例: [{"name": "verse", "start_bar": 8, "end_bar": 16}, ...]
    rows = []
    for s in sec:
        rows.append(
            {
                "section": s.get("name") or s.get("label"),
                "start_bar": int(s["start_bar"]),
                "end_bar": int(s["end_bar"]),
            }
        )
    return pd.DataFrame(rows)


def attach_sections(bars: pd.DataFrame, sections_df: pd.DataFrame) -> pd.DataFrame:
    bars = bars.copy()
    bars["section"] = None
    for _, r in sections_df.iterrows():
        mask = (bars.index >= r["start_bar"]) & (bars.index < r["end_bar"])
        bars.loc[mask, "section"] = r["section"]
    # Forward fill for gaps
    bars["section"].fillna(method="ffill", inplace=True)
    bars["section"].fillna("unknown", inplace=True)
    return bars


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bars", required=True, help="analysis/bars_with_slots.parquet")
    ap.add_argument("--sections", required=True, help="analysis/sections.json")
    ap.add_argument("--policy", required=True, help="policy/<song>.yaml")
    ap.add_argument("--outdir", default="role_bars", help="output dir")
    ap.add_argument("--instruments", nargs="*", default=DEFAULT_INSTRS)
    args = ap.parse_args()

    bars = pd.read_parquet(args.bars)
    sections_df = load_sections(Path(args.sections))
    bars = attach_sections(bars, sections_df)

    with open(args.policy, "r", encoding="utf-8") as f:
        policy = yaml.safe_load(f)

    # policy.sections.<name>.density_floor.{inst} を使って density_target を与える
    sec_cfg: Dict[str, Any] = policy.get("sections", {})
    Path(args.outdir).mkdir(parents=True, exist_ok=True)

    for inst in args.instruments:
        rows = []
        for bar_idx, r in bars.iterrows():
            sec = r.get("section", "unknown")
            secp = sec_cfg.get(sec, {})
            dens = None
            reg = None
            if isinstance(secp.get("density_floor"), dict):
                dens = float(secp["density_floor"].get(inst, 0.0))
            if isinstance(secp.get("register_pref"), dict):
                reg = secp["register_pref"].get(inst)
            # active_prob: density_floor をそのまま初期値に採用（0..1）
            rows.append(
                {
                    "bar": int(bar_idx),
                    "section": sec,
                    "active_prob": float(dens if dens is not None else 0.0),
                    "density_target": float(dens if dens is not None else 0.0),
                    "register_pref": reg or "auto",
                }
            )
        df = pd.DataFrame(rows)
        outp = Path(args.outdir) / f"{inst}_role_bars.parquet"
        df.to_parquet(outp, index=False)
        print(f"✅ wrote {outp} ({len(df)} rows)")


if __name__ == "__main__":
    main()
