#!/usr/bin/env python3
# scripts/backfill_bar_beat.py
import json
import math
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

START_KEYS = [
    "start_beats", "start_beat", "start",
    "offset_beats", "offset_ql", "offset"
]

def _beats_per_bar_from_ts(ts: str) -> float:
    # QL(四分音符)基準の"拍数"に正規化
    if not ts:
        return 4.0
    ts = str(ts).strip()
    if ts in ("4/4", "2/2"):  # 2/2 も 4 QL 相当扱いで安定運用
        return 4.0
    if ts == "3/4":
        return 3.0
    if ts == "5/4":
        return 5.0
    if ts == "6/8":
        # 6/8 は実運用上 "3 拍（付点 4 分）"として 3.0 が安定
        return 3.0
    # 未知は 4/4 相当
    return 4.0

def _load_plan(path: Path):
    data = json.loads(path.read_text(encoding="utf-8"))
    # 形： {events:[...]} / [...]/ {tracks:[{events:...},...]}
    if isinstance(data, dict) and "tracks" in data:
        # full_arrangement 形式
        tracks = []
        for tr in data["tracks"]:
            evs = tr.get("events", [])
            tracks.append((tr, evs))
        return ("tracks", data, tracks)
    elif isinstance(data, dict) and "events" in data:
        return ("events_obj", data, data["events"])
    elif isinstance(data, list):
        return ("events_list", data, data)
    else:
        raise ValueError("Unsupported plan JSON structure")

def _save_plan(kind, root, events, path: Path):
    if kind == "tracks":
        # events は参照で更新済み
        path.write_text(json.dumps(root, ensure_ascii=False, indent=2), encoding="utf-8")
    elif kind == "events_obj":
        root["events"] = events
        path.write_text(json.dumps(root, ensure_ascii=False, indent=2), encoding="utf-8")
    elif kind == "events_list":
        path.write_text(json.dumps(events, ensure_ascii=False, indent=2), encoding="utf-8")

def _extract_start_beats(e: dict):
    for k in START_KEYS:
        if k in e:
            try:
                return float(e[k])
            except Exception:
                pass
    return None

def backfill(plan_path: Path, bars_path: Path, out_path: Path):
    kind, root, payload = _load_plan(plan_path)

    bars = pd.read_parquet(bars_path)
    # 列名正規化
    if "bar_index" in bars.columns:
        bars["bar"] = bars["bar_index"]
    elif "bar" not in bars.columns:
        bars["bar"] = np.arange(len(bars))

    if "start_beat" not in bars.columns:
        # 旧最小 bars の場合は bar * beats_per_bar を復元
        beats_per_bar = _beats_per_bar_from_ts(str(bars.get("time_signature", "4/4").iloc[0] if len(bars) else "4/4"))
        bars["start_beat"] = bars["bar"].astype(float) * float(beats_per_bar)
    if "end_beat" not in bars.columns:
        bpb = _beats_per_bar_from_ts(str(bars.get("time_signature", "4/4").iloc[0] if len(bars) else "4/4"))
        bars["end_beat"] = bars["start_beat"] + float(bpb)

    # 参照配列
    starts = bars["start_beat"].to_numpy(dtype=float)
    ends   = bars["end_beat"].to_numpy(dtype=float)
    ts_col = bars["time_signature"].astype(str).to_numpy() if "time_signature" in bars.columns else np.array(["4/4"] * len(bars))
    sec_col = bars["section_label"].astype(str).to_numpy() if "section_label" in bars.columns else None

    def assign_bar_beat(e: dict):
        if "bar" in e and "beat" in e:
            return e  # すでに OK

        sb = _extract_start_beats(e)
        if sb is None:
            # どうしてもわからない場合は 0 にフォールバック
            e.setdefault("bar", 0)
            e.setdefault("beat", 0.0)
            return e

        # 区間検索（start_beat <= sb < end_beat）
        # starts が単調増加である前提
        idx = np.searchsorted(starts, sb, side="right") - 1
        if idx < 0:
            idx = 0
        elif idx >= len(starts):
            idx = len(starts) - 1

        # sb が bar の end を越えるケースをクランプ
        if sb >= ends[idx] and idx + 1 < len(starts):
            idx += 1

        bar_num = int(bars["bar"].iloc[idx])
        bar_start = float(starts[idx])
        ts = ts_col[idx] if idx < len(ts_col) else "4/4"
        bpb = _beats_per_bar_from_ts(ts)

        beat_in_bar = float(sb - bar_start)
        # 拍内へクランプ
        if beat_in_bar < 0:
            beat_in_bar = 0.0
        if beat_in_bar >= bpb:
            beat_in_bar = max(0.0, bpb - 1e-6)

        e["bar"] = int(bar_num)
        e.setdefault("beat", round(beat_in_bar, 6))
        if sec_col is not None and "section" not in e:
            e["section"] = sec_col[idx]
        return e

    if kind == "tracks":
        for tr, evs in payload:
            for ev in evs:
                assign_bar_beat(ev)
    else:
        for ev in payload:
            assign_bar_beat(ev)

    _save_plan(kind, root, payload, out_path)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", required=True, help="path to *_plan.json or full_arrangement.json (per-track supported)")
    ap.add_argument("--bars", required=True, help="bars.parquet with start_beat/end_beat/time_signature")
    ap.add_argument("--out", default=None, help="output path (default: overwrite)")
    args = ap.parse_args()

    plan_p = Path(args.plan)
    out_p = Path(args.out) if args.out else plan_p
    backfill(plan_p, Path(args.bars), out_p)
    print(f"✅ backfilled bar/beat → {out_p}")
