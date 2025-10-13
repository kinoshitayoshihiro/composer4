#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Stratified batch evaluator for A/B outputs.

Input structure (by gen_ab_stratified.py):
  output/
    drumgen_A/<tag>/*.mid
    drumgen_B/<tag>/*.mid

Output:
  --out-json summary with:
    - overall.summary.{A,B}
    - strata[ tag ].{A,B}.summary
  --out-csv  per-file rows (A/B with tag column)
"""

from __future__ import annotations
import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from typing import Tuple

import pretty_midi

GM_ROLE = {
    35: "KICK", 36: "KICK",
    38: "SNARE", 40: "SNARE",
    42: "HIHAT", 44: "HIHAT", 46: "HIHAT",
    41: "TOM", 43: "TOM", 45: "TOM", 47: "TOM", 48: "TOM", 50: "TOM",
    49: "CRASH", 57: "CRASH", 55: "CRASH", 52: "CRASH",
    51: "RIDE", 59: "RIDE", 53: "RIDE",
}


def parse_time_sig(s: str) -> Tuple[int, int]:
    try:
        a, b = s.split("/")
        return int(a), int(b)
    except Exception:
        return 4, 4


def bar_len_sec(bpm: float, tsig: str) -> float:
    num, den = parse_time_sig(tsig)
    return num * (60.0 / float(bpm)) * (4.0 / den)


def collect_notes(pm: pretty_midi.PrettyMIDI) -> List[Dict[str, Any]]:
    out = []
    for inst in pm.instruments:
        if not inst.is_drum:
            continue
        for n in inst.notes:
            out.append({
                "start": n.start, "end": n.end, "vel": n.velocity,
                "pitch": n.pitch, "role": GM_ROLE.get(n.pitch, "OTHER")
            })
    out.sort(key=lambda x: (x["start"], x["pitch"]))
    return out


def nearest_delta(t: float, grid: List[float]) -> float:
    import bisect
    i = bisect.bisect_left(grid, t)
    cand = []
    if i < len(grid):
        cand.append(grid[i])
    if i > 0:
        cand.append(grid[i - 1])
    return min((abs(t - g) for g in cand), default=1e9)


def make_grid(bars: int, bar_len: float, steps: int) -> List[float]:
    g = []
    step = bar_len / steps
    for b in range(bars):
        t0 = b * bar_len
        for k in range(steps):
            g.append(t0 + k * step)
    return g


def file_metrics(mid_path: Path, style_hint: str) -> Dict[str, Any]:
    pm = pretty_midi.PrettyMIDI(str(mid_path))
    meta_path = mid_path.with_suffix(".meta.json")
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    tempo = float(meta.get("tempo", 120))
    tsig = meta.get("time_sig", "4/4")
    bars = int(meta.get("length_bars", 16))
    barL = bar_len_sec(tempo, tsig)
    songL = bars * barL

    notes = collect_notes(pm)
    hats = [n for n in notes if n["role"] == "HIHAT"]
    snares = [n for n in notes if n["role"] == "SNARE"]
    kicks = [n for n in notes if n["role"] == "KICK"]
    crashes = [n for n in notes if n["role"] == "CRASH"]
    toms = [n for n in notes if n["role"] == "TOM"]

    # style→grid設定
    if style_hint == "shuffle":
        steps, eps = 12, 0.030
    elif style_hint == "rock":
        steps, eps = 8, 0.025
    else:
        steps, eps = 8, 0.020
    grid = make_grid(bars, barL, steps)

    # hat grid
    hat_on = sum(1 for h in hats if 0 <= h["start"] < songL and nearest_delta(h["start"], grid) <= eps)
    hat_grid = (hat_on / len(hats)) if hats else 1.0

    # backbeat（2&4近傍）
    num, den = parse_time_sig(tsig)
    quarters = num * (4.0 / den)
    backbeats = []
    if int(quarters) >= 4:
        for b in range(bars):
            t0 = b * barL
            backbeats += [t0 + barL * (1.0 / quarters), t0 + barL * (3.0 / quarters)]
    else:
        for b in range(bars):
            t0 = b * barL
            backbeats.append(t0 + barL * 0.5)
    bb_bars = 0
    for b in range(bars):
        t0 = b * barL
        t1 = t0 + barL
        tgt = [t for t in backbeats if t0 <= t < t1]
        ok = any(min(abs(s["start"] - t) for t in tgt) <= 0.035 for s in snares)
        if ok:
            bb_bars += 1
    snare_backbeat = bb_bars / max(bars, 1)

    # kick downbeat
    kd_bars = 0
    for b in range(bars):
        t0 = b * barL
        if any(abs(k["start"] - t0) <= 0.035 for k in kicks):
            kd_bars += 1
    kick_downbeat = kd_bars / max(bars, 1)

    # bar violation
    violations = sum(1 for n in notes if not (0 <= n["start"] < songL))
    bar_violation = violations / max(len(notes), 1)

    # velocity std
    vels = [n["vel"] for n in notes]
    vel_std = float(statistics.pstdev(vels)) if len(vels) > 1 else 0.0

    # densities per bar (role-wise)
    def role_density(role_list):
        if not role_list:
            return 0.0
        return len(role_list) / max(bars, 1)

    dens = {
        "notes_per_bar": len(notes) / max(bars, 1),
        "kick_per_bar": role_density(kicks),
        "snare_per_bar": role_density(snares),
        "hihat_per_bar": role_density(hats),
        "crash_per_bar": role_density(crashes),
        "tom_per_bar": role_density(toms),
    }

    # crash_on_bar1_rate
    c1 = 0
    for b in range(bars):
        t0 = b * barL
        if any(abs(c["start"] - t0) <= 0.05 for c in crashes):
            c1 += 1
    crash_on_bar1 = c1 / max(bars, 1)

    # fill_coverage: 最終1/4小節に CRASH/TOM または 急増密度がある割合
    fill_bars = 0
    for b in range(bars):
        t0 = b * barL
        t1 = t0 + barL
        win0 = t0 + 0.75 * barL
        hits = [n for n in notes if win0 <= n["start"] < t1]
        cond_any = any(n["role"] in {"CRASH", "TOM"} for n in hits)
        cond_dense = (len(hits) >= max(3, int(0.2 * dens["notes_per_bar"] * bars)))
        if cond_any or cond_dense:
            fill_bars += 1
    fill_cov = fill_bars / max(bars, 1)

    return {
        "file": str(mid_path),
        "style_hint": style_hint,
        "tempo": meta.get("tempo", 120),
        "bars": meta.get("length_bars", 16),
        "hat_grid_conform": round(hat_grid, 4),
        "snare_backbeat_rate": round(snare_backbeat, 4),
        "kick_downbeat_rate": round(kick_downbeat, 4),
        "bar_violation_rate": round(bar_violation, 6),
        "velocity_std": round(vel_std, 3),
        "notes_per_bar": round(dens["notes_per_bar"], 2),
        "kick_per_bar": round(dens["kick_per_bar"], 2),
        "snare_per_bar": round(dens["snare_per_bar"], 2),
        "hihat_per_bar": round(dens["hihat_per_bar"], 2),
        "crash_per_bar": round(dens["crash_per_bar"], 2),
        "tom_per_bar": round(dens["tom_per_bar"], 2),
        "crash_on_bar1_rate": round(crash_on_bar1, 4),
        "fill_coverage_rate": round(fill_cov, 4),
    }


def avg(rows, k, default=0.0):
    vals = [r[k] for r in rows if k in r]
    return round(sum(vals) / len(vals), 4) if vals else default


def summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not rows:
        return {"count": 0}
    return {
        "count": len(rows),
        "hat_grid_conform": avg(rows, "hat_grid_conform"),
        "snare_backbeat_rate": avg(rows, "snare_backbeat_rate"),
        "kick_downbeat_rate": avg(rows, "kick_downbeat_rate"),
        "bar_violation_rate": avg(rows, "bar_violation_rate"),
        "velocity_std": round(sum(r["velocity_std"] for r in rows) / len(rows), 3),
        "notes_per_bar": avg(rows, "notes_per_bar"),
        "kick_per_bar": avg(rows, "kick_per_bar"),
        "snare_per_bar": avg(rows, "snare_per_bar"),
        "hihat_per_bar": avg(rows, "hihat_per_bar"),
        "crash_per_bar": avg(rows, "crash_per_bar"),
        "tom_per_bar": avg(rows, "tom_per_bar"),
        "crash_on_bar1_rate": avg(rows, "crash_on_bar1_rate"),
        "fill_coverage_rate": avg(rows, "fill_coverage_rate"),
    }


def gather_files(root: Path) -> Dict[str, List[Path]]:
    """
    root/<tag>/*.mid を収集 → {tag: [paths...]}
    """
    out = {}
    for tag_dir in sorted(root.glob("*")):
        if not tag_dir.is_dir():
            continue
        mids = sorted(tag_dir.glob("*.mid"))
        if mids:
            out[tag_dir.name] = mids
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir-A", required=True)
    ap.add_argument("--dir-B", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-csv", default="")
    args = ap.parse_args()

    A = gather_files(Path(args.dir_A))
    B = gather_files(Path(args.dir_B))
    tags = sorted(set(A.keys()) | set(B.keys()))

    per_file = []
    strata = {}
    for tag in tags:
        rowsA = []
        rowsB = []
        for mid in A.get(tag, []):
            style_hint = tag.split("_")[0] if "_" in tag else "pop_straight"
            rowsA.append(file_metrics(mid, style_hint))
            per_file.append({"group": "A", "tag": tag, **rowsA[-1]})
        for mid in B.get(tag, []):
            style_hint = tag.split("_")[0] if "_" in tag else "pop_straight"
            rowsB.append(file_metrics(mid, style_hint))
            per_file.append({"group": "B", "tag": tag, **rowsB[-1]})
        strata[tag] = {"A": {"summary": summarize(rowsA), "count": len(rowsA)},
                       "B": {"summary": summarize(rowsB), "count": len(rowsB)}}

    overallA = summarize([r for r in per_file if r["group"] == "A"])
    overallB = summarize([r for r in per_file if r["group"] == "B"])

    out = {"overall": {"A": overallA, "B": overallB},
           "strata": strata,
           "counts": {"A": overallA.get("count", 0), "B": overallB.get("count", 0)}}

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.out_csv:
        with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
            cols = ["group", "tag", "file", "style_hint", "tempo", "bars",
                    "hat_grid_conform", "snare_backbeat_rate", "kick_downbeat_rate", "bar_violation_rate",
                    "velocity_std", "notes_per_bar", "kick_per_bar", "snare_per_bar", "hihat_per_bar", "crash_per_bar", "tom_per_bar",
                    "crash_on_bar1_rate", "fill_coverage_rate"]
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for r in per_file:
                w.writerow({k: r.get(k, "") for k in cols})

    print(f"✅ Wrote: {args.out_json}")


if __name__ == "__main__":
    main()
