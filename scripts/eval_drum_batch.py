#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch evaluator for drum generator outputs.
- Input: a directory containing *.mid and paired *.meta.json
- Output: JSON summary + optional CSV with per-file metrics

Metrics (per-file, then averaged):
- hat_grid_conform: Hi-hat onset grid conformity (0..1)
- snare_backbeat_rate: fraction of bars with snare on beats 2 & 4 (4/4)
- kick_downbeat_rate: fraction of bars with kick on bar downbeat
- bar_violation_rate: onsets outside expected song window [0, bars*bar_len)
- velocity_std: std of velocities across drum hits
- notes_per_bar: average total hits per bar
"""

from __future__ import annotations
import argparse
import json
import statistics
import csv
from pathlib import Path
from typing import Dict, Any, List, Tuple
import pretty_midi

# --- GM Drum pitch → ROLE ---
GM_ROLE = {
    # Kick
    35: "KICK", 36: "KICK",
    # Snare
    38: "SNARE", 40: "SNARE",
    # Hi-hat
    42: "HIHAT", 44: "HIHAT", 46: "HIHAT",
    # Toms
    41: "TOM", 43: "TOM", 45: "TOM", 47: "TOM", 48: "TOM",
    50: "TOM",
    # Cymbals
    49: "CRASH", 57: "CRASH", 55: "CRASH", 52: "CRASH",
    51: "RIDE", 59: "RIDE", 53: "RIDE",
}


def parse_time_sig(s: str) -> Tuple[int, int]:
    try:
        num, den = s.strip().split("/")
        return int(num), int(den)
    except Exception:
        return 4, 4


def bar_length_seconds(bpm: float, tsig: str) -> float:
    num, den = parse_time_sig(tsig)
    sec_per_beat = 60.0 / float(bpm)
    return num * sec_per_beat * (4.0 / den)


def nearest_grid_delta(t: float, grid: List[float]) -> float:
    import bisect
    i = bisect.bisect_left(grid, t)
    candidates = []
    if i < len(grid):
        candidates.append(grid[i])
    if i > 0:
        candidates.append(grid[i-1])
    return min((abs(t - g) for g in candidates), default=1e9)


def make_bar_grid(length_bars: int, bar_len: float, steps_per_bar: int) -> List[float]:
    grid = []
    for b in range(length_bars):
        start = b * bar_len
        step = bar_len / steps_per_bar
        for k in range(steps_per_bar):
            grid.append(start + k * step)
    return grid


def collect_drum_notes(pm: pretty_midi.PrettyMIDI) -> List[Dict[str, Any]]:
    notes = []
    for inst in pm.instruments:
        if not inst.is_drum:
            continue
        for n in inst.notes:
            role = GM_ROLE.get(n.pitch, "OTHER")
            notes.append({
                "start": n.start,
                "end": n.end,
                "vel": n.velocity,
                "pitch": n.pitch,
                "role": role
            })
    notes.sort(key=lambda x: (x["start"], x["pitch"]))
    return notes


def metrics_for_file(mid_path: Path) -> Dict[str, Any]:
    meta_path = mid_path.with_suffix(".meta.json")
    if not meta_path.exists():
        raise FileNotFoundError(f"Sidecar meta not found: {meta_path}")

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    tempo = float(meta.get("tempo", 120))
    tsig = meta.get("time_sig", "4/4")
    bars = int(meta.get("length_bars", 32))
    style = meta.get("style", "pop_straight")

    bar_len = bar_length_seconds(tempo, tsig)
    song_len = bars * bar_len

    pm = pretty_midi.PrettyMIDI(str(mid_path))
    notes = collect_drum_notes(pm)

    # role splits
    hats = [n for n in notes if n["role"] == "HIHAT"]
    snares = [n for n in notes if n["role"] == "SNARE"]
    kicks = [n for n in notes if n["role"] == "KICK"]

    # grid definition
    if style == "shuffle":
        steps = 12
        eps = 0.030
    elif style == "rock":
        steps = 8
        eps = 0.025
    else:
        steps = 8
        eps = 0.020
    grid = make_bar_grid(bars, bar_len, steps)

    # hat grid conformity
    hat_hits = len(hats)
    hat_on = 0
    for h in hats:
        if 0 <= h["start"] < song_len and nearest_grid_delta(h["start"], grid) <= eps:
            hat_on += 1
    hat_grid_conform = (hat_on / hat_hits) if hat_hits > 0 else 1.0

    # snare backbeat (4/4 only)
    num, den = parse_time_sig(tsig)
    beats = num * (4.0/den)
    backbeat_ok = 0
    for b in range(bars):
        bar_start = b * bar_len
        tgt = []
        if int(beats) >= 4:
            tgt = [
                bar_start + bar_len * (1.0/beats),  # beat 2
                bar_start + bar_len * (3.0/beats)   # beat 4
            ]
        else:
            tgt = [bar_start + bar_len * 0.5]
        ok = any(
            min(abs(s["start"]-t) for t in tgt) <= 0.035 and 
            0 <= s["start"] < (bar_start+bar_len)
            for s in snares
        )
        if ok:
            backbeat_ok += 1
    snare_backbeat_rate = backbeat_ok / max(bars, 1)

    # kick downbeat rate
    kd_ok = 0
    for b in range(bars):
        t0 = b * bar_len
        ok = any(abs(k["start"] - t0) <= 0.035 for k in kicks)
        if ok:
            kd_ok += 1
    kick_downbeat_rate = kd_ok / max(bars, 1)

    # bar violation
    violations = sum(1 for n in notes if not (0 <= n["start"] < song_len))
    bar_violation_rate = violations / max(len(notes), 1)

    # velocity std
    vels = [n["vel"] for n in notes]
    velocity_std = float(statistics.pstdev(vels)) if len(vels) > 1 else 0.0

    # notes per bar
    notes_per_bar = len(notes) / max(bars, 1)

    return {
        "file": str(mid_path),
        "tempo": tempo,
        "time_sig": tsig,
        "bars": bars,
        "style": style,
        "hat_grid_conform": round(hat_grid_conform, 4),
        "snare_backbeat_rate": round(snare_backbeat_rate, 4),
        "kick_downbeat_rate": round(kick_downbeat_rate, 4),
        "bar_violation_rate": round(bar_violation_rate, 6),
        "velocity_std": round(velocity_std, 3),
        "notes_per_bar": round(notes_per_bar, 2),
    }


def summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    def avg(k, default=0.0):
        vals = [r[k] for r in rows if k in r]
        return round(sum(vals)/len(vals), 4) if vals else default
    
    return {
        "count": len(rows),
        "hat_grid_conform": avg("hat_grid_conform"),
        "snare_backbeat_rate": avg("snare_backbeat_rate"),
        "kick_downbeat_rate": avg("kick_downbeat_rate"),
        "bar_violation_rate": avg("bar_violation_rate"),
        "velocity_std": round(sum(r["velocity_std"] for r in rows)/max(len(rows),1), 3),
        "notes_per_bar": avg("notes_per_bar"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", required=True, help="Directory with *.mid + *.meta.json")
    ap.add_argument("--output-json", required=True)
    ap.add_argument("--output-csv", default="")
    args = ap.parse_args()

    in_dir = Path(args.input_dir)
    mids = sorted(in_dir.glob("*.mid"))
    rows = [metrics_for_file(p) for p in mids]
    summary = summarize(rows)
    out = {"summary": summary, "files": rows}

    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).write_text(
        json.dumps(out, ensure_ascii=False, indent=2), 
        encoding="utf-8"
    )

    if args.output_csv:
        with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
            if rows:
                w.writeheader()
                w.writerows(rows)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
