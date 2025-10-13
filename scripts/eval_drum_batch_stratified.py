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

# Bass constants
BASS_RANGE_MIN = 28  # E1
BASS_RANGE_MAX = 55  # G3


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


def guess_bass_track(pm: pretty_midi.PrettyMIDI) -> int:
    """
    Guess bass track by lowest average pitch (simple & fast).
    Returns track index or -1 if no bass found.
    """
    cand = []
    for idx, inst in enumerate(pm.instruments):
        if inst.is_drum:
            continue
        if inst.notes:
            avg_pitch = sum(n.pitch for n in inst.notes) / len(inst.notes)
            name = (inst.name or "").lower()
            # Prioritize tracks with "bass" in name
            priority = -10 if "bass" in name else 0
            cand.append((avg_pitch + priority, idx))
    if not cand:
        return -1
    cand.sort()
    return cand[0][1]


def file_metrics_bass(mid_path: Path, drum_ref_mid: Path = None) -> Dict[str, Any]:
    """
    Evaluate bass-specific metrics from MIDI file.

    Metrics:
      - downbeat_anchor_rate: Fraction of bars with note on downbeat
      - range_ok_rate: Fraction of notes in E1-G3 range
      - velocity_std: Standard deviation of velocities
      - notes_per_bar: Average notes per bar
      - kick_align_rate: Alignment with drum kick (optional)
    """
    pm = pretty_midi.PrettyMIDI(str(mid_path))
    meta_path = mid_path.with_suffix(".meta.json")
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}

    tempo = float(meta.get("tempo", 120))
    tsig = meta.get("time_sig", "4/4")
    bars = int(meta.get("length_bars", 16))
    barL = bar_len_sec(tempo, tsig)

    # Guess bass track
    b_idx = guess_bass_track(pm)
    if b_idx < 0:
        return {
            "file": str(mid_path),
            "tempo": tempo, "bars": bars, "time_sig": tsig,
            "downbeat_anchor_rate": 0.0,
            "range_ok_rate": 0.0,
            "velocity_std": 0.0,
            "notes_per_bar": 0.0,
            "kick_align_rate": None,
        }

    bass_notes = [{"start": n.start, "vel": n.velocity, "pitch": n.pitch}
                  for n in pm.instruments[b_idx].notes]
    bass_notes.sort(key=lambda x: (x["start"], x["pitch"]))

    # 1) downbeat_anchor_rate
    anchor_bars = 0
    for b in range(bars):
        t0 = b * barL
        if any(abs(n["start"] - t0) <= 0.035 for n in bass_notes):
            anchor_bars += 1
    downbeat_anchor_rate = anchor_bars / max(bars, 1)

    # 2) range_ok_rate
    in_range = sum(1 for n in bass_notes if BASS_RANGE_MIN <= n["pitch"] <= BASS_RANGE_MAX)
    range_ok_rate = in_range / max(len(bass_notes), 1)

    # 3) velocity_std
    vels = [n["vel"] for n in bass_notes]
    velocity_std = float(statistics.pstdev(vels)) if len(vels) > 1 else 0.0

    # 4) notes_per_bar
    notes_per_bar = len(bass_notes) / max(bars, 1)

    # 5) kick_align_rate (optional, requires drum reference)
    kick_align_rate = None
    if drum_ref_mid is not None and Path(drum_ref_mid).exists():
        try:
            dpm = pretty_midi.PrettyMIDI(str(drum_ref_mid))
            kicks = []
            for inst in dpm.instruments:
                if not inst.is_drum:
                    continue
                for n in inst.notes:
                    if n.pitch in (35, 36):  # KICK
                        kicks.append(n.start)
            kicks.sort()
            if kicks:
                import bisect
                aligned = 0
                for n in bass_notes:
                    i = bisect.bisect_left(kicks, n["start"])
                    cand = []
                    if i < len(kicks):
                        cand.append(kicks[i])
                    if i > 0:
                        cand.append(kicks[i - 1])
                    if cand and min(abs(n["start"] - t) for t in cand) <= 0.040:
                        aligned += 1
                kick_align_rate = aligned / max(len(bass_notes), 1)
        except Exception:
            pass  # Ignore errors in reference loading

    return {
        "file": str(mid_path),
        "tempo": tempo, "bars": bars, "time_sig": tsig,
        "downbeat_anchor_rate": round(downbeat_anchor_rate, 4),
        "range_ok_rate": round(range_ok_rate, 4),
        "velocity_std": round(velocity_std, 3),
        "notes_per_bar": round(notes_per_bar, 2),
        "kick_align_rate": None if kick_align_rate is None else round(kick_align_rate, 4),
    }


# ========== Piano Evaluator ==========

def guess_piano_track(pm: pretty_midi.PrettyMIDI) -> pretty_midi.Instrument:
    """Guess piano track by program number (0 or 1-7 for acoustic/electric piano)."""
    for inst in pm.instruments:
        if not inst.is_drum and 0 <= inst.program <= 7:
            return inst
    # Fallback: first non-drum track
    for inst in pm.instruments:
        if not inst.is_drum:
            return inst
    return pretty_midi.Instrument(program=0, is_drum=False)


def file_metrics_piano(mid_path, chord_attrs: List[str] = None) -> Dict[str, Any]:
    """Compute Piano metrics: chord_tone_rate, hand_separation, velocity_std, bar_violation_rate, notes_per_bar.
    
    Args:
        mid_path: Path to MIDI file (str or Path)
        chord_attrs: List of [chord:X] strings for chord tone rate calculation
    
    Returns:
        Dict with metrics
    """
    pm = pretty_midi.PrettyMIDI(str(mid_path))
    track = guess_piano_track(pm)
    notes = track.notes
    
    # Metadata
    tempo = pm.estimate_tempo() if pm.get_tempo_changes()[1] else 120.0
    tsig = "4/4"
    if pm.time_signature_changes:
        ts = pm.time_signature_changes[0]
        tsig = f"{ts.numerator}/{ts.denominator}"
    
    # Bar calculation
    num, den = parse_time_sig(tsig)
    bar_len_sec = num * (60.0 / tempo) * (4.0 / den)
    total_len = pm.get_end_time()
    bars = math.ceil(total_len / bar_len_sec) if bar_len_sec > 1e-6 else 1
    
    if not notes:
        return {
            "file": str(mid_path),
            "tempo": round(tempo, 1), "bars": bars, "time_sig": tsig,
            "chord_tone_rate": 0.0,
            "hand_separation": 1.0,
            "velocity_std": 0.0,
            "bar_violation_rate": 0.0,
            "notes_per_bar": 0.0,
        }
    
    # 1) chord_tone_rate: 各小節の和音音に対する一致率
    chord_tone_rate = 0.0
    if chord_attrs:
        # Parse [chord:X] → pitch classes
        chord_pcs = []
        for attr in chord_attrs:
            if attr.startswith("[chord:") and attr.endswith("]"):
                chord_name = attr[len("[chord:"):-1]
                chord_pcs.append(_name_to_pitch_classes(chord_name))
        
        if chord_pcs:
            matches = 0
            for note in notes:
                bar_idx = int(note.start / bar_len_sec) % len(chord_pcs)
                expected_pcs = chord_pcs[bar_idx]
                note_pc = note.pitch % 12
                if note_pc in expected_pcs:
                    matches += 1
            chord_tone_rate = matches / len(notes)
    else:
        # Fallback: 甘めの三和音近似 (Major/Minor triad)
        # 全ノートのピッチクラスの分布を見て、最も多い3つが三和音を形成するか
        from collections import Counter
        pc_counts = Counter(n.pitch % 12 for n in notes)
        top3 = [pc for pc, _ in pc_counts.most_common(3)]
        if len(top3) == 3:
            # Check if it's a triad (0-4-7 or 0-3-7 pattern)
            top3_sorted = sorted(top3)
            intervals = [(top3_sorted[(i+1)%3] - top3_sorted[i]) % 12 for i in range(3)]
            if sorted(intervals) in [[3, 4, 5], [3, 5, 4], [4, 3, 5], [4, 5, 3]]:
                # Likely a triad
                matches = sum(1 for n in notes if (n.pitch % 12) in top3)
                chord_tone_rate = matches / len(notes)
    
    # 2) hand_separation: 同一タイムスタンプで上下（C4=60境界）が混在していない率
    # Group notes by timestamp (quantized to 0.01s)
    from collections import defaultdict
    ts_groups = defaultdict(list)
    for note in notes:
        ts_key = round(note.start, 2)  # 10ms quantization
        ts_groups[ts_key].append(note.pitch)
    
    separated = 0
    for pitches in ts_groups.values():
        if len(pitches) == 1:
            separated += 1
        else:
            # Check if all above or all below C4 (60)
            above = all(p >= 60 for p in pitches)
            below = all(p < 60 for p in pitches)
            if above or below:
                separated += 1
    
    hand_separation = separated / len(ts_groups) if ts_groups else 1.0
    
    # 3) velocity_std: ヒューマナイザの効き
    velocities = [n.velocity for n in notes]
    velocity_std = statistics.stdev(velocities) if len(velocities) > 1 else 0.0
    
    # 4) bar_violation_rate: 小節外はみ出し率
    violations = sum(1 for n in notes if n.end > bars * bar_len_sec)
    bar_violation_rate = violations / len(notes)
    
    # 5) notes_per_bar: 密度
    notes_per_bar = len(notes) / max(bars, 1)
    
    return {
        "file": str(mid_path),
        "tempo": round(tempo, 1), "bars": bars, "time_sig": tsig,
        "chord_tone_rate": round(chord_tone_rate, 4),
        "hand_separation": round(hand_separation, 4),
        "velocity_std": round(velocity_std, 3),
        "bar_violation_rate": round(bar_violation_rate, 4),
        "notes_per_bar": round(notes_per_bar, 2),
    }


def _name_to_pitch_classes(name: str) -> List[int]:
    """Chord name → pitch classes (0-11)."""
    ROOTS = {
        "C": 0, "C#": 1, "Db": 1,
        "D": 2, "D#": 3, "Eb": 3,
        "E": 4,
        "F": 5, "F#": 6, "Gb": 6,
        "G": 7, "G#": 8, "Ab": 8,
        "A": 9, "A#": 10, "Bb": 10,
        "B": 11,
    }
    
    # Parse root
    root_str = "".join([c for c in name if c.isalpha() or c in "#b"])
    if root_str not in ROOTS:
        return [0, 4, 7]  # C major fallback
    
    # Quality
    minor = ("m" in name and "maj" not in name)
    ext7 = ("7" in name)
    
    root = ROOTS[root_str]
    tri = [0, 3, 7] if minor else [0, 4, 7]
    pcs = [(root + i) % 12 for i in tri]
    
    if ext7:
        pcs.append((root + (10 if minor else 11)) % 12)
    
    return pcs


def avg(rows, k, default=0.0):
    vals = [r[k] for r in rows if k in r]
    return round(sum(vals) / len(vals), 4) if vals else default


def summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not rows:
        return {"count": 0}
    
    # Detect instrument type from first row
    is_bass = "downbeat_anchor_rate" in rows[0]
    is_piano = "chord_tone_rate" in rows[0]
    
    summary = {"count": len(rows)}
    
    if is_piano:
        # Piano metrics
        summary.update({
            "chord_tone_rate": avg(rows, "chord_tone_rate"),
            "hand_separation": avg(rows, "hand_separation"),
            "velocity_std": round(sum(r["velocity_std"] for r in rows if "velocity_std" in r) / len(rows), 3),
            "bar_violation_rate": avg(rows, "bar_violation_rate"),
            "notes_per_bar": avg(rows, "notes_per_bar"),
        })
    elif is_bass:
        # Bass metrics
        summary.update({
            "downbeat_anchor_rate": avg(rows, "downbeat_anchor_rate"),
            "range_ok_rate": avg(rows, "range_ok_rate"),
            "velocity_std": round(sum(r["velocity_std"] for r in rows if "velocity_std" in r) / len(rows), 3),
            "notes_per_bar": avg(rows, "notes_per_bar"),
            "kick_align_rate": avg(rows, "kick_align_rate") if any("kick_align_rate" in r and r["kick_align_rate"] is not None for r in rows) else None,
        })
    else:
        # Drum metrics
        summary.update({
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
        })
    
    return summary


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
    ap.add_argument("--instrument", default="drum", choices=["drum", "bass", "piano"])
    args = ap.parse_args()

    A = gather_files(Path(args.dir_A))
    B = gather_files(Path(args.dir_B))
    tags = sorted(set(A.keys()) | set(B.keys()))

    per_file = []
    strata = {}
    for tag in tags:
        rowsA = []
        rowsB = []
        
        if args.instrument == "piano":
            # Piano metrics (chord progression from tag or fallback)
            chord_attrs = None  # TODO: extract from .meta.json or tag
            for mid in A.get(tag, []):
                rowsA.append(file_metrics_piano(mid, chord_attrs))
                per_file.append({"group": "A", "tag": tag, **rowsA[-1]})
            for mid in B.get(tag, []):
                rowsB.append(file_metrics_piano(mid, chord_attrs))
                per_file.append({"group": "B", "tag": tag, **rowsB[-1]})
        elif args.instrument == "bass":
            # Bass metrics
            for mid in A.get(tag, []):
                rowsA.append(file_metrics_bass(mid))
                per_file.append({"group": "A", "tag": tag, **rowsA[-1]})
            for mid in B.get(tag, []):
                rowsB.append(file_metrics_bass(mid))
                per_file.append({"group": "B", "tag": tag, **rowsB[-1]})
        else:
            # Drum metrics (existing)
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
            if args.instrument == "piano":
                cols = ["group", "tag", "file", "tempo", "bars", "time_sig",
                        "chord_tone_rate", "hand_separation", "velocity_std", "bar_violation_rate", "notes_per_bar"]
            elif args.instrument == "bass":
                cols = ["group", "tag", "file", "tempo", "bars", "time_sig",
                        "downbeat_anchor_rate", "range_ok_rate", "velocity_std", "notes_per_bar", "kick_align_rate"]
            else:
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
