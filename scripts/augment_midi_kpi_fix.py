#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
augment_midi_kpi_fix.py
Reduce remaining KPI fails by adding minimal HH hits (density floor) and lifting snare backbeat velocities.
Assumes 4/4; uses pretty_midi.
"""
import argparse, random
from dataclasses import dataclass
from typing import Optional
import pretty_midi, numpy as np

KICK_PITCHES = (35, 36)
SNARE_PITCHES = (38, 40)
HAT_PITCHES = (42, 44, 46)

@dataclass
class Config:
    min_hat_density: float = 2.0
    hat_pitch: int = 42
    hat_vel: int = 54
    hat_microtiming_ms: float = 6.0
    backbeat_vel_floor: int = 56
    backbeat_window_ms: float = 45.0
    tempo_bpm: Optional[float] = None
    max_edits_per_bar: int = 2
    collision_ms: float = 30.0

def tempo_at_time(change_times, tempi, t_sec: float, fallback: float = 120.0) -> float:
    if len(tempi) == 0:
        return fallback
    idx = np.searchsorted(change_times, t_sec, side="right") - 1
    idx = int(np.clip(idx, 0, len(tempi)-1))
    return float(tempi[idx])

def extract_drums(pm: pretty_midi.PrettyMIDI):
    drums = [inst for inst in pm.instruments if inst.is_drum]
    notes = []
    for inst in drums:
        notes.extend(inst.notes)
    notes.sort(key=lambda n: n.start)
    return drums, notes

def within_ms(a: float, b: float, ms: float) -> bool:
    return abs(a - b) * 1000.0 <= ms

def bar_iter(total_duration: float, bar_duration: float):
    import math
    num = int(math.ceil(total_duration / bar_duration))
    for i in range(num):
        yield i, i*bar_duration, (i+1)*bar_duration

def add_hh(pm, drums, start, end, cfg: Config, sec_per_beat: float) -> int:
    hh = []
    for inst in drums:
        for n in inst.notes:
            if start <= n.start < end and n.pitch in HAT_PITCHES:
                hh.append(n)
    density = float(len(hh))
    edits = 0
    if density >= cfg.min_hat_density:
        return edits

    candidates_beats = [0.5, 2.5, 1.5, 3.5]
    cand_times = [start + b*sec_per_beat for b in candidates_beats]

    target_inst = drums[0] if drums else None
    if target_inst is None:
        return edits

    for ct in cand_times:
        if density >= cfg.min_hat_density or edits >= cfg.max_edits_per_bar:
            break
        conflict = any(within_ms(ct, n.start, cfg.collision_ms) for n in hh)
        if conflict:
            continue
        jitter = (random.random()*2 - 1) * (cfg.hat_microtiming_ms/1000.0)
        st = max(start, min(ct + jitter, end - 0.01))
        en = min(st + 0.06, end)
        n = pretty_midi.Note(velocity=int(min(127, max(1, cfg.hat_vel))), pitch=int(cfg.hat_pitch), start=float(st), end=float(en))
        target_inst.notes.append(n)
        hh.append(n); density += 1.0; edits += 1
    return edits

def lift_snare(pm, drums, start, end, cfg: Config, sec_per_beat: float) -> int:
    beats = [start + 1*sec_per_beat, start + 3*sec_per_beat]
    edits = 0
    for inst in drums:
        for n in inst.notes:
            if n.pitch not in SNARE_PITCHES:
                continue
            if any(within_ms(n.start, b, cfg.backbeat_window_ms) for b in beats):
                if n.velocity < cfg.backbeat_vel_floor:
                    n.velocity = int(min(127, cfg.backbeat_vel_floor))
                    edits += 1
    return edits

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--tempo-bpm', type=float, default=None)
    ap.add_argument('--min-hat-density', type=float, default=2.0)
    ap.add_argument('--hat-pitch', type=int, default=42)
    ap.add_argument('--hat-vel', type=int, default=54)
    ap.add_argument('--hat-microtiming-ms', type=float, default=6.0)
    ap.add_argument('--backbeat-vel-floor', type=int, default=56)
    ap.add_argument('--backbeat-window-ms', type=float, default=45.0)
    ap.add_argument('--max-edits-per-bar', type=int, default=2)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    random.seed(args.seed)

    cfg = Config(
        min_hat_density=args.min_hat_density,
        hat_pitch=args.hat_pitch,
        hat_vel=args.hat_vel,
        hat_microtiming_ms=args.hat_microtiming_ms,
        backbeat_vel_floor=args.backbeat_vel_floor,
        backbeat_window_ms=args.backbeat_window_ms,
        tempo_bpm=args.tempo_bpm,
        max_edits_per_bar=args.max_edits_per_bar,
    )

    pm = pretty_midi.PrettyMIDI(args.input)
    drums, _ = extract_drums(pm)
    if not drums:
        raise SystemExit("No drum track (is_drum=True) found.")
    change_times, tempi = pm.get_tempo_changes()
    base_tempo = cfg.tempo_bpm if cfg.tempo_bpm is not None else tempo_at_time(change_times, tempi, 0.0, 120.0)
    bar_duration = 60.0 / base_tempo * 4.0
    total = pm.get_end_time()

    total_hh = total_sn = 0
    for i, start, end in bar_iter(total, bar_duration):
        local_tempo = cfg.tempo_bpm if cfg.tempo_bpm is not None else tempo_at_time(change_times, tempi, (start+end)/2.0, base_tempo)
        spb = 60.0 / local_tempo
        total_hh += add_hh(pm, drums, start, end, cfg, spb)
        total_sn += lift_snare(pm, drums, start, end, cfg, spb)

    pm.write(args.output)
    print(f"✅ Wrote: {args.output}")
    print(f"   HH edits:    {total_hh}")
    print(f"   Snare edits: {total_sn}")

if __name__ == "__main__":
    main()
