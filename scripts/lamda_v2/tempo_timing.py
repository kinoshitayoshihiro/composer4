#!/usr/bin/env python3
from __future__ import annotations

"""
Lamda v2 — Phase1: Tempo/Timing primitives (minimal, production-safe).

Provided APIs
=============
- build_beat_grid(pm) -> dict with keys:
    tempo_map:     list[(time_sec: float, bpm: float)]
    timesig_map:   list[(index: int, sig: str)]           # coarse, from PM
    downbeats_sec: list[float]
    downbeats_ql:  list[float]                            # QL = quarter-length

- sec_to_ql(sec, tempo_map) -> float
- ql_to_sec(ql, tempo_map) -> float
- merge_min_dwell(events, min_ql=2.0) -> list[dict]
- snap_times_to_grid(times_ql, grid_ql) -> list[float]

Notes
=====
- Piecewise-constant tempo integration (exact for PM tempo steps).
- Safe defaults: if no tempo_map, assume 120 BPM.
- No external deps besides pretty_midi (for build_beat_grid).
"""
from typing import List, Tuple, Dict, Any

# ---------- core conversions ----------


def _tempo_map_or_default(tempo_map: List[Tuple[float, float]] | None) -> List[Tuple[float, float]]:
    if not tempo_map:
        return [(0.0, 120.0)]
    # ensure sorted and bpm > 0
    tmap = sorted((float(t), max(1e-6, float(b))) for t, b in tempo_map)
    if tmap[0][0] > 0.0:
        # prepend guard if first change isn't at 0
        tmap = [(0.0, tmap[0][1])] + tmap
    return tmap


def sec_to_ql(sec: float, tempo_map: List[Tuple[float, float]] | None) -> float:
    """Integrate piecewise-constant tempo to get QL at absolute time 'sec'."""
    t = max(0.0, float(sec))
    tmap = _tempo_map_or_default(tempo_map)
    ql = 0.0
    for i, (t0, bpm) in enumerate(tmap):
        t1 = tmap[i + 1][0] if i + 1 < len(tmap) else t
        if t <= t0:
            break
        span = min(t, t1) - t0 if t1 > t0 else 0.0
        if span > 0.0:
            ql += span * (bpm / 60.0) * 4.0
        if t <= t1:
            break
    return ql


def ql_to_sec(ql: float, tempo_map: List[Tuple[float, float]] | None) -> float:
    """Inverse of sec_to_ql by piecewise integration."""
    target = max(0.0, float(ql))
    tmap = _tempo_map_or_default(tempo_map)
    acc = 0.0  # accumulated QL
    for i, (t0, bpm) in enumerate(tmap):
        rate = (bpm / 60.0) * 4.0  # QL per second in this segment
        if rate <= 0:
            rate = 1e-6
        t1 = tmap[i + 1][0] if i + 1 < len(tmap) else None
        if t1 is None:
            # last segment: finish here
            dt = (target - acc) / rate
            return float(t0 + max(0.0, dt))
        # QL capacity of this segment up to t1
        seg_ql = (t1 - t0) * rate
        if acc + seg_ql >= target:
            dt = (target - acc) / rate
            return float(t0 + max(0.0, dt))
        acc += seg_ql
    # fallback
    last_t, last_bpm = tmap[-1]
    rate = (last_bpm / 60.0) * 4.0
    dt = (target - acc) / (rate if rate > 0 else 1e-6)
    return float(last_t + max(0.0, dt))


# ---------- grid construction ----------


def build_beat_grid(pm) -> Dict[str, Any]:
    """pretty_midi.PrettyMIDI -> beat grid dict.
    Downbeats come from PM; QL is computed from tempo_map.
    """
    try:
        changes, tempi = pm.get_tempo_changes()
        tempo_map = [(float(t), float(b)) for t, b in zip(changes, tempi)]
    except Exception:
        tempo_map = [(0.0, 120.0)]

    try:
        ts = getattr(pm, "time_signature_changes", []) or []
        timesig_map = [(i, f"{s.numerator}/{s.denominator}") for i, s in enumerate(ts)]
        if not timesig_map:
            timesig_map = [(0, "4/4")]
    except Exception:
        timesig_map = [(0, "4/4")]

    try:
        downbeats_sec = [float(x) for x in pm.get_downbeats()]
    except Exception:
        downbeats_sec = [0.0]

    downbeats_ql = [sec_to_ql(t, tempo_map) for t in downbeats_sec]

    return {
        "tempo_map": tempo_map,
        "timesig_map": timesig_map,
        "downbeats_sec": downbeats_sec,
        "downbeats_ql": downbeats_ql,
    }


# ---------- editing utilities ----------


def merge_min_dwell(events: List[Dict[str, Any]], min_ql: float = 2.0) -> List[Dict[str, Any]]:
    """Merge consecutive identical (root+quality) events.
    Assumes events sorted by 'time' in QL.
    Guarantees each segment spans at least min_ql by eliminating redundant splits.
    """
    if not events:
        return []
    merged: List[Dict[str, Any]] = []
    last = None
    for e in sorted(events, key=lambda x: float(x.get("time", 0.0))):
        r = (e.get("root") or "N", e.get("quality") or "")
        if last is None:
            merged.append(e)
            last = r
            continue
        if r == last:
            # skip exact duplicates (same chord continuing)
            continue
        # check min dwell of the previous segment
        if len(merged) >= 1:
            if float(e.get("time", 0.0)) - float(merged[-1].get("time", 0.0)) < float(min_ql):
                # too short: overwrite previous label with the new one (shift boundary)
                merged[-1] = {**e}
            else:
                merged.append(e)
        else:
            merged.append(e)
        last = r
    return merged


def snap_times_to_grid(times_ql: List[float], grid_ql: List[float]) -> List[float]:
    if not times_ql:
        return []
    if not grid_ql:
        return list(times_ql)
    out: List[float] = []
    for x in times_ql:
        # find nearest grid point
        best = min(grid_ql, key=lambda g: abs(g - x))
        out.append(float(best))
    return out
