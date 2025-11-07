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
import bisect

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
        timesig_map_time = [(float(s.time), f"{s.numerator}/{s.denominator}") for s in ts] if ts else [(0.0, "4/4")]
        if not timesig_map:
            timesig_map = [(0, "4/4")]
    except Exception:
        timesig_map = [(0, "4/4")]
        timesig_map_time = [(0.0, "4/4")]

    try:
        downbeats_sec = [float(x) for x in pm.get_downbeats()]
    except Exception:
        downbeats_sec = [0.0]

    downbeats_ql = [sec_to_ql(t, tempo_map) for t in downbeats_sec]

    grid = {
        "tempo_map": tempo_map,
        "timesig_map": timesig_map,
        "timesig_map_time": timesig_map_time,
        "downbeats_sec": downbeats_sec,
        "downbeats_ql": downbeats_ql,
    }
    
    # --- NEW: timesig sanitization (fix spurious 1/4 -> 4/4) ---
    try:
        _maybe_fix_one_four(grid)
    except Exception:
        pass
    
    return grid


def _maybe_fix_one_four(grid: dict, tol_ql: float = 0.65, min_bars: int = 16) -> None:
    """
    1/4 が多数出ているが、実際の小節長が ~4.0QL 前後なら 4/4 に補正。
    
    ガード条件:
      - 連続バー数 >= min_bars
      - 平均小節長QL ≈ 4.0 (±tol_ql)
      - 他の拍子が混ざっていない
    
    Parameters
    ----------
    grid : dict
        build_beat_grid()の出力辞書
    tol_ql : float
        許容誤差（デフォルト 0.65 QL）
    min_bars : int
        最小小節数（デフォルト 16）
    """
    ts_time = [sig for _, sig in grid.get("timesig_map_time", [])]
    if not ts_time:
        return
    # 全て1/4かチェック
    if not all(s == "1/4" for s in ts_time):
        return
    
    db = grid.get("downbeats_ql", [])
    if len(db) < (min_bars + 1):
        return
    
    # 小節長の平均を計算
    bar_ql = [db[i+1] - db[i] for i in range(len(db) - 1)]
    avg = sum(bar_ql) / max(1, len(bar_ql))
    
    # 平均が4.0QL付近でなければスキップ
    if abs(avg - 4.0) > tol_ql:
        return
    
    # 補正：timesig を 4/4 に置換
    grid["timesig_map"] = [(b, "4/4") for b, _ in grid.get("timesig_map", [])]
    grid["timesig_map_time"] = [(t, "4/4") for t, _ in grid.get("timesig_map_time", [])]


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
    """Snap times to nearest grid points using O(N log M) bisect search.
    
    Parameters
    ----------
    times_ql : List[float]
        Times to snap (in quarter lengths).
    grid_ql : List[float]
        Grid points to snap to (in quarter lengths).
    
    Returns
    -------
    List[float]
        Snapped times (same length as times_ql).
    
    Notes
    -----
    Improved from O(N*M) to O(N log M) using binary search.
    """
    if not times_ql:
        return []
    if not grid_ql:
        return list(times_ql)
    
    out: List[float] = []
    for x in times_ql:
        # Binary search for insertion point
        i = bisect.bisect_left(grid_ql, x)
        
        if i == 0:
            # x is before first grid point
            out.append(float(grid_ql[0]))
            continue
        
        if i == len(grid_ql):
            # x is after last grid point
            out.append(float(grid_ql[-1]))
            continue
        
        # Compare distances to neighbors
        a, b = grid_ql[i - 1], grid_ql[i]
        best = a if abs(a - x) <= abs(b - x) else b
        out.append(float(best))
    
    return out
