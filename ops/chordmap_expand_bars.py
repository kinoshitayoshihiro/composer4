#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ops/chordmap_expand_bars.py - Expand chordmap to bar-level granularity

既存の粗いchordmap（7イベント）を小節ごとに展開（137イベント）
"""
import argparse, json
from pathlib import Path


def expand_chordmap_to_bars(
    events: list,
    sections: dict,
    ql_per_bar: float = 4.0,
    min_dwell_bars: int = 1
) -> list:
    """Expand coarse chordmap to bar-level events
    
    Args:
        events: Original events [{time, root, quality, confidence?}, ...]
        sections: sections.json content
        ql_per_bar: QL per bar
        min_dwell_bars: Min chord duration in bars (for filtering)
    
    Returns:
        Expanded events (1 bar = 1 chord)
    """
    if not events:
        return []
    
    # Find total duration from sections
    max_bar = 0
    if "events" in sections:
        for sec in sections["events"]:
            end_ql = sec.get("end_ql", sec.get("time", 0))
            end_bar = int(end_ql / ql_per_bar)
            max_bar = max(max_bar, end_bar)
    
    if max_bar == 0:
        # Fallback: use last event time
        max_bar = int(events[-1]["time"] / ql_per_bar) + 10
    
    # Build bar-level grid
    expanded = []
    current_chord = {"root": events[0]["root"], "quality": events[0]["quality"]}
    event_idx = 0
    
    for bar in range(max_bar + 1):
        bar_time = bar * ql_per_bar
        
        # Update current chord if next event is reached
        while event_idx < len(events) - 1 and events[event_idx + 1]["time"] <= bar_time:
            event_idx += 1
            current_chord = {
                "root": events[event_idx]["root"],
                "quality": events[event_idx]["quality"]
            }
        
        # Emit event for this bar
        ev = {
            "time": bar_time,
            "root": current_chord["root"],
            "quality": current_chord["quality"]
        }
        
        # Carry confidence if available
        if "confidence" in events[event_idx]:
            ev["confidence"] = events[event_idx]["confidence"]
        
        expanded.append(ev)
    
    # Min dwell filter (merge consecutive same chords)
    if min_dwell_bars > 1:
        filtered = [expanded[0]]
        for ev in expanded[1:]:
            if ev["root"] == filtered[-1]["root"] and ev["quality"] == filtered[-1]["quality"]:
                # Same chord, skip (will extend previous)
                continue
            
            # Check duration of previous chord
            prev_bars = (ev["time"] - filtered[-1]["time"]) / ql_per_bar
            if prev_bars < min_dwell_bars:
                # Too short, extend to previous-previous chord
                if len(filtered) > 1:
                    filtered[-1] = filtered[-2].copy()
                    filtered[-1]["time"] = ev["time"] - ql_per_bar
            
            filtered.append(ev)
        expanded = filtered
    
    return expanded


def main():
    ap = argparse.ArgumentParser(description="Expand chordmap to bar-level (1 bar = 1 chord)")
    ap.add_argument("--chordmap", required=True, help="Input chordmap.json (coarse)")
    ap.add_argument("--sections", required=True, help="sections.json")
    ap.add_argument("--out", required=True, help="Output chordmap.json (bar-level)")
    ap.add_argument("--ql-per-bar", type=float, default=4.0, help="QL per bar")
    ap.add_argument("--min-dwell-bars", type=int, default=1, help="Min chord duration (bars)")
    
    args = ap.parse_args()
    
    chordmap_path = Path(args.chordmap)
    sections_path = Path(args.sections)
    out_path = Path(args.out)
    
    # Load inputs
    with chordmap_path.open("r", encoding="utf-8") as f:
        chordmap = json.load(f)
    
    with sections_path.open("r", encoding="utf-8") as f:
        sections = json.load(f)
    
    events = chordmap.get("events", [])
    if not events:
        print(f"[ERROR] No events in {chordmap_path}")
        return 1
    
    print(f"[INFO] Original: {len(events)} events")
    
    # Expand to bars
    expanded = expand_chordmap_to_bars(
        events,
        sections,
        ql_per_bar=args.ql_per_bar,
        min_dwell_bars=args.min_dwell_bars
    )
    
    print(f"[INFO] Expanded: {len(expanded)} events")
    
    # Output
    output = {"unit": "ql", "events": expanded}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print(f"[OK] Bar-level chordmap: {len(expanded)} events -> {out_path}")
    
    # Stats
    total_bars = int(events[-1]["time"] / args.ql_per_bar)
    density = len(expanded) / max(total_bars, 1)
    print(f"[INFO] Density: {len(expanded)} events / {total_bars} bars = {density:.2f} chords/bar")
    
    return 0


if __name__ == "__main__":
    exit(main())
