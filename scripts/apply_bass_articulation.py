#!/usr/bin/env python3
"""
apply_bass_articulation.py (V5.1: Hop-Aware & Bar Normalized)

Changes:
  - Dynamic Hop: Reads 'analysis_hop_sec' from vocal_guide metadata, or calculates from curve length.
  - Bar Normalization: Syncs 'bar' and 'bar_idx' to prevent logic errors.
"""

import argparse
import json
import random
from pathlib import Path
from typing import List, Dict, Any


# --- Tempo Map Logic ---
class TempoMap:
    def __init__(self, tempo_data: Dict):
        self.points = sorted(tempo_data.get("tempo_points", []), key=lambda x: x[0])
        self.beat_to_time = self._build_beat_to_time()

    def _build_beat_to_time(self):
        mapping = []
        current_time = 0.0
        if not self.points:
            return [(0.0, 0.0)]
        prev_beat, prev_bpm = self.points[0]
        if prev_beat > 0:
            mapping.append((0.0, 0.0))
            current_time += (prev_beat - 0.0) * (60.0 / prev_bpm)
        mapping.append((prev_beat, current_time))
        for i in range(1, len(self.points)):
            beat, bpm = self.points[i]
            dt = (beat - prev_beat) * (60.0 / prev_bpm)
            current_time += dt
            mapping.append((beat, current_time))
            prev_beat = beat
            prev_bpm = bpm
        return mapping

    def time_at_beat(self, target_beat: float) -> float:
        for i in range(len(self.beat_to_time) - 1):
            b1, t1 = self.beat_to_time[i]
            b2, t2 = self.beat_to_time[i + 1]
            if b1 <= target_beat <= b2:
                progress = (target_beat - b1) / (b2 - b1) if (b2 - b1) > 0 else 0
                return t1 + progress * (t2 - t1)
        last_b, last_t = self.beat_to_time[-1]
        last_bpm = self.points[-1][1]
        return last_t + (target_beat - last_b) * (60.0 / last_bpm)


def calculate_hop_sec(vocal_guide: Dict, song_duration: float = None) -> float:
    """
    Dynamically calculate hop_sec from vocal_guide metadata or curve length.
    Priority: metadata.analysis_hop_sec > calculated from curve length > fallback 0.05
    """
    # 1. Try metadata
    hop = vocal_guide.get("metadata", {}).get("analysis_hop_sec")
    if hop:
        return float(hop)
    
    # 2. Calculate from curve length if song_duration provided
    curve = vocal_guide.get("energy_curve", [])
    if song_duration and len(curve) > 0:
        calculated = song_duration / len(curve)
        return calculated
    
    # 3. Fallback to safe default (0.05s is common for librosa)
    return 0.05


# --- Vocal Guide Accessor (Hop-Aware) ---
def get_vocal_state(time_sec: float, vocal_guide: Dict, hop_sec: float) -> Dict[str, Any]:
    """Access vocal_guide using correct hop interval."""
    state = {"energy": "VOL_MEDIUM", "val_energy": 0.5}
    if not vocal_guide:
        return state
    curve = vocal_guide.get("energy_curve", [])
    # ★修正: 固定0.1ではなく、動的hop_secを使用
    idx = int(time_sec / hop_sec)
    val = 0.0
    if 0 <= idx < len(curve):
        val = curve[idx]
    state["val_energy"] = val
    if val > 0.6:
        state["energy"] = "VOL_LOUD"
    elif val < 0.1:
        state["energy"] = "VOL_SILENT"
    elif val < 0.3:
        state["energy"] = "VOL_SOFT"
    else:
        state["energy"] = "VOL_MEDIUM"
    return state


def load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Any, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def apply_articulation(
    plan: Dict[str, Any], vocal_guide: Dict, tempo_map: TempoMap
) -> Dict[str, Any]:
    events = plan.get("events", [])
    new_events = []
    cc_events = plan.get("cc_events", [])
    pitch_bend_events = plan.get("pitch_bend_events", [])

    # ★重要: Hop Size の取得（曲の長さから計算）
    song_duration = None
    if events:
        max_time_ql = max(ev.get("time_ql", 0) for ev in events)
        song_duration = tempo_map.time_at_beat(max_time_ql + 4.0)
    hop_sec = calculate_hop_sec(vocal_guide, song_duration)
    print(f"🎯 Using Analysis Hop: {hop_sec:.4f}s")

    events.sort(key=lambda x: x.get("time_ql", 0.0))
    print(f"Processing {len(events)} events for Bass Articulation (V5.1 Hop-Aware)...")

    prev_note = None
    prev_time = 0.0

    for i, ev in enumerate(events):
        # --- Bar Index Normalization (A案: bar_idx優先) ---
        if "bar_idx" in ev:
            ev["bar"] = ev["bar_idx"]  # Force sync
        elif "bar" in ev:
            ev["bar_idx"] = ev["bar"]  # Backfill

        time_ql = ev.get("time_ql", 0.0)
        time_sec = tempo_map.time_at_beat(time_ql)
        ev["time_sec"] = time_sec

        # Macro Context from Legacy Plan
        macro_density = ev.get("vocal_density", "medium")
        macro_energy = ev.get("energy", 0.5)

        # Micro Context from Vocal Guide (★ hop_sec 使用)
        v_state = get_vocal_state(time_sec, vocal_guide, hop_sec)
        val_energy = v_state["val_energy"]
        energy_token = v_state["energy"]

        # --- A. Slap & Dynamics ---
        orig_vel = ev.get("velocity", 100)

        if energy_token == "VOL_LOUD" or val_energy > 0.7:
            ev["velocity"] = min(127, int(orig_vel * 1.2))
            if ev.get("duration_ql", 1.0) > 0.5:
                ev["duration_ql"] = ev.get("duration_ql") * 0.75
                ev["duration_scale"] = 0.75
        elif energy_token == "VOL_SILENT":
            ev["velocity"] = int(orig_vel * 0.6)

        # --- B. Dynamic Slide ---
        current_note = int(ev.get("note", 36))
        if prev_note is not None:
            interval = current_note - prev_note
            time_diff = time_sec - prev_time
            if abs(interval) >= 3 and val_energy > 0.4 and time_diff < 0.6:
                bend_start = time_sec - 0.1
                bend_val = -4000 if interval > 0 else 4000
                pitch_bend_events.append({"time_sec": bend_start, "value": bend_val, "channel": 0})
                pitch_bend_events.append({"time_sec": time_sec, "value": 0, "channel": 0})

        new_events.append(ev)

        # --- C. Ghost Note Injection (緩和条件) ---
        # 条件: val_energy > 0.5, duration >= 0.5
        if val_energy > 0.5 and ev.get("duration_ql", 0) >= 0.5:
            next_time_ql = (
                events[i + 1].get("time_ql", time_ql + 4.0)
                if i + 1 < len(events)
                else time_ql + 4.0
            )

            ghost_time_ql = time_ql + 0.5

            if ghost_time_ql < next_time_ql - 0.1:
                ghost_note = ev.copy()
                ghost_note["time_ql"] = ghost_time_ql
                ghost_note["time_sec"] = tempo_map.time_at_beat(ghost_time_ql)
                ghost_note["velocity"] = 30
                ghost_note["duration_ql"] = 0.1
                ghost_note["type"] = "ghost"
                ghost_note["note"] = current_note
                new_events.append(ghost_note)

        prev_note = current_note
        prev_time = time_sec

    # CC Automation (★ hop_sec 基準で生成)
    guide_curve = vocal_guide.get("energy_curve", [])
    for idx, val in enumerate(guide_curve):
        t = idx * hop_sec  # ★修正: 正しい時間軸
        cc_events.append({"time_sec": t, "cc": 1, "value": int(val * 90)})
        cc_events.append({"time_sec": t, "cc": 11, "value": int(50 + val * 77)})

    plan["events"] = new_events
    plan["cc_events"] = cc_events
    plan["pitch_bend_events"] = pitch_bend_events
    plan["format_version"] = "V5_Hybrid_Fixed"
    return plan


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True)
    parser.add_argument("--vocal-guide", required=True)
    parser.add_argument("--tempo", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    plan = load_json(Path(args.plan))
    vocal_guide = load_json(Path(args.vocal_guide))
    tempo_map = TempoMap(load_json(Path(args.tempo)))

    new_plan = apply_articulation(plan, vocal_guide, tempo_map)
    save_json(new_plan, Path(args.out))
    print(f"🎸 Bass V5 (Hybrid Fixed) Saved: {args.out}")


if __name__ == "__main__":
    main()
