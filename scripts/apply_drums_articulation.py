#!/usr/bin/env python3
"""
apply_drums_articulation.py (V5.1: Hop-Aware & CC Fixed)

Changes:
  - Dynamic Hop: Reads 'analysis_hop_sec' or calculates from curve length.
  - Bar Normalization: Syncs 'bar' and 'bar_idx'.
"""

import argparse
import json
import random
from pathlib import Path
from typing import List, Dict, Any


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
    """Dynamically calculate hop_sec from vocal_guide or curve length."""
    hop = vocal_guide.get("metadata", {}).get("analysis_hop_sec")
    if hop:
        return float(hop)
    curve = vocal_guide.get("energy_curve", [])
    if song_duration and len(curve) > 0:
        return song_duration / len(curve)
    return 0.05


def get_vocal_state(time_sec: float, vocal_guide: Dict, hop_sec: float) -> Dict[str, Any]:
    """Access vocal_guide using correct hop interval."""
    state = {"energy": "VOL_MEDIUM", "val_energy": 0.5}
    if not vocal_guide:
        return state
    curve = vocal_guide.get("energy_curve", [])
    # ★修正: 動的hop_sec使用
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

    # ★重要: Hop Size の取得
    song_duration = None
    if events:
        max_time_ql = max(ev.get("time_ql", 0) for ev in events)
        song_duration = tempo_map.time_at_beat(max_time_ql + 4.0)
    hop_sec = calculate_hop_sec(vocal_guide, song_duration)
    print(f"🎯 Using Analysis Hop: {hop_sec:.4f}s")

    events.sort(key=lambda x: x.get("time_ql", 0.0))
    print(f"Processing {len(events)} events for Drums Articulation (V5.1 Hop-Aware)...")

    for ev in events:
        # --- Bar Index Normalization ---
        if "bar_idx" in ev:
            ev["bar"] = ev["bar_idx"]
        elif "bar" in ev:
            ev["bar_idx"] = ev["bar"]

        time_ql = ev.get("time_ql", 0.0)
        time_sec = tempo_map.time_at_beat(time_ql)
        ev["time_sec"] = time_sec

        # ★ hop_sec 使用
        v_state = get_vocal_state(time_sec, vocal_guide, hop_sec)
        val_energy = v_state["val_energy"]

        note = int(ev.get("note", 0))
        is_snare = (
            note in [38, 40] or ev.get("instrument") == "snare" or ev.get("drum_type") == "SNARE"
        )
        is_ghost = ev.get("type") == "ghost" or ev.get("velocity", 0) < 45
        is_hihat = note in [42, 44, 46]

        # --- 1. Snare Morphing ---
        if is_snare and not is_ghost:
            if val_energy < 0.3:
                ev["note"] = 37  # Side Stick
                ev["velocity"] = min(ev.get("velocity", 80), 80)
            elif val_energy > 0.7:
                ev["note"] = 40  # Rimshot
                clap = ev.copy()
                clap["note"] = 39
                clap["velocity"] = int(ev.get("velocity", 100) * 0.9)
                new_events.append(clap)
            else:
                ev["note"] = 38  # Normal Snare

        # --- 2. Ghost Texture ---
        if is_ghost and val_energy < 0.4:
            if random.random() < 0.5:
                ev["note"] = 54  # Tambourine

        # --- 3. Hi-Hat Openness (CC#4) ---
        if is_hihat:
            cc_val = int(val_energy * 100)
            cc_events.append(
                {"time_ql": time_ql, "type": "cc", "cc": 4, "value": cc_val, "channel": 9}
            )
            ev["velocity"] = int(ev.get("velocity", 90) * (0.7 + val_energy * 0.4))

        # --- 4. Generate CC curve for ALL events (energy expression) ---
        # ハイハットがなくても全イベントでEnergy CCを生成
        cc_events.append(
            {
                "time_ql": time_ql,
                "type": "cc",
                "cc": 11,  # Expression
                "value": int(50 + val_energy * 70),
                "channel": 9,
            }
        )

        new_events.append(ev)

    plan["events"] = new_events
    plan["cc_events"] = cc_events
    plan["format_version"] = "V5_1_HopAware"
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
    print(f"🥁 Drums V5 Fixed: {args.out}")


if __name__ == "__main__":
    main()
