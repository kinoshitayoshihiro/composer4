#!/usr/bin/env python3
"""
CREPE系plan JSON → MIDI変換ツール（Piano/Strings/Guitar対応）

Usage:
    python plan_to_midi.py <plan_json> <output_mid> --bpm 89.1 [--program 0]
"""

import argparse
import json
from pathlib import Path
import pretty_midi


def plan_to_midi(plan_path: Path, output_path: Path, bpm: float = 120.0, program: int = 0) -> None:
    """plan JSON → MIDI変換"""

    with open(plan_path) as f:
        plan = json.load(f)

    pm = pretty_midi.PrettyMIDI(initial_tempo=bpm)

    # tracks処理
    for track_data in plan.get("tracks", []):
        track_name = track_data.get("name", "Track")
        instrument = pretty_midi.Instrument(program=program, name=track_name)

        for event in track_data.get("events", []):
            # time/start_beats両対応
            time_ql = event.get("time", event.get("start_beats", 0))
            dur_ql = event.get("duration_ql", event.get("dur", 0.25))
            pitch = int(event.get("pitch_midi", event.get("pitch", 60)))
            vel = int(event.get("velocity", event.get("vel", 80)))

            # ql → seconds
            start_sec = time_ql * 60.0 / bpm
            end_sec = (time_ql + dur_ql) * 60.0 / bpm

            note = pretty_midi.Note(velocity=vel, pitch=pitch, start=start_sec, end=end_sec)
            instrument.notes.append(note)

        pm.instruments.append(instrument)

    pm.write(str(output_path))
    print(
        f"✅ {output_path.name}: {len(pm.instruments)} tracks, "
        f"{sum(len(inst.notes) for inst in pm.instruments)} notes"
    )


def main():
    parser = argparse.ArgumentParser(description="CREPE plan JSON → MIDI")
    parser.add_argument("plan_json", type=Path, help="Input plan JSON")
    parser.add_argument("output_mid", type=Path, help="Output MIDI file")
    parser.add_argument("--bpm", type=float, default=120.0, help="Tempo (default: 120.0)")
    parser.add_argument(
        "--program", type=int, default=0, help="MIDI program number (default: 0=Piano)"
    )

    args = parser.parse_args()

    plan_to_midi(args.plan_json, args.output_mid, args.bpm, args.program)


if __name__ == "__main__":
    main()
