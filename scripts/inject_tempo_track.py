#!/usr/bin/env python3
"""
Inject a clean tempo track (Track 0) into an existing MIDI and purge stray tempo metas.

Usage:
    # 固定BPM
    python scripts/inject_tempo_track.py \
        --in song.mid --out song_fixed.mid --bpm 74.68 --ts 4/4

    # sections.json使用
    python scripts/inject_tempo_track.py \
        --in song.mid --out song_fixed.mid \
        --tempo-map data/.../sections.json --ts 4/4 --beats-per-bar 4
"""
import argparse
import json
from pathlib import Path
from typing import List, Tuple, Optional

import mido


def parse_time_sig(ts: Optional[str]) -> Optional[Tuple[int, int]]:
    if not ts:
        return None
    if "/" not in ts:
        raise ValueError(f"--ts expects like '4/4', got {ts}")
    n, d = ts.split("/", 1)
    return int(n), int(d)


def bpm_to_uspt(bpm: float) -> int:
    """BPM → microseconds per beat"""
    return int(round(60_000_000 / float(bpm)))


def load_tempo_map(json_path: Path) -> List[Tuple[int, float]]:
    """
    Load tempo map from JSON (sections.json or tempo_map.json)

    Expected format:
        { "tempo_map": [[bar, bpm], ...] }

    Returns:
        List of (bar_index, bpm)
    """
    data = json.loads(json_path.read_text(encoding="utf-8"))

    if isinstance(data, dict) and "tempo_map" in data:
        tm = data["tempo_map"]
    else:
        tm = data

    out = []
    for pair in tm:
        if isinstance(pair, (list, tuple)) and len(pair) >= 2:
            bar, bpm = pair[0], pair[1]
            out.append((int(bar), float(bpm)))

    if not out:
        raise ValueError(f"No tempo_map entries parsed from {json_path}")

    out.sort(key=lambda x: x[0])
    return out


def purge_tempo_metas(track: mido.MidiTrack) -> mido.MidiTrack:
    """Remove all set_tempo and time_signature metas from track"""
    cleaned = mido.MidiTrack()

    for msg in track:
        if msg.is_meta and msg.type in ("set_tempo", "time_signature"):
            continue
        cleaned.append(msg.copy(time=msg.time))

    return cleaned


def make_tempo_track(
    ppq: int,
    constant_bpm: Optional[float],
    tempo_map: Optional[List[Tuple[int, float]]],
    beats_per_bar: int,
    time_sig: Optional[Tuple[int, int]],
) -> mido.MidiTrack:
    """
    Build Track 0 containing:
      - optional time_signature at time=0
      - one or more set_tempo metas (delta-times computed from abs ticks)
    """
    t0 = mido.MidiTrack()
    abs_events: List[Tuple[int, mido.Message]] = []

    # Time signature
    if time_sig:
        n, d = time_sig
        abs_events.append(
            (0, mido.MetaMessage("time_signature", numerator=n, denominator=d, time=0))
        )

    # Tempo events
    if constant_bpm is not None:
        abs_events.append(
            (0, mido.MetaMessage("set_tempo", tempo=bpm_to_uspt(constant_bpm), time=0))
        )
    elif tempo_map:
        for bar_idx, bpm in tempo_map:
            abs_tick = int(bar_idx * beats_per_bar * ppq)
            abs_events.append(
                (abs_tick, mido.MetaMessage("set_tempo", tempo=bpm_to_uspt(bpm), time=0))
            )
    else:
        # Fallback: 120 BPM
        abs_events.append((0, mido.MetaMessage("set_tempo", tempo=bpm_to_uspt(120.0), time=0)))

    # Convert absolute ticks to delta times
    abs_events.sort(key=lambda x: x[0])
    last_tick = 0

    for tick, msg in abs_events:
        delta = max(0, tick - last_tick)
        msg.time = delta
        t0.append(msg)
        last_tick = tick

    # End of track
    t0.append(mido.MetaMessage("end_of_track", time=0))

    return t0


def main():
    parser = argparse.ArgumentParser(
        description="Inject clean tempo track (Track 0) and purge others."
    )
    parser.add_argument("--in", dest="inp", required=True, type=Path, help="Input MIDI")
    parser.add_argument("--out", dest="out", required=True, type=Path, help="Output MIDI")
    parser.add_argument("--bpm", type=float, default=None, help="Constant BPM for whole song")
    parser.add_argument(
        "--tempo-map",
        type=Path,
        default=None,
        help="JSON with tempo_map [[bar,bpm],...] or sections.json",
    )
    parser.add_argument("--beats-per-bar", type=int, default=4, help="Beats per bar (default 4)")
    parser.add_argument("--ts", type=str, default=None, help="Time signature like 4/4")

    args = parser.parse_args()

    if (args.bpm is None) and (args.tempo_map is None):
        parser.error("Specify either --bpm or --tempo-map")

    tempo_map = load_tempo_map(args.tempo_map) if args.tempo_map else None
    ts = parse_time_sig(args.ts)

    mid = mido.MidiFile(args.inp)

    # 1) Purge tempo metas from all existing tracks
    cleaned_tracks = [purge_tempo_metas(t) for t in mid.tracks]

    # 2) Build fresh Track0 meta track
    t0 = make_tempo_track(
        ppq=mid.ticks_per_beat,
        constant_bpm=args.bpm,
        tempo_map=tempo_map,
        beats_per_bar=args.beats_per_bar,
        time_sig=ts,
    )

    # 3) Compose new midi: Track0 (tempo) + all cleaned tracks
    new_mid = mido.MidiFile(ticks_per_beat=mid.ticks_per_beat, type=mid.type)
    new_mid.tracks.append(t0)

    for tr in cleaned_tracks:
        new_mid.tracks.append(tr)

    new_mid.save(args.out)

    # Summary
    print(f"✅ Injected tempo → {args.out}")
    if args.bpm is not None:
        print(f"   BPM = {args.bpm}")
    if tempo_map:
        print(f"   tempo_map entries = {len(tempo_map)} (beats/bar={args.beats_per_bar})")
    print(f"   PPQ = {new_mid.ticks_per_beat}, Tracks = {len(new_mid.tracks)}")


if __name__ == "__main__":
    main()
