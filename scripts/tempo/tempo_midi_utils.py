#!/usr/bin/env python3
import json
from pathlib import Path

def load_tempo_map(tempo_map_path):
    obj = json.loads(Path(tempo_map_path).read_text(encoding="utf-8"))
    events = obj.get("events") or obj.get("tempo_points") or obj
    norm = []
    for e in events:
        t = e.get("time_ql") or e.get("time_qL") or (float(e.get("bar",0))*4.0 if "bar" in e else None)
        bpm = e.get("bpm") or e.get("tempo") or e.get("bpm_mean")
        if t is None or bpm is None: continue
        norm.append({"time_ql": float(t), "bpm": float(bpm)})
    norm.sort(key=lambda x: x["time_ql"])
    if not norm or norm[0]["time_ql"]>0.0:
        norm = [{"time_ql":0.0,"bpm": norm[0]["bpm"] if norm else 90.0}] + norm
    return norm

def bpm_to_mpqn(bpm: float) -> int:
    return int(round(60_000_000 / max(1e-6, bpm)))

def write_midi_with_tempo_map(tracks, tempo_map_path, out_mid, tpq=960):
    import mido
    pts = load_tempo_map(tempo_map_path)
    mid = mido.MidiFile(ticks_per_beat=tpq)
    t0 = mido.MidiTrack(); mid.tracks.append(t0)
    last = 0.0
    for pt in pts:
        delta = int(round((pt["time_ql"] - last) * tpq))
        if delta < 0: delta = 0
        t0.append(mido.MetaMessage("set_tempo", tempo=bpm_to_mpqn(pt["bpm"]), time=delta))
        last = pt["time_ql"]
    for tr in tracks:
        mt = mido.MidiTrack(); mid.tracks.append(mt)
        name = tr.get("name","Track"); mt.append(mido.MetaMessage("track_name",name=name,time=0))
        program = int(tr.get("program",0)); channel=int(tr.get("channel",0))
        mt.append(mido.Message("program_change", program=program, channel=channel, time=0))
        evs = sorted(tr.get("events",[]), key=lambda e: e.get("time_ql", e.get("time",0.0)))
        last_ql = 0.0
        for e in evs:
            tql = float(e.get("time_ql", e.get("time",0.0)))
            dql = float(e.get("dur_ql", e.get("duration_ql",0.25)))
            pitch = int(e.get("pitch", e.get("pitch_midi",60)))
            vel   = int(e.get("vel", e.get("velocity",80)))
            delta = int(round((tql - last_ql)*tpq)); 
            if delta < 0: delta = 0
            mt.append(mido.Message("note_on", note=pitch, velocity=vel, channel=channel, time=delta))
            mt.append(mido.Message("note_off", note=pitch, velocity=0, channel=channel, time=int(round(dql*tpq))))
            last_ql = tql + dql
    mid.save(out_mid)
    return out_mid
