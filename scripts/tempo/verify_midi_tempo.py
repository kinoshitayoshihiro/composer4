#!/usr/bin/env python3
import json, statistics
from pathlib import Path
import mido

def mpqn_to_bpm(mpqn:int)->float: return 60_000_000/float(mpqn)

def load_ref(path):
    obj=json.loads(Path(path).read_text(encoding="utf-8"))
    evs=obj.get("events") or obj.get("tempo_points") or obj
    out=[]
    for e in evs:
        t = e.get("time_ql") or e.get("time_qL") or (e.get("bar",0)*4.0 if "bar" in e else None)
        bpm = e.get("bpm") or e.get("tempo") or e.get("bpm_mean")
        if t is not None and bpm is not None: out.append((float(t), float(bpm)))
    out.sort(key=lambda x:x[0])
    if out and out[0][0]>0.0:
        out=[(0.0, out[0][1])] + out
    return out

def main():
    import argparse
    ap=argparse.ArgumentParser()
    ap.add_argument("--midi", required=True)
    ap.add_argument("--tempo-map", required=True)
    a=ap.parse_args()
    mid=mido.MidiFile(a.midi)
    tpq=mid.ticks_per_beat
    cur=0
    midi_t=[]
    for msg in mid.tracks[0]:
        cur+=msg.time
        if msg.type=="set_tempo":
            midi_t.append((cur/float(tpq), mpqn_to_bpm(msg.tempo)))
    ref=load_ref(a.tempo_map)
    med_midi = statistics.median([b for _,b in midi_t]) if midi_t else None
    med_ref  = statistics.median([b for _,b in ref]) if ref else None
    print("Tempo events  MIDI:", len(midi_t), "  Map:", len(ref))
    print("Median BPM     MIDI:", med_midi, "  Map:", med_ref)
    ok = (med_midi is not None and med_ref is not None and abs(med_midi - med_ref) <= 0.5)
    print("Result:", "OK" if ok else "MISMATCH")

if __name__=='__main__': main()
