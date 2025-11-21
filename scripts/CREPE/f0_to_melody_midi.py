#!/usr/bin/env python3
import argparse, numpy as np, pandas as pd
from pathlib import Path
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--vocal-f0",required=True); ap.add_argument("--out-mid",required=True)
    ap.add_argument("--pb-range",type=float,default=2.0); ap.add_argument("--min-dur-ql",type=float,default=0.25)
    a=ap.parse_args()
    try: import mido
    except Exception as e: raise SystemExit("pip install mido") from e
    vdf=pd.read_parquet(a.vocal_f0); t=vdf["time_ql"] if "time_ql" in vdf.columns else (vdf["time_s"]*4.0)
    f0m=vdf["f0_midi"].values; vp=vdf["voicing_prob"].values; voiced=vp>0.6
    notes=[]; 
    if len(t):
        st=None
        for i in range(len(t)):
            if voiced[i] and st is None: st=i
            if ((not voiced[i]) or i==len(t)-1) and st is not None:
                ed = i if not voiced[i] else i+1; dur = float(t.iloc[ed-1] - t.iloc[st])
                if dur>=a.min_dur_ql:
                    pitch=int(round(np.nanmedian(f0m[st:ed]))); notes.append((float(t.iloc[st]),dur,pitch))
                st=None
    tpq=480; mid=mido.MidiFile(ticks_per_beat=tpq); tr=mido.MidiTrack(); mid.tracks.append(tr)
    def ql2ticks(x): return int(x*tpq)
    last=0.0
    for t0,dur,p in notes:
        tr.append(mido.Message("note_on", note=p, velocity=96, time=max(0, ql2ticks(t0-last))))
        tr.append(mido.Message("note_off",note=p, velocity=0,  time=ql2ticks(dur)))
        last=t0+dur
    mid.save(a.out_mid); print("Wrote:",a.out_mid,"notes:",len(notes))
if __name__=="__main__": main()
