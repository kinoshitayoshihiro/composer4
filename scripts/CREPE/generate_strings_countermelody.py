#!/usr/bin/env python3
import argparse, json, math, yaml, numpy as np, pandas as pd
from pathlib import Path
def loadj(p): return json.loads(Path(p).read_text(encoding="utf-8"))
def default_policy(): return {"enable_sections":["chorus","bridge"],"lane":{"interval_from_vocal_semitones":[5,12],"avoid_unison":True,"avoid_minor2":True},"motion":{"slope_thr_st_per_s":1.0,"prefer_contrary_ratio":0.7},"rhythm":{"notes_per_bar":[2,4],"prefer_offbeat":True}}
def parse_sections(sec):
    it=sec.get("sections",[]);out=[]
    if it and "start_bar" in it[0]: 
        for x in it: out.append((int(x["start_bar"]),int(x["end_bar"]),str(x.get("label","")).lower()))
    else:
        it=sorted([(int(x.get("bar",0)),str(x.get("label","")).lower()) for x in it], key=lambda x:x[0])
        for i,(b,l) in enumerate(it): e=it[i+1][0]-1 if i+1<len(it) else b+7; out.append((b,e,l))
    return out
def select_bars(ranges,en): 
    s=set()
    for st,ed,l in ranges:
        if any(k in l for k in en):
            for b in range(st,ed+1): s.add(b)
    return sorted(s)
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--vocal-f0",required=True); ap.add_argument("--sections",required=True)
    ap.add_argument("--chordmap",required=True); ap.add_argument("--policy",default=None)
    ap.add_argument("--out",required=True)
    a=ap.parse_args()
    vdf=pd.read_parquet(a.vocal_f0); sec=loadj(a.sections); pol=default_policy()
    if a.policy: pol.update(yaml.safe_load(Path(a.policy).read_text(encoding="utf-8")) or {})
    ranges=parse_sections(sec); en=pol["enable_sections"]
    qlpb=4; 
    if "time_ql" not in vdf.columns: vdf["time_ql"]=vdf["time_s"]*4.0
    bars=select_bars(ranges,en)
    offs=[0.75,1.5,2.25,3.0]
    evs=[]
    for b in bars:
        m=(vdf["time_ql"]>=b*qlpb)&(vdf["time_ql"]<(b+1)*qlpb)
        vb=vdf.loc[m]
        if vb.empty: continue
        base=float(np.nanmedian(vb["f0_midi"]))
        if np.isnan(base): continue
        lo,hi=pol["lane"]["interval_from_vocal_semitones"]
        slope=float(np.nanmedian(np.gradient(vb["f0_smooth_hz"].fillna(0.0))))
        contrary=abs(slope)>=pol["motion"]["slope_thr_st_per_s"]
        num=int(np.clip(round(np.interp(abs(slope),[0,3],[2,4])),2,4))
        for off in offs[:num]:
            pitch=np.clip(base+(-2.0 if slope>0 else 2.0) if contrary else base, base+lo, base+hi)
            dur=0.4 if contrary else 0.6
            evs.append({"bar":int(b),"time":float(b*qlpb+off),"pitch_midi":float(round(pitch)),"duration_ql":float(dur),"velocity":80})
    plan={"meta":{"role":"strings","generator":"strings_countermelody","ql_per_bar":qlpb,"enable_sections":en},"tracks":[{"name":"Strings","role":"strings","events":evs}]}
    Path(a.out).write_text(json.dumps(plan,ensure_ascii=False,indent=2),encoding="utf-8")
    print("Wrote:",a.out,"events:",len(evs))
if __name__=="__main__": main()
