#!/usr/bin/env python3
import argparse, json, yaml, numpy as np, pandas as pd
from pathlib import Path
def loadj(p): return json.loads(Path(p).read_text(encoding="utf-8"))
def default_policy(): return {"enable_sections":["chorus","bridge"],"base_qL":0.5,"min_qL":0.25,"max_qL":1.2,"vocal_proximity":{"stress_window_qL":0.1,"shorten_factor":0.6},"rest_extend":{"voicing_prob_lt":0.4,"extend_factor":1.4},"slope":{"T_st_per_s":1.0,"when_up":{"mult":0.85},"when_down":{"mult":1.15}}}
def parse_sections(sec):
    it=sec.get("sections",[]);out=[]
    if it and "start_bar" in it[0]: 
        for x in it: out.append((int(x["start_bar"]),int(x["end_bar"]),str(x.get("label","")).lower()))
    else:
        it=sorted([(int(x.get("bar",0)),str(x.get("label","")).lower()) for x in it], key=lambda x:x[0])
        for i,(b,l) in enumerate(it): e=it[i+1][0]-1 if i+1<len(it) else b+7; out.append((b,e,l))
    return out
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--guitar-plan",required=True); ap.add_argument("--vocal-f0",required=True)
    ap.add_argument("--vocal-events",default=None); ap.add_argument("--sections",required=True)
    ap.add_argument("--policy",default=None); ap.add_argument("--out",required=True); ap.add_argument("--csv",required=True)
    a=ap.parse_args()
    plan=loadj(a.guitar_plan); qlpb=int(plan.get("meta",{}).get("ql_per_bar",4))
    vdf=pd.read_parquet(a.vocal_f0); 
    if "time_ql" not in vdf.columns: vdf["time_ql"]=vdf["time_s"]*4.0
    evdf=pd.read_parquet(a.vocal_events) if a.vocal_events else None
    sec=loadj(a.sections); ranges=parse_sections(sec); pol=default_policy()
    if a.policy: pol.update(yaml.safe_load(Path(a.policy).read_text(encoding="utf-8")) or {})
    def in_en(b): 
        for s,e,l in ranges:
            if s<=b<=e and any(k in l for k in pol["enable_sections"]): return True
        return False
    rows=[]
    for tr in plan.get("tracks",[]):
        ne=[]
        for ev in tr.get("events",[]):
            b=int(ev.get("time",0)//qlpb)
            dur=float(ev.get("duration_ql",pol["base_qL"]))
            if in_en(b):
                t0=float(ev.get("time",0))
                # stress shorten
                if evdf is not None and "time_ql" in evdf.columns and "classes" in evdf.columns:
                    w=pol["vocal_proximity"]["stress_window_qL"]
                    m=evdf["classes"].apply(lambda x: "stress" in [c.lower() for c in (x if isinstance(x,list) else [])])
                    st=evdf.loc[m,"time_ql"]; 
                    if ((st>=t0-w)&(st<=t0+w)).any(): dur*=pol["vocal_proximity"]["shorten_factor"]
                # rest extend
                m=(vdf["time_ql"]>=t0)&(vdf["time_ql"]<t0+dur)
                vp=float(vdf.loc[m,"voicing_prob"].mean()) if m.any() else 0.0
                if vp<pol["rest_extend"]["voicing_prob_lt"]: dur*=pol["rest_extend"]["extend_factor"]
                # slope tweak
                mb=(vdf["time_ql"]>=b*qlpb)&(vdf["time_ql"]<(b+1)*qlpb)
                sl=float(np.nanmedian(vdf.loc[mb,"f0_slope_hz_per_s"])) if mb.any() else 0.0
                T=pol["slope"]["T_st_per_s"]
                if sl>T: dur*=pol["slope"]["when_up"]["mult"]
                elif sl<-T: dur*=pol["slope"]["when_down"]["mult"]
                dur=float(np.clip(dur, pol["min_qL"], pol["max_qL"]))
                if abs(dur-ev.get("duration_ql",dur))>1e-3:
                    rows.append({"bar":b,"time_ql":t0,"old":ev.get("duration_ql"),"new":dur})
            ev2=dict(ev); ev2["duration_ql"]=dur; ne.append(ev2)
        tr["events"]=ne
    Path(a.out).write_text(json.dumps(plan,ensure_ascii=False,indent=2),encoding="utf-8")
    pd.DataFrame(rows).to_csv(a.csv,index=False)
    print("Wrote:",a.out,"changes:",len(rows))
if __name__=="__main__": main()
