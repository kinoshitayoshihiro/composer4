#!/usr/bin/env python3
import argparse, json, numpy as np, pandas as pd, yaml
from pathlib import Path
def J(p): return json.loads(Path(p).read_text(encoding="utf-8"))
def chord_third(root_midi, quality="maj"):
    return root_midi + (4 if "m" not in quality else 3)
def nearest_third_or_root(pitch, root, quality):
    third = chord_third(root, quality); import numpy as np
    cand = np.array([root, third])
    return int(cand[np.argmin(np.abs(cand - pitch))])
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--strings-plan",required=True)
    ap.add_argument("--chordmap",required=True)
    ap.add_argument("--out",required=True)
    ap.add_argument("--kpi-csv",required=True)
    a=ap.parse_args()
    plan=J(a.strings_plan); cm=J(a.chordmap); qlpb=int(plan.get("meta",{}).get("ql_per_bar",4))
    evs_cm = cm.get("events", [])
    bybar={}
    for ev in evs_cm:
        b=int(ev.get("time",0)//qlpb); 
        bybar.setdefault(b, ev)
    changes=0; total=0
    for tr in plan.get("tracks",[]):
        bars = {}
        for ev in tr["events"]:
            bars.setdefault(int(ev["bar"]), []).append(ev)
        for b, lst in bars.items():
            lst.sort(key=lambda x:x["time"])
            last = lst[-1]
            cm_ev = bybar.get(b)
            if not cm_ev: continue
            name2 = {"C":60,"C#":61,"DB":61,"D":62,"D#":63,"EB":63,"E":64,"F":65,"F#":66,"GB":66,"G":67,"G#":68,"AB":68,"A":69,"A#":70,"BB":70,"B":71}
            root = name2.get((cm_ev.get("root") or "C").upper(),60)
            qual = cm_ev.get("quality","maj")
            target = nearest_third_or_root(int(round(last["pitch_midi"])), root, qual)
            total+=1
            if abs(target - last["pitch_midi"])>=1:
                last["pitch_midi"]=int(target); changes+=1
    import pandas as pd
    pd.DataFrame([{"bars_considered": total, "resolved_changes": changes, "resolution_rate": (changes/total if total else 0.0)}]).to_csv(a.kpi_csv, index=False)
    Path(a.out).write_text(json.dumps(plan,ensure_ascii=False,indent=2),encoding="utf-8")
    print("Wrote:",a.out,"changes:",changes,"bars:",total)
if __name__=="__main__": main()
