#!/usr/bin/env python3
import argparse, json, pandas as pd
from pathlib import Path
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--pitch-curve-json", required=True)
    ap.add_argument("--out-csv", required=True)
    a=ap.parse_args()
    data = json.loads(Path(a.pitch_curve_json).read_text(encoding="utf-8"))
    rows=[]
    for i,n in enumerate(data.get("notes",[])):
        for t,c in n.get("curve",[]):
            rows.append({"note_index":i,"start_ql":n["start_ql"],"dur_ql":n["dur_ql"],"base_pitch":n["pitch_midi"],"rel_ql":t,"cents":c})
    pd.DataFrame(rows).to_csv(a.out_csv, index=False)
    print("Wrote:", a.out_csv, "rows:", len(rows))
if __name__=="__main__": main()
