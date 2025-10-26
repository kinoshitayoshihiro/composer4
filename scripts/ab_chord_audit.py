import os, json, csv, glob
from typing import Dict, Any, List, Tuple

def _load_cm(path: str) -> Dict[str,Any]:
    with open(path,"r",encoding="utf-8") as f:
        return json.load(f)

def _events_to_bar_labels(cm: Dict[str,Any], bars: int) -> List[str]:
    """4QL=1bar と仮定し、各bar先頭の (root+quality) を採用"""
    ev = cm.get("events") or []
    lbl = [""] * bars
    if not ev: return lbl
    j = 0
    for b in range(bars):
        t_ql = float(b*4.0)
        while j+1 < len(ev) and float(ev[j+1].get("time",0.0)) <= t_ql:
            j += 1
        r = (ev[j].get("root") or "N")
        q = (ev[j].get("quality") or "")
        lbl[b] = f"{r}{q}" if r!="N" else "N"
    return lbl

def audit_pair(cmA: Dict[str,Any], cmB: Dict[str,Any]) -> Dict[str,Any]:
    def last_bar(cm):
        ev=cm.get("events") or []
        return int(max([e.get("time",0.0) for e in ev]+[0.0])//4)+1
    bars = max(1, max(last_bar(cmA), last_bar(cmB)))
    A = _events_to_bar_labels(cmA, bars)
    B = _events_to_bar_labels(cmB, bars)
    same = sum(1 for a,b in zip(A,B) if a==b)
    diff_idx = [i for i,(a,b) in enumerate(zip(A,B)) if a!=b]
    return {
        "bars": bars,
        "match_rate": same/float(bars),
        "diff_bars": diff_idx[:20],
        "A_first5": A[:5],
        "B_first5": B[:5]
    }

def run_audit(ext_dir: str, int_dir: str, out_csv: str):
    ext_index = {os.path.splitext(os.path.basename(p))[0]: p
                 for p in glob.glob(os.path.join(ext_dir,"*.chordmap.json"))}
    int_index = {os.path.splitext(os.path.basename(p))[0]: p
                 for p in glob.glob(os.path.join(int_dir,"*.chordmap.json"))}
    ids = sorted(set(ext_index.keys()) & set(int_index.keys()))
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    with open(out_csv,"w",newline="",encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["file_id","bars","match_rate","n_diff","diff_bars","A_first5","B_first5"])
        for fid in ids:
            a = _load_cm(ext_index[fid])
            b = _load_cm(int_index[fid])
            res = audit_pair(a,b)
            w.writerow([
                fid, res["bars"], f'{res["match_rate"]:.4f}',
                len(res["diff_bars"]), "|".join(map(str,res["diff_bars"])),
                "|".join(res["A_first5"]), "|".join(res["B_first5"])
            ])
    print(f"OK: {len(ids)} files → {out_csv}")

if __name__=="__main__":
    import argparse
    ap = argparse.ArgumentParser(description="A/B audit for chordmaps (external vs internal)")
    ap.add_argument("--ext-dir", required=True, help="external chordmaps dir (LAMDA)")
    ap.add_argument("--int-dir", required=True, help="internal chordmaps dir (extracted)")
    ap.add_argument("--out-csv", required=True)
    args = ap.parse_args()
    run_audit(args.ext_dir, args.int_dir, args.out_csv)
