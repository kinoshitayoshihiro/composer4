#!/usr/bin/env python3
from __future__ import annotations
"""
Audit consistency of external chordmaps (dir A) vs GOLD Stage2 chordmaps (dir GOLD).
Outputs CSV: per-file bar-level chord match rate.
Usage:
  python scripts/audit_vs_gold.py \
    --gold-dir data/GOLD_stage2_json \
    --ext-dir  data/lamda_chordmaps \
    --out-csv  analysis/ext_vs_gold.csv
"""
import os, json, argparse, csv, glob
from typing import Dict, Any, List

def _load_cm_json(path: str) -> Dict[str,Any]:
    try:
        return json.load(open(path, "r", encoding="utf-8"))
    except Exception:
        return {}

def _events_to_bar_labels(cm: Dict[str,Any], bars: int) -> List[str]:
    ev = (cm or {}).get("events") or []
    lbl = [""] * bars
    if not ev: return lbl
    j = 0
    for b in range(bars):
        t_ql = float(b*4.0)
        while j+1 < len(ev) and float(ev[j+1].get("time",0.0)) <= t_ql:
            j += 1
        r = (ev[j].get("root") or "N")
        q = (ev[j].get("quality") or "")
        lbl[b] = r if r=="N" else f"{r}{q}"
    return lbl

def _bars_len(cm: Dict[str,Any]) -> int:
    ev = (cm or {}).get("events") or []
    last = float(max([e.get("time",0.0) for e in ev] + [0.0]))
    return int(last//4.0) + 1

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold-dir", required=True)
    ap.add_argument("--ext-dir", required=True)
    ap.add_argument("--out-csv", required=True)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    ext = {os.path.splitext(os.path.basename(p))[0]: p for p in glob.glob(os.path.join(args.ext_dir, "*.json"))}
    gold = {os.path.splitext(os.path.basename(p))[0].replace(".stage2",""): p for p in glob.glob(os.path.join(args.gold_dir, "*.json"))}

    keys = sorted(set(ext.keys()) & set(gold.keys()))
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["file_id","bars","match_rate","n_diff","diff_bars"])
        for k in keys:
            g = _load_cm_json(gold[k])
            gcm = (g.get("chordmap") or {})
            e = _load_cm_json(ext[k])
            ecm = e if "events" in e else (e.get("chordmap") or {})
            gb = _bars_len(gcm); eb = _bars_len(ecm); bars = max(gb, eb)
            A = _events_to_bar_labels(gcm, bars)
            B = _events_to_bar_labels(ecm, bars)
            same = sum(1 for a,b in zip(A,B) if a==b)
            diffs = [i for i,(a,b) in enumerate(zip(A,B)) if a!=b]
            w.writerow([k, bars, f"{same/float(bars):.4f}", len(diffs), "|".join(map(str, diffs[:20]))])
    print(f"OK: wrote {args.out_csv} (pairs={len(keys)})")

if __name__ == "__main__":
    main()
