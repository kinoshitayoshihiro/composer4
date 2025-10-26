#!/usr/bin/env python3
from __future__ import annotations
"""
Audit consistency between KILO_CHORDS_DATA (one pickle) and CHORDS_DATA (162 pickles).
Outputs a per-file CSV with bar-level chord match rate and diff bars.
Usage:
  python scripts/audit_kilo_vs_chords.py \
    --kilo data/KILO_CHORDS_DATA.pickle \
    --chords-dir data/CHORDS_DATA \
    --out-csv analysis/kilo_vs_chords_audit.csv \
    --tpq 480 \
    --max-files 0
"""
import os, glob, csv, pickle, argparse
from typing import Dict, Any, List
from adapters.lamda_kilo_reader import iter_kilo_pickle
from adapters.lamda_chords_decoder import decode_chord_seq_to_events

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

def _load_chords_dir(chords_dir: str):
    for pkl in glob.glob(os.path.join(chords_dir, "LAMDa_CHORDS_DATA_*.pickle")):
        data = pickle.load(open(pkl, "rb"))
        for item in data:
            if isinstance(item, (list,tuple)) and len(item)>=2:
                yield str(item[0]), item[1]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kilo", required=True)
    ap.add_argument("--chords-dir", required=True)
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--tpq", type=int, default=480)
    ap.add_argument("--max-files", type=int, default=0, help="0=all")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    seen = 0
    chords_map = {}
    for fid, seq in _load_chords_dir(args.chords_dir):
        chords_map[fid] = seq
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["file_id","bars","match_rate","n_diff","diff_bars","kilo_first5","chords_first5"])
        for fid, kseq in iter_kilo_pickle(args.kilo):
            if args.max_files and seen >= args.max_files:
                break
            if fid not in chords_map:
                continue
            cseq = chords_map[fid]
            km = decode_chord_seq_to_events(kseq, tpq=args.tpq, token_map_yaml="adapters/lamda_chords_token_map.yaml")
            cm = decode_chord_seq_to_events(cseq, tpq=args.tpq, token_map_yaml="adapters/lamda_chords_token_map.yaml")
            kb = _bars_len(km); cb = _bars_len(cm); bars = max(kb, cb)
            A = _events_to_bar_labels(km, bars)
            B = _events_to_bar_labels(cm, bars)
            same = sum(1 for a,b in zip(A,B) if a==b)
            diffs = [i for i,(a,b) in enumerate(zip(A,B)) if a!=b]
            w.writerow([fid, bars, f"{same/float(bars):.4f}", len(diffs), "|".join(map(str,diffs[:20])), "|".join(A[:5]), "|".join(B[:5])])
            seen += 1
    print(f"OK: wrote {args.out_csv} (files={seen})")

if __name__ == "__main__":
    main()
