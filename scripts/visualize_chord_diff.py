#!/usr/bin/env python3
from __future__ import annotations
"""
Visualize bar-level chord differences between GOLD and EXT chordmaps.
- Produces a 1xN heatmap (0=match, 1=diff) and saves PNG.

Usage:
  python scripts/visualize_chord_diff.py \
    --gold-json data/GOLD_stage2_json/xxx.stage2.json \
    --ext-json  data/lamda_chordmaps/xxx.json \
    --out-png   analysis/xxx_diff.png
"""
import os, json, argparse
import numpy as np

def _load_cm_json(path):
    try:
        j = json.load(open(path, "r", encoding="utf-8"))
        return j if "events" in j else (j.get("chordmap") or {})
    except Exception:
        return {}

def _events_to_bar_labels(cm, bars):
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

def _bars_len(cm):
    ev = (cm or {}).get("events") or []
    last = float(max([e.get("time",0.0) for e in ev] + [0.0]))
    return int(last//4.0) + 1

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold-json", required=True)
    ap.add_argument("--ext-json", required=True)
    ap.add_argument("--out-png", required=True)
    args = ap.parse_args()

    gcm = _load_cm_json(args.gold_json)
    ecm = _load_cm_json(args.ext_json)
    bars = max(_bars_len(gcm), _bars_len(ecm))
    A = _events_to_bar_labels(gcm, bars)
    B = _events_to_bar_labels(ecm, bars)
    diff = np.array([[0.0 if a==b else 1.0 for a,b in zip(A,B)]], dtype=float)

    # single-plot heatmap (no color specified)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.figure()
    plt.imshow(diff, aspect="auto")
    plt.yticks([]); plt.xlabel("bar index")
    plt.title("Chord diff (0=match, 1=diff)")
    os.makedirs(os.path.dirname(args.out_png) or ".", exist_ok=True)
    plt.savefig(args.out_png, bbox_inches="tight")
    plt.close()
    print(f"OK: wrote {args.out_png}  |  bars={bars}  diffs={int(diff.sum())}")

if __name__ == "__main__":
    main()
