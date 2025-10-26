#!/usr/bin/env python3
from __future__ import annotations
"""
Report the ring buffer as CSV and optional PNG time series.

Usage:
  python scripts/ringbuffer_report_png.py \
    --ring   analysis/consistency_ring.jsonl \
    --out-csv analysis/consistency_ring.csv \
    --out-png analysis/consistency_ring.png \
    --tag    ext_vs_gold   # optional filter
"""
import os, json, argparse, csv

def _load_ring(path):
    rows = []
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                try: rows.append(json.loads(line))
                except Exception: pass
    rows.sort(key=lambda r: r.get("ts",""))
    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ring", required=True)
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--out-png", default=None)
    ap.add_argument("--tag", default=None, help="filter by tag; default=all")
    args = ap.parse_args()

    rows = _load_ring(args.ring)
    if args.tag:
        rows = [r for r in rows if r.get("tag")==args.tag]

    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["ts","tag","mean_match","n","source"])
        for r in rows:
            w.writerow([
                r.get("ts",""),
                r.get("tag",""),
                f'{float(r.get("mean_match",0.0)):.4f}',
                int(r.get("n",0)),
                r.get("source","")
            ])
    print(f"OK: wrote {args.out_csv} (rows={len(rows)})")

    if args.out_png:
        # matplotlib single-plot without specifying colors
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        xs = list(range(len(rows)))
        ys = [float(r.get("mean_match",0.0)) for r in rows]
        plt.figure()
        plt.plot(xs, ys, marker="o")
        plt.title(args.tag or "consistency")
        plt.xlabel("snapshot")
        plt.ylabel("mean_match")
        plt.grid(True)
        os.makedirs(os.path.dirname(args.out_png) or ".", exist_ok=True)
        plt.savefig(args.out_png, bbox_inches="tight")
        plt.close()
        print(f"OK: wrote {args.out_png}")

if __name__ == "__main__":
    main()
