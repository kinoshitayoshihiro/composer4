#!/usr/bin/env python3
from __future__ import annotations
"""
Report the ring buffer as CSV.
Usage:
  python scripts/ringbuffer_report.py \
    --ring analysis/consistency_ring.jsonl \
    --out-csv analysis/consistency_ring.csv
"""
import os, json, argparse, csv

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ring", required=True)
    ap.add_argument("--out-csv", required=True)
    args = ap.parse_args()

    rows = []
    if os.path.exists(args.ring):
        with open(args.ring, "r", encoding="utf-8") as f:
            for line in f:
                try: rows.append(json.loads(line))
                except Exception: pass
    rows.sort(key=lambda r: r.get("ts",""))
    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["ts","tag","mean_match","n","source"])
        for r in rows:
            w.writerow([r.get("ts",""), r.get("tag",""), f'{float(r.get("mean_match",0.0)):.4f}', int(r.get("n",0)), r.get("source","")])
    print(f"OK: wrote {args.out_csv} (rows={len(rows)})")

if __name__ == "__main__":
    main()
