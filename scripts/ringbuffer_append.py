#!/usr/bin/env python3
from __future__ import annotations
"""
Append a summary snapshot from an audit CSV into a ring buffer (JSONL).
Usage:
  python scripts/ringbuffer_append.py \
    --csv analysis/ext_vs_gold.csv \
    --ring analysis/consistency_ring.jsonl \
    --tag ext_vs_gold \
    --max-entries 200
"""
import os, csv, json, argparse, datetime

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--ring", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--max-entries", type=int, default=200)
    args = ap.parse_args()

    n = 0; s = 0.0
    with open(args.csv, "r", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        for row in rd:
            try:
                s += float(row["match_rate"]); n += 1
            except Exception:
                continue
    entry = {
        "ts": datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "tag": args.tag,
        "mean_match": (s/n if n else 0.0),
        "n": n,
        "source": os.path.abspath(args.csv)
    }
    items = []
    if os.path.exists(args.ring):
        with open(args.ring, "r", encoding="utf-8") as f:
            for line in f:
                try: items.append(json.loads(line))
                except Exception: pass
    items.append(entry)
    if args.max_entries and len(items) > args.max_entries:
        items = items[-args.max_entries:]
    os.makedirs(os.path.dirname(args.ring) or ".", exist_ok=True)
    with open(args.ring, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    print(f"OK: appended -> {args.ring} (entries={len(items)})")

if __name__ == "__main__":
    main()
