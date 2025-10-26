#!/usr/bin/env python3
from __future__ import annotations
"""
Top-N diff extractor -> batch thumbnail PNG generator.

- Input: audit CSV (columns: file_id, bars, match_rate, n_diff, diff_bars)
- Select top-N by (low match_rate, high n_diff)
- Resolve GOLD/EXT chordmap JSON paths (with optional mapping CSV)
- Call visualize_chord_diff.py to render 1xN heatmap PNGs

Usage:
  python scripts/build_topn_thumbs.py \
    --audit-csv analysis/ext_vs_gold.csv \
    --gold-dir  data/GOLD_stage2_json \
    --ext-dir   data/lamda_chordmaps \
    --out-dir   analysis/thumbs \
    --top-n 50 \
    [--mapping-csv mappings/file_map.csv]
"""
import os, csv, argparse, subprocess, shlex, glob

def _load_mapping(mapping_csv):
    if not mapping_csv or not os.path.exists(mapping_csv):
        return {}
    mp = {}
    with open(mapping_csv, "r", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        def pick(row, keys):
            for k in keys:
                if k in row and row[k]: return row[k]
            return ""
        for row in rd:
            fid = pick(row, ["file_id","id","fid"])
            extb = pick(row, ["ext_base","ext","ext_name"])
            goldb = pick(row, ["gold_base","gold","gold_name"])
            if fid:
                mp[fid] = (extb or fid, goldb or fid)
    return mp

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit-csv", required=True)
    ap.add_argument("--gold-dir", required=True)
    ap.add_argument("--ext-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--top-n", type=int, default=50)
    ap.add_argument("--mapping-csv", default=None)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    mapping = _load_mapping(args.mapping_csv)

    # load audit rows
    rows = []
    with open(args.audit_csv, "r", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        for r in rd:
            try:
                r["_match"] = float(r.get("match_rate", 0.0))
                r["_ndiff"] = int(r.get("n_diff", 0))
            except Exception:
                continue
            rows.append(r)

    # sort by (match_rate asc, n_diff desc)
    rows.sort(key=lambda r: (r["_match"], -r["_ndiff"]))
    rows = rows[: args.top_n]

    # index files
    gold = {os.path.splitext(os.path.basename(p))[0].replace(".stage2",""): p
            for p in glob.glob(os.path.join(args.gold_dir, "*.json"))}
    ext  = {os.path.splitext(os.path.basename(p))[0]: p
            for p in glob.glob(os.path.join(args.ext_dir, "*.json"))}

    # render thumbs
    ok = 0
    for r in rows:
        fid = r["file_id"]
        # pick GOLD base
        gb = fid
        if fid in mapping: gb = mapping[fid][1] or gb
        if gb not in gold:
            # try plain fid
            if fid in gold: gb = fid
            else: continue
        # pick EXT base
        eb = fid
        if fid in mapping: eb = mapping[fid][0] or eb
        if eb not in ext:
            # try GOLD name also as EXT key
            if gb in ext: eb = gb
            else: continue
        gpath = gold[gb]
        epath = ext[eb]
        out_png = os.path.join(args.out_dir, f"{fid}.png")
        cmd = f'python scripts/visualize_chord_diff.py --gold-json "{gpath}" --ext-json "{epath}" --out-png "{out_png}"'
        try:
            subprocess.run(shlex.split(cmd), check=True)
            ok += 1
        except Exception as e:
            print("skip:", fid, "|", e)
    print(f"OK: generated {ok}/{len(rows)} thumbs -> {args.out_dir}")

if __name__ == "__main__":
    main()
