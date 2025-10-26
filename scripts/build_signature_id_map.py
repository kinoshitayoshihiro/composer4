#!/usr/bin/env python3
"""
Build a mapping from LAMDA SIGNATURES_DATA numeric IDs to time-signature strings.
Strategy:
- Input A: SIGNATURES_DATA pickle: [[file_id, [[sig_id, count], ...]], ...]
- Input B: Stage2 JSON dir that (ideally) already contains sections/timesig_map labels
  e.g., {"timesig_map": [(0, "4/4")]}
- Assumption: Stage2 filename base == LAMDA file_id (or provide CSV mapping)
- Output: YAML map {sig_id: "4/4"} with counts (choose the majority per ID)

Usage:
  python scripts/build_signature_id_map.py \
    --signatures-pickle data/SIGNATURES_DATA.pickle \
    --stage2-json-dir output/stage2/<dataset>/json \
    --out-yaml adapters/signature_id_map.yaml
"""
from __future__ import annotations
import os, json, argparse, pickle, yaml
from collections import defaultdict

def _load_stage2_timesig_label(path: str) -> str:
    try:
        j = json.load(open(path, "r", encoding="utf-8"))
        # try common places
        tsm = j.get("timesig_map") or j.get("meta", {}).get("timesig_map")
        if isinstance(tsm, list) and tsm:
            # take first label
            ent = tsm[0]
            if isinstance(ent, (list,tuple)) and len(ent)>=2:
                return str(ent[1])
        return ""
    except Exception:
        return ""

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--signatures-pickle", required=True)
    ap.add_argument("--stage2-json-dir", required=True)
    ap.add_argument("--out-yaml", required=True)
    args = ap.parse_args()

    # index Stage2 labels by file_id (assumes <file_id>.stage2.json)
    stage2_label = {}
    for fn in os.listdir(args.stage2_json_dir):
        if not fn.endswith(".json") and not fn.endswith(".stage2.json"):
            continue
        fid = os.path.splitext(fn)[0].replace(".stage2","")
        path = os.path.join(args.stage2_json_dir, fn)
        lab = _load_stage2_timesig_label(path)
        if lab:
            stage2_label[fid] = lab

    sig2label_counts = defaultdict(lambda: defaultdict(int))
    data = pickle.load(open(args.signatures_pickle, "rb"))
    for item in data:
        if not isinstance(item, (list,tuple)) or len(item)<2: 
            continue
        fid = str(item[0])
        pairs = item[1] or []
        if fid not in stage2_label:
            continue
        label = stage2_label[fid]
        # take the dominant sig_id in this file
        if not pairs: continue
        top_id = max(pairs, key=lambda x: x[1])[0]
        sig2label_counts[int(top_id)][label] += 1

    # choose majority label per sig_id
    result = {}
    for sig_id, labs in sig2label_counts.items():
        label = max(labs.items(), key=lambda kv: kv[1])[0]
        result[sig_id] = label

    out = {"map": {int(k): str(v) for k,v in sorted(result.items(), key=lambda kv: kv[0])}}
    os.makedirs(os.path.dirname(args.out_yaml) or ".", exist_ok=True)
    with open(args.out_yaml, "w", encoding="utf-8") as f:
        yaml.safe_dump(out, f, allow_unicode=True, sort_keys=True)
    print(f"OK: wrote {args.out_yaml} ({len(result)} entries)")

if __name__ == "__main__":
    main()
