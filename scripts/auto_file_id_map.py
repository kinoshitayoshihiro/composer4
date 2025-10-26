#!/usr/bin/env python3
from __future__ import annotations
"""
Auto-build a mapping CSV between EXT chordmaps and GOLD Stage2 JSONs.
Heuristics: label-seq hash, bar length, tempo (if available), Hamming distance on labels.

Usage:
  python scripts/auto_file_id_map.py \
    --gold-dir data/GOLD_stage2_json \
    --ext-dir  data/lamda_chordmaps \
    --out-csv  mappings/file_map.csv
"""
import os, json, argparse, csv, glob, hashlib
from typing import Dict, Any, List, Tuple

def _load_cm_json(path: str) -> Dict[str,Any]:
    try:
        j = json.load(open(path, "r", encoding="utf-8"))
        return j if "events" in j else (j.get("chordmap") or {})
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

def _tempo_rough(stage2_like: Dict[str,Any]) -> float:
    # Prefer tempo_map[0], else infer from downbeats_ql spacing (if present)
    tm = stage2_like.get("tempo_map") or []
    if isinstance(tm, list) and tm:
        try:
            return float(tm[0][1])  # (time_sec, bpm)
        except Exception:
            pass
    db = stage2_like.get("downbeats_ql") or stage2_like.get("downbeats") or []
    if isinstance(db, list) and len(db) >= 2:
        # assume ~ 1 bar = 4 QL. We can't get seconds; return pseudo-tempo in QL
        # Use "bars per 100 QL" as a rough scalar to compare sequences (unitless)
        bars = len(db) - 1
        span = float(db[-1] - db[0]) if db else 0.0
        if span > 0:
            return 100.0 * (bars * 4.0) / span  # higher = 'faster'
    return -1.0

def _sig_hash(labels: List[str]) -> str:
    s = "|".join(labels)
    return hashlib.md5(s.encode("utf-8")).hexdigest()

def _hamming(a: List[str], b: List[str]) -> int:
    m = min(len(a), len(b))
    return sum(1 for i in range(m) if a[i] != b[i]) + abs(len(a) - len(b))

def _gold_payload(path: str) -> Tuple[List[str], int, float]:
    j = json.load(open(path, "r", encoding="utf-8"))
    cm = (j.get("chordmap") or {})
    bars = _bars_len(cm)
    lab = _events_to_bar_labels(cm, bars)
    tempo = _tempo_rough(j)
    return lab, bars, tempo

def _ext_payload(path: str) -> Tuple[List[str], int]:
    cm = _load_cm_json(path)
    bars = _bars_len(cm)
    lab = _events_to_bar_labels(cm, bars)
    return lab, bars

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold-dir", required=True)
    ap.add_argument("--ext-dir", required=True)
    ap.add_argument("--out-csv", required=True)
    args = ap.parse_args()

    gold_files = {os.path.splitext(os.path.basename(p))[0].replace(".stage2",""): p
                  for p in glob.glob(os.path.join(args.gold_dir, "*.json"))}
    ext_files  = {os.path.splitext(os.path.basename(p))[0]: p
                  for p in glob.glob(os.path.join(args.ext_dir, "*.json"))}

    # preprocess features
    gold_feat = {}
    for k,p in gold_files.items():
        try:
            lab, bars, tempo = _gold_payload(p)
            gold_feat[k] = {
                "labels": lab,
                "bars": bars,
                "tempo": tempo,
                "hash": _sig_hash(lab)
            }
        except Exception:
            pass

    ext_feat = {}
    for k,p in ext_files.items():
        try:
            lab, bars = _ext_payload(p)
            ext_feat[k] = {
                "labels": lab,
                "bars": bars,
                "hash": _sig_hash(lab)
            }
        except Exception:
            pass

    # exact-hash pass
    mapping = {}
    used_ext = set()
    for gk, gf in gold_feat.items():
        for ek, ef in ext_feat.items():
            if ef["hash"] == gf["hash"]:
                mapping[gk] = ek
                used_ext.add(ek)
                break

    # approximate pass
    for gk, gf in gold_feat.items():
        if gk in mapping: continue
        candidates = []
        for ek, ef in ext_feat.items():
            if ek in used_ext: continue
            if abs(ef["bars"] - gf["bars"]) > 2:
                continue
            # if tempo known, require close
            if gf["tempo"] > 0.0:
                # ext has no tempo -> skip check; otherwise compare pseudo-tempo if available
                pass
            ham = _hamming(gf["labels"], ef["labels"])
            candidates.append((ham, ek))
        if candidates:
            candidates.sort()
            mapping[gk] = candidates[0][1]
            used_ext.add(mapping[gk])

    # write CSV
    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["file_id","ext_base","gold_base"])
        for gk, ek in mapping.items():
            w.writerow([gk, ek, gk])
    print(f"OK: wrote mapping -> {args.out_csv} (pairs={len(mapping)})")

if __name__ == "__main__":
    main()
