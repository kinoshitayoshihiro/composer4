#!/usr/bin/env python3
from __future__ import annotations
"""
Augment Stage2 JSONs with LAMDA META/KILO/SIGNATURES/TOTALS (v1.2)
- Adds meta normalization via adapters.meta_key_normalizer
- NO-OP safe
"""
import os, json, argparse, glob, pickle, yaml
from typing import Dict, Any, List, Optional
from adapters.lamda_kilo_reader import lookup_in_kilo
from adapters.lamda_meta_reader import iter_meta_pickle, extract_stage2_overlays
from adapters.meta_key_normalizer import normalize_meta
from adapters.lamda_totals_prior import build_priors, score_against_priors
from adapters.lamda_chords_decoder import decode_chord_seq_to_events

def _load_json(path: str) -> Dict[str,Any]:
    return json.load(open(path,"r",encoding="utf-8"))

def _save_json(path: str, obj: Dict[str,Any]):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path,"w",encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def _load_signature_map(path: Optional[str]) -> Dict[int,str]:
    if not path or not os.path.exists(path):
        return {}
    y = yaml.safe_load(open(path,"r",encoding="utf-8")) or {}
    return {int(k): str(v) for k,v in (y.get("map") or {}).items()}

def _timesig_from_signatures(fid: str, signatures_pickle: Optional[str], sig_map: Dict[int,str]) -> List:
    if not signatures_pickle or not os.path.exists(signatures_pickle):
        return []
    data = pickle.load(open(signatures_pickle, "rb"))
    for item in data:
        if isinstance(item, (list,tuple)) and len(item)>=2 and str(item[0]) == fid:
            pairs = item[1] or []
            if not pairs: 
                return []
            top = max(pairs, key=lambda x: x[1])
            sig_id = int(top[0])
            label = sig_map.get(sig_id, f"unknown:{sig_id}")
            return [(0, label)]
    return []

def _meta_for(fid: str, meta_globs: List[str]) -> Dict[str,Any]:
    if not meta_globs: return {}
    for pattern in meta_globs:
        for pkl in glob.glob(pattern):
            for f_id, md in iter_meta_pickle(pkl):
                if f_id == fid:
                    norm = normalize_meta(md)
                    return norm
    return {}

def _totals_priors(totals_pickle: Optional[str]) -> Optional[Dict[str,Any]]:
    if not totals_pickle or not os.path.exists(totals_pickle):
        return None
    totals = pickle.load(open(totals_pickle, "rb"))
    if isinstance(totals, dict):
        return build_priors(totals)
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage2-json", required=True)
    ap.add_argument("--file-id", required=True)
    ap.add_argument("--kilo-pickle", default=None)
    ap.add_argument("--meta-pickle", action="append", default=[],
                    help="glob patterns accepted; can be given multiple times")
    ap.add_argument("--signatures-pickle", default=None)
    ap.add_argument("--signature-map", default=None)
    ap.add_argument("--totals-pickle", default=None)
    ap.add_argument("--write-back", type=int, default=1)
    ap.add_argument("--out-json", default=None)
    args = ap.parse_args()

    j = _load_json(args.stage2_json)

    if args.kilo_pickle and os.path.exists(args.kilo_pickle):
        seq = lookup_in_kilo(args.kilo_pickle, args.file_id)
        if seq is not None:
            cm = decode_chord_seq_to_events(seq, tpq=480, token_map_yaml="adapters/lamda_chords_token_map.yaml")
            if cm.get("events"):
                j["chordmap"] = cm

    meta = _meta_for(args.file_id, args.meta_pickle)
    if meta:
        j["lamda_meta"] = meta

    sig_map = _load_signature_map(args.signature_map)
    tsmap = _timesig_from_signatures(args.file_id, args.signatures_p, sig_map) if False else _timesig_from_signatures(args.file_id, args.signatures_pickle, sig_map)
    if tsmap:
        j["timesig_map"] = tsmap

    priors = _totals_priors(args.totals_pickle)
    if priors:
        j["global_priors"] = priors

    j["schema_version"] = "lamda_v2.2"
    outp = args.out_json or args.stage2_json
    if args.write_back:
        _save_json(outp, j)
        print(f"OK: augmented {outp}")
    else:
        print(json.dumps(j, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
