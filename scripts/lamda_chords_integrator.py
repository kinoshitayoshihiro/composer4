import os, glob, pickle, json
from pathlib import Path
from typing import Dict, Any, List
from adapters.lamda_chords_decoder import decode_chord_seq_to_events

def build_index(chords_dir: str, out_dir: str, tpq: int=480, token_map_yaml: str=None):
    os.makedirs(out_dir, exist_ok=True)
    idx: Dict[str,str] = {}
    for pkl in glob.glob(os.path.join(chords_dir, "LAMDa_CHORDS_DATA_*.pickle")):
        data = pickle.load(open(pkl,"rb"))
        for file_id, chord_seq in data:
            cm = decode_chord_seq_to_events(chord_seq, tpq=tpq, token_map_yaml=token_map_yaml)
            if not cm.get("events"): continue
            out_path = os.path.join(out_dir, f"{file_id}.chordmap.json")
            with open(out_path,"w",encoding="utf-8") as f:
                json.dump(cm, f, ensure_ascii=False, indent=2)
            idx[file_id]=out_path
    with open(os.path.join(out_dir,"index.json"),"w",encoding="utf-8") as f:
        json.dump(idx, f, ensure_ascii=False, indent=2)
    print(f"OK: {len(idx)} chordmaps → {out_dir}")

if __name__=="__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--chords-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--tpq", type=int, default=480)
    ap.add_argument("--token-map", default="adapters/lamda_chords_token_map.yaml")
    args = ap.parse_args()
    build_index(args.chords_dir, args.out_dir, tpq=args.tpq, token_map_yaml=args.token_map)
