"""
LAMDA CHORDS_DATA → chordmap.json 索引生成（再実行安全）
Usage:
  python scripts/lamda_chords_to_index.py \
    --chords-dir data/Los-Angeles-MIDI/CHORDS_DATA \
    --out-dir data/lamda_chordmaps \
    --tpq 480 \
    --token-map adapters/lamda_chords_token_map.yaml \
    --resume
"""
import os, glob, json, pickle, argparse
from adapters.lamda_chords_decoder import decode_chord_seq_to_events

def main():
    ap = argparse.ArgumentParser(description="LAMDA CHORDS → chordmap.json 索引生成（再実行安全）")
    ap.add_argument("--chords-dir", required=True, help="LAMDA CHORDS_DATA directory")
    ap.add_argument("--out-dir", required=True, help="Output directory for chordmap.json files")
    ap.add_argument("--tpq", type=int, default=480, help="Ticks per quarter note (default: 480)")
    ap.add_argument("--token-map", default="adapters/lamda_chords_token_map.yaml",
                    help="YAML token map for LAMDA-specific chord encoding")
    ap.add_argument("--resume", action="store_true", 
                    help="Skip existing output files (resume-safe)")
    ap.add_argument("--out-index", default="index.json",
                    help="Output index filename (default: index.json)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    index = {}
    idx_path = os.path.join(args.out_dir, args.out_index)
    
    # Resume: load existing index
    if args.resume and os.path.exists(idx_path):
        try:
            with open(idx_path, "r", encoding="utf-8") as f:
                index = json.load(f)
            print(f"📂 Resuming from existing index: {len(index)} entries")
        except Exception as e:
            print(f"⚠️  Could not load existing index: {e}")
            index = {}

    pkl_files = sorted(glob.glob(os.path.join(args.chords_dir, "LAMDa_CHORDS_DATA_*.pickle")))
    print(f"🎵 Found {len(pkl_files)} CHORDS pickle files")
    
    processed = 0
    skipped = 0
    errors = 0
    
    for pkl in pkl_files:
        print(f"\n📁 Processing: {os.path.basename(pkl)}")
        try:
            data = pickle.load(open(pkl, "rb"))
            for file_id, chord_seq in data:
                out_json = os.path.join(args.out_dir, f"{file_id}.chordmap.json")
                
                # Resume-safe: skip existing
                if args.resume and os.path.exists(out_json):
                    index[file_id] = out_json
                    skipped += 1
                    continue
                
                # Decode chord sequence
                try:
                    cm = decode_chord_seq_to_events(
                        chord_seq, 
                        tpq=args.tpq, 
                        token_map_yaml=args.token_map
                    )
                    
                    # Skip empty chordmaps
                    if not cm.get("events"):
                        print(f"  ⚠️  Empty chordmap for {file_id}, skipping")
                        continue
                    
                    # Write chordmap JSON
                    with open(out_json, "w", encoding="utf-8") as f:
                        json.dump(cm, f, ensure_ascii=False, indent=2)
                    
                    index[file_id] = out_json
                    processed += 1
                    
                    # Progress report
                    if processed % 100 == 0:
                        print(f"  ✅ Processed {processed} files...")
                        
                except Exception as e:
                    print(f"  ❌ Error decoding {file_id}: {e}")
                    errors += 1
                    continue
                    
        except Exception as e:
            print(f"❌ Error loading pickle {pkl}: {e}")
            errors += 1
            continue
    
    # Write index
    with open(idx_path, "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=2)
    
    print(f"\n{'='*60}")
    print(f"✅ SUCCESS: {len(index)} total chordmaps → {args.out_dir}")
    print(f"   📊 Statistics:")
    print(f"      - Processed: {processed}")
    print(f"      - Skipped (resume): {skipped}")
    print(f"      - Errors: {errors}")
    print(f"      - Index: {idx_path}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
