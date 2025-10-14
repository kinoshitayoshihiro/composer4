#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Piano training data preparation:
Stage1/2 cleaned MIDI → REMI tokens → train/val/test splits

Usage:
    python scripts/piano_train_prepare.py \\
      --midi-dir output/piano_cleaned \\
      --out-dir data/piano_splits \\
      --seed 1234
"""

import argparse
import json
import hashlib
import random
from pathlib import Path
from typing import List, Dict, Any

import pretty_midi

# Import token utilities
import sys
sys.path.insert(0, str(Path(__file__).parent))
from token_utils import load_remi_tokenizer, encode_pm


def list_midis(root: Path) -> List[Path]:
    """Recursively find all MIDI files."""
    paths = []
    for ext in ["*.mid", "*.midi"]:
        paths.extend(root.rglob(ext))
    return sorted(set(paths))


def sha256_file(path: Path) -> str:
    """Compute SHA256 hash of file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def extract_metadata(midi_path: Path) -> Dict[str, Any]:
    """
    Extract metadata from MIDI sidecar (.meta.json) if available.
    Returns: {"style": str, "tempo": int, "density": str, "key": str}
    """
    meta_path = midi_path.with_suffix(".meta.json")
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text("utf-8"))
            conditions = meta.get("conditions", {})
            return {
                "style": conditions.get("style", "unknown"),
                "tempo": int(conditions.get("tempo", 120)),
                "density": conditions.get("density", "mid"),
                "key": conditions.get("key", "C")
            }
        except Exception:
            pass
    
    # Fallback: infer from filename or defaults
    name = midi_path.stem.lower()
    style = "block" if "block" in name else ("arpeggio" if "arpeggio" in name else "unknown")
    tempo = 120
    density = "mid"
    key = "C"
    
    return {"style": style, "tempo": tempo, "density": density, "key": key}


def crop_midi_by_bars(pm: pretty_midi.PrettyMIDI, max_bars: int, tempo: float = 120.0) -> pretty_midi.PrettyMIDI:
    """
    Crop MIDI to max_bars length.
    
    Args:
        pm: PrettyMIDI object
        max_bars: Maximum number of bars
        tempo: Assumed tempo for bar calculation (default 120 BPM, 4/4)
    
    Returns:
        Cropped PrettyMIDI object
    """
    bar_len = 4 * (60.0 / tempo)  # 4 beats × seconds per beat
    limit = max_bars * bar_len
    
    for inst in pm.instruments:
        inst.notes = [n for n in inst.notes if n.start < limit]
    
    return pm


def stratified_split(toks: List[Dict], val_ratio: float, test_ratio: float, seed: int) -> tuple:
    """
    Stratified split by style/tempo/density/key to ensure balanced representation.
    
    Args:
        toks: List of tokenized samples with metadata
        val_ratio: Validation split ratio
        test_ratio: Test split ratio
        seed: Random seed
    
    Returns:
        (train, val, test) lists
    """
    from collections import defaultdict
    
    # Group by strata (style, tempo_bucket, density)
    strata = defaultdict(list)
    for tok in toks:
        meta = tok.get("metadata", {})
        style = meta.get("style", "unknown")
        tempo = meta.get("tempo", 120)
        density = meta.get("density", "mid")
        
        # Bucket tempo: slow(<100), mid(100-140), fast(>140)
        tempo_bucket = "slow" if tempo < 100 else ("fast" if tempo > 140 else "mid")
        
        stratum_key = f"{style}_{tempo_bucket}_{density}"
        strata[stratum_key].append(tok)
    
    print(f"[info] Stratified split: {len(strata)} strata found")
    for k, v in sorted(strata.items()):
        print(f"  - {k}: {len(v)} samples")
    
    # Split each stratum proportionally
    train_all, val_all, test_all = [], [], []
    rng = random.Random(seed)
    
    for stratum_key, stratum_toks in strata.items():
        rng.shuffle(stratum_toks)
        n = len(stratum_toks)
        n_test = max(1, int(n * test_ratio))
        n_val = max(1, int(n * val_ratio))
        
        test_all.extend(stratum_toks[:n_test])
        val_all.extend(stratum_toks[n_test:n_test + n_val])
        train_all.extend(stratum_toks[n_test + n_val:])
    
    # Final shuffle
    rng.shuffle(train_all)
    rng.shuffle(val_all)
    rng.shuffle(test_all)
    
    return train_all, val_all, test_all


def main():
    ap = argparse.ArgumentParser(description="Prepare Piano training data from cleaned MIDI")
    ap.add_argument("--midi-dir", required=True, help="Stage1/2 passed Piano MIDI folder")
    ap.add_argument("--out-dir", required=True, help="Output directory: data/piano_splits/")
    ap.add_argument("--seed", type=int, default=1234, help="Random seed")
    ap.add_argument("--val-ratio", type=float, default=0.05, help="Validation split ratio")
    ap.add_argument("--test-ratio", type=float, default=0.05, help="Test split ratio")
    ap.add_argument("--max-bars", type=int, default=64, help="Max bars per sample (for stability)")
    ap.add_argument("--min-length", type=int, default=32, help="Min token length (filter out too short)")
    args = ap.parse_args()

    random.seed(args.seed)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Load tokenizer
    print("[info] Loading REMI tokenizer...")
    tk = load_remi_tokenizer(remi_enabled=True)

    # Find MIDI files
    midi_dir = Path(args.midi_dir)
    mids = list_midis(midi_dir)
    if not mids:
        raise SystemExit(f"No MIDI files found under: {midi_dir}")
    print(f"[info] Found {len(mids)} MIDI files")

    # Tokenize with metadata extraction
    toks = []
    skipped = 0
    for i, mp in enumerate(mids):
        if (i + 1) % 100 == 0:
            print(f"[progress] {i+1}/{len(mids)} processed...")
        
        try:
            pm = pretty_midi.PrettyMIDI(str(mp))
            
            # Crop if needed
            if args.max_bars:
                pm = crop_midi_by_bars(pm, args.max_bars)
            
            # Encode
            ids = encode_pm(tk, pm)
            
            if not isinstance(ids, (list, tuple)) or len(ids) < args.min_length:
                skipped += 1
                continue
            
            # Extract metadata for stratification
            metadata = extract_metadata(mp)

            toks.append({
                "midi_path": str(mp.relative_to(midi_dir)),
                "length": len(ids),
                "ids": ids,
                "metadata": metadata
            })
        except Exception as e:
            print(f"[skip] {mp.name}: {e}")
            skipped += 1

    print(f"[info] Tokenized: {len(toks)}, Skipped: {skipped}")

    # Stratified split (maintains style/tempo/density distribution)
    train, val, test = stratified_split(toks, args.val_ratio, args.test_ratio, args.seed)
    N = len(toks)

    # Write JSONL
    def dump_jsonl(rows: List[Dict], path: Path):
        with open(path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    dump_jsonl(train, out / "train.jsonl")
    dump_jsonl(val, out / "val.jsonl")
    dump_jsonl(test, out / "test.jsonl")
    print(f"[saved] train={len(train)}, val={len(val)}, test={len(test)}")

    # Dataset hash (reproducibility)
    h = hashlib.sha256()
    for p in mids:
        h.update(sha256_file(p).encode())
    ds_hash = h.hexdigest()[:16]

    # Metadata
    meta = {
        "dataset_size": N,
        "splits": {"train": len(train), "val": len(val), "test": len(test)},
        "dataset_hash": ds_hash,
        "tokenizer": getattr(tk, "REMI_VERSION", "unknown"),
        "max_bars": args.max_bars,
        "min_length": args.min_length,
        "seed": args.seed
    }
    (out / "dataset_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print(f"[done] Metadata: {meta}")


if __name__ == "__main__":
    main()
