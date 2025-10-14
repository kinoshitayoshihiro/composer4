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
from collections import defaultdict
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


def _tempo_bucket(bpm: int) -> str:
    """Bucket tempo into slow/mid/fast."""
    return "slow" if bpm < 100 else ("mid" if bpm <= 140 else "fast")


def _stable_key(p: Path, base_dir: Path) -> str:
    """Deterministic key independent of glob order."""
    rel = str(p.relative_to(base_dir)).encode("utf-8")
    return hashlib.sha1(rel).hexdigest()


def _nearest_bucket(src: str) -> str:
    """Merge extreme tempo buckets into mid for stability."""
    # slow -> mid, fast -> mid, mid stays mid
    return "mid" if src in ("slow", "fast") else "mid"


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


def stratified_split(toks: List[Dict], val_ratio: float, test_ratio: float, seed: int, midi_dir: Path = None) -> tuple:
    """
    Stratified split by style/tempo/density/key to ensure balanced representation.
    
    Args:
        toks: List of tokenized samples with metadata
        val_ratio: Validation split ratio
        test_ratio: Test split ratio
        seed: Random seed
        midi_dir: Base directory for stable sorting (optional)
    
    Returns:
        (train, val, test, audit_info) tuple
    """
    rng = random.Random(seed)
    
    # 1) Deterministic sort (remove glob order dependency)
    if midi_dir:
        toks = sorted(toks, key=lambda t: _stable_key(Path(t["midi_path"]), midi_dir))
    
    # 2) Group by strata (style, tempo_bucket, density)
    strata = defaultdict(list)
    for tok in toks:
        meta = tok.get("metadata", {})
        style = meta.get("style", "unknown")
        tempo = int(meta.get("tempo", 120))
        density = meta.get("density", "mid")
        
        tempo_bucket = _tempo_bucket(tempo)
        
        stratum_key = (style, tempo_bucket, density)
        strata[stratum_key].append(tok)
    
    # 3) Absorb micro-strata (len < 3) into nearest tempo bucket
    moved = []
    for (sty, tb, den), lst in list(strata.items()):
        if len(lst) < 3 and len(lst) > 0:
            nb = _nearest_bucket(tb)
            if nb != tb:
                strata[(sty, nb, den)].extend(lst)
                moved.append({
                    "from": f"{sty}/{tb}/{den}",
                    "to": f"{sty}/{nb}/{den}",
                    "count": len(lst)
                })
                strata[(sty, tb, den)] = []
    
    # Clean up empty strata
    strata = {k: v for k, v in strata.items() if v}
    
    print(f"[info] Stratified split: {len(strata)} strata found")
    for k, v in sorted(strata.items()):
        print(f"  - {k[0]}/{k[1]}/{k[2]}: {len(v)} samples")
    
    if moved:
        print(f"[info] Absorbed {len(moved)} micro-strata into nearest buckets")
    
    # 4) Split each stratum proportionally
    train_all, val_all, test_all = [], [], []
    
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
    
    # 5) Audit info for reproducibility checking
    dist = {}
    for (sty, tb, den), lst in strata.items():
        dist[f"{sty}/{tb}/{den}"] = len(lst)
    
    audit_info = {
        "moved_micro_strata": moved,
        "distribution": dist
    }
    
    return train_all, val_all, test_all, audit_info


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
    train, val, test, audit_info = stratified_split(toks, args.val_ratio, args.test_ratio, args.seed, midi_dir)
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

    # Save strata distribution audit info
    (out / "strata_distribution.json").write_text(
        json.dumps(audit_info, indent=2, ensure_ascii=False)
    )
    print(f"[saved] Strata distribution audit: {len(audit_info['distribution'])} strata")

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
