#!/usr/bin/env python3
"""Drum adapter CLI for batch generation."""
from __future__ import annotations
import argparse
import json
import time
import hashlib
from pathlib import Path
from generator.drum.adapter import DrumAdapter


def save_pm(pm, out_mid: Path):
    """Save PrettyMIDI to file."""
    out_mid.parent.mkdir(parents=True, exist_ok=True)
    pm.write(str(out_mid))
    return out_mid


def write_sidecar(midi_path: Path, meta: dict):
    """Write .meta.json sidecar file."""
    sc = midi_path.with_suffix(".meta.json")
    sc.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description="Generate drum patterns via DrumAdapter")
    ap.add_argument("--n", type=int, default=4, help="Number of samples")
    ap.add_argument("--tempo", type=int, default=120, help="Tempo in BPM")
    ap.add_argument("--time-sig", default="4/4", help="Time signature")
    ap.add_argument("--length-bars", type=int, default=64, help="Length in bars")
    ap.add_argument("--style", default="pop_straight", help="Pattern style")
    ap.add_argument("--density", default="mid", choices=["low", "mid", "high"])
    ap.add_argument("--swing", type=float, default=0.0, help="Swing amount")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    ap.add_argument("--no-humanize", action="store_true", help="Skip humanization")
    ap.add_argument("--patterns-dir", default="data/drum_patterns")
    ap.add_argument("--out", default="output/drumgen", help="Output directory")
    args = ap.parse_args()

    adapter = DrumAdapter(patterns_dir=args.patterns_dir)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    batch = []
    
    for i in range(args.n):
        print(f"Generating sample {i+1}/{args.n}...", end=" ", flush=True)
        
        r = adapter.generate_one(
            tempo=args.tempo,
            time_sig=args.time_sig,
            length_bars=args.length_bars,
            style=args.style,
            density=args.density,
            swing=args.swing,
            seed=args.seed + i,
            apply_humanizer=not args.no_humanize,
        )
        
        pm = r["pretty_midi"]
        tokens = r["tokens"]
        
        mid = out_dir / f"drum_{args.style}_{args.tempo}bpm_{args.length_bars}bars_{args.seed+i}.mid"
        save_pm(pm, mid)

        h = hashlib.sha1()
        h.update(mid.read_bytes())
        h.update(str(args.seed+i).encode())
        h.update(str(time.time()).encode())
        gen_id = h.hexdigest()[:16]

        meta = {
            "gen_id": gen_id,
            "seed": args.seed + i,
            "tempo": args.tempo,
            "time_sig": args.time_sig,
            "length_bars": args.length_bars,
            "style": args.style,
            "density": args.density,
            "swing": args.swing,
            "remi_version": "1.1.0",
            "token_count": len(tokens) if tokens else 0,
            "artifacts": {"midi_path": str(mid)},
        }
        write_sidecar(mid, meta)
        batch.append({"midi_path": str(mid), "meta": meta})
        
        print(f"✓ {mid.name}")

    (out_dir / "batch_meta.json").write_text(
        json.dumps({"items": batch}, ensure_ascii=False, indent=2), "utf-8"
    )
    
    print(f"\n✅ Generated {len(batch)} samples to {out_dir}")


if __name__ == "__main__":
    main()
