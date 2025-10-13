#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generate stratified A/B drum sets by calling the adapter-based generator.
A and B share the same seeds so we can compare apples-to-apples per stratum.

Example:
  python scripts/gen_ab_stratified.py \
    --styles pop_straight,shuffle,rock \
    --densities low,mid,high \
    --tempos 100,120,140 \
    --length-bars 16 \
    --n-per-stratum 3 \
    --A.humanize true \
    --B.humanize true \
    --A.style-override "" \
    --B.style-override "" \
    --out-root output
"""

from __future__ import annotations
import argparse
import json
import time
from pathlib import Path

# 既存の薄層アダプタを再利用
try:
    from adapters.run_drum_adapter import DrumAdapter
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from adapters.run_drum_adapter import DrumAdapter


def parse_csv(s: str) -> list:
    return [x.strip() for x in s.split(",") if x.strip()]


def boolish(s: str) -> bool:
    return str(s).lower() in {"1", "true", "t", "yes", "y", "on"}


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def save_pm(pm, out_mid: Path):
    out_mid.parent.mkdir(parents=True, exist_ok=True)
    pm.write(str(out_mid))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--styles", default="pop_straight,shuffle,rock")
    ap.add_argument("--densities", default="low,mid,high")
    ap.add_argument("--tempos", default="100,120,140")
    ap.add_argument("--length-bars", type=int, default=16)
    ap.add_argument("--time-sig", default="4/4")
    ap.add_argument("--n-per-stratum", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-root", default="output")

    # A/B toggles（必要に応じて増やせます）
    ap.add_argument("--A-humanize", dest="A_humanize", default="true")
    ap.add_argument("--B-humanize", dest="B_humanize", default="true")
    ap.add_argument("--A-style-override", dest="A_style_override", default="")
    ap.add_argument("--B-style-override", dest="B_style_override", default="")
    args = ap.parse_args()

    styles = parse_csv(args.styles)
    densities = parse_csv(args.densities)
    tempos = [int(t) for t in parse_csv(args.tempos)]
    bars = args.length_bars
    time_sig = args.time_sig
    n_per = args.n_per_stratum
    seed0 = args.seed

    outA = Path(args.out_root) / "drumgen_A"
    outB = Path(args.out_root) / "drumgen_B"
    ensure_dir(outA)
    ensure_dir(outB)

    A_hum = boolish(getattr(args, "A_humanize", "true"))
    B_hum = boolish(getattr(args, "B_humanize", "true"))
    A_style_override = getattr(args, "A_style_override", "").strip() or None
    B_style_override = getattr(args, "B_style_override", "").strip() or None

    adapter = DrumAdapter()

    manifest = {"created_at": time.time(), "items": []}

    for style in styles:
        for dens in densities:
            for tempo in tempos:
                tag = f"{style}_{dens}_{tempo}bpm_{bars}bars"
                # A/B 同一seedで n-per を吐く
                for i in range(n_per):
                    seed = seed0 + i

                    # ---- A ----
                    styleA = A_style_override or style
                    rA = adapter.generate_one(
                        tempo=tempo, time_sig=time_sig, length_bars=bars,
                        style=styleA, density=dens, swing=0.0, seed=seed,
                        apply_humanizer=A_hum,
                    )
                    pmA = rA["pretty_midi"]
                    midA = outA / tag / f"A_{tag}_seed{seed}_{i}.mid"
                    save_pm(pmA, midA)

                    # ---- B ----
                    styleB = B_style_override or style
                    rB = adapter.generate_one(
                        tempo=tempo, time_sig=time_sig, length_bars=bars,
                        style=styleB, density=dens, swing=0.0, seed=seed,
                        apply_humanizer=B_hum,
                    )
                    pmB = rB["pretty_midi"]
                    midB = outB / tag / f"B_{tag}_seed{seed}_{i}.mid"
                    save_pm(pmB, midB)

                    manifest["items"].append({
                        "tag": tag,
                        "seed": seed,
                        "A": str(midA),
                        "B": str(midB),
                        "style": style,
                        "density": dens,
                        "tempo": tempo,
                        "bars": bars,
                        "A_humanize": A_hum,
                        "B_humanize": B_hum,
                        "A_style": styleA,
                        "B_style": styleB,
                    })

    man_path = Path(args.out_root) / "drumgen_AB_manifest.json"
    man_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"✅ Wrote manifest: {man_path} (items={len(manifest['items'])})")


if __name__ == "__main__":
    main()
