#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ops/stem_harmony_batch.py

ステム→chordmap.json をディレクトリ単位で一括実行。
各 song 直下に analysis/chordmap.json を生成。

例:
python ops/stem_harmony_batch.py \
  --root data/suno_ai \
  --glob "*/**/stems" \
  --exclude Vocals --exclude "Backing Vocals" \
  --include-N \
  --stem-weight "bass=1.3" --stem-weight "keys=1.2" --stem-weight "fx=0.6"
"""
from __future__ import annotations
import argparse, subprocess, sys
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--glob", default="**/stems")
    ap.add_argument("--exclude", action="append", default=[])
    ap.add_argument("--sections-name", default="analysis/sections.json")
    ap.add_argument("--out-name", default="analysis/chordmap.json")
    # YAML config support
    ap.add_argument("--config", help="YAML/JSON config for stem_harmony.py")
    # passthrough for stem_harmony.py
    ap.add_argument("--include-N", action="store_true")
    ap.add_argument("--stem-weight", action="append", default=[])
    ap.add_argument("--sr", type=int, default=22050)
    ap.add_argument("--bins-per-octave", type=int, default=36)
    ap.add_argument("--stay", type=float, default=0.93)
    ap.add_argument("--near", type=float, default=0.03)
    ap.add_argument("--gamma-global", type=float, default=0.15)
    ap.add_argument("--gamma-local", type=float, default=0.30)
    ap.add_argument("--n-energy-gamma", type=float, default=1.0)
    ap.add_argument("--n-conf-gamma", type=float, default=2.0)
    ap.add_argument("--ql-per-beat", type=float, default=1.0)
    args = ap.parse_args()

    root = Path(args.root)
    stems = list(root.glob(args.glob))
    if not stems:
        print(f"[WARN] No stems matched under {root} with glob={args.glob}", file=sys.stderr)
        sys.exit(0)

    ok = 0
    for stems_dir in stems:
        song_dir = stems_dir.parent
        sections = song_dir / args.sections_name
        out      = song_dir / args.out_name
        cmd = [
            sys.executable, "ops/stem_harmony.py",
            "--stems", str(stems_dir),
            "--out",   str(out),
            "--sr",    str(args.sr),
            "--bins-per-octave", str(args.bins_per_octave),
            "--stay",  str(args.stay),
            "--near",  str(args.near),
            "--gamma-global", str(args.gamma_global),
            "--gamma-local",  str(args.gamma_local),
            "--n-energy-gamma", str(args.n_energy_gamma),
            "--n-conf-gamma",   str(args.n_conf_gamma),
            "--ql-per-beat",    str(args.ql_per_beat),
        ]
        if args.config:
            cmd += ["--config", args.config]
        if args.include_N:
            cmd.append("--include-N")
        for ex in args.exclude:
            cmd += ["--exclude", ex]
        if sections.exists():
            cmd += ["--sections", str(sections)]
        for w in (args.stem_weight or []):
            cmd += ["--stem-weight", w]

        print("[RUN]", " ".join(cmd))
        r = subprocess.run(cmd)
        if r.returncode == 0:
            ok += 1
            print(f"[OK] {out}")
        else:
            print(f"[FAIL] {stems_dir}", file=sys.stderr)

    print(f"[DONE] {ok}/{len(stems)} chordmaps generated.")

if __name__ == "__main__":
    main()
