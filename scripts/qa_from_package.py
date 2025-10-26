#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
qa_from_package.py
Lightweight QA for a song_package.yaml.
- Checks presence/paths, reads bars.parquet if possible, summarizes durations, parts, diagnostics.
- Outputs qa_report.json (+ optional CSV with --csv).

Dependencies:
  - PyYAML (pip install pyyaml)
  - (optional) pandas + pyarrow (pip install pandas pyarrow) for bars.parquet
  - mido (pip install mido) for MIDI timing summary

Example:
  python qa_from_package.py \
    --package /path/to/midi_guide/SONG123/song_package.yaml \
    --out /path/to/qa/SONG123_qa.json \
    --csv /path/to/qa/SONG123_qa.csv
"""
import argparse, json, os, sys
from pathlib import Path

def load_yaml(path: Path):
    try:
        import yaml
    except Exception:
        print("[ERR] PyYAML is required. pip install pyyaml", file=sys.stderr)
        raise
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def try_read_bars_parquet(p: Path):
    try:
        import pandas as pd
        df = pd.read_parquet(p)
        return {
            "path": str(p),
            "rows": int(len(df)),
            "cols": list(df.columns),
            "head": df.head(3).to_dict(orient="records")
        }
    except Exception as e:
        return {"path": str(p), "error": str(e)}

def midi_summary(p: Path):
    try:
        from mido import MidiFile
        mf = MidiFile(str(p))
        # mido length is approximate (default tempo if not specified)
        return {
            "ticks_per_beat": mf.ticks_per_beat,
            "tracks": len(mf.tracks),
            "approx_length_s": round(mf.length, 3)
        }
    except Exception as e:
        return {"error": str(e)}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--package", required=True, help="Path to song_package.yaml")
    ap.add_argument("--out", required=True, help="Path to qa_report.json")
    ap.add_argument("--csv", default=None, help="Optional CSV mirror of key metrics")
    args = ap.parse_args()

    pkg_path = Path(args.package)
    pkg = load_yaml(pkg_path)
    base_dir = pkg_path.parent

    report = {
        "package": str(pkg_path),
        "ids": pkg.get("ids"),
        "spec_present": {},
        "diagnostics_present": {},
        "bars": None,
        "midi_parts": {},
        "warnings": []
    }

    # hub: bars
    bars_rel = (pkg.get("hub") or {}).get("bars_parquet")
    if bars_rel:
        bars_path = (base_dir / bars_rel).resolve()
        if bars_path.exists():
            report["bars"] = try_read_bars_parquet(bars_path)
        else:
            report["warnings"].append(f"bars_parquet missing: {bars_path}")
    else:
        report["warnings"].append("bars_parquet not set in package.hub")

    # spec
    spec = pkg.get("spec") or {}
    for key in ("sections", "chordmap", "anchors"):
        if key in spec:
            p = (base_dir / spec[key]).resolve()
            report["spec_present"][key] = p.exists()
            if not p.exists():
                report["warnings"].append(f"spec missing: {key} -> {p}")
        else:
            report["spec_present"][key] = False

    # diagnostics
    diag = pkg.get("diagnostics") or {}
    for key, rel in diag.items():
        if isinstance(rel, dict):
            # dataset_level or nested
            for k2, r2 in rel.items():
                p = (base_dir / r2).resolve()
                report["diagnostics_present"][f"{key}.{k2}"] = p.exists()
                if not p.exists():
                    report["warnings"].append(f"diagnostics missing: {key}.{k2} -> {p}")
        else:
            p = (base_dir / rel).resolve()
            report["diagnostics_present"][key] = p.exists()
            if not p.exists():
                report["warnings"].append(f"diagnostics missing: {key} -> {p}")

    # MIDI parts
    mids = (pkg.get("guides") or {}).get("midi") or {}
    for name, rel in mids.items():
        p = (base_dir / rel).resolve()
        if p.exists():
            report["midi_parts"][name] = midi_summary(p)
        else:
            report["midi_parts"][name] = {"missing": True}
            report["warnings"].append(f"midi part missing: {name} -> {p}")

    # write json
    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] wrote QA report: {outp}")

    # optional CSV
    if args.csv:
        try:
            import csv
            with open(args.csv, "w", newline="", encoding="utf-8") as w:
                wr = csv.writer(w)
                wr.writerow(["key","value"])
                wr.writerow(["package", str(pkg_path)])
                wr.writerow(["song_id", (pkg.get("ids") or {}).get("song_id","")])
                wr.writerow(["dataset", (pkg.get("ids") or {}).get("dataset","")])
                wr.writerow(["has_sections", report["spec_present"].get("sections")])
                wr.writerow(["has_chordmap", report["spec_present"].get("chordmap")])
                wr.writerow(["has_anchors", report["spec_present"].get("anchors")])
                wr.writerow(["bars_rows", (report["bars"] or {}).get("rows","")])
                wr.writerow(["midi_parts", ",".join(sorted(mids.keys()))])
            print(f"[OK] wrote CSV: {args.csv}")
        except Exception as e:
            print(f"[WARN] failed to write CSV: {e}", file=sys.stderr)

if __name__ == "__main__":
    main()
