#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
batch_from_packages.py
Run render and/or QA over many song_package.yaml files under a unified LOCAL_LAMDA tree.

Features
- Discovers packages: Local_Lamda_midi/midi_guide/*/song_package.yaml
- Supports multiple tasks: render, qa (either or both)
- Parallel execution (--workers N)
- Dataset inference (from ids.dataset, or from hub path "wav_guide/<dataset>/...")
- Skips up-to-date results unless --force
- Writes an index CSV summarizing results

Dependencies: PyYAML (and for QA: pandas/pyarrow; for render: mido + Fluidsynth+SF2 if rendering)
"""
import argparse, csv, json, os, sys, time, subprocess, re
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

def load_yaml(path: Path):
    try:
        import yaml
    except Exception:
        print("[ERR] PyYAML is required (pip install pyyaml)", file=sys.stderr)
        raise
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def discover_packages(base: Path):
    pkg_root = base / "Local_Lamda_midi" / "midi_guide"
    return sorted(pkg_root.glob("*/song_package.yaml"))

def infer_dataset(pkg_dict: dict, pkg_path: Path):
    # 1) ids.dataset
    ids = pkg_dict.get("ids") or {}
    ds = ids.get("dataset")
    if ds: return ds
    # 2) hub path includes wav_guide/<dataset>/
    hub = pkg_dict.get("hub") or {}
    bars_rel = hub.get("bars_parquet")
    if bars_rel:
        full = (pkg_path.parent / bars_rel).resolve()
        parts = [p for p in full.parts]
        # look for "wav_guide/<dataset>/"
        for i, p in enumerate(parts[:-1]):
            if p == "wav_guide" and i+1 < len(parts):
                return parts[i+1]
    return "unknown"

def is_up_to_date(target: Path, *sources: Path):
    if not target.exists():
        return False
    t_mtime = target.stat().st_mtime
    for s in sources:
        if s.exists() and s.stat().st_mtime > t_mtime:
            return False
    return True

def run_subprocess(args_list, cwd=None):
    p = subprocess.run(args_list, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return p.returncode, p.stdout[-4000:], p.stderr[-4000:]

def process_one(base: Path, pkg_path: Path, tasks: set, soundfont: str, preset_map: str,
                render_out_root: Path, qa_out_root: Path, force=False, add_audio_chordmap=False, include_dataset_level=False):
    pkg = load_yaml(pkg_path)
    dataset = infer_dataset(pkg, pkg_path)
    song_id = (pkg.get("ids") or {}).get("song_id") or pkg_path.parent.name

    results = {"song_id": song_id, "dataset": dataset, "package": str(pkg_path), "render_ok": "", "qa_ok": "", "render_out": "", "qa_out": ""}

    # RENDER
    if "render" in tasks:
        outdir = render_out_root / dataset / song_id
        outdir.mkdir(parents=True, exist_ok=True)
        rendered_stem = outdir / "piano.wav"  # heuristic check; any stem triggers "done"
        if force or not rendered_stem.exists():
            cmd = [
                sys.executable, str(Path(__file__).parent / "render_from_package.py"),
                "--package", str(pkg_path),
                "--outdir", str(outdir),
                "--preset-map", preset_map,
            ]
            if soundfont: cmd += ["--soundfont", soundfont]
            rc, so, se = run_subprocess(cmd)
            results["render_ok"] = (rc == 0)
            results["render_out"] = str(outdir)
            if rc != 0:
                results["render_err"] = se
        else:
            results["render_ok"] = True
            results["render_out"] = str(outdir)

    # QA
    if "qa" in tasks:
        qa_dir = qa_out_root / dataset
        qa_dir.mkdir(parents=True, exist_ok=True)
        qa_json = qa_dir / f"{song_id}_qa.json"
        if force or not is_up_to_date(qa_json, pkg_path):
            cmd = [
                sys.executable, str(Path(__file__).parent / "qa_from_package.py"),
                "--package", str(pkg_path),
                "--out", str(qa_json),
                "--csv", str(qa_dir / f"{song_id}_qa.csv")
            ]
            rc, so, se = run_subprocess(cmd)
            results["qa_ok"] = (rc == 0)
            results["qa_out"] = str(qa_json)
            if rc != 0:
                results["qa_err"] = se
        else:
            results["qa_ok"] = True
            results["qa_out"] = str(qa_json)

    return results

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="Path to LOCAL_LAMDA base")
    ap.add_argument("--dataset", action="append", default=None,
                    help="Filter to dataset(s). Repeat or comma-separated (e.g., 'moisesdb,musdb18').")
    ap.add_argument("--tasks", default="render,qa", help="Comma-separated: render,qa")
    ap.add_argument("--soundfont", default=None, help="Path to .sf2 for render")
    ap.add_argument("--preset-map", default='{\"piano\":0, \"guitar\":24, \"bass\":32, \"drums\":128, \"vocal\":0}')
    ap.add_argument("--render-out", default=None, help="Root for renders (default: {base}/renders)")
    ap.add_argument("--qa-out", default=None, help="Root for QA reports (default: {base}/qa)")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--force", action="store_true", help="Ignore up-to-date checks")
    ap.add_argument("--index-out", default=None, help="Write a CSV index summarizing results")
    args = ap.parse_args()

    base = Path(args.base)
    tasks = {t.strip() for t in args.tasks.split(",") if t.strip()}
    filt_ds = None
    if args.dataset:
        filt_ds = set()
        for item in args.dataset:
            for s in item.split(","):
                s = s.strip()
                if s: filt_ds.add(s)

    render_out_root = Path(args.render_out) if args.render_out else (base / "renders")
    qa_out_root = Path(args.qa_out) if args.qa_out else (base / "qa")

    packages = discover_packages(base)
    if not packages:
        print("[WARN] no song_package.yaml found under Local_Lamda_midi/midi_guide/*", file=sys.stderr)
        return

    # Filter by dataset if requested
    selected = []
    for p in packages:
        try:
            pkg = load_yaml(p)
        except Exception:
            continue
        ds = infer_dataset(pkg, p)
        if filt_ds and ds not in filt_ds:
            continue
        selected.append(p)

    print(f"[INFO] {len(selected)} packages selected out of {len(packages)} found.")

    rows = []
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
        futs = []
        for p in selected:
            futs.append(ex.submit(process_one, base, p, tasks, args.soundfont, args.preset_map,
                                  render_out_root, qa_out_root, args.force))
        for fu in as_completed(futs):
            res = fu.result()
            rows.append(res)
            ok_r = ("render" not in tasks) or res.get("render_ok", False)
            ok_q = ("qa" not in tasks) or res.get("qa_ok", False)
            status = "OK" if (ok_r and ok_q) else "ERR"
            print(f"[{status}] {res.get('dataset')}/{res.get('song_id')}")

    dur = time.time() - t0
    print(f"[DONE] {len(rows)} processed in {dur:.1f}s")

    if args.index_out:
        outp = Path(args.index_out); outp.parent.mkdir(parents=True, exist_ok=True)
        cols = ["dataset","song_id","package","render_ok","render_out","qa_ok","qa_out","render_err","qa_err"]
        with open(outp, "w", newline="", encoding="utf-8") as w:
            wr = csv.DictWriter(w, fieldnames=cols); wr.writeheader()
            for r in rows:
                wr.writerow({k: r.get(k, "") for k in cols})
        print(f"[INDEX] wrote {outp}")

if __name__ == "__main__":
    main()
