#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
render_from_package.py
Quick, reproducible stems rendering from a song_package.yaml.
- Uses Fluidsynth (if installed) + an SF2 soundfont for simple guide stems.
- Injects program changes into per-part MIDI if needed.
- Produces stems and a render_config.yaml + render_report.json.

Dependencies:
  - PyYAML (pip install pyyaml)
  - mido (pip install mido)
  - (optional) Fluidsynth CLI installed and a .sf2 soundfont

Example:
  python render_from_package.py \
    --package /path/to/midi_guide/SONG123/song_package.yaml \
    --soundfont /path/to/GeneralUser.sf2 \
    --outdir /path/to/renders/SONG123 \
    --preset-map '{"piano":0, "guitar":24, "bass":32, "drums":128, "vocal":0}'

Notes:
- drums preset 128 means "force channel 10 (9 in zero-based)".
- You can skip rendering and only write a ready-to-run render_config.yaml by omitting --soundfont.
"""
import argparse, json, os, subprocess, shutil, sys
from pathlib import Path

# ---------- small YAML loader ----------
def load_yaml(path: Path):
    try:
        import yaml
    except Exception as e:
        print("[ERR] PyYAML is required. pip install pyyaml", file=sys.stderr)
        raise
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def dump_yaml(data: dict, path: Path):
    try:
        import yaml
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)
    except Exception as e:
        # fallback minimal JSON if yaml is missing (shouldn't happen as we read YAML above)
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

def inject_program_change(in_mid: Path, out_mid: Path, program: int, force_drum=False):
    """Create a copy of MIDI with program assignment at track start.
       If force_drum=True, move all drum events to channel 9 (10th)."""
    from mido import MidiFile, MidiTrack, Message, MetaMessage
    mf = MidiFile(str(in_mid))
    # Create new file with same ticks_per_beat
    out = MidiFile(ticks_per_beat=mf.ticks_per_beat)
    for ti, track in enumerate(mf.tracks):
        newt = MidiTrack()
        # insert program change at delta=0 for non-drum parts
        if force_drum:
            # ensure channel 9
            newt.append(Message('program_change', channel=9, program=0, time=0))
            remap_ch = 9
        else:
            # default to channel 0
            newt.append(Message('program_change', channel=0, program=max(0, min(127, program)), time=0))
            remap_ch = 0
        abs_time = 0
        for msg in track:
            abs_time += msg.time
            if msg.is_meta:
                # keep meta (tempo, time_signature, etc.)
                newt.append(msg.copy(time=msg.time))
            else:
                m = msg.copy()
                if hasattr(m, "channel"):
                    m.channel = remap_ch
                newt.append(m)
        out.tracks.append(newt)
    out.save(str(out_mid))

def which(prog: str):
    return shutil.which(prog)

def render_with_fluidsynth(sf2: Path, midi: Path, out_wav: Path, sample_rate=48000):
    cmd = [
        "fluidsynth", "-ni", str(sf2), str(midi),
        "-F", str(out_wav), "-r", str(sample_rate)
    ]
    return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--package", required=True, help="Path to song_package.yaml")
    ap.add_argument("--soundfont", default=None, help="Path to .sf2 (if omitted, only config files are written)")
    ap.add_argument("--outdir", required=True, help="Where to write stems and configs")
    ap.add_argument("--preset-map", default='{\"piano\":0, \"guitar\":24, \"bass\":32, \"drums\":128, \"vocal\":0}',
                    help='JSON: part->GM program. drums=128 means force channel10.')
    ap.add_argument("--sample-rate", type=int, default=48000)
    args = ap.parse_args()

    pkg = load_yaml(Path(args.package))
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    # collect midi parts
    base_dir = Path(args.package).parent
    guides = (pkg.get("guides") or {}).get("midi") or {}
    parts = {}
    for name, rel in guides.items():
        midi_path = (base_dir / rel).resolve()
        if midi_path.exists():
            parts[name] = midi_path

    preset_map = json.loads(args.preset_map)

    # write render_config.yaml
    render_conf = {
        "package": os.path.relpath(args.package, start=outdir),
        "soundfont": args.soundfont,
        "sample_rate": args.sample_rate,
        "parts": [],
        "notes": "Programs follow GM numbering. drums=128 means channel10."
    }
    for name, mp in parts.items():
        program = int(preset_map.get(name, 0))
        force_drum = (program >= 128)
        render_conf["parts"].append({
            "name": name,
            "midi": os.path.relpath(mp, start=outdir),
            "program": program if not force_drum else 0,
            "force_drum_channel": bool(force_drum),
            "stem_out": f"{name}.wav"
        })
    dump_yaml(render_conf, outdir / "render_config.yaml")

    # if no fluidsynth, stop after config
    if args.soundfont is None:
        print("[INFO] soundfont not provided; wrote render_config.yaml only.")
        return
    if which("fluidsynth") is None:
        print("[WARN] fluidsynth not found in PATH; wrote render_config.yaml only.")
        return

    # render stems
    report = {"stems": [], "errors": []}
    for p in render_conf["parts"]:
        name = p["name"]
        src_mid = (outdir / f"__tmp_{name}.mid")
        try:
            inject_program_change(
                in_mid=(Path(outdir) / p["midi"]).resolve() if (outdir / p["midi"]).exists() else (Path(args.package).parent / p["midi"]).resolve(),
                out_mid=src_mid,
                program=p["program"],
                force_drum=p["force_drum_channel"]
            )
            out_wav = outdir / p["stem_out"]
            r = render_with_fluidsynth(Path(args.soundfont), src_mid, out_wav, sample_rate=args.sample_rate)
            ok = (r.returncode == 0 and out_wav.exists())
            report["stems"].append({
                "name": name, "midi": p["midi"], "out": str(out_wav), "ok": ok,
                "stdout": r.stdout[-4000:], "stderr": r.stderr[-4000:]
            })
            if not ok:
                report["errors"].append({"name": name, "stderr": r.stderr})
        except Exception as e:
            report["errors"].append({"name": name, "exception": str(e)})
        finally:
            if src_mid.exists():
                try: src_mid.unlink()
                except: pass

    (outdir / "render_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[DONE] wrote stems and render_report.json")

if __name__ == "__main__":
    main()
