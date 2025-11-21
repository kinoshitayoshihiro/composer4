#!/usr/bin/env python3
import argparse, json, yaml, numpy as np, pandas as pd
from pathlib import Path


def parse_sections(sec):
    it = sec.get("sections", [])
    out = []
    if it and "start_bar" in it[0]:
        for x in it:
            out.append((int(x["start_bar"]), int(x["end_bar"]), str(x.get("label", "")).lower()))
    else:
        it = sorted(
            [(int(x.get("bar", 0)), str(x.get("label", "")).lower()) for x in it],
            key=lambda x: x[0],
        )
        for i, (b, l) in enumerate(it):
            e = it[i + 1][0] - 1 if i + 1 < len(it) else b + 7
            out.append((b, e, l))
    return out


def scale(v, src, dst):
    a, b = src
    c, d = dst
    if b - a == 0:
        return (c + d) / 2
    t = (v - a) / (b - a)
    t = np.clip(t, 0, 1)
    return c + t * (d - c)


def default():
    return {
        "enable_sections": ["chorus", "bridge"],
        "ql_per_bar": 4,
        "cutoff": {"midi_min": 100, "midi_max": 1000, "f0_midi_range": [60, 84]},
        "volume": {"min": 70, "max": 100, "use_voicing_prob": True},
        "lfo": {"min": 0, "max": 50, "vibrato_depth_cents_range": [0, 50]},
        "downsample_every_ql": 0.25,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocal-f0", required=True)
    ap.add_argument("--sections", required=True)
    ap.add_argument("--policy", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    vdf = pd.read_parquet(a.vocal_f0)
    if "time_ql" not in vdf.columns:
        vdf["time_ql"] = vdf["time_s"] * 4.0
    sec = json.loads(Path(a.sections).read_text(encoding="utf-8"))
    pol = default()
    if a.policy:
        import yaml

        pol.update(yaml.safe_load(Path(a.policy).read_text(encoding="utf-8")) or {})
    ranges = parse_sections(sec)
    en = set(pol["enable_sections"])
    qlpb = pol["ql_per_bar"]
    t_knots = np.arange(
        float(vdf["time_ql"].min()), float(vdf["time_ql"].max()), pol["downsample_every_ql"]
    )
    cce = []
    for tq in t_knots:
        bar = int(tq // qlpb)
        lbl = ""
        for s, e, L in ranges:
            if s <= bar <= e:
                lbl = L
                break
        if not any(k in lbl for k in en):
            continue
        idx = int(np.abs(vdf["time_ql"] - tq).argmin())
        row = vdf.iloc[idx]
        f0m = float(row.get("f0_midi", 69.0))
        # NaN処理
        if np.isnan(f0m):
            f0m = 69.0
        cutoff = int(
            scale(
                f0m,
                pol["cutoff"]["f0_midi_range"],
                [pol["cutoff"]["midi_min"], pol["cutoff"]["midi_max"]],
            )
        )
        vp = float(row.get("voicing_prob", 0.0))
        if np.isnan(vp):
            vp = 0.0
        vol = (
            int(scale(vp, [0, 1], [pol["volume"]["min"], pol["volume"]["max"]]))
            if pol["volume"]["use_voicing_prob"]
            else int((pol["volume"]["min"] + pol["volume"]["max"]) / 2)
        )
        vdepth = float(row.get("vibrato_depth_cents", 0.0))
        if np.isnan(vdepth):
            vdepth = 0.0
        lfo = int(
            scale(
                vdepth,
                pol["lfo"]["vibrato_depth_cents_range"],
                [pol["lfo"]["min"], pol["lfo"]["max"]],
            )
        )
        cce += [
            {"time_ql": float(tq), "cc": 74, "value": cutoff},
            {"time_ql": float(tq), "cc": 7, "value": vol},
            {"time_ql": float(tq), "cc": 1, "value": lfo},
        ]
    out = {
        "meta": {"role": "pad_synth", "generator": "synth_pad_automation", "ql_per_bar": qlpb},
        "automation": cce,
    }
    Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Wrote:", a.out, "events:", len(cce))


if __name__ == "__main__":
    main()
