#!/usr/bin/env python3
import argparse, json, numpy as np, pandas as pd, yaml
from pathlib import Path


def J(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def default_policy():
    return {
        "enable_sections": ["verse", "pre", "chorus", "bridge"],
        "comp": {"notes_per_bar": [2, 6], "prefer_offbeat": True, "stress_guard_ql": 0.08},
        "line": {
            "use_f0": True,
            "sustain_min_ql": 0.5,
            "sustain_max_ql": 2.0,
            "slope_T_st_per_s": 1.2,
            "tail_extend_on_vibrato_cents": 25,
        },
        "ql_per_bar": 4,
    }


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocal-f0", required=True)
    ap.add_argument("--vocal-events", required=True)
    ap.add_argument("--sections", required=True)
    ap.add_argument("--policy", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    pol = default_policy()
    if a.policy:
        pol.update(yaml.safe_load(Path(a.policy).read_text(encoding="utf-8")) or {})
    vdf = pd.read_parquet(a.vocal_f0)
    evdf = pd.read_parquet(a.vocal_events)
    sec = J(a.sections)
    if "time_ql" not in vdf.columns:
        vdf["time_ql"] = vdf["time_s"] * 4.0
    qlpb = pol["ql_per_bar"]
    ranges = parse_sections(sec)
    events = []

    def in_en(b):
        for s, e, l in ranges:
            if s <= b <= e and any(k in l for k in pol["enable_sections"]):
                return True
        return False

    offs = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    maxn = lambda slope: int(np.clip(round(np.interp(abs(slope), [0, 3], [2, 6])), 2, 6))
    bars = sorted({int(x // qlpb) for x in vdf["time_ql"]})
    for b in bars:
        if not in_en(b):
            continue
        vb = vdf[(vdf["time_ql"] >= b * qlpb) & (vdf["time_ql"] < (b + 1) * qlpb)]
        slope = (
            float(np.nanmedian(np.gradient(vb["f0_smooth_hz"].fillna(0.0))))
            if not vb.empty
            else 0.0
        )
        for off in offs[: maxn(slope)]:
            events.append(
                {
                    "bar": b,
                    "time": b * qlpb + off,
                    "duration_ql": 0.35,
                    "velocity": 70,
                    "pitch_midi": 64,
                }
            )
    for b in bars:
        if not in_en(b):
            continue
        vb = vdf[(vdf["time_ql"] >= b * qlpb) & (vdf["time_ql"] < (b + 1) * qlpb)]
        if vb.empty:
            continue
        tail = vb.iloc[-1]
        dur = 0.8
        f0_midi = tail.get("f0_midi", 64)
        if np.isnan(f0_midi):
            f0_midi = 64
        events.append(
            {
                "bar": b,
                "time": float(b * qlpb + 3.25),
                "duration_ql": dur,
                "velocity": 60,
                "pitch_midi": int(round(f0_midi)),
            }
        )
    plan = {
        "meta": {"role": "piano", "generator": "piano_hybrid", "ql_per_bar": qlpb},
        "tracks": [{"name": "Piano", "role": "piano", "events": events}],
    }
    Path(a.out).write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Wrote:", a.out, "events:", len(events))


if __name__ == "__main__":
    main()
