#!/usr/bin/env python3
import argparse, json, numpy as np, pandas as pd, yaml, random
from pathlib import Path


def J(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def default_policy():
    return {
        "phase_ms": {"base_delay_ms": [20, 40], "allow_negative_ms": False},
        "humanize": {
            "start_gauss_std": 8,
            "start_clamp": 15,
            "dur_gauss_std": 12,
            "dur_clamp": 25,
            "swing_16th_pct": 6,
            "drift_guard_ms": 30,
        },
        "ql_per_bar": 4,
        "bpm": 90.0,
        "section_override": {"chorus": {"base_delay_ms": [10, 25], "swing_16th_pct": 4}},
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


def ms_per_ql(bpm):
    return (60.0 / bpm) * 1000.0


def clamp(x, mn, mx):
    return max(mn, min(mx, x))


def bar_label(bar, ranges):
    for s, e, l in ranges:
        if s <= bar <= e:
            return l
    return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--guitar-plan", required=True)
    ap.add_argument("--sections", required=True)
    ap.add_argument("--policy", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--csv", required=True)
    a = ap.parse_args()
    plan = J(a.guitar_plan)
    sec = J(a.sections)
    pol = default_policy()
    if a.policy:
        pol.update(yaml.safe_load(Path(a.policy).read_text(encoding="utf-8")) or {})
    qlpb = int(plan.get("meta", {}).get("ql_per_bar", pol["ql_per_bar"]))
    ranges = []
    it = sec.get("sections", [])
    if it and ("start_bar" in it[0] or "start" in it[0]):
        for x in it:
            sb = int(x.get("start_bar", x.get("start", 0)))
            eb = int(x.get("end_bar", x.get("end", sb)))
            ranges.append((sb, eb, str(x.get("label", "")).lower()))
    bpm = float(pol.get("bpm", 90.0))
    ms_perql = ms_per_ql(bpm)
    rows = []
    for tr in plan.get("tracks", []):
        for ev in tr.get("events", []):
            # start_beats または time を使用
            time_val = ev.get("time", ev.get("start_beats", 0))
            bar = int(time_val // qlpb)
            lbl = bar_label(bar, ranges)
            base_delay = pol["phase_ms"]["base_delay_ms"]
            if "chorus" in lbl and "chorus" in pol.get("section_override", {}):
                base_delay = pol["section_override"]["chorus"]["base_delay_ms"]
            delay = random.uniform(base_delay[0], base_delay[1])
            if not pol["phase_ms"].get("allow_negative_ms", False):
                delay = abs(delay)
            sj = clamp(
                random.gauss(0, pol["humanize"]["start_gauss_std"]),
                -pol["humanize"]["start_clamp"],
                pol["humanize"]["start_clamp"],
            )
            dj = clamp(
                random.gauss(0, pol["humanize"]["dur_gauss_std"]),
                -pol["humanize"]["dur_clamp"],
                pol["humanize"]["dur_clamp"],
            )
            frac = time_val % 0.5
            swing_ms = (
                (ms_perql * 0.25) * (pol["humanize"]["swing_16th_pct"] / 100.0)
                if 0.25 <= frac < 0.5
                else 0.0
            )
            ev["time_shift_ms"] = float(delay + sj + swing_ms)
            ev["dur_shift_ms"] = float(dj)
            rows.append(
                {
                    "bar": bar,
                    "time_ql": time_val,
                    "time_shift_ms": ev["time_shift_ms"],
                    "dur_shift_ms": ev["dur_shift_ms"],
                    "section": lbl,
                }
            )
    Path(a.out).write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
    import pandas as pd

    pd.DataFrame(rows).to_csv(a.csv, index=False)
    print("Wrote:", a.out, "rows:", len(rows))


if __name__ == "__main__":
    main()
