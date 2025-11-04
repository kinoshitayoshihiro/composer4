#!/usr/bin/env python3
import json, argparse
from pathlib import Path
import pandas as pd

def default_channel(role: str) -> int:
    role = (role or "").lower()
    if role == "drums":   return 9   # GM: ch10
    if role == "bass":    return 1
    if role == "guitar":  return 2
    if role == "piano":   return 3
    if role == "strings": return 4
    return 0

def beats_per_bar_from_parquet(bars_path: Path, default_bpb=4):
    try:
        df = pd.read_parquet(bars_path)
        if {"start_beat","end_beat"}.issubset(df.columns):
            bpb = (df["end_beat"] - df["start_beat"]).round(6).mode().iloc[0]
            return int(bpb) if float(bpb).is_integer() else float(bpb)
        if {"start_beats","end_beats"}.issubset(df.columns):
            bpb = (df["end_beats"] - df["start_beats"]).round(6).mode().iloc[0]
            return int(bpb) if float(bpb).is_integer() else float(bpb)
    except Exception:
        pass
    return default_bpb

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", required=True)
    ap.add_argument("--out", dest="out_path", required=True)
    ap.add_argument("--bars", dest="bars_parquet", default=None)
    ap.add_argument("--default-vel", type=int, default=80)
    args = ap.parse_args()

    plan = json.loads(Path(args.in_path).read_text(encoding="utf-8"))
    bpb = beats_per_bar_from_parquet(Path(args.bars_parquet)) if args.bars_parquet else 4

    total, fixed = 0, {"bar":0,"beat":0,"start_beats":0,"end_beats":0,"dur_beats":0,"vel":0,"velocity":0,"channel":0}
    for tr in plan.get("tracks", []):
        role = tr.get("name") or tr.get("role") or ""
        ch   = tr.get("channel")
        if ch is None:
            ch = default_channel(role)
            tr["channel"] = ch

        for ev in tr.get("events", []):
            total += 1

            # velocity/vel 同期
            if "vel" not in ev and "velocity" in ev:
                ev["vel"] = int(ev["velocity"]); fixed["vel"] += 1
            if "velocity" not in ev and "vel" in ev:
                ev["velocity"] = int(ev["vel"]); fixed["velocity"] += 1
            if "vel" not in ev and "velocity" not in ev:
                ev["vel"] = ev["velocity"] = int(args.default_vel); fixed["vel"] += 1; fixed["velocity"] += 1

            # channel 補完（イベント側に無ければ付ける）
            if "channel" not in ev:
                ev["channel"] = int(ch); fixed["channel"] += 1

            # dur_beats / duration_beats 同期
            if "dur_beats" not in ev and "duration_beats" in ev:
                ev["dur_beats"] = float(ev["duration_beats"]); fixed["dur_beats"] += 1
            if "duration_beats" not in ev and "dur_beats" in ev:
                ev["duration_beats"] = float(ev["dur_beats"])

            # bar/beat が無い場合の救済（start_beats から割り戻し）
            if "bar" not in ev or "beat" not in ev:
                if "start_beats" in ev:
                    sb = float(ev["start_beats"])
                    bar = int(sb // bpb)
                    beat = sb - bar * bpb
                    if "bar" not in ev:
                        ev["bar"] = bar; fixed["bar"] += 1
                    if "beat" not in ev:
                        ev["beat"] = round(beat, 6); fixed["beat"] += 1

            # start/end_beats を埋める
            if "start_beats" not in ev:
                if "bar" in ev and "beat" in ev:
                    ev["start_beats"] = float(ev["bar"]) * bpb + float(ev["beat"]); fixed["start_beats"] += 1
            if "end_beats" not in ev:
                dur = float(ev.get("dur_beats", ev.get("duration_beats", 0.0)))
                if "start_beats" in ev and dur > 0:
                    ev["end_beats"] = float(ev["start_beats"]) + dur; fixed["end_beats"] += 1

    Path(args.out_path).write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"✅ normalized: {total} events | fixes: {fixed} | bpb={bpb}")

if __name__ == "__main__":
    main()
