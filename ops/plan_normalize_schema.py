#!/usr/bin/env python3
# ops/plan_normalize_schema.py
"""
Plan schema normalizer: イベントへ channel/velocity を補完
- トラックヘッダの channel をイベントへ継承
- vel → velocity を同値コピー
- その他の内容（タイミング/音高/長さ）は一切変更しない
"""
import json
from pathlib import Path
import argparse

DEFAULT_CH_BY_ROLE = {
    "drums": 9,     # GMのCh10(=index9)
    "bass": 1,
    "guitar": 2,
    "piano": 3,
    "strings": 4,
}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", required=True, help="Input plan JSON")
    ap.add_argument("--out", dest="out_path", required=True, help="Output plan JSON")
    args = ap.parse_args()

    in_path = Path(args.in_path)
    out_path = Path(args.out_path)

    plan = json.loads(in_path.read_text(encoding="utf-8"))

    tracks = plan.get("tracks")
    if not isinstance(tracks, list):
        raise SystemExit("Plan format error: top-level 'tracks' list not found")

    added_ch = 0
    added_vel = 0
    fixed_tracks = 0

    for tr in tracks:
        name = (tr.get("name") or "").lower()
        role = (tr.get("role") or name).lower()

        # トラックヘッダの channel 確保
        ch = tr.get("channel")
        if ch is None:
            ch = DEFAULT_CH_BY_ROLE.get(role, 0)
            tr["channel"] = ch
            fixed_tracks += 1

        # イベントへ channel/velocity/vel/dur_beats 補完
        events = tr.get("events", [])
        for ev in events:
            if not isinstance(ev, dict):
                continue
            if "channel" not in ev:
                ev["channel"] = ch
                added_ch += 1
            # velocity/vel双方向補完
            if "velocity" not in ev and "vel" in ev:
                ev["velocity"] = ev["vel"]
                added_vel += 1
            if "vel" not in ev and "velocity" in ev:
                ev["vel"] = ev["velocity"]
                added_vel += 1
            # dur_beats補完（start_beats/end_beatsから計算）
            if "dur_beats" not in ev:
                if "end_beats" in ev and "start_beats" in ev:
                    ev["dur_beats"] = max(0.0, float(ev["end_beats"]) - float(ev["start_beats"]))
                elif "duration" in ev:
                    ev["dur_beats"] = float(ev["duration"])
                else:
                    ev["dur_beats"] = 0.5

    # 出力
    out_path.write_text(json.dumps(plan, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")

    print(f"✅ wrote: {out_path}")
    print(f"  tracks fixed(header channel added): {fixed_tracks}")
    print(f"  events added channel: {added_ch}")
    print(f"  events added velocity(from vel): {added_vel}")

if __name__ == "__main__":
    main()
