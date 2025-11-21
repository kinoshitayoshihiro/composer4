#!/usr/bin/env python3
"""Combine manual chord symbols with timing/metadata from bars + auto chordmap."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


def load_json(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def detect_root(symbol: str) -> Tuple[str, str]:
    """Split symbol into root and the rest of the quality string."""
    if not symbol:
        return "", ""
    s = symbol.strip()
    # handle slash chords first (e.g. Cmaj7/E)
    base = s.split("/")[0]
    if not base:
        return "", ""
    root = base[0]
    if len(base) >= 2 and base[1] in ("#", "b"):
        root += base[1]
        remainder = base[2:]
    else:
        remainder = base[1:]
    return root, remainder


def split_tensions(symbol: str) -> Tuple[str, List[str], Optional[str]]:
    """Extract base part, tension list, and slash bass from a symbol string."""
    if not symbol:
        return "", [], None
    base, bass = symbol, None
    if "/" in symbol:
        base, bass = symbol.split("/", 1)
        bass = bass.strip() or None
    tensions: List[str] = []
    if "(" in base and base.endswith(")"):
        body, tail = base.split("(", 1)
        base = body
        tail = tail[:-1]  # drop ')' (safe as endswith already checked)
        tensions = [t.strip() for t in tail.split(",") if t.strip()]
    return base, tensions, bass


def normalize_auto_events(auto_obj: Dict, bars_df: pd.DataFrame) -> Dict[int, Dict]:
    """Map auto chord events to bar index using start/end seconds."""
    events = auto_obj.get("events") or []
    bars_records = [
        {
            "bar": int(row.bar_index),
            "start_sec": float(row.start_sec),
            "end_sec": float(row.end_sec),
        }
        for row in bars_df.itertuples()
    ]
    bar_lookup: Dict[int, Dict] = {}
    for ev in events:
        time_sec = float(ev.get("time", ev.get("time_sec", 0.0)))
        bar_idx = None
        for br in bars_records:
            if br["start_sec"] - 1e-6 <= time_sec < br["end_sec"] + 1e-6:
                bar_idx = br["bar"]
                break
        if bar_idx is None:
            continue
        bar_lookup[bar_idx] = {
            "auto_time_sec": time_sec,
            "auto_root": ev.get("root"),
            "auto_quality": ev.get("quality"),
            "auto_symbol": ev.get("symbol"),
            "auto_raw": ev,
        }
    return bar_lookup


def build_event(manual_event: Dict, bars_df: pd.DataFrame, auto_info: Optional[Dict]) -> Dict:
    bar = int(manual_event["bar"])
    symbol = manual_event.get("symbol", "").strip()
    row = bars_df.loc[bars_df["bar_index"] == bar].iloc[0]
    base, tensions, bass = split_tensions(symbol)
    root, quality = detect_root(base)
    event = {
        "bar": bar,
        "symbol": symbol,
        "root": root,
        "quality": quality,
        "tensions": tensions,
        "bass": bass,
        "time": float(row["start_sec"]),
        "time_ql": float(row["start_beat"]),
        "duration_sec": float(row["end_sec"] - row["start_sec"]),
        "duration_ql": float(row["end_beat"] - row["start_beat"]),
        "source": "manual_enriched",
    }
    if auto_info:
        event.update(auto_info)
    return event


def main() -> None:
    parser = argparse.ArgumentParser(description="Enrich manual_chordmap with timing/meta data.")
    parser.add_argument("--manual", type=Path, required=True, help="manual_chordmap.json path")
    parser.add_argument("--bars", type=Path, required=True, help="bars.parquet path")
    parser.add_argument("--out", type=Path, required=True, help="output enriched chordmap path")
    parser.add_argument("--auto", type=Path, help="optional auto chordmap for reference")
    args = parser.parse_args()

    manual = load_json(args.manual)
    events = manual.get("events") or manual.get("chords")
    if not events:
        raise SystemExit("manual chordmap has no events")

    bars_df = pd.read_parquet(args.bars)
    auto_lookup = {}
    if args.auto:
        auto_lookup = normalize_auto_events(load_json(args.auto), bars_df)

    enriched_events = []
    for ev in events:
        bar = int(ev.get("bar", 0))
        auto_info = auto_lookup.get(bar)
        enriched_events.append(build_event(ev, bars_df, auto_info))

    enriched = {
        "unit": "bar",
        "meta": {
            "base": str(args.manual),
            "bars": str(args.bars),
            "auto": str(args.auto) if args.auto else None,
            "notes": "manual symbols enriched with timing/root/tension metadata",
        },
        "events": enriched_events,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(enriched, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
