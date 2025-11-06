#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
chordmap_to_music21.py (patched)
--------------------------------
- Robust mapping from chordmap.json -> music21.harmony.ChordSymbol
- Resolves symbol/quality mismatches in favor of normalized "quality" (with safe fallbacks)
- Expands shorthand (e.g., 7alt -> 7(#9#5), C(b9) -> C7(b9))
- Keeps QL timing; derives durations from successive "time" values
- Optional: writes preview .txt / normalized .json / MIDI (if music21 present)

Usage:
    python chordmap_to_music21.py --input chordmap.json \
        --out-preview chordmap_m21_preview.txt \
        --out-json chordmap.normalized.json \
        --out-midi chordmap.mid

Notes:
    - If music21 is not installed, the script still produces preview/json without failing.
    - When "symbol" and "quality" disagree, we prefer the computed symbol from (root, quality).
      If you want to trust file symbols instead, pass --prefer-symbol.
"""
from __future__ import annotations
import json, argparse, sys, pathlib, statistics
from typing import Dict, Any, Tuple, List

try:
    from music21.harmony import ChordSymbol
    from music21.stream import Stream
    from music21.note import Rest

    MUSIC21_AVAILABLE = True
except Exception:
    ChordSymbol = None  # type: ignore
    Stream = None  # type: ignore
    Rest = None  # type: ignore
    MUSIC21_AVAILABLE = False

QUALITY_MAP: Dict[str, str] = {
    "maj9": "maj9",
    "m7b5": "m7b5",
    "7alt": "7(#9#5)",  # canonical expansion for analysis
    "m9": "m9",
    "9": "9",  # dominant 9th
    "add9": "add9",
    "sus4": "sus4",
    "maj7": "maj7",
    "7": "7",
    "m6": "m6",
    "6": "6",
    "sus2": "sus2",
    "m7": "m7",
    "7b9": "7(b9)",
    "": "",
    None: "",
}


def compute_symbol(root: str, quality: str | None) -> str:
    q = (quality or "").strip()
    if q in ("", None):
        return root
    if q == "7alt":
        return f"{root}7(#9#5)"
    if q == "7b9":
        return f"{root}7(b9)"
    suffix = QUALITY_MAP.get(q, q)
    return f"{root}{suffix}"


def normalize_symbol(sym: str, root: str, quality: str | None) -> str:
    s = (sym or "").strip()
    # Normalize shorthand: "C(b9)" -> "C7(b9)"
    if s == f"{root}(b9)":
        return f"{root}7(b9)"
    return s


def same_chord(a: str, b: str) -> bool:
    # Simple fallback equality (string). When music21 present, attempt parse & compare pitch sets.
    if a.strip() == b.strip():
        return True
    if not MUSIC21_AVAILABLE:
        return False
    try:
        ca = ChordSymbol(a)
        cb = ChordSymbol(b)
        # Compare normalForms as a crude equivalence proxy
        return sorted(p.ps for p in ca.pitches) == sorted(p.ps for p in cb.pitches)
    except Exception:
        return False


def derive_durations(
    events: List[Dict[str, Any]], min_duration_ql: float | None = None
) -> List[float]:
    times = [e["time"] for e in events]
    intervals = [max(0.0, t2 - t1) for t1, t2 in zip(times, times[1:])]
    if not intervals:
        default_last = 4.0
    else:
        try:
            default_last = (
                statistics.median([i for i in intervals if i > 0.0]) or intervals[-1] or 4.0
            )
        except statistics.StatisticsError:
            default_last = intervals[-1] or 4.0
    durs = intervals + [default_last]
    if min_duration_ql is not None:
        durs = [max(min_duration_ql, d) for d in durs]
    return durs


def convert(
    chordmap: Dict[str, Any], prefer_symbol: bool = False, min_duration_ql: float | None = None
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Returns (normalized_events, report)."""
    events = sorted(chordmap.get("events", []), key=lambda e: e.get("time", 0.0))
    durs = derive_durations(events, min_duration_ql=min_duration_ql)

    report: List[Dict[str, Any]] = []
    normalized: List[Dict[str, Any]] = []

    for e, dur in zip(events, durs):
        root = e.get("root", "")
        quality = e.get("quality", "")
        sym_in = (e.get("symbol") or "").strip()
        sym_in_norm = normalize_symbol(sym_in, root, quality) if sym_in else ""
        sym_calc = compute_symbol(root, quality)

        if prefer_symbol and sym_in_norm:
            chosen = sym_in_norm
            # if symbol obviously wrong (e.g., Fmaj7 vs quality m7), fall back with report
            if not same_chord(chosen, sym_calc):
                report.append(
                    {
                        "time_ql": e.get("time"),
                        "root": root,
                        "quality": quality,
                        "symbol_in": sym_in,
                        "normalized_symbol": sym_in_norm,
                        "computed": sym_calc,
                        "decision": "prefer_symbol_but_warn",
                    }
                )
        else:
            if sym_in_norm and same_chord(sym_in_norm, sym_calc):
                chosen = sym_in_norm
            else:
                chosen = sym_calc
                if sym_in:
                    report.append(
                        {
                            "time_ql": e.get("time"),
                            "root": root,
                            "quality": quality,
                            "symbol_in": sym_in,
                            "normalized_symbol": sym_in_norm,
                            "computed": sym_calc,
                            "decision": "prefer_computed",
                        }
                    )

        normalized.append(
            {
                "time": float(e.get("time", 0.0)),
                "root": root,
                "quality": quality,
                "symbol": chosen,
                "duration_ql": float(dur),
            }
        )

    return normalized, report


def build_stream(normalized_events: List[Dict[str, Any]]):
    if not MUSIC21_AVAILABLE:
        return None
    s = Stream()
    for e in normalized_events:
        sym = e["symbol"]
        dur = float(e.get("duration_ql", 4.0))
        try:
            ch = ChordSymbol(sym)
            ch.duration.quarterLength = dur
            s.append(ch)
        except Exception:
            # Fallback: insert rest to keep timeline consistent
            r = Rest(quarterLength=dur)
            s.append(r)
    return s


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="chordmap.json", help="path to chordmap.json")
    ap.add_argument("--bars-file", default=None, help="path to bars.parquet for bar assignment")
    ap.add_argument("--out-preview", default=None, help="write human-readable preview .txt")
    ap.add_argument("--out-json", default=None, help="write normalized chordmap .json")
    ap.add_argument("--out-midi", default=None, help="write MIDI via music21 (optional)")
    ap.add_argument(
        "--prefer-symbol", action="store_true", help="prefer symbols in file when available"
    )
    ap.add_argument(
        "--min-duration-ql", type=float, default=None, help="minimum duration (QL) to enforce"
    )
    ap.add_argument(
        "--add-bar-info", action="store_true", help="add bar field to events using bars.parquet"
    )
    ap.add_argument(
        "--add-symbol", action="store_true", help="add symbol field (root+quality) if missing"
    )
    args = ap.parse_args(argv)

    path = pathlib.Path(args.input)
    with path.open("r", encoding="utf-8") as f:
        chordmap = json.load(f)

    # Optional: add bar info
    if args.add_bar_info and args.bars_file:
        try:
            import pandas as pd

            bars = pd.read_parquet(args.bars_file)
            for e in chordmap.get("events", []):
                time_sec = e.get("time", 0.0)
                bar_idx = -1
                for idx, row in bars.iterrows():
                    if row["start_sec"] <= time_sec < row["end_sec"]:
                        bar_idx = int(idx)
                        break
                e["bar"] = bar_idx
            print(f"[INFO] Added bar field to {len(chordmap['events'])} events")
        except Exception as ex:
            print(f"[WARN] Failed to add bar info: {ex}", file=sys.stderr)

    # Optional: add symbol field
    if args.add_symbol:
        for e in chordmap.get("events", []):
            if "symbol" not in e or not e["symbol"]:
                root = e.get("root", "")
                quality = e.get("quality", "")
                e["symbol"] = compute_symbol(root, quality)
        print(f"[INFO] Added/updated symbol field to {len(chordmap['events'])} events")

    normalized_events, report = convert(
        chordmap, prefer_symbol=args.prefer_symbol, min_duration_ql=args.min_duration_ql
    )

    # Preview
    if args.out_preview:
        lines = [
            f'{e["time"]:>7.1f} ql -> {e["symbol"]:<12s} (dur={e["duration_ql"]:.1f} ql)'
            for e in normalized_events
        ]
        pathlib.Path(args.out_preview).write_text("\n".join(lines), encoding="utf-8")

    # Normalized JSON
    if args.out_json:
        # Include bar field if present in original events
        out = {"unit": "ql", "events": []}
        for i, e in enumerate(normalized_events):
            ev_out = {k: e[k] for k in ("time", "root", "quality", "symbol", "duration_ql")}
            # Copy bar field from original if exists
            if i < len(chordmap.get("events", [])):
                orig = chordmap["events"][i]
                if "bar" in orig:
                    ev_out["bar"] = orig["bar"]
            out["events"].append(ev_out)

        with pathlib.Path(args.out_json).open("w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)

    # MIDI (optional)
    if args.out_midi:
        if not MUSIC21_AVAILABLE:
            print("[WARN] music21 is not available. Skipping MIDI export.", file=sys.stderr)
        else:
            s = build_stream(normalized_events)
            s.write("midi", fp=str(pathlib.Path(args.out_midi)))

    # Diagnostics to stderr
    if report:
        print(f"[INFO] Mismatches / normalizations: {len(report)}", file=sys.stderr)
    else:
        print("[INFO] No symbol/quality mismatches detected.", file=sys.stderr)


if __name__ == "__main__":
    main()
