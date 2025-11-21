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
    # Basic triads
    "": "",
    "major": "",
    "maj": "",
    "minor": "m",
    "m": "m",
    "min": "m",
    "dim": "dim",
    "aug": "aug",
    "+": "aug",
    # 7th chords
    "7": "7",
    "maj7": "maj7",
    "M7": "maj7",
    "m7": "m7",
    "min7": "m7",
    "dim7": "dim7",
    "m7b5": "m7b5",
    "half-diminished": "m7b5",
    # 6th chords
    "6": "6",
    "m6": "m6",
    "6/9": "6/9",
    # Suspended
    "sus2": "sus2",
    "sus4": "sus4",
    "7sus4": "7sus4",
    # Extended (9th, 11th, 13th) - PRESERVE TENSIONS
    "9": "9",
    "maj9": "maj9",
    "M9": "maj9",
    "m9": "m9",
    "min9": "m9",
    "add9": "add9",
    "add2": "add9",  # add2 = add9
    "11": "11",
    "m11": "m11",
    "13": "13",
    "maj13": "maj13",
    # Alterations
    "7b9": "7(b9)",
    "7#9": "7(#9)",
    "7b13": "7(b13)",
    "7#5": "7(#5)",
    "7b5": "7(b5)",
    "7alt": "7(#9#5)",  # canonical expansion
    "alt": "7(#9#5)",
    # Special cases
    "m(maj7)": "m(maj7)",
    "mM7": "m(maj7)",
    None: "",
}


def normalize_quality(quality: str | None) -> str:
    """
    quality を正規化。

    重要: minor の二重 "m" を防ぐため、root から独立して処理

    Examples:
        "m" -> "m"
        "minor" -> "m"
        "maj7" -> "maj7"
        "major" -> ""
        "9" -> "9" (preserve tension)
    """
    q = (quality or "").strip().lower()

    # QUALITY_MAP から検索
    if q in QUALITY_MAP:
        return QUALITY_MAP[q]

    # 未知の quality はそのまま返す（ログに残す）
    return q


def compute_symbol(root: str, quality: str | None) -> str:
    """
    root + quality から symbol を生成。

    CRITICAL: root に "m" が含まれていても quality は独立処理

    Examples:
        ("E", "m") -> "Em"
        ("Em", "m") -> "Em" (NOT "Emm")
        ("A", "m7") -> "Am7"
        ("Am", "m7") -> "Am7" (NOT "Amm7")
        ("G", "9") -> "G9" (preserve tension)
        ("C", "add9") -> "Cadd9"
    """
    r = (root or "").strip()
    q_norm = normalize_quality(quality)

    # root が既に minor 記号 "m" で終わり、quality も "m" で始まる場合は重複を除去
    # 例: root="Em", quality="m" -> "Em" (not "Emm")
    # 例: root="Am", quality="m7" -> "Am7" (not "Amm7")
    if r.endswith("m") and q_norm.startswith("m"):
        # root の末尾 "m" を除去
        r = r[:-1]

    if not q_norm:
        return r

    return f"{r}{q_norm}"


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
    """
    QL（四分音符）時間を正（source of truth）として duration を計算。

    CRITICAL: time_ql を優先し、秒（time）はフォールバックのみ。
    後段（json2midi.py）でテンポマップを適用して秒変換する。
    """
    # time_ql 優先（QL が source of truth）
    times = [e.get("time_ql", e.get("time", 0.0)) for e in events]
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
    # time_ql 優先でソート（QL が source of truth）
    events = sorted(chordmap.get("events", []), key=lambda e: e.get("time_ql", e.get("time", 0.0)))
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
                        "time_ql": e.get("time_ql", e.get("time", 0.0)),
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
                            "time_ql": e.get("time_ql", e.get("time", 0.0)),
                            "root": root,
                            "quality": quality,
                            "symbol_in": sym_in,
                            "normalized_symbol": sym_in_norm,
                            "computed": sym_calc,
                            "decision": "prefer_computed",
                        }
                    )

        # CRITICAL: time_ql を正として normalized に格納（秒は別フィールド time_sec で保持）
        normalized.append(
            {
                "time_ql": float(e.get("time_ql", e.get("time", 0.0))),
                "time_sec": float(e.get("time", 0.0)),  # 秒は参考値として保持
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
        "--min-duration-ql",
        type=float,
        default=0.5,
        help="minimum duration (QL) to enforce (default: 0.5 for short chord preservation)",
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
            f'{e.get("time_ql", 0.0):>7.1f} ql -> {e.get("symbol", ""):<12s} (dur={e.get("duration_ql", 0.0):.1f} ql)'
            for e in normalized_events
        ]
        pathlib.Path(args.out_preview).write_text("\n".join(lines), encoding="utf-8")

    # Normalized JSON
    if args.out_json:
        # Include bar field if present in original events
        out = {"unit": "ql", "events": []}
        for i, e in enumerate(normalized_events):
            ev_out = {
                "time": e.get("time_ql", 0.0),  # Use time_ql as primary time
                "root": e.get("root", ""),
                "quality": e.get("quality", ""),
                "symbol": e.get("symbol", ""),
                "duration_ql": e.get("duration_ql", 0.0),
            }
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
