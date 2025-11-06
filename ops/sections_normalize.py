#!/usr/bin/env python3
# ops/sections_normalize.py
import json, argparse, datetime, sys
from pathlib import Path


def load_json(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def infer_total_bars(bars_parquet):
    try:
        import pandas as pd

        df = pd.read_parquet(bars_parquet)
        if "bar_index" in df.columns:
            return int(df["bar_index"].max()) + 1, df
    except Exception:
        pass
    return None, None


def to_spans(items, total_bars):
    """items: either [{bar,label}] or [{start_bar,end_bar,label}]"""
    if not items:
        return []
    # already spans?
    if "start_bar" in items[0] and "end_bar" in items[0]:
        return sorted(items, key=lambda x: x["start_bar"])
    # pointer list → spans
    bps = sorted([(int(x["bar"]), x["label"]) for x in items], key=lambda x: x[0])
    spans = []
    for i, (start, lab) in enumerate(bps):
        end = (bps[i + 1][0] - 1) if i + 1 < len(bps) else (total_bars - 1 if total_bars else start)
        end = max(end, start)
        spans.append({"start_bar": int(start), "end_bar": int(end), "label": lab})
    return spans


def attach_seconds(spans, bars_df):
    if bars_df is None or not {"start_sec", "end_sec", "bar_index"} <= set(bars_df.columns):
        return spans
    idx = bars_df.set_index("bar_index")
    out = []
    for s in spans:
        a, b = s["start_bar"], s["end_bar"]
        s2 = dict(s)
        try:
            s2["start_sec"] = float(idx.loc[a, "start_sec"])
            s2["end_sec"] = float(idx.loc[b, "end_sec"])
        except Exception:
            pass
        out.append(s2)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", dest="out", required=True)
    ap.add_argument("--bars", dest="bars_parquet", default=None)
    ap.add_argument("--total-bars", type=int, default=None)
    args = ap.parse_args()

    raw = load_json(args.inp)
    items = raw.get("sections", raw) if isinstance(raw, dict) else raw

    total, bars_df = (
        infer_total_bars(args.bars_parquet) if args.bars_parquet else (args.total_bars, None)
    )
    total = total or args.total_bars
    spans = to_spans(items, total_bars=total)
    spans = attach_seconds(spans, bars_df)

    out = {
        "meta": {
            "schema": "sections/v2",
            "generated_at": datetime.datetime.utcnow().isoformat() + "Z",
            "source": Path(args.inp).name,
        },
        "sections": spans,
    }
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"✅ normalized → {args.out} (sections={len(spans)})")


if __name__ == "__main__":
    main()
