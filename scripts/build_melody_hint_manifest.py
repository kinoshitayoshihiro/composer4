#!/usr/bin/env python3
"""Build a lightweight manifest describing CREPE-derived melody hints per bar."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import pandas as pd

from melody_hint_utils import build_melody_hint_manifest_payload, build_melody_hint_table


def load_bars(path: Path) -> pd.DataFrame:
    bars = pd.read_parquet(path)
    if "bar_idx" not in bars.columns:
        if "bar_index" in bars.columns:
            bars = bars.rename(columns={"bar_index": "bar_idx"})
        else:
            bars["bar_idx"] = range(len(bars))
    if "section_label" not in bars.columns:
        bars["section_label"] = "unknown"
    return bars


def load_vocal_f0(path: Optional[Path]) -> Optional[pd.DataFrame]:
    if path is None:
        return None
    if not path.exists():
        raise FileNotFoundError(f"vocal_f0 file not found: {path}")
    return pd.read_parquet(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build melody_hint manifest from CREPE data")
    parser.add_argument("--bars", required=True, type=Path, help="Path to bars_with_slots.parquet")
    parser.add_argument(
        "--vocal-f0", required=True, type=Path, help="Path to vocal_f0_crepe.parquet"
    )
    parser.add_argument("--out", required=True, type=Path, help="Output JSON manifest path")
    parser.add_argument("--song-id", help="Optional song identifier for metadata")
    args = parser.parse_args()

    bars = load_bars(args.bars)
    vocal_f0 = load_vocal_f0(args.vocal_f0)

    hints = build_melody_hint_table(bars, vocal_f0)

    manifest = build_melody_hint_manifest_payload(
        hints,
        bars_total=len(bars),
        song_id=args.song_id or Path(args.bars).parent.name,
        bars_path=args.bars,
        vocal_f0_path=args.vocal_f0,
        out_path=args.out,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Melody hint manifest saved to {args.out} (hints={len(hints)})")


if __name__ == "__main__":
    main()
