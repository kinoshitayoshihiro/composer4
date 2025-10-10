#!/usr/bin/env python3
"""A/B summarize for Stage2 runs with stratified analysis.

Compares two Stage2 runs and generates a Markdown report with:
- Overall pass rate, score distribution
- Stratified analysis by BPM band, confidence level, preset applied
- Audio adaptive statistics
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


def _iter_jsonl(path_or_dir: str | Path) -> Iterable[Dict[str, Any]]:
    """Iterate JSONL files from path or directory."""
    path_or_dir = Path(path_or_dir)
    paths = []
    if path_or_dir.is_dir():
        paths = list(path_or_dir.glob("*.jsonl"))
    else:
        paths = [path_or_dir]

    for p in paths:
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)


def _bpm_bin(bpm: float | None) -> str:
    """Bin BPM into categories."""
    if bpm is None:
        return "unknown"
    if bpm <= 95:
        return "≤95"
    if bpm <= 130:
        return "≤130"
    return ">130"


def _conf_bin(c: float | None) -> str:
    """Bin confidence into categories."""
    if c is None:
        return "unknown"
    if c < 0.5:
        return "<0.5"
    if c < 0.7:
        return "0.5–0.7"
    if c < 0.85:
        return "0.7–0.85"
    return "≥0.85"


def _summary(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute summary statistics for a list of metrics."""
    scores = [float(r.get("score", 0.0)) for r in rows]
    axes = [r.get("axes_raw", {}) for r in rows]

    def pct(x: List[float], q: float) -> float:
        if not x:
            return float("nan")
        s = sorted(x)
        i = max(0, min(len(s) - 1, int(q * (len(s) - 1))))
        return s[i]

    return {
        "n": len(rows),
        "pass_rate": (sum(1 for r in rows if float(r.get("score", 0)) >= 50.0) / max(1, len(rows))),
        "p50": pct(scores, 0.50),
        "p75": pct(scores, 0.75),
        "p90": pct(scores, 0.90),
        "vel_med": pct([float(a.get("velocity", 0)) for a in axes], 0.5),
        "str_med": pct([float(a.get("structure", 0)) for a in axes], 0.5),
    }


def _collect(path: str | Path) -> List[Dict[str, Any]]:
    """Collect all metrics from path."""
    return list(_iter_jsonl(path))


def _strata_key(
    r: Dict[str, Any],
    names: List[str],
) -> Tuple:
    """Generate stratification key from row based on strata names."""
    k = []
    for n in names:
        if n == "bpm_bin":
            bpm_val = float(r.get("bpm", 0) or r.get("tempo", 0) or 0)
            k.append(_bpm_bin(bpm_val))
        elif n == "audio.min_confidence_bin":
            audio = r.get("audio", {})
            if audio:
                k.append(_conf_bin(float(audio.get("min_confidence", 0))))
            else:
                k.append(_conf_bin(None))
        elif n == "preset_applied":
            st = r.get("_retry_state") or {}
            k.append(st.get("last_preset") or "none")
        else:
            k.append(str(r.get(n)))
    return tuple(k)


def _markdown_table(
    rows: List[Tuple],
    headers: List[str],
) -> str:
    """Generate Markdown table from rows and headers."""
    out = []
    out.append("| " + " | ".join(headers) + " |")
    out.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for r in rows:
        out.append("| " + " | ".join(str(x) for x in r) + " |")
    return "\n".join(out)


def main() -> int:
    """Main entry point."""
    ap = argparse.ArgumentParser(description="A/B summarize for Stage2 runs with strata")
    ap.add_argument(
        "--a",
        required=True,
        help="Dir or metrics_score.jsonl (A)",
    )
    ap.add_argument(
        "--b",
        required=True,
        help="Dir or metrics_score.jsonl (B)",
    )
    ap.add_argument(
        "--strata",
        nargs="*",
        default=["bpm_bin", "audio.min_confidence_bin", "preset_applied"],
        help="Stratification keys (default: bpm_bin, confidence, preset)",
    )
    ap.add_argument(
        "--out",
        required=True,
        help="Output Markdown file path",
    )
    args = ap.parse_args()

    path_a = Path(args.a)
    path_b = Path(args.b)

    if not path_a.exists():
        print(f"ERROR: Path A not found: {path_a}", file=sys.stderr)
        return 1
    if not path_b.exists():
        print(f"ERROR: Path B not found: {path_b}", file=sys.stderr)
        return 1

    A = _collect(path_a)
    B = _collect(path_b)
    overallA = _summary(A)
    overallB = _summary(B)

    lines = []
    lines.append("# A/B Summary\n")
    lines.append(f"**A**: `{args.a}` / **B**: `{args.b}`\n")
    lines.append("## Overall")
    lines.append(
        _markdown_table(
            [
                (
                    "A",
                    overallA["n"],
                    f"{overallA['pass_rate']:.3f}",
                    f"{overallA['p50']:.2f}",
                    f"{overallA['p75']:.2f}",
                    f"{overallA['p90']:.2f}",
                ),
                (
                    "B",
                    overallB["n"],
                    f"{overallB['pass_rate']:.3f}",
                    f"{overallB['p50']:.2f}",
                    f"{overallB['p75']:.2f}",
                    f"{overallB['p90']:.2f}",
                ),
            ],
            ["Run", "N", "PassRate(>=50)", "p50", "p75", "p90"],
        )
    )
    lines.append("")

    # Stratified analysis
    lines.append("## Stratified Analysis")
    grpA: Dict[Tuple, List[Dict[str, Any]]] = defaultdict(list)
    grpB: Dict[Tuple, List[Dict[str, Any]]] = defaultdict(list)

    for r in A:
        grpA[_strata_key(r, args.strata)].append(r)
    for r in B:
        grpB[_strata_key(r, args.strata)].append(r)

    rows = []
    keys = sorted(set(list(grpA.keys()) + list(grpB.keys())))
    for k in keys:
        sA = _summary(grpA.get(k, []))
        sB = _summary(grpB.get(k, []))
        rows.append(
            (
                " / ".join(k) if isinstance(k, tuple) else k,
                sA["n"],
                f"{sA['pass_rate']:.3f}",
                f"{sA['p50']:.2f}",
                sB["n"],
                f"{sB['pass_rate']:.3f}",
                f"{sB['p50']:.2f}",
            )
        )

    lines.append(
        _markdown_table(
            rows,
            ["Strata", "A.N", "A.Pass", "A.p50", "B.N", "B.Pass", "B.p50"],
        )
    )

    out_path = Path(args.out)
    with out_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
