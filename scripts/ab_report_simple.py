#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Read two batch-eval JSONs and produce a Markdown A/B report.
- Input: --eval-a <A.json> --eval-b <B.json>
- Output: --out-md <report.md>
- Also prints acceptance check result to stdout and returns nonzero exit on failure (optional).
"""

from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

METRICS = [
    ("hat_grid_conform", "Hi-hat Grid Conform ↑", True),
    ("snare_backbeat_rate", "Snare Backbeat Rate ↑", True),
    ("kick_downbeat_rate", "Kick Downbeat Rate ↑", True),
    ("velocity_std", "Velocity Std ↑", True),
    ("notes_per_bar", "Notes per Bar (info)", None),
    ("bar_violation_rate", "Bar Violation ↓", False),
]

# Acceptance defaults (can be tweaked)
DEFAULT_THRESHOLDS = {
    "bar_violation_rate_max": 0.0,
    "hat_grid_conform_min": 0.85,
    "snare_backbeat_rate_min": 0.80,
    "kick_downbeat_rate_min": 0.90,
    "velocity_std_min": 8.0,
}


def load(path: str):
    obj = json.loads(Path(path).read_text(encoding="utf-8"))
    return obj["summary"], obj.get("files", [])


def emoji_win(delta: float, higher_is_better: bool | None):
    if higher_is_better is None:
        return "•"
    if (delta > 0 and higher_is_better) or (delta < 0 and not higher_is_better):
        return "✅"
    elif delta == 0:
        return "➖"
    else:
        return "❌"


def check_acceptance(s, thr):
    ok = True
    reasons = []
    if s["bar_violation_rate"] > thr["bar_violation_rate_max"]:
        ok = False
        reasons.append(
            f"bar_violation_rate {s['bar_violation_rate']} > {thr['bar_violation_rate_max']}"
        )
    if s["hat_grid_conform"] < thr["hat_grid_conform_min"]:
        ok = False
        reasons.append(
            f"hat_grid_conform {s['hat_grid_conform']} < {thr['hat_grid_conform_min']}"
        )
    if s["snare_backbeat_rate"] < thr["snare_backbeat_rate_min"]:
        ok = False
        reasons.append(
            f"snare_backbeat_rate {s['snare_backbeat_rate']} < {thr['snare_backbeat_rate_min']}"
        )
    if s["kick_downbeat_rate"] < thr["kick_downbeat_rate_min"]:
        ok = False
        reasons.append(
            f"kick_downbeat_rate {s['kick_downbeat_rate']} < {thr['kick_downbeat_rate_min']}"
        )
    if s["velocity_std"] < thr["velocity_std_min"]:
        ok = False
        reasons.append(
            f"velocity_std {s['velocity_std']} < {thr['velocity_std_min']}"
        )
    return ok, reasons


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-a", required=True)
    ap.add_argument("--eval-b", required=True)
    ap.add_argument("--out-md", required=True)
    ap.add_argument("--name-a", default="Baseline (A)")
    ap.add_argument("--name-b", default="Candidate (B)")
    ap.add_argument("--strict-exit", action="store_true",
                    help="Exit 1 if B fails acceptance")
    args = ap.parse_args()

    sA, filesA = load(args.eval_a)
    sB, filesB = load(args.eval_b)

    lines = []
    lines.append("# Drum A/B Report")
    lines.append("")
    lines.append(f"- A: **{args.name_a}** — n={sA['count']}")
    lines.append(f"- B: **{args.name_b}** — n={sB['count']}")
    lines.append("")

    # Table
    lines.append("| Metric | A | B | Δ(B−A) | Note |")
    lines.append("|---|---:|---:|---:|:--|")
    for key, label, hib in METRICS:
        a = sA.get(key, 0.0)
        b = sB.get(key, 0.0)
        delta = round(b - a, 4)
        lines.append(
            f"| {label} | {a:.4f} | {b:.4f} | {delta:+.4f} | {emoji_win(delta, hib)} |"
        )

    # Acceptance check for B
    ok, reasons = check_acceptance(sB, DEFAULT_THRESHOLDS)
    lines.append("")
    lines.append("## Acceptance")
    if ok:
        lines.append("**✅ PASS** — thresholds satisfied.")
    else:
        lines.append("**❌ FAIL** — thresholds not met:")
        for r in reasons:
            lines.append(f"- {r}")

    Path(args.out_md).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_md).write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))

    if args.strict_exit and not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
