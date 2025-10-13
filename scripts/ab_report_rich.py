#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Rich A/B report:
- Markdown tables (overall + per-stratum)
- Simple plots (bar charts) saved as PNGs
- Optional strict exit if B fails thresholds

NOTE: Do not specify colors; single-plot per figure; matplotlib only.
"""

from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

METRICS = [
    ("hat_grid_conform", "Hi-hat Grid Conform ↑", True),
    ("snare_backbeat_rate", "Snare Backbeat Rate ↑", True),
    ("kick_downbeat_rate", "Kick Downbeat Rate ↑", True),
    ("velocity_std", "Velocity Std ↑", True),
    ("notes_per_bar", "Notes/Bar (info)", None),
    ("bar_violation_rate", "Bar Violation ↓", False),
    ("crash_on_bar1_rate", "Crash on Bar1 ↑", True),
    ("fill_coverage_rate", "Fill Coverage ↑", True),
]

THR_DEFAULT = {
    "bar_violation_rate_max": 0.0,
    "hat_grid_conform_min": 0.85,
    "snare_backbeat_rate_min": 0.80,
    "kick_downbeat_rate_min": 0.90,
    "velocity_std_min": 8.0,
}


def check_accept(summary, thr):
    ok = True
    reasons = []
    if summary["bar_violation_rate"] > thr["bar_violation_rate_max"]:
        ok = False
        reasons.append(f"bar_violation_rate {summary['bar_violation_rate']} > {thr['bar_violation_rate_max']}")
    if summary["hat_grid_conform"] < thr["hat_grid_conform_min"]:
        ok = False
        reasons.append(f"hat_grid_conform {summary['hat_grid_conform']} < {thr['hat_grid_conform_min']}")
    if summary["snare_backbeat_rate"] < thr["snare_backbeat_rate_min"]:
        ok = False
        reasons.append(f"snare_backbeat_rate {summary['snare_backbeat_rate']} < {thr['snare_backbeat_rate_min']}")
    if summary["kick_downbeat_rate"] < thr["kick_downbeat_rate_min"]:
        ok = False
        reasons.append(f"kick_downbeat_rate {summary['kick_downbeat_rate']} < {thr['kick_downbeat_rate_min']}")
    if summary["velocity_std"] < thr["velocity_std_min"]:
        ok = False
        reasons.append(f"velocity_std {summary['velocity_std']} < {thr['velocity_std_min']}")
    return ok, reasons


def emoji(delta, hib):
    if hib is None:
        return "•"
    if (delta > 0 and hib) or (delta < 0 and not hib):
        return "✅"
    if delta == 0:
        return "➖"
    return "❌"


def barplot(save_to, title, labels, A_vals, B_vals):
    # Single-plot figure, no explicit colors
    plt.figure()
    x = range(len(labels))
    width = 0.4
    plt.bar([i - width / 2 for i in x], A_vals, width, label="A")
    plt.bar([i + width / 2 for i in x], B_vals, width, label="B")
    plt.xticks(list(x), labels, rotation=45, ha="right")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_to)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ab-json", required=True, help="eval_drum_batch_stratified.py output")
    ap.add_argument("--out-md", required=True)
    ap.add_argument("--plot-dir", default="output/reports/plots")
    ap.add_argument("--strict-exit", action="store_true")
    args = ap.parse_args()

    data = json.loads(Path(args.ab_json).read_text(encoding="utf-8"))
    overallA = data["overall"]["A"]
    overallB = data["overall"]["B"]
    strata = data["strata"]

    lines = []
    lines.append("# Drum A/B Report (Stratified)")
    lines.append("")
    lines.append(f"- n(A)={overallA.get('count', 0)}, n(B)={overallB.get('count', 0)}")
    lines.append("")

    # Overall table
    lines.append("## Overall")
    lines.append("| Metric | A | B | Δ(B−A) | Note |")
    lines.append("|---|---:|---:|---:|:--|")
    for key, label, hib in METRICS:
        a = overallA.get(key, 0.0)
        b = overallB.get(key, 0.0)
        d = round(b - a, 4)
        lines.append(f"| {label} | {a:.4f} | {b:.4f} | {d:+.4f} | {emoji(d, hib)} |")
    lines.append("")

    # Per-stratum tables + plots
    plot_dir = Path(args.plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)
    lines.append("## Stratified")
    for tag in sorted(strata.keys()):
        sA = strata[tag]["A"]["summary"]
        sB = strata[tag]["B"]["summary"]
        lines.append(f"### {tag}")
        lines.append("| Metric | A | B | Δ(B−A) | Note |")
        lines.append("|---|---:|---:|---:|:--|")
        labels = []
        Avals = []
        Bvals = []
        for key, label, hib in METRICS:
            a = sA.get(key, 0.0)
            b = sB.get(key, 0.0)
            d = round(b - a, 4)
            lines.append(f"| {label} | {a:.4f} | {b:.4f} | {d:+.4f} | {emoji(d, hib)} |")
            labels.append(label)
            Avals.append(a)
            Bvals.append(b)
        # plot
        img = plot_dir / f"{tag.replace('/', '_')}_bars.png"
        barplot(str(img), f"{tag} — A/B", labels, Avals, Bvals)
        lines.append(f"![]({img})")
        lines.append("")

    # Acceptance (overall B)
    ok, reasons = check_accept(overallB, dict(THR_DEFAULT))
    lines.append("## Acceptance (overall B)")
    if ok:
        lines.append("**✅ PASS**")
    else:
        lines.append("**❌ FAIL**")
        for r in reasons:
            lines.append(f"- {r}")

    Path(args.out_md).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_md).write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))

    if args.strict_exit and not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
