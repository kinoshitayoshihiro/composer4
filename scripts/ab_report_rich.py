#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Rich A/B report with instrument auto-detection:
- Markdown tables (overall + per-stratum)
- Simple plots (bar charts) saved as PNGs
- Optional strict exit if B fails thresholds
- --instrument flag for explicit column selection

NOTE: Do not specify colors; single-plot per figure; matplotlib only.
"""

from __future__ import annotations
import argparse
import json
import math
import sys
from pathlib import Path
from collections import defaultdict, Counter
import statistics

import matplotlib.pyplot as plt

# Metric columns per instrument (keys not in JSON are auto-skipped)
METRIC_COLUMNS = {
    "drum": ["hat_grid_conform", "snare_backbeat_rate", "kick_downbeat_rate", "bar_violation_rate", "velocity_std", "notes_per_bar"],
    "bass": ["downbeat_anchor_rate", "range_ok_rate", "velocity_std", "kick_align_rate", "bar_violation_rate", "notes_per_bar"],
    "piano": ["chord_tone_rate", "hand_separation", "velocity_std", "bar_violation_rate", "notes_per_bar"],
    "guitar": ["strum_consistency", "chord_tone_rate", "range_ok_rate", "velocity_std", "bar_violation_rate", "notes_per_bar"],
    "strings": ["legato_ratio", "sustain_stability", "range_ok_rate", "velocity_std", "bar_violation_rate", "notes_per_bar"],
}

# Acceptance thresholds per instrument (keys not in JSON are skipped)
THRESHOLDS = {
    "drum": {
        "hat_grid_conform_min": 0.95,
        "snare_backbeat_rate_min": 0.60,
        "kick_downbeat_rate_min": 0.60,
        "bar_violation_rate_max": 0.0,
        "velocity_std_min": 6.5,
    },
    "bass": {
        "downbeat_anchor_rate_min": 0.85,
        "range_ok_rate_min": 0.95,
        "velocity_std_min": 7.0,
        "kick_align_rate_min": 0.55,
        "bar_violation_rate_max": 0.0,
    },
    "piano": {
        "chord_tone_rate_min": 0.70,
        "hand_separation_min": 0.60,  # 0.85→0.60: block style同時発音を許容
        "velocity_std_min": 7.0,      # 8.0→7.0: template engineの自然なばらつき
        "bar_violation_rate_max": 0.0,
    },
    "guitar": {
        "strum_consistency_min": 0.75,
        "chord_tone_rate_min": 0.70,
        "range_ok_rate_min": 0.90,
        "velocity_std_min": 7.0,
        "bar_violation_rate_max": 0.0,
    },
    "strings": {
        "legato_ratio_min": 0.60,
        "sustain_stability_min": 0.80,
        "range_ok_rate_min": 0.95,
        "velocity_std_min": 5.0,
        "bar_violation_rate_max": 0.0,
    },
}


def _select_columns(instrument: str, available_keys):
    """Select metric columns for instrument, filtering out missing keys."""
    cols = METRIC_COLUMNS.get(instrument, METRIC_COLUMNS["drum"])
    return [c for c in cols if c in available_keys]


def _judge_accept(overall: dict, instrument: str) -> tuple:
    """Judge acceptance based on instrument-specific thresholds."""
    thr = THRESHOLDS.get(instrument, {})
    fails = []
    
    # *_min thresholds
    for k, v in thr.items():
        if k.endswith("_min"):
            metric = k[:-4]
            if metric in overall and overall[metric] is not None:
                if overall[metric] < v:
                    fails.append(f"{metric} {overall[metric]:.3f} < {v}")
    
    # *_max thresholds
    for k, v in thr.items():
        if k.endswith("_max"):
            metric = k[:-4]
            if metric in overall and overall[metric] is not None:
                if overall[metric] > v:
                    fails.append(f"{metric} {overall[metric]:.3f} > {v}")
    
    return (len(fails) == 0, fails)


def emoji(delta):
    """Simple delta emoji (no higher_is_better logic)."""
    if abs(delta) < 0.0001:
        return "➖"
    return "📊"


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
    ap.add_argument("--plot-dir", default="")
    ap.add_argument("--title", default="A/B Report")
    ap.add_argument("--strict-exit", action="store_true")
    ap.add_argument("--instrument", default="drum",
                    choices=["drum", "bass", "piano", "guitar", "strings"],
                    help="Metric columns and acceptance criteria auto-switch")
    args = ap.parse_args()

    data = json.loads(Path(args.ab_json).read_text("utf-8"))
    overallA = data["overall"]["A"]
    overallB = data["overall"]["B"]
    strata = data.get("strata", {})
    counts = data.get("counts", {})

    all_keys = set(overallA.keys()) | set(overallB.keys())
    columns = _select_columns(args.instrument, all_keys)
    if not columns:
        columns = sorted([k for k in all_keys if k not in ("count", "tempo", "bars", "time_sig")])

    title = f"{args.title} [{args.instrument}]"
    md = []
    md.append(f"# {title}")
    md.append("")
    md.append(f"- n(A)={counts.get('A', overallA.get('count', 0))}, n(B)={counts.get('B', overallB.get('count', 0))}")
    md.append("")

    # Acceptance check (A and B both)
    if args.strict_exit:
        okA, failsA = _judge_accept(overallA, args.instrument)
        okB, failsB = _judge_accept(overallB, args.instrument)
        if okA and okB:
            md.append("\n**✅ Acceptance: PASS**")
        else:
            msg = []
            if not okA:
                msg.append("A: " + "; ".join(failsA))
            if not okB:
                msg.append("B: " + "; ".join(failsB))
            md.append("\n**❌ Acceptance: FAIL**  \n" + "<br/>".join(msg))
            Path(args.out_md).write_text("\n".join(md), encoding="utf-8")
            raise SystemExit(1)

    # Overall table
    md.append("## Overall")
    md.append("| Metric | A | B | Δ(B−A) |")
    md.append("|---|---:|---:|---:|")
    for key in columns:
        a = overallA.get(key, 0.0)
        b = overallB.get(key, 0.0)
        if a is None:
            a = 0.0
        if b is None:
            b = 0.0
        d = round(b - a, 4)
        md.append(f"| {key} | {a:.4f} | {b:.4f} | {d:+.4f} {emoji(d)} |")
    md.append("")

    # Per-stratum tables + plots
    md.append("## Stratified")
    for tag in sorted(strata.keys()):
        sA = strata[tag]["A"]["summary"]
        sB = strata[tag]["B"]["summary"]
        md.append(f"### {tag}")
        md.append("| Metric | A | B | Δ(B−A) |")
        md.append("|---|---:|---:|---:|")
        
        labels = []
        Avals = []
        Bvals = []
        for key in columns:
            a = sA.get(key, 0.0)
            b = sB.get(key, 0.0)
            if a is None:
                a = 0.0
            if b is None:
                b = 0.0
            d = round(b - a, 4)
            md.append(f"| {key} | {a:.4f} | {b:.4f} | {d:+.4f} {emoji(d)} |")
            labels.append(key)
            Avals.append(a)
            Bvals.append(b)
        
        # Plot (if plot_dir specified)
        if args.plot_dir:
            plot_dir = Path(args.plot_dir)
            plot_dir.mkdir(parents=True, exist_ok=True)
            img = plot_dir / f"{tag.replace('/', '_')}_bars.png"
            barplot(str(img), f"{tag} — A/B", labels, Avals, Bvals)
            md.append(f"![]({img})")
        md.append("")

    Path(args.out_md).write_text("\n".join(md), encoding="utf-8")
    print(f"Wrote {args.out_md}")


if __name__ == "__main__":
    main()
