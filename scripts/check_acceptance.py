#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Threshold-only acceptance checker for a single eval JSON (eval_drum_batch.py output).
Use when baseline isn't available.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

DEFAULT_THRESHOLDS = {
    "bar_violation_rate_max": 0.0,
    "hat_grid_conform_min": 0.85,
    "snare_backbeat_rate_min": 0.80,
    "kick_downbeat_rate_min": 0.90,
    "velocity_std_min": 8.0,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-json", required=True)
    ap.add_argument("--strict-exit", action="store_true")
    # スタイル別に緩める場合は追加: --style shuffle で hat_grid を 0.80 に下げる等
    ap.add_argument("--style", default="")
    args = ap.parse_args()

    s = json.loads(Path(args.eval_json).read_text(encoding="utf-8"))["summary"]
    thr = dict(DEFAULT_THRESHOLDS)

    if args.style == "shuffle":
        thr["hat_grid_conform_min"] = 0.80  # shuffleはゆるめ

    ok = True
    reasons = []
    if s["bar_violation_rate"] > thr["bar_violation_rate_max"]:
        ok = False
        reasons.append(f"bar_violation_rate {s['bar_violation_rate']} > {thr['bar_violation_rate_max']}")
    if s["hat_grid_conform"] < thr["hat_grid_conform_min"]:
        ok = False
        reasons.append(f"hat_grid_conform {s['hat_grid_conform']} < {thr['hat_grid_conform_min']}")
    if s["snare_backbeat_rate"] < thr["snare_backbeat_rate_min"]:
        ok = False
        reasons.append(f"snare_backbeat_rate {s['snare_backbeat_rate']} < {thr['snare_backbeat_rate_min']}")
    if s["kick_downbeat_rate"] < thr["kick_downbeat_rate_min"]:
        ok = False
        reasons.append(f"kick_downbeat_rate {s['kick_downbeat_rate']} < {thr['kick_downbeat_rate_min']}")
    if s["velocity_std"] < thr["velocity_std_min"]:
        ok = False
        reasons.append(f"velocity_std {s['velocity_std']} < {thr['velocity_std_min']}")

    print(f"[ACCEPTANCE] summary={s}")
    if ok:
        print("✅ PASS — thresholds satisfied.")
    else:
        print("❌ FAIL — thresholds not met:")
        for r in reasons:
            print(" -", r)
        if args.strict_exit:
            sys.exit(1)


if __name__ == "__main__":
    main()
