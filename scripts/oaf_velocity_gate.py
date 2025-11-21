#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
oaf_velocity_gate.py — OaF適用後のPianoダイナミクス整形（平滑＋差分制限＋上下限）

使い方:
  python3 scripts/oaf_velocity_gate.py \
    --plan-in  data/.../piano_plan.doctored.json \
    --out-plan data/.../piano_plan.gated.json \
    --role piano --window-notes 5 --delta-max 18 --min-vel 30 --max-vel 112
"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List


def _load_json(p: Path) -> Any:
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def _save_json(p: Path, obj: Any) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _iter_events_with_role(plan: Dict[str, Any], wanted: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if isinstance(plan.get("tracks"), list) and plan["tracks"]:
        for tr in plan["tracks"]:
            role = (tr.get("role") or tr.get("name") or "").lower()
            evs = tr.get("events") or []
            if role == wanted:
                out.extend(evs)
    else:
        evs = plan.get("events") or []
        # event単体にroleがあるケースにも対応
        for ev in evs:
            r = (ev.get("role") or "").lower()
            if r == wanted or wanted == "any":
                out.append(ev)
    return out


def _clamp(x: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, x))


def _median(xs: List[int]) -> int:
    if not xs:
        return 0
    s = sorted(xs)
    n = len(s)
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) // 2


def _running_median(vec: List[int], win: int) -> List[int]:
    if win <= 1:
        return vec[:]
    half = win // 2
    out = []
    for i in range(len(vec)):
        lo = max(0, i - half)
        hi = min(len(vec), i + half + 1)
        out.append(_median(vec[lo:hi]))
    return out


def main():
    ap = argparse.ArgumentParser(description="Post OaF velocity gating for piano")
    ap.add_argument("--plan-in", required=True, type=Path)
    ap.add_argument("--out-plan", required=True, type=Path)
    ap.add_argument("--role", default="piano")
    ap.add_argument("--window-notes", type=int, default=5, help="移動中央値の窓幅（ノート単位）")
    ap.add_argument("--delta-max", type=int, default=18, help="隣接ノート間の最大差分")
    ap.add_argument("--min-vel", type=int, default=30)
    ap.add_argument("--max-vel", type=int, default=112)
    args = ap.parse_args()

    plan = _load_json(args.plan_in)
    evs = _iter_events_with_role(plan, args.role.lower())

    # velocity列を抽出（無ければ127相当とみなす）
    v = [int(ev.get("velocity", ev.get("vel", 64))) for ev in evs]
    if not v:
        print("[WARN] target role has no events — skipped")
        _save_json(args.out_plan, plan)
        return

    # (1) 移動中央値で平滑
    v_smooth = _running_median(v, max(1, args.window_notes))

    # (2) 差分クリップ（過大ジャンプ抑制）
    v_delta = [v_smooth[0]]
    for i in range(1, len(v_smooth)):
        prev = v_delta[-1]
        # up/dnの順で差分制限
        candidate = v_smooth[i]
        if candidate > prev:
            clamped = min(prev + args.delta_max, candidate)
        else:
            clamped = max(prev - args.delta_max, candidate)
        v_delta.append(_clamp(clamped, args.min_vel, args.max_vel))

    # (3) 上下限クリップ
    v_final = [_clamp(x, args.min_vel, args.max_vel) for x in v_delta]

    # 書き戻し
    for ev, newv in zip(evs, v_final):
        ev["velocity"] = int(newv)

    # provenance
    meta = plan.get("meta") or {}
    prov = meta.get("provenance") or {}
    prov["oaf_velocity_gate"] = {
        "enabled": True,
        "window_notes": args.window_notes,
        "delta_max": args.delta_max,
        "min_vel": args.min_vel,
        "max_vel": args.max_vel,
        "timestamp": int(time.time()),
    }
    meta["provenance"] = prov
    plan["meta"] = meta

    _save_json(args.out_plan, plan)
    print(f"[OK] gated -> {args.out_plan} (n={len(v_final)})")


if __name__ == "__main__":
    main()
