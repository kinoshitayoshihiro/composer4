#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plan_doctor.py — Plan正規化ユーティリティ（dur/dur_beats補完・負値クリップ・メタ付与）

使い方:
  python3 scripts/plan_doctor.py \
    --in-plan  data/.../piano_plan.json \
    --out-plan data/.../piano_plan.doctored.json \
    --min-dur-beats 0.03125  # 1/32拍相当
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

SchemaVersion = "v1.2"


def _load_json(p: Path) -> Any:
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def _save_json(p: Path, obj: Any) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _iter_events(plan: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """events配列へのビューを返す。
    - tracks[].events があればそれを、なければ top-level events を対象にする（最初のトラックのみを想定）。
    """
    if isinstance(plan.get("tracks"), list) and plan["tracks"]:
        tr0 = plan["tracks"][0]
        evs = tr0.get("events") or []
        return evs, tr0
    evs = plan.get("events") or []
    return evs, None


def _as_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _fix_event(ev: Dict[str, Any], min_dur_beats: float) -> Dict[str, Any]:
    """1イベントの正規化。dur_beats優先で補完。start_beatsもできるだけ揃える。"""
    # start_beats 救済
    if "start_beats" not in ev:
        # 互換キー候補
        for k in ("start", "offset_beats", "offset"):
            if k in ev:
                ev["start_beats"] = _as_float(ev[k])
                break

    # end_beats があるなら差分を候補に
    cand_diff = None
    if "end_beats" in ev and "start_beats" in ev:
        cand_diff = _as_float(ev["end_beats"]) - _as_float(ev["start_beats"])

    # dur_beats 決定
    if "dur_beats" in ev and _as_float(ev["dur_beats"]) > 0:
        pass  # そのまま
    elif "dur" in ev and _as_float(ev["dur"]) > 0:
        ev["dur_beats"] = _as_float(ev["dur"])
    elif cand_diff is not None:
        ev["dur_beats"] = cand_diff
    else:
        # どうしても無い場合の最小長
        ev["dur_beats"] = min_dur_beats

    # クリップ（負値→最小長、ゼロ→最小長）
    if _as_float(ev["dur_beats"]) <= 0:
        ev["dur_beats"] = min_dur_beats
    if "start_beats" in ev and _as_float(ev["start_beats"]) < 0:
        ev["start_beats"] = max(0.0, _as_float(ev["start_beats"]))

    return ev


def main():
    ap = argparse.ArgumentParser(description="Normalize/repair instrument plan JSON")
    ap.add_argument("--in-plan", required=True, type=Path)
    ap.add_argument("--out-plan", type=Path, default=None)
    ap.add_argument(
        "--bars", type=Path, default=None, help="Optional bars.parquet for boundary reference"
    )
    ap.add_argument("--min-dur-beats", type=float, default=0.03125, help="下限（既定1/32拍）")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    plan = _load_json(args.in_plan)
    events, track_ref = _iter_events(plan)

    before_n = len(events)
    fixed_cnt = zero_or_neg_before = 0
    inserted_dur = inserted_start = 0

    for ev in events:
        # 事前計測
        if _as_float(ev.get("dur_beats", ev.get("dur", 0.0))) <= 0:
            zero_or_neg_before += 1
        if "dur_beats" not in ev and "dur" in ev:
            inserted_dur += 1
        if "start_beats" not in ev:
            inserted_start += 1

        _fix_event(ev, args.min_dur_beats)
        fixed_cnt += 1

    # メタ／スキーマ
    meta = plan.get("meta") or {}
    prov = meta.get("provenance") or meta.get("context_sources") or {}
    prov["plan_doctor"] = {
        "enabled": True,
        "schema_version": SchemaVersion,
        "min_dur_beats": args.min_dur_beats,
        "timestamp": int(time.time()),
    }
    # 保存先
    meta["provenance"] = prov
    plan["meta"] = meta
    plan["plan_schema"] = SchemaVersion

    # 出力
    out_path = args.out_plan or args.in_plan
    if args.dry_run:
        print(
            f"[DRY-RUN] events={before_n}, fixed={fixed_cnt}, "
            f"zero_or_neg_before={zero_or_neg_before}, "
            f"inserted_dur={inserted_dur}, inserted_start={inserted_start}"
        )
        sys.exit(0)
    _save_json(out_path, plan)
    print(f"[OK] doctored: {out_path}")
    print(
        f"  events={before_n}, fixed={fixed_cnt}, zero/neg_before={zero_or_neg_before}, "
        f"ins_dur={inserted_dur}, ins_start={inserted_start}"
    )


if __name__ == "__main__":
    main()
