#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fix_plan_drum_channel.py
- full_arrangement.json のドラム系トラック/イベントを MIDI チャンネル10 (index=9) へ統一
- velocity/vel フィールドの同期と1..127クランプ
- GMドラム音域(35..81)での自動検出オプション

Usage:
  python3 scripts/fix_plan_drum_channel.py \
    --in song_packages/.../full_arrangement.json \
    --out song_packages/.../full_arrangement.json \
    --required-channel 9 \
    --detect-gm \
    --min-gm-ratio 0.6

戻り値: 0=成功
"""

from __future__ import annotations
import argparse, json, sys, re
from typing import Any, Dict, List, Tuple

GM_DRUM_PITCH_MIN = 35
GM_DRUM_PITCH_MAX = 81
NAME_DRUM_PAT = re.compile(r"(drum|percussion)", re.IGNORECASE)


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(d: Dict[str, Any], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(d, f, ensure_ascii=False, indent=2)


def is_drum_by_name_or_flag(track: Dict[str, Any]) -> bool:
    role = (track.get("role") or "").strip()
    name = (track.get("name") or "").strip()
    if track.get("is_drum") is True:
        return True
    if NAME_DRUM_PAT.search(role) or NAME_DRUM_PAT.search(name):
        return True
    return False


def gm_drum_ratio(track: Dict[str, Any]) -> float:
    evs = track.get("events") or []
    if not evs:
        return 0.0
    hits, total = 0, 0
    for ev in evs:
        if "pitch" in ev and isinstance(ev["pitch"], (int, float)):
            total += 1
            p = int(ev["pitch"])
            if GM_DRUM_PITCH_MIN <= p <= GM_DRUM_PITCH_MAX:
                hits += 1
    return (hits / total) if total else 0.0


def sync_velocity_fields(ev: Dict[str, Any], vmin: int, vmax: int) -> Tuple[bool, int]:
    """
    velocity/vel を相互に同期（存在する方をソースに）し、1..127へクランプ。
    変更があれば (changed=True, normalized_vel) を返す
    """
    changed = False
    vel_src = None
    if "vel" in ev:
        vel_src = ev["vel"]
    if vel_src is None and "velocity" in ev:
        vel_src = ev["velocity"]

    if vel_src is None:
        return changed, None  # 触らない（他の正規化ステージに任せる）

    try:
        vel_val = int(vel_src)
    except Exception:
        return changed, None

    if vel_val < vmin:
        vel_val = vmin
        changed = True
    if vel_val > vmax:
        vel_val = vmax
        changed = True

    # 双方向同期
    if ev.get("vel") != vel_val:
        ev["vel"] = vel_val
        changed = True
    if ev.get("velocity") != vel_val:
        ev["velocity"] = vel_val
        changed = True

    return changed, vel_val


def ensure_channel(ev: Dict[str, Any], required_ch: int) -> bool:
    """
    イベントの channel を required_ch へ。変更があれば True。
    """
    ch = ev.get("channel")
    if ch is None or int(ch) != int(required_ch):
        ev["channel"] = int(required_ch)
        return True
    return False


def fix_drum_track(
    track: Dict[str, Any], required_ch: int, sync_vel: bool, vmin: int, vmax: int
) -> Tuple[int, int]:
    """
    トラックレベルの channel と各イベントの channel/velocity を修正。
    戻り値: (changed_event_channels, normalized_vel_count)
    """
    changed_ev_ch = 0
    normalized_vel = 0

    # トラックレベル channel を合わせる（MIDI ch10=9）
    if track.get("channel") is None or int(track.get("channel")) != int(required_ch):
        track["channel"] = int(required_ch)
    # ドラム明示（将来の判定を安定化）
    track["is_drum"] = True

    # イベント側
    events = track.get("events") or []
    for ev in events:
        if ensure_channel(ev, required_ch):
            changed_ev_ch += 1
        if sync_vel:
            changed, _ = sync_velocity_fields(ev, vmin, vmax)
            if changed:
                normalized_vel += 1

    return changed_ev_ch, normalized_vel


def main():
    ap = argparse.ArgumentParser(
        description="Fix full_arrangement.json: drum tracks → MIDI ch10 (index 9), sync velocities."
    )
    ap.add_argument("--in", dest="inp", required=True, help="Input full_arrangement.json")
    ap.add_argument(
        "--out", dest="outp", default=None, help="Output path (default: overwrite input)"
    )
    ap.add_argument(
        "--required-channel",
        "--channel",
        dest="required_channel",
        type=int,
        default=9,
        help="Required channel for drums (0-15). Default=9 (MIDI ch10).",
    )
    ap.add_argument(
        "--detect-gm",
        action="store_true",
        help="Also detect drum tracks by GM drum pitch ratio (35..81).",
    )
    ap.add_argument(
        "--min-gm-ratio",
        type=float,
        default=0.6,
        help="Threshold of GM drum pitch ratio to classify as drums. Default=0.6",
    )
    ap.add_argument(
        "--no-velocity-sync", action="store_true", help="Do not sync/normalize velocity fields."
    )
    ap.add_argument(
        "--vel-min", type=int, default=1, help="Minimum velocity when normalizing (default=1)."
    )
    ap.add_argument(
        "--vel-max", type=int, default=127, help="Maximum velocity when normalizing (default=127)."
    )
    ap.add_argument("--audit", action="store_true", help="Print detection summary.")
    args = ap.parse_args()

    plan = load_json(args.inp)
    tracks: List[Dict[str, Any]] = plan.get("tracks") or []

    drum_indices: List[int] = []
    summary = []

    # まず判定
    for i, tr in enumerate(tracks):
        by_name = is_drum_by_name_or_flag(tr)
        by_gm = False
        gm_ratio = None
        if args.detect_gm:
            gm_ratio = gm_drum_ratio(tr)
            by_gm = gm_ratio >= args.min_gm_ratio
        is_drum = by_name or by_gm
        summary.append(
            {
                "idx": i,
                "name": tr.get("name") or tr.get("role") or f"track_{i}",
                "by_name": by_name,
                "gm_ratio": gm_ratio,
                "by_gm": by_gm,
                "decided": is_drum,
                "current_ch": tr.get("channel"),
            }
        )
        if is_drum:
            drum_indices.append(i)

    changed_tracks = 0
    changed_event_channels_total = 0
    normalized_vel_total = 0

    # 修正
    for i in drum_indices:
        tr = tracks[i]
        before_ch = tr.get("channel")
        ev_ch_changed, vel_norm = fix_drum_track(
            tr,
            required_ch=args.required_channel,
            sync_vel=(not args.no_velocity_sync),
            vmin=args.vel_min,
            vmax=args.vel_max,
        )
        changed_event_channels_total += ev_ch_changed
        normalized_vel_total += vel_norm
        if before_ch is None or int(before_ch) != int(args.required_channel):
            changed_tracks += 1

    if args.audit:
        print("=== Drum track detection summary ===")
        for s in summary:
            print(
                f"[{s['idx']:02d}] {s['name']}: by_name={s['by_name']}, "
                f"gm_ratio={s['gm_ratio']}, by_gm={s['by_gm']}, decided={s['decided']}, "
                f"current_ch={s['current_ch']}"
            )

    out_path = args.outp or args.inp
    save_json(plan, out_path)

    print("✅ Fix completed.")
    print(f"   Drum tracks detected: {len(drum_indices)}")
    print(f"   Tracks channel set to {args.required_channel}: {changed_tracks}")
    print(f"   Events channel normalized: {changed_event_channels_total}")
    if not args.no_velocity_sync:
        print(f"   Events velocity synced/clamped: {normalized_vel_total}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
