#!/usr/bin/env python3
"""
validate_plan.py
----------------
Plan JSONスキーマ検証

Usage:
    python3 scripts/validate_plan.py song_packages/suno_project/song_001/bass_plan.json
    python3 scripts/validate_plan.py song_packages/suno_project/song_001/full_arrangement.json --require-drum-channel 9
"""
import json
import argparse
import re
from pathlib import Path
from typing import Dict, Any, List


def is_drum_track(track: Dict[str, Any]) -> bool:
    """トラックがドラム系かを判定。role/name に 'drum' または 'percussion' を含む、あるいは is_drum=True。"""
    role = (track.get("role") or track.get("name") or "").lower()
    if any(k in role for k in ("drum", "percussion")):
        return True
    if track.get("is_drum") is True:
        return True
    return False


def validate_drums_channel(plan: Dict[str, Any], required_ch: int = 9) -> List[Dict[str, Any]]:
    """
    ドラム系トラックに対して channel=required_ch（デフォルト9=10ch）を強制。
    - トラックレベル channel が存在する場合はそれも検証
    - イベント側 channel があればそれも検証（無ければトラック値を継承して検証）
    返り値: 違反イベントの列挙（最大件数制限は呼び出し側で行う）
    """
    violations: List[Dict[str, Any]] = []
    tracks = plan.get("tracks", [])
    for ti, tr in enumerate(tracks):
        if not is_drum_track(tr):
            continue
        track_name = tr.get("name") or tr.get("role") or f"track_{ti}"
        track_ch = tr.get("channel")
        # トラックレベルで channel が付いていてズレていたら即違反
        if track_ch is not None and int(track_ch) != int(required_ch):
            violations.append(
                {
                    "track_index": ti,
                    "track_name": track_name,
                    "event_index": -1,
                    "channel": track_ch,
                    "reason": f"Track-level channel must be {required_ch} for drums",
                }
            )
        # 各イベントの検証（イベント側に無ければ track_ch を継承して判定）
        for ei, ev in enumerate(tr.get("events", []) or []):
            ev_ch = ev.get("channel", track_ch)
            if ev_ch is None or int(ev_ch) != int(required_ch):
                violations.append(
                    {
                        "track_index": ti,
                        "track_name": track_name,
                        "event_index": ei,
                        "channel": ev_ch,
                        "reason": f"Drum event must be on channel {required_ch}",
                    }
                )
    return violations


def validate(plan_path: Path, strict: bool = True, require_drum_channel: int = None) -> bool:
    """
    Plan JSONの必須フィールド検証（厳格化版）

    Args:
        plan_path: Plan JSONファイルパス
        strict: 厳格モード（dur_beats/vel必須、範囲チェック）
        require_drum_channel: ドラム系トラックの必須チャンネル（0-15）。Noneなら検証しない

    Returns:
        True: valid, False: invalid
    """
    try:
        data = json.loads(plan_path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"❌ Failed to parse JSON: {e}")
        return False

    # トップレベル必須フィールド
    if "ppq" not in data:
        print(f"❌ Missing: ppq")
        return False
    if not isinstance(data["ppq"], int) or data["ppq"] <= 0:
        print(f"❌ Invalid ppq: {data.get('ppq')} (must be positive integer)")
        return False

    if "tempo_bpm" not in data:
        print(f"❌ Missing: tempo_bpm")
        return False
    if not isinstance(data["tempo_bpm"], (int, float)) or data["tempo_bpm"] <= 0:
        print(f"❌ Invalid tempo_bpm: {data.get('tempo_bpm')} (must be positive number)")
        return False

    if "tracks" not in data:
        print(f"❌ Missing: tracks")
        return False
    if not isinstance(data["tracks"], list) or len(data["tracks"]) == 0:
        print(f"❌ Invalid tracks: must be non-empty list")
        return False

    # --- Drums channel validation (MIDI ch10=9) ---
    if require_drum_channel is not None:
        drum_violations = validate_drums_channel(data, required_ch=require_drum_channel)
        if drum_violations:
            print(
                f"❌ Drum channel violations detected (drums must be on MIDI ch{require_drum_channel+1} = index {require_drum_channel}):"
            )
            for v in drum_violations[:20]:
                where = f"track#{v['track_index']} '{v['track_name']}'"
                if v["event_index"] >= 0:
                    where += f" event#{v['event_index']}"
                print(f"   - {where}: channel={v['channel']} -> {v['reason']}")
            remain = max(0, len(drum_violations) - 20)
            if remain:
                print(f"   ... and {remain} more")
            return False

    # 各トラック検証
    for idx, tr in enumerate(data["tracks"]):
        pfx = f"track[{idx}]"
        for req in ["name", "role", "channel", "program", "events"]:
            if req not in tr:
                print(f"❌ {pfx}: Missing {req}")
                return False

        # Channel/Program範囲チェック
        if not (0 <= tr["channel"] <= 15):
            print(f"❌ {pfx}: Invalid channel {tr['channel']} (must be 0-15)")
            return False
        if not (0 <= tr["program"] <= 127):
            print(f"❌ {pfx}: Invalid program {tr['program']} (must be 0-127)")
            return False

        # Events必須
        if not isinstance(tr["events"], list):
            print(f"❌ {pfx}: events must be list")
            return False

        # イベント検証（厳格化）
        for eidx, ev in enumerate(tr["events"]):
            epfx = f"{pfx}.events[{eidx}]"

            # bar/beat必須
            if "bar" not in ev or "beat" not in ev:
                print(f"❌ {epfx}: Missing bar/beat")
                return False

            # pitch or chord必須
            if "pitch" not in ev and "chord" not in ev:
                print(f"❌ {epfx}: Missing pitch or chord")
                return False

            # 厳格モード：dur_beats/vel必須
            if strict:
                if "dur_beats" not in ev:
                    print(f"❌ {epfx}: Missing dur_beats (strict mode)")
                    return False
                if "vel" not in ev:
                    print(f"❌ {epfx}: Missing vel (strict mode)")
                    return False

                # Velocity範囲チェック
                if not (0 <= ev["vel"] <= 127):
                    print(f"❌ {epfx}: Invalid vel {ev['vel']} (must be 0-127)")
                    return False

                # Duration正値チェック
                if ev["dur_beats"] <= 0:
                    print(f"❌ {epfx}: Invalid dur_beats {ev['dur_beats']} (must be positive)")
                    return False

            # Pitch範囲チェック（存在する場合）
            if "pitch" in ev:
                if isinstance(ev["pitch"], int):
                    if not (0 <= ev["pitch"] <= 127):
                        print(f"❌ {epfx}: Invalid pitch {ev['pitch']} (must be 0-127)")
                        return False
                elif isinstance(ev["pitch"], list):
                    for p in ev["pitch"]:
                        if not (0 <= p <= 127):
                            print(f"❌ {epfx}: Invalid pitch in list {p} (must be 0-127)")
                            return False

    print(f"✅ Valid: {plan_path}")
    return True


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Plan JSON検証（厳格化版＋ドラムチャンネル検証）")
    ap.add_argument("plan", type=Path, nargs="+", help="Plan JSON file(s)")
    ap.add_argument(
        "--lenient", action="store_true", help="Lenient mode (skip dur_beats/vel checks)"
    )
    ap.add_argument(
        "--require-drum-channel",
        type=int,
        default=None,
        help="Required MIDI channel (0-15) for drums. Default=None (skip drum channel validation). Use 9 for MIDI ch10.",
    )
    args = ap.parse_args()

    all_ok = True
    for p in args.plan:
        if not validate(p, strict=not args.lenient, require_drum_channel=args.require_drum_channel):
            all_ok = False

    exit(0 if all_ok else 1)
