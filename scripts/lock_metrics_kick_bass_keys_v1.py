#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
lock_metrics_kick_bass_keys_v1.py

song_004 仕様の

- drums_plan_v2.json
- bass_plan_v2.json
- keys_plan_v1.json

から Kick / Bass / Keys の発音タイミングを取得し、
ロック度 (0.0〜1.0) を計算する最小実装。

ロック度の定義（簡易版）:
- pair_lock(kick, bass):
  Kick の各オンセットについて ±tolerance_beats 内に
  Bass のオンセットが存在する割合
- pair_lock(kick, keys):
  Keys（役割が伴奏系のみ対象）のオンセットとの同期割合
- pair_lock(bass, keys):
  Bass と Keys の同期割合
- triple_lock:
  1つの Kick の近傍に Bass も Keys も同時に居る割合

rockness_score（総合指標）:
  0.5 * kick_bass_lock + 0.3 * kick_keys_lock + 0.2 * bass_keys_lock
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import yaml


# ===== Dataclasses =====


@dataclass
class TimedEvent:
    time_beats: float
    velocity: int
    role: Optional[str] = None

    def __lt__(self, other: "TimedEvent") -> bool:
        return self.time_beats < other.time_beats


# ===== Kick pattern tag 正規化 =====


def normalize_kick_pattern_tag(raw: Optional[str]) -> Optional[str]:
    """
    drums_plan の metadata 等に入る kick_pattern_tag を正規化。

    例:
    - "4 on the floor" -> "FOUR_ON_THE_FLOOR"
    - "rock_8beat_standard" -> "ROCK_8BEAT_STANDARD"
    """
    if not raw:
        return None

    s = raw.strip().lower()
    s = re.sub(r"\s+", "_", s)
    s = s.replace("-", "_")

    mapping = {
        "4_on_the_floor": "FOUR_ON_THE_FLOOR",
        "four_on_the_floor": "FOUR_ON_THE_FLOOR",
        "four_on_floor": "FOUR_ON_THE_FLOOR",
        "four_on_the_floor_disco": "FOUR_ON_THE_FLOOR",
        "rock_8beat": "ROCK_8BEAT_STANDARD",
        "rock_8beat_standard": "ROCK_8BEAT_STANDARD",
        "punk_fast_2beat": "PUNK_FAST_2BEAT",
        "two_beat_punk": "PUNK_FAST_2BEAT",
    }

    return mapping.get(s, s.upper())


# ===== ローダ群 =====


def _event_bar(ev: Dict[str, Any]) -> int:
    if "bar" in ev:
        return int(ev["bar"])
    if "bar_index" in ev:
        return int(ev["bar_index"])
    if "measure" in ev:
        return int(ev["measure"])
    return 0


def _event_beat(ev: Dict[str, Any]) -> float:
    return float(ev.get("beat", 0.0))


def load_kick_events(
    drums_plan_path: Path | str,
    beats_per_bar: int = 4,
    kick_pitches: Iterable[int] = (35, 36),
) -> Tuple[List[TimedEvent], Optional[str]]:
    """
    drums_plan_v2.json から Kick イベントだけ抽出。
    - drum_type == "kick" を優先
    - 無ければ pitch in (35,36) を Kick とみなす
    """
    p = Path(drums_plan_path)
    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)

    events_raw = data.get("events", data if isinstance(data, list) else [])
    metadata = data.get("metadata", {}) if isinstance(data, dict) else {}
    pattern_tag_raw = metadata.get("kick_pattern_tag")

    kick_events: List[TimedEvent] = []
    for ev in events_raw:
        drum_type = ev.get("drum_type")
        pitch = int(ev.get("pitch", 0))

        if drum_type == "kick" or pitch in kick_pitches:
            bar = _event_bar(ev)
            beat = _event_beat(ev)
            time_beats = bar * float(beats_per_bar) + beat
            vel = int(ev.get("velocity", 100))
            kick_events.append(TimedEvent(time_beats=time_beats, velocity=vel, role="kick"))

    kick_events.sort()
    return kick_events, normalize_kick_pattern_tag(pattern_tag_raw)


def load_bass_events(
    bass_plan_path: Path | str,
    beats_per_bar: int = 4,
) -> List[TimedEvent]:
    """
    bass_plan_v2.json から Bass のオンセット群を読み取る。
    """
    p = Path(bass_plan_path)
    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)

    events_raw = data.get("events", data if isinstance(data, list) else [])
    bass_events: List[TimedEvent] = []

    for ev in events_raw:
        bar = _event_bar(ev)
        beat = _event_beat(ev)
        time_beats = bar * float(beats_per_bar) + beat
        vel = int(ev.get("velocity", 100))
        role = ev.get("role")
        bass_events.append(TimedEvent(time_beats=time_beats, velocity=vel, role=role))

    bass_events.sort()
    return bass_events


_ACTIVE_KEYS_ROLES = {
    "COMP_FOUNDATION",
    "PUMP_OFFBEAT",
    "ARP_MOTION",
    "HOOK_RIFF",
    "PAD_WIDE",
    "COUNTER_MELODY",
}


def load_keys_events(
    keys_plan_path: Path | str,
    beats_per_bar: int = 4,
    active_roles_only: bool = True,
) -> List[TimedEvent]:
    """
    keys_plan_v1.json から Keys のオンセット群を読み取る。
    active_roles_only=True の場合は、伴奏・土台系の役割だけを対象にする。
    """
    p = Path(keys_plan_path)
    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)

    events_raw = data.get("events", data if isinstance(data, list) else [])
    keys_events: List[TimedEvent] = []

    for ev in events_raw:
        bar = _event_bar(ev)
        beat = _event_beat(ev)
        time_beats = bar * float(beats_per_bar) + beat
        vel = int(ev.get("velocity", 100))
        role = ev.get("keys_role") or ev.get("role")

        if active_roles_only and role and role not in _ACTIVE_KEYS_ROLES:
            continue

        keys_events.append(TimedEvent(time_beats=time_beats, velocity=vel, role=role))

    keys_events.sort()
    return keys_events


# ===== ロック度メトリクス =====


def _has_near_hit(
    time_beats: float,
    sorted_times: List[float],
    tolerance_beats: float,
) -> bool:
    """昇順ソート済み times の中に ±tolerance 内の値があるかを高速チェック。"""
    if not sorted_times:
        return False

    # 2-pointer でも良いが、ここでは単純に線形で十分（最小実装）
    for t in sorted_times:
        if t > time_beats + tolerance_beats:
            break
        if abs(t - time_beats) <= tolerance_beats:
            return True
    return False


def compute_pair_lock(
    primary: List[TimedEvent],
    reference: List[TimedEvent],
    tolerance_beats: float = 0.05,
) -> float:
    """
    primary 側の各イベントについて、reference 側に ±tolerance 以内のイベントが
    存在する割合を返す。
    """
    if not primary or not reference:
        return 0.0

    ref_times = [e.time_beats for e in reference]
    ref_times.sort()

    locked = 0
    for ev in primary:
        if _has_near_hit(ev.time_beats, ref_times, tolerance_beats):
            locked += 1

    return locked / float(len(primary))


def compute_triple_lock(
    kicks: List[TimedEvent],
    bass: List[TimedEvent],
    keys: List[TimedEvent],
    tolerance_beats: float = 0.05,
) -> float:
    """
    各 Kick イベントについて、
    - Bass にも ±tolerance ビート以内のヒット
    - Keys にも ±tolerance ビート以内のヒット
    が同時に存在する割合。
    """
    if not kicks or not bass or not keys:
        return 0.0

    bass_times = [e.time_beats for e in bass]
    keys_times = [e.time_beats for e in keys]
    bass_times.sort()
    keys_times.sort()

    locked = 0
    for ev in kicks:
        tb = ev.time_beats
        if _has_near_hit(tb, bass_times, tolerance_beats) and _has_near_hit(
            tb, keys_times, tolerance_beats
        ):
            locked += 1

    return locked / float(len(kicks))


def compute_lock_metrics(
    kicks: List[TimedEvent],
    bass: List[TimedEvent],
    keys: List[TimedEvent],
    tolerance_beats: float = 0.05,
) -> Dict[str, float]:
    """
    Kick × Bass × Keys のロック度まとめ。
    """
    kick_bass = compute_pair_lock(kicks, bass, tolerance_beats)
    kick_keys = compute_pair_lock(kicks, keys, tolerance_beats)
    bass_keys = compute_pair_lock(bass, keys, tolerance_beats)
    triple = compute_triple_lock(kicks, bass, keys, tolerance_beats)

    rockness = 0.5 * kick_bass + 0.3 * kick_keys + 0.2 * bass_keys

    return {
        "kick_bass_lock": kick_bass,
        "kick_keys_lock": kick_keys,
        "bass_keys_lock": bass_keys,
        "triple_lock": triple,
        "rockness_score": rockness,
    }


# ===== CLI =====


def _cli() -> None:
    ap = argparse.ArgumentParser(description="Kick×Bass×Keys ロック度メトリクス計算")
    ap.add_argument(
        "--drums-plan",
        type=str,
        required=True,
        help="drums_plan_v2.json へのパス（Kick を含む）",
    )
    ap.add_argument(
        "--bass-plan",
        type=str,
        required=True,
        help="bass_plan_v2.json へのパス",
    )
    ap.add_argument(
        "--keys-plan",
        type=str,
        required=True,
        help="keys_plan_v1.json へのパス",
    )
    ap.add_argument(
        "--beats-per-bar",
        type=int,
        default=4,
        help="1小節あたりの拍数（デフォルト: 4）",
    )
    ap.add_argument(
        "--tolerance-beats",
        type=float,
        default=0.05,
        help="ロック判定の許容ビート幅（デフォルト: 0.05）",
    )
    ap.add_argument(
        "--out",
        type=str,
        default=None,
        help="出力JSONパス（任意）",
    )
    ap.add_argument(
        "--expectations",
        type=str,
        default="config/lock_expectations_v1.yaml",
        help="QA基準値YAMLパス（デフォルト: config/lock_expectations_v1.yaml）",
    )
    args = ap.parse_args()

    kicks, pattern_tag = load_kick_events(args.drums_plan, beats_per_bar=args.beats_per_bar)
    bass = load_bass_events(args.bass_plan, beats_per_bar=args.beats_per_bar)
    keys = load_keys_events(args.keys_plan, beats_per_bar=args.beats_per_bar)

    metrics = compute_lock_metrics(
        kicks=kicks,
        bass=bass,
        keys=keys,
        tolerance_beats=args.tolerance_beats,
    )

    result: Dict[str, Any] = {
        "kick_pattern_tag": pattern_tag,
        "num_kick_events": len(kicks),
        "num_bass_events": len(bass),
        "num_keys_events": len(keys),
        "tolerance_beats": args.tolerance_beats,
    }
    result.update(metrics)

    # QA判定（expectations YAML読み込み）
    qa_status = "PASS"
    qa_messages: List[str] = []

    if args.expectations and Path(args.expectations).exists():
        try:
            with open(args.expectations, "r", encoding="utf-8") as f:
                expectations = yaml.safe_load(f) or {}

            # デフォルト期待値
            defaults = expectations.get("defaults", {})
            default_kick_bass_range = defaults.get("kick_bass_sync_rate_range", [0.5, 0.8])
            default_lock_score_range = defaults.get("lock_score_range", [0.45, 0.75])

            # pattern_tag別の期待値を検索（style_section_emotionから）
            matched_expectation = None
            for entry in expectations.get("style_section_emotion", []):
                if entry.get("style_id") == pattern_tag:
                    matched_expectation = entry
                    break

            if matched_expectation:
                kick_bass_range = matched_expectation.get(
                    "kick_bass_sync_rate_range", default_kick_bass_range
                )
                lock_score_range = matched_expectation.get(
                    "lock_score_range", default_lock_score_range
                )
            else:
                kick_bass_range = default_kick_bass_range
                lock_score_range = default_lock_score_range

            # QA判定
            min_kick_bass, max_kick_bass = kick_bass_range
            min_lock_score, max_lock_score = lock_score_range

            if metrics["kick_bass_lock"] < min_kick_bass:
                qa_status = "WARNING"
                qa_messages.append(
                    f"Kick×Bass lock {metrics['kick_bass_lock']:.1%} < minimum {min_kick_bass:.1%}"
                )

            if metrics["rockness_score"] < min_lock_score:
                qa_status = "WARNING"
                qa_messages.append(
                    f"Rockness score {metrics['rockness_score']:.1%} < minimum {min_lock_score:.1%}"
                )

            result["qa_status"] = qa_status
            result["qa_messages"] = qa_messages
            result["expectations_file"] = str(args.expectations)
            result["expected_kick_bass_range"] = kick_bass_range
            result["expected_lock_score_range"] = lock_score_range

        except Exception as e:
            result["qa_status"] = "ERROR"
            result["qa_error"] = str(e)

    # 結果表示
    print("🎵 Kick×Bass×Keys Lock Metrics")
    print(f"   Kick pattern: {pattern_tag or 'N/A'}")
    print(f"   Kick events: {len(kicks)}")
    print(f"   Bass events: {len(bass)}")
    print(f"   Keys events: {len(keys)} (active roles only)")
    print(f"   Tolerance: {args.tolerance_beats} beats\n")

    print("📊 Lock Scores:")
    print(f"   Kick×Bass lock:    {metrics['kick_bass_lock']:.1%}")
    print(f"   Kick×Keys lock:    {metrics['kick_keys_lock']:.1%}")
    print(f"   Bass×Keys lock:    {metrics['bass_keys_lock']:.1%}")
    print(f"   Triple lock:       {metrics['triple_lock']:.1%}")
    print(f"   Rockness score:    {metrics['rockness_score']:.1%}\n")

    # QA判定結果表示
    if "qa_status" in result:
        status_emoji = {"PASS": "✅", "WARNING": "⚠️", "ERROR": "❌"}.get(result["qa_status"], "❓")
        print(f"🔍 QA Status: {status_emoji} {result['qa_status']}")
        if result.get("qa_messages"):
            for msg in result["qa_messages"]:
                print(f"   - {msg}")
        print()

    # JSON出力
    if args.out:
        Path(args.out).write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"✅ Saved: {args.out}")
    else:
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    _cli()
