#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
apply_strings_playfulness_v0_1.py

Strings人間味注入アルゴリズム v0.1（最小仕様）

Drum/Bass/Guitarと同じ「小さな揺れ＋呼吸＋句読点」をStringsに移植。
strings_dynamics_plan の playfulness_bias を"人間味注入の強度ノブ"として使用。

主要機能:
1) 小さな揺れ（micro-timing）: 控えめな±4ms（弦はアタックが丸い）
2) 呼吸（最重要）: bar内のゆっくりした微小スウェル（velocity波形）
3) 句読点: セクション境界のswell確率上昇・上行ライン追加
4) ロール別最適化: PAD/SWELL/COUNTERで呼吸の強度調整

使い方:
    from apply_strings_playfulness_v0_1 import apply_strings_playfulness_to_plan

    humanized_plan = apply_strings_playfulness_to_plan(
        strings_plan=strings_plan,
        dynamics_plan=strings_dynamics_plan,
        bars_parquet=bars_df,
        sections_json=sections_data,
        seed=42
    )
"""

from __future__ import annotations

import json
import random
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


@dataclass
class StringsPlayfulnessContext:
    """Strings人間味注入コンテキスト"""

    bar_index: int
    playfulness_bias: float = 0.08
    role: str = "PAD"
    section: str = "VERSE"
    swell_probability: float = 0.0
    articulation: str = "legato"
    is_section_edge: bool = False


def _apply_breathing_wave(
    events: List[Dict[str, Any]], ctx: StringsPlayfulnessContext, playfulness_scale: float
) -> None:
    """
    bar内のゆっくりした微小スウェル（velocity波形）を適用

    PAD/SWELL roleで特に効果大
    """
    if not events:
        return

    # role別の呼吸強度
    role_boost = 1.0 if ctx.role in ("PAD", "SWELL") else 0.6
    amp = 0.06 * playfulness_scale * role_boost  # ~6% max

    n = len(events)
    for i, ev in enumerate(events):
        phase = i / max(1, n - 1)  # 0..1
        # 0→1→0 の簡易カーブ（山型）
        wave = 1.0 - abs(phase * 2 - 1.0)
        factor = 1.0 + (wave - 0.5) * 2 * amp

        original_vel = ev.get("velocity", 80)
        ev["velocity"] = int(_clamp(original_vel * factor, 1, 127))


def _insert_swell_punctuation(
    events: List[Dict[str, Any]],
    bar_index: int,
    ctx: StringsPlayfulnessContext,
    playfulness_scale: float,
    rng: random.Random,
) -> None:
    """
    句読点としてのswell/上行ライン挿入

    セクション境界 or 4小節周期で軽いアクセント
    """
    # 確率計算（COUNTER/SWELL roleで高め）
    base_prob = 0.18
    role_bonus = 0.12 if ctx.role in ("COUNTER", "SWELL") else 0.0
    punctuation_chance = (base_prob + role_bonus) * playfulness_scale

    if rng.random() > punctuation_chance or not events:
        return

    # 最後のイベントを軽く持ち上げる（swell風）
    tail = events[-1]

    # velocity強調
    vel_boost = int(6 + 6 * playfulness_scale)
    original_vel = tail.get("velocity", 80)
    tail["velocity"] = int(_clamp(original_vel + vel_boost, 1, 127))

    # duration延長（余韻）
    dur_boost = 1.0 + 0.08 * playfulness_scale
    tail["duration_beats"] = tail.get("duration_beats", 1.0) * dur_boost


def apply_playfulness_to_event(
    event: Dict[str, Any],
    ctx: StringsPlayfulnessContext,
    playfulness_scale: float,
    rng: random.Random,
) -> Dict[str, Any]:
    """
    単一イベントに人間味を適用（タイミング揺れのみ）

    Args:
        event: Strings event辞書
        ctx: 人間味コンテキスト
        playfulness_scale: 強度スケール（0.0~1.0）
        rng: 乱数生成器

    Returns:
        人間味適用済みイベント
    """
    event = deepcopy(event)

    # === 小さな揺れ（micro-timing）: 控えめ ===
    dt_ms = rng.uniform(-4, 4) * playfulness_scale

    # ms -> beats変換（BPM=120想定: 1beat=500ms）
    dt_beats = dt_ms / 500.0

    # タイミング適用
    original_beat = event.get("beat", 0.0)
    event["beat"] = original_beat + dt_beats

    return event


def apply_strings_playfulness_to_plan(
    strings_plan: Dict[str, Any],
    dynamics_plan: Dict[str, Any],
    bars_parquet: Optional[Any] = None,
    sections_json: Optional[Dict[str, Any]] = None,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Strings planに人間味を注入

    Args:
        strings_plan: Strings plan JSON（events含む）
        dynamics_plan: Strings dynamics plan JSON（bars含む）
        bars_parquet: bars.parquet DataFrame（任意、BPM取得用）
        sections_json: sections.json（任意、セクション境界判定用）
        seed: 乱数シード

    Returns:
        人間味注入済みStrings plan
    """
    rng = random.Random(seed)

    # Dynamics planからbar別contextを構築
    bar_contexts: Dict[int, StringsPlayfulnessContext] = {}
    for bar_data in dynamics_plan.get("bars", []):
        bar_idx = bar_data.get("bar_index", 0)
        bar_contexts[bar_idx] = StringsPlayfulnessContext(
            bar_index=bar_idx,
            playfulness_bias=bar_data.get("playfulness_bias", 0.08),
            role=bar_data.get("role", "PAD"),
            section=bar_data.get("section", "VERSE"),
            swell_probability=bar_data.get("swell_probability", 0.0),
            articulation=bar_data.get("articulation", "legato"),
            is_section_edge=False,
        )

    # セクション境界判定
    if sections_json:
        section_bars = sections_json.get("bars", [])
        section_changes = set()
        prev_section = None
        for sb in section_bars:
            bar_idx = sb.get("bar_index", 0)
            section = sb.get("section", "verse")
            if prev_section and section != prev_section:
                section_changes.add(bar_idx)
            prev_section = section

        for bar_idx in section_changes:
            if bar_idx in bar_contexts:
                bar_contexts[bar_idx].is_section_edge = True

    # Bar別にイベントをグループ化
    events_by_bar: Dict[int, List[Dict[str, Any]]] = {}
    original_events = strings_plan.get("events", [])

    for event in original_events:
        bar_idx = event.get("bar", 0)
        if bar_idx not in events_by_bar:
            events_by_bar[bar_idx] = []
        events_by_bar[bar_idx].append(event)

    # Bar別に人間味適用
    humanized_events = []
    punctuation_stats = {"swell_count": 0}

    for bar_idx in sorted(events_by_bar.keys()):
        ctx = bar_contexts.get(bar_idx, StringsPlayfulnessContext(bar_index=bar_idx))
        bar_events = events_by_bar[bar_idx]

        S = _clamp(ctx.playfulness_bias / 0.12, 0.0, 1.0)

        # 1) タイミング揺れ適用
        bar_humanized = []
        for event in bar_events:
            humanized_event = apply_playfulness_to_event(event, ctx, S, rng)
            bar_humanized.append(humanized_event)

        # 2) 呼吸（velocity波形）適用
        _apply_breathing_wave(bar_humanized, ctx, S)

        # 3) 句読点挿入
        if ctx.is_section_edge or bar_idx % 4 == 3:
            before_count = len(bar_humanized)
            _insert_swell_punctuation(bar_humanized, bar_idx, ctx, S, rng)
            if len(bar_humanized) > before_count:
                punctuation_stats["swell_count"] += 1

        humanized_events.extend(bar_humanized)

    # beat順ソート
    humanized_events.sort(key=lambda e: (e.get("bar", 0), e.get("beat", 0.0)))

    # Planコピーして更新
    result_plan = deepcopy(strings_plan)
    result_plan["events"] = humanized_events

    # メタデータ追加
    if "metadata" not in result_plan:
        result_plan["metadata"] = {}

    result_plan["metadata"]["playfulness_version"] = "v0.1"
    result_plan["metadata"]["playfulness_applied"] = True
    result_plan["metadata"]["original_event_count"] = len(original_events)
    result_plan["metadata"]["humanized_event_count"] = len(humanized_events)
    result_plan["metadata"]["swell_punctuation_count"] = punctuation_stats["swell_count"]

    # Role別統計
    role_stats = {}
    for event in humanized_events:
        role = event.get("role", "unknown")
        if role not in role_stats:
            role_stats[role] = {"count": 0, "avg_velocity": 0, "avg_duration": 0}
        role_stats[role]["count"] += 1
        role_stats[role]["avg_velocity"] += event.get("velocity", 0)
        role_stats[role]["avg_duration"] += event.get("duration_beats", 0)

    for role, stats in role_stats.items():
        if stats["count"] > 0:
            stats["avg_velocity"] /= stats["count"]
            stats["avg_duration"] /= stats["count"]

    result_plan["metadata"]["role_statistics"] = role_stats

    return result_plan


def main():
    """CLI実行用メイン"""
    import argparse

    parser = argparse.ArgumentParser(description="Strings人間味注入 v0.1")
    parser.add_argument("--strings-plan", required=True, help="Strings plan JSON")
    parser.add_argument("--dynamics-plan", required=True, help="Strings dynamics plan JSON")
    parser.add_argument("--sections-json", help="sections.json（任意）")
    parser.add_argument("--out", required=True, help="出力先JSON")
    parser.add_argument("--seed", type=int, default=42, help="乱数シード")

    args = parser.parse_args()

    # 読込
    with open(args.strings_plan, "r") as f:
        strings_plan = json.load(f)

    with open(args.dynamics_plan, "r") as f:
        dynamics_plan = json.load(f)

    sections_json = None
    if args.sections_json:
        with open(args.sections_json, "r") as f:
            sections_json = json.load(f)

    # 人間味注入
    result = apply_strings_playfulness_to_plan(
        strings_plan=strings_plan,
        dynamics_plan=dynamics_plan,
        sections_json=sections_json,
        seed=args.seed,
    )

    # 出力
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"✅ Strings人間味注入完了:")
    print(f"   Original events: {result['metadata']['original_event_count']}")
    print(f"   Humanized events: {result['metadata']['humanized_event_count']}")
    print(f"   Swell punctuation: {result['metadata']['swell_punctuation_count']}")
    print(f"\n   Role statistics:")
    for role, stats in result["metadata"]["role_statistics"].items():
        print(
            f"     {role}: count={stats['count']}, "
            f"avg_vel={stats['avg_velocity']:.1f}, "
            f"avg_dur={stats['avg_duration']:.3f}"
        )
    print(f"\n   Output: {args.out}")


if __name__ == "__main__":
    main()
