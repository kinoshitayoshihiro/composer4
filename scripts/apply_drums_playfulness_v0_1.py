#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
apply_drums_playfulness_v0_1.py

Drums人間味注入アルゴリズム v0.1（Performance層のみ）

二層方針:
  Expression層（Generator）: cymbal_mode, hihat_breathing, snare_ghost_style, 
                              tom_fill_cycle, fill grammar, crash punctuation
  Performance層（Apply）: micro timing jitter, velocity nuance, 既存イベント強弱補正のみ

Performance層の責務:
- マイクロタイミング揺らぎ（小さめ、±3~5ms）
- ベロシティ呼吸（既存イベントの微調整）
- 句読点は既存イベントの強弱補正のみ（新規ノート追加禁止）

使い方:
    from apply_drums_playfulness_v0_1 import apply_drums_playfulness_to_plan
    
    humanized_plan = apply_drums_playfulness_to_plan(
        drums_plan=drums_plan,
        dynamics_plan=drums_dynamics_plan,
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
class DrumsPlayfulnessContext:
    """Drums Performance層コンテキスト"""
    bar_index: int
    playfulness_bias: float = 0.08
    section: str = "VERSE"
    intensity: float = 0.5
    is_section_edge: bool = False


def _is_kick(event: Dict[str, Any]) -> bool:
    """キックドラムか判定"""
    drum_type = event.get("drum_type", "")
    midi_note = event.get("midi_note", 0)
    return drum_type in ("KICK", "BD") or midi_note in (35, 36)


def _is_snare(event: Dict[str, Any]) -> bool:
    """スネアドラムか判定"""
    drum_type = event.get("drum_type", "")
    midi_note = event.get("midi_note", 0)
    return drum_type in ("SNARE", "SD") or midi_note in (38, 40)


def _is_hihat(event: Dict[str, Any]) -> bool:
    """ハイハットか判定"""
    drum_type = event.get("drum_type", "")
    midi_note = event.get("midi_note", 0)
    return drum_type in ("HH_CLOSED", "HH_OPEN", "HH_PEDAL") or midi_note in (42, 44, 46)


def _is_crash(event: Dict[str, Any]) -> bool:
    """クラッシュシンバルか判定"""
    drum_type = event.get("drum_type", "")
    midi_note = event.get("midi_note", 0)
    return drum_type in ("CRASH", "CRASH1", "CRASH2") or midi_note in (49, 57)


def apply_playfulness_to_event(
    event: Dict[str, Any],
    ctx: DrumsPlayfulnessContext,
    playfulness_scale: float,
    rng: random.Random
) -> Dict[str, Any]:
    """
    単一イベントに人間味を適用（Performance層のみ）
    
    Args:
        event: Drums event辞書
        ctx: 人間味コンテキスト
        playfulness_scale: 強度スケール（0.0~1.0）
        rng: 乱数生成器
    
    Returns:
        人間味適用済みイベント
    """
    event = deepcopy(event)
    
    # === 1) マイクロタイミング揺らぎ（小さめ、Drums特性） ===
    # Drumsは他楽器より控えめ（±3~5ms）
    dt_ms = rng.uniform(-3, 5) * playfulness_scale
    
    # 楽器別微調整
    if _is_kick(event):
        # キックは安定感重視、揺らぎ半減
        dt_ms *= 0.5
    elif _is_hihat(event):
        # ハイハットは細かい揺れを許容
        dt_ms *= 1.2
    elif _is_crash(event):
        # クラッシュは少し前ノリ
        dt_ms += rng.uniform(-2, 0) * playfulness_scale
    
    # ms -> beats変換（BPM=120想定: 1beat=500ms）
    dt_beats = dt_ms / 500.0
    
    # === 2) ベロシティ呼吸（既存イベントの微調整のみ） ===
    dv = rng.randint(-4, 4) * playfulness_scale
    
    # 楽器別ダイナミクス保護
    if _is_kick(event):
        # キックは弱くしすぎない（ドラムの土台）
        dv = max(dv, -2)
    elif _is_snare(event):
        # スネアはアクセント保持
        if event.get("velocity", 80) > 90:
            dv = max(dv, -1)
    elif _is_crash(event):
        # クラッシュは強調
        dv += rng.randint(2, 5) * playfulness_scale
    
    # タイミング適用
    original_beat = event.get("beat", 0.0)
    event["beat"] = original_beat + dt_beats
    
    # ベロシティ適用
    original_vel = event.get("velocity", 80)
    event["velocity"] = int(_clamp(original_vel + dv, 1, 127))
    
    return event


def apply_drums_playfulness_to_plan(
    drums_plan: Dict[str, Any],
    dynamics_plan: Optional[Dict[str, Any]] = None,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Drums planに人間味を注入（Performance層のみ）
    
    Args:
        drums_plan: Drums plan JSON（events含む）
        dynamics_plan: Drums dynamics plan JSON（bars含む、任意）
        seed: 乱数シード
    
    Returns:
        人間味注入済みDrums plan
    """
    rng = random.Random(seed)
    
    # Dynamics planからbar別contextを構築
    bar_contexts: Dict[int, DrumsPlayfulnessContext] = {}
    
    if dynamics_plan:
        for bar_data in dynamics_plan.get("bars", []):
            bar_idx = bar_data.get("bar_index", 0)
            bar_contexts[bar_idx] = DrumsPlayfulnessContext(
                bar_index=bar_idx,
                playfulness_bias=bar_data.get("playfulness_bias", 0.08),
                section=bar_data.get("section", "VERSE"),
                intensity=bar_data.get("intensity", 0.5),
                is_section_edge=False,
            )
    
    # Eventsに人間味適用
    original_events = drums_plan.get("events", [])
    humanized_events = []
    
    for event in original_events:
        bar_idx = event.get("bar", 0)
        ctx = bar_contexts.get(bar_idx, DrumsPlayfulnessContext(bar_index=bar_idx))
        
        S = _clamp(ctx.playfulness_bias / 0.12, 0.0, 1.0)
        
        # Performance層のみ適用
        humanized_event = apply_playfulness_to_event(event, ctx, S, rng)
        humanized_events.append(humanized_event)
    
    # beat順ソート
    humanized_events.sort(key=lambda e: (e.get("bar", 0), e.get("beat", 0.0)))
    
    # Planコピーして更新
    result_plan = deepcopy(drums_plan)
    result_plan["events"] = humanized_events
    
    # メタデータ追加（統一仕様）
    if "metadata" not in result_plan:
        result_plan["metadata"] = {}
    
    result_plan["metadata"]["playfulness_version"] = "v0.1"
    result_plan["metadata"]["playfulness_applied"] = True
    result_plan["metadata"]["apply_mode"] = "performance_only"
    result_plan["metadata"]["event_count_before"] = len(original_events)
    result_plan["metadata"]["event_count_after"] = len(humanized_events)
    result_plan["metadata"]["new_events_added"] = 0  # Performance層は追加禁止
    
    # 楽器別統計
    drum_stats = {}
    for event in humanized_events:
        drum_type = event.get("drum_type", "unknown")
        if drum_type not in drum_stats:
            drum_stats[drum_type] = {"count": 0, "avg_velocity": 0}
        drum_stats[drum_type]["count"] += 1
        drum_stats[drum_type]["avg_velocity"] += event.get("velocity", 0)
    
    for drum_type, stats in drum_stats.items():
        if stats["count"] > 0:
            stats["avg_velocity"] /= stats["count"]
    
    result_plan["metadata"]["drum_statistics"] = drum_stats
    
    return result_plan


def main():
    """CLI実行用メイン"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Drums人間味注入 v0.1 (Performance層のみ)")
    parser.add_argument("--drums-plan", required=True, help="Drums plan JSON")
    parser.add_argument("--dynamics-plan", help="Drums dynamics plan JSON（任意）")
    parser.add_argument("--out", required=True, help="出力先JSON")
    parser.add_argument("--seed", type=int, default=42, help="乱数シード")
    
    args = parser.parse_args()
    
    # 読込
    with open(args.drums_plan, "r") as f:
        drums_plan = json.load(f)
    
    dynamics_plan = None
    if args.dynamics_plan:
        with open(args.dynamics_plan, "r") as f:
            dynamics_plan = json.load(f)
    
    # 人間味注入
    result = apply_drums_playfulness_to_plan(
        drums_plan=drums_plan,
        dynamics_plan=dynamics_plan,
        seed=args.seed
    )
    
    # 出力
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Drums人間味注入完了 (Performance層のみ):")
    print(f"   Apply mode: {result['metadata']['apply_mode']}")
    print(f"   Event count: {result['metadata']['event_count_before']} → {result['metadata']['event_count_after']}")
    print(f"   New events added: {result['metadata']['new_events_added']}")
    print(f"   Drum types: {len(result['metadata']['drum_statistics'])}")
    print(f"   Output: {args.out}")


if __name__ == "__main__":
    main()
