#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
drum_dynamics_policy_v1_1.py

A. ダイナミクス設計（Drum）を
section_types / emotion_axes の共通ラベル体系に準拠して適用する。

出力:
- plans/drum_dynamics_plan.json

効果:
- DrumPhraseGenerator が rulebook の拡張キーと同じ要領で
  cymbal/ghost/tom/breathing/playfulness/fill を bar単位で受け取れる。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


# -----------------------------
# ユーティリティ
# -----------------------------


def _safe_read_json(path: str | Path, default: Any) -> Any:
    p = Path(path)
    if not p.exists():
        return default
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def _safe_read_yaml(path: str | Path, default: Any) -> Any:
    p = Path(path)
    if not p.exists():
        return default
    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


# -----------------------------
# ポリシー構造
# -----------------------------


@dataclass
class DrumDynamicsDecision:
    bar_index: int
    section: str
    emotion_tag: str = "unknown"
    energy: float = 0.5
    valence: float = 0.0
    tension: float = 0.0

    valence_bucket: str = "NEU"
    arousal_bucket: str = "MID"

    cymbal_mode: str = "HH_ONLY"
    hihat_breathing: str = "AUTO_LIGHT"
    snare_ghost_style: str = "LIGHT_16"
    tom_fill_cycle: str = "OFF"
    playfulness_bias: float = 0.08

    fill_probability: float = 0.08
    crash_punct_probability: float = 0.05

    preferred_fill_tokens: List[str] = field(default_factory=list)
    debug_source: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "bar_index": self.bar_index,
            "section": self.section,
            "emotion_tag": self.emotion_tag,
            "energy": self.energy,
            "valence": self.valence,
            "tension": self.tension,
            "valence_bucket": self.valence_bucket,
            "arousal_bucket": self.arousal_bucket,
            "cymbal_mode": self.cymbal_mode,
            "hihat_breathing": self.hihat_breathing,
            "snare_ghost_style": self.snare_ghost_style,
            "tom_fill_cycle": self.tom_fill_cycle,
            "playfulness_bias": self.playfulness_bias,
            "fill_probability": self.fill_probability,
            "crash_punct_probability": self.crash_punct_probability,
            "preferred_fill_tokens": self.preferred_fill_tokens or [],
            "debug_source": self.debug_source or {},
        }


class DrumDynamicsPolicyV11:
    def __init__(self, policy_yaml_path: str | Path) -> None:
        self.policy_path = Path(policy_yaml_path)
        self.policy = _safe_read_yaml(self.policy_path, default={}) or {}

        self.defaults = self.policy.get("defaults", {})
        self.section_profiles = self.policy.get("section_profiles", {})
        self.axis_modifiers = self.policy.get("axis_modifiers", {})
        self.low_guard = self.policy.get("low_tension_guard", {})
        self.edge_rules = self.policy.get("section_edge_rules", {})
        self.clamp_cfg = self.policy.get("clamp", {})

        self.section_norm = (self.policy.get("section_normalization", {}) or {}).get("mapping", {})
        self.emotion_buckets = self.policy.get("emotion_buckets", {}) or {}

    # ---- section 正規化 ----
    def normalize_section(self, raw_section: str) -> str:
        if not raw_section:
            return "UNKNOWN"
        key = str(raw_section).strip()
        key_l = key.lower()
        return self.section_norm.get(key_l, self.section_norm.get(key, key)).upper()

    # ---- valence/arousal 量子化 ----
    def bucket_valence(self, valence: float) -> str:
        vb = self.emotion_buckets.get("valence", {})
        NEG = vb.get("NEG", {"min": -1.0, "max": -0.3})
        NEU = vb.get("NEU", {"min": -0.3, "max": 0.3})
        POS = vb.get("POS", {"min": 0.3, "max": 1.0})

        if valence < NEG["max"]:
            return "NEG"
        if valence > POS["min"]:
            return "POS"
        return "NEU"

    def bucket_arousal(self, energy: float) -> str:
        ab = self.emotion_buckets.get("arousal", {})
        LOW = ab.get("LOW", {"min": 0.0, "max": 0.35})
        MID = ab.get("MID", {"min": 0.35, "max": 0.7})
        HIGH = ab.get("HIGH", {"min": 0.7, "max": 1.0})

        if energy < LOW["max"]:
            return "LOW"
        if energy > HIGH["min"]:
            return "HIGH"
        return "MID"

    # ---- section edge 判定 ----
    def is_section_edge(self, sections: List[str], bar_index: int, window: int = 1) -> bool:
        if not sections:
            return False
        cur = sections[bar_index] if 0 <= bar_index < len(sections) else "UNKNOWN"
        # 次セクションが違うなら edge とみなす
        nxt_idx = bar_index + 1
        if nxt_idx < len(sections):
            nxt = sections[nxt_idx]
            if cur != nxt:
                return True
        return False

    # ---- 決定ロジック ----
    def decide_for_bar(
        self,
        bar_index: int,
        section_id: str,
        energy: float,
        valence: float,
        tension: float,
        emotion_tag: str,
        is_edge: bool,
    ) -> DrumDynamicsDecision:

        ar = self.bucket_arousal(energy)
        vb = self.bucket_valence(valence)

        debug = {
            "base_profile": f"section_profiles.{section_id}.{ar}",
            "axis_modifiers_applied": [],
            "low_tension_guard_applied": False,
            "section_edge_applied": False,
        }

        # 1) defaults
        cymbal_mode = self.defaults.get("cymbal_mode", "HH_ONLY")
        hihat_breathing = self.defaults.get("hihat_breathing", "AUTO_LIGHT")
        snare_ghost_style = self.defaults.get("snare_ghost_style", "LIGHT_16")
        tom_fill_cycle = self.defaults.get("tom_fill_cycle", "OFF")
        playfulness_bias = float(self.defaults.get("playfulness_bias", 0.08))

        fill_prob = float(
            (self.defaults.get("fill_probability_by_arousal", {}) or {}).get(ar, 0.08)
        )
        crash_prob = float(
            (self.defaults.get("crash_punct_probability_by_arousal", {}) or {}).get(ar, 0.05)
        )

        # 2) section × arousal base
        sec_blob = (self.section_profiles.get(section_id, {}) or {}).get(ar, {}) or {}
        cymbal_mode = sec_blob.get("cymbal_mode", cymbal_mode)
        hihat_breathing = sec_blob.get("hihat_breathing", hihat_breathing)
        snare_ghost_style = sec_blob.get("snare_ghost_style", snare_ghost_style)
        tom_fill_cycle = sec_blob.get("tom_fill_cycle", tom_fill_cycle)
        playfulness_bias = float(sec_blob.get("playfulness_bias", playfulness_bias))

        # 3) axis modifiers
        if vb == "NEG" and "valence_NEG" in self.axis_modifiers:
            m = self.axis_modifiers["valence_NEG"]
            playfulness_bias += float(m.get("playfulness_add", 0.0))
            pref = m.get("prefer_cymbal_modes", [])
            if pref:
                cymbal_mode = pref[0]
            debug["axis_modifiers_applied"].append("valence_NEG")

        if vb == "POS" and "valence_POS" in self.axis_modifiers:
            m = self.axis_modifiers["valence_POS"]
            playfulness_bias += float(m.get("playfulness_add", 0.0))
            pref = m.get("prefer_cymbal_modes", [])
            if pref:
                cymbal_mode = pref[0]
            debug["axis_modifiers_applied"].append("valence_POS")

        if ar == "HIGH" and "arousal_HIGH" in self.axis_modifiers:
            m = self.axis_modifiers["arousal_HIGH"]
            fill_prob += float(m.get("extra_fill_probability_add", 0.0))
            pref_tom = m.get("prefer_tom_fill_cycles", [])
            if pref_tom:
                tom_fill_cycle = pref_tom[0]
            debug["axis_modifiers_applied"].append("arousal_HIGH")

        # 4) low tension guard（簡易版：energyのみ判定）
        if self.low_guard.get("enable", False):
            cond = self.low_guard.get("condition", {}) or {}
            if energy <= float(cond.get("energy_max", 0.35)):
                enf = self.low_guard.get("enforce", {}) or {}
                fill_prob = float(enf.get("fill_probability", 0.02))
                crash_prob = float(enf.get("crash_punct_probability", 0.01))
                tom_fill_cycle = enf.get("tom_fill_cycle", tom_fill_cycle)
                max_p = float(enf.get("playfulness_bias_max", 0.08))
                playfulness_bias = min(playfulness_bias, max_p)
                debug["low_tension_guard_applied"] = True

        # 5) section edge boost
        preferred_tokens = []
        if self.edge_rules.get("enable", False) and is_edge:
            preferred_tokens = self.edge_rules.get("preferred_fill_tokens", []) or []
            add_map = self.edge_rules.get("extra_fill_probability_add_by_arousal", {}) or {}
            fill_prob += float(add_map.get(ar, 0.0))
            debug["section_edge_applied"] = True

        # 6) clamp
        clamp_pb = self.clamp_cfg.get("playfulness_bias", {}) or {}
        playfulness_bias = _clamp(
            playfulness_bias, float(clamp_pb.get("min", 0.0)), float(clamp_pb.get("max", 0.25))
        )

        fill_prob = _clamp(fill_prob, 0.0, 1.0)
        crash_prob = _clamp(crash_prob, 0.0, 1.0)

        return DrumDynamicsDecision(
            bar_index=bar_index,
            section=section_id,
            emotion_tag=emotion_tag or "unknown",
            energy=_clamp(energy, 0.0, 1.0),
            valence=_clamp(valence, -1.0, 1.0),
            tension=_clamp(tension, 0.0, 1.0),
            valence_bucket=vb,
            arousal_bucket=ar,
            cymbal_mode=cymbal_mode,
            hihat_breathing=hihat_breathing,
            snare_ghost_style=snare_ghost_style,
            tom_fill_cycle=tom_fill_cycle,
            playfulness_bias=playfulness_bias,
            fill_probability=fill_prob,
            crash_punct_probability=crash_prob,
            preferred_fill_tokens=preferred_tokens,
            debug_source=debug,
        )


# -----------------------------
# 外部公開API
# -----------------------------


def apply_drum_dynamics_policy_v1_1(
    contexts: List[Any],
    policy_yaml_path: str = "config/drum_dynamics_policy_v1_1.yaml",
    sections_json_path: str = "plans/sections.json",
    emotion_profile_json_path: str = "plans/emotion_profile.json",
    out_plan_path: str = "plans/drum_dynamics_plan.json",
) -> Dict[str, Any]:
    """
    BarContextリストに drum_dynamics を注入しつつ、
    drum_dynamics_plan.json を生成して返す。

    contexts の各要素は BarContext 互換を想定。
    最低限:
      - bar_index
      - section (または section_name)
      - emotion_tag/energy/valence/tension があれば利用
    """

    policy = DrumDynamicsPolicyV11(policy_yaml_path)

    # sections.json から補助的に section_name を拾えるようにする
    sections_doc = _safe_read_json(sections_json_path, default={})
    # 期待例: { "bars": [{ "bar_index":0, "section":"verse" }, ...] }
    sec_map: Dict[int, str] = {}
    for b in sections_doc.get("bars") or []:
        try:
            idx = int(b.get("bar_index"))
            sec_map[idx] = str(b.get("section") or b.get("section_name") or "")
        except Exception:
            continue

    # emotion_profile.json から bar別の axis を拾う
    emo_doc = _safe_read_json(emotion_profile_json_path, default={})
    emo_map: Dict[int, Dict[str, Any]] = {}
    for b in emo_doc.get("bars") or []:
        try:
            idx = int(b.get("bar_index"))
            emo_map[idx] = b
        except Exception:
            continue

    # contexts から section一覧を構築（edge判定用）
    sections_norm_list: List[str] = []
    for ctx in contexts:
        idx = getattr(ctx, "bar_index", None)
        raw = getattr(ctx, "section", None) or getattr(ctx, "section_name", None)
        if (raw is None or raw == "") and idx is not None:
            raw = sec_map.get(int(idx), "unknown")
        sections_norm_list.append(
            policy.normalize_section(str(raw) if raw is not None else "unknown")
        )

    edge_window = int((policy.edge_rules.get("edge_window_bars", 1)) or 1)

    decisions: List[DrumDynamicsDecision] = []

    for i, ctx in enumerate(contexts):
        bar_index = int(getattr(ctx, "bar_index", i))

        raw_section = getattr(ctx, "section", None) or getattr(ctx, "section_name", None)
        if raw_section is None or raw_section == "":
            raw_section = sec_map.get(bar_index, "unknown")
        section_id = policy.normalize_section(str(raw_section))

        # axis/タグは BarContext > emotion_profile.json の順で補完
        emo_row = emo_map.get(bar_index, {}) or {}

        energy = getattr(ctx, "energy", None)
        if energy is None:
            energy = emo_row.get("energy", 0.5)

        valence = getattr(ctx, "valence", None)
        if valence is None:
            valence = emo_row.get("valence", 0.0)

        tension = getattr(ctx, "tension", None)
        if tension is None:
            tension = emo_row.get("tension", 0.0)

        emotion_tag = getattr(ctx, "emotion_tag", None) or emo_row.get("emotion_tag", "unknown")

        is_edge = policy.is_section_edge(sections_norm_list, i, window=edge_window)

        dec = policy.decide_for_bar(
            bar_index=bar_index,
            section_id=section_id,
            energy=float(energy),
            valence=float(valence),
            tension=float(tension),
            emotion_tag=str(emotion_tag),
            is_edge=is_edge,
        )
        decisions.append(dec)

        # BarContextへ注入
        setattr(ctx, "section", section_id)
        setattr(
            ctx,
            "drum_dynamics",
            {
                "cymbal_mode": dec.cymbal_mode,
                "hihat_breathing": dec.hihat_breathing,
                "snare_ghost_style": dec.snare_ghost_style,
                "tom_fill_cycle": dec.tom_fill_cycle,
                "playfulness_bias": dec.playfulness_bias,
                "fill_probability": dec.fill_probability,
                "crash_punct_probability": dec.crash_punct_probability,
                "preferred_fill_tokens": dec.preferred_fill_tokens or [],
                "valence_bucket": dec.valence_bucket,
                "arousal_bucket": dec.arousal_bucket,
            },
        )

    plan = {
        "version": "1.1",
        "policy_id": str(policy.policy.get("policy_id", "DRM_DYN_V1_1")),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "bars": [d.to_dict() for d in decisions],
    }

    out_p = Path(out_plan_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")

    return plan


# -----------------------------
# CLI
# -----------------------------


def main():
    import argparse
    import pandas as pd

    parser = argparse.ArgumentParser(description="Drum Dynamics Policy v1.1 - CLI")
    parser.add_argument(
        "--bars",
        type=Path,
        required=True,
        help="bars.parquet path (contains bar_index, section, energy, etc.)",
    )
    parser.add_argument(
        "--sections",
        type=Path,
        help="sections.json path (optional, for section補完)",
    )
    parser.add_argument(
        "--emotion-profile",
        type=Path,
        help="emotion_profile.json path (optional, for emotion補完)",
    )
    parser.add_argument(
        "--policy",
        type=Path,
        default="config/drum_dynamics_policy_v1_1.yaml",
        help="Policy YAML path (default: config/drum_dynamics_policy_v1_1.yaml)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output drums_dynamics_plan.json path",
    )

    args = parser.parse_args()

    # Load bars.parquet
    bars_df = pd.read_parquet(args.bars)

    # Convert DataFrame rows to BarContext-like objects
    @dataclass
    class BarContextLite:
        bar_index: int
        section: Optional[str] = None
        energy: float = 0.5
        valence: float = 0.0
        tension: float = 0.0
        emotion_tag: str = "unknown"

    contexts = []
    for idx, row in bars_df.iterrows():
        ctx = BarContextLite(
            bar_index=int(row.get("bar_index", idx)),
            section=row.get("section_label", row.get("section", None)),
            energy=float(row.get("energy", 0.5)),
            valence=float(row.get("valence", 0.0)),
            tension=float(row.get("tension", 0.0)),
            emotion_tag=row.get("emotion_tag", "unknown"),
        )
        contexts.append(ctx)

    # Apply policy
    plan = apply_drum_dynamics_policy_v1_1(
        contexts=contexts,
        policy_yaml_path=str(args.policy),
        sections_json_path=str(args.sections) if args.sections else "plans/sections.json",
        emotion_profile_json_path=(
            str(args.emotion_profile) if args.emotion_profile else "plans/emotion_profile.json"
        ),
        out_plan_path=str(args.out),
    )

    print(f"✅ Drum Dynamics Plan generated: {args.out}")
    print(f"   Bars: {len(plan['bars'])}")
    print(f"   Policy ID: {plan['policy_id']}")


if __name__ == "__main__":
    main()
