#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
apply_piano_dynamics_policy_v1_1.py

Provides a helper to apply a per‑bar dynamics policy for a Piano instrument.
This policy adheres to version 1.1 of the instrument dynamics specification and
is inspired by the drum dynamics policy. Each bar of a song is annotated with
its musical context (section, emotion buckets) and used to derive suitable
dynamics parameters for the piano line, such as role (FOUNDATION, DRIVE,
MELODIC), root anchor and pedal point biases, passing tone and octave motion
density, syncopation, register and articulation, and an overall
playfulness bias.

The YAML policy defines base settings for each section × arousal level along
with modifiers for valence and arousal, a low tension guard, section edge
boosts and value clamping. The function exported here will attach a
`piano_dynamics` dictionary to each supplied context and emit a JSON plan for
debugging.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

# Section normalizer統合
sys.path.insert(0, str(Path(__file__).parent))
from section_normalizer import SectionNormalizer


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


@dataclass
class PianoDynamicsDecision:
    bar_index: int
    section: str
    emotion_tag: str = "unknown"
    energy: float = 0.5
    valence: float = 0.0
    tension: float = 0.0

    valence_bucket: str = "NEU"
    arousal_bucket: str = "MID"

    role: str = "FOUNDATION"
    voicing_density: float = 0.7
    tension_ratio: float = 0.2
    voice_leading_strictness: float = 0.08
    arpeggio_density: float = 0.06
    pedal_sustain_bias: float = 0.08
    register_bias: str = "LOW_MID"
    articulation: str = "LEGATO"
    playfulness_bias: float = 0.06

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
            "role": self.role,
            "voicing_density": self.voicing_density,
            "tension_ratio": self.tension_ratio,
            "voice_leading_strictness": self.voice_leading_strictness,
            "arpeggio_density": self.arpeggio_density,
            "pedal_sustain_bias": self.pedal_sustain_bias,
            "register_bias": self.register_bias,
            "articulation": self.articulation,
            "playfulness_bias": self.playfulness_bias,
        }


class PianoDynamicsPolicyV11:
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
        # Section normalizer統合（priority制御付き）
        self.normalizer = SectionNormalizer(policy_yaml_path)

    def normalize_section(self, raw_section: str) -> str:
        """セクション名正規化（SectionNormalizer利用）"""
        return self.normalizer.normalize(raw_section)

    def bucket_valence(self, valence: float) -> str:
        vb = self.emotion_buckets.get("valence", {})
        NEG = vb.get("NEG", {"min": -1.0, "max": -0.3})
        POS = vb.get("POS", {"min": 0.3, "max": 1.0})
        if valence < NEG.get("max", -0.3):
            return "NEG"
        if valence > POS.get("min", 0.3):
            return "POS"
        return "NEU"

    def bucket_arousal(self, energy: float) -> str:
        ab = self.emotion_buckets.get("arousal", {})
        LOW = ab.get("LOW", {"min": 0.0, "max": 0.35})
        HIGH = ab.get("HIGH", {"min": 0.7, "max": 1.0})
        if energy < LOW.get("max", 0.35):
            return "LOW"
        if energy > HIGH.get("min", 0.7):
            return "HIGH"
        return "MID"

    def is_section_edge(self, sections: List[str], bar_index: int, window: int = 1) -> bool:
        if not sections:
            return False
        cur = sections[bar_index] if 0 <= bar_index < len(sections) else "UNKNOWN"
        nxt_idx = bar_index + 1
        if nxt_idx < len(sections):
            nxt = sections[nxt_idx]
            if cur != nxt:
                return True
        return False

    def decide_for_bar(
        self,
        bar_index: int,
        section_id: str,
        energy: float,
        valence: float,
        tension: float,
        emotion_tag: str,
        is_edge: bool,
    ) -> PianoDynamicsDecision:
        ar = self.bucket_arousal(energy)
        vb = self.bucket_valence(valence)
        debug_source = {
            "base_profile": f"section_profiles.{section_id}.{ar}",
            "axis_modifiers_applied": [],
            "low_tension_guard_applied": False,
            "section_edge_applied": False,
        }
        # Defaults
        role = self.defaults.get("role", "FOUNDATION")
        voicing_density = float(self.defaults.get("voicing_density", 0.7))
        tension_ratio = float(self.defaults.get("tension_ratio", 0.2))
        voice_leading_strictness = float(self.defaults.get("voice_leading_strictness", 0.08))
        arpeggio_density = float(self.defaults.get("arpeggio_density", 0.06))
        pedal_sustain_bias = float(self.defaults.get("pedal_sustain_bias", 0.08))
        register_bias = self.defaults.get("register_bias", "LOW_MID")
        articulation = self.defaults.get("articulation", "LEGATO")
        playfulness_bias = float(self.defaults.get("playfulness_bias", 0.06))

        # Section × arousal base
        sec_blob = (self.section_profiles.get(section_id, {}) or {}).get(ar, {}) or {}
        role = sec_blob.get("role", role)
        voicing_density = float(sec_blob.get("voicing_density", voicing_density))
        tension_ratio = float(sec_blob.get("tension_ratio", tension_ratio))
        voice_leading_strictness = float(
            sec_blob.get("voice_leading_strictness", voice_leading_strictness)
        )
        arpeggio_density = float(sec_blob.get("arpeggio_density", arpeggio_density))
        pedal_sustain_bias = float(sec_blob.get("pedal_sustain_bias", pedal_sustain_bias))
        register_bias = sec_blob.get("register_bias", register_bias)
        articulation = sec_blob.get("articulation", articulation)
        playfulness_bias = float(sec_blob.get("playfulness_bias", playfulness_bias))

        # Axis modifiers
        if vb == "NEG" and "valence_NEG" in self.axis_modifiers:
            mod = self.axis_modifiers["valence_NEG"]
            playfulness_bias += float(mod.get("playfulness_add", 0.0))
            tension_ratio += float(mod.get("pedal_point_add", 0.0))
            pedal_sustain_bias += float(mod.get("syncopation_add", 0.0))
            debug_source["axis_modifiers_applied"].append("valence_NEG")
        if vb == "POS" and "valence_POS" in self.axis_modifiers:
            mod = self.axis_modifiers["valence_POS"]
            playfulness_bias += float(mod.get("playfulness_add", 0.0))
            tension_ratio += float(mod.get("pedal_point_add", 0.0))
            arpeggio_density += float(mod.get("octave_motion_add", 0.0))
            debug_source["axis_modifiers_applied"].append("valence_POS")
        if ar == "HIGH" and "arousal_HIGH" in self.axis_modifiers:
            mod = self.axis_modifiers["arousal_HIGH"]
            voice_leading_strictness += float(mod.get("passing_tone_add", 0.0))
            pedal_sustain_bias += float(mod.get("syncopation_add", 0.0))
            debug_source["axis_modifiers_applied"].append("arousal_HIGH")

        # Low tension guard
        if self.low_guard.get("enable", False):
            cond = self.low_guard.get("condition", {}) or {}
            if energy <= float(cond.get("energy_max", 0.35)):
                enf = self.low_guard.get("enforce", {}) or {}
                voice_leading_strictness = float(
                    enf.get("voice_leading_strictness", voice_leading_strictness)
                )
                arpeggio_density = float(enf.get("arpeggio_density", arpeggio_density))
                pedal_sustain_bias = float(enf.get("pedal_sustain_bias", pedal_sustain_bias))
                tension_ratio = float(enf.get("tension_ratio", tension_ratio))
                max_p = float(enf.get("playfulness_bias_max", playfulness_bias))
                playfulness_bias = min(playfulness_bias, max_p)
                debug_source["low_tension_guard_applied"] = True

        # Section edge rules
        if self.edge_rules.get("enable", False) and is_edge:
            boost = self.edge_rules.get("boost", {}) or {}
            pass_add_map = boost.get("passing_tone_add_by_arousal", {}) or {}
            sync_add_map = boost.get("syncopation_add_by_arousal", {}) or {}
            voice_leading_strictness += float(pass_add_map.get(ar, 0.0))
            pedal_sustain_bias += float(sync_add_map.get(ar, 0.0))
            debug_source["section_edge_applied"] = True

        # Clamp parameters
        def clamp_param(name: str, value: float) -> float:
            conf = self.clamp_cfg.get(name, {}) or {}
            min_v = float(conf.get("min", 0.0))
            max_v = float(conf.get("max", 1.0))
            return _clamp(value, min_v, max_v)

        voicing_density = clamp_param("voicing_density", voicing_density)
        tension_ratio = clamp_param("tension_ratio", tension_ratio)
        voice_leading_strictness = clamp_param("voice_leading_strictness", voice_leading_strictness)
        arpeggio_density = clamp_param("arpeggio_density", arpeggio_density)
        pedal_sustain_bias = clamp_param("pedal_sustain_bias", pedal_sustain_bias)
        playfulness_bias = clamp_param("playfulness_bias", playfulness_bias)

        return PianoDynamicsDecision(
            bar_index=bar_index,
            section=section_id,
            emotion_tag=emotion_tag or "unknown",
            energy=_clamp(energy, 0.0, 1.0),
            valence=_clamp(valence, -1.0, 1.0),
            tension=_clamp(tension, 0.0, 1.0),
            valence_bucket=vb,
            arousal_bucket=ar,
            role=role,
            voicing_density=voicing_density,
            tension_ratio=tension_ratio,
            voice_leading_strictness=voice_leading_strictness,
            arpeggio_density=arpeggio_density,
            pedal_sustain_bias=pedal_sustain_bias,
            register_bias=register_bias,
            articulation=articulation,
            playfulness_bias=playfulness_bias,
        )


def apply_piano_dynamics_policy_v1_1(
    contexts: List[Any],
    policy_yaml_path: str = "config/piano_dynamics_policy_v1_1.yaml",
    sections_json_path: str = "plans/sections.json",
    emotion_profile_json_path: str = "plans/emotion_profile.json",
    out_plan_path: Optional[str] = None,  # None時はYAML outputs優先
) -> Dict[str, Any]:
    """
    Apply the piano dynamics policy to a list of BarContext-like objects.
    Each context receives a `piano_dynamics` dictionary and a plan JSON is
    written out. The returned dictionary contains the plan.

    Args:
        out_plan_path: 出力ファイルパス。Noneの場合はYAML outputs.per_bar_dynamics_jsonを使用。
    """
    policy = PianoDynamicsPolicyV11(policy_yaml_path)

    # YAML outputs優先参照（レビュー対応：問題2）
    if out_plan_path is None:
        yaml_outputs = policy.policy.get("outputs", {})
        out_plan_path = yaml_outputs.get("per_bar_dynamics_json", "plans/piano_dynamics_plan.json")
    sections_doc = _safe_read_json(sections_json_path, default={})
    sec_map: Dict[int, str] = {}
    for b in sections_doc.get("bars") or []:
        try:
            idx = int(b.get("bar_index"))
            sec_map[idx] = str(b.get("section") or b.get("section_name") or "")
        except Exception:
            continue
    emo_doc = _safe_read_json(emotion_profile_json_path, default={})
    emo_map: Dict[int, Dict[str, Any]] = {}
    for b in emo_doc.get("bars") or []:
        try:
            idx = int(b.get("bar_index"))
            emo_map[idx] = b
        except Exception:
            continue
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
    decisions: List[PianoDynamicsDecision] = []
    for i, ctx in enumerate(contexts):
        bar_index = int(getattr(ctx, "bar_index", i))
        raw_section = getattr(ctx, "section", None) or getattr(ctx, "section_name", None)
        if raw_section is None or raw_section == "":
            raw_section = sec_map.get(bar_index, "unknown")
        section_id = policy.normalize_section(str(raw_section))
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
            bar_index,
            section_id,
            float(energy),
            float(valence),
            float(tension),
            str(emotion_tag),
            is_edge,
        )
        decisions.append(dec)
        # Inject
        setattr(ctx, "section", section_id)
        setattr(
            ctx,
            "piano_dynamics",
            {
                "role": dec.role,
                "voicing_density": dec.voicing_density,
                "tension_ratio": dec.tension_ratio,
                "voice_leading_strictness": dec.voice_leading_strictness,
                "arpeggio_density": dec.arpeggio_density,
                "pedal_sustain_bias": dec.pedal_sustain_bias,
                "register_bias": dec.register_bias,
                "articulation": dec.articulation,
                "playfulness_bias": dec.playfulness_bias,
                "valence_bucket": dec.valence_bucket,
                "arousal_bucket": dec.arousal_bucket,
            },
        )
    plan = {
        "version": "1.1",
        "policy_id": str(policy.policy.get("policy_id", "PIANO_DYN_V1_1")),
        "instrument": "PIANO",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "bars": [d.to_dict() for d in decisions],
    }
    out_p = Path(out_plan_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
    return plan
