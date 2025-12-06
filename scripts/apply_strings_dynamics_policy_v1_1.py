#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
apply_strings_dynamics_policy_v1_1.py

This module provides a helper to apply a per‑bar dynamics policy for a Strings
instrument based on the YAML specification at run time. It is modelled on the
drum dynamics policy and generalises the same ideas to strings: a bar is
annotated with a section (normalised against the official section vocabulary),
a valence bucket and arousal bucket (derived from the track's emotional
profile), and from these a base set of dynamics parameters is selected.

The YAML policy must follow the schema described in the companion YAML
file. It exposes keys such as `role`, `pad_density`, `ostinato_density`,
`countermelody_density`, `swell_probability`, `crescendo_bias`,
`register_bias`, `articulation`, and `playfulness_bias`. Additional axes
modifiers, low tension guards and section edge rules operate similarly to
those of the drum policy.

Example usage:

    from apply_strings_dynamics_policy_v1_1 import apply_strings_dynamics_policy_v1_1
    contexts = load_bar_contexts(...)  # list of BarContext objects
    plan = apply_strings_dynamics_policy_v1_1(
        contexts,
        policy_yaml_path="config/strings_dynamics_policy_v1_1.yaml",
        sections_json_path="plans/sections.json",
        emotion_profile_json_path="plans/emotion_profile.json",
        out_plan_path="plans/strings_dynamics_plan.json"
    )

The function will enrich each BarContext with a `strings_dynamics` attribute
containing the computed settings and produce a JSON plan file for audit and
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
class StringsDynamicsDecision:
    bar_index: int
    section: str
    emotion_tag: str = "unknown"
    energy: float = 0.5
    valence: float = 0.0
    tension: float = 0.0

    valence_bucket: str = "NEU"
    arousal_bucket: str = "MID"

    role: str = "PAD"
    pad_density: float = 0.5
    ostinato_density: float = 0.0
    countermelody_density: float = 0.0
    swell_probability: float = 0.0
    crescendo_bias: float = 0.0
    register_bias: str = "MID"
    articulation: str = "LEGATO"
    playfulness_bias: float = 0.0

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
            "pad_density": self.pad_density,
            "ostinato_density": self.ostinato_density,
            "countermelody_density": self.countermelody_density,
            "swell_probability": self.swell_probability,
            "crescendo_bias": self.crescendo_bias,
            "register_bias": self.register_bias,
            "articulation": self.articulation,
            "playfulness_bias": self.playfulness_bias,
        }


class StringsDynamicsPolicyV11:
    def __init__(self, policy_yaml_path: str | Path) -> None:
        self.policy_path = Path(policy_yaml_path)
        self.policy = _safe_read_yaml(self.policy_path, default={}) or {}

        self.defaults = self.policy.get("defaults", {})
        self.section_profiles = self.policy.get("section_profiles", {})
        self.axis_modifiers = self.policy.get("axis_modifiers", {})
        self.low_guard = self.policy.get("low_tension_guard", {})
        self.edge_rules = self.policy.get("section_edge_rules", {})
        self.clamp_cfg = self.policy.get("clamp", {})

        # Section normalisation mapping
        self.section_norm = (self.policy.get("section_normalization", {}) or {}).get("mapping", {})
        # Emotion bucket definitions
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
    ) -> StringsDynamicsDecision:
        ar = self.bucket_arousal(energy)
        vb = self.bucket_valence(valence)
        debug_source = {
            "base_profile": f"section_profiles.{section_id}.{ar}",
            "axis_modifiers_applied": [],
            "low_tension_guard_applied": False,
            "section_edge_applied": False,
        }
        # Start from defaults
        role = self.defaults.get("role", "PAD")
        pad_density = float(self.defaults.get("pad_density", 0.55))
        ostinato_density = float(self.defaults.get("ostinato_density", 0.0))
        countermelody_density = float(self.defaults.get("countermelody_density", 0.0))
        swell_probability = float(self.defaults.get("swell_probability", 0.0))
        crescendo_bias = float(self.defaults.get("crescendo_bias", 0.0))
        register_bias = self.defaults.get("register_bias", "MID")
        articulation = self.defaults.get("articulation", "LEGATO")
        playfulness_bias = float(self.defaults.get("playfulness_bias", 0.0))

        # Apply section × arousal profile
        sec_blob = (self.section_profiles.get(section_id, {}) or {}).get(ar, {}) or {}
        role = sec_blob.get("role", role)
        pad_density = float(sec_blob.get("pad_density", pad_density))
        ostinato_density = float(sec_blob.get("ostinato_density", ostinato_density))
        countermelody_density = float(sec_blob.get("countermelody_density", countermelody_density))
        swell_probability = float(sec_blob.get("swell_probability", swell_probability))
        crescendo_bias = float(sec_blob.get("crescendo_bias", crescendo_bias))
        register_bias = sec_blob.get("register_bias", register_bias)
        articulation = sec_blob.get("articulation", articulation)
        playfulness_bias = float(sec_blob.get("playfulness_bias", playfulness_bias))

        # Axis modifiers
        if vb == "NEG" and "valence_NEG" in self.axis_modifiers:
            mod = self.axis_modifiers["valence_NEG"]
            playfulness_bias += float(mod.get("playfulness_add", 0.0))
            # articulation or register modifications
            if mod.get("articulation_preference"):
                articulation = mod["articulation_preference"][0]
            if mod.get("register_bias"):
                register_bias = mod["register_bias"]
            debug_source["axis_modifiers_applied"].append("valence_NEG")
        if vb == "POS" and "valence_POS" in self.axis_modifiers:
            mod = self.axis_modifiers["valence_POS"]
            playfulness_bias += float(mod.get("playfulness_add", 0.0))
            if mod.get("articulation_preference"):
                articulation = mod["articulation_preference"][0]
            if mod.get("register_bias"):
                register_bias = mod["register_bias"]
            debug_source["axis_modifiers_applied"].append("valence_POS")
        if ar == "HIGH" and "arousal_HIGH" in self.axis_modifiers:
            mod = self.axis_modifiers["arousal_HIGH"]
            # These modifiers may adjust various densities or probabilities
            countermelody_density += float(mod.get("countermelody_add", 0.0))
            swell_probability += float(mod.get("swell_probability_add", 0.0))
            debug_source["axis_modifiers_applied"].append("arousal_HIGH")

        # Low tension guard: energy only considered
        if self.low_guard.get("enable", False):
            cond = self.low_guard.get("condition", {}) or {}
            if energy <= float(cond.get("energy_max", 0.35)):
                enf = self.low_guard.get("enforce", {}) or {}
                countermelody_density = float(
                    enf.get("countermelody_density", countermelody_density)
                )
                ostinato_density = float(enf.get("ostinato_density", ostinato_density))
                swell_probability = float(enf.get("swell_probability", swell_probability))
                max_p = float(enf.get("playfulness_bias_max", playfulness_bias))
                playfulness_bias = min(playfulness_bias, max_p)
                debug_source["low_tension_guard_applied"] = True

        # Section edge rules: boost crescendo and swell on edges
        if self.edge_rules.get("enable", False) and is_edge:
            boost = self.edge_rules.get("boost", {}) or {}
            cres_add_map = boost.get("crescendo_bias_add_by_arousal", {}) or {}
            swell_add_map = boost.get("swell_probability_add_by_arousal", {}) or {}
            crescendo_bias += float(cres_add_map.get(ar, 0.0))
            swell_probability += float(swell_add_map.get(ar, 0.0))
            debug_source["section_edge_applied"] = True

        # Clamp values
        def clamp_param(name: str, value: float) -> float:
            conf = self.clamp_cfg.get(name, {}) or {}
            min_v = float(conf.get("min", 0.0))
            max_v = float(conf.get("max", 1.0))
            return _clamp(value, min_v, max_v)

        pad_density = clamp_param("pad_density", pad_density)
        ostinato_density = clamp_param("ostinato_density", ostinato_density)
        countermelody_density = clamp_param("countermelody_density", countermelody_density)
        swell_probability = clamp_param("swell_probability", swell_probability)
        crescendo_bias = clamp_param("crescendo_bias", crescendo_bias)
        playfulness_bias = clamp_param("playfulness_bias", playfulness_bias)

        return StringsDynamicsDecision(
            bar_index=bar_index,
            section=section_id,
            emotion_tag=emotion_tag or "unknown",
            energy=_clamp(energy, 0.0, 1.0),
            valence=_clamp(valence, -1.0, 1.0),
            tension=_clamp(tension, 0.0, 1.0),
            valence_bucket=vb,
            arousal_bucket=ar,
            role=role,
            pad_density=pad_density,
            ostinato_density=ostinato_density,
            countermelody_density=countermelody_density,
            swell_probability=swell_probability,
            crescendo_bias=crescendo_bias,
            register_bias=register_bias,
            articulation=articulation,
            playfulness_bias=playfulness_bias,
        )


def apply_strings_dynamics_policy_v1_1(
    contexts: List[Any],
    policy_yaml_path: str = "config/strings_dynamics_policy_v1_1.yaml",
    sections_json_path: Optional[str] = "plans/sections.json",
    emotion_profile_json_path: Optional[str] = "plans/emotion_profile.json",
    out_plan_path: Optional[str] = "plans/strings_dynamics_plan.json",
) -> Dict[str, Any]:
    """
    Apply a strings dynamics policy to the list of BarContext-like objects.

    Each context is expected to have at least the following attributes:
      - bar_index
      - section or section_name
      - energy, valence, tension, emotion_tag (optional, fallback to emotion_profile.json)

    The function updates each context by attaching a `strings_dynamics` dictionary
    with the per‑bar settings and writes out a comprehensive plan JSON for
    auditing. It returns the plan dictionary.

    Missing files are tolerated: contexts will still be enriched using default
    values if no emotion profile or sections mapping is available.

    Args:
        out_plan_path: 出力ファイルパス。Noneの場合はYAML outputs.per_bar_dynamics_jsonを使用。
    """
    policy = StringsDynamicsPolicyV11(policy_yaml_path)

    # YAML outputs優先参照（レビュー対応：問題2）
    if out_plan_path is None:
        yaml_outputs = policy.policy.get("outputs", {})
        out_plan_path = yaml_outputs.get(
            "per_bar_dynamics_json", "plans/strings_dynamics_plan.json"
        )

    # Load auxiliary data (handle None paths)
    sections_doc = _safe_read_json(sections_json_path, default={}) if sections_json_path else {}
    sec_map: Dict[int, str] = {}
    for b in sections_doc.get("bars") or []:
        try:
            idx = int(b.get("bar_index"))
            sec_map[idx] = str(b.get("section") or b.get("section_name") or "")
        except Exception:
            continue

    emo_doc = (
        _safe_read_json(emotion_profile_json_path, default={}) if emotion_profile_json_path else {}
    )
    emo_map: Dict[int, Dict[str, Any]] = {}
    for b in emo_doc.get("bars") or []:
        try:
            idx = int(b.get("bar_index"))
            emo_map[idx] = b
        except Exception:
            continue

    # Normalise sections list for edge detection
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
    decisions: List[StringsDynamicsDecision] = []

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

        # Inject into context
        setattr(ctx, "section", section_id)
        setattr(
            ctx,
            "strings_dynamics",
            {
                "role": dec.role,
                "pad_density": dec.pad_density,
                "ostinato_density": dec.ostinato_density,
                "countermelody_density": dec.countermelody_density,
                "swell_probability": dec.swell_probability,
                "crescendo_bias": dec.crescendo_bias,
                "register_bias": dec.register_bias,
                "articulation": dec.articulation,
                "playfulness_bias": dec.playfulness_bias,
                "valence_bucket": dec.valence_bucket,
                "arousal_bucket": dec.arousal_bucket,
            },
        )

    plan = {
        "version": "1.1",
        "policy_id": str(policy.policy.get("policy_id", "STR_DYN_V1_1")),
        "instrument": "STRINGS",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "bars": [d.to_dict() for d in decisions],
    }

    # Write plan to file (if path provided)
    if out_plan_path:
        out_p = Path(out_plan_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")

    return plan
