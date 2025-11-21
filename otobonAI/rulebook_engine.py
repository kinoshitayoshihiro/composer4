# otobonAI/rulebook_engine.py

"""
Rulebook Engine for OtobonAI.

Loads and processes composer rules from rulebook.yaml, matching them to
song contexts (section, harmony, emotion) to drive GuideToneAI and EmotionAI.
"""

from __future__ import annotations

import json
import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class RuleActionEmotion:
    """EmotionAI action derived from rule mechanics."""

    energy_delta: float = 0.0
    tension_delta: float = 0.0
    brightness_delta: float = 0.0
    valence_delta: float = 0.0
    density_delta: float = 0.0
    velocity_scale: Optional[float] = None  # Phase 2.0: 直接指定
    duration_scale: Optional[float] = None  # Phase 2.0: 直接指定
    density_scale: Optional[float] = None  # Phase 2.0: 直接指定（density_deltaと別）
    tags_add: List[str] = field(default_factory=list)


@dataclass
class RuleActionGuideTone:
    """GuideToneAI action derived from rule mechanics."""

    priority_tones: List[str] = field(default_factory=list)  # e.g. ["3rd", "7th", "9th"]
    default_register: str = "mid"  # "low" | "mid" | "high"
    motion: str = "step"  # "step" | "leap_ok" | "hold"
    phrase_shape: Optional[str] = None  # "arch", "uphill", "downhill"
    notes_per_bar: Optional[float] = None  # density hint


@dataclass
class Rule:
    """Wrapper for a single rule from rulebook.json."""

    raw: Dict[str, Any]

    @property
    def id(self) -> str:
        return self.raw.get("id", "")

    @property
    def name(self) -> str:
        return self.raw.get("name", "")

    @property
    def domain(self) -> str:
        """harmony, melody, bass, rhythm, form, guidetone, emotion."""
        return self.raw.get("domain", "")

    @property
    def emotion_tags(self) -> List[str]:
        return self.raw.get("emotion_tags", [])

    @property
    def guide_tone_tags(self) -> List[str]:
        return self.raw.get("guide_tone_tags", [])

    @property
    def context(self) -> Dict[str, Any]:
        """Context conditions: sections, tempo_bpm_range, genre, difficulty."""
        return self.raw.get("context", {})

    @property
    def mechanics(self) -> Dict[str, Any]:
        """Detailed mechanics: melody, harmony, bass, rhythm."""
        return self.raw.get("mechanics", {})

    @property
    def emotion_effect(self) -> str:
        return self.raw.get("emotion_effect", "")

    def get_emotion_action(self) -> Optional[RuleActionEmotion]:
        """
        Extract emotion deltas from mechanics and emotion_tags.

        Phase 2.0: mechanics.emotion から直接取得（優先）
        Fallback: emotion_tags からヒューリスティック推定
        """
        if self.domain not in ("emotion", "harmony", "melody"):
            return None

        # Phase 2.0: mechanics.emotion から直接取得
        emotion_mech = self.mechanics.get("emotion", {})
        if emotion_mech:
            return RuleActionEmotion(
                energy_delta=emotion_mech.get("energy_delta", 0.0),
                tension_delta=emotion_mech.get("tension_delta", 0.0),
                brightness_delta=emotion_mech.get("brightness_delta", 0.0),
                valence_delta=emotion_mech.get("valence_delta", 0.0),
                density_delta=emotion_mech.get("density_delta", 0.0),
                velocity_scale=emotion_mech.get("velocity_scale"),
                duration_scale=emotion_mech.get("duration_scale"),
                density_scale=emotion_mech.get("density_scale"),
                tags_add=list(self.emotion_tags),
            )

        # Fallback: emotion_tags からヒューリスティック推定
        energy = 0.0
        tension = 0.0
        brightness = 0.0
        valence = 0.0
        density = 0.0

        for tag in self.emotion_tags:
            if tag in ("bright", "happy", "refreshing", "hopeful", "epic"):
                energy += 0.05
                brightness += 0.05
                valence += 0.05
            elif tag in ("sad", "dark", "lonely"):
                energy -= 0.05
                brightness -= 0.05
                valence -= 0.05
            elif tag in ("tense", "dramatic"):
                tension += 0.1
                energy += 0.03
            elif tag in ("relaxed", "floating"):
                tension -= 0.1
                energy -= 0.03
            elif tag in ("nostalgic", "bittersweet"):
                valence -= 0.02
                brightness -= 0.02

        # Density from rhythm mechanics
        rhythm = self.mechanics.get("rhythm", {})
        density_desc = str(rhythm.get("density", "")).lower()
        if "増やす" in density_desc or "厚く" in density_desc:
            density += 0.1
        elif "薄め" in density_desc or "控えめ" in density_desc:
            density -= 0.1

        return RuleActionEmotion(
            energy_delta=energy,
            tension_delta=tension,
            brightness_delta=brightness,
            valence_delta=valence,
            density_delta=density,
            tags_add=list(self.emotion_tags),
        )

    def get_guidetone_action(self) -> Optional[RuleActionGuideTone]:
        """
        Extract guide tone hints from guide_tone_tags and mechanics.

        Phase 2.0: mechanics.guide_tone から直接取得（優先）
        Fallback: guide_tone_tags と mechanics.melody から推定
        """
        if self.domain not in ("guide_tone", "guidetone", "harmony", "melody"):
            return None

        # Phase 2.0: mechanics.guide_tone から直接取得
        gt_mech = self.mechanics.get("guide_tone", {})
        if gt_mech:
            priority = gt_mech.get("priority_tones", [])
            if not priority:
                priority = ["3rd", "7th"]

            return RuleActionGuideTone(
                priority_tones=priority,
                default_register=gt_mech.get("register", "mid"),
                motion=gt_mech.get("motion", "step"),
                phrase_shape=gt_mech.get("phrase_shape"),
                notes_per_bar=gt_mech.get("notes_per_bar"),
            )

        # Fallback: guide_tone_tags と mechanics.melody から推定
        # Extract priority tones from tags
        priority = []
        for tag in self.guide_tone_tags:
            if tag in ("3rd", "7th", "9th", "11th", "13th", "root", "5th"):
                priority.append(tag)

        if not priority:
            # Default: 3rd and 7th (guide tones)
            priority = ["3rd", "7th"]

        # Register from melody contour
        melody = self.mechanics.get("melody", {})
        contour = str(melody.get("contour", "")).lower()
        register = "mid"
        if "高い" in contour or "レンジを上げ" in contour:
            register = "high"
        elif "低め" in contour or "狭い" in contour:
            register = "low"

        # Motion from rhythm description
        rhythm_desc = str(melody.get("rhythm", "")).lower()
        motion = "step"
        if "ジャンプ" in rhythm_desc or "跳躍" in rhythm_desc:
            motion = "leap_ok"
        elif "伸ばす" in rhythm_desc or "長い音価" in rhythm_desc:
            motion = "hold"

        # Notes per bar from rhythm density
        notes_per_bar = None
        if "少なめ" in rhythm_desc or "間を大事" in rhythm_desc:
            notes_per_bar = 0.8
        elif "細かい" in rhythm_desc or "16分" in rhythm_desc:
            notes_per_bar = 2.0

        return RuleActionGuideTone(
            priority_tones=priority,
            default_register=register,
            motion=motion,
            phrase_shape=None,  # Could extract from contour
            notes_per_bar=notes_per_bar,
        )

    def matches(self, ctx: Dict[str, Any]) -> bool:
        """
        Check if this rule matches the given song context.

        Context keys:
        - section: str (e.g. "chorus", "verse")
        - position_in_section: str ("start", "middle", "end")
        - scale_degree: int (1-7)
        - function: str ("tonic", "subdominant", "dominant")
        - chord_symbol: str
        - tempo_bpm: float
        - song_emotion_tags: List[str]
        """
        rule_ctx = self.context

        # Section matching
        sections = rule_ctx.get("sections", [])
        if sections and ctx.get("section") not in sections:
            return False

        # Tempo range
        tempo_range = rule_ctx.get("tempo_bpm_range")
        if tempo_range and len(tempo_range) == 2:
            tempo = ctx.get("tempo_bpm", 120)
            if not (tempo_range[0] <= tempo <= tempo_range[1]):
                return False

        # Emotion tags (OR match)
        song_tags = set(ctx.get("song_emotion_tags", []))
        rule_tags = set(self.emotion_tags)
        if rule_tags and not song_tags.intersection(rule_tags):
            # Allow partial match if no song tags provided
            if song_tags:
                return False

        return True


class Rulebook:
    """Container for composition rules loaded from rulebook.json."""

    def __init__(self, rules: List[Rule], meta: Dict[str, Any]):
        self.rules = rules
        self.meta = meta

    @classmethod
    def load(cls, path: str | Path) -> "Rulebook":
        """Load rulebook from JSON or YAML file."""
        path = Path(path)
        if path.suffix in (".yaml", ".yml"):
            data = yaml.safe_load(path.read_text(encoding="utf-8"))
        else:
            data = json.loads(path.read_text(encoding="utf-8"))
        meta = data.get("meta", {})
        rules_raw = data.get("rules", [])
        rules = [Rule(r) for r in rules_raw]
        return cls(rules, meta)

    # ---- Filter API ----

    def for_domain(self, *domains: str) -> List[Rule]:
        """Get rules matching any of the specified domains."""
        dset = set(domains)
        return [r for r in self.rules if r.domain in dset]

    def for_section(self, section: str) -> List[Rule]:
        """Get rules applicable to a section."""
        return [r for r in self.rules if section in r.context.get("sections", [])]

    def for_emotion_tags(self, *tags: str) -> List[Rule]:
        """Get rules with any of the specified emotion tags."""
        tset = set(tags)
        return [r for r in self.rules if tset.intersection(r.emotion_tags)]

    def query(self, context: Dict[str, Any], domain: str) -> Dict[str, Any]:
        """
        Phase 2.0統一インターフェース：contextからdomain別にアクションをクエリ。

        Args:
            context: 統一context dict（section, role, emotion, lyric, chord_symbol等）
            domain: "emotion" | "guide_tone"

        Returns:
            マージされたaction dict
        """
        matched_rules = self.list_matched_rules(context, domain)

        # Reverse to apply low-specificity first, high-specificity last (most specific wins)
        matched_rules = list(reversed(matched_rules))

        # Merge actions from all matched rules
        merged_actions: dict[str, Any] = {}

        for rule in matched_rules:
            if domain == "emotion":
                action = rule.get_emotion_action()
                if action:
                    # Accumulate deltas
                    merged_actions["energy_delta"] = (
                        merged_actions.get("energy_delta", 0.0) + action.energy_delta
                    )
                    merged_actions["tension_delta"] = (
                        merged_actions.get("tension_delta", 0.0) + action.tension_delta
                    )
                    merged_actions["brightness_delta"] = (
                        merged_actions.get("brightness_delta", 0.0) + action.brightness_delta
                    )
                    merged_actions["valence_delta"] = (
                        merged_actions.get("valence_delta", 0.0) + action.valence_delta
                    )
                    merged_actions["density_delta"] = (
                        merged_actions.get("density_delta", 0.0) + action.density_delta
                    )

                    # Phase 2.0: Scales (last wins - most specific rule)
                    if action.velocity_scale is not None:
                        merged_actions["velocity_scale"] = action.velocity_scale
                    if action.duration_scale is not None:
                        merged_actions["duration_scale"] = action.duration_scale
                    if action.density_scale is not None:
                        merged_actions["density_scale"] = action.density_scale

                    # Tags: union
                    tags = merged_actions.setdefault("tags_add", [])
                    tags.extend(action.tags_add)

            elif domain == "guide_tone":
                action = rule.get_guidetone_action()
                if action:
                    # Priority tones: union
                    priority = merged_actions.setdefault("priority_tones", [])
                    priority.extend(action.priority_tones)

                    # Register: last wins (most specific rule)
                    if action.default_register:
                        merged_actions["register"] = action.default_register

                    # Motion: last wins
                    if action.motion:
                        merged_actions["motion"] = action.motion

                    # Notes per bar: max (allow density increase)
                    if action.notes_per_bar:
                        merged_actions["notes_per_bar"] = max(
                            merged_actions.get("notes_per_bar", 0), action.notes_per_bar
                        )

                    # Phrase shape: last wins
                    if action.phrase_shape:
                        merged_actions["phrase_shape"] = action.phrase_shape

        return merged_actions

    def list_matched_rules(self, context: Dict[str, Any], domain: str) -> List[Rule]:
        """
        Phase 2.0：contextにマッチするルールをリスト（デバッグ用）。

        Args:
            context: 統一context dict
            domain: "emotion" | "guide_tone"

        Returns:
            マッチしたルールのリスト（specificity順）
        """
        # Domain filter
        candidates = [r for r in self.rules if r.domain == domain or r.domain == "harmony"]

        # Context matching
        matched = []
        for rule in candidates:
            if self._matches_context(rule, context):
                matched.append(rule)

        # Sort by specificity
        matched.sort(key=lambda r: self._specificity_score(r, context), reverse=True)

        return matched

    def _matches_context(self, rule: Rule, context: Dict[str, Any]) -> bool:
        """
        Phase 2.0：統一contextとルールのマッチング判定。

        Context keys:
        - section: str
        - role: str ("strings", "piano", "bass")
        - emotion: {"energy": float, "tension": float}
        - lyric: {"phrase_role": str, "stress_level": float, "is_silent": bool}
        - chord_symbol: str
        - key_center: str
        """
        rule_ctx = rule.context

        # Section matching
        sections = rule_ctx.get("sections", [])
        if sections and context.get("section") not in sections:
            return False

        # Role matching (新規)
        roles = rule_ctx.get("roles", [])
        if roles and context.get("role") not in roles:
            return False

        # Emotion threshold (新規)
        if "emotion" in context:
            emo = context["emotion"]
            if "energy_gte" in rule_ctx:
                if emo.get("energy", 0.5) < rule_ctx["energy_gte"]:
                    return False
            if "tension_gte" in rule_ctx:
                if emo.get("tension", 0.5) < rule_ctx["tension_gte"]:
                    return False

        # Lyric phrase role matching (新規)
        if "lyric" in context:
            lyric = context["lyric"]
            phrase_roles = rule_ctx.get("phrase_roles", [])
            if phrase_roles and lyric.get("phrase_role", "none") not in phrase_roles:
                return False

        # Tempo range (既存)
        tempo_range = rule_ctx.get("tempo_bpm_range")
        if tempo_range and len(tempo_range) == 2:
            tempo = context.get("tempo_bpm", 120)
            if not (tempo_range[0] <= tempo <= tempo_range[1]):
                return False

        return True

    def _specificity_score(self, rule: Rule, context: Dict[str, Any]) -> int:
        """ルールの specificity スコア計算（高いほど優先）"""
        score = 0
        rule_ctx = rule.context

        if rule_ctx.get("sections"):
            score += 10
        if rule_ctx.get("roles"):
            score += 8
        if "energy_gte" in rule_ctx or "tension_gte" in rule_ctx:
            score += 5
        if rule_ctx.get("phrase_roles"):
            score += 7
        if rule_ctx.get("tempo_bpm_range"):
            score += 3
        if rule.emotion_tags:
            score += len(rule.emotion_tags)

        return score

    def find_matching(self, ctx: Dict[str, Any], *domains: str) -> List[Rule]:
        """
        Find all rules matching the context, optionally filtered by domain.

        Returns rules sorted by relevance (more specific matches first).
        """
        candidates = self.for_domain(*domains) if domains else self.rules
        matched = [r for r in candidates if r.matches(ctx)]

        # Sort by specificity: rules with more constraints are more specific
        def specificity_score(rule: Rule) -> int:
            score = 0
            if rule.context.get("sections"):
                score += 10
            if rule.context.get("tempo_bpm_range"):
                score += 5
            if rule.emotion_tags:
                score += len(rule.emotion_tags)
            return score

        return sorted(matched, key=specificity_score, reverse=True)
