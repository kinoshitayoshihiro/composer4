#!/usr/bin/env python3
"""
generate_guitar_plan_v2.py - Slot-based guitar renderer for fill/riff system.

Architecture:
- Slot Planner: bars_with_slots.parquet (riff_slot: where to fire)
- Policy YAML: density/riff_types/articulation (how to fire)
- Chord Source: chordmap_locked_extended.json (what notes to play)
- Output: plans/guitar_plan.json

Design Philosophy:
"位置決めはbars/sections。造形は楽器別レンダラ。music21は和声支援のみ。"
"""
import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

# Import shared chordmap utilities
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
from chordmap_utils import parse_symbol, get_chord_tones
from ai_hook_utils import load_reference_layers
from v2_common import (
    ensure_register,
    choose_tensions,
    voice_lead_voicing,
    load_humanize_config,
    apply_timing_humanize,
    apply_velocity_humanize,
    apply_rhythm_vocab_annotations,
    record_emotion_snapshot,
    summarize_emotion_log,
)

# Phase 2.0 OtobonAI hooks
try:
    from otobonAI.lyric_index import LyricAnchorIndex
    from otobonAI.emotion_ai_v2 import EmotionAI as EmotionAIv2
    from otobonAI.guide_tone_ai_v2 import GuideToneAI as GuideToneAIv2
    from otobonAI.rulebook_engine import Rulebook

    OTOBON_AI_AVAILABLE = True
except ImportError as e:  # pragma: no cover - optional dependency
    OTOBON_AI_AVAILABLE = False
    print(f"⚠️  OtobonAI Phase 2.0 modules not available: {e}")
    LyricAnchorIndex = None  # type: ignore
    EmotionAIv2 = None  # type: ignore
    GuideToneAIv2 = None  # type: ignore
    Rulebook = None  # type: ignore

try:
    from otobonAI.duration_humanize_ai import DurationHumanizeAI
except Exception as exc:  # pragma: no cover - optional dependency
    print(f"⚠️  DurationHumanizeAI unavailable: {exc}")
    DurationHumanizeAI = None  # type: ignore

# Continue module (Stage3 / RhythmAI)
try:  # pragma: no cover - optional dependency validated via integration tests
    from continue_module import (
        ContinueModule,
        load_stage3,
        build_rhythm_ai,
        load_events as continue_load_events,
    )
except Exception as exc:  # pragma: no cover - degrade gracefully when unavailable
    ContinueModule = None  # type: ignore

    def load_stage3(_path):  # type: ignore
        raise RuntimeError("continue_module import failed, Stage3 data unavailable")

    def build_rhythm_ai(*_args, **_kwargs):  # type: ignore
        return None

    def continue_load_events(_path):  # type: ignore
        raise RuntimeError("continue_module import failed, motif loading unavailable")


# Guitar range: E2 (MIDI 40) - B5 (MIDI 83)
GUITAR_MIN_PITCH = 40
GUITAR_MAX_PITCH = 83
BEATS_PER_BAR = 4.0


@dataclass
class ContinueSettings:
    enabled: bool = False
    section_overrides: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    default_source_bars: int = 1
    default_target_bars: int = 8
    stage3_path: Optional[str] = None
    stage3_loop_id: Optional[str] = None
    rhythm_manifest: Optional[str] = None
    groove_vocab: Optional[str] = None
    motif_path: Optional[str] = None
    require_riff_slot: bool = True
    beats_per_bar: float = BEATS_PER_BAR
    seed: int = 42

    def section_params(self, label: str) -> Dict[str, int]:
        params = self.section_overrides.get(label.lower(), {})
        return {
            "source_bars": int(params.get("source_bars", self.default_source_bars)),
            "target_bars": int(params.get("target_bars", self.default_target_bars)),
        }

    def enabled_sections(self) -> List[str]:
        return sorted(self.section_overrides.keys())


class GuitarContinueController:
    def __init__(self, settings: ContinueSettings):
        self.settings = settings
        self._module: Optional[ContinueModule] = None
        self._external_motif: Optional[List[Dict[str, Any]]] = None
        if settings.motif_path:
            self._external_motif = self._load_external_motif(Path(settings.motif_path))

    # ------------------------------------------------------------------ public
    def apply(
        self, events: List[Dict[str, Any]], bars_df: pd.DataFrame
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        if not self.settings.enabled or not self.settings.section_overrides:
            return events, {"applied": False}

        sequences = self._collect_sequences(bars_df)
        if not sequences:
            return events, {"applied": False}

        skip_ranges: List[Tuple[int, int]] = []
        injected_events: List[Dict[str, Any]] = []
        segments_meta: List[Dict[str, Any]] = []

        for seq in sequences:
            bars = seq["bars"]
            section_key = seq["section_label"]
            section_label = seq.get("section_label_display", section_key)
            section_params = self.settings.section_params(section_key)
            source_bars = max(1, section_params["source_bars"])
            target_default = max(source_bars + 1, section_params["target_bars"])
            idx = 0
            while idx + source_bars < len(bars):
                remaining = len(bars) - idx
                block_len = min(target_default, remaining)
                if block_len <= source_bars:
                    break
                block_start_bar = bars[idx]
                block_end_bar = block_start_bar + block_len
                motif_events = self._select_motif(events, block_start_bar, source_bars)
                if not motif_events:
                    break
                continue_events, cond_meta = self._run_continue(
                    motif_events=motif_events,
                    section_label=section_label,
                    source_bars=source_bars,
                    target_bars=block_len,
                    block_start_bar=block_start_bar,
                )
                if not continue_events:
                    break
                skip_ranges.append((block_start_bar, block_end_bar))
                injected_events.extend(continue_events)
                segments_meta.append(
                    {
                        "section": section_label,
                        "start_bar": block_start_bar,
                        "bars": block_len,
                        "conditions": cond_meta,
                        "motif_source": "external" if self._external_motif else "plan",
                    }
                )
                idx += block_len

        if not injected_events:
            return events, {"applied": False}

        filtered_events: List[Dict[str, Any]] = []
        for event in events:
            bar_idx = self._event_bar(event)
            if any(start <= bar_idx < end for start, end in skip_ranges):
                continue
            filtered_events.append(event)
        filtered_events.extend(injected_events)
        return filtered_events, {
            "applied": True,
            "segments": segments_meta,
            "sections": sorted({seg["section"] for seg in segments_meta}),
        }

    # ------------------------------------------------------------------ helpers
    def _collect_sequences(self, bars_df: pd.DataFrame) -> List[Dict[str, Any]]:
        sequences: List[Dict[str, Any]] = []
        current: List[int] = []
        current_label: Optional[str] = None
        current_display: Optional[str] = None
        for row in bars_df.itertuples():
            label = str(getattr(row, "section_label", ""))
            key = label.lower()
            riff_slot = bool(getattr(row, "riff_slot", 0))
            eligible = key in self.settings.section_overrides
            if self.settings.require_riff_slot:
                eligible = eligible and riff_slot
            if not eligible:
                if current:
                    self._flush_sequence(sequences, current, current_label, current_display)
                    current, current_label, current_display = [], None, None
                continue
            if current and (row.bar_idx != current[-1] + 1 or key != current_label):
                self._flush_sequence(sequences, current, current_label, current_display)
                current = []
            current.append(int(row.bar_idx))
            current_label = key
            current_display = label
        if current:
            self._flush_sequence(sequences, current, current_label, current_display)
        return sequences

    def _flush_sequence(
        self,
        sequences: List[Dict[str, Any]],
        bars: List[int],
        label: Optional[str],
        label_display: Optional[str],
    ) -> None:
        if not bars or not label:
            return
        min_required = max(2, self.settings.section_params(label)["source_bars"] + 1)
        if len(bars) < min_required:
            return
        sequences.append(
            {
                "bars": bars[:],
                "section_label": label,
                "section_label_display": label_display or label,
            }
        )

    def _select_motif(
        self, events: List[Dict[str, Any]], start_bar: int, source_bars: int
    ) -> List[Dict[str, Any]]:
        if self._external_motif:
            return [dict(evt) for evt in self._external_motif]

        lower = start_bar
        upper = start_bar + source_bars
        offset = lower * self.settings.beats_per_bar
        motif: List[Dict[str, Any]] = []
        fallback: List[Dict[str, Any]] = []
        for event in events:
            bar_idx = self._event_bar(event)
            if lower <= bar_idx < upper:
                cloned = {
                    "time_ql": float(event.get("time_ql", 0.0) - offset),
                    "duration_ql": float(event.get("duration_ql", 0.25)),
                    "velocity": int(event.get("velocity", 80)),
                }
                if event.get("is_riff") or "riff" in str(event.get("event_type", "")):
                    motif.append(cloned)
                else:
                    fallback.append(cloned)
        return motif or fallback

    def _run_continue(
        self,
        *,
        motif_events: List[Dict[str, Any]],
        section_label: str,
        source_bars: int,
        target_bars: int,
        block_start_bar: int,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        module = self._ensure_module()
        if module is None:
            return [], []
        try:
            result = module.extend(
                motif_events,
                source_bars=source_bars,
                target_bars=target_bars,
                instrument="guitar",
                section_label=section_label,
            )
        except ValueError:
            return [], []

        offset = block_start_bar * self.settings.beats_per_bar
        shifted: List[Dict[str, Any]] = []
        for evt in result.get("events", []):
            cloned = dict(evt)
            cloned["time_ql"] = float(evt.get("time_ql", 0.0) + offset)
            cloned["duration_ql"] = float(evt.get("duration_ql", 0.25))
            cloned["bar_idx"] = self._event_bar(cloned)
            cloned["section_label"] = section_label
            cloned["event_type"] = "continue_riff"
            cloned["is_riff"] = True
            shifted.append(cloned)
        return shifted, result.get("metadata", {}).get("stage3_conditions", [])

    def _ensure_module(self) -> Optional[ContinueModule]:
        if self._module is not None:
            return self._module
        if ContinueModule is None:
            return None

        stage3_df = pd.DataFrame()
        if self.settings.stage3_path:
            stage3_df = load_stage3(Path(self.settings.stage3_path).expanduser())
        rhythm_ai = build_rhythm_ai(
            Path(self.settings.groove_vocab).expanduser() if self.settings.groove_vocab else None,
            (
                Path(self.settings.rhythm_manifest).expanduser()
                if self.settings.rhythm_manifest
                else None
            ),
        )

        self._module = ContinueModule(
            rhythm_ai=rhythm_ai,
            stage3_df=stage3_df,
            stage3_loop_id=self.settings.stage3_loop_id,
            beats_per_bar=self.settings.beats_per_bar,
            seed=self.settings.seed,
        )
        return self._module

    def _event_bar(self, event: Dict[str, Any]) -> int:
        bar_idx = event.get("bar_idx")
        if bar_idx is not None:
            try:
                return int(bar_idx)
            except (TypeError, ValueError):
                pass
        time_ql = float(event.get("time_ql", 0.0))
        return int(time_ql // self.settings.beats_per_bar)

    def _load_external_motif(self, path: Path) -> List[Dict[str, Any]]:
        if not path.exists():
            raise FileNotFoundError(f"Continue motif file not found: {path}")
        payload = continue_load_events(path)
        motif: List[Dict[str, Any]] = []
        for evt in payload:
            motif.append(
                {
                    "time_ql": float(evt.get("time_ql", 0.0)),
                    "duration_ql": float(evt.get("duration_ql", 0.25)),
                    "velocity": int(evt.get("velocity", 80)),
                }
            )
        return motif


def _parse_sections_value(value: Any) -> Dict[str, Dict[str, Any]]:
    if not value:
        return {}
    if isinstance(value, dict):
        return {str(k).lower(): dict(v or {}) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return {str(item).lower(): {} for item in value}
    if isinstance(value, str):
        return {value.lower(): {}}
    return {}


def resolve_continue_settings(
    guitar_cfg: Dict[str, Any],
    policy_sections: Dict[str, Any],
    overrides: Optional[Dict[str, Any]] = None,
) -> ContinueSettings:
    cfg = guitar_cfg.get("continue", {}) if isinstance(guitar_cfg, dict) else {}
    overrides = overrides or {}

    enabled = bool(cfg.get("enabled", False))
    if overrides.get("force_disable"):
        enabled = False
    if overrides.get("force_enable"):
        enabled = True

    section_map = _parse_sections_value(cfg.get("sections"))
    for sec_name, sec_cfg in (policy_sections or {}).items():
        cont_cfg = sec_cfg.get("guitar_continue") if isinstance(sec_cfg, dict) else None
        if not cont_cfg:
            continue
        if isinstance(cont_cfg, dict):
            section_map[sec_name.lower()] = dict(cont_cfg)
        elif bool(cont_cfg):
            section_map.setdefault(sec_name.lower(), {})

    if overrides.get("sections"):
        section_map = {name.lower(): {} for name in overrides["sections"]}

    default_source = int(overrides.get("source_bars") or cfg.get("source_bars", 1))
    default_target = int(overrides.get("target_bars") or cfg.get("target_bars", 8))

    settings = ContinueSettings(
        enabled=enabled and bool(section_map),
        section_overrides=section_map,
        default_source_bars=max(1, default_source),
        default_target_bars=max(2, default_target),
        stage3_path=overrides.get("stage3_path") or cfg.get("stage3_conditions"),
        stage3_loop_id=overrides.get("stage3_loop_id") or cfg.get("stage3_loop_id"),
        rhythm_manifest=overrides.get("rhythm_manifest") or cfg.get("rhythm_manifest"),
        groove_vocab=overrides.get("groove_vocab") or cfg.get("groove_vocab"),
        motif_path=overrides.get("motif_path") or cfg.get("motif_path"),
        require_riff_slot=bool(cfg.get("require_riff_slot", True)),
        seed=int(overrides.get("seed") or cfg.get("seed", 42)),
    )

    if overrides.get("require_riff_slot") is not None:
        settings.require_riff_slot = bool(overrides["require_riff_slot"])

    if settings.stage3_path is None:
        default_stage3 = Path("outputs/stage3/conditions.parquet")
        if default_stage3.exists():
            settings.stage3_path = str(default_stage3)

    return settings


def load_bars(bars_path: str) -> pd.DataFrame:
    """Load bars.parquet with riff_slot."""
    bars = pd.read_parquet(bars_path)
    required = ["section_label"]
    missing = [c for c in required if c not in bars.columns]
    if missing:
        raise ValueError(f"bars.parquet missing columns: {missing}")

    if "riff_slot" not in bars.columns:
        bars["riff_slot"] = 0
        print("ℹ️  bars file missing riff_slot column, defaulting to 0")

    # Ensure bar_index exists (rename if needed)
    if "bar_index" in bars.columns and "bar_idx" not in bars.columns:
        bars = bars.rename(columns={"bar_index": "bar_idx"})
    elif "bar_idx" not in bars.columns:
        bars["bar_idx"] = range(len(bars))

    return bars


def load_sections(sections_path: str) -> List[Dict[str, Any]]:
    """Load sections.json (handle both list and {\"sections\": [...]})."""
    with open(sections_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    elif isinstance(data, dict) and "sections" in data:
        return data["sections"]
    else:
        raise ValueError('sections.json must be a list or {"sections": [...]}')


def build_section_lookup(sections: List[Dict[str, Any]]) -> Dict[int, str]:
    """Return bar_index -> section_label mapping derived from sections.json."""
    lookup: Dict[int, str] = {}
    for sec in sections:
        start = int(sec.get("start_bar", sec.get("bar_start", 0)))
        end = int(sec.get("end_bar", sec.get("bar_end", start)))
        label = sec.get("label", "verse")
        for bar in range(start, end):
            lookup[bar] = label
    return lookup


def load_chordmap(chordmap_path: str) -> List[Dict[str, Any]]:
    """Load chordmap_locked_extended.json."""
    with open(chordmap_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    elif isinstance(data, dict) and "events" in data:
        return data["events"]
    else:
        raise ValueError('chordmap must be a list or {"events": [...]}')


def get_chord_at_bar(chordmap: List[Dict[str, Any]], bar_idx: int) -> Dict[str, Any]:
    """Find chord event overlapping with bar_idx."""
    bar_start_ql = bar_idx * 4.0
    for chord in chordmap:
        chord_start = chord.get("time_ql", 0.0)
        chord_end = chord_start + chord.get("duration_ql", 4.0)
        if chord_start <= bar_start_ql < chord_end:
            return chord
    # Default to first chord if not found
    return chordmap[0] if chordmap else {}


def make_riff_pattern(
    bar_idx: int,
    bar_data: pd.Series,
    chord: Dict[str, Any],
    guitar_cfg: Dict[str, Any],
    policy: Dict[str, Any],
    section_label: str,
    riff_type: str = "strum",
    n_notes: int = 4,
    use_extensions: bool = False,
    prev_voicing: List[int] = None,
    emotion_params=None,
    guide_params=None,
) -> List[Dict[str, Any]]:
    """
    Generate guitar riff pattern with role-based voicing and register enforcement.

    Phase 2 Enhancement:
        - Tension adoption (choose_tensions)
        - Voice-leading optimization (voice_lead_voicing)

    Args:
        bar_idx: Bar index
        bar_data: Row from bars.parquet
        chord: Chord from chordmap
        guitar_cfg: policy['instruments']['guitar']
        policy: Full policy dictionary
        section_label: Section name (verse, chorus, etc.)
        riff_type: "strum", "broken_chord", "single_note"
        n_notes: Number of notes to use
        use_extensions: Whether to use tension notes
        prev_voicing: Previous voicing for voice-leading (Phase 2)
        emotion_params: EmotionAI params for this bar (Phase 2.0 hook)
        guide_params: GuideToneAI plan for this bar (Phase 2.0 hook)

    Returns:
        Tuple of (events, final_voicing) for Phase 2 voice-leading tracking
    """
    start_ql = bar_idx * 4.0
    events = []
    emotion_log: Dict[int, Dict[str, Any]] = {}
    emotion_log: Dict[int, Dict[str, Any]] = {}

    velocity_scale = 1.0
    duration_scale = 1.0
    density_hint: Optional[int] = None
    register_shift = 0
    phrase_velocity_boost = 0

    if emotion_params is not None:
        velocity_scale = float(getattr(emotion_params, "velocity_scale", 1.0) or 1.0)
        duration_scale = float(getattr(emotion_params, "duration_scale", 1.0) or 1.0)

    if guide_params is not None:
        notes_hint = getattr(guide_params, "notes_per_bar", None)
        if notes_hint is not None:
            try:
                density_hint = max(1, int(notes_hint))
            except (TypeError, ValueError):
                density_hint = None

        register_pref = str(getattr(guide_params, "register", "")).lower()
        if register_pref == "high":
            register_shift = 12
        elif register_pref == "low":
            register_shift = -12

        phrase_shape = getattr(guide_params, "phrase_shape", None)
        if phrase_shape == "uphill":
            phrase_velocity_boost = 8
        elif phrase_shape == "downhill":
            phrase_velocity_boost = -6

    # Parse chord symbol
    symbol = chord.get("symbol", "C")
    parsed = parse_symbol(symbol)

    # === Phase 2 改修: テンション"ピン留め"フロー ===
    # 1) 基本トーン取得 (triad/7th)
    base_tones = get_chord_tones(parsed, bass_octave=5)  # C4 baseline
    if not base_tones:
        base_tones = [60, 64, 67]  # C major fallback

    # 2) テンション選択（確率＆mode）→ ピン留めリスト
    section_cfg = policy.get("sections", {}).get(section_label, {})
    tension_mode = section_cfg.get("tension_mode", "none")

    pinned_tensions = []
    if use_extensions and tension_mode != "none":
        allow_tensions = guitar_cfg.get("tensions", {}).get("allow", [9, 11, 13])
        tensions = choose_tensions(symbol, allow_tensions, tension_mode)

        if tensions:
            root_pc = parsed.get("root_midi", 60) % 12
            reg_cfg = policy.get("instruments", {}).get("guitar", {}).get("register", {})
            reg_pref = reg_cfg.get("octave_prefer", 55)

            for t in tensions:
                # Tension pitch class
                tension_pc = (root_pc + t) % 12
                # Preferred octave近くに配置
                tension_note = tension_pc + ((reg_pref // 12) * 12)
                pinned_tensions.append(tension_note)
                print(f"[DEBUG] Pinned tension {t} -> MIDI {tension_note} (PC {tension_pc})")

    # 3) 優先度付きマージ（pinned は必ず残す順）
    # 重複除去しつつ順序保持: pinned_tensions + base_tones
    all_tones = []
    seen = set()
    for t in pinned_tensions + base_tones:
        if t not in seen:
            all_tones.append(t)
            seen.add(t)

    pinned_mask = [True] * len(pinned_tensions) + [False] * (len(all_tones) - len(pinned_tensions))

    # 4) 音数決定：pinned を確保しつつ上限まで拡張
    max_per_bar = guitar_cfg.get("max_notes_per_bar", 8)
    min_per_bar = guitar_cfg.get("min_notes_per_bar", 1)
    # section density から target 計算
    section_density = section_cfg.get("density", 0.6)
    target_n = max(min_per_bar, int(section_density * max_per_bar))
    if density_hint is not None:
        target_n = max(min_per_bar, min(max_per_bar, density_hint))
    # pinned を収容できない場合は target_n を引き上げ（上限で頭打ち）
    target_n = min(max_per_bar, max(target_n, len(pinned_tensions)))

    # スライス（上限まで）
    chord_tones = all_tones[:target_n]
    pinned_mask = pinned_mask[:target_n]

    print(
        f"[DEBUG] Final selection: {len(chord_tones)} notes ({len([p for p in pinned_mask if p])} pinned)"
    )

    if register_shift:
        all_tones = [note + register_shift for note in all_tones]

    # 5) レンジ適合（折り返し）
    chord_tones = ensure_register(
        chord_tones, "guitar", policy, section_label, pinned_mask=pinned_mask
    )

    # === Phase 2: Voice-leading (NEW - redesigned to preserve tensions) ===
    if prev_voicing:
        reg_min = guitar_cfg.get("register", {}).get("min", GUITAR_MIN_PITCH)
        reg_max = guitar_cfg.get("register", {}).get("max", GUITAR_MAX_PITCH)

        print(f"[DEBUG] Before voice_lead_voicing: {sorted(chord_tones)}")
        chord_tones = voice_lead_voicing(
            chord_tones=chord_tones,
            prev_voicing=prev_voicing,
            reg_min=reg_min,
            reg_max=reg_max,
            max_step=7,  # Allow up to P5 movement
        )
        print(f"[DEBUG] After voice_lead_voicing: {sorted(chord_tones)}")

    # 6) イベント生成（is_tension メタデータ付与）
    # Humanization
    humanize_ms = guitar_cfg.get("humanize_timing_ms", 10)
    humanize_vel = guitar_cfg.get("humanize_velocity", 6)
    base_velocity = guitar_cfg.get("base_velocity", 85)
    base_velocity = int(np.clip(base_velocity * velocity_scale, 40, 120))
    base_velocity = int(np.clip(base_velocity + phrase_velocity_boost, 35, 120))

    if riff_type == "strum":
        # Downstroke strum (8th note feel)
        for eighth in [0, 2]:  # Beat 1, 3
            time_ql = start_ql + eighth + np.random.uniform(-humanize_ms / 1000, humanize_ms / 1000)
            # Strum all chord tones (no slicing - already limited by target_n)
            for i, note in enumerate(chord_tones):
                note_clipped = int(np.clip(note, GUITAR_MIN_PITCH, GUITAR_MAX_PITCH))
                vel = base_velocity + np.random.randint(-humanize_vel, humanize_vel)
                # Stagger timing (0-30ms)
                stagger = i * 0.02

                event = {
                    "bar_idx": bar_idx,
                    "time_ql": float(time_ql + stagger),
                    "note": note_clipped,
                    "velocity": int(np.clip(vel, 60, 110)),
                    "duration_ql": 0.5,
                    "type": f"riff_{riff_type}",
                    "pattern": riff_type,
                    "is_riff": True,
                    "event_type": "riff",
                }
                # Phase 2: Mark tensions for QAgate
                if i < len(pinned_mask) and pinned_mask[i]:
                    event["is_tension"] = True
                events.append(event)

    elif riff_type == "broken_chord":
        # Arpeggio (16th note ascending) - all chord_tones
        for i, note in enumerate(chord_tones):
            time_ql = (
                start_ql + i * 0.25 + np.random.uniform(-humanize_ms / 1000, humanize_ms / 1000)
            )
            note_clipped = int(np.clip(note, GUITAR_MIN_PITCH, GUITAR_MAX_PITCH))
            vel = base_velocity + np.random.randint(-humanize_vel, humanize_vel)
            duration = 0.25 * duration_scale

            event = {
                "bar_idx": bar_idx,
                "time_ql": float(time_ql),
                "note": note_clipped,
                "velocity": int(np.clip(vel, 55, 110)),
                "duration_ql": float(duration),
                "type": f"riff_{riff_type}",
                "pattern": riff_type,
                "is_riff": True,
                "event_type": "riff",
            }
            # Phase 2: Mark tensions for QAgate
            if i < len(pinned_mask) and pinned_mask[i]:
                event["is_tension"] = True
            events.append(event)

    elif riff_type == "single_note":
        # Power chord (root + fifth, palm mute feel)
        root = chord_tones[0]
        fifth = chord_tones[2] if len(chord_tones) > 2 else root + 7
        power_chord = [root, fifth]
        for beat in [0, 1, 2, 3]:
            time_ql = start_ql + beat + np.random.uniform(-humanize_ms / 1000, humanize_ms / 1000)
            for j, note in enumerate(power_chord):
                note_clipped = int(np.clip(note, GUITAR_MIN_PITCH, GUITAR_MAX_PITCH))
                vel = base_velocity - 10 + np.random.randint(-humanize_vel, humanize_vel)
                duration = 0.5 * duration_scale

                event = {
                    "bar_idx": bar_idx,
                    "time_ql": float(time_ql),
                    "note": note_clipped,
                    "velocity": int(np.clip(vel, 50, 105)),
                    "duration_ql": float(duration),
                    "type": f"riff_{riff_type}",
                    "pattern": riff_type,
                    "is_riff": True,
                    "event_type": "riff",
                }
                # Phase 2: Mark tensions (root usually isn't, fifth rarely)
                # Map power_chord[j] back to original chord_tones index
                original_idx = 0 if j == 0 else (2 if len(chord_tones) > 2 else -1)
                if (
                    original_idx >= 0
                    and original_idx < len(pinned_mask)
                    and pinned_mask[original_idx]
                ):
                    event["is_tension"] = True
                events.append(event)

    # Phase 2: Return (events, final_voicing) for voice-leading tracking
    # Use all chord_tones (already limited to target_n)
    final_voicing = sorted(chord_tones)
    return (events, final_voicing)


def ensure_activity_floor(
    bar_idx: int,
    bar_events: List[Dict[str, Any]],
    chord: Dict[str, Any],
    guitar_cfg: Dict[str, Any],
    min_notes: int,
) -> List[Dict[str, Any]]:
    """
    If bar_events is too sparse, add sustained chordpad notes to meet min_notes.
    """
    # count events in this bar (simple count)
    existing_notes = len(bar_events)
    if existing_notes >= min_notes:
        return bar_events

    need = max(0, min_notes - existing_notes)
    parsed = parse_symbol(chord.get("symbol", "C"))
    chord_tones = get_chord_tones(parsed, bass_octave=4)
    if not chord_tones:
        chord_tones = [60, 64, 67]

    start_ql = bar_idx * 4.0
    base_velocity = guitar_cfg.get("base_velocity", 80)
    for i in range(need):
        note = int(np.clip(chord_tones[i % len(chord_tones)], GUITAR_MIN_PITCH, GUITAR_MAX_PITCH))
        bar_events.append(
            {
                "bar_idx": bar_idx,
                "time_ql": float(start_ql + 0.0 + 0.001 * i),
                "note": note,
                "velocity": int(np.clip(base_velocity - 5, 60, 110)),
                "duration_ql": 4.0,
                "type": "chordpad_floor",
                "pattern": "sustain",
                "event_type": "pad",
            }
        )

    return bar_events


def enforce_bar_note_cap(
    bar_events: List[Dict[str, Any]], max_notes: int | None
) -> List[Dict[str, Any]]:
    """Trim low-priority events if a bar exceeds the policy max."""
    if not bar_events or not max_notes or max_notes <= 0:
        return bar_events

    if len(bar_events) <= max_notes:
        return bar_events

    priority: List[Dict[str, Any]] = []
    support: List[Dict[str, Any]] = []
    floor: List[Dict[str, Any]] = []

    for ev in sorted(bar_events, key=lambda e: e.get("time_ql", 0.0)):
        ev_type = str(ev.get("type", "")).lower()
        if ev.get("is_riff"):
            priority.append(ev)
        elif "floor" in ev_type:
            floor.append(ev)
        else:
            support.append(ev)

    kept: List[Dict[str, Any]] = []
    for bucket in (priority, support, floor):
        if len(kept) >= max_notes:
            break
        space = max_notes - len(kept)
        kept.extend(bucket[:space])

    return sorted(kept, key=lambda e: e.get("time_ql", 0.0))


def apply_humanize(
    events: List[Dict[str, Any]],
    timing_std_ms: float = 8.0,
    vel_jitter: float = 6.0,
    legato_chance: float = 0.15,
) -> List[Dict[str, Any]]:
    """Apply lightweight humanization (timing jitter and velocity jitter)."""
    for e in events:
        # jitter in beats: timing_std_ms ms -> beats (assuming quarter note = 1 beat and 60s BPM not known here). Use ms->seconds->approx beats via 0.5s per beat at 120bpm -> rough.
        jitter_s = np.random.normal(0, timing_std_ms / 1000.0)
        # approximate beats by dividing by 0.5 (approx 120bpm) - rough but acceptable for small jitters
        jitter_beats = jitter_s / 0.5
        e["time_ql"] = float(e.get("time_ql", 0.0) + jitter_beats)
        if "velocity" in e:
            e["velocity"] = int(
                np.clip(
                    e["velocity"] + np.random.randint(-int(vel_jitter), int(vel_jitter) + 1),
                    40,
                    127,
                )
            )
        # legato: extend duration slightly
        if np.random.random() < legato_chance:
            e["duration_ql"] = e.get("duration_ql", 0.25) * 1.15
    return events


def make_comping_pattern(
    bar_idx: int,
    bar_data: pd.Series,
    chord: Dict[str, Any],
    guitar_cfg: Dict[str, Any],
    section_density: float,
    policy: Dict[str, Any],
    section_label: str,
    use_extensions: bool,
    emotion_params=None,
    guide_params=None,
) -> List[Dict[str, Any]]:
    """
    Generate guitar comping (offbeat chords).

    Args:
        bar_idx: Bar index
        bar_data: Row from bars.parquet
        chord: Chord from chordmap
        guitar_cfg: policy['instruments']['guitar']
        section_density: sections[section_label]['guitar']
        emotion_params: EmotionAI params for this bar (Phase 2.0 hook)
        guide_params: GuideToneAI plan for this bar (Phase 2.0 hook)

    Returns:
        List of guitar events
    """
    start_ql = bar_idx * 4.0
    events = []

    # Phase 2.0 hooks (EmotionAI / GuideToneAI influence comping dynamics)
    velocity_scale = 1.0
    duration_scale = 1.0
    density_scale = 1.0
    register_shift = 0
    phrase_velocity_boost = 0

    if emotion_params is not None:
        velocity_scale = float(getattr(emotion_params, "velocity_scale", 1.0) or 1.0)
        duration_scale = float(getattr(emotion_params, "duration_scale", 1.0) or 1.0)
        density_scale = float(getattr(emotion_params, "density_scale", 1.0) or 1.0)

    if guide_params is not None:
        notes_hint = getattr(guide_params, "notes_per_bar", None)
        if notes_hint is not None:
            try:
                relative = float(notes_hint) / max(1.0, guitar_cfg.get("min_notes_per_bar", 2))
                density_scale *= float(np.clip(relative, 0.4, 1.8))
            except (TypeError, ValueError):
                pass

        register_pref = str(getattr(guide_params, "register", "")).lower()
        if register_pref == "high":
            register_shift = 12
        elif register_pref == "low":
            register_shift = -12

        phrase_shape = getattr(guide_params, "phrase_shape", None)
        if phrase_shape == "uphill":
            phrase_velocity_boost = 6
        elif phrase_shape == "downhill":
            phrase_velocity_boost = -6

    # Check if comping is active
    guitar_active = bar_data.get("guitar_active", 1.0)
    if pd.isna(guitar_active):
        guitar_active = 1.0

    if guitar_active < 0.3:
        return []

    # Effective density (Emotion/Guide hooks adjust density floor)
    effective_density = float(section_density * guitar_active * density_scale)

    if effective_density < 0.2:
        return []

    # Parse chord symbol and get tones
    symbol = chord.get("symbol", "C")
    parsed = parse_symbol(symbol)
    chord_tones = get_chord_tones(parsed, bass_octave=4)  # Guitar octave 4

    if not chord_tones:
        chord_tones = [60, 64, 67]  # C major fallback

    section_cfg = policy.get("sections", {}).get(section_label, {})
    tension_mode = section_cfg.get("tension_mode", "none")
    pinned_tensions: List[int] = []
    if use_extensions and tension_mode != "none":
        allow_tensions = guitar_cfg.get("tensions", {}).get("allow", [9, 11])
        tensions = choose_tensions(symbol, allow_tensions, tension_mode)
        reg_pref = guitar_cfg.get("register", {}).get("octave_prefer", 55)
        root_midi = parsed.get("root_midi", 48)
        interval_map = {9: 14, 11: 17, 13: 21}
        for t in tensions:
            note = root_midi + interval_map.get(t, 14)
            while note < GUITAR_MIN_PITCH:
                note += 12
            while note > GUITAR_MAX_PITCH:
                note -= 12
            if note < reg_pref - 7:
                note += 12
            elif note > reg_pref + 7:
                note -= 12
            pinned_tensions.append(int(note))

    voicing_notes = chord_tones[:3]
    is_tension_flags = [False] * len(voicing_notes)
    if use_extensions and pinned_tensions:
        if voicing_notes:
            voicing_notes[-1] = pinned_tensions[0]
            is_tension_flags[-1] = True
        else:
            voicing_notes = [pinned_tensions[0]]
            is_tension_flags = [True]

        for idx, tension_note in enumerate(pinned_tensions[1:], start=0):
            if idx < len(voicing_notes):
                voicing_notes[idx] = tension_note
                is_tension_flags[idx] = True

    if register_shift:
        voicing_notes = [note + register_shift for note in voicing_notes]

    # Humanization
    humanize_ms = guitar_cfg.get("humanize_timing_ms", 10)
    humanize_vel = guitar_cfg.get("humanize_velocity", 6)
    base_velocity = guitar_cfg.get("base_velocity", 85)
    base_velocity = int(np.clip(base_velocity * velocity_scale, 40, 120))
    base_velocity = int(np.clip(base_velocity + phrase_velocity_boost, 35, 120))

    # Offbeat chords (beats 2, 4)
    comp_prob = min(effective_density, 1.0)
    for beat in [1, 3]:  # Offbeat (2, 4)
        if np.random.random() < comp_prob:
            time_ql = start_ql + beat + np.random.uniform(-humanize_ms / 1000, humanize_ms / 1000)
            # Play top 3 notes (voicing)
            for idx, note in enumerate(voicing_notes):
                note_clipped = int(np.clip(note, GUITAR_MIN_PITCH, GUITAR_MAX_PITCH))
                vel = base_velocity + np.random.randint(-humanize_vel, humanize_vel)
                duration = 0.5 * duration_scale
                event = {
                    "bar_idx": bar_idx,
                    "time_ql": float(time_ql),
                    "note": note_clipped,
                    "velocity": int(np.clip(vel, 55, 115)),
                    "duration_ql": float(duration),
                    "pattern": "offbeat_chord",
                    "event_type": "comping",
                }
                if idx < len(is_tension_flags) and is_tension_flags[idx]:
                    event["is_tension"] = True
                events.append(event)

    return events


def generate_guitar_plan(
    bars: pd.DataFrame,
    sections: List[Dict[str, Any]],
    chordmap: List[Dict[str, Any]],
    policy: Dict[str, Any],
    lyric_index: Optional[LyricAnchorIndex] = None,
    emotion_ai: Optional[EmotionAIv2] = None,
    guidetone_ai: Optional[GuideToneAIv2] = None,
    reference_layers: Optional[Dict[str, Any]] = None,
    continue_overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Main logic: Generate slot-based guitar plan.

    Args:
        bars: bars_with_slots.parquet
        sections: sections.json
        chordmap: chordmap_locked_extended.json
        policy: policy YAML
        lyric_index: Optional LyricAnchorIndex (Phase 2.0)
        emotion_ai: Optional EmotionAI instance (Phase 2.0)
        guidetone_ai: Optional GuideToneAI instance (Phase 2.0)
        reference_layers: Optional CREPE/Onsets reference summary

    Returns:
        Guitar plan JSON
    """
    guitar_cfg = policy.get("instruments", {}).get("guitar", {})
    sections_density = policy.get("sections", {})
    riff_on_sections = guitar_cfg.get("riff_on_sections", ["pre_chorus", "chorus", "bridge"])
    if "verse" not in riff_on_sections:
        riff_on_sections.append("verse")
    section_lookup = build_section_lookup(sections)
    slots_cfg = policy.get("slots", {})
    riff_auto_prob = float(slots_cfg.get("riff_default", 0.5))

    continue_settings = resolve_continue_settings(guitar_cfg, sections_density, continue_overrides)
    continue_controller: Optional[GuitarContinueController] = None
    continue_report: Dict[str, Any] = {"applied": False}
    if continue_settings.enabled:
        continue_controller = GuitarContinueController(continue_settings)

    # Riff type distribution
    riff_types_cfg = guitar_cfg.get("riff_types", [])
    if isinstance(riff_types_cfg, list) and len(riff_types_cfg) > 0:
        riff_types = {rt["type"]: rt["probability"] for rt in riff_types_cfg}
    else:
        riff_types = {"strum": 0.5, "broken_chord": 0.3, "single_note": 0.2}

    events = []
    prev_voicing = None  # Phase 2: Track voicing for voice-leading
    reference_layers = reference_layers or {}
    emotion_log: Dict[int, Dict[str, Any]] = {}

    for _, bar_row in bars.iterrows():
        bar_idx = int(bar_row["bar_idx"])
        section_label = section_lookup.get(bar_idx, bar_row.get("section_label", "verse"))
        section_label = str(section_label)
        section_lower = section_label.lower()
        riff_slot = bar_row.get("riff_slot", False)

        # Get chord
        chord = get_chord_at_bar(chordmap, bar_idx)

        # Get section density
        section_cfg = sections_density.get(section_label, {})
        guitar_density = section_cfg.get("guitar", 0.5)

        # Phase 2.0: Build AI context
        lyric_info = None
        if lyric_index:
            lyric_info = lyric_index.get_bar_info(bar_idx)

        context = {
            "bar_index": bar_idx,
            "bar": bar_idx,
            "section": section_label,
            "role": "guitar",
            "chord_symbol": chord.get("symbol", "C"),
            "slots": {"riff": bool(riff_slot)},
        }

        if lyric_info and lyric_info.get("has_anchor"):
            context["lyric"] = {
                "phrase_role": lyric_info["phrase_role"],
                "stress_level": lyric_info.get("stress_level", 0.0),
                "is_silent": lyric_info.get("is_silent", False),
            }

        if reference_layers:
            context["reference_layers"] = reference_layers

        emotion_params = None
        if emotion_ai:
            try:
                emotion_params = emotion_ai.get_params(context)
                record_emotion_snapshot(
                    emotion_log,
                    bar_idx=bar_idx,
                    section_label=section_label,
                    emotion_params=emotion_params,
                )
            except Exception as exc:
                print(f"⚠️  EmotionAI v2 error at bar {bar_idx}: {exc}")

        guide_params = None
        if guidetone_ai:
            try:
                guide_params = guidetone_ai.get_plan(context)
            except Exception as exc:
                print(f"⚠️  GuideToneAI v2 error at bar {bar_idx}: {exc}")

        if emotion_params is not None:
            density_scale = getattr(emotion_params, "density_scale", None)
            if density_scale is not None:
                try:
                    guitar_density *= float(density_scale)
                except (TypeError, ValueError):
                    pass

        # Determine tension adoption for this section (from policy sections, not sections.json)
        policy_sections = policy.get("sections", {})
        t_ratio = (
            float(policy_sections.get(section_label, {}).get("tension_ratio", 0.0))
            if isinstance(policy_sections, dict)
            else 0.0
        )
        use_ext = (t_ratio > 0.0) or (np.random.random() < t_ratio)
        print(
            f"[DEBUG-MAIN] Bar {bar_idx}: section={section_label}, t_ratio={t_ratio:.2f}, use_ext={use_ext}"
        )

        # Decide number of notes based on section density and instrument config
        d = (
            float(policy_sections.get(section_label, {}).get("density", 0.6))
            if isinstance(policy_sections, dict)
            else 0.6
        )
        min_np = guitar_cfg.get("min_notes_per_bar", 3)
        max_np = guitar_cfg.get("max_notes_per_bar", 12)
        n_notes = max(min_np, int(round(d * max_np)))

        slot_forced = bool(riff_slot)
        priority_section = section_label in riff_on_sections or "chorus" in section_lower
        auto_riff = False
        if priority_section:
            auto_riff = "chorus" in section_lower or (np.random.random() < riff_auto_prob)
        should_riff = slot_forced or auto_riff

        # Decision: Riff or Comping
        if should_riff:
            # Fire riff (Phase 2: returns tuple)
            riff_type = np.random.choice(list(riff_types.keys()), p=list(riff_types.values()))
            result = make_riff_pattern(
                bar_idx,
                bar_row,
                chord,
                guitar_cfg,
                policy,  # Added for Phase 1
                section_label,  # Added for Phase 1
                riff_type,
                n_notes=n_notes,
                use_extensions=use_ext,
                prev_voicing=prev_voicing,  # Phase 2: pass previous voicing
                emotion_params=emotion_params,
                guide_params=guide_params,
            )
            # Phase 2: Unpack tuple
            bar_events, prev_voicing = result
        else:
            # Comping
            bar_events = make_comping_pattern(
                bar_idx,
                bar_row,
                chord,
                guitar_cfg,
                guitar_density,
                policy,
                section_label,
                use_ext,
                emotion_params=emotion_params,
                guide_params=guide_params,
            )

        # Ensure activity floor
        bar_events = ensure_activity_floor(
            bar_idx, bar_events, chord, guitar_cfg, min_notes=guitar_cfg.get("min_notes_per_bar", 3)
        )

        bar_events = enforce_bar_note_cap(bar_events, guitar_cfg.get("max_notes_per_bar", 8))

        # Phase 3: Add metadata for humanize (beat_in_bar, section_label)
        bar_start_ql = bar_idx * 4.0
        for ev in bar_events:
            if "bar_idx" not in ev:
                ev["bar_idx"] = bar_idx
            if "beat_in_bar" not in ev:
                ev["beat_in_bar"] = (ev.get("time_ql", 0.0) - bar_start_ql) / 1.0
            if "section_label" not in ev:
                ev["section_label"] = section_label

        events.extend(bar_events)

    if continue_controller:
        events, continue_report = continue_controller.apply(events, bars)
        for ev in events:
            if "beat_in_bar" not in ev:
                bar_idx = int(ev.get("bar_idx", 0))
                bar_start_ql = bar_idx * BEATS_PER_BAR
                ev["beat_in_bar"] = (ev.get("time_ql", 0.0) - bar_start_ql) / 1.0

    # Sort by time
    events = sorted(events, key=lambda e: e["time_ql"]) if events else events

    # Phase 3: Apply humanization (section-aware)
    # Group events by section for section-specific humanize
    from itertools import groupby

    humanized_events = []
    tempo_bpm = policy.get("global", {}).get("tempo_bpm", 120.0)

    for section_label, section_events in groupby(
        events, key=lambda e: e.get("section_label", "verse")
    ):
        section_events = list(section_events)
        humanize_cfg = load_humanize_config(policy, section_label, "guitar")

        section_events = apply_timing_humanize(section_events, humanize_cfg, tempo_bpm)
        section_events = apply_velocity_humanize(section_events, humanize_cfg, base_velocity=80)

        humanized_events.extend(section_events)

    events = sorted(humanized_events, key=lambda e: e["time_ql"])

    # Metadata
    metadata = {
        "instrument": "guitar",
        "num_bars": int(len(bars)),
        "num_events": len(events),
        "riff_slots_used": int(bars["riff_slot"].sum()),
        "generator": "generate_guitar_plan_v2.py",
        "humanize_profile": policy.get("humanize", {}).get("profile", "pop_easy"),
        "ai_hooks": {
            "lyric_anchor": bool(lyric_index),
            "emotion_ai": bool(emotion_ai),
            "guide_tone_ai": bool(guidetone_ai),
            "reference_layers": bool(reference_layers),
        },
    }

    if continue_settings.enabled or continue_report.get("applied"):
        metadata["continue"] = {
            "enabled": continue_settings.enabled,
            "sections": continue_settings.enabled_sections(),
            "applied": continue_report.get("applied", False),
            "segments": continue_report.get("segments", []),
            "stage3_path": continue_settings.stage3_path,
            "stage3_loop_id": continue_settings.stage3_loop_id,
        }

    if reference_layers:
        metadata["reference_layers"] = reference_layers

    emotion_tracking = summarize_emotion_log(emotion_log)
    if emotion_tracking:
        metadata["emotion_tracking"] = emotion_tracking

    return {"metadata": metadata, "events": events}


def main():
    parser = argparse.ArgumentParser(description="Generate guitar plan (slot-based V2)")
    parser.add_argument("--bars", required=True, help="Path to bars_with_slots.parquet")
    parser.add_argument("--sections", required=True, help="Path to sections.json")
    parser.add_argument("--chordmap", required=True, help="Path to chordmap_locked_extended.json")
    parser.add_argument("--policy", required=True, help="Path to policy YAML")
    parser.add_argument("--out", required=True, help="Output guitar_plan.json")
    parser.add_argument("--seed", type=int, help="Random seed for reproducibility")
    parser.add_argument("--lyric-anchors", help="Path to lyric_anchors.json (optional)")
    parser.add_argument("--emotion-profile", help="Path to emotion_profile.json (optional)")
    parser.add_argument("--guide-hints", help="Path to guide_tone_hints.json (optional)")
    parser.add_argument("--rulebook", help="Path to rulebook.yaml (optional)")
    parser.add_argument("--vocal-f0", help="Path to vocal_f0_crepe.parquet (optional)")
    parser.add_argument("--piano-oaf", help="Path to piano_onsets_and_frames.json (optional)")
    parser.add_argument(
        "--rhythm-manifest",
        help="Override rhythm_vocab manifest path (default: data/rhythm_vocab.yaml)",
    )
    parser.add_argument(
        "--continue-enable",
        action="store_true",
        help="Force-enable Continue (Stage3) riff overrides regardless of policy",
    )
    parser.add_argument(
        "--continue-disable",
        action="store_true",
        help="Force-disable Continue overrides even if policy enables them",
    )
    parser.add_argument(
        "--continue-sections",
        help="Comma-separated list of section labels to feed into Continue",
    )
    parser.add_argument("--continue-stage3", help="Path to Stage3 conditions file override")
    parser.add_argument("--continue-loop-id", help="Stage3 loop_id override for Continue")
    parser.add_argument("--continue-groove", help="Groove vocab parquet override for Continue")
    parser.add_argument("--continue-manifest", help="Rhythm manifest override for Continue")
    parser.add_argument(
        "--continue-motif",
        help="Path to Continue motif events (JSON exported from Continue module)",
    )
    parser.add_argument(
        "--continue-source-bars",
        type=int,
        help="How many bars of motif to capture before handing off to Continue",
    )
    parser.add_argument(
        "--continue-target-bars",
        type=int,
        help="Number of bars Continue should render per block",
    )
    parser.add_argument("--continue-seed", type=int, help="RNG seed for Continue module")
    parser.add_argument(
        "--continue-allow-non-slot",
        action="store_true",
        help="Allow Continue overrides even when riff_slot is 0",
    )
    args = parser.parse_args()

    # Set random seed for reproducibility
    if args.seed is not None:
        np.random.seed(args.seed)
        print(f"🎲 Random seed set to {args.seed}")

    print(f"📖 Loading bars from {args.bars}")
    bars = load_bars(args.bars)

    print(f"📖 Loading sections from {args.sections}")
    sections = load_sections(args.sections)

    print(f"📖 Loading chordmap from {args.chordmap}")
    chordmap = load_chordmap(args.chordmap)

    print(f"📖 Loading policy from {args.policy}")
    with open(args.policy, "r", encoding="utf-8") as f:
        policy = yaml.safe_load(f)

    reference_layers = load_reference_layers(args.vocal_f0, args.piano_oaf)
    if reference_layers:
        print(
            "🔗 Reference layers loaded:",
            ", ".join(
                f"{k}={v.get('frames', v.get('notes', 0))}" for k, v in reference_layers.items()
            ),
        )

    rulebook = None
    lyric_index = None
    emotion_ai = None
    guidetone_ai = None

    if args.rulebook and Rulebook:
        print(f"📖 Loading Rulebook from {args.rulebook}")
        rulebook = Rulebook.load(Path(args.rulebook))

    if args.lyric_anchors and LyricAnchorIndex:
        print(f"📝 Loading LyricAnchorIndex from {args.lyric_anchors}")
        with open(args.lyric_anchors, "r", encoding="utf-8") as f:
            anchors_data = json.load(f)
        tempo_bpm = policy.get("global", {}).get("tempo_bpm", 120)
        lyric_index = LyricAnchorIndex(anchors=anchors_data, tempo_bpm=tempo_bpm)

    if args.emotion_profile and EmotionAIv2 and rulebook:
        print(f"🎭 Loading EmotionAI v2 from {args.emotion_profile}")
        with open(args.emotion_profile, "r", encoding="utf-8") as f:
            emotion_profile_data = json.load(f)
        emotion_ai = EmotionAIv2(emotion_profile=emotion_profile_data, rulebook=rulebook)

    if args.guide_hints and GuideToneAIv2 and rulebook:
        print(f"🎵 Loading GuideToneAI v2 from {args.guide_hints}")
        with open(args.guide_hints, "r", encoding="utf-8") as f:
            guide_tone_data = json.load(f)
        guidetone_ai = GuideToneAIv2(guide_tone_hints=guide_tone_data, rulebook=rulebook)

    continue_overrides: Dict[str, Any] = {}
    if args.continue_enable:
        continue_overrides["force_enable"] = True
    if args.continue_disable:
        continue_overrides["force_disable"] = True
    if args.continue_sections:
        sections = [seg.strip() for seg in args.continue_sections.split(",") if seg.strip()]
        if sections:
            continue_overrides["sections"] = sections
    if args.continue_stage3:
        continue_overrides["stage3_path"] = args.continue_stage3
    if args.continue_loop_id:
        continue_overrides["stage3_loop_id"] = args.continue_loop_id
    if args.continue_groove:
        continue_overrides["groove_vocab"] = args.continue_groove
    if args.continue_manifest:
        continue_overrides["rhythm_manifest"] = args.continue_manifest
    if args.continue_motif:
        continue_overrides["motif_path"] = args.continue_motif
    if args.continue_source_bars is not None:
        continue_overrides["source_bars"] = args.continue_source_bars
    if args.continue_target_bars is not None:
        continue_overrides["target_bars"] = args.continue_target_bars
    if args.continue_seed is not None:
        continue_overrides["seed"] = args.continue_seed
    if args.continue_allow_non_slot:
        continue_overrides["require_riff_slot"] = False

    print(f"🎸 Generating guitar plan ({len(bars)} bars)")
    plan = generate_guitar_plan(
        bars,
        sections,
        chordmap,
        policy,
        lyric_index=lyric_index,
        emotion_ai=emotion_ai,
        guidetone_ai=guidetone_ai,
        reference_layers=reference_layers,
        continue_overrides=continue_overrides,
    )

    plan_meta = plan.setdefault("metadata", {})
    default_section = plan_meta.get("default_section") or policy.get("metadata", {}).get(
        "default_section"
    )
    song_id = plan_meta.get("song_id") or policy.get("metadata", {}).get("song_id")
    rhythm_stats = apply_rhythm_vocab_annotations(
        plan.get("events", []),
        instrument="guitar",
        policy=policy,
        default_section=default_section,
        song_id=song_id,
    )
    if rhythm_stats.get("assigned", 0) > 0:
        plan_meta["rhythm_vocab_ids"] = rhythm_stats["used_ids"]
        plan_meta["rhythm_vocab_instrument"] = "guitar"
        plan_meta.setdefault("ai_hooks", {}).update({"rhythm_vocab_policy": True})

    rhythm_manifest_path = Path(args.rhythm_manifest).expanduser() if args.rhythm_manifest else None
    if DurationHumanizeAI is not None:
        try:
            duration_ai = DurationHumanizeAI(
                instrument="guitar",
                policy=policy,
                tempo_bpm=policy.get("global", {}).get("tempo_bpm", 120),
                rhythm_manifest_path=rhythm_manifest_path,
                vocab_instrument="guitar",
            )
            duration_ai.annotate_plan(plan)
            plan_meta.setdefault("ai_hooks", {}).update({"duration_humanize_ai": True})
        except Exception as exc:
            print(f"⚠️  DurationHumanizeAI annotation skipped: {exc}")

    # Save
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(plan, f, indent=2, ensure_ascii=False)

    print(f"✅ Guitar plan saved to {out_path}")
    print(f"   Events: {plan['metadata']['num_events']}")
    print(f"   Riff slots used: {plan['metadata']['riff_slots_used']}/{len(bars)}")


if __name__ == "__main__":
    main()
