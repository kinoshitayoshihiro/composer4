#!/usr/bin/env python3
"""
generate_bass_plan_from_harmony.py - harmony_beat.json専用ベース生成

Architecture:
- Slot Planner: bars_with_slots.parquet (always_active: bass は常に演奏)
- Policy YAML: density/pattern_types/articulation (how to fire)
- Chord Source: harmony_beat.json (感情・機能・XMusic統合済み)
- Output: plans/bass_plan.json

Design Philosophy:
"harmony_beat.jsonの感情データを活かして、コントラストのあるベースラインを生成"

Bass is ALWAYS active (always_active: true in policy).
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import yaml

# Import harmony_beat utilities
sys.path.insert(0, str(Path(__file__).parent))
from harmony_utils import load_harmony_beat, get_harmony_chord_at_bar, HarmonyChordEvent
from chordmap_utils import parse_symbol, get_chord_tones
from ai_hook_utils import load_reference_layers
from v2_common import (
    ensure_activity_floor,
    apply_humanize,
    ensure_register,
    select_role,
    extend_last_note_per_bar,  # Phase 3.6
    record_emotion_snapshot,
    summarize_emotion_log,
)

# Optional: RhythmAI
try:
    from otobonAI.rhythm_ai import RhythmAI
except ImportError as exc:  # pragma: no cover - optional dependency
    print(f"⚠️  RhythmAI unavailable for bass plan: {exc}")
    RhythmAI = None  # type: ignore

try:
    from otobonAI.duration_humanize_ai import DurationHumanizeAI
except Exception as exc:  # pragma: no cover - DurationHumanizeAI is required
    raise RuntimeError(
        "DurationHumanizeAI import failed; ensure otobonAI package is installed and configured"
    ) from exc

try:
    from otobonAI.lyric_index import LyricAnchorIndex
    from otobonAI.emotion_ai_v2 import EmotionAI as EmotionAIv2
    from otobonAI.guide_tone_ai_v2 import GuideToneAI as GuideToneAIv2
    from otobonAI.rulebook_engine import Rulebook
except ImportError as exc:  # pragma: no cover - optional dependency
    print(f"⚠️  OtobonAI Phase 2.0 modules unavailable for bass plan: {exc}")
    LyricAnchorIndex = None  # type: ignore
    EmotionAIv2 = None  # type: ignore
    GuideToneAIv2 = None  # type: ignore
    Rulebook = None  # type: ignore

# Bass range: E1 (MIDI 28) - D3 (MIDI 50)
BASS_MIN_PITCH = 28
BASS_MAX_PITCH = 50


def density_bucket_from_value(value: float) -> str:
    """Map numeric density multipliers to RhythmAI-friendly buckets."""

    val = max(0.0, float(value))
    if val <= 0.4:
        return "sparse"
    if val <= 0.75:
        return "medium"
    if val <= 1.2:
        return "dense"
    return "wall"


def resolve_bass_pattern_type(
    rng: np.random.RandomState,
    section_label: str,
    section_density: float,
    pattern_types: Dict[str, float],
    bass_cfg: Dict[str, Any],
    rhythm_ai: Optional[RhythmAI],
    rhythm_vocab_instrument: Optional[str] = None,
    harmony_event: Optional[HarmonyChordEvent] = None,  # ← NEW
) -> tuple[str, Optional[str]]:
    """
    Select a bass pattern type using RhythmAI manifest when available.

    NEW: harmony_event.xmusic_emotion活用
    - "joy"/"pride"/"determination" → 8分系を優先
    - "despair"/"loyalty"/"melancholy" → 4分レガートを優先
    """

    available = list(pattern_types.keys()) or ["root_quarter"]
    probs = np.array([float(pattern_types.get(k, 0.0)) for k in available], dtype=float)

    # NEW: XMusic emotion調整
    if harmony_event and harmony_event.xmusic_emotion:
        emotion = harmony_event.xmusic_emotion.lower()
        if emotion in ("happy", "bright", "intense"):  # joy/pride/determination相当
            # エネルギッシュ → 8分系優先
            pattern_types_adjusted = {"root_eighth": 0.6, "walking": 0.3, "root_quarter": 0.1}
            available = list(pattern_types_adjusted.keys())
            probs = np.array([pattern_types_adjusted[k] for k in available], dtype=float)
        elif emotion in ("dark", "calm", "melancholy"):  # despair/loyalty相当
            # 落ち着き → 4分レガート優先
            pattern_types_adjusted = {"root_quarter": 0.7, "root_eighth": 0.2, "walking": 0.1}
            available = list(pattern_types_adjusted.keys())
            probs = np.array([pattern_types_adjusted[k] for k in available], dtype=float)

    if probs.sum() > 0:
        probs = probs / probs.sum()
        fallback = rng.choice(available, p=probs)
    else:
        fallback = rng.choice(available)

    if rhythm_ai is None or not rhythm_ai.has_manifest():
        return fallback, None

    rhythm_cfg = bass_cfg.get("rhythm_vocab", {}) or {}
    if not rhythm_cfg:
        return fallback, None

    section_overrides = rhythm_cfg.get("sections", {}).get(section_label, {}) or {}
    preferred_ids: List[str] = []
    preferred_ids.extend(rhythm_cfg.get("default_ids", []) or [])
    preferred_ids.extend(section_overrides.get("preferred_ids", []) or [])
    descriptors = section_overrides.get("descriptors") or rhythm_cfg.get("descriptors", []) or []
    density_hint = (
        section_overrides.get("density")
        or rhythm_cfg.get("density")
        or density_bucket_from_value(section_density)
    )

    instrument_key = (rhythm_vocab_instrument or "bass").strip() or "bass"

    entry = rhythm_ai.choose_vocab_entry(
        instrument_key,
        section_label=section_label,
        density_hint=density_hint,
        descriptors=descriptors,
        preferred_ids=preferred_ids,
    )

    if not entry:
        return fallback, None

    pattern_override = (
        entry.ai_hooks.get("bass_pattern_type")
        or entry.ai_hooks.get("pattern_type")
        or section_overrides.get("pattern_type")
    )
    if not pattern_override:
        pattern_map = rhythm_cfg.get("pattern_map", {}) or {}
        pattern_override = pattern_map.get(entry.pattern_ref)

    if pattern_override and str(pattern_override) in pattern_types:
        return str(pattern_override), entry.id

    return fallback, None


def load_bars(bars_path: str) -> pd.DataFrame:
    """Load bars.parquet."""
    bars = pd.read_parquet(bars_path)
    required = ["section_label"]
    missing = [c for c in required if c not in bars.columns]
    if missing:
        raise ValueError(f"bars.parquet missing columns: {missing}")

    # Ensure bar_idx exists
    if "bar_index" in bars.columns and "bar_idx" not in bars.columns:
        bars = bars.rename(columns={"bar_index": "bar_idx"})
    elif "bar_idx" not in bars.columns:
        bars["bar_idx"] = range(len(bars))

    return bars


def get_section_at_bar(harmony: Dict[str, Any], bar_idx: int) -> str:
    """
    harmony_beat.jsonから指定barのsectionラベルを取得。

    Args:
        harmony: load_harmony_beat()の戻り値
        bar_idx: 小節インデックス

    Returns:
        sectionラベル（例: "intro", "verse", "chorus"）
        見つからない場合は"unknown"
    """
    for chord in harmony.get("chords", []):
        # Get the bar and duration from the chord event
        chord_bar = chord.get("bar", 0)
        duration_beats = chord.get("duration_beats", 4.0)
        # Calculate the bar span for this chord
        bar_span = int(np.ceil(duration_beats / 4.0))

        # Check if bar_idx falls within this chord's span
        if chord_bar <= bar_idx < chord_bar + bar_span:
            return chord.get("section", "unknown")

    return "unknown"


def make_bass_pattern(
    bar_idx: int,
    bar_data: pd.Series,
    chord: Dict[str, Any],
    bass_cfg: Dict[str, Any],
    policy: Dict[str, Any],
    section_density: float,
    section_label: str,
    pattern_type: str = "root_quarter",
    emotion_params=None,
    guide_params=None,
    rhythm_pattern_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Generate bass pattern (root foundation).

    Args:
        bar_idx: Bar index
        bar_data: Row from bars.parquet
        chord: Chord from chordmap
        bass_cfg: policy['instruments']['bass']
        policy: Full policy dictionary
        section_density: sections[section_label]['bass']
        section_label: verse, chorus, etc.
        pattern_type: "root_quarter", "root_eighth", "walking"
        emotion_params: Optional EmotionAI parameters (Phase 2.0 hook)
        guide_params: Optional GuideToneAI plan (Phase 2.0 hook)
        rhythm_pattern_id: Optional ID from rhythm_vocab.yaml (Phase 3)

    Returns:
        List of bass events
    """
    start_ql = bar_idx * 4.0
    events = []

    # Parse chord symbol and get tones
    symbol = chord.get("symbol", "C")
    parsed = parse_symbol(symbol)
    chord_tones = get_chord_tones(parsed, bass_octave=2)  # Bass octave 2 (E2=40, C2=36)

    # Use root as bass note
    bass_root = chord_tones[0] if chord_tones else 36  # C2 fallback

    velocity_scale = 1.0
    duration_scale = 1.0
    register_shift = 0
    phrase_velocity_boost = 0

    if emotion_params is not None:
        velocity_scale = float(getattr(emotion_params, "velocity_scale", 1.0) or 1.0)
        duration_scale = float(getattr(emotion_params, "duration_scale", 1.0) or 1.0)

    if guide_params is not None:
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

    if register_shift:
        chord_tones = [note + register_shift for note in chord_tones]
        bass_root = chord_tones[0] if chord_tones else bass_root

    # === Phase 1: Register enforcement (NO open voicing for bass) ===
    section_cfg = policy.get("sections", {}).get(section_label, {})
    _role = select_role(section_cfg, "bass", default_role="root")

    # Enforce register (clamp to bass range) - bass should NOT use tensions
    chord_tones = ensure_register(chord_tones, "bass", policy, section_label)
    # Bass does NOT use open voicing (low register foundation)

    # Update bass_root after register enforcement
    bass_root = chord_tones[0] if chord_tones else 36

    # Humanization
    humanize_ms = bass_cfg.get("humanize_timing_ms", 12)
    humanize_vel = bass_cfg.get("humanize_velocity", 7)
    base_velocity = bass_cfg.get("base_velocity", 80)
    base_velocity = int(np.clip(base_velocity * velocity_scale, 45, 120))
    base_velocity = int(np.clip(base_velocity + phrase_velocity_boost, 40, 120))

    if pattern_type == "root_quarter":
        # Quarter notes on beats 1, 3
        for beat in [0, 2]:
            time_ql = start_ql + beat + np.random.uniform(-humanize_ms / 1000, humanize_ms / 1000)
            vel = base_velocity + np.random.randint(-humanize_vel, humanize_vel)
            duration = 1.0 * duration_scale
            event = {
                "bar_idx": bar_idx,
                "time_ql": float(time_ql),
                "note": int(np.clip(bass_root, BASS_MIN_PITCH, BASS_MAX_PITCH)),
                "velocity": int(np.clip(vel, 60, 110)),
                "duration_ql": float(duration),
                "pattern": pattern_type,
                "event_type": pattern_type,
                "role": "bass",  # ★追加
                "instrument": "bass",  # ★追加
            }
            if rhythm_pattern_id:
                event["rhythm_pattern_id"] = rhythm_pattern_id
            event["section"] = section_label
            event["section_label"] = section_label
            events.append(event)

    elif pattern_type == "root_eighth":
        # Eighth notes (feel-good pop beat)
        for eighth in range(8):
            time_ql = (
                start_ql + eighth * 0.5 + np.random.uniform(-humanize_ms / 1000, humanize_ms / 1000)
            )
            vel = base_velocity - 5 + np.random.randint(-humanize_vel, humanize_vel)
            duration = 0.5 * duration_scale
            event = {
                "bar_idx": bar_idx,
                "time_ql": float(time_ql),
                "note": int(np.clip(bass_root, BASS_MIN_PITCH, BASS_MAX_PITCH)),
                "velocity": int(np.clip(vel, 55, 100)),
                "duration_ql": float(duration),
                "pattern": pattern_type,
                "event_type": pattern_type,
                "role": "bass",  # ★追加
                "instrument": "bass",  # ★追加
            }
            if rhythm_pattern_id:
                event["rhythm_pattern_id"] = rhythm_pattern_id
            event["section"] = section_label
            event["section_label"] = section_label
            events.append(event)

    elif pattern_type == "walking":
        # Walking bass (chorus/bridge uplifting)
        # Use chord tones for walking: root, 3rd, root, 5th
        if len(chord_tones) >= 3:
            walking_notes = [
                chord_tones[0],  # Root
                chord_tones[1],  # 3rd
                chord_tones[0],  # Root
                chord_tones[2] if len(chord_tones) > 2 else chord_tones[0],  # 5th or root
            ]
        else:
            walking_notes = [bass_root] * 4

        for i, note in enumerate(walking_notes):
            time_ql = start_ql + i + np.random.uniform(-humanize_ms / 1000, humanize_ms / 1000)
            vel = base_velocity + np.random.randint(-humanize_vel, humanize_vel)
            duration = 1.0 * duration_scale
            event = {
                "bar_idx": bar_idx,
                "time_ql": float(time_ql),
                "note": int(np.clip(note, BASS_MIN_PITCH, BASS_MAX_PITCH)),
                "velocity": int(np.clip(vel, 65, 105)),
                "duration_ql": float(duration),
                "pattern": pattern_type,
                "event_type": pattern_type,
                "role": "bass",  # ★追加
                "instrument": "bass",  # ★追加
            }
            if rhythm_pattern_id:
                event["rhythm_pattern_id"] = rhythm_pattern_id
            event["section"] = section_label
            event["section_label"] = section_label
            events.append(event)

    return events


def generate_bass_plan(
    bars: pd.DataFrame,
    harmony: Dict[str, Any],  # ← chordmap → harmony
    policy: Dict[str, Any],
    rng_seed: int = 42,
    lyric_index: Optional[LyricAnchorIndex] = None,
    emotion_ai: Optional[EmotionAIv2] = None,
    guidetone_ai: Optional[GuideToneAIv2] = None,
    reference_layers: Optional[Dict[str, Any]] = None,
    rhythm_ai: Optional[RhythmAI] = None,
    rhythm_vocab_instrument: str = "bass",
) -> Dict[str, Any]:
    """
    Main logic: Generate slot-based bass plan from harmony_beat.json.

    Args:
        bars: bars_with_slots.parquet
        harmony: harmony_beat.json (感情・機能・XMusic統合済み)
        policy: policy YAML
        lyric_index: Optional LyricAnchorIndex (Phase 2.0)
        emotion_ai: Optional EmotionAI instance (Phase 2.0)
        guidetone_ai: Optional GuideToneAI instance (Phase 2.0)
        reference_layers: Optional CREPE/Onsets summary for context
        rhythm_ai: Optional RhythmAI instance for rhythm_vocab lookups

    Returns:
        Bass plan JSON
    """

    bass_cfg = policy.get("instruments", {}).get("bass", {})
    sections_density = policy.get("sections", {})
    global_h = policy.get("global", {}).get("humanize", {})
    min_notes = bass_cfg.get("min_notes_per_bar", 1)

    rng = np.random.RandomState(rng_seed)

    pattern_types_cfg = bass_cfg.get("pattern_types", [])
    if isinstance(pattern_types_cfg, list) and pattern_types_cfg:
        pattern_types = {p["type"]: p["probability"] for p in pattern_types_cfg}
    else:
        # CHANGED 2025-11-30: Walking bass を default に (15% → 60%)
        pattern_types = {"walking": 0.60, "root_quarter": 0.30, "root_eighth": 0.10}

    events: List[Dict[str, Any]] = []
    emotion_log: Dict[int, Dict[str, Any]] = {}
    reference_layers = reference_layers or {}
    selected_rhythm_ids: List[str] = []

    for _, bar_row in bars.iterrows():
        bar_idx = int(bar_row["bar_idx"])

        # harmony_beat.jsonからsection取得
        section_label = get_section_at_bar(harmony, bar_idx)
        section_key = section_label.lower()  # ★小文字統一

        bar_start_ql = float(bar_row.get("start_ql", bar_idx * 4.0))
        bar_end_ql = float(bar_row.get("end_ql", (bar_idx + 1) * 4.0))

        # ========== harmony_beat.json対応 ==========
        harmony_event = get_harmony_chord_at_bar(harmony, bar_idx)
        if harmony_event is None:
            continue  # コードなし → スキップ

        chord = {
            "symbol": harmony_event.symbol,
            "function": harmony_event.function,
            "emotion": harmony_event.emotion,
            "xmusic_emotion": harmony_event.xmusic_emotion,
        }
        # ===========================================

        chordpad_pitches = [36, 40, 43, 48]  # C2-C3 bass range

        # ★section_keyでpolicyアクセス
        section_cfg = sections_density.get(section_key, {})
        bass_density = section_cfg.get("bass", 1.0)

        pattern_type, pattern_entry_id = resolve_bass_pattern_type(
            rng,
            section_key,  # ★section_key使用
            bass_density,
            pattern_types,
            bass_cfg,
            rhythm_ai,
            rhythm_vocab_instrument,
            harmony_event,  # ← NEW: harmony_event追加
        )

        lyric_info = lyric_index.get_bar_info(bar_idx) if lyric_index else None

        context = {
            "bar_index": bar_idx,
            "bar": bar_idx,
            "section": section_label,  # 表示用は元の名前保持
            "role": "bass",
            "chord_symbol": chord.get("symbol", "C"),
            # NEW: harmony_beat感情を直接注入
            "harmony_emotion": harmony_event.emotion,
            "xmusic_emotion": harmony_event.xmusic_emotion,
            "valence": harmony_event.valence,
            "tension": harmony_event.tension,
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
                    bass_density *= float(density_scale)
                except (TypeError, ValueError):
                    pass

        original_pattern_type = pattern_type
        if guide_params is not None:
            notes_hint = getattr(guide_params, "notes_per_bar", None)
            if notes_hint is not None:
                available = list(pattern_types.keys())
                if notes_hint >= 6 and "root_eighth" in available:
                    pattern_type = "root_eighth"
                elif notes_hint >= 4 and "walking" in available:
                    pattern_type = "walking"
                elif "root_quarter" in available:
                    pattern_type = "root_quarter"

        if pattern_type != original_pattern_type:
            pattern_entry_id = None

        bar_events = make_bass_pattern(
            bar_idx,
            bar_row,
            chord,
            bass_cfg,
            policy,
            bass_density,
            section_label,
            pattern_type,
            emotion_params=emotion_params,
            guide_params=guide_params,
            rhythm_pattern_id=pattern_entry_id,
        )

        bar_events = ensure_activity_floor(
            bar_events, bar_start_ql, bar_end_ql, min_notes, chordpad_pitches, velocity=85
        )

        events.extend(bar_events)
        if pattern_entry_id:
            selected_rhythm_ids.append(pattern_entry_id)

    events = sorted(events, key=lambda e: e["time_ql"])

    events = apply_humanize(
        events,
        timing_std_ms=global_h.get("timing_std_ms", 8),
        velocity_jitter=global_h.get("velocity_jitter", 6),
        legato_chance=global_h.get("legato_chance", 0.0),
        rng=rng,
    )

    inst_cfg = policy.get("instruments", {}).get("bass", {})
    sustain_cfg = inst_cfg.get("sustain", {})
    min_dur = sustain_cfg.get("min_duration_ql", 1.5)
    bar_span = sustain_cfg.get("max_bar_span", 1)

    events = extend_last_note_per_bar(
        events=events,
        bars_df=bars,
        role="bass",
        min_duration_ql=min_dur,
        max_bar_span=bar_span,
    )

    metadata = {
        "instrument": "bass",
        "num_bars": int(len(bars)),
        "num_events": len(events),
        "always_active": True,
        "generator": "generate_bass_plan_from_harmony.py",
        "sustain_mode": "last_note_per_bar",
        "sustain_config": {"min_duration_ql": min_dur, "max_bar_span": bar_span},
        "ai_hooks": {
            "lyric_anchor": bool(lyric_index),
            "emotion_ai": bool(emotion_ai),
            "guide_tone_ai": bool(guidetone_ai),
            "reference_layers": bool(reference_layers),
            "rhythm_ai_vocab": bool(rhythm_ai and rhythm_ai.has_manifest()),
        },
    }

    if reference_layers:
        metadata["reference_layers"] = reference_layers
    if selected_rhythm_ids:
        metadata["rhythm_vocab_ids"] = sorted(set(selected_rhythm_ids))
    metadata["rhythm_vocab_instrument"] = rhythm_vocab_instrument

    emotion_tracking = summarize_emotion_log(emotion_log)
    if emotion_tracking:
        metadata["emotion_tracking"] = emotion_tracking

    return {"metadata": metadata, "events": events}


def main():
    parser = argparse.ArgumentParser(description="Generate bass plan from harmony_beat.json")
    parser.add_argument("--bars", required=True, help="Path to bars_with_slots.parquet")
    parser.add_argument("--harmony-beat", required=True, help="Path to harmony_beat.json")
    parser.add_argument("--policy", required=True, help="Path to policy YAML")
    parser.add_argument("--out", required=True, help="Output bass_plan.json")
    parser.add_argument("--seed", type=int, help="Random seed for reproducibility")
    parser.add_argument("--lyric-anchors", help="Path to lyric_anchors.json (optional)")
    parser.add_argument("--emotion-profile", help="Path to emotion_profile.json (optional)")
    parser.add_argument("--guide-hints", help="Path to guide_tone_hints.json (optional)")
    parser.add_argument("--rulebook", help="Path to rulebook.yaml (optional)")
    parser.add_argument("--vocal-f0", help="Path to vocal_f0_crepe.parquet (optional)")
    parser.add_argument("--piano-oaf", help="Path to piano_onsets_and_frames.json (optional)")
    parser.add_argument(
        "--groove-vocab",
        help="Optional groove_vocab.parquet path for RhythmAI (defaults to data/groove_vocab.parquet)",
    )
    parser.add_argument(
        "--rhythm-manifest",
        help="Optional rhythm_vocab.yaml path for RhythmAI (defaults to data/rhythm_vocab.yaml)",
    )
    parser.add_argument(
        "--rhythm-vocab",
        default="bass",
        help="Instrument key inside rhythm_vocab manifest to bias selection (default: bass)",
    )
    args = parser.parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)
        print(f"🎲 Random seed set to {args.seed}")

    print(f"📖 Loading bars from {args.bars}")
    bars = load_bars(args.bars)

    print(f"📖 Loading harmony_beat from {args.harmony_beat}")
    harmony = load_harmony_beat(Path(args.harmony_beat))

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

    rhythm_ai = None
    if RhythmAI is not None:
        try:
            groove_vocab_path = Path(args.groove_vocab).expanduser() if args.groove_vocab else None
            rhythm_manifest_path = (
                Path(args.rhythm_manifest).expanduser() if args.rhythm_manifest else None
            )
            rhythm_ai = RhythmAI(
                vocab_path=groove_vocab_path,
                rhythm_manifest_path=rhythm_manifest_path,
            )
            if rhythm_ai.has_manifest():
                print(f"🥁 Rhythm vocab manifest loaded from {rhythm_ai.rhythm_manifest_path}")
        except Exception as exc:  # pragma: no cover
            print(f"⚠️  RhythmAI unavailable at runtime: {exc}")
            rhythm_ai = None

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

    print(f"🎸 Generating bass plan ({len(bars)} bars)")
    plan = generate_bass_plan(
        bars,
        harmony,  # ← chordmap → harmony
        policy,
        lyric_index=lyric_index,
        emotion_ai=emotion_ai,
        guidetone_ai=guidetone_ai,
        reference_layers=reference_layers,
        rhythm_ai=rhythm_ai,
        rhythm_vocab_instrument=args.rhythm_vocab or "bass",
    )

    try:
        duration_ai = DurationHumanizeAI(
            instrument="bass",
            policy=policy,
            tempo_bpm=policy.get("global", {}).get("tempo_bpm", 120),
            rhythm_manifest_path=getattr(rhythm_ai, "rhythm_manifest_path", None),
            vocab_instrument=args.rhythm_vocab or "bass",
        )
        duration_ai.annotate_plan(plan)
        plan.setdefault("metadata", {}).setdefault("ai_hooks", {}).update(
            {"duration_humanize_ai": True}
        )
    except Exception as exc:  # pragma: no cover - diagnostics only
        raise RuntimeError(f"DurationHumanizeAI annotation failed for bass plan: {exc}") from exc

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(plan, f, indent=2, ensure_ascii=False)

    print(f"✅ Bass plan saved to {out_path}")
    print(f"   Events: {plan['metadata']['num_events']}")
    print(f"   Always active: {plan['metadata']['always_active']}")


if __name__ == "__main__":
    main()
