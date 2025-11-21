# --- START OF FILE utilities/humanizer.py (役割特化版) ---
# ruff: noqa
import copy
import logging
import math
import random
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

logger = logging.getLogger("otokotoba.humanizer")

try:
    from cyext.humanize import (
        apply_swing as cy_apply_swing,
        humanize_velocities as cy_humanize_velocities,
        apply_envelope as cy_apply_envelope,
        apply_velocity_histogram as cy_apply_velocity_histogram,
    )
except Exception as exc:  # pragma: no cover - optional
    logger.info(
        "Cython extensions not available (%s). Install Cython and reinstall for speed.",
        exc,
    )
    cy_apply_swing = None
    cy_humanize_velocities = None
    cy_apply_envelope = None
    cy_apply_velocity_histogram = None

from .ghost_jitter import apply_ghost_jitter  # re-export

# music21 のサブモジュールを正しい形式でインポート
import music21.chord as m21chord  # check_imports.py の期待する形式 (スペースに注意)
import music21.expressions as expressions
import music21.instrument as instrument
import music21.key as key
import music21.meter as meter
import music21.note as note
import music21.stream as stream
import music21.tempo as tempo
import music21.volume as volume
from music21 import exceptions21

__all__ = [
    "apply_ghost_jitter",
    "apply_velocity_histogram",
    "apply_velocity_histogram_profile",
    "get_velocity_histogram",
    "VELO_PROFILES",
    "apply_envelope",
    "swing_offset",
    "load_event_type_profiles",
    "load_event_type_profiles_from_yaml",
]

USE_EXPR_CC11 = False
USE_AFTERTOUCH = False


def set_cc_flags(use_expr_cc11: bool, use_aftertouch: bool) -> None:
    """Configure global CC mapping flags."""
    global USE_EXPR_CC11, USE_AFTERTOUCH
    USE_EXPR_CC11 = bool(use_expr_cc11)
    USE_AFTERTOUCH = bool(use_aftertouch)


# MIN_NOTE_DURATION_QL は core_music_utils からインポートすることを推奨
try:
    from .core_music_utils import MIN_NOTE_DURATION_QL
except ImportError:
    MIN_NOTE_DURATION_QL = 0.125

try:
    import quantize  # type: ignore
except Exception:  # pragma: no cover - optional dependency

    class _DummyQuantize:
        def setSwingRatio(self, ratio: float) -> None:
            pass

    quantize = _DummyQuantize()


class _QuantizeConfig:
    def __init__(self) -> None:
        self.swing_ratio = 0.5

    def setSwingRatio(self, ratio: float) -> None:
        self.swing_ratio = float(ratio)


# quantizeモジュールの代替として常にこのインスタンスを使う
quantize = _QuantizeConfig()

# 既存の関数があれば残しつつ、下記を追記 -----------------------
# ------------------------------------------------------------
# 1) グローバルプロファイル・レジストリ
# ------------------------------------------------------------
_PROFILE_REGISTRY: dict[str, dict[str, Any]] = {}
PROFILES = _PROFILE_REGISTRY


_EventTypeTables = tuple[dict[str, str], dict[str, dict[str, Any]], dict[str, Any]]
_EVENT_TYPE_TABLES: _EventTypeTables = ({}, {}, {})


def _normalize_event_type_profile(data: Mapping[str, Any]) -> dict[str, Any]:
    profile: dict[str, Any] = {}
    per_inst_raw = data.get("per_instrument")
    for field_name, value in data.items():
        if field_name == "per_instrument":
            continue
        profile[field_name] = value
    if isinstance(per_inst_raw, Mapping):
        profile["per_instrument"] = {
            str(inst).lower(): dict(inst_cfg)
            for inst, inst_cfg in per_inst_raw.items()
            if isinstance(inst_cfg, Mapping)
        }
    else:
        profile["per_instrument"] = {}
    return profile


def _extract_event_type_section(raw: Mapping[str, Any] | None) -> Mapping[str, Any] | None:
    if not isinstance(raw, Mapping):
        return None
    section: Mapping[str, Any] | None = raw
    if "humanize" in raw and isinstance(raw["humanize"], Mapping):
        humanize_section = raw["humanize"]
        section = humanize_section.get("event_types")
    elif "event_types" in raw and isinstance(raw["event_types"], Mapping):
        section = raw["event_types"]
    return section if isinstance(section, Mapping) else None


def _normalize_event_type_config(
    raw: Mapping[str, Any] | None,
) -> _EventTypeTables:
    section = _extract_event_type_section(raw)
    if section is None:
        return ({}, {}, {})
    aliases: dict[str, str] = {}
    profiles: dict[str, dict[str, Any]] = {}
    default_profile: dict[str, Any] = {}
    for name, data in section.items():
        if name == "aliases" and isinstance(data, Mapping):
            aliases = {str(alias).lower(): str(target).lower() for alias, target in data.items()}
            continue
        if not isinstance(data, Mapping):
            continue
        normalized = _normalize_event_type_profile(data)
        key = str(name).lower()
        if key == "default":
            default_profile = normalized
        else:
            profiles[key] = normalized
    return (aliases, profiles, default_profile)


def load_event_type_profiles(config: Mapping[str, Any] | None) -> None:
    """Register event-type humanization profiles from a mapping."""
    global _EVENT_TYPE_TABLES
    _EVENT_TYPE_TABLES = _normalize_event_type_config(config)


def load_event_type_profiles_from_yaml(path: str | Path) -> None:
    """Load event-type profiles from a YAML file."""
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(cfg_path)
    import yaml  # local import to avoid mandatory dependency at import time

    data = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    if not isinstance(data, Mapping):
        load_event_type_profiles(None)
        return
    load_event_type_profiles(data)


def _auto_load_default_event_types() -> None:
    if any(_EVENT_TYPE_TABLES):
        return
    cfg_path = Path(__file__).resolve().parents[1] / "configs" / "plan_humanize.yaml"
    if not cfg_path.exists():
        return
    try:
        load_event_type_profiles_from_yaml(cfg_path)
    except Exception as exc:  # pragma: no cover - best effort only
        logger.debug("Failed to auto-load event type profiles: %s", exc)


_auto_load_default_event_types()


def _resolve_event_type_tables(
    global_settings: Mapping[str, Any] | None,
) -> _EventTypeTables:
    if global_settings and hasattr(global_settings, "get"):
        for key_name in ("humanize_event_types", "event_type_profiles", "event_types"):
            candidate = global_settings.get(key_name)
            if isinstance(candidate, Mapping):
                return _normalize_event_type_config(candidate)
    return _EVENT_TYPE_TABLES


def _prepare_profile_for_instrument(
    raw_profile: Mapping[str, Any] | None,
    instrument_family: str,
) -> dict[str, Any]:
    if not isinstance(raw_profile, Mapping):
        return {}
    flat: dict[str, Any] = {
        key: value for key, value in raw_profile.items() if key != "per_instrument"
    }
    per_inst = raw_profile.get("per_instrument")
    if isinstance(per_inst, Mapping):
        inst_cfg = per_inst.get(instrument_family)
        if isinstance(inst_cfg, Mapping):
            flat.update(inst_cfg)
    return flat


def _resolve_event_type_profile(
    event_name: str,
    instrument_family: str,
    tables: _EventTypeTables,
) -> dict[str, Any]:
    aliases, profiles, default_profile = tables
    resolved = aliases.get(event_name, event_name)
    profile: dict[str, Any] = {}
    if default_profile:
        profile.update(_prepare_profile_for_instrument(default_profile, instrument_family))
    specific = profiles.get(resolved)
    if specific:
        profile.update(_prepare_profile_for_instrument(specific, instrument_family))
    return profile


def _ms_to_quarter_length(ms: float, tempo_bpm: float) -> float:
    if tempo_bpm <= 0:
        return 0.0
    return (ms / 1000.0) * (tempo_bpm / 60.0)


def _infer_instrument_family(part: stream.Part) -> str:
    candidates: list[str | None] = [getattr(part, "partName", None), getattr(part, "id", None)]
    inst = part.getInstrument(returnDefault=False)
    if inst is not None:
        best_name = inst.bestName() if hasattr(inst, "bestName") else None
        candidates.extend(
            [
                getattr(inst, "partName", None),
                getattr(inst, "instrumentName", None),
                best_name,
            ]
        )

    def _family_from_name(name: str | None) -> str | None:
        if not name:
            return None
        lower = name.lower()
        if "bass" in lower:
            return "bass"
        if "guitar" in lower:
            return "guitar"
        if any(token in lower for token in ("piano", "keys", "keyboard")):
            return "piano"
        if any(token in lower for token in ("drum", "percussion")):
            return "drums"
        if any(token in lower for token in ("string", "violin", "cello")):
            return "strings"
        if any(token in lower for token in ("vocal", "voice")):
            return "vocal"
        return None

    for cand in candidates:
        family = _family_from_name(cand)
        if family:
            return family
    return "default"


def _resolve_tempo_bpm(part: stream.Part, global_settings: Mapping[str, Any] | None) -> float:
    mark = part.recurse().getElementsByClass(tempo.MetronomeMark).first()
    if mark and mark.number:
        try:
            return float(mark.number)
        except Exception:  # pragma: no cover - defensive only
            pass
    if global_settings and hasattr(global_settings, "get"):
        bpm = global_settings.get("tempo_bpm")
        if bpm is not None:
            try:
                return float(bpm)
            except Exception:  # pragma: no cover - defensive only
                pass
    return 120.0


def _strip_wrapper(text: str) -> str:
    stripped = text
    while len(stripped) > 2 and stripped[0] in "[{(" and stripped[-1] in "]})":
        stripped = stripped[1:-1].strip()
    return stripped


def _normalize_event_type_value(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, (list, tuple, set)):
        for item in value:
            normalized = _normalize_event_type_value(item)
            if normalized:
                return normalized
        return None
    text = str(value).strip()
    if not text:
        return None
    text = _strip_wrapper(text)
    lowered = text.lower()
    for sep in (":", "="):
        for prefix in ("event_type", "event", "evt"):
            token = f"{prefix}{sep}"
            if lowered.startswith(token):
                lowered = lowered[len(token) :].strip()
                break
        else:
            continue
        break
    lowered = _strip_wrapper(lowered)
    return lowered or None


def _extract_event_type_tag(element: note.NotRest | m21chord.Chord) -> str | None:
    candidates: list[Any] = []
    for attr_name in ("humanize_event_type", "event_type", "eventType"):
        candidates.append(getattr(element, attr_name, None))
    editorial = getattr(element, "editorial", None)
    if editorial is not None:
        for attr_name in ("humanize_event_type", "event_type", "eventType", "comment"):
            candidates.append(getattr(editorial, attr_name, None))
        misc = getattr(editorial, "misc", None)
        if isinstance(misc, Mapping):
            candidates.append(misc.get("event_type"))
            candidates.append(misc.get("humanize_event_type"))
    lyric = getattr(element, "lyric", None)
    if lyric is not None:
        candidates.append(lyric)
    lyrics = getattr(element, "lyrics", None)
    if isinstance(lyrics, list):
        for lyr in lyrics:
            if hasattr(lyr, "text"):
                candidates.append(getattr(lyr, "text", None))
            else:
                candidates.append(lyr)
    exprs = getattr(element, "expressions", None)
    if exprs:
        for expr in exprs:
            if isinstance(expr, expressions.TextExpression):
                candidates.append(expr.content)
    for candidate in candidates:
        normalized = _normalize_event_type_value(candidate)
        if normalized:
            return normalized
    return None


def _apply_event_type_adjustments(
    element: note.Note | m21chord.Chord,
    profile: Mapping[str, Any],
    tempo_bpm: float,
) -> None:
    if not profile:
        return
    timing_jitter_ms = float(profile.get("timing_jitter_ms", 0.0) or 0.0)
    timing_push_ms = float(profile.get("timing_push_ms", 0.0) or 0.0)
    duration_scale = float(profile.get("duration_scale", 1.0) or 1.0)
    velocity_jitter = int(profile.get("velocity_jitter", 0) or 0)
    velocity_bias = int(profile.get("velocity_bias", 0) or 0)

    offset_delta = 0.0
    if timing_jitter_ms:
        offset_delta += _ms_to_quarter_length(
            random.uniform(-timing_jitter_ms, timing_jitter_ms), tempo_bpm
        )
    if timing_push_ms:
        offset_delta += _ms_to_quarter_length(timing_push_ms, tempo_bpm)
    if offset_delta:
        new_offset = max(0.0, float(element.offset) + offset_delta)
        element.offset = new_offset

    if duration_scale and element.duration:
        try:
            new_q_len = max(
                MIN_NOTE_DURATION_QL / 8,
                float(element.duration.quarterLength) * duration_scale,
            )
            element.duration.quarterLength = new_q_len
        except exceptions21.DurationException:
            pass

    if velocity_jitter or velocity_bias:
        targets: list[note.Note] = []
        if isinstance(element, m21chord.Chord):
            targets.extend([n for n in element.notes if isinstance(n, note.Note)])
        elif isinstance(element, note.Note):
            targets.append(element)
        for tgt in targets:
            if tgt.volume is None:
                tgt.volume = volume.Volume(velocity=64)
            vel = tgt.volume.velocity or 64
            if velocity_jitter:
                vel += random.randint(-velocity_jitter, velocity_jitter)
            vel += velocity_bias
            tgt.volume.velocity = max(1, min(127, vel))


def _apply_event_type_profile_to_element(
    element: note.Note | m21chord.Chord,
    tables: _EventTypeTables,
    instrument_family: str,
    tempo_bpm: float,
) -> None:
    aliases, profiles, default_profile = tables
    if not aliases and not profiles and not default_profile:
        return
    tag = _extract_event_type_tag(element)
    if tag is None and not default_profile:
        return
    profile = _resolve_event_type_profile(tag or "default", instrument_family, tables)
    if profile:
        _apply_event_type_adjustments(element, profile, tempo_bpm)


def load_profiles(dict_like: dict[str, Any]) -> None:
    """
    YAML から読み取った humanize_profiles セクションを登録する。
    例: cfg['humanize_profiles'] をそのまま渡す。
    """
    global _PROFILE_REGISTRY
    _PROFILE_REGISTRY = dict_like
    logger.info(f"Loaded {len(_PROFILE_REGISTRY)} humanize profiles")


def get_profile(name: str) -> dict[str, Any]:
    if name in _PROFILE_REGISTRY:
        return _PROFILE_REGISTRY[name]
    raise KeyError(f"Humanize profile '{name}' not found.")


# ------------------------------------------------------------
# 2) 適用関数
# ------------------------------------------------------------
def apply(
    part_stream: stream.Part,
    profile_name: str | None = None,
    *,
    swing_ratio: float | None = None,
    kick_leak_velocity_jitter: int = 0,
    global_settings: Mapping[str, Any] | None = None,
) -> None:
    """music21.stream.Part に in-place でヒューマナイズを適用"""
    prof = get_profile(profile_name) if profile_name else {}
    off = prof.get("offset_ms", {})  # {mean, stdev}
    vel = prof.get("velocity", {})
    dur = prof.get("duration_pct", {})

    notes = list(part_stream.flatten().notes)
    event_tables = _resolve_event_type_tables(global_settings)
    instrument_family = _infer_instrument_family(part_stream)
    tempo_bpm = _resolve_tempo_bpm(part_stream, global_settings)

    if any(event_tables):
        for element in part_stream.recurse().notes:
            _apply_event_type_profile_to_element(
                element,
                event_tables,
                instrument_family,
                tempo_bpm,
            )

    for n in notes:
        # (a) onset shift
        if off:
            jitter = random.normalvariate(off.get("mean", 0.0), off.get("stdev", 0.0))
            n.offset += (
                jitter / 1000.0
            )  # ms → sec (assuming QL = 1 sec @ 60bpm; precise shiftは later pass で)

        # (b) velocity tweak
        if vel and n.volume.velocity is not None:
            tweak = random.normalvariate(vel.get("mean", 0.0), vel.get("stdev", 0.0))
            n.volume.velocity = int(max(1, min(127, n.volume.velocity + tweak)))

        # (c) duration ratio (legato / staccato 感)
        if dur:
            factor = random.normalvariate(dur.get("mean", 100), dur.get("stdev", 0)) / 100.0
            n.quarterLength *= factor

    if prof.get("gliss_pairs"):
        extra = getattr(part_stream, "extra_cc", [])
        for a, b in zip(notes, notes[1:]):
            if int(a.pitch.midi) != int(b.pitch.midi):
                dur = float(b.offset) - float(a.offset)
                val = max(0, min(127, int(dur * 127)))
                off = float(a.offset)
                extra.append({"time": off, "cc": 5, "val": val})
                extra.append({"time": off, "cc": 65, "val": 127})
        part_stream.extra_cc = extra

    if kick_leak_velocity_jitter:
        bpm_el = part_stream.recurse().getElementsByClass(tempo.MetronomeMark).first()
        bpm = bpm_el.number if bpm_el else 120
        thr = 0.06 * bpm / 60.0
        kicks = [n for n in notes if int(n.pitch.midi) == 36]
        hats = [n for n in notes if int(n.pitch.midi) in {42, 44, 46}]
        for h in hats:
            if any(abs(float(h.offset) - float(k.offset)) <= thr for k in kicks):
                if h.volume.velocity is None:
                    continue
                delta = random.randint(-kick_leak_velocity_jitter, kick_leak_velocity_jitter)
                h.volume.velocity = int(max(1, min(127, h.volume.velocity + delta)))

    if swing_ratio is None and profile_name:
        swing_ratio = cast(float | None, PROFILES.get(profile_name, {}).get("swing"))

    if swing_ratio is not None:
        try:
            quantize.setSwingRatio(swing_ratio)
        except Exception:
            pass
        swing_offset = round(swing_ratio * 0.5, 2)
        for n in part_stream.recurse().getElementsByClass(["Note", "Chord", "Rest"]):
            if n.quarterLength >= 1.0:
                continue
            beat_pos = float(n.offset) % 1.0
            beat_start = n.offset - beat_pos
            if abs(beat_pos - 0.5) < 0.05:
                n.offset = beat_start + swing_offset
            elif beat_pos < 0.05 or beat_pos > 0.95:
                n.offset = beat_start

    gs = global_settings or {}
    expr = bool(gs.get("use_expr_cc11", USE_EXPR_CC11))
    aft = bool(gs.get("use_aftertouch", USE_AFTERTOUCH))
    if expr or aft:
        _humanize_velocities(
            part_stream,
            amount=4,
            global_settings=gs,
            expr_curve=str(gs.get("expr_curve", "linear")),
            kick_leak_jitter=int(gs.get("kick_leak_jitter", 0)),
        )


def _validate_ratio(ratio: float) -> float:
    """Return a safe swing ratio allowing ``0`` for straight feel."""
    if ratio <= 0:
        # ``0`` disables swing; no warning necessary
        return 0.0
    if ratio >= 1:
        logger.warning("Swing ratio %.3f out of range (0,1). Using 0.5", ratio)
        return 0.5
    return ratio


def _apply_swing_py(part_stream: stream.Part, swing_ratio: float, subdiv: int = 8) -> None:
    """Shift off-beats according to ``swing_ratio`` and grid ``subdiv``.

    ``swing_ratio`` represents the relative position of the off-beat note
    within a pair (0.5 means straight). ``subdiv`` describes the number of
    divisions in a 4/4 measure. 8 results in eighth‑note swing.
    """
    swing_ratio = _validate_ratio(swing_ratio)
    if subdiv <= 0:
        return

    step = 4.0 / subdiv  # length of the smallest grid
    pair = step * 2.0  # span of an on/off pair
    tol = step * 0.1

    for n in part_stream.recurse().notes:
        pos = float(n.offset)
        pair_start = math.floor(pos / pair) * pair
        within = pos - pair_start
        if abs(within - step) < tol:
            n.offset = pair_start + pair * swing_ratio


def swing_offset(offset: float, swing_ratio: float, subdiv: int = 8) -> float:
    """Return swung ``offset`` without modifying an element."""
    swing_ratio = _validate_ratio(swing_ratio)
    if subdiv <= 0:
        return offset
    step = 4.0 / subdiv
    pair = step * 2.0
    tol = step * 0.1
    pair_start = math.floor(offset / pair) * pair
    within = offset - pair_start
    if abs(within - step) < tol:
        return pair_start + pair * swing_ratio
    return offset


def _apply_swing(part_stream: stream.Part, swing_ratio: float, subdiv: int = 8) -> None:
    if cy_apply_swing is not None:
        cy_apply_swing(part_stream, float(swing_ratio), subdiv)
    else:
        _apply_swing_py(part_stream, swing_ratio, subdiv)


def _apply_variable_swing(
    part_stream: stream.Part, ratios: Mapping[int, float], subdiv: int = 8
) -> None:
    """Apply swing where the ratio can vary per measure index."""
    if subdiv <= 0:
        return

    step = 4.0 / subdiv
    pair = step * 2.0
    tol = step * 0.1

    ts = part_stream.recurse().getElementsByClass(meter.TimeSignature).first()
    measure_len = float(ts.barDuration.quarterLength) if ts else 4.0

    for n in part_stream.recurse().notes:
        pos = float(n.offset)
        pair_start = math.floor(pos / pair) * pair
        within = pos - pair_start
        if abs(within - step) < tol:
            measure_idx = int(pair_start // measure_len)
            ratio = ratios.get(measure_idx, ratios.get("default", 0.5))
            ratio = _validate_ratio(ratio)
            n.offset = pair_start + pair * ratio


def _guess_subdiv(part: stream.Part) -> int:
    """Return default swing subdivision based on time signature."""
    ts = part.recurse().getElementsByClass(meter.TimeSignature).first()
    if ts:
        num, denom = ts.numerator, ts.denominator
        if denom == 4 and num in (3, 4):
            return 8
        if denom == 8 and num in (6, 12):
            return 12
    return 8


def apply_swing(
    part: stream.Part,
    ratio: float | Sequence[float] | Mapping[int, float],
    subdiv: int | None = 8,
) -> None:
    """Public API to apply swing in-place.

    Parameters
    ----------
    part : :class:`music21.stream.Part`
        Target part to modify.
    ratio : float | Sequence[float] | Mapping[int, float]
        Relative position of the off-beat note (0.5 = straight).
    subdiv : int | None
        Number of grid subdivisions per measure. ``8`` for typical eighth swing.
        When ``None``, a suitable value is inferred from the part's time signature
        (e.g. ``8`` for ``4/4`` or ``3/4`` and ``12`` for ``6/8`` or ``12/8``).
    """
    if ratio is None:
        return

    if subdiv is None:
        subdiv = _guess_subdiv(part)

    if isinstance(ratio, (int, float)):
        r = float(ratio)
        if abs(r) < 1e-6:
            return
        r = _validate_ratio(r)
        _apply_swing(part, r, subdiv=subdiv)
    else:
        if isinstance(ratio, Sequence):
            ratios = {idx: float(r) for idx, r in enumerate(ratio)}
        else:
            ratios = {int(k): float(v) for k, v in ratio.items()}
        _apply_variable_swing(part, ratios, subdiv=subdiv)


def _apply_envelope_py(part: stream.Part, start: int, end: int, scale: float) -> None:
    for n in part.recurse().notes:
        if start <= n.offset < end and n.volume and n.volume.velocity is not None:
            n.volume.velocity = int(max(1, min(127, round(n.volume.velocity * scale))))


def apply_envelope(part: stream.Part, start: int, end: int, scale: float) -> None:
    """Scale note velocities between start and end beats."""
    if cy_apply_envelope is not None:
        cy_apply_envelope(part, start, end, scale)
    else:
        _apply_envelope_py(part, start, end, scale)


def _humanize_velocities_py(
    part: stream.Part,
    amount: int = 4,
    use_expr_cc11: bool = False,
    use_aftertouch: bool = False,
    expr_curve: str = "linear",
    kick_leak_jitter: int = 0,
) -> None:
    notes = list(part.recurse().notes)
    kicks = [n for n in notes if int(n.pitch.midi) == 36]
    for n in notes:
        if n.volume is None:
            n.volume = volume.Volume(velocity=64)
        vel = n.volume.velocity or 64
        delta = random.randint(-amount, amount)
        new_vel = max(1, min(127, vel + delta))
        if kick_leak_jitter and int(n.pitch.midi) in {42, 44, 46}:
            bpm_el = part.recurse().getElementsByClass(tempo.MetronomeMark).first()
            bpm = bpm_el.number if bpm_el else 120
            thr = 0.1 * bpm / 60.0
            if any(abs(float(n.offset) - float(k.offset)) <= thr for k in kicks):
                jitter = random.randint(-kick_leak_jitter, kick_leak_jitter)
                new_vel = max(1, min(127, new_vel + jitter))
        n.volume.velocity = new_vel
        cc_val = new_vel
        if expr_curve == "cubic-in":
            cc_val = int(max(1, min(127, (new_vel / 127.0) ** 3 * 127)))
        if use_expr_cc11:
            part.extra_cc = getattr(part, "extra_cc", []) + [
                {"time": float(n.offset), "cc": 11, "val": cc_val}
            ]
        if use_aftertouch:
            part.extra_cc = getattr(part, "extra_cc", []) + [
                {"time": float(n.offset), "cc": 74, "val": cc_val}
            ]


def _humanize_velocities(
    part: stream.Part,
    amount: int = 4,
    *,
    global_settings: Mapping[str, Any] | None = None,
    expr_curve: str = "linear",
    kick_leak_jitter: int = 0,
) -> None:
    """Randomise note velocities and optionally emit CC messages."""
    gs = global_settings or {}
    use_expr = bool(gs.get("use_expr_cc11", False))
    use_at = bool(gs.get("use_aftertouch", False))
    if cy_humanize_velocities is not None:
        cy_humanize_velocities(part, amount, use_expr, use_at, expr_curve, kick_leak_jitter)
    else:
        _humanize_velocities_py(part, amount, use_expr, use_at, expr_curve, kick_leak_jitter)


def _apply_velocity_histogram_py(part: stream.Part, histogram: dict[int, float]) -> stream.Part:
    """Assign note velocities by sampling from ``histogram``.

    Parameters
    ----------
    part:
        Target part whose note velocities will be replaced.
    histogram:
        Mapping of velocity value to weight.

    Returns
    -------
    stream.Part
        The modified part.
    """
    if not histogram:
        return part
    choices = [int(v) for v in histogram.keys()]
    weights = [float(w) for w in histogram.values()]
    for n in part.recurse().notes:
        vel = random.choices(choices, weights)[0]
        if n.volume is None:
            n.volume = volume.Volume(velocity=vel)
        else:
            n.volume.velocity = vel
    return part


def apply_velocity_histogram(
    part: stream.Part,
    histogram: dict[int, float] | None = None,
    *,
    profile: str | dict[int, float] | None = None,
) -> stream.Part:
    """Apply a velocity histogram mapping.

    Provide ``histogram`` directly or pass ``profile`` as either a preset name
    or mapping. When ``profile`` is a string it will be resolved via
    :data:`VELO_PROFILES`.
    """
    if isinstance(profile, str):
        try:
            profile = VELO_PROFILES[profile]
        except KeyError:
            raise KeyError(f"velocity histogram profile '{profile}' not found") from None
    if histogram is None and profile is None:
        raise TypeError("either histogram or profile must be provided")
    if histogram is None:
        histogram = profile  # type: ignore[assignment]
    elif profile is not None:
        histogram = profile  # type: ignore[assignment]
    if cy_apply_velocity_histogram is not None:
        return cy_apply_velocity_histogram(part, histogram)
    return _apply_velocity_histogram_py(part, histogram)


def apply_velocity_histogram_profile(part: stream.Part, profile: str = "piano_soft") -> stream.Part:
    """Apply a named velocity histogram profile."""
    try:
        hist = VELO_PROFILES[profile]
    except KeyError:
        raise KeyError(f"velocity histogram profile '{profile}' not found") from None
    return apply_velocity_histogram(part, hist)


def apply_offset_profile(part: stream.Part, profile_name: str | None) -> None:
    """Shift note offsets according to a registered profile."""
    if not profile_name:
        return
    try:
        profile = get_profile(profile_name)
    except KeyError:
        logger.warning(f"Offset profile '{profile_name}' not found.")
        return

    if "shift_ql" in profile:
        try:
            shift = float(profile["shift_ql"])
        except (TypeError, ValueError):
            shift = 0.0
        for n in part.recurse().getElementsByClass(["Note", "Chord", "Rest"]):
            n.offset += shift
        for cc in getattr(part, "extra_cc", []):
            cc["time"] += shift
        return

    pattern = profile.get("offsets_ql") or profile.get("pattern")
    if not isinstance(pattern, (list, tuple)) or not pattern:
        logger.warning(f"Offset profile '{profile_name}' has no usable 'shift_ql' or 'offsets_ql'.")
        return

    shifts = [float(x) for x in pattern]
    notes = list(part.recurse().getElementsByClass(["Note", "Chord", "Rest"]))
    for idx, el in enumerate(notes):
        shift = shifts[idx % len(shifts)]
        el.offset += shift
    for idx, cc in enumerate(getattr(part, "extra_cc", [])):
        shift = shifts[idx % len(shifts)]
        cc["time"] += shift


# ------------------------------------------------------------
# 3) CLI テスト用メイン（任意）
# ------------------------------------------------------------
if __name__ == "__main__":  # python utilities/humanizer.py main_cfg.yml
    import sys

    import yaml

    cfg_path = Path(sys.argv[1])
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    load_profiles(cfg["humanize_profiles"])
    print("Profiles loaded:", list(_PROFILE_REGISTRY))


HUMANIZATION_TEMPLATES: dict[str, dict[str, Any]] = {
    "default_subtle": {
        "time_variation": 0.01,
        "duration_percentage": 0.03,
        "velocity_variation": 5,
        "use_fbm_time": False,
    },
    "piano_gentle_arpeggio": {
        "time_variation": 0.008,
        "duration_percentage": 0.02,
        "velocity_variation": 4,
        "use_fbm_time": True,
        "fbm_time_scale": 0.005,
        "fbm_hurst": 0.7,
    },
    "piano_block_chord": {
        "time_variation": 0.015,
        "duration_percentage": 0.04,
        "velocity_variation": 7,
        "use_fbm_time": False,
    },
    "drum_tight": {
        "time_variation": 0.005,
        "duration_percentage": 0.01,
        "velocity_variation": 3,
        "use_fbm_time": False,
    },
    "drum_loose_fbm": {
        "time_variation": 0.02,
        "duration_percentage": 0.05,
        "velocity_variation": 8,
        "use_fbm_time": True,
        "fbm_time_scale": 0.01,
        "fbm_hurst": 0.6,
    },
    "guitar_strum_loose": {
        "time_variation": 0.025,
        "duration_percentage": 0.06,
        "velocity_variation": 10,
        "use_fbm_time": True,
        "fbm_time_scale": 0.015,
    },
    "guitar_arpeggio_precise": {
        "time_variation": 0.008,
        "duration_percentage": 0.02,
        "velocity_variation": 4,
        "use_fbm_time": False,
    },
    "vocal_ballad_smooth": {
        "time_variation": 0.025,
        "duration_percentage": 0.05,
        "velocity_variation": 4,
        "use_fbm_time": True,
        "fbm_time_scale": 0.01,
        "fbm_hurst": 0.7,
    },
    "vocal_pop_energetic": {
        "time_variation": 0.015,
        "duration_percentage": 0.02,
        "velocity_variation": 8,
        "use_fbm_time": True,
        "fbm_time_scale": 0.008,
    },
    "flam_legato_ghost": {
        "time_variation": 0.005,
        "duration_percentage": 1.2,
        "velocity_variation": [-0.2, 0.2],
        "apply_to_elements": ["note"],
    },
}

# Simple velocity histogram profiles
VELO_PROFILES: dict[str, dict[int, float]] = {
    "piano_soft": {50: 0.2, 60: 0.5, 70: 0.3},
    "piano_hard": {90: 0.3, 100: 0.5, 110: 0.2},
}


def get_velocity_histogram(name: str) -> dict[int, float]:
    """Return velocity histogram preset by name."""
    try:
        return VELO_PROFILES[name]
    except KeyError:
        raise KeyError(f"velocity histogram profile '{name}' not found") from None


try:
    import numpy as np

    NUMPY_AVAILABLE = True
except ImportError:
    np = None  # type: ignore
    NUMPY_AVAILABLE = False


def generate_fractional_noise(
    length: int, hurst: float = 0.7, scale_factor: float = 1.0
) -> list[float]:
    if not NUMPY_AVAILABLE:
        logger.debug(
            f"Humanizer (FBM): NumPy not available. Using Gaussian noise for length {length}."
        )
        return [random.gauss(0, scale_factor / 3) for _ in range(length)]
    if length <= 0:
        return []
    white_noise = np.random.randn(length)
    fft_white = np.fft.fft(white_noise)
    freqs = np.fft.fftfreq(length)
    freqs[0] = 1e-6 if freqs.size > 0 and freqs[0] == 0 else freqs[0]
    filter_amplitude = np.abs(freqs) ** (-hurst)
    if freqs.size > 0:
        filter_amplitude[0] = 0
    fft_fbm = fft_white * filter_amplitude
    fbm_noise = np.fft.ifft(fft_fbm).real
    std_dev = np.std(fbm_noise)
    if std_dev != 0:
        fbm_norm = scale_factor * (fbm_noise - np.mean(fbm_noise)) / std_dev
    else:
        fbm_norm = np.zeros(length)
    return fbm_norm.tolist()


def apply_humanization_to_element(
    m21_element_obj: note.Note | m21chord.Chord,
    template_name: str | None = None,
    custom_params: dict[str, Any] | None = None,
) -> note.Note | m21chord.Chord:
    if not isinstance(m21_element_obj, (note.Note, m21chord.Chord)):
        logger.warning(
            f"Humanizer: apply_humanization_to_element received non-Note/Chord object: {type(m21_element_obj)}"
        )
        return m21_element_obj

    actual_template_name = (
        template_name
        if template_name and template_name in HUMANIZATION_TEMPLATES
        else "default_subtle"
    )
    params = HUMANIZATION_TEMPLATES.get(actual_template_name, {}).copy()

    if custom_params:
        params.update(custom_params)

    element_copy = copy.deepcopy(m21_element_obj)
    time_var = params.get("time_variation", 0.01)
    dur_perc = params.get("duration_percentage", 0.03)
    vel_var = params.get("velocity_variation", 5)
    use_fbm = params.get("use_fbm_time", False)
    fbm_scale = params.get("fbm_time_scale", 0.01)
    fbm_h = params.get("fbm_hurst", 0.6)

    if use_fbm and NUMPY_AVAILABLE:
        time_shift = generate_fractional_noise(1, hurst=fbm_h, scale_factor=fbm_scale)[0]
    else:
        if use_fbm and not NUMPY_AVAILABLE:
            logger.debug(
                "Humanizer: FBM time shift requested but NumPy not available. Using uniform random."
            )
        time_shift = random.uniform(-time_var, time_var)

    original_offset = element_copy.offset
    element_copy.offset += time_shift
    if element_copy.offset < 0:
        element_copy.offset = 0.0

    if element_copy.duration:
        original_ql = element_copy.duration.quarterLength
        duration_change = original_ql * random.uniform(-dur_perc, dur_perc)
        new_ql = max(MIN_NOTE_DURATION_QL / 8, original_ql + duration_change)
        try:
            element_copy.duration.quarterLength = new_ql
        except exceptions21.DurationException as e:
            logger.warning(
                f"Humanizer: DurationException for {element_copy}: {e}. Skip dur change."
            )

    notes_to_affect = (
        element_copy.notes if isinstance(element_copy, m21chord.Chord) else [element_copy]
    )
    for n_obj_affect in notes_to_affect:
        if isinstance(n_obj_affect, note.Note):
            base_vel = (
                n_obj_affect.volume.velocity
                if hasattr(n_obj_affect, "volume")
                and n_obj_affect.volume
                and n_obj_affect.volume.velocity is not None
                else 64
            )
            if isinstance(vel_var, (list, tuple)) and len(vel_var) == 2:
                scale = 1.0 + random.uniform(float(vel_var[0]), float(vel_var[1]))
                final_vel = int(max(1, min(127, round(base_vel * scale))))
            else:
                vel_range = int(vel_var)
                vel_change = random.randint(-vel_range, vel_range)
                final_vel = max(1, min(127, base_vel + vel_change))

            if hasattr(n_obj_affect, "volume") and n_obj_affect.volume is not None:
                n_obj_affect.volume.velocity = final_vel
            else:
                n_obj_affect.volume = volume.Volume(velocity=final_vel)

    return element_copy


def apply_humanization_to_part(
    part_to_humanize: stream.Part,
    template_name: str | None = None,
    custom_params: dict[str, Any] | None = None,
) -> stream.Part:
    if not isinstance(part_to_humanize, stream.Part):
        logger.error("Humanizer: apply_humanization_to_part expects a music21.stream.Part object.")
        return part_to_humanize

    # part_to_humanize.id が int の場合もあるので、文字列に変換してから連結する
    if part_to_humanize.id:
        base_id = str(part_to_humanize.id)
        new_id = f"{base_id}_humanized"
    else:
        new_id = "HumanizedPart"
    humanized_part = stream.Part(id=new_id)
    for el_class_item in [
        instrument.Instrument,
        tempo.MetronomeMark,
        meter.TimeSignature,
        key.KeySignature,
        expressions.TextExpression,
    ]:
        for item_el in part_to_humanize.getElementsByClass(el_class_item):
            humanized_part.insert(item_el.offset, copy.deepcopy(item_el))

    elements_to_process = []
    for element_item in part_to_humanize.recurse().getElementsByClass(["Note", "Chord", "Rest"]):
        elements_to_process.append(element_item)

    elements_to_process.sort(key=lambda el_sort: el_sort.getOffsetInHierarchy(part_to_humanize))

    for element_proc in elements_to_process:
        original_hierarchical_offset = element_proc.getOffsetInHierarchy(part_to_humanize)

        if isinstance(element_proc, (note.Note, m21chord.Chord)):
            humanized_element = apply_humanization_to_element(
                element_proc, template_name, custom_params
            )
            offset_shift_from_humanize = humanized_element.offset - element_proc.offset
            final_insert_offset = original_hierarchical_offset + offset_shift_from_humanize
            if final_insert_offset < 0:
                final_insert_offset = 0.0

            humanized_part.insert(final_insert_offset, humanized_element)
        elif isinstance(element_proc, note.Rest):
            humanized_part.insert(original_hierarchical_offset, copy.deepcopy(element_proc))

    return humanized_part


# --- END OF FILE utilities/humanizer.py ---
