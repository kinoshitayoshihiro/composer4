"""Library for rendering control curves onto PrettyMIDI objects.

This module intentionally exposes a small surface area; command line parsing
is handled by :mod:`utilities.apply_controls_cli`.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable, Mapping

try:  # optional dependency
    import pretty_midi  # type: ignore
except Exception:  # pragma: no cover - fallback stub in tests
    from tests._stubs import pretty_midi  # type: ignore

from .controls_spline import ControlCurve

logger = logging.getLogger(__name__)


_INITIAL_RPN_WINDOW = 0.05
_RPN_NUMBERS = {101, 100, 6, 38}
_RPN_ORDER = {101: 0, 100: 1, 6: 2, 38: 3}


def _rpn_time(
    rpn_at: float,
    first_pb_time: float | None,
    first_cc_time: float | None = None,
    *,
    lead: float = 1e-6,
) -> float:
    """Return an RPN timestamp slightly earlier than other controller events."""

    candidates: list[float] = []
    for val in (rpn_at, first_pb_time, first_cc_time):
        if val is None:
            continue
        try:
            candidates.append(float(val))
        except (TypeError, ValueError):  # pragma: no cover - defensive
            continue
    base = min(candidates) if candidates else 0.0
    t = base - float(lead)
    return t if t >= 0.0 else 0.0


def _rpn_block_times(base: float, length: int, *, eps: float = 1e-12) -> list[float]:
    """Return monotonically increasing timestamps ending at ``base``."""

    base = float(max(0.0, base))
    if length <= 1:
        return [base]

    local_eps = float(eps)
    if base <= eps * (length - 1):
        local_eps = min(eps, max(base / (length + 1e-6), eps * 1e-3))

    times = [base - local_eps * (length - 1 - idx) for idx in range(length)]
    for i in range(1, len(times)):
        if times[i] <= times[i - 1]:
            times[i] = times[i - 1] + local_eps
    for i in range(len(times)):
        if times[i] > base:
            times[i] = base
    return times


def ensure_instrument_for_channel(pm: pretty_midi.PrettyMIDI, ch: int) -> pretty_midi.Instrument:
    """Return an instrument routed to MIDI channel ``ch``.

    Events in PrettyMIDI carry no channel attribute; instead we keep a
    dedicated instrument per channel named ``"channel{ch}"``.
    """

    name = f"channel{ch}"
    for inst in pm.instruments:
        if inst.name == name:
            return inst
    inst = pretty_midi.Instrument(program=0, name=name)
    pm.instruments.append(inst)
    return inst


def _sort_events(inst: pretty_midi.Instrument) -> None:
    """Sort control and bend events so that RPNs come first."""

    events: list[tuple[float, int, int, int, str, object]] = []
    for cc in inst.control_changes:
        num = int(cc.number)
        pr = 0 if num in _RPN_NUMBERS else 1
        order = _RPN_ORDER.get(num, num)
        events.append((float(cc.time), pr, order, int(cc.value), "cc", cc))
    for pb in inst.pitch_bends:
        events.append((float(pb.time), 2, 0, int(getattr(pb, "pitch", 0)), "pb", pb))

    def _sort_key(e: tuple[float, int, int, int, str, object]) -> tuple[float, int, int, int]:
        time, pr, number, value, _kind, _obj = e
        # Make RPN formals (101/100/6/38) win ties by a tiny epsilon lead.
        if pr == 0:
            time -= 1e-9
        order = _RPN_ORDER.get(number, number)
        return (time, pr, order, value)

    events.sort(key=_sort_key)
    inst.control_changes = _dedup_rpn_blocks([e[5] for e in events if e[4] == "cc"])
    inst.pitch_bends = [e[5] for e in events if e[4] == "pb"]


def _dedup_rpn_blocks(
    ccs: list[pretty_midi.ControlChange],
) -> list[pretty_midi.ControlChange]:
    """Keep only the initial RPN block cluster in *ccs*.

    Consecutive 101/100/6[/38] events sharing the same timestamp within the
    first :data:`_INITIAL_RPN_WINDOW` seconds are collapsed, while later blocks
    are preserved verbatim to allow bend range changes mid-song.
    """

    out: list[pretty_midi.ControlChange] = []
    seen_initial_times: set[float] = set()
    i = 0
    while i < len(ccs):
        cc = ccs[i]
        if (
            (cc.number, cc.value) == (101, 0)
            and i + 2 < len(ccs)
            and (ccs[i + 1].number, ccs[i + 1].value) == (100, 0)
            and ccs[i + 2].number == 6
        ):
            block_len = 3
            if i + 3 < len(ccs) and ccs[i + 3].number == 38:
                block_len = 4
            block_time = float(cc.time)
            same_time_cluster = all(
                abs(float(ccs[i + offset].time) - block_time) <= 1e-6 for offset in range(block_len)
            )
            if block_time <= _INITIAL_RPN_WINDOW and same_time_cluster:
                key = round(block_time, 6)
                if key in seen_initial_times:
                    i += block_len
                    continue
                seen_initial_times.add(key)
            out.extend(ccs[i : i + block_len])
            i += block_len
            continue
        out.append(cc)
        i += 1
    return out


def write_bend_range_rpn(
    inst: pretty_midi.Instrument,
    range_semitones: float,
    *,
    at_time: float = 0.0,
    precision: str = "midi128",
) -> None:
    """Emit bend-range RPN (101/100 → 6/38) using 1/128 semitone steps by default.

    ``precision`` may be set to ``"cent"`` to encode the fractional part in
    1/100‑semitone increments for legacy pipelines that persisted this layout.
    """

    mode = precision.lower()
    if mode not in {"midi128", "cent"}:
        raise ValueError(f"invalid precision '{precision}'")
    scale = 128.0 if mode == "midi128" else 100.0
    lsb_max = 127 if mode == "midi128" else 99

    base_time = max(0.0, float(at_time))
    value = max(0.0, float(range_semitones))
    msb = int(math.floor(value))
    frac = value - msb
    if math.isclose(frac, 0.0, abs_tol=1e-9):
        lsb = 0
    else:
        lsb = int(round(frac * scale))
    msb = max(0, min(24, msb))
    lsb = max(0, min(lsb_max, lsb))

    ccs = inst.control_changes
    if not isinstance(ccs, list):
        ccs = list(ccs)
        inst.control_changes = ccs

    # Try to reuse an existing leading RPN block if present.
    for i in range(len(ccs) - 2):
        a, b, c = ccs[i : i + 3]
        if (a.number, a.value) == (101, 0) and (b.number, b.value) == (100, 0) and c.number == 6:
            d = ccs[i + 3] if i + 3 < len(ccs) and ccs[i + 3].number == 38 else None
            existing_scale = 100.0
            if d is not None and d.value > 99:
                existing_scale = 128.0
            cur = c.value + (d.value if d else 0) / existing_scale
            inst._rpn_written = True  # type: ignore[attr-defined]
            if abs(cur - range_semitones) <= 0.01:
                inst._rpn_range = cur  # type: ignore[attr-defined]
                inst._rpn_precision = mode  # type: ignore[attr-defined]
                return
            # Update in-place and align times as a tight block ending at base_time.
            c.value = msb
            if d is not None:
                d.value = lsb
            block_times = _rpn_block_times(base_time, 4 if d is not None else 3)
            a.time = block_times[0]
            b.time = block_times[1]
            c.time = block_times[2]
            if d is not None:
                d.time = block_times[3]
            inst._rpn_range = msb + lsb / scale  # type: ignore[attr-defined]
            inst._rpn_precision = mode  # type: ignore[attr-defined]
            return

    # Otherwise append a fresh block at (or just before) base_time.
    block_times = _rpn_block_times(base_time, 4)
    ccs.extend(
        [
            pretty_midi.ControlChange(number=101, value=0, time=block_times[0]),
            pretty_midi.ControlChange(number=100, value=0, time=block_times[1]),
            pretty_midi.ControlChange(number=6, value=msb, time=block_times[2]),
            pretty_midi.ControlChange(number=38, value=lsb, time=block_times[3]),
        ]
    )
    inst._rpn_written = True  # type: ignore[attr-defined]
    inst._rpn_range = msb + lsb / scale  # type: ignore[attr-defined]
    inst._rpn_precision = mode  # type: ignore[attr-defined]


_CC_MAP = {"cc11": 11, "cc64": 64}


def apply_controls(
    pm: pretty_midi.PrettyMIDI,
    curves_by_channel: Mapping[int, Mapping[str, ControlCurve]],
    *,
    bend_range_semitones: float = 2.0,
    write_rpn: bool = True,
    rpn_at: float = 0.0,
    sample_rate_hz: Mapping[str, float] | None = None,
    max_events: Mapping[str, int] | None = None,
    value_eps: float = 1e-6,
    time_eps: float = 1e-9,
    tempo_map: float | list[tuple[float, float]] | Callable[[float], float] | None = None,
    simplify_mode: str = "rdp",
    total_max_events: int | None = None,
) -> pretty_midi.PrettyMIDI:
    """Render ``curves_by_channel`` onto ``pm``.

    Parameters mirror :meth:`ControlCurve.to_midi_cc` and
    :meth:`ControlCurve.to_pitch_bend` and are forwarded accordingly.
    """

    sr_map = sample_rate_hz or {
        "cc": 30.0,
        "cc11": 30.0,
        "cc64": 30.0,
        "bend": 120.0,
    }
    max_map = max_events or {}
    tempo_pairs = None
    if isinstance(tempo_map, (list, tuple)):
        tempo_pairs = sorted((float(b), float(bpm)) for b, bpm in tempo_map)

    def _beats_to_seconds(beat: float) -> float:
        tm = tempo_map
        if tm is None:
            return float(beat) * 0.5
        if hasattr(tm, "sec_at"):
            try:
                return float(tm.sec_at(beat))
            except Exception:  # pragma: no cover - fallback when tempo_map misbehaves
                pass
        if isinstance(tm, (int, float)):
            bpm = float(tm)
            if bpm <= 0 or not math.isfinite(bpm):
                raise ValueError("bpm must be positive and finite")
            return float(beat) * 60.0 / bpm
        if callable(tm):
            bpm = float(tm(beat))
            if bpm <= 0 or not math.isfinite(bpm):
                raise ValueError("bpm must be positive and finite")
            return float(beat) * 60.0 / bpm
        pairs = tempo_pairs or []
        if not pairs:
            return float(beat) * 0.5
        t_acc = 0.0
        for i, (b0, bpm) in enumerate(pairs):
            b1 = pairs[i + 1][0] if i + 1 < len(pairs) else None
            if i == 0 and beat <= b0:
                return (beat - b0) * 60.0 / bpm
            if b1 is None or beat < b1:
                return t_acc + (beat - b0) * 60.0 / bpm
            t_acc += (b1 - b0) * 60.0 / bpm
        return t_acc

    def _maybe_fix_cc_times(
        curve: ControlCurve,
        events: list[pretty_midi.ControlChange],
        target: str,
    ) -> None:
        # Only fix if (a) beats domain, (b) tempo_map provided, (c) initial constant time cluster.
        if not events or tempo_map is None or getattr(curve, "domain", None) != "beats":
            return
        first = float(events[0].time)
        if any(abs(float(ev.time) - first) > max(time_eps, 1e-9) for ev in events[1:]):
            return
        sr_val = sr_map.get(target)
        if sr_val is None:
            sr_val = sr_map.get("cc")
        if sr_val is None or float(sr_val) <= 0:
            sr_val = getattr(curve, "sample_rate_hz", 0.0)
        try:
            sr = float(sr_val)
        except (TypeError, ValueError):
            sr = 0.0
        if sr <= 0:
            return
        start_beat = float(curve.times[0]) if len(curve.times) else 0.0
        step = 1.0 / sr
        offset = float(getattr(curve, "offset_sec", 0.0))
        for idx, ev in enumerate(events):
            beat = start_beat + idx * step
            ev.time = _beats_to_seconds(beat) + offset

    for ch, mapping in curves_by_channel.items():
        inst = ensure_instrument_for_channel(pm, ch)
        bend_requested = False
        for name, curve in mapping.items():
            if name == "bend":
                bend_requested = True
                pre = len(inst.pitch_bends)
                curve.to_pitch_bend(
                    inst,
                    bend_range_semitones=bend_range_semitones,
                    sample_rate_hz=sr_map.get(name),
                    max_events=max_map.get(name),
                    value_eps=value_eps,
                    time_eps=time_eps,
                    tempo_map=tempo_map,
                    simplify_mode=simplify_mode,
                )
                new_pb = inst.pitch_bends[pre:]
                if new_pb:
                    rendered_start = new_pb[0].time
                    rendered_end = new_pb[-1].time
                    start = float(curve.times[0]) if len(curve.times) else 0.0
                    end = float(curve.times[-1]) if len(curve.times) else 0.0
                    if curve.domain == "beats":
                        start, end = curve._beats_to_times([start, end], tempo_map or 120.0)
                    start += curve.offset_sec
                    end += curve.offset_sec
                    # In beats domain we trust the renderer's times.
                    if curve.domain == "beats":
                        start = rendered_start
                        end = rendered_end
                    if abs(start) <= max(time_eps, 1e-12):
                        start = 0.0
                    else:
                        start = max(0.0, start)
                    new_pb[0].time = start
                    new_pb[-1].time = end
                    # Ensure non-negative strictly increasing times
                    if new_pb and new_pb[0].time < 0.0:
                        new_pb[0].time = 0.0
                    if len(new_pb) >= 2 and new_pb[-1].time <= new_pb[0].time:
                        tiny = max(time_eps, 4e-12)
                        new_pb[-1].time = new_pb[0].time + tiny
                    # Drop penultimate if it coincides with last at different pitch.
                    if (
                        len(new_pb) >= 2
                        and abs(new_pb[-1].time - new_pb[-2].time) <= time_eps
                        and getattr(new_pb[-1], "pitch", None) == 0
                        and getattr(new_pb[-2], "pitch", None) != 0
                    ):
                        inst.pitch_bends.pop(-2)
                continue

            if name in _CC_MAP:
                cc_num = _CC_MAP[name]
                sr = sr_map.get(name)
                # Respect generic cap via "cc" when controller-specific is absent.
                max_ev = max_map.get(name)
                if max_ev is None:
                    max_ev = max_map.get("cc")
                pre = len(inst.control_changes)
                curve.to_midi_cc(
                    inst,
                    cc_num,
                    sample_rate_hz=sr,
                    max_events=max_ev,
                    value_eps=value_eps,
                    time_eps=time_eps,
                    tempo_map=tempo_map,
                    simplify_mode=simplify_mode,
                )
                new_cc = [c for c in inst.control_changes[pre:] if c.number == cc_num]
                if new_cc:
                    _maybe_fix_cc_times(curve, new_cc, name)
                    rendered_start = new_cc[0].time
                    rendered_end = new_cc[-1].time
                    start = float(curve.times[0]) if len(curve.times) else 0.0
                    end = float(curve.times[-1]) if len(curve.times) else 0.0
                    if curve.domain == "beats":
                        converted = curve._beats_to_times([start, end], tempo_map or 120.0)
                        if converted:
                            start = float(converted[0])
                            end = float(converted[-1])
                        else:
                            start = end = 0.0
                    start += curve.offset_sec
                    end += curve.offset_sec
                    new_cc[0].time = start
                    new_cc[-1].time = end

        # Emit RPN once per channel if requested and bends (or bends requested) exist.
        if (
            write_rpn
            and (bend_requested or inst.pitch_bends)
            and not getattr(inst, "_rpn_written", False)
        ):
            first_pb_time = (
                min(float(pb.time) for pb in inst.pitch_bends) if inst.pitch_bends else None
            )
            non_rpn_cc_times = [
                float(cc.time) for cc in inst.control_changes if int(cc.number) not in _RPN_NUMBERS
            ]
            first_cc_time = min(non_rpn_cc_times) if non_rpn_cc_times else None
            rpn_timestamp = _rpn_time(rpn_at, first_pb_time, first_cc_time)
            write_bend_range_rpn(
                inst,
                bend_range_semitones,
                at_time=rpn_timestamp,
                precision="midi128",
            )

        _sort_events(inst)

    if total_max_events:
        _enforce_total_cap(pm, total_max_events)

    return pm


def _downsample_keep_endpoints(events: list, target: int) -> list:
    if target <= 0:
        return []
    if target >= len(events):
        return list(events)
    if target == 1:
        return [events[0]]
    idxs = [round(i * (len(events) - 1) / (target - 1)) for i in range(target)]
    seen: set[int] = set()
    out = []
    for idx in idxs:
        if idx not in seen:
            out.append(events[idx])
            seen.add(idx)
    return out


def _enforce_total_cap(pm: pretty_midi.PrettyMIDI, cap: int) -> None:
    if cap <= 0:
        for inst in pm.instruments:
            inst.notes = []
            inst.control_changes = []
            inst.pitch_bends = []
            _sort_events(inst)
        return

    eff_cap = max(0, int(cap))

    groups: list[dict[str, object]] = []
    total_non_rpn = 0
    total_rpn = 0
    preserved_rpn: dict[pretty_midi.Instrument, list[pretty_midi.ControlChange]] = {}
    for inst in pm.instruments:
        rpn_ccs = [cc for cc in inst.control_changes if int(cc.number) in _RPN_NUMBERS]
        preserved_rpn[inst] = rpn_ccs
        total_rpn += len(rpn_ccs)
        notes = list(inst.notes)
        if notes:
            groups.append({"inst": inst, "kind": "note", "key": None, "events": notes})
            total_non_rpn += len(notes)
        cc_map: dict[int, list[pretty_midi.ControlChange]] = {}
        for cc in inst.control_changes:
            if int(cc.number) in _RPN_NUMBERS:
                continue
            cc_map.setdefault(cc.number, []).append(cc)
        for number, changes in cc_map.items():
            groups.append({"inst": inst, "kind": "cc", "key": number, "events": list(changes)})
            total_non_rpn += len(changes)
        bends = list(inst.pitch_bends)
        if bends:
            groups.append({"inst": inst, "kind": "bend", "key": None, "events": bends})
            total_non_rpn += len(bends)

    budget = eff_cap - total_rpn
    if budget < 0 and total_non_rpn > 0:
        total_rpn = 0
        budget = eff_cap
        for inst in preserved_rpn:
            preserved_rpn[inst] = []

    if total_non_rpn + total_rpn <= eff_cap and not any(preserved_rpn.values()):
        if total_non_rpn <= eff_cap:
            return

    budget = max(0, budget)

    if total_non_rpn <= budget and budget >= 0 and not any(preserved_rpn.values()):
        return

    total = total_non_rpn

    allocations: list[dict[str, object]] = []
    sum_targets = 0
    for group in groups:
        events = list(group["events"])
        n = len(events)
        if n == 0:
            min_keep = 0
            ideal = 0.0
            target = 0
        else:
            min_keep = 1 if n == 1 else min(2, n)
            ideal = (n / total) * budget if total > 0 else 0.0
            base = int(math.floor(ideal))
            target = min(n, max(min_keep, base))
        allocations.append(
            {
                "inst": group["inst"],
                "kind": group["kind"],
                "key": group["key"],
                "events": events,
                "min": min_keep,
                "ideal": ideal,
                "target": target,
            }
        )
        sum_targets += target

    if sum_targets > budget:
        while sum_targets > budget:
            reducible = [alloc for alloc in allocations if int(alloc["target"]) > int(alloc["min"])]
            if reducible:
                reducible.sort(
                    key=lambda a: (
                        int(a["target"]) - int(a["min"]),
                        float(a["ideal"]),
                    ),
                    reverse=True,
                )
                candidate = reducible[0]
                current = int(candidate["target"])
                minimum = int(candidate["min"])
                new_target = max(minimum, current - 1)
                candidate["target"] = new_target
                sum_targets -= current - new_target
                continue
            break

        sum_targets = sum(int(a["target"]) for a in allocations)

    remainder = budget - sum_targets
    if remainder > 0:
        expandable = [a for a in allocations if int(a["target"]) < len(a["events"])]
        while remainder > 0 and expandable:
            expandable.sort(
                key=lambda a: (
                    float(a["ideal"]) - float(a["target"]),
                    len(a["events"]),
                ),
                reverse=True,
            )
            candidate = expandable[0]
            current = int(candidate["target"])
            if current >= len(candidate["events"]):
                expandable.pop(0)
                continue
            candidate["target"] = current + 1
            remainder -= 1

    new_notes: dict[pretty_midi.Instrument, list[pretty_midi.Note]] = {}
    new_cc: dict[pretty_midi.Instrument, dict[int, list[pretty_midi.ControlChange]]] = {}
    new_pb: dict[pretty_midi.Instrument, list[pretty_midi.PitchBend]] = {}
    for alloc in allocations:
        target = int(alloc.get("target", 0))
        events = alloc["events"]
        kept = _downsample_keep_endpoints(events, target)
        inst = alloc["inst"]
        kind = alloc["kind"]
        key = alloc["key"]
        if kind == "note":
            new_notes[inst] = kept
        elif kind == "bend":
            new_pb[inst] = kept
        elif kind == "cc" and isinstance(key, int):
            new_cc.setdefault(inst, {})[key] = kept

    for inst in pm.instruments:
        preserved = list(preserved_rpn.get(inst, []))
        if inst in new_notes:
            inst.notes = sorted(
                new_notes[inst],
                key=lambda n: (
                    float(n.start),
                    float(n.end),
                    int(getattr(n, "pitch", 0)),
                    int(getattr(n, "velocity", 0)),
                ),
            )
        if inst in new_pb:
            inst.pitch_bends = sorted(
                new_pb[inst], key=lambda pb: (float(pb.time), int(getattr(pb, "pitch", 0)))
            )
        if inst in new_cc:
            merged: list[pretty_midi.ControlChange] = list(preserved)
            for number, events in new_cc[inst].items():
                deduped: list[pretty_midi.ControlChange] = []
                seen: set[tuple[int, float, int]] = set()
                for cc in sorted(events, key=lambda c: (float(c.time), int(c.value))):
                    key = (number, float(cc.time), int(cc.value))
                    if key in seen:
                        continue
                    seen.add(key)
                    deduped.append(cc)
                merged.extend(deduped)
            merged.sort(key=lambda c: (float(c.time), int(c.number), int(c.value)))
            inst.control_changes = merged
        else:
            inst.control_changes = preserved
        _sort_events(inst)


__all__ = [
    "apply_controls",
    "ensure_instrument_for_channel",
    "write_bend_range_rpn",
]


def _parse_kv_pairs(spec: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for part in spec.split(","):
        if not part:
            continue
        if "=" not in part:
            continue
        key, val = part.split("=", 1)
        try:
            out[key.strip()] = float(val.strip())
        except ValueError:
            continue
    return out


def _load_curve(path: str) -> ControlCurve:
    import json

    with open(path) as fh:
        data = json.load(fh)
    domain = data.get("domain", "time")
    knots = data.get("knots", [])
    times = [float(t) for t, _ in knots]
    values = [float(v) for _, v in knots]
    sr = data.get("sample_rate_hz") or data.get("resolution_hz")
    return ControlCurve(times, values, domain=domain, sample_rate_hz=sr or 0.0)


def main(argv: list[str] | None = None) -> int:
    import argparse
    import json
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Apply control curves to MIDI")
    parser.add_argument("in_mid")
    parser.add_argument("routing_json")
    parser.add_argument("--out")
    parser.add_argument("--bend-range-semitones", type=float, default=2.0)
    parser.add_argument("--write-rpn", action="store_true")
    parser.add_argument("--rpn-at", type=float, default=0.0)
    parser.add_argument("--sample-rate-hz", default="")
    parser.add_argument("--max-bend", type=int)
    parser.add_argument("--max-cc11", type=int)
    parser.add_argument("--max-cc64", type=int)
    parser.add_argument("--value-eps", type=float, default=1e-6)
    parser.add_argument("--time-eps", type=float, default=1e-9)
    parser.add_argument("--tempo-map")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    pm = pretty_midi.PrettyMIDI(args.in_mid)
    routing_path = Path(args.routing_json)
    with routing_path.open() as fh:
        routing = json.load(fh)

    curves_by_channel: dict[int, dict[str, ControlCurve]] = {}
    for ch_str, mapping in routing.items():
        ch = int(ch_str)
        ch_curves: dict[str, ControlCurve] = {}
        for tgt, path in mapping.items():
            ch_curves[tgt] = _load_curve(path)
        curves_by_channel[ch] = ch_curves

    sr_map = _parse_kv_pairs(args.sample_rate_hz) if args.sample_rate_hz else None
    max_map: dict[str, int] = {}
    if args.max_bend is not None:
        max_map["bend"] = args.max_bend
    if args.max_cc11 is not None:
        max_map["cc11"] = args.max_cc11
    if args.max_cc64 is not None:
        max_map["cc64"] = args.max_cc64
    tempo_map = None
    if args.tempo_map:
        tm_path = Path(args.tempo_map)
        if tm_path.exists():
            with tm_path.open() as fh:
                tempo_map = json.load(fh)
        else:
            tempo_map = json.loads(args.tempo_map)

    apply_controls(
        pm,
        curves_by_channel,
        bend_range_semitones=args.bend_range_semitones,
        write_rpn=args.write_rpn,
        rpn_at=args.rpn_at,
        sample_rate_hz=sr_map,
        max_events=max_map or None,
        value_eps=args.value_eps,
        time_eps=args.time_eps,
        tempo_map=tempo_map,
    )

    cc_count = sum(len(inst.control_changes) for inst in pm.instruments)
    bend_count = sum(len(inst.pitch_bends) for inst in pm.instruments)
    print(f"Applied controls: {cc_count} CC, {bend_count} bend events")

    if not args.dry_run:
        out_path = args.out or args.in_mid
        pm.write(str(out_path))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry
    raise SystemExit(main())
