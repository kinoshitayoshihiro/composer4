"""groove_sampler_v2 JSON を MIDI へ変換するユーティリティ。

複数ファイルを一括処理し、テンポや人間味付けの調整を行えます。

CLI tool to convert groove_sampler_v2 JSON event lists into drum MIDI files.
Supports batch conversion, swing, humanization and YAML mapping options.
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import random
import sys
from pathlib import Path

import yaml

import pretty_midi

__version__ = "0.2.1"

logger = logging.getLogger(__name__)

DEFAULT_PITCH_MAP: dict[str, int] = {
    "kick": 36,
    "snare": 38,
    "chh": 42,
    "hh_edge": 46,
    "hh_pedal": 44,
    "snare_brush": 96,
    "ride": 51,
    "tom_low": 45,
    "tom_mid": 47,
    "tom_high": 50,
}

# GM Program numbers for melodic instruments (0-indexed)
DEFAULT_PROGRAMS: dict[str, int] = {
    "piano": 0,  # Acoustic Grand Piano
    "guitar": 25,  # Acoustic Guitar (steel)
    "bass": 33,  # Electric Bass (finger)
    "strings": 48,  # String Ensemble 1
    "pad": 88,  # Pad 1 (new age)
    "synth": 80,  # Lead 1 (square)
}


def _instrument_family_from_name(name: str) -> str:
    lowered = (name or "").lower()
    if any(keyword in lowered for keyword in ["kick", "snare", "hh", "drum", "tom", "perc"]):
        return "drums"
    if "bass" in lowered:
        return "bass"
    if "guitar" in lowered:
        return "guitar"
    if any(k in lowered for k in ["piano", "keys", "keyboard"]):
        return "piano"
    if any(k in lowered for k in ["string", "violin", "cello"]):
        return "strings"
    return "default"


class MatrixHumanizeHook:
    """Deterministic humanize hook driven by plan_humanize.yaml matrix."""

    def __init__(self, config_path: Path | None, seed: int | None = None):
        self.enabled = False
        self.matrix: dict = {}
        self.rng = random.Random(seed or 42)
        if config_path and config_path.exists():
            try:
                with config_path.open("r", encoding="utf-8") as fh:
                    data = yaml.safe_load(fh) or {}
                self.matrix = data.get("humanize", {}).get("matrix", {}) or {}
                if isinstance(self.matrix, dict) and self.matrix:
                    self.enabled = True
            except Exception as exc:
                logger.warning("Failed to load humanize config %s: %s", config_path, exc)

    def _resolve_profile(self, family: str, section: str) -> dict:
        if not self.matrix:
            return {}
        role_cfg = self.matrix.get(family)
        if not isinstance(role_cfg, dict):
            return {}
        profile: dict = {}
        base_cfg = {}
        for key in ("default", "base"):
            candidate = role_cfg.get(key)
            if isinstance(candidate, dict):
                base_cfg = dict(candidate)
                break
        if not base_cfg:
            base_cfg = {
                k: v
                for k, v in role_cfg.items()
                if k not in {"sections", "default", "base"}
            }
        profile.update(base_cfg)

        sections_cfg = role_cfg.get("sections")
        if isinstance(sections_cfg, dict):
            section_override = sections_cfg.get(section)
            if section_override is None:
                section_override = sections_cfg.get("default")
            if isinstance(section_override, dict):
                profile.update(section_override)
        return profile

    def apply(self, event: dict, bpm: float, start_sec: float, end_sec: float, velocity: int) -> tuple[float, float, int]:
        if not self.enabled or not event:
            return start_sec, end_sec, velocity

        family = _instrument_family_from_name(str(event.get("instrument") or event.get("track") or ""))
        section = str(
            event.get("section")
            or event.get("section_label")
            or event.get("section_name")
            or event.get("phrase_section")
            or "verse"
        ).lower()
        profile = self._resolve_profile(family, section)
        if not profile:
            return start_sec, end_sec, velocity

        timing_ms = float(profile.get("timing_jitter_ms", 0.0))
        timing_scale = float(profile.get("timing_scale", 1.0))
        push_ms = float(profile.get("timing_push_ms", 0.0))
        total_shift_ms = push_ms
        if timing_ms > 0:
            total_shift_ms += self.rng.uniform(-timing_ms, timing_ms) * timing_scale
        if total_shift_ms:
            shift_sec = total_shift_ms / 1000.0
            start_sec = max(0.0, start_sec + shift_sec)
            end_sec = max(start_sec + 1e-4, end_sec + shift_sec)

        duration_scale = float(profile.get("duration_scale", 1.0))
        if duration_scale and duration_scale != 1.0:
            duration = max(1e-4, (end_sec - start_sec) * duration_scale)
            end_sec = start_sec + duration

        vel_jitter = float(profile.get("velocity_jitter", 0.0))
        if vel_jitter > 0:
            velocity += int(round(self.rng.uniform(-vel_jitter, vel_jitter)))
        vel_shift = float(profile.get("velocity_shift", 0.0))
        if vel_shift:
            velocity += int(round(vel_shift))
        velocity = max(1, min(127, velocity))

        return start_sec, end_sec, velocity


def _load_json(path: Path) -> dict | list:
    try:
        with path.open() as fh:
            return json.load(fh)
    except FileNotFoundError:
        sys.exit(f"File not found: {path}")
    except json.JSONDecodeError as exc:
        sys.exit(f"JSON parse error in {path}: {exc}")


def _load_mapping(path: Path) -> dict[str, int]:
    ext = path.suffix.lower()
    if ext in {".yaml", ".yml"}:
        try:
            from ruamel.yaml import YAML  # type: ignore
        except Exception:
            sys.exit("ruamel.yaml required for YAML mapping")
        with path.open() as fh:
            data = YAML(typ="safe").load(fh)
    else:
        data = _load_json(path)
    if not isinstance(data, dict):
        sys.exit("Mapping file must define a dictionary")
    return {k: int(v) for k, v in data.items()}


def _beat_to_seconds(beat: float, tempo_changes: list[list[float]] | None, bpm: float) -> float:
    if not tempo_changes:
        return beat * 60.0 / bpm
    tempo_changes = sorted(tempo_changes, key=lambda x: x[0])
    sec = 0.0
    prev_b = 0.0
    prev_t = bpm
    for b, t in tempo_changes:
        if beat < b:
            sec += (beat - prev_b) * 60.0 / prev_t
            return sec
        sec += (b - prev_b) * 60.0 / prev_t
        prev_b, prev_t = b, t
    sec += (beat - prev_b) * 60.0 / prev_t
    return sec


            pm = convert_events(
    events: list[dict[str, float | str]],
    bpm: float,
    mapping: dict[str, int],
    *,
    swing: float = 0.0,
    humanize_timing_ms: float = 0.0,
    humanize_vel_pct: float = 0.0,
    split_tracks: bool = False,
    repeat: int = 1,
                quiet=ns.quiet,
                humanize_hook=matrix_hook,
    quiet: bool = False,
    humanize_hook: MatrixHumanizeHook | None = None,
) -> pretty_midi.PrettyMIDI:
    pattern_len = max(float(ev.get("offset", 0)) + float(ev.get("duration", 0)) for ev in events)
    pm = pretty_midi.PrettyMIDI(initial_tempo=bpm)
    instruments: dict[str, pretty_midi.Instrument] = {}
    if not split_tracks:
        instruments["drums"] = pretty_midi.Instrument(program=0, is_drum=True)
    warned: set[str] = set()

    total = len(events) * repeat
    bar = None
    if not quiet and total > 100:
        try:
            from tqdm import tqdm

            bar = tqdm(total=total, unit="ev", desc="events")
        except ImportError:
            pass

    for rep in range(repeat):
        for ev in events:
            name = str(ev.get("instrument", "unknown"))

            # Support both drum mapping (kick, snare, etc.) and instrument names (Bass, Piano, etc.)
            pitch = mapping.get(name)
            if pitch is None:
                # Try lowercase
                pitch = mapping.get(name.lower())
            if pitch is None:
                # For melodic instruments, use pitch from event (if available) or default to middle C
                if "pitch" in ev:
                    pitch = int(ev["pitch"])
                elif "note" in ev:
                    # Support both MIDI number (int) and note name (str)
                    note_val = ev["note"]
                    if isinstance(note_val, (int, float)):
                        pitch = int(note_val)
                    else:
                        # Convert note name string to MIDI pitch
                        note_str = str(note_val)
                        try:
                            pitch = pretty_midi.note_name_to_number(note_str)
                        except Exception:
                            pitch = 60  # Middle C
                else:
                    pitch = 60  # Middle C for melodic instruments
                    if name not in warned:
                        logger.warning(
                            "Unknown instrument %s: using pitch from event or default", name
                        )
                        warned.add(name)

            # Support multiple event formats:
            # 1. "time" (in seconds) - arrangement_plan.json format
            # 2. "time_ql" (in quarter notes/beats) - plan format
            # 3. "offset" (in beats) - original groove_sampler format
            if "time" in ev:
                # Direct time in seconds
                start = float(ev["time"]) + (pattern_len * rep * 60.0 / bpm if rep > 0 else 0)
                end = start + float(ev.get("duration", 0.25))
            elif "time_ql" in ev:
                # Time in quarter notes (beats)
                start_beat = float(ev["time_ql"]) + pattern_len * rep
                duration_ql = float(ev.get("duration_ql", ev.get("duration", 0.25)))
                end_beat = start_beat + duration_ql
                start = _beat_to_seconds(start_beat, tempo_changes, bpm)
                end = _beat_to_seconds(end_beat, tempo_changes, bpm)
            else:
                # Original offset format
                start_beat = float(ev.get("offset", 0)) + pattern_len * rep
                end_beat = start_beat + float(ev.get("duration", 0))
                if abs(start_beat % 1 - 0.5) < 1e-6:
                    shift = swing * 0.25
                    start_beat = max(start_beat - shift, 0.0)
                    end_beat = max(end_beat - shift, 0.0)
                start = _beat_to_seconds(start_beat, tempo_changes, bpm)
                end = _beat_to_seconds(end_beat, tempo_changes, bpm)

            start += random.uniform(-humanize_timing_ms, humanize_timing_ms) / 1000.0

            # Support both velocity_factor (0-1) and velocity (0-127)
            if "velocity" in ev:
                velocity = int(min(max(float(ev["velocity"]), 1), 127))
            else:
                velocity = int(min(max(float(ev.get("velocity_factor", 1)) * 127, 1), 127))

            if humanize_vel_pct:
                jitter = random.uniform(-humanize_vel_pct, humanize_vel_pct) / 100.0
                velocity = int(min(max(velocity * (1 + jitter), 1), 127))

            if humanize_hook and humanize_hook.enabled:
                start, end, velocity = humanize_hook.apply(ev, bpm, start, end, velocity)

            # Determine if instrument is drum or melodic
            is_drum = name.lower() in mapping or name in [
                "kick",
                "snare",
                "chh",
                "hh_edge",
                "hh_pedal",
                "ride",
                "tom_low",
                "tom_mid",
                "tom_high",
                "drums",
            ]

            # Determine program number for melodic instruments
            program = 0
            if not is_drum:
                # Try to match instrument name to GM program
                name_lower = name.lower()
                for key, prog in DEFAULT_PROGRAMS.items():
                    if key in name_lower:
                        program = prog
                        break

            inst = instruments.setdefault(
                name if split_tracks else ("drums" if is_drum else name),
                pretty_midi.Instrument(program=program, is_drum=is_drum),
            )
            inst.notes.append(
                pretty_midi.Note(velocity=velocity, pitch=int(pitch), start=start, end=end)
            )
            if bar:
                bar.update(1)
    if bar:
        bar.close()

    pm.instruments.extend(instruments.values())

    if tempo_changes:
        # TODO: replace _tick_scales hack with official API when available
        pm._tick_scales = []
        for beat, tbpm in sorted(tempo_changes, key=lambda x: x[0]):
            tick = int(round(beat * pm.resolution))
            scale = 60.0 / (tbpm * pm.resolution)
            pm._tick_scales.append((tick, scale))
        if pm._tick_scales[0][0] != 0:
            pm._tick_scales.insert(0, (0, 60.0 / (bpm * pm.resolution)))
        max_tick = int(round(pattern_len * repeat * pm.resolution)) + 1
        pm._update_tick_to_time(max_tick)

    return pm


def _load_tempo_map(path: Path) -> list[list[float]]:
    """Load tempo_map.json and convert to tempo_changes format.

    Supports two formats:
    1. Standard format with entries:
        {
          "entries": [
            {"bar": 0, "beat_in_bar": 0, "time_sec": 0.0, "bpm": 120.0, "time_signature": [4, 4]},
            ...
          ]
        }
    2. Simple tempo_points format:
        {
          "tempo_points": [[time_sec, bpm], [time_sec, bpm], ...]
        }

    Returns: [[beat, bpm], [beat, bpm], ...]
    """
    data = _load_json(path)
    if not isinstance(data, dict):
        sys.exit(f"tempo_map.json must be a dict: {path}")

    # Try tempo_points format first (time_sec, bpm pairs)
    if "tempo_points" in data:
        tempo_points = data["tempo_points"]
        if not tempo_points:
            logger.warning("tempo_map.json has no tempo_points, using fixed tempo")
            return []

        # Convert time_sec to beat positions (assuming 120 BPM as base for conversion)
        # We'll use the first BPM as the initial tempo for time->beat conversion
        first_bpm = tempo_points[0][1] if tempo_points else 120.0
        tempo_changes = []

        for time_sec, bpm in tempo_points:
            # Convert time in seconds to beat position (approximate)
            beat_position = (float(time_sec) * first_bpm) / 60.0
            tempo_changes.append([beat_position, float(bpm)])

        return tempo_changes

    # Try standard entries format
    entries = data.get("entries", [])
    if not entries:
        logger.warning("tempo_map.json has no entries, using fixed tempo")
        return []

    tempo_changes = []
    for entry in entries:
        bar = entry.get("bar", 0)
        beat_in_bar = entry.get("beat_in_bar", 0)
        bpm = entry.get("bpm", 120.0)
        time_sig = entry.get("time_signature", [4, 4])
        beats_per_bar = time_sig[0] if isinstance(time_sig, list) and len(time_sig) >= 1 else 4

        # Convert bar + beat_in_bar to absolute beat position
        absolute_beat = bar * beats_per_bar + beat_in_bar
        tempo_changes.append([float(absolute_beat), float(bpm)])

    return tempo_changes


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description="Convert groove_sampler_v2 JSON to MIDI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("input_json", nargs="+", help="input JSON or glob pattern")
    ap.add_argument("-o", "--out", type=Path, default=None, help="output MIDI path")
    ap.add_argument(
        "-b", "--bpm", type=float, default=120.0, help="tempo in BPM (fallback if no tempo_map)"
    )
    ap.add_argument(
        "--tempo-map", type=Path, default=None, help="tempo_map.json for variable tempo"
    )
    ap.add_argument("--swing", type=float, default=0.0, help="swing amount 0-1")
    ap.add_argument("--humanize-timing", type=float, default=0.0, help="timing jitter ms")
    ap.add_argument("--humanize-vel", type=float, default=0.0, help="velocity jitter %%")
    ap.add_argument("--split-tracks", action="store_true", help="split tracks per instrument")
    ap.add_argument(
        "--beats-per-bar",
        type=int,
        default=4,
        help="Beats per bar when events specify {bar, beat}. Default: 4",
    )
    ap.add_argument("--repeat", type=int, default=1, help="repeat pattern")
    ap.add_argument("--seed", type=int, default=None, help="random seed")
    ap.add_argument(
        "-m",
        "--map",
        dest="mapping",
        type=Path,
        default=None,
        help="mapping file (JSON/YAML)",
    )
    ap.add_argument(
        "--humanize-config",
        type=Path,
        default=None,
        help="Optional plan_humanize.yaml path for deterministic matrix hook",
    )
    ap.add_argument(
        "--humanize-seed",
        type=int,
        default=None,
        help="Seed for deterministic humanize hook (defaults to --seed)",
    )
    ap.add_argument("-q", "--quiet", action="store_true", help="suppress progress and summary")
    ap.add_argument("-V", "--version", action="version", version=__version__)
    ns = ap.parse_args(argv)

    swing = max(0.0, min(ns.swing, 1.0))
    if swing != ns.swing:
        logger.error("Invalid swing value: %s", ns.swing)
        sys.exit(1)
    timing = max(0.0, ns.humanize_timing)
    if timing != ns.humanize_timing:
        logger.error("Invalid humanize-timing value: %s", ns.humanize_timing)
        sys.exit(1)
    vel_jitter = max(0.0, ns.humanize_vel)
    if vel_jitter != ns.humanize_vel:
        logger.error("Invalid humanize-vel value: %s", ns.humanize_vel)
        sys.exit(1)

    if ns.seed is not None:
        random.seed(ns.seed)

    humanize_config = ns.humanize_config
    if humanize_config is None:
        default_cfg = Path("configs/plan_humanize.yaml")
        if default_cfg.exists():
            humanize_config = default_cfg
    hook_seed = ns.humanize_seed if ns.humanize_seed is not None else ns.seed
    matrix_hook = MatrixHumanizeHook(humanize_config, seed=hook_seed)

    inputs: list[Path] = []
    for pattern in ns.input_json:
        matched = list(glob.glob(pattern))
        if not matched:
            sys.exit(f"Input not found: {pattern}")
        inputs.extend(Path(p) for p in matched)

    mapping = DEFAULT_PITCH_MAP
    if ns.mapping is not None:
        mapping = _load_mapping(ns.mapping)

    # Load external tempo_map if provided
    external_tempo_map = None
    if ns.tempo_map is not None:
        external_tempo_map = _load_tempo_map(ns.tempo_map)
        if not ns.quiet:
            logger.info(f"Loaded tempo_map from {ns.tempo_map} ({len(external_tempo_map)} changes)")

    single_out = ns.out if ns.out and len(inputs) == 1 else None

    for in_path in inputs:
        try:
            data = _load_json(Path(in_path))
            if isinstance(data, dict):
                # Handle arrangement_plan.json format with "tracks" array
                if "tracks" in data and isinstance(data["tracks"], list) and ns.split_tracks:
                    # For split-tracks mode, process each track separately
                    # Create a PrettyMIDI object and add each track as a separate Instrument
                    pm = pretty_midi.PrettyMIDI(initial_tempo=ns.bpm)

                    # Check if arrangement_plan.json has meta.tempo_map_path
                    tempo_changes = data.get("tempo_changes")
                    if not tempo_changes and "meta" in data:
                        meta_tempo_map = data["meta"].get("tempo_map_path")
                        if meta_tempo_map and not external_tempo_map:
                            # Auto-load tempo_map from meta
                            meta_path = Path(meta_tempo_map)
                            if not meta_path.is_absolute():
                                # Resolve relative to input file
                                meta_path = Path(in_path).parent / meta_path
                            if meta_path.exists():
                                external_tempo_map = _load_tempo_map(meta_path)
                                if not ns.quiet:
                                    logger.info(f"Auto-loaded tempo_map from meta: {meta_path}")

                    # External tempo_map overrides embedded tempo_changes
                    if external_tempo_map:
                        tempo_changes = external_tempo_map

                    # Process each track separately
                    for track_data in data["tracks"]:
                        if not isinstance(track_data, dict):
                            continue
                        track_events = track_data.get("events", [])
                        if not track_events:
                            continue

                        # Get instrument name
                        inst_name = track_data.get("instrument") or track_data.get(
                            "name", "unknown"
                        )

                        # Add instrument field to each event
                        for ev in track_events:
                            if "instrument" not in ev:
                                ev["instrument"] = inst_name

                        # Determine if drum
                        is_drum = inst_name.lower() in ["drums", "percussion"]

                        # Determine program
                        program = 0
                        if not is_drum:
                            name_lower = inst_name.lower()
                            for key, prog in DEFAULT_PROGRAMS.items():
                                if key in name_lower:
                                    program = prog
                                    break

                        # Create instrument
                        inst = pretty_midi.Instrument(
                            program=program, is_drum=is_drum, name=inst_name.capitalize()
                        )

                        # Convert events to notes
                        for ev in track_events:
                            # Get pitch - support int, float, digit string, and note name
                            pitch = None
                            if "pitch" in ev:
                                pitch = int(ev["pitch"])
                            elif "note" in ev:
                                note_val = ev["note"]
                                # Support MIDI number (int/float) or note name string
                                if isinstance(note_val, (int, float)):
                                    pitch = int(note_val)
                                else:
                                    note_str = str(note_val).strip()
                                    # Check if it's a digit string first
                                    if note_str.isdigit() or (
                                        note_str.startswith("-") and note_str[1:].isdigit()
                                    ):
                                        pitch = int(note_str)
                                    else:
                                        # Try to parse as note name (e.g., "C4", "A#3")
                                        try:
                                            pitch = pretty_midi.note_name_to_number(note_str)
                                        except Exception:
                                            logger.warning(
                                                f"Could not parse note '{note_str}' in track {inst_name}, using middle C"
                                            )
                                            pitch = 60

                            if pitch is None:
                                logger.warning(
                                    f"No pitch/note in event for track {inst_name}, skipping"
                                )
                                continue

                            # ---- Get time and duration (ABSOLUTE beats first, then convert to seconds) ----
                            # Priority: bar/beat > start_beats > time_ql > time(sec) > fallback
                            # Note: bar field indicates local bar/beat coordinates, must convert first

                            if "bar" in ev:
                                # Local (bar, beat) → absolute beats
                                # When bar is present, time_ql/time are often relative to bar start
                                bar_num = float(ev["bar"])
                                beat_in_bar = ev.get("beat", ev.get("beat_in_bar", 0.0))
                                try:
                                    beat_in_bar = 0.0 if beat_in_bar is None else float(beat_in_bar)
                                except Exception:
                                    beat_in_bar = 0.0
                                offset_beats = float(ev.get("offset_beats", 0.0) or 0.0)
                                t_beats = bar_num * ns.beats_per_bar + beat_in_bar + offset_beats
                            elif "start_beats" in ev:
                                # Absolute beat position
                                t_beats = float(ev["start_beats"])
                            elif "time_ql" in ev:
                                # Quarter note beats (already absolute if no bar field)
                                t_beats = float(ev["time_ql"])
                            elif "time" in ev:
                                # Absolute seconds → convert to beats (approximate)
                                # Use first tempo as approximation
                                first_bpm = tempo_changes[0][1] if tempo_changes else ns.bpm
                                t_beats = float(ev["time"]) * first_bpm / 60.0
                            else:
                                # Fallback to offset
                                t_beats = float(ev.get("offset", 0))

                            # ---- Get duration (beats) ----
                            # Priority: dur_beats > duration_beats > end_beats - start > duration(sec) > default

                            if "dur_beats" in ev:
                                d_beats = float(ev["dur_beats"])
                            elif "duration_beats" in ev:
                                d_beats = float(ev["duration_beats"])
                            elif "duration_ql" in ev:
                                d_beats = float(ev["duration_ql"])
                            elif "end_beats" in ev:
                                d_beats = float(ev["end_beats"]) - t_beats
                            elif "duration" in ev:
                                # Duration in seconds → convert to beats
                                first_bpm = tempo_changes[0][1] if tempo_changes else ns.bpm
                                d_beats = float(ev["duration"]) * first_bpm / 60.0
                            else:
                                # Default gate: 0.25 beats (16th note)
                                d_beats = 0.25

                            # Convert beats to seconds (simple for now, tempo_changes will be applied later via _tick_scales)
                            # For initial note timing, use a simple conversion
                            bpm_for_conversion = tempo_changes[0][1] if tempo_changes else ns.bpm
                            start = t_beats * 60.0 / bpm_for_conversion
                            end = start + d_beats * 60.0 / bpm_for_conversion

                            # Get velocity
                            if "velocity" in ev:
                                velocity = int(min(max(float(ev["velocity"]), 1), 127))
                            else:
                                velocity = int(
                                    min(max(float(ev.get("velocity_factor", 1)) * 127, 1), 127)
                                )

                            inst.notes.append(
                                pretty_midi.Note(
                                    velocity=velocity, pitch=pitch, start=start, end=end
                                )
                            )

                        pm.instruments.append(inst)

                    # Apply tempo changes
                    if tempo_changes:
                        pm._tick_scales = []
                        for beat, tbpm in sorted(tempo_changes, key=lambda x: x[0]):
                            tick = int(round(beat * pm.resolution))
                            scale = 60.0 / (tbpm * pm.resolution)
                            pm._tick_scales.append((tick, scale))
                        if pm._tick_scales[0][0] != 0:
                            pm._tick_scales.insert(0, (0, 60.0 / (ns.bpm * pm.resolution)))
                        # Estimate max tick
                        max_time = (
                            max(n.end for inst in pm.instruments for n in inst.notes)
                            if pm.instruments
                            else 10.0
                        )
                        max_tick = int(round(max_time * ns.bpm / 60.0 * pm.resolution)) + 1
                        pm._update_tick_to_time(max_tick)

                    # Write output
                    out_path = single_out if single_out else Path(in_path).with_suffix(".mid")
                    pm.write(str(out_path))
                    if not ns.quiet:
                        msg = f"Saved {out_path} (tracks: {len(pm.instruments)}, events: {sum(len(i.notes) for i in pm.instruments)})"
                        logger.info(msg)
                        print(msg)
                    continue  # Skip normal processing

                elif "tracks" in data and isinstance(data["tracks"], list):
                    # Merge all events from all tracks (non-split mode)
                    events = []
                    for track in data["tracks"]:
                        if isinstance(track, dict) and "events" in track:
                            track_events = track["events"]
                            # Add instrument field if missing
                            for ev in track_events:
                                if "instrument" not in ev:
                                    ev["instrument"] = track.get(
                                        "instrument", track.get("name", "unknown")
                                    )
                            events.extend(track_events)
                else:
                    events = data.get("events", [])

                tempo_changes = data.get("tempo_changes")

                # Check if arrangement_plan.json has meta.tempo_map_path
                if not tempo_changes and "meta" in data:
                    meta_tempo_map = data["meta"].get("tempo_map_path")
                    if meta_tempo_map and not external_tempo_map:
                        # Auto-load tempo_map from meta
                        meta_path = Path(meta_tempo_map)
                        if not meta_path.is_absolute():
                            # Resolve relative to input file
                            meta_path = Path(in_path).parent / meta_path
                        if meta_path.exists():
                            external_tempo_map = _load_tempo_map(meta_path)
                            if not ns.quiet:
                                logger.info(f"Auto-loaded tempo_map from meta: {meta_path}")

                # External tempo_map overrides embedded tempo_changes
                if external_tempo_map:
                    tempo_changes = external_tempo_map

                if not isinstance(events, list):
                    raise ValueError("Invalid events")
            else:
                events = data
                tempo_changes = external_tempo_map
            pm = convert_events(
                events,
                ns.bpm,
                mapping,
                swing=swing,
                humanize_timing_ms=timing,
                humanize_vel_pct=vel_jitter,
                split_tracks=ns.split_tracks,
                repeat=ns.repeat,
                tempo_changes=tempo_changes,
                quiet=ns.quiet,
            )
            out_path = single_out if single_out else Path(in_path).with_suffix(".mid")
            pm.write(str(out_path))
            if not ns.quiet:
                msg = f"Saved {out_path} (events: {len(events)}, bpm: {ns.bpm})"
                logger.info(msg)
                print(msg)
        except Exception as exc:
            logger.error("Failed on %s: %s", in_path, exc, exc_info=True)
            sys.exit(1)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
