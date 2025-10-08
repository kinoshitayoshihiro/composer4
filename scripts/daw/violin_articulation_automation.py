#!/usr/bin/env python3
# pyright: reportMissingTypeStubs=false
# pyright: reportUnknownMemberType=false
# pyright: reportUnknownArgumentType=false
# pyright: reportUnknownVariableType=false
# pyright: reportUnknownParameterType=false
# pyright: reportUnknownLambdaType=false
"""Utility to inject articulation automation into a MIDI file.

This helper loads a MIDI clip, inserts keyswitch notes and CC defaults based on
`configs/labels/technique_map.yaml`, and optionally writes a lightweight
DAWDreamer session description for batch rendering.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import pretty_midi
import yaml


@dataclass
class TechniqueConfig:
    instrument: str
    technique: str
    lead_time_s: float
    plugin_name: Optional[str]
    audio_out_path: Optional[Path]


def load_tech_map(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def inject_keyswitch_and_cc(
    midi: pretty_midi.PrettyMIDI,
    instrument_name: str,
    technique: str,
    tech_map: Dict[str, Any],
    lead_time_s: float,
) -> None:
    if not midi.instruments:
        raise ValueError("MIDI file does not contain any instrument tracks.")

    instruments_cfg = dict(tech_map.get("instruments", {}) or {})
    instrument_cfg_raw = instruments_cfg.get(instrument_name)
    if not isinstance(instrument_cfg_raw, dict):
        raise KeyError(f"Instrument '{instrument_name}' not found in technique map.")
    instrument_cfg: Dict[str, Any] = dict(instrument_cfg_raw)

    inst = midi.instruments[0]
    inst.name = inst.name or instrument_name
    inst.is_drum = False

    program_override = instrument_cfg.get("program")
    if program_override is not None:
        inst.program = int(program_override)

    keyswitch_map = dict(instrument_cfg.get("keyswitches", {}) or {})
    keyswitch_cfg = keyswitch_map.get(technique, {}) or {}

    cc_defaults_raw = dict(instrument_cfg.get("cc_defaults", {}) or {})
    overrides_map = dict(instrument_cfg.get("cc_overrides", {}) or {})
    cc_overrides_raw = overrides_map.get(technique, {}) or {}

    def _normalize_cc(source: Any) -> Dict[int, int]:
        result: Dict[int, int] = {}
        if isinstance(source, dict):
            for key, value in source.items():
                try:
                    result[int(key)] = int(value)
                except (TypeError, ValueError):
                    continue
        return result

    cc_defaults = _normalize_cc(cc_defaults_raw)
    cc_overrides = _normalize_cc(cc_overrides_raw)

    for cc, value in cc_defaults.items():
        inst.control_changes.append(
            pretty_midi.ControlChange(number=int(cc), value=value, time=0.0)
        )
    for cc, value in cc_overrides.items():
        inst.control_changes.append(
            pretty_midi.ControlChange(number=int(cc), value=value, time=0.0)
        )

    if isinstance(keyswitch_cfg, dict) and keyswitch_cfg:
        note_value = keyswitch_cfg.get("note")
        if note_value is None:
            raise ValueError(f"Technique '{technique}' is missing a 'note' keyswitch entry")
        ks_pitch = int(note_value)
        ks_velocity = int(keyswitch_cfg.get("velocity", 1))
        ks_duration = float(keyswitch_cfg.get("hold_ms", 30)) / 1000.0
        new_notes: List[pretty_midi.Note] = []
        for note in inst.notes:
            ks_start = max(0.0, note.start - lead_time_s)
            ks_end = ks_start + ks_duration
            ks_note = pretty_midi.Note(
                pitch=ks_pitch,
                velocity=ks_velocity,
                start=ks_start,
                end=ks_end,
            )
            new_notes.append(ks_note)
            new_notes.append(note)
        inst.notes = new_notes
    else:
        print(f"[WARN] No keyswitch mapping for technique='{technique}'.")

    # pyright: ignore[reportUnknownLambdaType]
    inst.notes.sort(key=lambda note: note.start)
    # pyright: ignore[reportUnknownLambdaType]
    inst.control_changes.sort(key=lambda change: (change.time, change.number))


def write_session_json(
    out_path: Path,
    plugin_name: str,
    midi_path: Path,
    audio_out_path: Path,
    sample_rate: int = 48_000,
    block_size: int = 1024,
) -> None:
    session = {
        "engine": {"sample_rate": sample_rate, "block_size": block_size},
        "tracks": [
            {
                "name": "violin",
                "plugins": [
                    {
                        "type": "vst3",
                        "name": plugin_name,
                        "midi_input": str(midi_path),
                        "audio_output": str(audio_out_path),
                        "params": {},
                    }
                ],
            }
        ],
    }
    out_path.write_text(json.dumps(session, ensure_ascii=False, indent=2), encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--in-midi",
        type=Path,
        required=True,
        help="Input MIDI file",
    )
    parser.add_argument(
        "--out-midi",
        type=Path,
        required=True,
        help="Output MIDI with automation",
    )
    parser.add_argument(
        "--technique",
        required=True,
        help="Technique label to apply (e.g. spiccato)",
    )
    parser.add_argument(
        "--tech-map",
        type=Path,
        required=True,
        help="Path to technique_map.yaml",
    )
    parser.add_argument(
        "--instrument",
        default="violin",
        help="Instrument key in the technique map",
    )
    parser.add_argument(
        "--lead-time",
        type=float,
        default=0.03,
        help="Seconds before note onset to place keyswitch",
    )
    parser.add_argument(
        "--session-json",
        type=Path,
        help="Optional path to write a DAWDreamer session JSON",
    )
    parser.add_argument(
        "--plugin-name",
        default="Synchron Solo Violin I",
        help="Plugin name for session JSON",
    )
    parser.add_argument(
        "--audio-out",
        type=Path,
        help="Audio path for session JSON (defaults to out-midi with .wav)",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    tech_map = load_tech_map(args.tech_map)
    midi = pretty_midi.PrettyMIDI(str(args.in_midi))

    inject_keyswitch_and_cc(
        midi=midi,
        instrument_name=args.instrument,
        technique=args.technique,
        tech_map=tech_map,
        lead_time_s=args.lead_time,
    )

    args.out_midi.parent.mkdir(parents=True, exist_ok=True)
    midi.write(str(args.out_midi))

    if args.session_json:
        audio_out = args.audio_out or args.out_midi.with_suffix(".wav")
        write_session_json(
            out_path=args.session_json,
            plugin_name=args.plugin_name,
            midi_path=args.out_midi,
            audio_out_path=audio_out,
        )


if __name__ == "__main__":
    main()
