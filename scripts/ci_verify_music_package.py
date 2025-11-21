#!/usr/bin/env python3
"""Compatibility wrapper for ops/ci_verify_music_package.py."""
from __future__ import annotations

import argparse
import pathlib
import runpy
import sys
from typing import Iterable, List

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
TARGET = REPO_ROOT / "ops" / "ci_verify_music_package.py"


def _ensure_target_exists() -> None:
    if TARGET.exists():
        return
    sys.stderr.write(f"ERROR: Missing target script: {TARGET}\n")
    sys.exit(1)


def _resolve_song_dir(raw: str) -> pathlib.Path:
    song_dir = pathlib.Path(raw).expanduser()
    if not song_dir.is_absolute():
        song_dir = (pathlib.Path.cwd() / song_dir).resolve()
    else:
        song_dir = song_dir.resolve()
    if not song_dir.exists():
        sys.stderr.write(f"ERROR: song_dir not found: {song_dir}\n")
        sys.exit(1)
    return song_dir


def _find_existing(song_dir: pathlib.Path, candidates: Iterable[str]) -> pathlib.Path | None:
    for rel in candidates:
        cand = song_dir / rel
        if cand.exists():
            return cand
    return None


def _pick_midi(song_dir: pathlib.Path) -> pathlib.Path:
    midi = _find_existing(
        song_dir,
        (
            "full_arrangement.mid",
            "full_arrangement_real.mid",
            "full_arrangement_stem.mid",
            "full_arrangement_complete.mid",
            "full_arrangement_6tracks_real.mid",
            "full_arrangement_5tracks.mid",
            "full_arranged.mid",
            "arranged.mid",
        ),
    )
    if midi:
        return midi
    sys.stderr.write(
        "ERROR: Could not locate a full arrangement MIDI under the provided song_dir.\n"
        "Create the arrangement (full_arrangement.mid) or pass --midi explicitly.\n"
    )
    sys.exit(1)


def _pick_bars(song_dir: pathlib.Path) -> pathlib.Path:
    bars = _find_existing(
        song_dir,
        (
            "bars.parquet",
            "bars_with_slots.parquet",
            "bars_with_emotion.parquet",
            "bars_extended.parquet",
            "analysis/bars.parquet",
            "analysis/bars_with_slots.parquet",
            "analysis/bars_with_emotion.parquet",
        ),
    )
    if bars:
        return bars
    sys.stderr.write(
        "ERROR: Could not locate bars*.parquet under the provided song_dir.\n"
        "Run Phase A or pass --bars explicitly.\n"
    )
    sys.exit(1)


def _extend_arg(args_list: List[str], flag: str, value: object | None) -> None:
    if value is None:
        return
    args_list.extend([flag, str(value)])


def main() -> None:
    _ensure_target_exists()

    parser = argparse.ArgumentParser(
        description=(
            "Backward-compatible entry point that infers required paths before "
            "calling ops/ci_verify_music_package.py"
        )
    )
    parser.add_argument("--song-dir", required=True, help="SongPackage root directory")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Legacy flag; kept for compatibility but currently informational only.",
    )
    parser.add_argument("--midi", help="Override path to the arrangement MIDI")
    parser.add_argument("--bars", help="Override path to bars parquet")
    parser.add_argument("--tempo-bpm", type=float, help="Forwarded to CI verifier")
    parser.add_argument("--beats-per-bar", type=float, help="Forwarded to CI verifier")
    parser.add_argument("--duration-tolerance", type=float, help="Forwarded tolerance")
    parser.add_argument(
        "--downbeats-tolerance", type=int, help="Forwarded downbeats tolerance"
    )
    parser.add_argument("--gate-config", help="KPI Gate YAML path")
    parser.add_argument("--kpi-threshold", type=float, help="KPI pass-rate threshold")
    parser.add_argument("--python-bin", help="Python interpreter for KPI Gate")
    parser.add_argument("--report", help="Override CI report output path")
    parser.add_argument("--inst-activity", action="store_true", help="Forward flag")
    parser.add_argument("--enable-crepe", action="store_true", help="Forward flag")
    parser.add_argument("--enable-oaf", action="store_true", help="Forward flag")
    parser.add_argument("--ab-csv", help="AB metric CSV output path")
    parser.add_argument("--drums-mode", help="Forward drums mode")
    args, passthrough = parser.parse_known_args()

    song_dir = _resolve_song_dir(args.song_dir)
    midi_path = pathlib.Path(args.midi).expanduser().resolve() if args.midi else _pick_midi(song_dir)
    bars_path = pathlib.Path(args.bars).expanduser().resolve() if args.bars else _pick_bars(song_dir)

    target_argv: List[str] = [str(TARGET)]
    target_argv.extend(["--midi", str(midi_path)])
    target_argv.extend(["--bars", str(bars_path)])
    target_argv.extend(["--song-dir", str(song_dir)])

    _extend_arg(target_argv, "--tempo-bpm", args.tempo_bpm)
    _extend_arg(target_argv, "--beats-per-bar", args.beats_per_bar)
    _extend_arg(target_argv, "--duration-tolerance", args.duration_tolerance)
    _extend_arg(target_argv, "--downbeats-tolerance", args.downbeats_tolerance)
    _extend_arg(target_argv, "--gate-config", args.gate_config)
    _extend_arg(target_argv, "--kpi-threshold", args.kpi_threshold)
    _extend_arg(target_argv, "--python-bin", args.python_bin)
    _extend_arg(target_argv, "--report", args.report)
    _extend_arg(target_argv, "--ab-csv", args.ab_csv)
    _extend_arg(target_argv, "--drums-mode", args.drums_mode)

    if args.inst_activity:
        target_argv.append("--inst-activity")
    if args.enable_crepe:
        target_argv.append("--enable-crepe")
    if args.enable_oaf:
        target_argv.append("--enable-oaf")

    # Preserve any additional future flags by forwarding them unchanged.
    target_argv.extend(passthrough)

    # Provide visibility when strict mode is requested (no behavioral change yet).
    if args.strict:
        print("[ci_verify] Strict mode requested; using default verifier thresholds.")

    sys.argv = target_argv
    runpy.run_path(str(TARGET), run_name="__main__")


if __name__ == "__main__":
    main()
