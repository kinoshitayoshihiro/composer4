#!/usr/bin/env python3
"""Utility presets for retrying low-scoring Stage 2 drum loops."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterator

import numpy as np

try:
    import pretty_midi as pm
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("pretty_midi is required for retry presets") from exc

# MIDI pitch sets -------------------------------------------------------------

KICK_PITCHES = {35, 36}
SNARE_PITCHES = {38, 40}
HAT_PITCHES = {42, 44, 46}


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _iter_midis(indir: Path) -> Iterator[Path]:
    for midi_path in sorted(indir.rglob("*.mid")):
        yield midi_path


def preset_velocity_smooth(
    midi: pm.PrettyMIDI,
    *,
    vmin: int = 20,
    vmax: int = 115,
    compress: float = 0.85,
) -> pm.PrettyMIDI:
    """Lift quiet notes, tame peaks, then compress towards the mean."""

    for instrument in midi.instruments:
        for note in instrument.notes:
            velocity = int(note.velocity)
            velocity = max(vmin, min(vmax, velocity))
            mean_target = 0.5 * (vmin + vmax)
            velocity = int(round((1 - compress) * velocity + compress * mean_target))
            note.velocity = max(1, min(127, velocity))
    return midi


def preset_velocity_boost(
    midi: pm.PrettyMIDI,
    *,
    lo: int = 34,
    hi: int = 120,
    compress: float = 0.7,
) -> pm.PrettyMIDI:
    """Expand the velocity range while biasing towards energetic accents."""

    lo = max(1, min(126, lo))
    hi = max(lo + 1, min(127, hi))
    mean_target = 0.5 * (lo + hi)
    for instrument in midi.instruments:
        for note in instrument.notes:
            velocity = int(note.velocity)
            scaled = lo + (velocity / 127.0) * (hi - lo)
            boosted = (1 - compress) * scaled + compress * mean_target
            note.velocity = max(1, min(127, int(round(boosted))))
    return midi


def preset_velocity_chain(
    midi: pm.PrettyMIDI,
    *,
    smooth_vmin: int = 30,
    smooth_vmax: int = 112,
    smooth_compress: float = 0.82,
    boost_lo: int = 34,
    boost_hi: int = 120,
    boost_compress: float = 0.7,
) -> pm.PrettyMIDI:
    """Apply smoothing first, then apply a range-expanding boost."""

    midi = preset_velocity_smooth(
        midi,
        vmin=smooth_vmin,
        vmax=smooth_vmax,
        compress=smooth_compress,
    )
    midi = preset_velocity_boost(
        midi,
        lo=boost_lo,
        hi=boost_hi,
        compress=boost_compress,
    )
    return midi


def _snap_pitch(pitch: int) -> int:
    if 33 <= pitch <= 47:
        if pitch in HAT_PITCHES or 41 <= pitch <= 47:
            return min(HAT_PITCHES, key=lambda value: abs(value - pitch))
        if pitch in SNARE_PITCHES or 37 <= pitch <= 41:
            return min(SNARE_PITCHES, key=lambda value: abs(value - pitch))
        if pitch in KICK_PITCHES or 33 <= pitch <= 37:
            return min(KICK_PITCHES, key=lambda value: abs(value - pitch))
    return pitch


def preset_role_snap(midi: pm.PrettyMIDI) -> pm.PrettyMIDI:
    """Snap ambiguous percussion pitches towards canonical drum roles."""

    for instrument in midi.instruments:
        for note in instrument.notes:
            note.pitch = _snap_pitch(int(note.pitch))
    return midi


def preset_structure_smooth(
    midi: pm.PrettyMIDI,
    *,
    grid_divisions: int = 16,
    window_beats: float = 1.0,
) -> pm.PrettyMIDI:
    """Rebalance per-slot density by pruning low-velocity hits from hotspots."""

    tempo = midi.estimate_tempo()
    if tempo <= 0:
        tempo = 120.0
    beat_duration = 60.0 / tempo
    cell_duration = beat_duration / 4.0 / grid_divisions

    note_bins: dict[int, list[tuple[pm.Instrument, pm.Note]]] = {}
    for instrument in midi.instruments:
        for note in list(instrument.notes):
            slot = int(note.start // cell_duration)
            note_bins.setdefault(slot, []).append((instrument, note))

    indices = np.array(sorted(note_bins.keys()))
    if indices.size == 0:
        return midi
    values = np.array([len(note_bins[idx]) for idx in indices], dtype=np.float64)
    window = max(1, int(round(window_beats * grid_divisions / 4.0)))
    kernel = np.ones(window, dtype=np.float64) / float(window)
    smoothed = np.convolve(values, kernel, mode="same")
    threshold = smoothed.mean() * 1.25

    for idx, bucket in note_bins.items():
        if len(bucket) <= threshold:
            continue
        bucket.sort(key=lambda entry: entry[1].velocity)
        remove = int(len(bucket) - threshold)
        for instrument, note in bucket[:remove]:
            instrument.notes.remove(note)
    return midi


def _process_file(midi_path: Path, outdir: Path, preset: str) -> None:
    midi = pm.PrettyMIDI(str(midi_path))
    if preset == "velocity":
        midi = preset_velocity_smooth(midi)
    elif preset == "velocity_boost":
        midi = preset_velocity_boost(midi)
    elif preset == "velocity_chain":
        midi = preset_velocity_chain(midi)
    elif preset == "rolesnap":
        midi = preset_role_snap(midi)
    elif preset == "structure":
        midi = preset_structure_smooth(midi)
    else:  # pragma: no cover - CLI guards
        raise ValueError(f"Unknown preset: {preset}")
    midi.write(str(outdir / midi_path.name))


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply retry presets to MIDI files")
    parser.add_argument("--in", dest="indir", required=True, help="Input directory")
    parser.add_argument("--out", dest="outdir", required=True, help="Output directory")
    parser.add_argument(
        "--preset",
        choices=[
            "velocity",
            "velocity_boost",
            "velocity_chain",
            "rolesnap",
            "structure",
        ],
        required=True,
        help="Preset to apply",
    )
    args = parser.parse_args()

    indir = Path(args.indir)
    outdir = Path(args.outdir)
    _ensure_dir(outdir)

    for midi_path in _iter_midis(indir):
        _process_file(midi_path, outdir, args.preset)


if __name__ == "__main__":
    main()
