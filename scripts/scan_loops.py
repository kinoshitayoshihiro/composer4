import argparse
import csv
import logging
import re
from pathlib import Path
from tempfile import NamedTemporaryFile

import pretty_midi

from utilities.groove_sampler_v2 import _safe_read_bpm, mido
from utilities.pretty_midi_safe import pm_to_mido

if mido is None:  # pragma: no cover - dependency is required
    raise RuntimeError("mido is required for loop scanning; install via 'pip install mido'")

logger = logging.getLogger(__name__)


def _ensure_tempo(pm: pretty_midi.PrettyMIDI, bpm: float, tempo_source: str) -> pretty_midi.PrettyMIDI:
    """Return ``pm`` ensuring a tempo meta message exists.

    The input ``pm`` is converted to :mod:`mido` to inspect tempo messages. If
    none are present, a ``set_tempo`` is inserted at time ``0`` and the MIDI is
    reloaded via a temporary file.  The original ``pm`` is never modified in
    place. The returned object carries ``tempo_injected`` and ``tempo_source``
    attributes.
    """

    def _pm_to_mido_raw(midi_pm: pretty_midi.PrettyMIDI) -> "mido.MidiFile":
        if mido is None:  # pragma: no cover - handled at import time
            raise RuntimeError("mido is required for tempo injection")
        tmp_path: str | None = None
        with NamedTemporaryFile(suffix=".mid", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            if not hasattr(midi_pm, "write"):
                raise RuntimeError("PrettyMIDI.write unavailable in this environment")
            midi_pm.write(tmp_path)
            return mido.MidiFile(tmp_path)
        finally:
            if tmp_path:
                try:
                    Path(tmp_path).unlink(missing_ok=True)  # type: ignore[arg-type]
                except Exception:
                    pass

    midi = _pm_to_mido_raw(pm)

    def _track_has_explicit_tempo(track) -> bool:
        return any(getattr(msg, "type", "") == "set_tempo" for msg in track)

    target_idx = 1 if len(midi.tracks) > 1 else 0
    if target_idx < len(midi.tracks) and _track_has_explicit_tempo(midi.tracks[target_idx]):
        setattr(pm, "tempo_injected", False)
        setattr(pm, "tempo_source", tempo_source)
        return pm

    enriched_midi = pm_to_mido(pm)

    tmp_path: str | None = None
    with NamedTemporaryFile(suffix=".mid", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        enriched_midi.save(tmp_path)
        pm2 = pretty_midi.PrettyMIDI(tmp_path)
    finally:
        if tmp_path:
            try:
                Path(tmp_path).unlink(missing_ok=True)  # type: ignore[arg-type]
            except Exception:
                pass

    setattr(pm2, "tempo_injected", True)
    setattr(pm2, "tempo_source", tempo_source)
    return pm2


def _analyze(path: Path):
    logger.info("Analyzing %s", path)
    pm = pretty_midi.PrettyMIDI(str(path))
    try:
        _times, tempi = pm.get_tempo_changes()
        if len(tempi) == 0 or not float(tempi[0]) > 0:
            scale = 60.0 / (120.0 * pm.resolution)
            pm._tick_scales = [(0, scale)]
            if hasattr(pm, "_update_tick_to_time"):
                pm._update_tick_to_time(pm.resolution)
    except Exception:
        pass
    bpm = _safe_read_bpm(pm, default_bpm=120.0, fold_halves=False)
    source = getattr(_safe_read_bpm, "last_source", "file")
    if source == "default":
        logger.warning("Tempo not found in %s; using default 120 BPM", path)
    pm = _ensure_tempo(pm, bpm, source)
    midi = pm_to_mido(pm)
    ticks_per_beat = midi.ticks_per_beat
    total_ticks = max(sum(msg.time for msg in tr) for tr in midi.tracks)
    ts_msg = None
    for tr in midi.tracks:
        for msg in tr:
            if msg.type == "time_signature":
                ts_msg = msg
                break
        if ts_msg:
            break
    beats_per_bar = ts_msg.numerator * (4 / ts_msg.denominator) if ts_msg else 4.0
    bars = total_ticks / (ticks_per_beat * beats_per_bar)
    all_notes = [n for inst in pm.instruments for n in inst.notes]
    notes = len(all_notes)
    uniq = len({n.pitch for n in all_notes})
    is_drum = any(inst.is_drum for inst in pm.instruments)
    is_fill = bool(re.search(r"\bfill\b", path.name, re.IGNORECASE))
    keep_drum = is_drum
    keep_pitched = not is_drum
    return [
        str(path),
        path.stat().st_size,
        f"{bpm:.2f}",
        f"{bars:.2f}",
        notes,
        uniq,
        int(is_drum),
        int(is_fill),
        int(keep_drum),
        int(keep_pitched),
        int(getattr(pm, "tempo_injected", False)),
        getattr(pm, "tempo_source", source),
    ]


def main():
    ap = argparse.ArgumentParser(description="Scan loops and write inventory CSV")
    ap.add_argument("--root", type=Path, default=Path("data/loops"))
    ap.add_argument("--out", type=Path, default=Path("loop_inventory.csv"))
    ns = ap.parse_args()

    rows = []
    for p in ns.root.rglob("*.mid"):
        try:
            rows.append(_analyze(p))
        except Exception as exc:  # pragma: no cover
            logger.warning("Failed to analyze %s: %s", p, exc)
    injected = sum(r[-2] for r in rows)
    logger.info("tempo injected on %d/%d files", injected, len(rows))
    with ns.out.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "path",
                "bytes",
                "bpm",
                "bars",
                "notes",
                "uniq_pitches",
                "drum",
                "is_fill",
                "keep_drum",
                "keep_pitched",
                "tempo_injected",
                "tempo_source",
            ]
        )
        writer.writerows(rows)


if __name__ == "__main__":  # pragma: no cover
    main()
