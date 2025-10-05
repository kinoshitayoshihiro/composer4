#!/usr/bin/env python3
"""Generate instrument-specific JSONL manifests from the LAMDa dataset.

This script scans the META_DATA shards to extract metadata for MIDI files,
filters them by General MIDI program families (piano, guitar, bass, strings,
percussion), deduplicates entries using SIGNATURES_DATA profiles, enriches the
metadata with tempo/time-signature/key information, and records the index of the
hash within TOTALS_MATRIX when available.

Outputs are written as ``manifests/lamd_<instrument>.jsonl`` files by default,
matching the instrument categories requested by the user.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import pickle
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import (
    Dict,
    Iterable,
    Iterator,
    List,
    Mapping,
    MutableMapping,
    Optional,
    Sequence,
    Set,
    Tuple,
)

logger = logging.getLogger(__name__)

LAMDA_ROOT_DEFAULT = Path("data/Los-Angeles-MIDI")
MANIFEST_DIR_DEFAULT = Path("manifests")

PIANO_PATCHES = frozenset(range(0, 8))
GUITAR_PATCHES = frozenset(range(24, 32))
BASS_PATCHES = frozenset(range(32, 40))
STRINGS_PATCHES = frozenset(range(40, 56))
DRUM_PATCHES = frozenset(
    {
        128,
        131,
        132,
        139,
        144,
        160,
        161,
        172,
        185,
        192,
        193,
        200,
        206,
        207,
        224,
        234,
        238,
        244,
        248,
        250,
        255,
    }
)

INSTRUMENT_DEFINITIONS: Dict[str, frozenset[int]] = {
    "piano": PIANO_PATCHES,
    "guitar": GUITAR_PATCHES,
    "bass": BASS_PATCHES,
    "strings": STRINGS_PATCHES,
    "drum": DRUM_PATCHES,
}


@dataclass(frozen=True)
class ManifestConfig:
    root: Path
    meta_location: Path
    signatures_path: Path
    totals_path: Path
    out_dir: Path
    instruments: Tuple[str, ...]


def parse_args(argv: Optional[Sequence[str]] = None) -> ManifestConfig:
    parser = argparse.ArgumentParser(description="Generate instrument manifests for LAMDa")
    parser.add_argument(
        "--root",
        type=Path,
        default=LAMDA_ROOT_DEFAULT,
        help="Base directory of the LAMDa dataset (default: data/Los-Angeles-MIDI)",
    )
    parser.add_argument(
        "--meta",
        type=Path,
        help="Path to the META_DATA directory or a specific shard (default: <root>/META_DATA)",
    )
    parser.add_argument(
        "--signatures",
        type=Path,
        default=LAMDA_ROOT_DEFAULT / "SIGNATURES_DATA" / "LAMDa_SIGNATURES_DATA.pickle",
        help="Path to LAMDa_SIGNATURES_DATA.pickle",
    )
    parser.add_argument(
        "--totals",
        type=Path,
        default=LAMDA_ROOT_DEFAULT / "TOTALS_MATRIX" / "LAMDa_TOTALS.pickle",
        help="Path to LAMDa_TOTALS.pickle",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=MANIFEST_DIR_DEFAULT,
        help="Directory where manifests will be written (default: manifests)",
    )
    parser.add_argument(
        "--instrument",
        dest="instruments",
        action="append",
        choices=sorted(INSTRUMENT_DEFINITIONS.keys()),
        help="Instrument(s) to generate. Defaults to all categories if omitted.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging verbosity",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=getattr(logging, args.log_level.upper()))

    root = args.root
    meta_location = args.meta or (root / "META_DATA")

    if not root.exists():
        raise SystemExit(f"LAMDa root directory not found: {root}")
    if not args.signatures.exists():
        raise SystemExit(f"SIGNATURES_DATA pickle not found: {args.signatures}")
    if not args.totals.exists():
        raise SystemExit(f"TOTALS_MATRIX pickle not found: {args.totals}")
    if not meta_location.exists():
        raise SystemExit(f"META_DATA location not found: {meta_location}")

    instruments = (
        tuple(args.instruments) if args.instruments else tuple(INSTRUMENT_DEFINITIONS.keys())
    )

    return ManifestConfig(
        root=root,
        meta_location=meta_location,
        signatures_path=args.signatures,
        totals_path=args.totals,
        out_dir=args.out,
        instruments=instruments,
    )


def load_signatures(path: Path) -> Dict[str, Tuple[Tuple[int, int], ...]]:
    logger.info("Loading signatures from %s", path)
    with path.open("rb") as fh:
        data = pickle.load(fh)
    signatures: Dict[str, Tuple[Tuple[int, int], ...]] = {}
    for entry in data:
        if not isinstance(entry, Sequence) or len(entry) < 2:
            continue
        hash_id = entry[0]
        raw_profile = entry[1]
        if not isinstance(hash_id, str) or not isinstance(raw_profile, Sequence):
            continue
        cleaned: List[Tuple[int, int]] = []
        for item in raw_profile:
            if (
                isinstance(item, Sequence)
                and len(item) >= 2
                and isinstance(item[0], (int, float))
                and isinstance(item[1], (int, float))
            ):
                cleaned.append((int(item[0]), int(item[1])))
        signatures[hash_id] = tuple(cleaned)
    logger.info("Loaded %d signature profiles", len(signatures))
    return signatures


def load_totals_index(path: Path) -> Dict[str, int]:
    logger.info("Loading totals index from %s", path)
    with path.open("rb") as fh:
        data = pickle.load(fh)
    if not isinstance(data, Sequence) or len(data) < 2:
        logger.warning("Unrecognised TOTALS structure; skipping index mapping")
        return {}
    ids = data[1]
    if not isinstance(ids, Sequence):
        logger.warning("TOTALS hash list is missing or invalid; skipping index mapping")
        return {}
    lookup: Dict[str, int] = {}
    for idx, value in enumerate(ids):
        if isinstance(value, str) and value not in lookup:
            lookup[value] = idx
    logger.info("Created totals lookup for %d ids", len(lookup))
    return lookup


def iter_meta_records(meta_location: Path) -> Iterator[Tuple[str, List[Sequence[object]]]]:
    if meta_location.is_file():
        targets = [meta_location]
    else:
        targets = sorted(meta_location.glob("LAMDa_META_DATA_*.pickle"))
    if not targets:
        raise SystemExit(f"No META_DATA pickle files found under {meta_location}")

    for path in targets:
        logger.debug("Loading META shard %s", path)
        with path.open("rb") as fh:
            data = pickle.load(fh)
        if not isinstance(data, Sequence):
            logger.warning("Skipping malformed META shard: %s", path)
            continue
        for entry in data:
            if isinstance(entry, Sequence) and len(entry) == 2 and isinstance(entry[0], str):
                yield entry[0], entry[1]


def to_dict_safely(pairs_or_events: Iterable[Sequence[object]]) -> Dict[str, object]:
    """Convert the leading key/value pairs into a dict, stop at the first MIDI event."""

    stop_tags = {"note", "control_change", "patch_change", "track_name", "text_event", "marker"}
    result: Dict[str, object] = {}
    for item in pairs_or_events:
        if not isinstance(item, Sequence) or not item:
            break
        tag = item[0]
        if not isinstance(tag, str):
            break
        if tag in stop_tags:
            break
        if len(item) >= 2:
            value = item[1] if len(item) == 2 else list(item[1:])
            result.setdefault(tag, value)
    return result


def signature_digest(profile: Tuple[Tuple[int, int], ...]) -> str:
    hasher = hashlib.sha1()
    for pitch, weight in profile:
        hasher.update(pitch.to_bytes(2, "big", signed=False))
        hasher.update(weight.to_bytes(4, "big", signed=False))
    return hasher.hexdigest()


KEY_NAMES_MAJOR = {
    -7: "Cb",
    -6: "Gb",
    -5: "Db",
    -4: "Ab",
    -3: "Eb",
    -2: "Bb",
    -1: "F",
    0: "C",
    1: "G",
    2: "D",
    3: "A",
    4: "E",
    5: "B",
    6: "F#",
    7: "C#",
}
KEY_NAMES_MINOR = {
    -7: "Abm",
    -6: "Ebm",
    -5: "Bbm",
    -4: "Fm",
    -3: "Cm",
    -2: "Gm",
    -1: "Dm",
    0: "Am",
    1: "Em",
    2: "Bm",
    3: "F#m",
    4: "C#m",
    5: "G#m",
    6: "D#m",
    7: "A#m",
}


def parse_time_signature(value: object) -> Optional[str]:
    if not isinstance(value, Sequence) or len(value) < 3:
        return None
    numerator = value[1]
    denominator_power = value[2]
    if not isinstance(numerator, (int, float)) or not isinstance(denominator_power, (int, float)):
        return None
    denominator = 2 ** int(denominator_power)
    return f"{int(numerator)}/{denominator}"


def parse_key_signature(value: object) -> Optional[str]:
    if not isinstance(value, Sequence) or len(value) < 3:
        return None
    key = value[1]
    scale = value[2]
    if not isinstance(key, (int, float)) or not isinstance(scale, (int, float)):
        return None
    key = int(key)
    scale = int(scale)
    mapping = KEY_NAMES_MINOR if scale else KEY_NAMES_MAJOR
    return mapping.get(key)


def parse_tempo(value: object) -> Tuple[Optional[float], Optional[int]]:
    if not isinstance(value, Sequence) or len(value) < 2:
        return None, None
    us_per_beat = value[1]
    if not isinstance(us_per_beat, (int, float)) or us_per_beat <= 0:
        return None, None
    tempo_microseconds = int(us_per_beat)
    bpm = 60_000_000 / tempo_microseconds
    return bpm, tempo_microseconds


def resolve_midi_path(root: Path, song_id: str, meta_map: Mapping[str, object]) -> Optional[Path]:
    midi_path = root / "MIDIs" / song_id[0] / f"{song_id}.mid"
    if midi_path.exists():
        return midi_path

    alt_path = meta_map.get("midi_path")
    if isinstance(alt_path, str):
        candidate = Path(alt_path)
        if not candidate.is_absolute():
            candidate = root / candidate
        if candidate.exists():
            return candidate

    logger.debug("MIDI file missing for %s", song_id)
    return None


def build_meta_payload(
    meta_map: Mapping[str, object], totals_index: Optional[int], relevant_patches: Set[int]
) -> Dict[str, object]:
    tempo_bpm, tempo_microseconds = parse_tempo(meta_map.get("set_tempo"))
    time_signature = parse_time_signature(meta_map.get("time_signature"))
    key_signature = parse_key_signature(meta_map.get("key_signature"))

    patches = meta_map.get("midi_patches")
    patch_list: List[int] = []
    if isinstance(patches, Sequence):
        for value in patches:
            if isinstance(value, (int, float)):
                patch_list.append(int(value))

    payload: Dict[str, object] = {
        "patches": sorted(set(patch_list)),
        "matched_patches": sorted(relevant_patches & set(patch_list)),
        "tempo_bpm": tempo_bpm,
        "tempo_us_per_beat": tempo_microseconds,
        "time_signature": time_signature,
        "key_signature": key_signature,
        "tempo_change_count": meta_map.get("tempo_change_count"),
        "total_number_of_tracks": meta_map.get("total_number_of_tracks"),
        "total_number_of_opus_midi_events": meta_map.get("total_number_of_opus_midi_events"),
        "total_number_of_chords": meta_map.get("total_number_of_chords"),
        "totals_index": totals_index,
    }
    return {key: value for key, value in payload.items() if value is not None}


def generate_manifests(config: ManifestConfig) -> Dict[str, Dict[str, int]]:
    signatures = load_signatures(config.signatures_path)
    signature_hashes = {
        hash_id: signature_digest(profile) for hash_id, profile in signatures.items()
    }
    totals_lookup = load_totals_index(config.totals_path)

    config.out_dir.mkdir(parents=True, exist_ok=True)
    writers = {
        instrument: (config.out_dir / f"lamd_{instrument}.jsonl").open("w", encoding="utf-8")
        for instrument in config.instruments
    }

    seen_signatures: Dict[str, Set[str]] = {instrument: set() for instrument in config.instruments}
    stats: Dict[str, Dict[str, int]] = {
        instrument: defaultdict(int)  # type: ignore[assignment]
        for instrument in config.instruments
    }

    try:
        for song_id, details in iter_meta_records(config.meta_location):
            meta_map = to_dict_safely(details)
            if not meta_map:
                continue
            signature_profile = signatures.get(song_id)
            signature_hash = signature_hashes.get(song_id)
            if signature_profile is None or signature_hash is None:
                logger.debug("Missing signature for %s; skipping", song_id)
                continue

            patches_value = meta_map.get("midi_patches")
            if not isinstance(patches_value, Sequence):
                continue
            patches = {int(p) for p in patches_value if isinstance(p, (int, float))}

            midi_path = resolve_midi_path(config.root, song_id, meta_map)
            if midi_path is None:
                for instrument in config.instruments:
                    stats[instrument]["missing_path"] += 1
                continue

            totals_index = totals_lookup.get(song_id)

            for instrument in config.instruments:
                patch_family = INSTRUMENT_DEFINITIONS[instrument]
                matched = patches & patch_family
                if not matched:
                    continue

                if signature_hash in seen_signatures[instrument]:
                    stats[instrument]["duplicates_skipped"] += 1
                    continue

                payload = {
                    "id": song_id,
                    "path": str(midi_path),
                    "instrument": instrument,
                    "source": "LAMDa",
                    "signature_digest": signature_hash,
                    "meta": build_meta_payload(meta_map, totals_index, matched),
                }
                json.dump(payload, writers[instrument], ensure_ascii=False)
                writers[instrument].write("\n")

                seen_signatures[instrument].add(signature_hash)
                stats[instrument]["written"] += 1
    finally:
        for fh in writers.values():
            fh.close()

    return {instrument: dict(counter) for instrument, counter in stats.items()}


def main(argv: Optional[Sequence[str]] = None) -> int:
    config = parse_args(argv)
    stats = generate_manifests(config)
    for instrument, counter in stats.items():
        logger.info(
            "%s: written=%d duplicates=%d missing_path=%d",
            instrument,
            counter.get("written", 0),
            counter.get("duplicates_skipped", 0),
            counter.get("missing_path", 0),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
