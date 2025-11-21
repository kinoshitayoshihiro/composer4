#!/usr/bin/env python3
"""Generate Lamda-compatible manifest rows for the XMIDI emotion dataset.

The script scans XMIDI MIDI files, derives deterministic Lamda IDs, attaches
valence/arousal metadata from ``config/xmidi_mapping.yaml``, and produces:

1. ``manifests/lamd_xmidi.jsonl`` – Lamda-style manifest rows.
2. ``outputs/stage3/xmidi_labels.csv`` – CSV that Stage3 ``collect_conditions`` expects.

Example::

    PYTHONPATH=. .venv311/bin/python scripts/build_xmidi_manifest.py \
        --xmidi-root data/test_xmidi_small \
        --output-manifest manifests/lamd_xmidi_sample.jsonl \
        --output-labels outputs/stage3/xmidi_labels_sample.csv \
        --sample-limit 50
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import yaml

CHUNK_SIZE = 2**20  # 1 MiB


@dataclass
class XMIDIMapping:
    emotions: dict[str, dict[str, Any]]
    genres: dict[str, dict[str, Any]]
    id_namespace: str
    defaults: dict[str, Any]
    source: str


@dataclass
class DrumLabelMapping:
    defaults: dict[str, Any]
    labels: dict[str, dict[str, Any]]
    aliases: dict[str, str]


def load_mapping(path: Path) -> XMIDIMapping:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return XMIDIMapping(
        emotions={k: v for k, v in data.get("emotions", {}).items()},
        genres={k: v for k, v in data.get("genres", {}).items()},
        id_namespace=data.get("id_namespace", "xmidi"),
        defaults=data.get("defaults", {}),
        source=data.get("source", "XMIDI"),
    )


def load_drum_mapping(path: Path | None) -> DrumLabelMapping | None:
    if path is None or not path.exists():
        return None
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return DrumLabelMapping(
        defaults=data.get("defaults", {}),
        labels=data.get("labels", {}),
        aliases=data.get("aliases", {}),
    )


def list_xmidi_files(root: Path) -> list[Path]:
    patterns = ("XMIDI_*.mid", "XMIDI_*.midi")
    files: list[Path] = []
    for pattern in patterns:
        files.extend(root.rglob(pattern))
    files = [f for f in files if f.is_file()]
    files.sort()
    return files


def parse_xmidi_tokens(path: Path) -> tuple[str, str, str]:
    name = path.stem
    parts = name.split("_")
    if len(parts) < 4 or parts[0] != "XMIDI":
        raise ValueError(f"Unexpected XMIDI filename: {path.name}")
    emotion = parts[1].lower()
    genre = parts[2].lower()
    clip_id = "_".join(parts[3:])
    return emotion, genre, clip_id


def deterministic_id(namespace: str, rel_path: str) -> str:
    encoded = f"{namespace}:{rel_path}".encode("utf-8")
    return hashlib.sha1(encoded).hexdigest()


def compute_signature(path: Path, mode: str) -> str:
    hasher = hashlib.sha1()
    if mode == "content":
        with path.open("rb") as fh:
            while chunk := fh.read(CHUNK_SIZE):
                hasher.update(chunk)
    elif mode == "path":
        hasher.update(path.as_posix().encode("utf-8"))
    else:
        raise ValueError(f"Unsupported signature mode: {mode}")
    return hasher.hexdigest()


def ensure_relative(path: Path, base: Path) -> str:
    try:
        rel = path.relative_to(base)
        return rel.as_posix()
    except ValueError:
        return path.as_posix()


def match_drum_label(
    emotion: str,
    genre: str,
    valence: float,
    arousal: float,
    drum_mapping: DrumLabelMapping | None,
) -> tuple[str, dict[str, Any], dict[str, float]]:
    """Match XMIDI metadata to drum label.

    Returns:
        (label_name, drum_traits, axis_bias)
    """
    if drum_mapping is None:
        return "neutral_pocket", {}, {}

    # Try each label in order
    for label_name, label_spec in drum_mapping.labels.items():
        match_spec = label_spec.get("match", {})

        # Check emotion match
        emotions = match_spec.get("emotion", [])
        if emotions and emotion not in emotions:
            continue

        # Check genre match (optional)
        genres = match_spec.get("genre", [])
        if genres and genre not in genres:
            continue

        # Check arousal bounds (optional)
        arousal_min = match_spec.get("arousal_min")
        arousal_max = match_spec.get("arousal_max")
        if arousal_min is not None and arousal < arousal_min:
            continue
        if arousal_max is not None and arousal > arousal_max:
            continue

        # Check valence bounds (optional)
        valence_min = match_spec.get("valence_min")
        valence_max = match_spec.get("valence_max")
        if valence_min is not None and valence < valence_min:
            continue
        if valence_max is not None and valence > valence_max:
            continue

        # Match found
        drum_traits = label_spec.get("drum_traits", {})
        axis_bias = label_spec.get("axis_bias", {})
        return label_name, drum_traits, axis_bias

    # No match - use fallback
    fallback = drum_mapping.defaults.get("fallback_label", "neutral_pocket")
    return fallback, {}, {}


def write_manifest(entries: Iterable[dict[str, Any]], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as fh:
        for entry in entries:
            fh.write(json.dumps(entry, ensure_ascii=False))
            fh.write("\n")


def write_labels(rows: list[dict[str, Any]], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = (
        list(rows[0].keys())
        if rows
        else [
            "loop_id",
            "emotion",
            "genre",
            "valence",
            "arousal",
            "clip_id",
            "relative_path",
        ]
    )
    with destination.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Lamda manifest for XMIDI dataset")
    parser.add_argument(
        "--xmidi-root",
        type=Path,
        default=Path("data/XMIDI_Dataset"),
        help="Directory that contains XMIDI_*.mid(i) files",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Root used to derive relative manifest paths",
    )
    parser.add_argument(
        "--mapping",
        type=Path,
        default=Path("config/xmidi_mapping.yaml"),
        help="YAML file with emotion/genre metadata",
    )
    parser.add_argument(
        "--drum-mapping",
        type=Path,
        help="YAML file with drum label mapping (optional)",
    )
    parser.add_argument(
        "--output-manifest",
        type=Path,
        default=Path("manifests/lamd_xmidi.jsonl"),
        help="Output JSONL manifest path",
    )
    parser.add_argument(
        "--output-labels",
        type=Path,
        default=Path("outputs/stage3/xmidi_labels.csv"),
        help="Output CSV for Stage3 collect_conditions",
    )
    parser.add_argument(
        "--instrument",
        type=str,
        default=None,
        help="Instrument label to store in manifest (defaults to mapping value)",
    )
    parser.add_argument(
        "--signature-mode",
        choices=["content", "path"],
        default=None,
        help="Digest mode for signature_digest (defaults to mapping value)",
    )
    parser.add_argument(
        "--sample-limit",
        type=int,
        help="Process only the first N files (debugging aid)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    mapping = load_mapping(args.mapping)
    drum_mapping = load_drum_mapping(args.drum_mapping) if args.drum_mapping else None
    xmidi_root = args.xmidi_root.expanduser().resolve()
    project_root = args.project_root.expanduser().resolve()

    files = list_xmidi_files(xmidi_root)
    if args.sample_limit is not None:
        files = files[: args.sample_limit]
    if not files:
        raise SystemExit(f"No XMIDI files found under {xmidi_root}")

    manifest_entries: list[dict[str, Any]] = []
    label_rows: list[dict[str, Any]] = []

    instrument = args.instrument or mapping.defaults.get("instrument", "arrangement")
    signature_mode = args.signature_mode or mapping.defaults.get("signature_digest_mode", "content")
    include_file_stats = mapping.defaults.get("include_file_stats", True)

    for idx, midi_path in enumerate(files, 1):
        emotion, genre, clip_id = parse_xmidi_tokens(midi_path)
        emotion_meta = mapping.emotions.get(emotion)
        if not emotion_meta:
            logging.warning("Unknown emotion '%s' in %s; skipping", emotion, midi_path.name)
            continue
        genre_meta = mapping.genres.get(genre, {})

        rel_path = ensure_relative(midi_path, project_root)
        loop_id = deterministic_id(mapping.id_namespace, rel_path)
        signature_digest = compute_signature(midi_path, signature_mode)
        stats = midi_path.stat()
        timestamp = datetime.fromtimestamp(stats.st_mtime, tz=timezone.utc).isoformat()

        valence = emotion_meta["valence"]
        arousal = emotion_meta["arousal"]

        # Match drum label
        drum_label, drum_traits, axis_bias = match_drum_label(
            emotion, genre, valence, arousal, drum_mapping
        )

        meta: dict[str, Any] = {
            "dataset": mapping.source,
            "emotion": emotion,
            "genre": genre,
            "clip_id": clip_id,
            "valence": valence,
            "arousal": arousal,
            "emotion_meta": emotion_meta,
            "genre_meta": genre_meta,
            "drum_label": drum_label,
            "drum_traits": drum_traits,
            "axis_bias": axis_bias,
            "drum_label_source": "matched" if drum_traits else "fallback",
        }
        if include_file_stats:
            meta["file_size_bytes"] = stats.st_size
            meta["modified_utc"] = timestamp

        entry = {
            "id": loop_id,
            "path": rel_path,
            "instrument": instrument,
            "source": mapping.defaults.get("source", mapping.source),
            "signature_digest": signature_digest,
            "meta": meta,
        }
        manifest_entries.append(entry)

        label_rows.append(
            {
                "loop_id": loop_id,
                "emotion": emotion,
                "genre": genre,
                "valence": valence,
                "arousal": arousal,
                "clip_id": clip_id,
                "relative_path": rel_path,
                "drum_label": drum_label,
                "drum_traits_json": json.dumps(drum_traits, ensure_ascii=False),
                "axis_bias_json": json.dumps(axis_bias, ensure_ascii=False),
            }
        )

        if idx % 1000 == 0:
            logging.info("Processed %d files", idx)

    write_manifest(manifest_entries, args.output_manifest)
    write_labels(label_rows, args.output_labels)

    logging.info(
        "Finished XMIDI manifest: %d entries → %s", len(manifest_entries), args.output_manifest
    )
    logging.info("Stage3 labels CSV written to %s", args.output_labels)
    if drum_mapping:
        logging.info(
            "Drum labels applied: %d unique labels", len(set(r["drum_label"] for r in label_rows))
        )


if __name__ == "__main__":
    main()
