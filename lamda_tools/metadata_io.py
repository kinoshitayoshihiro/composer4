"""Helpers for working with sharded LAMDa drum-loop metadata."""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

DEFAULT_LIGHT_COLUMNS: Sequence[str] = (
    "md5",
    "filename",
    "genre",
    "bpm",
    "note_count",
    "duration_ticks",
    "metrics.swing_ratio",
    "metrics.ghost_rate",
    "metrics.layering_rate",
    "metrics.velocity_mean",
)


def load_metadata_index(path: Path) -> Dict[str, Any]:
    """Load the aggregated index pickle produced by the drum-loop builder."""

    resolved = Path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"Metadata index not found: {resolved}")
    with resolved.open("rb") as stream:
        return pickle.load(stream)


def _resolve_shard_path(
    shard_entry: Dict[str, Any],
    metadata_dir: Optional[Path],
    index_path: Optional[Path],
) -> Path:
    raw_path = Path(shard_entry.get("path", ""))
    candidates = [raw_path]
    if metadata_dir is not None:
        candidates.append(metadata_dir / raw_path.name)
        candidates.append(metadata_dir / raw_path)
    if index_path is not None:
        base = Path(index_path).resolve().parent
        candidates.append(base / raw_path.name)
        candidates.append(base / raw_path)
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return raw_path.resolve()


def iter_loop_records(
    index_data: Dict[str, Any],
    *,
    metadata_dir: Optional[Path] = None,
    index_path: Optional[Path] = None,
) -> Iterator[Dict[str, Any]]:
    """Yield loop entries across all shards with attached shard metadata.

    Supports both aggregated index files that reference shard pickle files and
    direct shard payloads that contain a ``loops`` list at the top level.
    """

    shards = index_data.get("shards", []) or []
    if shards:
        for shard_entry in shards:
            shard_path = _resolve_shard_path(shard_entry, metadata_dir, index_path)
            if not shard_path.exists():
                continue
            with shard_path.open("rb") as stream:
                shard_payload = pickle.load(stream)
            loops = shard_payload.get("loops", []) or []
            shard_summary = (
                shard_entry.get("metrics_summary")
                or shard_payload.get("summary")
                or shard_payload.get("metrics_summary")
            )
            shard_index = shard_entry.get("index")
            for loop in loops:
                yield {
                    "loop": loop,
                    "shard_index": shard_index,
                    "shard_path": str(shard_path),
                    "shard_summary": shard_summary,
                }
        return

    loops = index_data.get("loops", []) or []
    if not loops:
        return

    shard_path = None
    if index_path is not None:
        shard_path = str(index_path.resolve())
    shard_index = index_data.get("shard_index")
    shard_summary = index_data.get("summary") or index_data.get("metrics_summary")

    for loop in loops:
        yield {
            "loop": loop,
            "shard_index": shard_index,
            "shard_path": shard_path,
            "shard_summary": shard_summary,
        }


def flatten_loop_record(
    record: Dict[str, Any],
    *,
    metric_keys: Optional[Sequence[str]] = None,
    include_paths: bool = True,
    include_instrument_distribution: bool = False,
    light: bool = False,
) -> Dict[str, Any]:
    """Flatten a loop record (with shard metadata) into a simple row."""

    loop: Dict[str, Any] = record.get("loop", {})
    metrics: Dict[str, Any] = loop.get("metrics", {}) or {}
    flat: Dict[str, Any] = {
        "md5": loop.get("md5"),
        "filename": loop.get("filename"),
        "genre": loop.get("genre"),
        "bpm": loop.get("bpm"),
        "note_count": loop.get("note_count"),
        "duration_ticks": loop.get("duration_ticks"),
        "shard_index": record.get("shard_index"),
        "shard_path": record.get("shard_path"),
    }
    if include_paths and not light:
        flat["input_path"] = loop.get("input_path")
        flat["output_path"] = loop.get("output_path")
    pitches = loop.get("pitches") or {}
    if not light:
        flat["pitches.sum"] = pitches.get("sum")
        flat["pitches.counts"] = json.dumps(pitches.get("counts"), ensure_ascii=False)
        flat["pitches.distribution"] = json.dumps(pitches.get("distribution"), ensure_ascii=False)
    keys = metric_keys or sorted(metrics.keys())
    for key in keys:
        flat[f"metrics.{key}"] = metrics.get(key)
    if include_instrument_distribution and "instrument_distribution" in metrics:
        flat["metrics.instrument_distribution"] = json.dumps(
            metrics.get("instrument_distribution"), ensure_ascii=False
        )
    if light:
        light_keys = set(DEFAULT_LIGHT_COLUMNS)
        light_keys.update({"shard_index", "shard_path"})
        return {key: flat.get(key) for key in light_keys if key in flat}
    return flat


def collect_flat_rows(
    index_path: Path,
    metadata_dir: Optional[Path] = None,
    *,
    metric_keys: Optional[Sequence[str]] = None,
    limit: Optional[int] = None,
    light: bool = False,
    include_paths: bool = True,
    include_distribution: bool = False,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Load and flatten metadata rows from an index file."""

    index_data = load_metadata_index(index_path)
    rows: List[Dict[str, Any]] = []
    for entry in iter_loop_records(
        index_data,
        metadata_dir=metadata_dir,
        index_path=index_path,
    ):
        rows.append(
            flatten_loop_record(
                entry,
                metric_keys=metric_keys,
                include_paths=include_paths,
                include_instrument_distribution=include_distribution,
                light=light,
            )
        )
        if limit is not None and len(rows) >= limit:
            break
    return rows, index_data
