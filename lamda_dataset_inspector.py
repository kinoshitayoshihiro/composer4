"""Utility helpers to inspect supplementary LAMDa dataset pickles.

This module provides lightweight summaries for the auxiliary pickle files that
ship with the Los Angeles MIDI (LAMDa) dataset:

* ``KILO_CHORDS_DATA``
* ``SIGNATURES_DATA``
* ``TOTALS_MATRIX``
* ``META_DATA`` (multiple shards)

The goal is to expose quick statistics without forcing callers to load the full
data into memory.  Each summariser uses a configurable sample window and guards
against missing files so that it can be safely plugged into setup diagnostics or
unit tests.
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import pickle
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

# Reasonable cap to prevent huge in-memory processing when computing statistics
DEFAULT_SAMPLE_SIZE = 1_000
META_SAMPLE_SIZE = 5_000


def _load_pickle(path: Path) -> Any:
    with path.open("rb") as fp:
        return pickle.load(fp)


def _safe_mean(values: Sequence[float]) -> float:
    return float(statistics.fmean(values)) if values else 0.0


def _normalise_path(base: Path, relative: str) -> Path:
    path = base / relative
    return path if path.exists() else Path(relative)


class LAMDaDatasetInspector:
    """Produce summaries for auxiliary LAMDa pickle datasets."""

    def __init__(self, base_dir: Path) -> None:
        self.base_dir = Path(base_dir)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def summarize_all(self) -> Dict[str, Dict[str, Any]]:
        """Return summaries for every discoverable supplementary dataset."""

        summaries: Dict[str, Dict[str, Any]] = {}

        kilo_summary = self.summarize_kilo_chords()
        if kilo_summary:
            summaries["kilo_chords"] = kilo_summary

        signature_summary = self.summarize_signatures()
        if signature_summary:
            summaries["signatures"] = signature_summary

        totals_summary = self.summarize_totals_matrix()
        if totals_summary:
            summaries["totals_matrix"] = totals_summary

        meta_summary = self.summarize_meta_data()
        if meta_summary:
            summaries["meta_data"] = meta_summary

        return summaries

    # ------------------------------------------------------------------
    def summarize_kilo_chords(
        self,
        sample_size: int = DEFAULT_SAMPLE_SIZE,
        pickle_path: Optional[Path | str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Summarize the ``KILO_CHORDS_DATA`` pickle."""

        resolved_path = (
            Path(pickle_path)
            if pickle_path
            else _normalise_path(self.base_dir, "KILO_CHORDS_DATA/LAMDa_KILO_CHORDS_DATA.pickle")
        )
        if not resolved_path.exists():
            logger.debug("KILO_CHORDS_DATA pickle not found at %s", resolved_path)
            return None

        data: List[Sequence[Any]] = _load_pickle(resolved_path)
        total_entries = len(data)
        if total_entries == 0:
            return {"entries": 0}

        sample_entries = data[: min(sample_size, total_entries)]
        entry_lengths: List[int] = []
        pitch_counter: Counter[int] = Counter()

        for entry in sample_entries:
            if len(entry) < 2 or not isinstance(entry[1], Sequence):
                continue
            vector = entry[1]
            entry_lengths.append(len(vector))
            pitch_counter.update(
                itertools.islice((int(v) for v in vector if isinstance(v, (int, float))), 256)
            )

        common_pitches = pitch_counter.most_common(5)

        return {
            "entries": total_entries,
            "sampled_entries": len(sample_entries),
            "avg_vector_length": _safe_mean(entry_lengths),
            "median_vector_length": statistics.median(entry_lengths) if entry_lengths else 0,
            "top_pitches": common_pitches,
        }

    # ------------------------------------------------------------------
    def summarize_signatures(
        self,
        sample_size: int = DEFAULT_SAMPLE_SIZE,
        pickle_path: Optional[Path | str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Summarize the ``SIGNATURES_DATA`` pickle."""

        resolved_path = (
            Path(pickle_path)
            if pickle_path
            else _normalise_path(self.base_dir, "SIGNATURES_DATA/LAMDa_SIGNATURES_DATA.pickle")
        )
        if not resolved_path.exists():
            logger.debug("SIGNATURES_DATA pickle not found at %s", resolved_path)
            return None

        data: List[Sequence[Any]] = _load_pickle(resolved_path)
        total_entries = len(data)
        if total_entries == 0:
            return {"entries": 0}

        sample_entries = data[: min(sample_size, total_entries)]
        profile_lengths: List[int] = []
        signature_counter: Counter[int] = Counter()

        for entry in sample_entries:
            if len(entry) < 2 or not isinstance(entry[1], Sequence):
                continue
            signatures = entry[1]
            profile_lengths.append(len(signatures))
            for value, count in itertools.islice(signatures, 64):
                if isinstance(value, (int, float)) and isinstance(count, (int, float)):
                    signature_counter[int(value)] += int(count)

        return {
            "entries": total_entries,
            "sampled_entries": len(sample_entries),
            "avg_profile_length": _safe_mean(profile_lengths),
            "dominant_signatures": signature_counter.most_common(5),
        }

    # ------------------------------------------------------------------
    def summarize_totals_matrix(
        self, pickle_path: Optional[Path | str] = None
    ) -> Optional[Dict[str, Any]]:
        """Summarize the ``TOTALS_MATRIX`` pickle."""

        resolved_path = (
            Path(pickle_path)
            if pickle_path
            else _normalise_path(self.base_dir, "TOTALS_MATRIX/LAMDa_TOTALS.pickle")
        )
        if not resolved_path.exists():
            logger.debug("TOTALS_MATRIX pickle not found at %s", resolved_path)
            return None

        data = _load_pickle(resolved_path)
        if not isinstance(data, Sequence) or len(data) < 2:
            return None

        global_totals = data[0][0] if data[0] else None
        hash_ids = data[1]

        summary: Dict[str, Any] = {
            "hash_ids": len(hash_ids) if isinstance(hash_ids, Sequence) else 0,
        }

        if isinstance(global_totals, Sequence) and global_totals:
            # The first list stores headline counters followed by multiple detailed arrays.
            # We expose the raw numeric headline values to avoid guessing semantics.
            headline_numbers = [value for value in global_totals if isinstance(value, (int, float))]
            summary["headline_values"] = headline_numbers[:5]
            array_shapes: List[Tuple[int, ...]] = []
            for item in global_totals:
                if (
                    isinstance(item, Sequence)
                    and item
                    and all(isinstance(sub, Sequence) for sub in item)
                ):
                    first = item[0]
                    if isinstance(first, Sequence):
                        array_shapes.append((len(item), len(first)))
                    else:
                        array_shapes.append((len(item),))
            if array_shapes:
                summary["matrix_shapes"] = array_shapes[:5]

        return summary

    # ------------------------------------------------------------------
    def summarize_meta_data(
        self,
        sample_size: int = META_SAMPLE_SIZE,
        meta_location: Optional[Path | str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Summarize all ``META_DATA`` shards."""

        meta_root = (
            Path(meta_location) if meta_location else _normalise_path(self.base_dir, "META_DATA")
        )
        if not meta_root.exists():
            logger.debug("META_DATA location not found at %s", meta_root)
            return None

        if meta_root.is_file():
            pickle_files = [meta_root]
        elif meta_root.is_dir():
            pickle_files = sorted(meta_root.glob("LAMDa_META_DATA_*.pickle"))
        else:
            logger.debug("META_DATA path is neither file nor directory: %s", meta_root)
            return None

        if not pickle_files:
            return None

        total_entries = 0
        sampled_entries = 0
        track_counts: List[int] = []
        opus_event_counts: List[int] = []
        chord_counts: List[int] = []
        tempo_change_counts: List[int] = []
        patch_counter: Counter[int] = Counter()

        for pickle_path in pickle_files:
            data: List[Sequence[Any]] = _load_pickle(pickle_path)
            total_entries += len(data)
            for hash_id, metrics in data:
                if sampled_entries >= sample_size:
                    break
                sampled_entries += 1
                metric_map = _to_metric_map(metrics)
                track_counts.append(int(metric_map.get("total_number_of_tracks", 0)))
                opus_event_counts.append(int(metric_map.get("total_number_of_opus_midi_events", 0)))
                chord_counts.append(int(metric_map.get("total_number_of_chords", 0)))
                tempo_change_counts.append(int(metric_map.get("tempo_change_count", 0)))
                for program, count in _extract_patch_counts(metric_map.get("total_patches_counts")):
                    patch_counter[program] += count
            if sampled_entries >= sample_size:
                break

        return {
            "entries": total_entries,
            "sampled_entries": sampled_entries,
            "avg_tracks": _safe_mean(track_counts),
            "avg_opus_events": _safe_mean(opus_event_counts),
            "avg_chord_count": _safe_mean(chord_counts),
            "avg_tempo_changes": _safe_mean(tempo_change_counts),
            "top_patches": patch_counter.most_common(5),
        }


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _to_metric_map(metrics: Iterable[Sequence[Any]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for item in metrics:
        if not item:
            continue
        key = item[0]
        if not isinstance(key, str):
            continue
        if key in result:
            # Keep only the first occurrence (e.g., multiple track_name entries)
            continue
        if len(item) == 2:
            result[key] = item[1]
        else:
            result[key] = item[1:]
    return result


def _extract_patch_counts(raw_value: Any) -> Iterable[Tuple[int, int]]:
    if not isinstance(raw_value, Sequence):
        return ()
    for entry in raw_value:
        if (
            isinstance(entry, Sequence)
            and len(entry) >= 2
            and isinstance(entry[0], (int, float))
            and isinstance(entry[1], (int, float))
        ):
            yield int(entry[0]), int(entry[1])


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarise LAMDa auxiliary pickle datasets")
    parser.add_argument(
        "--base",
        type=Path,
        help="Base directory containing the auxiliary datasets (defaults to data/Los-Angeles-MIDI if present)",
    )
    parser.add_argument(
        "--kilo", type=Path, help="Optional explicit path to KILO_CHORDS_DATA pickle"
    )
    parser.add_argument(
        "--sigs", type=Path, help="Optional explicit path to SIGNATURES_DATA pickle"
    )
    parser.add_argument(
        "--totals", type=Path, help="Optional explicit path to TOTALS_MATRIX pickle"
    )
    parser.add_argument(
        "--meta", type=Path, help="Optional explicit path to META_DATA directory or shard"
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        help="Override sample size when summarising KILO_CHORDS_DATA and SIGNATURES_DATA",
    )
    parser.add_argument(
        "--meta-sample",
        type=int,
        help="Override sample size when summarising META_DATA",
    )
    parser.add_argument(
        "--out", type=Path, help="Write JSON summary to this path instead of stdout"
    )
    parser.add_argument(
        "--pretty", action="store_true", help="Pretty-print JSON output with indentation"
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging verbosity",
    )
    return parser.parse_args(argv)


def _dump_summary(summary: Dict[str, Any], out_path: Optional[Path], pretty: bool) -> None:
    text_kwargs = {"indent": 2} if pretty else {}
    if out_path is None:
        print(json.dumps(summary, ensure_ascii=False, **text_kwargs))
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as writer:
        json.dump(summary, writer, ensure_ascii=False, **text_kwargs)
        writer.write("\n")


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper()))

    if args.base is not None:
        base_dir = args.base
    else:
        default_base = Path("data/Los-Angeles-MIDI")
        base_dir = default_base if default_base.exists() else Path.cwd()

    inspector = LAMDaDatasetInspector(base_dir)
    target_specified = any([args.kilo, args.sigs, args.totals, args.meta])
    summaries: Dict[str, Dict[str, Any]] = {}

    sample_size = (
        args.sample_size if args.sample_size and args.sample_size > 0 else DEFAULT_SAMPLE_SIZE
    )
    meta_sample = (
        args.meta_sample if args.meta_sample and args.meta_sample > 0 else META_SAMPLE_SIZE
    )

    if args.kilo or not target_specified:
        summary = inspector.summarize_kilo_chords(sample_size=sample_size, pickle_path=args.kilo)
        if summary:
            summaries["kilo_chords"] = summary

    if args.sigs or not target_specified:
        summary = inspector.summarize_signatures(sample_size=sample_size, pickle_path=args.sigs)
        if summary:
            summaries["signatures"] = summary

    if args.totals or not target_specified:
        summary = inspector.summarize_totals_matrix(pickle_path=args.totals)
        if summary:
            summaries["totals_matrix"] = summary

    if args.meta or not target_specified:
        summary = inspector.summarize_meta_data(sample_size=meta_sample, meta_location=args.meta)
        if summary:
            summaries["meta_data"] = summary

    if not summaries:
        logger.warning("No summaries generated. Check provided paths and options.")
        return 1

    _dump_summary(summaries, args.out, args.pretty)
    return 0


if __name__ == "__main__":
    sys.exit(main())
