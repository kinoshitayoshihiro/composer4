#!/usr/bin/env python3
"""Reusable drum-loop builder utilities for LAMDa-style datasets."""
# pylint: disable=broad-except,import-outside-toplevel
from __future__ import annotations

import hashlib
import random
import statistics
import sys
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    cast,
)

import importlib

from .metrics import (
    MetricConfig,
    MetricsAggregator,
    compute_loop_metrics,
)


def _load_tqdm() -> Callable[..., Iterable[Any]]:
    try:
        module = importlib.import_module("tqdm")
        return module.tqdm  # type: ignore[attr-defined]
    except ModuleNotFoundError:  # pragma: no cover - fallback for docs/tests

        def _noop(iterable: Iterable[Any], **_: Any) -> Iterable[Any]:
            return iterable

        return _noop


tqdm = _load_tqdm()


@dataclass
class DrumLoopBuildConfig:
    """Configuration for the drum-loop deduplication pipeline."""

    dataset_name: str = "Drum Loops"
    input_dir: Path = Path("data/loops")
    output_dir: Path = Path("output/drumloops_cleaned")
    metadata_dir: Path = Path("output/drumloops_metadata")
    tmidix_path: Path = Path("data/Los-Angeles-MIDI/CODE")
    min_notes: int = 16
    max_file_size: int = 1_000_000
    save_interval: int = 5_000
    start_file_number: int = 0
    random_seed: Optional[int] = 42
    require_polyphony: bool = True
    allowed_extensions: Sequence[str] = (".mid", ".midi")
    ensure_hex_subdirs: bool = True
    metrics: MetricConfig = field(default_factory=MetricConfig)
    metadata_shard_max_loops: Optional[int] = 20000
    metadata_shard_prefix: str = "drumloops_metadata_v2_shard"

    def normalized(self) -> "DrumLoopBuildConfig":
        """Return a copy with all paths resolved for downstream use."""

        return DrumLoopBuildConfig(
            dataset_name=self.dataset_name,
            input_dir=Path(self.input_dir),
            output_dir=Path(self.output_dir),
            metadata_dir=Path(self.metadata_dir),
            tmidix_path=Path(self.tmidix_path),
            min_notes=self.min_notes,
            max_file_size=self.max_file_size,
            save_interval=self.save_interval,
            start_file_number=self.start_file_number,
            random_seed=self.random_seed,
            require_polyphony=self.require_polyphony,
            allowed_extensions=tuple(ext.lower() for ext in self.allowed_extensions),
            ensure_hex_subdirs=self.ensure_hex_subdirs,
            metrics=MetricConfig(**asdict(self.metrics)),
            metadata_shard_max_loops=self.metadata_shard_max_loops,
            metadata_shard_prefix=self.metadata_shard_prefix,
        )


def _import_tmidix(path: Path):
    """Import TMIDIX dynamically after adding the provided path."""

    resolved = path.resolve()
    if str(resolved) not in sys.path:
        sys.path.append(str(resolved))
    return importlib.import_module("TMIDIX")


def _collect_midi_files(
    input_dir: Path,
    allowed_extensions: Iterable[str],
) -> List[Path]:
    extensions = {ext.lower() for ext in allowed_extensions}
    midi_files = [
        path
        for path in input_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in extensions
    ]
    return sorted({p.resolve() for p in midi_files})


def _initialise_output_dirs(
    output_dir: Path,
    metadata_dir: Path,
    ensure_hex_subdirs: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    if ensure_hex_subdirs:
        for digit in "0123456789abcdef":
            (output_dir / digit).mkdir(exist_ok=True)


def _extract_filename_metadata(filename: str) -> Dict[str, Any]:
    parts = filename.replace(".midi", "").replace(".mid", "").split("_")
    genre = parts[1] if len(parts) > 1 else "unknown"
    try:
        bpm = int(parts[0])
    except ValueError:
        bpm = 120
    return {"genre": genre, "bpm": bpm}


def build_drumloops(
    config: DrumLoopBuildConfig,
    *,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Execute the drum-loop deduplication pipeline.

    Parameters
    ----------
    config:
        Pipeline configuration (input/output paths, thresholds, etc.).
    dry_run:
        When ``True`` the function only scans files and reports the would-be
        configuration without performing deduplication.

    Returns
    -------
    Dict[str, Any]
        Summary statistics from the run (number of scanned/unique loops, etc.).
    """

    cfg = config.normalized()
    TMIDIX = _import_tmidix(cfg.tmidix_path)
    metadata_base_v2 = cfg.metadata_dir / "drumloops_metadata_v2"
    legacy_metadata_base = cfg.metadata_dir / "drumloops_metadata"
    pickle_writer: Callable[..., Any] = getattr(
        TMIDIX,
        "Tegridy_Any_Pickle_File_Writer",
    )
    midi2score: Callable[[bytes], Any] = getattr(TMIDIX, "midi2score")

    print("=" * 70)
    print(f"{cfg.dataset_name} LAMDa Dataset Builder")
    print("=" * 70)
    print("Input directory :", cfg.input_dir)
    print("Output directory:", cfg.output_dir)
    print("Metadata dir   :", cfg.metadata_dir)
    print("Min notes      :", cfg.min_notes)
    print("Max file size  :", cfg.max_file_size)
    print("Save interval  :", cfg.save_interval)
    print("Require chords :", cfg.require_polyphony)
    print("=" * 70)

    if not cfg.input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {cfg.input_dir}")

    midi_files = _collect_midi_files(cfg.input_dir, cfg.allowed_extensions)
    print(f"Found {len(midi_files)} MIDI files")
    print("=" * 70)

    if not midi_files:
        raise FileNotFoundError("No MIDI files discovered – aborting.")

    if dry_run:
        print("Dry-run mode: skipping deduplication; returning summary only.")
        return {
            "config": asdict(cfg),
            "total_scanned": len(midi_files),
            "unique_loops": None,
            "metrics_summary": None,
            "dry_run": True,
        }

    _initialise_output_dirs(
        cfg.output_dir,
        cfg.metadata_dir,
        cfg.ensure_hex_subdirs,
    )

    pickle_writer(
        [str(path) for path in midi_files],
        str(cfg.metadata_dir / "drumloops_filez"),
    )

    if cfg.random_seed is not None:
        random.seed(cfg.random_seed)
    random.shuffle(midi_files)

    start_idx = cfg.start_file_number
    save_interval = cfg.save_interval

    input_files_count = start_idx
    files_count = 0

    all_md5_names: List[str] = []
    all_pitches_sums: List[int] = []
    all_pitches_counts: List[List[int]] = []
    all_pitches_and_counts: List[List[List[int]]] = []

    genre_stats: Counter[str] = Counter()
    bpm_stats: List[int] = []
    metrics_aggregator = MetricsAggregator()
    if cfg.metadata_shard_max_loops and cfg.metadata_shard_max_loops > 0:
        max_loops_per_shard = cfg.metadata_shard_max_loops
    else:
        max_loops_per_shard = None
    shard_loops: List[Dict[str, Any]] = []
    shard_metrics = MetricsAggregator()
    shard_genre_stats: Counter[str] = Counter()
    shard_bpm_stats: List[int] = []
    shard_records: List[Dict[str, Any]] = []

    def _build_bpm_summary(
        values: Sequence[int],
    ) -> Optional[Dict[str, float]]:
        if not values:
            return None
        return {
            "count": len(values),
            "min": min(values),
            "max": max(values),
            "mean": statistics.mean(values),
            "median": statistics.median(values),
        }

    def _record_shard(shard_index: int) -> None:
        bpm_summary = _build_bpm_summary(shard_bpm_stats)
        metrics_summary = (
            shard_metrics.summary(digits=cfg.metrics.round_digits) if shard_metrics.count else None
        )
        shard_base = cfg.metadata_dir / (f"{cfg.metadata_shard_prefix}_{shard_index:04d}")
        payload: Dict[str, Any] = {
            "version": "2.0",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "config": asdict(cfg),
            "shard_index": shard_index,
            "loop_count": len(shard_loops),
            "summary": {
                "genre_distribution": dict(shard_genre_stats),
                "bpm": bpm_summary,
                "metrics": metrics_summary,
            },
            "loops": list(shard_loops),
        }
        pickle_writer(payload, str(shard_base))
        shard_records.append(
            {
                "index": shard_index,
                "path": str(shard_base.with_suffix(".pickle")),
                "loop_count": len(shard_loops),
                "genre_distribution": dict(shard_genre_stats),
                "bpm_summary": bpm_summary,
                "metrics_summary": metrics_summary,
            }
        )

    def _flush_shard() -> None:
        nonlocal shard_loops, shard_metrics, shard_genre_stats, shard_bpm_stats
        if not shard_loops:
            return
        shard_index = len(shard_records)
        _record_shard(shard_index)
        shard_loops = []
        shard_metrics = MetricsAggregator()
        shard_genre_stats = Counter()
        shard_bpm_stats = []

    def _build_index_payload(include_pending: bool = False) -> Dict[str, Any]:
        metrics_summary = (
            metrics_aggregator.summary(digits=cfg.metrics.round_digits)
            if metrics_aggregator.count
            else None
        )
        bpm_summary = _build_bpm_summary(bpm_stats)
        pending_summary: Optional[Dict[str, Any]] = None
        if include_pending and shard_loops:
            pending_summary = {
                "loop_count": len(shard_loops),
                "genre_distribution": dict(shard_genre_stats),
                "bpm": _build_bpm_summary(shard_bpm_stats),
                "metrics": (
                    shard_metrics.summary(digits=cfg.metrics.round_digits)
                    if shard_metrics.count
                    else None
                ),
            }

        return {
            "version": "2.0",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "config": asdict(cfg),
            "summary": {
                "total_scanned": input_files_count,
                "unique_loops": files_count,
                "dedupe_ratio": (files_count / input_files_count if input_files_count else 0.0),
                "genre_distribution": dict(genre_stats),
                "bpm": bpm_summary,
                "metrics": metrics_summary,
            },
            "shards": list(shard_records),
            "pending_shard": pending_summary,
            "dedupe_state": {
                "all_md5": list(all_md5_names),
                "all_pitches_sums": list(all_pitches_sums),
                "all_pitches_counts": list(all_pitches_counts),
                "all_pitches_and_counts": list(all_pitches_and_counts),
            },
        }

    for midi_path in tqdm(midi_files[start_idx:], desc="Processing"):
        try:
            input_files_count += 1

            filename = midi_path.name
            meta = _extract_filename_metadata(filename)
            genre = meta["genre"]
            bpm = meta["bpm"]

            file_size = midi_path.stat().st_size
            if file_size > cfg.max_file_size:
                continue

            fdata = midi_path.read_bytes()
            md5sum = hashlib.md5(fdata).hexdigest()
            md5name = f"{md5sum}.mid"

            if str(md5sum) in all_md5_names:
                continue

            score = midi2score(fdata)
            events_matrix: List[List[Any]] = []
            for track in score[1:]:
                events_matrix.extend(track)
            events_matrix.sort(key=lambda event: event[1])
            notes = [event for event in events_matrix if event and event[0] == "note"]

            if len(notes) < cfg.min_notes:
                continue

            times = [note[1] for note in notes]
            durs = [note[2] for note in notes]

            if min(times) < 0 or min(durs) < 0:
                continue

            if cfg.require_polyphony and len(times) <= len(set(times)):
                continue

            all_md5_names.append(str(md5sum))

            pitches = [note[4] for note in notes]
            pitches_sum = sum(pitches)
            if pitches_sum in all_pitches_sums:
                continue
            all_pitches_sums.append(pitches_sum)

            counts_iter = Counter(pitches).most_common()
            pitches_and_counts = sorted(
                [[key, val] for key, val in counts_iter],
                reverse=True,
                key=lambda pair: pair[1],
            )
            pitches_counts = [item[1] for item in pitches_and_counts]

            if pitches_counts in all_pitches_counts:
                continue

            loop_metrics = compute_loop_metrics(notes, config=cfg.metrics)

            output_subdir = cfg.output_dir / md5name[0]
            output_subdir.mkdir(exist_ok=True)
            output_path = output_subdir / md5name
            output_path.write_bytes(fdata)

            metrics_aggregator.add(loop_metrics)
            shard_metrics.add(loop_metrics)
            record: Dict[str, Any] = {
                "md5": md5sum,
                "filename": filename,
                "input_path": str(midi_path),
                "output_path": str(output_path),
                "genre": genre,
                "bpm": bpm,
                "note_count": len(notes),
                "duration_ticks": loop_metrics.duration_ticks,
                "pitches": {
                    "sum": pitches_sum,
                    "counts": list(pitches_counts),
                    "distribution": pitches_and_counts,
                },
                "metrics": loop_metrics.to_dict(
                    digits=cfg.metrics.round_digits,
                ),
            }
            shard_loops.append(record)

            all_pitches_counts.append(pitches_counts)
            all_pitches_and_counts.append(pitches_and_counts)
            genre_stats[genre] += 1
            bpm_stats.append(bpm)
            shard_genre_stats[genre] += 1
            shard_bpm_stats.append(bpm)
            files_count += 1

            if max_loops_per_shard and len(shard_loops) >= max_loops_per_shard:
                _flush_shard()

            if files_count % save_interval == 0:
                _flush_shard()
                index_payload = _build_index_payload()
                pickle_writer(index_payload, str(metadata_base_v2))
                pickle_writer(
                    [
                        [str(path) for path in midi_files],
                        all_md5_names,
                        all_pitches_sums,
                        all_pitches_and_counts,
                        genre_stats,
                        bpm_stats,
                    ],
                    str(cfg.metadata_dir / "drumloops_metadata"),
                )
                ratio = files_count / input_files_count if input_files_count else 0.0
                print()
                print("=" * 70)
                print(
                    "CHECKPOINT: "
                    f"{files_count} unique / {input_files_count} total"
                    f" → {ratio:.2%}"
                )
                print("=" * 70)

        except KeyboardInterrupt:  # pragma: no cover - CLI convenience
            print()
            print("=" * 70)
            print("Keyboard interrupt detected. Saving progress...")
            print("=" * 70)
            break
        except Exception:  # pragma: no cover - LAMDa pipeline is permissive
            continue

    _flush_shard()
    index_payload = _build_index_payload()
    pickle_writer(index_payload, str(metadata_base_v2))
    pickle_writer(
        [
            [str(path) for path in midi_files],
            all_md5_names,
            all_pitches_sums,
            all_pitches_and_counts,
            genre_stats,
            bpm_stats,
        ],
        str(legacy_metadata_base),
    )

    dedupe_ratio = files_count / input_files_count if input_files_count else 0.0
    metrics_summary = index_payload["summary"]["metrics"]
    bpm_summary = index_payload["summary"]["bpm"]

    summary: Dict[str, Any] = {
        "config": asdict(cfg),
        "total_scanned": input_files_count,
        "unique_loops": files_count,
        "dedupe_ratio": dedupe_ratio,
        "genre_stats": dict(genre_stats),
        "bpm_stats": bpm_stats,
        "bpm_summary": bpm_summary,
        "metrics_summary": metrics_summary,
        "loops_metadata_records": files_count,
        "metadata_files": {
            "legacy": str(legacy_metadata_base.with_suffix(".pickle")),
            "v2_index": str(metadata_base_v2.with_suffix(".pickle")),
            "v2_shards": [record["path"] for record in shard_records],
        },
        "dry_run": False,
    }

    print("=" * 70)
    print("PROCESSING COMPLETE!")
    print("=" * 70)
    print(f"Total MIDI files scanned: {input_files_count}")
    print(f"Unique drum loops extracted: {files_count}")
    if input_files_count:
        print(f"Deduplication ratio: {dedupe_ratio:.2%}")
    shard_count = len(shard_records)
    print(f"Metadata shards written: {shard_count}")
    print("=" * 70)
    print("Genre distribution (top 10):")
    for genre, count in genre_stats.most_common(10):
        print(f"  {genre}: {count}")
    print("=" * 70)
    if bpm_stats:
        print("BPM statistics:")
        print(f"  Min: {min(bpm_stats)}")
        print(f"  Max: {max(bpm_stats)}")
        print(f"  Average: {statistics.mean(bpm_stats):.1f}")
        print(f"  Median: {statistics.median(bpm_stats):.1f}")
        print("=" * 70)
    averages: Dict[str, float] = {}
    if isinstance(metrics_summary, dict):
        metrics_summary_dict = cast(Dict[str, Any], metrics_summary)
        raw_averages = metrics_summary_dict.get("averages")
        if isinstance(raw_averages, dict):
            averages = cast(Dict[str, float], raw_averages)
    if averages:
        print("Average groove metrics:")
        for key in (
            "swing_ratio",
            "ghost_rate",
            "layering_rate",
            "velocity_mean",
        ):
            value = averages.get(key)
            if value is not None:
                print(f"  {key}: {value:.4f}")
        print("=" * 70)
    print("Output location:", cfg.output_dir)
    print("Metadata directory:", cfg.metadata_dir)
    print("  Legacy pickle :", legacy_metadata_base.with_suffix(".pickle"))
    print("  Index (v2)    :", metadata_base_v2.with_suffix(".pickle"))
    if shard_records:
        print("  Shard files   :")
        preview = shard_records[:5]
        for record in preview:
            print(f"    - {record['path']}")
        if len(shard_records) > len(preview):
            print(f"    ... {len(shard_records)} shards total")
    else:
        print("  Shard files   : (none)")
    print("=" * 70)
    print("Done! 🥁")
    print("=" * 70)

    return summary
