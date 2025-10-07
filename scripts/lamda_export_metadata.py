#!/usr/bin/env python3
"""Export LAMDa drum-loop metadata shards to tabular formats."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, cast

from lamda_tools.metadata_io import DEFAULT_LIGHT_COLUMNS, collect_flat_rows

DEFAULT_INDEX_NAME = "drumloops_metadata_v2.pickle"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=("Flatten sharded drum-loop metadata into CSV/JSONL tables."),
    )
    parser.add_argument(
        "--metadata-dir",
        type=Path,
        default=Path("output/drumloops_metadata"),
        help="Directory containing the metadata index and shard pickles.",
    )
    parser.add_argument(
        "--index",
        type=Path,
        help=(
            "Explicit metadata index path. Defaults to" " metadata_dir/drumloops_metadata_v2.pickle"
        ),
    )
    parser.add_argument(
        "--csv-out",
        type=Path,
        help="Optional destination for CSV output.",
    )
    parser.add_argument(
        "--jsonl-out",
        type=Path,
        help="Optional destination for JSONL output.",
    )
    parser.add_argument(
        "--summary-out",
        type=Path,
        help="Optional JSON file capturing run summary information.",
    )
    parser.add_argument(
        "--metrics",
        nargs="*",
        help=("Restrict exported metrics to the provided keys" " (default: include all metrics)."),
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Maximum number of loops to export (default: all).",
    )
    parser.add_argument(
        "--light",
        action="store_true",
        help="Emit only the DEFAULT_LIGHT_COLUMNS (plus shard metadata).",
    )
    parser.add_argument(
        "--include-distribution",
        action="store_true",
        help=("Include instrument_distribution JSON in the output" " (ignored with --light)."),
    )
    parser.add_argument(
        "--no-paths",
        action="store_true",
        help="Omit input/output path columns even in full exports.",
    )
    parser.add_argument(
        "--print-summary",
        action="store_true",
        help="Print row/column counts to stdout after exporting.",
    )
    return parser.parse_args()


def _resolve_index(metadata_dir: Path, explicit: Optional[Path]) -> Path:
    if explicit:
        return explicit
    return metadata_dir / DEFAULT_INDEX_NAME


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def _summarise(
    rows: Sequence[Dict[str, Any]],
    index_data: Dict[str, Any],
) -> Dict[str, Any]:
    raw_shards = cast(Iterable[Any], index_data.get("shards", []) or [])
    shards_info: List[Dict[str, Any]] = [
        shard_dict for shard_dict in raw_shards if isinstance(shard_dict, dict)
    ]
    shard_paths: List[Any] = [shard.get("path") for shard in shards_info]
    shard_stats: Dict[str, Any] = {
        "count": len(shards_info),
        "paths": shard_paths,
    }
    return {
        "rows": len(rows),
        "columns": sorted({key for row in rows for key in row.keys()}),
        "shards": shard_stats,
        "available_metrics": DEFAULT_LIGHT_COLUMNS,
    }


def main() -> None:
    args = _parse_args()
    metadata_dir = args.metadata_dir.resolve()
    index_path = _resolve_index(metadata_dir, args.index)
    metric_keys = args.metrics
    include_paths = not args.no_paths
    include_distribution = args.include_distribution and not args.light

    rows, index_data = collect_flat_rows(
        index_path,
        metadata_dir,
        metric_keys=metric_keys,
        limit=args.limit,
        light=args.light,
        include_paths=include_paths,
        include_distribution=include_distribution,
    )

    if args.csv_out:
        _write_csv(args.csv_out.resolve(), rows)
    if args.jsonl_out:
        _write_jsonl(args.jsonl_out.resolve(), rows)

    summary = _summarise(rows, index_data)
    if args.summary_out:
        args.summary_out.parent.mkdir(parents=True, exist_ok=True)
        args.summary_out.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    if args.print_summary:
        print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
