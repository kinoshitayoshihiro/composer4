#!/usr/bin/env python3
# pyright: reportMissingImports=false
# pyright: reportUnknownMemberType=false
# pyright: reportUnknownArgumentType=false
# pyright: reportUnknownVariableType=false
"""Generate similarity-based loop pairs from LAMDa metadata."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from lamda_tools.metadata_io import collect_flat_rows

_MODULES: Dict[str, Any] = {}

DEFAULT_METRIC_PREFIX = "metrics."
DEFAULT_MIN_SIMILARITY = 0.4
DEFAULT_TOP_K = 5
DEFAULT_INDEX_NAME = "drumloops_metadata_v2.pickle"


def _ensure_dependencies() -> Tuple[Any, Any]:
    numpy_mod = _MODULES.get("numpy")
    if numpy_mod is None:
        try:
            import numpy as np_mod  # type: ignore[import-not-found]
        except ModuleNotFoundError as exc:
            message = "This script requires numpy. Install it with " "'pip install numpy'."
            raise RuntimeError(message) from exc
        numpy_mod = np_mod
        _MODULES["numpy"] = numpy_mod

    pandas_mod = _MODULES.get("pandas")
    if pandas_mod is None:
        try:
            import pandas as pd_mod  # type: ignore[import-not-found]
        except ModuleNotFoundError as exc:
            message = "This script requires pandas. Install it with " "'pip install pandas'."
            raise RuntimeError(message) from exc
        pandas_mod = pd_mod
        _MODULES["pandas"] = pandas_mod

    return numpy_mod, pandas_mod


def _get_numpy() -> Any:
    numpy_mod, _ = _ensure_dependencies()
    return numpy_mod


def _get_pandas() -> Any:
    _, pandas_mod = _ensure_dependencies()
    return pandas_mod


def _filter_required_columns(frame: Any, columns: Sequence[str]) -> Any:
    if not columns:
        return frame
    pandas_mod = _get_pandas()
    missing = [col for col in columns if col not in frame.columns]
    if missing:
        raise KeyError(f"Required columns not present: {missing}")
    mask = pandas_mod.Series(True, index=frame.index)
    for column in columns:
        series = frame[column]
        column_mask = series.notna()
        column_mask &= series.astype(str).str.strip() != ""
        mask &= column_mask
    return frame.loc[mask, :].copy()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=("Build (input, target, similarity) pairs for DUV training."),
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
            "Explicit metadata index path. Defaults to " "metadata_dir/drumloops_metadata_v2.pickle"
        ),
    )
    parser.add_argument(
        "--query",
        type=str,
        help=("Semantic filter in the form 'source -> target' " "(substring match)."),
    )
    parser.add_argument(
        "--filter-column",
        type=str,
        default="genre",
        help="Column used for query substring matching (default: genre).",
    )
    parser.add_argument(
        "--metrics",
        nargs="*",
        help=("Metrics columns to use for similarity " "(defaults to all metrics.* columns)."),
    )
    parser.add_argument(
        "--require-columns",
        nargs="*",
        help="Drop loops missing these columns before pairing.",
    )
    parser.add_argument(
        "--require-paths",
        action="store_true",
        help="Ensure loops retain both input_path and output_path values.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help="Number of target candidates to keep per input (default: 5).",
    )
    parser.add_argument(
        "--max-pairs",
        type=int,
        help="Optional global cap on generated pairs.",
    )
    parser.add_argument(
        "--min-similarity",
        type=float,
        default=DEFAULT_MIN_SIMILARITY,
        help="Discard pairs below this cosine similarity (default: 0.4).",
    )
    parser.add_argument(
        "--allow-self",
        action="store_true",
        help=("Allow pairing loops with identical MD5 hashes " "(disabled by default)."),
    )
    parser.add_argument(
        "--jsonl-out",
        type=Path,
        required=True,
        help="Destination JSONL file for generated pairs.",
    )
    parser.add_argument(
        "--csv-out",
        type=Path,
        help="Optional CSV destination (same fields as JSONL).",
    )
    parser.add_argument(
        "--summary-out",
        type=Path,
        help="Optional JSON summary file describing the run.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help=("Limit the number of loops considered before pairing " "(useful for debugging)."),
    )
    parser.add_argument(
        "--print-summary",
        action="store_true",
        help="Print summary information to stdout upon completion.",
    )
    return parser.parse_args()


def _resolve_index(metadata_dir: Path, explicit: Optional[Path]) -> Path:
    if explicit:
        return explicit
    return metadata_dir / DEFAULT_INDEX_NAME


def _split_query(query: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    if not query:
        return None, None
    for token in ("->", "→"):
        if token in query:
            left, right = query.split(token, 1)
            return left.strip() or None, right.strip() or None
    cleaned = query.strip()
    return cleaned or None, None


def _apply_filter(
    frame: Any,
    keyword: Optional[str],
    column: str,
) -> Any:
    pandas_mod = _get_pandas()
    if keyword is None:
        return frame
    if column not in frame.columns:
        raise KeyError(f"Column '{column}' not present in metadata")
    lower_keyword = keyword.casefold()

    def _match(value: Any) -> bool:
        if value is None or pandas_mod.isna(value):
            return False
        return lower_keyword in str(value).casefold()

    mask = frame[column].map(_match)
    mask_bool = mask.astype(bool)
    result = frame.loc[mask_bool, :]
    return result


def _select_metric_columns(
    frame: Any,
    requested: Optional[Sequence[str]],
) -> List[str]:
    if requested:
        missing = [col for col in requested if col not in frame.columns]
        if missing:
            raise KeyError(f"Missing metric columns: {missing}")
        return list(requested)
    return [
        col
        for col in frame.columns
        if col.startswith(DEFAULT_METRIC_PREFIX) and col != "metrics.instrument_distribution"
    ]


def _standardise_vectors(matrix: Any) -> Any:
    numpy_mod = _get_numpy()
    data = numpy_mod.asarray(matrix, dtype=numpy_mod.float64)
    mean = data.mean(axis=0, keepdims=True)
    std = data.std(axis=0, keepdims=True)
    std[std < 1e-9] = 1.0
    normalised = (data - mean) / std
    norms = numpy_mod.linalg.norm(normalised, axis=1, keepdims=True)
    norms[norms < 1e-9] = 1.0
    return normalised / norms


def _build_pairs(
    frame: Any,
    metric_cols: Sequence[str],
    *,
    source_filter: Optional[str],
    target_filter: Optional[str],
    filter_column: str,
    top_k: int,
    max_pairs: Optional[int],
    min_similarity: float,
    allow_self: bool,
) -> List[Dict[str, Any]]:
    numpy_mod = _get_numpy()
    if not metric_cols:
        raise ValueError("No metric columns available for similarity")

    working = frame.reset_index(drop=True)
    metric_matrix = working.loc[:, metric_cols].to_numpy(
        dtype=float,
        na_value=0.0,
    )
    unit_vectors = _standardise_vectors(metric_matrix)

    sources = _apply_filter(working, source_filter, filter_column)
    targets = _apply_filter(working, target_filter, filter_column)
    if sources.empty:
        raise ValueError("Source filter matched no loops")
    if targets.empty:
        raise ValueError("Target filter matched no loops")

    source_idx = sources.index.to_numpy()
    target_idx = targets.index.to_numpy()

    source_vectors = unit_vectors[source_idx]
    target_vectors = unit_vectors[target_idx]

    similarities = source_vectors @ target_vectors.T

    pairs: List[Dict[str, Any]] = []
    for row_pos, src_idx in enumerate(source_idx):
        row_scores = similarities[row_pos]
        if max_pairs is not None and len(pairs) >= max_pairs:
            break
        if row_scores.size == 0:
            continue
        top = min(top_k, row_scores.size)
        candidate_idx = numpy_mod.argpartition(row_scores, -top)[-top:]
        order = numpy_mod.argsort(-row_scores[candidate_idx])
        candidate_idx = candidate_idx[order]
        for tgt_pos in candidate_idx:
            score = float(row_scores[tgt_pos])
            if score < min_similarity:
                continue
            tgt_idx = int(target_idx[tgt_pos])
            src_md5 = str(working.loc[src_idx, "md5"])
            tgt_md5 = str(working.loc[tgt_idx, "md5"])
            if not allow_self and src_md5 == tgt_md5:
                continue
            pair = {
                "input_md5": working.loc[src_idx, "md5"],
                "target_md5": working.loc[tgt_idx, "md5"],
                "similarity": score,
                "input_genre": working.loc[src_idx, filter_column],
                "target_genre": working.loc[tgt_idx, filter_column],
                "input_output_path": (
                    working.loc[src_idx, "output_path"]
                    if "output_path" in working.columns
                    else None
                ),
                "target_output_path": (
                    working.loc[tgt_idx, "output_path"]
                    if "output_path" in working.columns
                    else None
                ),
                "input_input_path": (
                    working.loc[src_idx, "input_path"] if "input_path" in working.columns else None
                ),
                "target_input_path": (
                    working.loc[tgt_idx, "input_path"] if "input_path" in working.columns else None
                ),
                "input_shard_index": working.loc[src_idx, "shard_index"],
                "target_shard_index": working.loc[tgt_idx, "shard_index"],
                "metrics": list(metric_cols),
            }
            pairs.append(pair)
            if max_pairs is not None and len(pairs) >= max_pairs:
                break
        if max_pairs is not None and len(pairs) >= max_pairs:
            break
    return pairs


def _write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")
        return
    import csv

    fieldnames = sorted({key for row in rows for key in row.keys()})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    _ensure_dependencies()
    args = _parse_args()
    metadata_dir = args.metadata_dir.resolve()
    index_path = _resolve_index(metadata_dir, args.index)

    rows, _ = collect_flat_rows(
        index_path,
        metadata_dir,
        limit=args.limit,
        include_paths=True,
    )
    pandas_mod = _get_pandas()
    frame = pandas_mod.DataFrame(rows)
    required_columns: List[str] = []
    if args.require_columns:
        required_columns.extend(args.require_columns)
    if args.require_paths:
        required_columns.extend(["input_path", "output_path"])
    if required_columns:
        required_columns = list(dict.fromkeys(required_columns))
        frame = _filter_required_columns(frame, required_columns)
    if frame.empty:
        raise ValueError("No loops remaining after applying column filters")
    filtered_rows = int(frame.shape[0])
    metric_cols = _select_metric_columns(frame, args.metrics)

    source_filter, target_filter = _split_query(args.query)

    pairs = _build_pairs(
        frame,
        metric_cols,
        source_filter=source_filter,
        target_filter=target_filter,
        filter_column=args.filter_column,
        top_k=args.top_k,
        max_pairs=args.max_pairs,
        min_similarity=args.min_similarity,
        allow_self=args.allow_self,
    )

    args.jsonl_out.parent.mkdir(parents=True, exist_ok=True)
    _write_jsonl(args.jsonl_out, pairs)
    if args.csv_out:
        _write_csv(args.csv_out, pairs)

    unique_sources = {pair["input_md5"] for pair in pairs}
    unique_targets = {pair["target_md5"] for pair in pairs}
    similarity_values = [pair["similarity"] for pair in pairs]
    similarity_stats: Optional[Dict[str, float]] = None
    if similarity_values:
        similarity_stats = {
            "min": min(similarity_values),
            "max": max(similarity_values),
            "mean": float(sum(similarity_values) / len(similarity_values)),
        }

    summary = {
        "pairs": len(pairs),
        "metric_columns": metric_cols,
        "source_filter": source_filter,
        "target_filter": target_filter,
        "index_path": str(index_path),
        "rows_total": len(rows),
        "rows_after_filters": filtered_rows,
        "required_columns": required_columns or None,
        "unique_sources": len(unique_sources),
        "unique_targets": len(unique_targets),
        "similarity": similarity_stats,
        "top_k": args.top_k,
        "min_similarity": args.min_similarity,
        "allow_self": args.allow_self,
    }
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
