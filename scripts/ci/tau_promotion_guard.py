#!/usr/bin/env python3
"""Guardrails for promoting proposed tau thresholds."""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


def _read_jsonl(path: str | None) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path or not os.path.exists(path):
        return rows
    with open(path, "r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _pluck(payload: Dict[str, Any], dotted: str) -> Any:
    current: Any = payload
    for token in dotted.split("."):
        if not isinstance(current, dict) or token not in current:
            return None
        current = current[token]
    return current


def _to_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_simple_yaml(path: str | None) -> Dict[str, Any]:
    if not path or not os.path.exists(path):
        return {}
    root: Dict[str, Any] = {}
    stack: List[Tuple[int, Dict[str, Any]]] = [(-1, root)]
    with open(path, "r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.rstrip("\n")
            if not line or line.lstrip().startswith("#"):
                continue
            indent = len(line) - len(line.lstrip(" "))
            if ":" not in line:
                continue
            key, _, rest = line.strip().partition(":")
            while stack and indent <= stack[-1][0]:
                stack.pop()
            parent = stack[-1][1] if stack else root
            content = rest.strip()
            if not content:
                child: Dict[str, Any] = {}
                parent[key] = child
                stack.append((indent, child))
                continue
            if content == "null":
                parent[key] = None
            else:
                try:
                    parent[key] = float(content)
                except ValueError:
                    parent[key] = content
    return root


def _tempo_bin(bpm: Any, edges: Sequence[float]) -> str:
    value = _to_float(bpm)
    if value is None:
        return "NA"
    for edge in edges:
        if value < edge:
            return f"<{edge}"
    return f">={edges[-1]}"


def _choose_tau(
    tau_map: Mapping[str, Any],
    instrument: str,
    tempo_bin: str,
    fallback: float | None,
) -> float | None:
    bit = tau_map.get("by_instrument_tempo")
    if isinstance(bit, dict):
        inst_map = bit.get(instrument)
        if isinstance(inst_map, dict):
            candidate = inst_map.get(tempo_bin)
            if isinstance(candidate, (int, float)):
                return float(candidate)
    bii = tau_map.get("by_instrument")
    if isinstance(bii, dict):
        candidate = bii.get(instrument)
        if isinstance(candidate, (int, float)):
            return float(candidate)
    byt = tau_map.get("by_tempo")
    if isinstance(byt, dict):
        candidate = byt.get(tempo_bin)
        if isinstance(candidate, (int, float)):
            return float(candidate)
    global_tau = tau_map.get("global")
    if isinstance(global_tau, (int, float)):
        return float(global_tau)
    return fallback


def _mismatch_at_tau(values: Iterable[Any], tau: float) -> float | None:
    floats = [_to_float(item) for item in values]
    filtered = [item for item in floats if item is not None]
    if not filtered:
        return None
    count = sum(1 for item in filtered if item < tau)
    return 100.0 * (count / len(filtered))


def _guard_single_metric(
    before_rows: Sequence[Dict[str, Any]],
    after_rows: Sequence[Dict[str, Any]],
    tau_yaml: str | None,
    cos_key: str,
    tempo_edges: Sequence[float],
    min_global_improve_pp: float,
    max_bucket_regression_pp: float,
    max_regression_buckets: int,
    min_bucket_n: int,
) -> Tuple[bool, Dict[str, Any]]:
    report: Dict[str, Any] = {
        "metric": cos_key,
        "global_before": None,
        "global_after": None,
        "global_delta": None,
        "tau_global": None,
        "bad_buckets": [],
    }
    tau_map = _parse_simple_yaml(tau_yaml)
    tau_global = tau_map.get("global")
    if isinstance(tau_global, (int, float)):
        report["tau_global"] = float(tau_global)
        before_rate = _mismatch_at_tau(
            (_pluck(row, cos_key) for row in before_rows),
            float(tau_global),
        )
        after_rate = _mismatch_at_tau(
            (_pluck(row, cos_key) for row in after_rows),
            float(tau_global),
        )
        report["global_before"] = before_rate
        report["global_after"] = after_rate
        if before_rate is not None and after_rate is not None:
            report["global_delta"] = after_rate - before_rate

    buckets: Dict[Tuple[str, str], Dict[str, List[float]]] = defaultdict(
        lambda: {"before": [], "after": []}
    )
    for label, rows in ("before", before_rows), ("after", after_rows):
        for row in rows:
            instrument = str(_pluck(row, "instrument") or "NA")
            tempo_bin = _tempo_bin(_pluck(row, "tempo.bpm"), tempo_edges)
            value = _to_float(_pluck(row, cos_key))
            if value is None:
                continue
            buckets[(instrument, tempo_bin)][label].append(value)

    bad_buckets: List[Dict[str, Any]] = []
    fallback = None
    if isinstance(tau_global, (int, float)):
        fallback = float(tau_global)
    for (instrument, tempo_bin), samples in buckets.items():
        size = max(len(samples["before"]), len(samples["after"]))
        if size < min_bucket_n:
            continue
        tau = _choose_tau(tau_map, instrument, tempo_bin, fallback)
        if tau is None:
            continue
        before_rate = _mismatch_at_tau(samples["before"], tau)
        after_rate = _mismatch_at_tau(samples["after"], tau)
        if before_rate is None or after_rate is None:
            continue
        delta = after_rate - before_rate
        if delta > max_bucket_regression_pp:
            bad_buckets.append(
                {
                    "inst": instrument,
                    "tempo_bin": tempo_bin,
                    "n": size,
                    "tau": tau,
                    "before": before_rate,
                    "after": after_rate,
                    "delta": delta,
                }
            )
    bad_buckets.sort(
        key=lambda item: (
            -item["delta"],
            -item["n"],
            item["inst"],
            item["tempo_bin"],
        )
    )
    report["bad_buckets"] = bad_buckets

    ok = True
    global_delta = report["global_delta"]
    if global_delta is not None and global_delta > -min_global_improve_pp:
        ok = False
    if len(bad_buckets) > max_regression_buckets:
        ok = False
    return ok, report


def _parse_edges(raw: str | None) -> List[float]:
    edges: List[float] = []
    if not raw:
        return edges
    for piece in raw.split(","):
        piece = piece.strip()
        if not piece:
            continue
        try:
            edges.append(float(piece))
        except ValueError:
            continue
    return edges


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--before", required=True)
    parser.add_argument("--after", required=True)
    parser.add_argument("--tau-file")
    parser.add_argument("--tau-file-mert")
    parser.add_argument("--tempo-edges", default="95,110,130,150")
    parser.add_argument("--min-global-improve-pp", type=float, default=1.0)
    parser.add_argument("--max-bucket-regression-pp", type=float, default=3.0)
    parser.add_argument("--max-regression-buckets", type=int, default=2)
    parser.add_argument("--min-bucket-n", type=int, default=10)
    parser.add_argument(
        "--out",
        help="Write guard summary JSON to this path (optional)",
    )
    args = parser.parse_args(argv)

    before_rows = _read_jsonl(args.before)
    after_rows = _read_jsonl(args.after)
    edges = _parse_edges(args.tempo_edges)
    if not edges:
        edges = [95.0, 110.0, 130.0, 150.0]

    results: List[Dict[str, Any]] = []
    ok_all = True

    if args.tau_file and os.path.exists(args.tau_file):
        ok, report = _guard_single_metric(
            before_rows,
            after_rows,
            args.tau_file,
            "metrics.text_audio_cos",
            edges,
            args.min_global_improve_pp,
            args.max_bucket_regression_pp,
            args.max_regression_buckets,
            args.min_bucket_n,
        )
        results.append({"kind": "CLAP", "ok": ok, "report": report})
        ok_all = ok_all and ok

    if args.tau_file_mert and os.path.exists(args.tau_file_mert):
        ok, report = _guard_single_metric(
            before_rows,
            after_rows,
            args.tau_file_mert,
            "metrics.text_audio_cos_mert",
            edges,
            args.min_global_improve_pp,
            args.max_bucket_regression_pp,
            args.max_regression_buckets,
            args.min_bucket_n,
        )
        results.append({"kind": "MERT", "ok": ok, "report": report})
        ok_all = ok_all and ok

    summary: Dict[str, Any] = {"ok": ok_all, "results": results}
    if args.out:
        directory = os.path.dirname(args.out)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, ensure_ascii=False, indent=2)
        print(f"[guard] wrote {args.out}")
    else:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if ok_all else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # pragma: no cover - defensive logging
        print(f"[tau_promotion_guard] error: {exc}", file=sys.stderr)
        raise SystemExit(1)
