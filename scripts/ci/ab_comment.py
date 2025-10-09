#!/usr/bin/env python3
"""Stage2 A/B summary commenter with tau preview support."""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import (
    Any,
    Dict,
    List,
    Mapping,
    MutableMapping,
    Optional,
    Sequence,
    Tuple,
)

MARKER_TEMPLATE = "<!-- {marker} -->"


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
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


def pluck(dct: Mapping[str, Any], dotted: str, default: Any = None) -> Any:
    current: Any = dct
    for part in dotted.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return default
        current = current[part]
    return current


def _median(values: Sequence[Optional[float]]) -> Optional[float]:
    cleaned = [v for v in values if isinstance(v, (int, float))]
    if not cleaned:
        return None
    return float(statistics.median(cleaned))


def fmt_val(value: Optional[float], suffix: str = "") -> str:
    if value is None:
        return "—"
    return f"{value:.3f}{suffix}"


def delta(a: Optional[float], b: Optional[float]) -> Optional[float]:
    if a is None or b is None:
        return None
    return b - a


def fmt_delta(value: Optional[float], suffix: str = "") -> str:
    if value is None:
        return "—"
    return f"{value:+.3f}{suffix}"


@dataclass
class MetricsBlock:
    axes: Dict[str, Optional[float]]
    cos_p50: Optional[float]
    mismatch_rate: Optional[float]


def collect_metrics(
    rows: Sequence[Mapping[str, Any]],
    axes: Sequence[str],
    tau: float,
) -> Dict[str, Dict[str, Optional[float]]]:
    axes_block: Dict[str, Optional[float]] = {}
    cos_values: List[Optional[float]] = []
    for axis in axes:
        axis_values: List[Optional[float]] = []
        for row in rows:
            axis_values.append(pluck(row, f"axes_raw.{axis}"))
        axes_block[axis] = _median(
            [
                float(value) if isinstance(value, (int, float)) else None
                for value in axis_values
            ]
        )

    for row in rows:
        value = pluck(row, "metrics.text_audio_cos")
        cos_values.append(
            float(value) if isinstance(value, (int, float)) else None
        )

    cos_filtered = [v for v in cos_values if v is not None]
    cos_p50 = _median(cos_filtered)

    mismatch_rate: Optional[float] = None
    if cos_filtered:
        mismatch_rate = 100.0 * (
            sum(1 for value in cos_filtered if value < tau) / len(cos_filtered)
        )

    return {
        "axes": axes_block,
        "text_audio_cos": {"p50": cos_p50},
        "mismatch_rate": {"rate": mismatch_rate},
    }


def collect_mert(
    rows: Sequence[Mapping[str, Any]],
    tau: Optional[float],
) -> Dict[str, Optional[float]]:
    values: List[Optional[float]] = []
    for row in rows:
        val = pluck(row, "metrics.text_audio_cos_mert")
        values.append(float(val) if isinstance(val, (int, float)) else None)

    filtered = [v for v in values if v is not None]
    if not filtered:
        return {"p50": None, "mismatch_rate": None}

    p50 = _median(filtered)
    mismatch: Optional[float] = None
    if tau is not None:
        mismatch = 100.0 * (
            sum(1 for value in filtered if value < tau) / len(filtered)
        )
    return {"p50": p50, "mismatch_rate": mismatch}


def _parse_simple_yaml(path: Optional[str]) -> Dict[str, Any]:
    if not path or not os.path.exists(path):
        return {}
    root: Dict[str, Any] = {}
    stack: List[Tuple[int, MutableMapping[str, Any]]] = [(-1, root)]

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


before_rows_cache: List[Dict[str, Any]] = []
after_rows_cache: List[Dict[str, Any]] = []


def _recalc_mismatch(
    rows: Sequence[Mapping[str, Any]],
    tau: float,
) -> Optional[float]:
    values: List[float] = []
    for row in rows:
        val = pluck(row, "metrics.text_audio_cos")
        if isinstance(val, (int, float)):
            values.append(float(val))
    if not values:
        return None
    return 100.0 * (sum(1 for entry in values if entry < tau) / len(values))


def _recalc_mismatch_key(
    rows: Sequence[Mapping[str, Any]],
    key: str,
    tau: float,
) -> Optional[float]:
    values: List[float] = []
    for row in rows:
        val = pluck(row, key)
        if isinstance(val, (int, float)):
            values.append(float(val))
    if not values:
        return None
    return 100.0 * (sum(1 for entry in values if entry < tau) / len(values))


def mark_comment(marker: str) -> str:
    return MARKER_TEMPLATE.format(marker=marker)


def make_comment(
    before: Dict[str, Any],
    after: Dict[str, Any],
    axes: Sequence[str],
    title: str,
    marker: str,
    tau_current: float,
    tau_proposed: Optional[float],
    tau_mert_current: Optional[float],
    tau_mert_proposed: Optional[float],
    before_rows_memo: Optional[Sequence[Mapping[str, Any]]] = None,
    after_rows_memo: Optional[Sequence[Mapping[str, Any]]] = None,
) -> str:
    lines: List[str] = []
    lines.append(mark_comment(marker))
    lines.append(f"## {title}")
    lines.append("")

    if tau_proposed is not None:
        lines.append(
            "_tau (current/proposed):"
            f" **{tau_current:.3f} → {tau_proposed:.3f}**_"
        )
        lines.append("")

    lines.append("**Axes (p50)**")
    for axis in axes:
        before_axis = before["axes"].get(axis)
        after_axis = after["axes"].get(axis)
        lines.append(
            "- `{axis}` | before: {before_val} → after: {after_val}"
            " | Δ {delta_val}".format(
                axis=axis,
                before_val=fmt_val(before_axis),
                after_val=fmt_val(after_axis),
                delta_val=fmt_delta(delta(before_axis, after_axis)),
            )
        )
    lines.append("")

    before_cos = before["text_audio_cos"].get("p50")
    after_cos = after["text_audio_cos"].get("p50")
    before_mismatch = before["mismatch_rate"].get("rate")
    after_mismatch = after["mismatch_rate"].get("rate")
    lines.append("**CLAP**")
    lines.append(
        "- `text_audio_cos.p50` | before: {before_val} → after: {after_val}"
        " | Δ {delta_val}".format(
            before_val=fmt_val(before_cos),
            after_val=fmt_val(after_cos),
            delta_val=fmt_delta(delta(before_cos, after_cos)),
        )
    )
    lines.append(
        "- `caption_mismatch_rate` | before: {before_val} → after: {after_val}"
        " | Δ {delta_val}".format(
            before_val=fmt_val(before_mismatch, "%"),
            after_val=fmt_val(after_mismatch, "%"),
            delta_val=fmt_delta(
                delta(before_mismatch, after_mismatch),
                "pp",
            ),
        )
    )

    if (
        tau_proposed is not None
        and before_rows_memo is not None
        and after_rows_memo is not None
    ):
        lines.append("")
        lines.append("**Tau Proposal Preview (global)**")
        lines.append(
            f"_Recomputed mismatch with proposed tau = {tau_proposed:.3f}_"
        )
        before_preview = _recalc_mismatch(before_rows_memo, tau_proposed)
        after_preview = _recalc_mismatch(after_rows_memo, tau_proposed)
        lines.append(
            "- `caption_mismatch_rate@tau_proposed` | before: {before_val}"
            " → after: {after_val}".format(
                before_val=fmt_val(before_preview, "%"),
                after_val=fmt_val(after_preview, "%"),
            )
        )

    if before_rows_memo is not None and after_rows_memo is not None:
        mert_before = collect_mert(before_rows_memo, tau_mert_current)
        mert_after = collect_mert(after_rows_memo, tau_mert_current)
        if mert_before["p50"] is not None or mert_after["p50"] is not None:
            lines.append("")
            lines.append("**MERT**")
            lines.append(
                "- `text_audio_cos_mert.p50` | before: {before_val}"
                " → after: {after_val} | Δ {delta_val}".format(
                    before_val=fmt_val(mert_before["p50"]),
                    after_val=fmt_val(mert_after["p50"]),
                    delta_val=fmt_delta(
                        delta(
                            mert_before["p50"],
                            mert_after["p50"],
                        )
                    ),
                )
            )
            if tau_mert_current is not None:
                lines.append(
                    "- `mismatch_rate@mert_tau` | before: {before_val}"
                    " → after: {after_val}".format(
                        before_val=fmt_val(
                            mert_before["mismatch_rate"],
                            "%",
                        ),
                        after_val=fmt_val(
                            mert_after["mismatch_rate"],
                            "%",
                        ),
                    )
                )
            if tau_mert_proposed is not None:
                lines.append("")
                lines.append(f"_@tau_mert_proposed = {tau_mert_proposed:.3f}_")
                before_mert_prev = _recalc_mismatch_key(
                    before_rows_memo,
                    "metrics.text_audio_cos_mert",
                    tau_mert_proposed,
                )
                after_mert_prev = _recalc_mismatch_key(
                    after_rows_memo,
                    "metrics.text_audio_cos_mert",
                    tau_mert_proposed,
                )
                lines.append(
                    f"- `mismatch_rate@tau_mert_proposed` | before: {fmt_val(before_mert_prev, '%')} → after: {fmt_val(after_mert_prev, '%')}"
                )

    lines.append("")
    lines.append("_auto-posted by AB summary bot_")
    return "\n".join(lines)


def gh_request(url: str, method: str = "GET", body: Optional[Mapping[str, Any]] = None, token: Optional[str] = None) -> Any:
    req = urllib.request.Request(url, method=method)
    req.add_header("Accept", "application/vnd.github+json")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    data = None
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, data=data) as resp:  # type: ignore[arg-type]
        return json.loads(resp.read().decode("utf-8"))


def find_existing_comment(repo: str, pr_number: int, marker: str, token: str) -> Optional[int]:
    page = 1
    while True:
        url = f"https://api.github.com/repos/{repo}/issues/{pr_number}/comments?per_page=100&page={page}"
        comments = gh_request(url, token=token)
        if not comments:
            return None
        for comment in comments:
            body = comment.get("body") or ""
            if marker in body:
                return int(comment["id"])
        page += 1


def upsert_comment(repo: str, pr_number: int, marker: str, body: str, token: str) -> None:
    existing = find_existing_comment(repo, pr_number, marker, token)
    if existing is not None:
        url = f"https://api.github.com/repos/{repo}/issues/comments/{existing}"
        gh_request(url, method="PATCH", body={"body": body}, token=token)
    else:
        url = f"https://api.github.com/repos/{repo}/issues/{pr_number}/comments"
        gh_request(url, method="POST", body={"body": body}, token=token)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--before", required=True, help="before JSONL path")
    parser.add_argument("--after", required=True, help="after JSONL path")
    parser.add_argument("--axes", action="append", default=["velocity", "structure"], help="axes to summarise (multiple allowed)")
    parser.add_argument("--tau", type=float, default=0.50, help="threshold for caption mismatch")
    parser.add_argument("--tau-file", default=None, help="YAML produced by auto_tau.py")
    parser.add_argument("--tau-mert", type=float, default=None, help="threshold for MERT mismatch")
    parser.add_argument("--tau-file-mert", default=None, help="auto_tau.py YAML for MERT")
    parser.add_argument("--title", default="Stage2 A/B Summary")
    parser.add_argument("--marker", default="AB_SUMMARY: composer4")
    parser.add_argument("--repo", default=os.environ.get("GITHUB_REPOSITORY"))
    parser.add_argument("--pr-number", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def load_tau(path: Optional[str]) -> Optional[float]:
    if not path:
        return None
    data = _parse_simple_yaml(path)
    value = data.get("global")
    return float(value) if isinstance(value, (int, float)) else None


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    before_rows = read_jsonl(args.before)
    after_rows = read_jsonl(args.after)

    global before_rows_cache, after_rows_cache
    before_rows_cache = before_rows
    after_rows_cache = after_rows

    before_metrics = collect_metrics(before_rows, args.axes, args.tau)
    after_metrics = collect_metrics(after_rows, args.axes, args.tau)

    tau_proposed = load_tau(args.tau_file)
    tau_mert_current = args.tau_mert
    tau_mert_proposed = load_tau(args.tau_file_mert)

    body = make_comment(
        before=before_metrics,
        after=after_metrics,
        axes=args.axes,
        title=args.title,
        marker=args.marker,
        tau_current=args.tau,
        tau_proposed=tau_proposed,
        tau_mert_current=tau_mert_current,
        tau_mert_proposed=tau_mert_proposed,
        before_rows_memo=before_rows,
        after_rows_memo=after_rows,
    )

    if args.dry_run:
        print(body)
        return 0

    repo = args.repo
    pr_number = args.pr_number
    token = os.environ.get("GITHUB_TOKEN")
    if not repo or not token:
        print("[ab_comment] Missing repo or token; use --dry-run for local preview.", file=sys.stderr)
        return 1
    if pr_number is None:
        event_path = os.environ.get("GITHUB_EVENT_PATH")
        if event_path and os.path.exists(event_path):
            with open(event_path, "r", encoding="utf-8") as handle:
                event = json.load(handle)
            pr_number = pluck(event, "pull_request.number")
    if pr_number is None:
        print("[ab_comment] Could not determine PR number; use --dry-run.", file=sys.stderr)
        return 1

    upsert_comment(repo=str(repo), pr_number=int(pr_number), marker=mark_comment(args.marker), body=body, token=str(token))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
