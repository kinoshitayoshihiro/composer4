#!/usr/bin/env python3
"""CLAP multi-crops summary commenter."""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

MARKER = "<!-- CLAP_CROPS -->"


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not os.path.exists(path):
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


def _median(values: List[Optional[float]]) -> Optional[float]:
    filtered = [float(v) for v in values if isinstance(v, (int, float))]
    if not filtered:
        return None
    return float(statistics.median(filtered))


def _fmt(value: Optional[float]) -> str:
    if value is None:
        return "—"
    return f"{value:.3f}"


def collect(rows: List[Dict[str, Any]]) -> Dict[str, Optional[float]]:
    return {
        "n": len(rows),
        "cos_mean_p50": _median([row.get("cos_mean") for row in rows]),
        "cos_std_p50": _median([row.get("cos_std") for row in rows]),
        "cos_iqr_p50": _median([row.get("cos_iqr") for row in rows]),
    }


def make_comment(stats: Dict[str, Optional[float]], title: str) -> str:
    lines: List[str] = []
    lines.append(MARKER)
    lines.append(f"## {title}")
    lines.append(f"- files: {stats['n']}")
    lines.append(f"- cos_mean.p50: {_fmt(stats['cos_mean_p50'])}")
    lines.append(f"- cos_std.p50: {_fmt(stats['cos_std_p50'])}")
    lines.append(f"- cos_iqr.p50: {_fmt(stats['cos_iqr_p50'])}")
    lines.append("")
    lines.append(
        "_狙い: 長尺・変動素材で一致スコアの安定化（分散・IQRの減少を確認）_"
    )
    return "\n".join(lines)


def _github_request(
    url: str,
    method: str = "GET",
    body: Optional[Dict[str, Any]] = None,
    token: Optional[str] = None,
) -> Any:
    request = urllib.request.Request(url, method=method)
    request.add_header("Accept", "application/vnd.github+json")
    if token:
        request.add_header("Authorization", f"Bearer {token}")
    data = None
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        request.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(  # type: ignore[arg-type]
        request,
        data=data,
    ) as response:
        payload = response.read().decode("utf-8")
    return json.loads(payload)


def _get_pr_number_from_event() -> Optional[int]:
    event_path = os.environ.get("GITHUB_EVENT_PATH")
    if not event_path or not os.path.exists(event_path):
        return None
    with open(event_path, "r", encoding="utf-8") as handle:
        event = json.load(handle)
    number = event.get("pull_request", {}).get("number")
    return int(number) if isinstance(number, int) else None


def _find_existing_comment(
    repo: str, pr_number: int, marker: str, token: str
) -> Tuple[Optional[int], Optional[str]]:
    page = 1
    while True:
        url = (
            "https://api.github.com/repos/"
            f"{repo}/issues/{pr_number}/comments?per_page=100&page={page}"
        )
        comments = _github_request(url, token=token)
        if not comments:
            return None, None
        for comment in comments:
            body = comment.get("body") or ""
            if marker in body:
                return int(comment.get("id")), body
        page += 1


def _create_or_update_comment(
    repo: str,
    pr_number: int,
    body: str,
    token: str,
) -> None:
    comment_id, _ = _find_existing_comment(repo, pr_number, MARKER, token)
    if comment_id is not None:
        url = (
            "https://api.github.com/repos/"
            f"{repo}/issues/comments/{comment_id}"
        )
        _github_request(url, method="PATCH", body={"body": body}, token=token)
    else:
        url = (
            "https://api.github.com/repos/"
            f"{repo}/issues/{pr_number}/comments"
        )
        _github_request(url, method="POST", body={"body": body}, token=token)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--jsonl",
        required=True,
        help="path to clap multicrops jsonl",
    )
    parser.add_argument("--title", default="CLAP Multi-crops (10s)")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    rows = read_jsonl(args.jsonl)
    if not rows:
        print("[clap_crops_comment] no rows; skip", file=sys.stderr)
        return 0

    stats = collect(rows)
    body = make_comment(stats, args.title)

    if args.dry_run:
        print(body)
        return 0

    token = os.environ.get("GITHUB_TOKEN")
    repo = os.environ.get("GITHUB_REPOSITORY")
    pr_number = _get_pr_number_from_event()
    if not all([token, repo, pr_number]):
        print(
            "[clap_crops_comment] Missing envs. Use --dry-run locally.",
            file=sys.stderr,
        )
        return 1

    assert repo is not None
    assert token is not None
    assert pr_number is not None
    _create_or_update_comment(str(repo), int(pr_number), body, str(token))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
