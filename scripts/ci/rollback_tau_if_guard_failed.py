#!/usr/bin/env python3
"""Rollback tau configs if guard summary reports failure."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from typing import List, Optional, Sequence, Tuple

PROMOTE_MARKER = "chore(metrics): promote proposed tau(s)"


def _run(
    cmd: List[str],
    *,
    cwd: Optional[str] = None,
    check: bool = True,
    capture: bool = True,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=cwd,
        check=check,
        capture_output=capture,
        text=True,
    )


def _read_guard(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _git_recent_commits(paths: List[str], max_count: int = 20) -> List[str]:
    completed = _run(
        ["git", "log", "--pretty=%H", "-n", str(max_count), "--", *paths],
        check=False,
    )
    if completed.returncode != 0:
        return []
    return [line.strip() for line in completed.stdout.splitlines() if line.strip()]


def _git_commit_message(commit: str) -> str:
    completed = _run(["git", "log", "-1", "--pretty=%s", commit], check=False)
    return (completed.stdout or "").strip()


def _find_previous_good_commit(
    paths: List[str],
    require_marker: bool,
    lookback: int = 50,
) -> Tuple[Optional[str], Optional[str]]:
    commits = _git_recent_commits(paths, max_count=lookback)
    if not commits:
        return None, None

    promote_index: Optional[int] = None
    for index, commit in enumerate(commits):
        if PROMOTE_MARKER in _git_commit_message(commit):
            promote_index = index
            break

    if require_marker and promote_index is None:
        return None, None

    if promote_index is None:
        if len(commits) >= 2:
            return commits[1], "no-marker"
        return None, None

    parent_index = promote_index + 1
    if parent_index < len(commits):
        return commits[parent_index], "promote-parent"

    return None, None


def _restore_files_from_commit(commit: str, paths: List[str]) -> List[str]:
    restored: List[str] = []
    for path in paths:
        result = _run(
            ["git", "checkout", commit, "--", path],
            check=False,
        )
        if result.returncode == 0:
            restored.append(path)
        else:
            print(
                f"[rollback] warning: could not restore {path}",
                file=sys.stderr,
            )
    return restored


def _commit_and_push(
    message: str,
    paths: Sequence[str],
    push: bool = True,
) -> bool:
    _run(["git", "config", "user.name", "github-actions[bot]"])
    _run(
        [
            "git",
            "config",
            "user.email",
            "github-actions[bot]@users.noreply.github.com",
        ]
    )
    if not paths:
        return False
    _run(["git", "add", *paths])
    commit = _run(["git", "commit", "-m", message], check=False)
    if commit.returncode != 0:
        return False
    if push:
        _run(["git", "push"], check=False)
    return True


def _diff_preview(commit: str, paths: List[str]) -> None:
    for path in paths:
        subprocess.run(["git", "diff", f"{commit}:{path}", path], check=False)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--guard-json",
        default="artifacts/tau_guard.summary.json",
        help="Path to guard summary JSON",
    )
    parser.add_argument(
        "--paths",
        default="configs/metrics/tau.yaml,configs/metrics/tau_mert.yaml",
        help="Comma-separated list of tau config files to restore",
    )
    parser.add_argument("--lookback", type=int, default=50)
    parser.add_argument(
        "--require-marker",
        action="store_true",
        help=f"Only rollback if history includes '{PROMOTE_MARKER}'",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    if not os.path.exists(args.guard_json):
        print("[rollback] guard file not found; nothing to do")
        return 0

    guard = _read_guard(args.guard_json)
    if guard.get("ok", True):
        print("[rollback] guard PASS; no rollback")
        return 0

    paths = [item.strip() for item in args.paths.split(",") if item.strip()]
    if not paths:
        print("[rollback] no paths specified", file=sys.stderr)
        return 2

    prev_commit, reason = _find_previous_good_commit(
        paths,
        args.require_marker,
        lookback=args.lookback,
    )
    if not prev_commit:
        print(
            "[rollback] no previous good commit found; abort",
            file=sys.stderr,
        )
        return 2

    print(f"[rollback] restoring files from {prev_commit} (reason={reason})")
    if args.dry_run:
        _diff_preview(prev_commit, paths)
        return 0

    restored_paths = _restore_files_from_commit(prev_commit, paths)
    if not restored_paths:
        print("[rollback] nothing to restore; abort", file=sys.stderr)
        return 2
    message = "chore(metrics): rollback tau to " f"{prev_commit[:7]} due to guard failure [skip ci]"
    if _commit_and_push(message, restored_paths, push=True):
        print("[rollback] rollback committed & pushed")
        return 0

    print("[rollback] rollback commit failed", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
