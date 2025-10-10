#!/usr/bin/env python3
"""Promote proposed tau files from artifacts into configs/metrics."""
from __future__ import annotations

import argparse
import os
import subprocess
import time
from typing import List, Tuple

BANNER = "# managed-by: promote_tau_from_artifacts.py\n"


def _exists(path: str | None) -> bool:
    return bool(path) and os.path.exists(str(path))


def _read(path: str) -> str | None:
    if not _exists(path):
        return None
    with open(path, "r", encoding="utf-8") as handle:
        return handle.read()


def _write(path: str, text: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)


def _promote_pair(source: str, destination: str) -> bool:
    payload = _read(source)
    if payload is None:
        return False
    if not payload.lstrip().startswith("#"):
        payload = BANNER + payload
    _write(destination, payload)
    return True


def _log_promotion(entries: List[Tuple[str, str, str]]) -> None:
    os.makedirs("artifacts", exist_ok=True)
    log_path = os.path.join("artifacts", "tau_promotion.log")
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    lines = [f"[{timestamp}] promoted:"]
    for label, src, dst in entries:
        lines.append(f"- {label}: {src} -> {dst}")
    with open(log_path, "a", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    print("\n".join(lines))


def _git_commit() -> None:
    subprocess.run(
        ["git", "config", "user.name", "github-actions[bot]"],
        check=False,
    )
    subprocess.run(
        [
            "git",
            "config",
            "user.email",
            "github-actions[bot]@users.noreply.github.com",
        ],
        check=False,
    )
    subprocess.run(
        [
            "git",
            "add",
            "configs/metrics/tau.yaml",
            "configs/metrics/tau_mert.yaml",
            "artifacts/tau_promotion.log",
        ],
        check=False,
    )
    result = subprocess.run(
        [
            "git",
            "commit",
            "-m",
            "chore(metrics): promote proposed tau(s) from artifacts [skip ci]",
        ],
        check=False,
    )
    if result.returncode == 0:
        subprocess.run(["git", "push"], check=False)
        print("[promote_tau] committed & pushed")
    else:
        print("[promote_tau] no commit created or commit failed")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clap-artifacts", default="artifacts/auto_tau.yaml")
    parser.add_argument(
        "--mert-artifacts",
        default="artifacts/auto_tau_mert.yaml",
    )
    parser.add_argument("--clap-dest", default="configs/metrics/tau.yaml")
    parser.add_argument("--mert-dest", default="configs/metrics/tau_mert.yaml")
    parser.add_argument("--git-commit", action="store_true")
    args = parser.parse_args()

    promoted: List[Tuple[str, str, str]] = []
    if _promote_pair(args.clap_artifacts, args.clap_dest):
        promoted.append(("CLAP", args.clap_artifacts, args.clap_dest))
    if _promote_pair(args.mert_artifacts, args.mert_dest):
        promoted.append(("MERT", args.mert_artifacts, args.mert_dest))

    if not promoted:
        print("[promote_tau] no source artifacts found; nothing to do")
        return 0

    _log_promotion(promoted)

    if args.git_commit:
        _git_commit()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
