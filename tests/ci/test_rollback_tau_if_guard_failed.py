# -*- coding: utf-8 -*-
"""Tests for rollback_tau_if_guard_failed script."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def _git(cmd: list[str], cwd: Path) -> None:
    subprocess.run(cmd, cwd=cwd, check=True, text=True)


def test_rollback_restores_previous_tau(tmp_path: Path) -> None:
    repo = tmp_path
    (repo / "configs" / "metrics").mkdir(parents=True)
    (repo / "artifacts").mkdir(parents=True)

    _git(["git", "init", "-b", "main"], repo)
    _git(["git", "config", "user.name", "tester"], repo)
    _git(["git", "config", "user.email", "tester@example.com"], repo)

    tau_path = repo / "configs" / "metrics" / "tau.yaml"
    tau_path.write_text("global: 0.50\n", encoding="utf-8")
    _git(["git", "add", "."], repo)
    _git(["git", "commit", "-m", "init"], repo)

    tau_path.write_text("global: 0.55\n", encoding="utf-8")
    _git(["git", "add", "."], repo)
    _git(["git", "commit", "-m", "chore(metrics): promote proposed tau(s)"], repo)

    tau_path.write_text("global: 0.60\n", encoding="utf-8")
    _git(["git", "add", "."], repo)
    _git(["git", "commit", "-m", "chore(metrics): promote proposed tau(s)"], repo)

    guard_payload = {"ok": False, "results": []}
    guard_path = repo / "artifacts" / "tau_guard.summary.json"
    guard_path.write_text(json.dumps(guard_payload), encoding="utf-8")

    script = (
        Path(__file__).resolve().parents[2] / "scripts" / "ci" / "rollback_tau_if_guard_failed.py"
    )
    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--guard-json",
            "artifacts/tau_guard.summary.json",
            "--require-marker",
        ],
        cwd=repo,
        text=True,
    )
    assert completed.returncode == 0

    result = tau_path.read_text(encoding="utf-8")
    assert "global: 0.55" in result
