# -*- coding: utf-8 -*-
"""Tests for guard summary rendering in the PR dashboard aggregator."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any


def _import_aggregator(repo_root: Path) -> Any:
    script = repo_root / "scripts" / "ci" / "pr_comment_aggregator.py"
    spec = importlib.util.spec_from_file_location("pr_comment_aggregator", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)  # type: ignore[arg-type]
    return module


def test_build_guard_section(tmp_path: Path) -> None:
    guard_payload = {
        "ok": False,
        "results": [
            {
                "kind": "CLAP",
                "ok": False,
                "report": {
                    "global_delta": -1.2,
                    "bad_buckets": [{"delta": 4.1}, {"delta": 3.0}],
                },
            },
            {
                "kind": "MERT",
                "ok": True,
                "report": {"global_delta": -0.6, "bad_buckets": []},
            },
        ],
    }
    guard_path = tmp_path / "tau_guard.summary.json"
    guard_path.write_text(json.dumps(guard_payload), encoding="utf-8")

    repo_root = Path(__file__).resolve().parents[2]
    module = _import_aggregator(repo_root)
    section = module.build_guard_section(str(guard_path))

    assert section is not None
    assert "Guard Summary" in section
    assert "FAIL" in section
    assert "CLAP: global Δ" in section
    assert "bad buckets: 2" in section
