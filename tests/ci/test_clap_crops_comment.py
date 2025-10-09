# -*- coding: utf-8 -*-
"""Tests for CLAP multi-crops commenter."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any, Dict, List


def _import_module(repo_root: Path):
    module_path = repo_root / "scripts" / "ci" / "clap_crops_comment.py"
    spec = importlib.util.spec_from_file_location(
        "clap_crops_comment", module_path
    )
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


def test_collect_and_make_comment(tmp_path: Path) -> None:
    rows: List[Dict[str, Any]] = [
        {"file": "a.wav", "cos_mean": 0.52, "cos_std": 0.03, "cos_iqr": 0.04},
        {"file": "b.wav", "cos_mean": 0.58, "cos_std": 0.05, "cos_iqr": 0.06},
        {"file": "c.wav", "cos_mean": 0.60, "cos_std": 0.06, "cos_iqr": 0.07},
        {"file": "d.wav", "cos_mean": None, "cos_std": None, "cos_iqr": None},
    ]

    jsonl_path = tmp_path / "clap.jsonl"
    jsonl_path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows),
        encoding="utf-8",
    )

    repo_root = Path(__file__).resolve().parents[2]
    module = _import_module(repo_root)

    loaded = module.read_jsonl(str(jsonl_path))
    stats = module.collect(loaded)

    assert stats["n"] == 4
    assert stats["cos_mean_p50"] == 0.58

    body = module.make_comment(stats, "CLAP Multi-crops (10s)")
    assert "<!-- CLAP_CROPS -->" in body
    assert "cos_mean.p50" in body
    assert "files: 4" in body
