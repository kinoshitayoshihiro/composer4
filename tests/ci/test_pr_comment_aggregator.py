# -*- coding: utf-8 -*-
"""Smoke tests for Composer4 PR dashboard aggregator."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def _import_dash(repo_root: Path):
    module_path = repo_root / "scripts" / "ci" / "pr_comment_aggregator.py"
    spec = importlib.util.spec_from_file_location("pr_comment_aggregator", module_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


def test_build_sections_smoke(tmp_path: Path) -> None:
    before_rows = [
        {
            "axes_raw": {"velocity": 0.40, "structure": 0.50},
            "metrics": {"text_audio_cos": 0.45},
        }
    ]
    after_rows = [
        {
            "axes_raw": {"velocity": 0.50, "structure": 0.55},
            "metrics": {"text_audio_cos": 0.60},
        }
    ]

    before_path = tmp_path / "before.jsonl"
    after_path = tmp_path / "after.jsonl"
    before_path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in before_rows),
        encoding="utf-8",
    )
    after_path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in after_rows),
        encoding="utf-8",
    )

    crops_rows = [
        {"cos_mean": 0.55, "cos_std": 0.04, "cos_iqr": 0.05},
    ]
    crops_path = tmp_path / "clap_multicrops.jsonl"
    crops_path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in crops_rows),
        encoding="utf-8",
    )

    repo_root = Path(__file__).resolve().parents[2]
    module = _import_dash(repo_root)

    before_loaded = module.read_jsonl(str(before_path))
    after_loaded = module.read_jsonl(str(after_path))
    crops_loaded = module.read_jsonl(str(crops_path))

    ab_section = module.build_ab_section(
        before_loaded,
        after_loaded,
        ["velocity", "structure"],
        tau=0.5,
        tau_file=None,
    )
    crops_section = module.build_crops_section(crops_loaded)

    assert ab_section is not None
    assert "A/B Summary" in ab_section

    assert crops_section is not None
    assert "CLAP Multi-crops" in crops_section
