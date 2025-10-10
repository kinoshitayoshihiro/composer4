# -*- coding: utf-8 -*-
"""Tests for optional MERT block in A/B commenter."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Dict, List


def _import_ab_comment(repo_root: Path):
    module_path = repo_root / "scripts" / "ci" / "ab_comment.py"
    spec = importlib.util.spec_from_file_location("ab_comment", module_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


def test_mert_optional_block(tmp_path: Path) -> None:
    before_rows: List[Dict[str, Any]] = [
        {
            "axes_raw": {"velocity": 0.40, "structure": 0.50},
            "metrics": {
                "text_audio_cos": 0.45,
                "text_audio_cos_mert": 0.40,
            },
        }
    ]
    after_rows: List[Dict[str, Any]] = [
        {
            "axes_raw": {"velocity": 0.50, "structure": 0.55},
            "metrics": {
                "text_audio_cos": 0.60,
                "text_audio_cos_mert": 0.50,
            },
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

    repo_root = Path(__file__).resolve().parents[2]
    module = _import_ab_comment(repo_root)

    loaded_before = module.read_jsonl(str(before_path))
    loaded_after = module.read_jsonl(str(after_path))

    module.before_rows_cache = loaded_before  # type: ignore[attr-defined]
    module.after_rows_cache = loaded_after  # type: ignore[attr-defined]

    metrics_before = module.collect_metrics(loaded_before, ["velocity", "structure"], 0.50)
    metrics_after = module.collect_metrics(loaded_after, ["velocity", "structure"], 0.50)

    mert_before = module.collect_mert(loaded_before, tau=0.45)
    mert_after = module.collect_mert(loaded_after, tau=0.45)

    assert mert_before["p50"] == 0.40
    assert mert_after["p50"] == 0.50
    assert mert_before["mismatch_rate"] is not None
    assert mert_after["mismatch_rate"] is not None

    body = module.make_comment(
        before=metrics_before,
        after=metrics_after,
        axes=["velocity", "structure"],
        title="Stage2 A/B Summary",
        marker="AB_SUMMARY",
        tau_current=0.50,
        tau_proposed=None,
        tau_mert_current=0.45,
        tau_mert_proposed=0.48,
        before_rows_memo=loaded_before,
        after_rows_memo=loaded_after,
    )

    assert "**MERT**" in body
    assert "text_audio_cos_mert.p50" in body
    assert "mismatch_rate@mert_tau" in body
    assert "tau_mert_proposed" in body
