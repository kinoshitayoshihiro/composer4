# -*- coding: utf-8 -*-
"""Tests for Stage2 A/B summary commenter."""
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


def test_make_comment_includes_tau_preview(tmp_path: Path) -> None:
    before_rows: List[Dict[str, Any]] = [
        {
            "loop_id": "a1",
            "axes_raw": {"velocity": 0.40, "structure": 0.50},
            "metrics": {
                "text_audio_cos": 0.45,
                "text_audio_cos_mert": 0.35,
            },
        },
        {
            "loop_id": "a2",
            "axes_raw": {"velocity": 0.43, "structure": 0.51},
            "metrics": {
                "text_audio_cos": 0.52,
                "text_audio_cos_mert": 0.38,
            },
        },
    ]
    after_rows: List[Dict[str, Any]] = [
        {
            "loop_id": "b1",
            "axes_raw": {"velocity": 0.48, "structure": 0.54},
            "metrics": {
                "text_audio_cos": 0.61,
                "text_audio_cos_mert": 0.50,
            },
        },
        {
            "loop_id": "b2",
            "axes_raw": {"velocity": 0.50, "structure": 0.57},
            "metrics": {
                "text_audio_cos": 0.63,
                "text_audio_cos_mert": 0.55,
            },
        },
    ]

    before_path = tmp_path / "before.jsonl"
    after_path = tmp_path / "after.jsonl"
    before_path.write_text(
        "\n".join(
            json.dumps(row, ensure_ascii=False) for row in before_rows
        ),
        encoding="utf-8",
    )
    after_path.write_text(
        "\n".join(
            json.dumps(row, ensure_ascii=False) for row in after_rows
        ),
        encoding="utf-8",
    )

    repo_root = Path(__file__).resolve().parents[2]
    ab_comment = _import_ab_comment(repo_root)

    loaded_before = ab_comment.read_jsonl(str(before_path))
    loaded_after = ab_comment.read_jsonl(str(after_path))

    axes = ["velocity", "structure"]
    tau_current = 0.50
    tau_proposed = 0.47
    tau_mert = 0.42
    tau_mert_proposed = 0.40

    ab_comment.before_rows_cache = loaded_before  # type: ignore[attr-defined]
    ab_comment.after_rows_cache = loaded_after  # type: ignore[attr-defined]

    metrics_before = ab_comment.collect_metrics(
        loaded_before,
        axes,
        tau_current,
    )
    metrics_after = ab_comment.collect_metrics(
        loaded_after,
        axes,
        tau_current,
    )

    body = ab_comment.make_comment(
        before=metrics_before,
        after=metrics_after,
        axes=axes,
        title="Stage2 A/B Summary",
        marker="AB_SUMMARY",
        tau_current=tau_current,
        tau_proposed=tau_proposed,
        tau_mert_current=tau_mert,
        tau_mert_proposed=tau_mert_proposed,
        before_rows_memo=loaded_before,
        after_rows_memo=loaded_after,
    )

    assert "Tau Proposal Preview" in body
    assert "caption_mismatch_rate@tau_proposed" in body
    assert "text_audio_cos_mert.p50" in body
    assert "tau_mert_proposed" in body
