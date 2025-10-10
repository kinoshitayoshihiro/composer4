# -*- coding: utf-8 -*-
"""Tests for tau bucket preview sections in PR dashboard aggregator."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any, Dict, List


def _import_aggregator() -> Any:
    root = Path(__file__).resolve().parents[2]
    script = root / "scripts" / "ci" / "pr_comment_aggregator.py"
    spec = importlib.util.spec_from_file_location("pr_comment_aggregator", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)  # type: ignore[arg-type]
    return module


def _write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    payload = "\n".join(json.dumps(row) for row in rows)
    path.write_text(payload, encoding="utf-8")


def test_tau_bucket_preview_clap_and_mert(tmp_path: Path) -> None:
    mod = _import_aggregator()

    before_rows = [
        {
            "instrument": "Drums",
            "tempo": {"bpm": 100},
            "metrics": {
                "text_audio_cos": 0.40,
                "text_audio_cos_mert": 0.35,
            },
        },
        {
            "instrument": "Drums",
            "tempo": {"bpm": 100},
            "metrics": {
                "text_audio_cos": 0.60,
                "text_audio_cos_mert": 0.55,
            },
        },
        {
            "instrument": "Piano",
            "tempo": {"bpm": 140},
            "metrics": {
                "text_audio_cos": 0.45,
                "text_audio_cos_mert": 0.42,
            },
        },
    ]
    after_rows = [
        {
            "instrument": "Drums",
            "tempo": {"bpm": 100},
            "metrics": {
                "text_audio_cos": 0.65,
                "text_audio_cos_mert": 0.60,
            },
        },
        {
            "instrument": "Drums",
            "tempo": {"bpm": 100},
            "metrics": {
                "text_audio_cos": 0.62,
                "text_audio_cos_mert": 0.58,
            },
        },
        {
            "instrument": "Piano",
            "tempo": {"bpm": 140},
            "metrics": {
                "text_audio_cos": 0.50,
                "text_audio_cos_mert": 0.45,
            },
        },
    ]

    before_path = tmp_path / "before.jsonl"
    after_path = tmp_path / "after.jsonl"
    _write_jsonl(before_path, before_rows)
    _write_jsonl(after_path, after_rows)

    clap_tau = tmp_path / "auto_tau.yaml"
    clap_tau.write_text(
        "\n".join(
            [
                "# auto_tau proposal",
                "global: 0.500000",
                "by_instrument:",
                "  Piano: 0.470000",
                "by_instrument_tempo:",
                "  Drums:",
                "    <110: 0.580000",
            ]
        ),
        encoding="utf-8",
    )

    mert_tau = tmp_path / "auto_tau_mert.yaml"
    mert_tau.write_text(
        "\n".join(
            [
                "global: 0.480000",
                "by_tempo:",
                "  <110: 0.520000",
                "  >=150: 0.500000",
            ]
        ),
        encoding="utf-8",
    )

    before_loaded = mod.read_jsonl(str(before_path))
    after_loaded = mod.read_jsonl(str(after_path))
    edges = [95.0, 110.0, 130.0, 150.0]

    clap_section = mod.build_tau_bucket_preview(
        "CLAP Tau Preview",
        before_loaded,
        after_loaded,
        str(clap_tau),
        "metrics.text_audio_cos",
        edges,
    )
    mert_section = mod.build_tau_bucket_preview(
        "MERT Tau Preview",
        before_loaded,
        after_loaded,
        str(mert_tau),
        "metrics.text_audio_cos_mert",
        edges,
    )

    assert clap_section is not None
    assert "CLAP Tau Preview" in clap_section
    assert "Drums × <110" in clap_section

    assert mert_section is not None
    assert "MERT Tau Preview" in mert_section
    # ensure by_tempo override is reflected in the output
    assert "<110" in mert_section
