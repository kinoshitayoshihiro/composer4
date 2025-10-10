# -*- coding: utf-8 -*-
"""Tests for the tau_promotion_guard script."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def _script_path() -> Path:
    root = Path(__file__).resolve().parents[2]
    return root / "scripts" / "ci" / "tau_promotion_guard.py"


def _write_jsonl(target: Path, rows: list[dict[str, object]]) -> None:
    payload = "\n".join(json.dumps(row) for row in rows)
    target.write_text(payload, encoding="utf-8")


def test_tau_promotion_guard_allows_expected_delta(tmp_path: Path) -> None:
    tau_file = tmp_path / "tau.yaml"
    tau_file.write_text("global: 0.500000\n", encoding="utf-8")

    before = tmp_path / "before.jsonl"
    after = tmp_path / "after.jsonl"

    before_rows: list[dict[str, object]] = [
        {
            "instrument": "Drums",
            "tempo": {"bpm": 110},
            "metrics": {"text_audio_cos": 0.45},
        },
        {
            "instrument": "Drums",
            "tempo": {"bpm": 110},
            "metrics": {"text_audio_cos": 0.40},
        },
        {
            "instrument": "Drums",
            "tempo": {"bpm": 110},
            "metrics": {"text_audio_cos": 0.48},
        },
        {
            "instrument": "Drums",
            "tempo": {"bpm": 110},
            "metrics": {"text_audio_cos": 0.49},
        },
    ]
    after_rows: list[dict[str, object]] = [
        {
            "instrument": "Drums",
            "tempo": {"bpm": 110},
            "metrics": {"text_audio_cos": 0.55},
        },
        {
            "instrument": "Drums",
            "tempo": {"bpm": 110},
            "metrics": {"text_audio_cos": 0.58},
        },
        {
            "instrument": "Drums",
            "tempo": {"bpm": 110},
            "metrics": {"text_audio_cos": 0.62},
        },
        {
            "instrument": "Drums",
            "tempo": {"bpm": 110},
            "metrics": {"text_audio_cos": 0.60},
        },
    ]

    _write_jsonl(before, before_rows)
    _write_jsonl(after, after_rows)

    proc = subprocess.run(
        [
            sys.executable,
            str(_script_path()),
            "--before",
            str(before),
            "--after",
            str(after),
            "--tau-file",
            str(tau_file),
            "--tempo-edges",
            "95,110,130",
            "--min-bucket-n",
            "1",
            "--min-global-improve-pp",
            "0.1",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )

    assert proc.returncode == 0
    payload = json.loads(proc.stdout)
    assert payload["ok"] is True


def test_tau_promotion_guard_blocks_large_delta(tmp_path: Path) -> None:
    tau_file = tmp_path / "tau.yaml"
    tau_file.write_text("global: 0.500000\n", encoding="utf-8")

    before = tmp_path / "before.jsonl"
    after = tmp_path / "after.jsonl"

    before_rows: list[dict[str, object]] = [
        {
            "instrument": "Piano",
            "tempo": {"bpm": 90},
            "metrics": {"text_audio_cos": 0.60},
        },
        {
            "instrument": "Piano",
            "tempo": {"bpm": 90},
            "metrics": {"text_audio_cos": 0.58},
        },
        {
            "instrument": "Piano",
            "tempo": {"bpm": 90},
            "metrics": {"text_audio_cos": 0.62},
        },
    ]
    after_rows: list[dict[str, object]] = [
        {
            "instrument": "Piano",
            "tempo": {"bpm": 90},
            "metrics": {"text_audio_cos": 0.40},
        },
        {
            "instrument": "Piano",
            "tempo": {"bpm": 90},
            "metrics": {"text_audio_cos": 0.42},
        },
        {
            "instrument": "Piano",
            "tempo": {"bpm": 90},
            "metrics": {"text_audio_cos": 0.38},
        },
    ]

    _write_jsonl(before, before_rows)
    _write_jsonl(after, after_rows)

    proc = subprocess.run(
        [
            sys.executable,
            str(_script_path()),
            "--before",
            str(before),
            "--after",
            str(after),
            "--tau-file",
            str(tau_file),
            "--tempo-edges",
            "80,100,120",
            "--min-bucket-n",
            "1",
            "--min-global-improve-pp",
            "0.1",
            "--max-bucket-regression-pp",
            "1.0",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )

    assert proc.returncode == 2
    payload = json.loads(proc.stdout)
    assert payload["ok"] is False
    assert payload["results"]
