# -*- coding: utf-8 -*-
"""Ensure dashboard prefers config tau files when no CLI overrides."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def _script_path() -> Path:
    root = Path(__file__).resolve().parents[2]
    return root / "scripts" / "ci" / "pr_comment_aggregator.py"


def _write_jsonl(target: Path, rows: list[dict[str, object]]) -> None:
    payload = "\n".join(json.dumps(row) for row in rows)
    target.write_text(payload, encoding="utf-8")


def test_dashboard_prefers_config_tau_when_present(tmp_path: Path) -> None:
    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir()

    clap_crops = artifacts_dir / "clap_multicrops.jsonl"
    mert_crops = artifacts_dir / "mert_multicrops.jsonl"
    _write_jsonl(
        clap_crops,
        [
            {
                "instrument": "Piano",
                "tempo_bucket": "128",
                "cos_mean": 0.5,
                "cos_delta": 0.1,
            }
        ],
    )
    _write_jsonl(
        mert_crops,
        [
            {
                "instrument": "Piano",
                "tempo_bucket": "128",
                "metric": "cosine",
                "cos_mean": 0.6,
                "cos_delta": 0.05,
            }
        ],
    )

    before = artifacts_dir / "before.jsonl"
    after = artifacts_dir / "after.jsonl"
    before_rows: list[dict[str, object]] = [
        {
            "instrument": "Piano",
            "tempo": {"bpm": 128},
            "metrics": {
                "text_audio_cos": 0.42,
                "text_audio_cos_mert": 0.43,
            },
        },
        {
            "instrument": "Piano",
            "tempo": {"bpm": 128},
            "metrics": {
                "text_audio_cos": 0.41,
                "text_audio_cos_mert": 0.40,
            },
        },
    ]
    after_rows: list[dict[str, object]] = [
        {
            "instrument": "Piano",
            "tempo": {"bpm": 128},
            "metrics": {
                "text_audio_cos": 0.58,
                "text_audio_cos_mert": 0.57,
            },
        },
        {
            "instrument": "Piano",
            "tempo": {"bpm": 128},
            "metrics": {
                "text_audio_cos": 0.60,
                "text_audio_cos_mert": 0.59,
            },
        },
    ]
    _write_jsonl(before, before_rows)
    _write_jsonl(after, after_rows)

    va_summary = artifacts_dir / "va_summary.json"
    va_summary.write_text(
        json.dumps(
            {
                "before": {"accuracy": 0.75, "f1": 0.7},
                "after": {"accuracy": 0.8, "f1": 0.74},
                "delta": {"accuracy": 0.05, "f1": 0.04},
            }
        ),
        encoding="utf-8",
    )

    va_buckets = artifacts_dir / "va_buckets.json"
    va_buckets.write_text(
        json.dumps(
            {
                "axes": ["velocity"],
                "buckets": [
                    {
                        "name": "velocity",
                        "before": {"accuracy": 0.72},
                        "after": {"accuracy": 0.79},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    configs_metrics = tmp_path / "configs" / "metrics"
    configs_metrics.mkdir(parents=True)
    (configs_metrics / "tau.yaml").write_text(
        "global: 0.50\n",
        encoding="utf-8",
    )
    (configs_metrics / "tau_mert.yaml").write_text(
        "global: 0.55\n",
        encoding="utf-8",
    )

    proc = subprocess.run(
        [
            sys.executable,
            str(_script_path()),
            "--before",
            str(before),
            "--after",
            str(after),
            "--crops-jsonl",
            str(clap_crops),
            "--mert-crops-jsonl",
            str(mert_crops),
            "--va-json",
            str(va_summary),
            "--va-buckets-json",
            str(va_buckets),
            "--tempo-edges",
            "95,110,130,150",
            "--dry-run",
        ],
        cwd=tmp_path,
        check=True,
        text=True,
        capture_output=True,
    )

    output = proc.stdout
    assert "Tau Preview" in output
    assert "τ=0.500" in output
    assert "τ=0.550" in output
