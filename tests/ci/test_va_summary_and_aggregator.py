# -*- coding: utf-8 -*-
"""Integration smoke test for VA summary and dashboard aggregator."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any, Dict, List


def _import_module(repo_root: Path, relative: str, name: str):
    module_path = repo_root / relative
    spec = importlib.util.spec_from_file_location(name, module_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


def test_va_pipeline(tmp_path: Path) -> None:
    after_rows: List[Dict[str, Any]] = [
        {"metrics": {"emopia": {"valence": 0.60, "arousal": 0.40}}},
        {
            "metrics": {
                "emopia": {
                    "valence_seq": [0.20, 0.70, 0.80],
                    "arousal_seq": [0.10, 0.30, 0.90],
                }
            }
        },
        {"metrics": {"emopia": {"valence": 0.50}}},
    ]
    after_path = tmp_path / "after.jsonl"
    after_path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in after_rows),
        encoding="utf-8",
    )

    repo_root = Path(__file__).resolve().parents[2]
    va = _import_module(repo_root, "scripts/ci/va_summary.py", "va_summary")

    entries = list(va.read_jsonl(str(after_path)))
    assert len(entries) == 3

    summary = va.summarize(entries)
    assert summary["n_valence"] >= 2
    assert summary["n_arousal"] >= 1

    out_path = tmp_path / "va_summary.json"
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False)

    aggregator = _import_module(
        repo_root,
        "scripts/ci/pr_comment_aggregator.py",
        "pr_comment_aggregator",
    )
    section = aggregator.build_va_section(str(out_path))

    assert section is not None
    assert "EMOPIA Valence/Arousal" in section
    assert "valence.p50" in section
    assert "arousal.p50" in section
