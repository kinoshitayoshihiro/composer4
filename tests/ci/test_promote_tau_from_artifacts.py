# -*- coding: utf-8 -*-
"""Integration smoke tests for promote_tau_from_artifacts script."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def _script_path() -> Path:
    root = Path(__file__).resolve().parents[2]
    return root / "scripts" / "ci" / "promote_tau_from_artifacts.py"


def test_promote_tau_creates_config_and_log(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()

    clap_artifact = artifact_dir / "auto_tau.yaml"
    clap_artifact.write_text("global: 0.620000\n", encoding="utf-8")

    mert_artifact = artifact_dir / "auto_tau_mert.yaml"
    mert_artifact.write_text("global: 0.580000\n", encoding="utf-8")

    metrics_dir = tmp_path / "configs" / "metrics"
    metrics_dir.mkdir(parents=True)

    artifacts_dir = tmp_path / "artifacts"
    log_path = artifacts_dir / "tau_promotion.log"

    proc = subprocess.run(
        [
            sys.executable,
            str(_script_path()),
            "--clap-artifacts",
            str(clap_artifact),
            "--mert-artifacts",
            str(mert_artifact),
            "--clap-dest",
            str(metrics_dir / "tau.yaml"),
            "--mert-dest",
            str(metrics_dir / "tau_mert.yaml"),
        ],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )

    assert proc.returncode == 0
    clap_config = metrics_dir / "tau.yaml"
    mert_config = metrics_dir / "tau_mert.yaml"

    assert clap_config.exists()
    assert mert_config.exists()
    assert log_path.exists()

    log_text = log_path.read_text(encoding="utf-8")
    assert "CLAP" in log_text
    assert "MERT" in log_text
    assert log_text.splitlines()[1].endswith(f"{clap_artifact} -> {clap_config}")
    assert log_text.splitlines()[2].endswith(f"{mert_artifact} -> {mert_config}")
