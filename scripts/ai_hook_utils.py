"""Utility helpers for Phase 2.0 AI hook wiring."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd


def load_reference_layers(
    vocal_f0_path: Optional[str],
    oaf_path: Optional[str],
) -> Dict[str, Any]:
    """Load optional CREPE / Onsets-and-Frames reference summaries."""

    summary: Dict[str, Any] = {}

    if vocal_f0_path:
        path = Path(vocal_f0_path)
        if path.exists():
            try:
                df = pd.read_parquet(path)
                summary["crepe"] = {
                    "frames": int(len(df)),
                    "path": str(path),
                }
            except Exception as exc:  # pragma: no cover - telemetry only
                print(f"⚠️  Failed to load CREPE reference {path}: {exc}")

    if oaf_path:
        path = Path(oaf_path)
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                notes = data.get("notes") or data.get("events") or []
                note_count = len(notes) if isinstance(notes, list) else 0
                summary["onsets_and_frames"] = {
                    "notes": int(note_count),
                    "path": str(path),
                }
            except Exception as exc:  # pragma: no cover - telemetry only
                print(f"⚠️  Failed to load Onsets-and-Frames reference {path}: {exc}")

    return summary
