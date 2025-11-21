import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.generate_guitar_plan_v2 import (
    generate_guitar_plan,
    resolve_continue_settings,
)


def _build_minimal_policy(stage3_path: Path, motif_path: Path) -> dict:
    return {
        "global": {"tempo_bpm": 120},
        "sections": {
            "Chorus": {
                "density": 0.8,
                "guitar": 0.7,
                "guitar_continue": {"target_bars": 4},
            }
        },
        "slots": {"riff_default": 1.0},
        "instruments": {
            "guitar": {
                "min_notes_per_bar": 2,
                "max_notes_per_bar": 4,
                "riff_types": [{"type": "strum", "probability": 1.0}],
                "continue": {
                    "enabled": True,
                    "sections": {"chorus": {"target_bars": 4}},
                    "stage3_conditions": str(stage3_path),
                    "motif_path": str(motif_path),
                    "source_bars": 1,
                    "target_bars": 4,
                    "seed": 99,
                },
            }
        },
    }


def test_continue_replaces_riffs(tmp_path):
    stage3_path = tmp_path / "stage3_conditions.csv"
    stage3_df = pd.DataFrame(
        [
            {
                "loop_id": "loopA",
                "backbeat_strength": 0.9,
                "n_downbeats": 16,
                "arousal": 0.2,
                "swing_pct": 12,
                "valence": 0.1,
            }
        ]
    )
    stage3_df.to_csv(stage3_path, index=False)

    motif_path = tmp_path / "motif.json"
    motif_path.write_text(
        json.dumps(
            {
                "events": [
                    {"time_ql": 0.0, "duration_ql": 0.5, "velocity": 90},
                    {"time_ql": 0.5, "duration_ql": 0.5, "velocity": 85},
                ]
            }
        ),
        encoding="utf-8",
    )

    bars = pd.DataFrame(
        {
            "bar_idx": [0, 1, 2, 3],
            "section_label": ["Chorus"] * 4,
            "riff_slot": [1, 1, 1, 1],
        }
    )
    sections = [{"label": "Chorus", "start_bar": 0, "end_bar": 4}]
    chordmap = [{"time_ql": 0.0, "duration_ql": 16.0, "symbol": "C"}]
    policy = _build_minimal_policy(stage3_path, motif_path)

    np.random.seed(0)
    plan = generate_guitar_plan(
        bars,
        sections,
        chordmap,
        policy,
        continue_overrides={"force_enable": True},
    )

    assert plan["metadata"].get("continue", {}).get("applied") is True
    assert plan["metadata"]["continue"]["sections"] == ["chorus"]
    event_types = {evt.get("event_type") for evt in plan["events"]}
    assert event_types == {"continue_riff"}
    assert len(plan["events"]) > 0


def test_resolve_continue_settings_uses_section_policy():
    guitar_cfg = {"continue": {"enabled": True}}
    policy_sections = {"Verse": {"guitar_continue": {"source_bars": 2, "target_bars": 4}}}
    settings = resolve_continue_settings(guitar_cfg, policy_sections)

    assert settings.enabled is True
    assert "verse" in settings.section_overrides
    assert settings.section_params("verse")["source_bars"] == 2
