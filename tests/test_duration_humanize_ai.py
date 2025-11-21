"""Unit tests for DurationHumanizeAI emotion scalar ingestion."""

from __future__ import annotations

from otobonAI.duration_humanize_ai import DurationHumanizeAI


def test_emotion_scalars_populate_velocity_and_density() -> None:
    policy = {
        "humanize": {
            "global": {
                "timing_std_ms": 5.0,
                "duration_scale_mean": 1.0,
                "duration_scale_jitter": 0.1,
                "staccato_prob": 0.1,
                "phrase_end_extend": 1.05,
                "max_shift_ms": 20.0,
            }
        }
    }
    plan = {
        "metadata": {
            "emotion_tracking": {
                "per_bar": {
                    "0": {
                        "energy": 0.8,
                        "valence": 0.2,
                    }
                }
            }
        },
        "events": [
            {
                "bar_idx": 0,
                "time_ql": 0.0,
                "duration_ql": 1.0,
                "velocity": 80,
            }
        ],
    }

    duration_ai = DurationHumanizeAI(
        instrument="piano",
        policy=policy,
        tempo_bpm=120.0,
        rhythm_manifest_path=None,
        vocab_instrument="piano",
    )

    duration_ai.annotate_plan(plan)

    humanize_payload = plan["events"][0].get("humanize")
    assert humanize_payload, "DurationHumanizeAI should annotate events"
    assert "velocity_scale" in humanize_payload
    assert "density_scale" in humanize_payload
    emotion_payload = humanize_payload.get("emotion")
    assert emotion_payload and "velocity_scale" in emotion_payload
    assert emotion_payload["velocity_scale"] == humanize_payload["velocity_scale"]
