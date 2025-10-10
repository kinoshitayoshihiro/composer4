import importlib.util
from typing import Any, Dict, Mapping, Optional

import numpy as np
import pandas as pd
import pytest

from scripts.lamda_stage2_extractor import (
    AUDIO_LOOP_DEFAULTS,
    AudioGuidanceRow,
    AudioAdaptiveAxisCap,
    AudioAdaptiveFusion,
    AudioAdaptiveFusionSource,
    AudioAdaptiveRule,
    AudioAdaptiveState,
    AudioAdaptiveWeights,
    VelocityScoringConfig,
    _build_audio_summary,
    _load_audio_adaptive_weights,
    _apply_audio_adaptive_weights,
    _evaluate_audio_adaptive_rule,
    _scale_velocity_phase_compensation,
    _write_audio_embeddings,
)


def _make_audio_row(**kwargs) -> AudioGuidanceRow:
    return AudioGuidanceRow(**kwargs)


def _build_loop_row(
    loop_id: str,
    audio_row: Optional[AudioGuidanceRow],
    summary: Dict[str, object],
) -> Dict[str, object]:
    row: Dict[str, object] = {"loop_id": loop_id}
    row.update(AUDIO_LOOP_DEFAULTS)
    if audio_row is not None:
        row.update(audio_row.as_loop_row())
    row["text_audio_cos"] = summary.get("text_audio_cos")
    row["caption_en"] = summary.get("caption_en")
    return row


def test_stage2_audio_output_variants(tmp_path):
    clap_embed = np.arange(4, dtype=np.float32) * 0.1
    text_embed = np.linspace(-1.0, 1.0, 4, dtype=np.float32)

    cases = {
        "loop_a": _make_audio_row(
            text_audio_cos=0.62,
            caption_en="energetic rock with punchy drums",
            audio_embed_clap=clap_embed,
            text_embed=text_embed,
            audio_model="clap",
        ),
        "loop_b": _make_audio_row(
            text_audio_cos=0.41,
            audio_model="mert",
        ),
        "loop_c": _make_audio_row(
            caption_en="gentle ambient texture",
            audio_model="clap",
        ),
        "loop_d": None,
    }

    loop_rows = []
    embedding_rows = []
    summaries: Dict[str, Dict[str, object]] = {}

    for loop_id, audio_row in cases.items():
        summary, embedding_entry = _build_audio_summary(loop_id, audio_row)
        loop_rows.append(_build_loop_row(loop_id, audio_row, summary))
        summaries[loop_id] = summary
        if embedding_entry is not None:
            embedding_rows.append(embedding_entry)

    loop_df = pd.DataFrame(loop_rows)
    csv_path = tmp_path / "loop_summary.csv"
    loop_df.to_csv(csv_path, index=False)

    loaded = pd.read_csv(csv_path)
    row_a = loaded.loc[loaded["loop_id"] == "loop_a"].iloc[0]
    assert pytest.approx(row_a["text_audio_cos"], 1e-6) == 0.62
    assert row_a["caption_en"] == "energetic rock with punchy drums"

    row_b = loaded.loc[loaded["loop_id"] == "loop_b"].iloc[0]
    assert pytest.approx(row_b["text_audio_cos"], 1e-6) == 0.41
    assert pd.isna(row_b["caption_en"])

    row_c = loaded.loc[loaded["loop_id"] == "loop_c"].iloc[0]
    assert pd.isna(row_c["text_audio_cos"])
    assert row_c["caption_en"] == "gentle ambient texture"

    row_d = loaded.loc[loaded["loop_id"] == "loop_d"].iloc[0]
    assert pd.isna(row_d["text_audio_cos"])
    assert pd.isna(row_d["caption_en"])

    parquet_path = tmp_path / "audio_embeddings.parquet"
    _write_audio_embeddings(embedding_rows, parquet_path)

    if importlib.util.find_spec("pyarrow") is None:
        assert not parquet_path.exists()
        return

    import pyarrow.parquet as pq  # type: ignore

    table = pq.read_table(parquet_path)
    assert table.num_rows == len(embedding_rows)
    records = table.to_pydict()
    assert "loop_a" in records["loop_id"]
    first_index = records["loop_id"].index("loop_a")
    clap_values = records["audio_embed_clap"][first_index]
    assert clap_values == pytest.approx(clap_embed.tolist())
    text_values = records["text_embed"][first_index]
    assert text_values == pytest.approx(text_embed.tolist())

    assert summaries["loop_a"]["model"] == "clap"
    assert summaries["loop_b"]["text_audio_cos"] == pytest.approx(0.41)
    assert summaries["loop_c"]["caption_en"] == "gentle ambient texture"
    assert summaries["loop_d"] == {}


def test_apply_audio_adaptive_weights_with_rule():
    base = {"timing": 1.0, "velocity": 1.0}
    rule = AudioAdaptiveRule(
        operator=">=",
        threshold=0.5,
        multipliers={"timing": 1.2, "velocity": 0.8},
        name="rule_high",
    )
    adaptive = AudioAdaptiveWeights(
        pivot_path=("metrics", "text_audio_cos"),
        rules=(rule,),
    )
    context = {"metrics": {"text_audio_cos": 0.55}}

    weights, applied, pivot, state = _apply_audio_adaptive_weights(
        base,
        adaptive,
        context,
    )

    assert abs(weights["timing"] - 1.2) < 1e-6
    assert abs(weights["velocity"] - 0.8) < 1e-6
    assert applied is rule
    assert pivot is not None and abs(pivot - 0.55) < 1e-6
    assert isinstance(state, AudioAdaptiveState)
    assert state.last_rule_name == "rule_high"


def test_apply_audio_adaptive_weights_no_match():
    base = {"timing": 1.0, "velocity": 0.9}
    rule = AudioAdaptiveRule(
        operator=">",
        threshold=0.8,
        multipliers={"timing": 1.5},
        name="rule_upper",
    )
    adaptive = AudioAdaptiveWeights(
        pivot_path=("audio", "text_audio_cos"),
        rules=(rule,),
    )
    context = {"audio": {"text_audio_cos": 0.6}}

    weights, applied, pivot, state = _apply_audio_adaptive_weights(
        base,
        adaptive,
        context,
    )

    assert weights == base
    assert applied is None
    assert pivot is not None and abs(pivot - 0.6) < 1e-6
    assert isinstance(state, AudioAdaptiveState)
    assert state.last_rule_name is None


def test_apply_audio_adaptive_weights_with_fusion_sources():
    base = {"timing": 1.0}
    rule = AudioAdaptiveRule(
        operator=">=",
        threshold=0.6,
        multipliers={"timing": 1.1},
        name="fusion_rule",
    )
    fusion = AudioAdaptiveFusion(
        sources=(
            AudioAdaptiveFusionSource(
                path=("audio", "text_audio_cos"),
                weight=0.7,
            ),
            AudioAdaptiveFusionSource(
                path=("metrics", "text_audio_cos_mert"),
                weight=0.3,
            ),
        ),
        clamp_min=0.0,
        clamp_max=1.0,
    )
    adaptive = AudioAdaptiveWeights(
        pivot_path=("metrics", "text_audio_cos"),
        rules=(rule,),
        fusion=fusion,
    )
    context = {
        "audio": {"text_audio_cos": 0.5},
        "metrics": {"text_audio_cos_mert": 0.9},
    }

    weights, applied, pivot, state = _apply_audio_adaptive_weights(
        base,
        adaptive,
        context,
    )

    expected_pivot = (0.5 * 0.7) + (0.9 * 0.3)
    assert pytest.approx(pivot, 1e-6) == expected_pivot
    assert applied is rule
    assert pytest.approx(weights["timing"], 1e-6) == 1.1
    assert state is not None and state.last_rule_name == "fusion_rule"


def test_audio_adaptive_caps_and_normalization():
    base = {"timing": 2.0, "velocity": 1.5}
    rule = AudioAdaptiveRule(
        operator=">=",
        threshold=0.4,
        multipliers={"timing": 2.0, "velocity": 0.5},
        name="scale_rule",
    )
    adaptive = AudioAdaptiveWeights(
        pivot_path=("metrics", "pivot"),
        rules=(rule,),
        min_scale=0.7,
        max_scale=1.4,
        axis_caps={
            "timing": AudioAdaptiveAxisCap(min_scale=0.9, max_scale=1.1),
        },
        normalize_sum=True,
    )
    context = {"metrics": {"pivot": 0.5}}

    weights, applied, _, _ = _apply_audio_adaptive_weights(
        base,
        adaptive,
        context,
    )

    # weights are clamped against per-axis and global limits before
    # normalization, and the final sum matches the base sum.
    assert 2.0 * 0.9 <= weights["timing"] <= 2.0 * 1.1
    assert 1.5 * 0.7 <= weights["velocity"] <= 1.5 * 1.4
    total = sum(weights.values())
    assert abs(total - sum(base.values())) < 1e-6
    assert applied is rule


def test_audio_adaptive_hysteresis_and_cooldown():
    base = {"timing": 1.0}
    rule_high = AudioAdaptiveRule(
        operator=">=",
        threshold=0.7,
        multipliers={"timing": 1.2},
        name="high",
    )
    rule_low = AudioAdaptiveRule(
        operator="<",
        threshold=0.4,
        multipliers={"timing": 0.8},
        name="low",
    )
    adaptive = AudioAdaptiveWeights(
        pivot_path=("metrics", "pivot"),
        rules=(rule_high, rule_low),
        hysteresis_margin=0.1,
        cooldown_loops=2,
    )
    state = AudioAdaptiveState()

    weights, applied, pivot, state = _apply_audio_adaptive_weights(
        base,
        adaptive,
        {"metrics": {"pivot": 0.72}},
        state=state,
    )
    assert applied is rule_high
    assert state is not None
    assert state.cooldown_remaining == 2

    weights, applied, pivot, state = _apply_audio_adaptive_weights(
        weights,
        adaptive,
        {"metrics": {"pivot": 0.68}},
        state=state,
    )
    assert applied is rule_high
    assert state is not None
    assert state.cooldown_remaining == 1

    weights, applied, pivot, state = _apply_audio_adaptive_weights(
        weights,
        adaptive,
        {"metrics": {"pivot": 0.3}},
        state=state,
    )
    assert applied is rule_low
    assert state is not None
    assert state.last_rule_name == "low"
    assert state.cooldown_remaining == 2

    _ = pivot


def test_evaluate_rule_and_apply_with_precomputed():
    base = {"timing": 1.0, "structure": 1.0}
    rule = AudioAdaptiveRule(
        operator=">=",
        threshold=0.6,
        multipliers={"timing": 1.1},
        name="preview",
    )
    adaptive = AudioAdaptiveWeights(
        pivot_path=("metrics", "score"),
        rules=(rule,),
        hysteresis_margin=0.02,
        cooldown_loops=1,
    )
    context: Mapping[str, Any] = {"audio": {}, "metrics": {"score": 0.7}}
    state = AudioAdaptiveState()
    decided_rule, pivot, state = _evaluate_audio_adaptive_rule(
        adaptive,
        context,
        state,
    )
    assert decided_rule is rule
    assert pivot is not None
    assert pivot == pytest.approx(0.7)
    apply_context: Dict[str, Any] = dict(context)
    apply_context["axes_raw"] = {"timing": 0.9, "structure": 0.8}
    weights, applied, pivot_out, _ = _apply_audio_adaptive_weights(
        base,
        adaptive,
        apply_context,
        state=state,
        evaluated=(decided_rule, pivot),
    )
    assert applied is rule
    assert pivot_out is not None
    assert pivot_out == pytest.approx(0.7)
    assert weights["timing"] == pytest.approx(1.1)
    assert weights["structure"] == pytest.approx(1.0)


def test_scale_velocity_phase_compensation_scales_maps():
    cfg = VelocityScoringConfig(
        nbins=4,
        targets={"global": np.full(4, 0.25, dtype=np.float64)},
        weights={"global": 1.0},
        metric_weights={"js": 0.5, "cos": 0.5},
        phase_compensation=True,
        phase_adjust_db={"global": 1.0, "downbeat": 2.0},
        tempo_phase_adjust=(
            (120.0, {"downbeat": 1.5}),
            (None, {"offbeat": -1.0}),
        ),
    )
    scaled = _scale_velocity_phase_compensation(cfg, 0.5)
    assert scaled is not cfg
    assert scaled is not None
    assert scaled.phase_adjust_db["downbeat"] == pytest.approx(1.0)
    assert scaled.phase_adjust_db["global"] == pytest.approx(0.5)
    assert scaled.tempo_phase_adjust[0][1]["downbeat"] == pytest.approx(0.75)
    assert scaled.tempo_phase_adjust[1][1]["offbeat"] == pytest.approx(-0.5)
    unchanged = _scale_velocity_phase_compensation(cfg, None)
    assert unchanged is cfg


def test_load_audio_adaptive_weights_from_config():
    score_cfg: Dict[str, Any] = {
        "audio_adaptive_weights": {
            "pivot": "audio.text_audio_cos",
            "rules": [
                {
                    "if": ">= 0.6",
                    "axis_multipliers": {
                        "timing": 1.3,
                        "structure": 0.9,
                    },
                }
            ],
        }
    }

    adaptive = _load_audio_adaptive_weights(score_cfg)

    assert adaptive is not None
    assert adaptive.pivot_path == ("audio", "text_audio_cos")
    assert len(adaptive.rules) == 1
    first_rule = adaptive.rules[0]
    assert first_rule.operator == ">="
    assert abs(first_rule.threshold - 0.6) < 1e-6
    assert first_rule.multipliers == {"timing": 1.3, "structure": 0.9}
    assert first_rule.name == "rule_0"
    assert adaptive.fusion is None
    assert adaptive.axis_caps == {}


def test_load_audio_adaptive_weights_with_extras():
    score_cfg: Dict[str, Any] = {
        "audio_adaptive_weights": {
            "pivot": "metrics.text_audio_cos",
            "fusion": {
                "sources": [
                    {"path": "audio.text_audio_cos", "weight": 0.6},
                    {"path": "metrics.text_audio_cos_mert", "weight": 0.4},
                ],
                "clamp_min": 0.0,
                "clamp_max": 1.0,
            },
            "caps": {
                "min_scale": 0.8,
                "max_scale": 1.3,
                "axes": {
                    "timing": {"min_scale": 0.9, "max_scale": 1.2},
                },
            },
            "normalize": {
                "enabled": True,
                "target_sum": 6.0,
            },
            "hysteresis_margin": 0.05,
            "cooldown_loops": 3,
            "rules": [
                {
                    "name": "hi",
                    "if": ">= 0.65",
                    "multipliers": {"timing": 1.2},
                    "extras": {"phase_comp_factor": 0.8},
                },
            ],
        }
    }

    adaptive = _load_audio_adaptive_weights(score_cfg)

    assert adaptive is not None
    assert abs((adaptive.min_scale or 0.0) - 0.8) < 1e-6
    assert abs((adaptive.max_scale or 0.0) - 1.3) < 1e-6
    assert "timing" in adaptive.axis_caps
    assert abs((adaptive.axis_caps["timing"].max_scale or 0.0) - 1.2) < 1e-6
    assert adaptive.normalize_sum is True
    assert abs((adaptive.normalize_target or 0.0) - 6.0) < 1e-6
    assert abs(adaptive.hysteresis_margin - 0.05) < 1e-6
    assert adaptive.cooldown_loops == 3
    assert adaptive.fusion is not None
    assert len(adaptive.fusion.sources) == 2
    assert adaptive.rules[0].name == "hi"
    assert adaptive.rules[0].extras == {"phase_comp_factor": 0.8}
