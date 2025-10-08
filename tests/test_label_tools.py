from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from tools.label_rules import LabelRuleEngine, LabelSchema
from tools.validate_labels import validate_records


def _build_schema() -> LabelSchema:
    return LabelSchema(
        version="test",
        emotions=["happy", "sad"],
        genres=["pop", "rock"],
        techniques={"ghost", "pizzicato"},
        license_origins={"research_only", "cc"},
        default_emotion="happy",
        default_genre="pop",
    )


def test_label_rule_engine_applies_defaults(tmp_path: Path) -> None:
    engine = LabelRuleEngine(_build_schema())
    record: Dict[str, Any] = {
        "id": "loop123",
        "source": "test",
        "tempo_bpm": 120.0,
        "time_signature": "4/4",
        "duration_seconds": 8.0,
        "metrics": {"swing_ratio": 0.35, "backbeat_strength": 0.5},
        "label": {},
    }
    engine.apply(record)

    label = record["label"]
    assert label["emotion"] == "happy"
    assert label["genre"] == "pop"
    assert label["grid_class"] == "swing"
    assert label["license_origin"] == "research_only"


def test_validate_records_with_fix(tmp_path: Path) -> None:
    schema_path = Path("configs/contracts/loop_summary.v2025_10.json")
    contract_schema = json.loads(schema_path.read_text(encoding="utf-8"))
    schema = _build_schema()
    engine = LabelRuleEngine(schema)

    record: Dict[str, Any] = {
        "id": "loop4567",
        "source": "test",
        "tempo_bpm": 100.0,
        "time_signature": "4/4",
        "duration_seconds": 4.0,
        "metrics": {
            "swing_ratio": 0.1,
            "backbeat_strength": 0.6,
        },
        "label": {
            "emotion": None,
            "genre": None,
            "technique": ["ghost", "unknown"],
            "key": None,
            "grid_class": None,
            "caption": None,
            "license_origin": "invalid",
        },
    }

    validated, failures = validate_records(
        [record],
        schema,
        contract_schema,
        engine,
        apply_fix=True,
    )
    assert not failures
    assert validated
    fixed_label = validated[0]["label"]
    assert fixed_label["emotion"] == "happy"
    assert fixed_label["technique"] == ["ghost"]
    assert fixed_label["license_origin"] == "research_only"


def test_validate_records_reports_errors(tmp_path: Path) -> None:
    schema_path = Path("configs/contracts/loop_summary.v2025_10.json")
    contract_schema = json.loads(schema_path.read_text(encoding="utf-8"))
    schema = _build_schema()
    engine = LabelRuleEngine(schema)

    record: Dict[str, Any] = {
        "id": "loop7890",
        "source": "test",
        "tempo_bpm": 60.0,
        "time_signature": "4/4",
        "duration_seconds": 4.0,
        "metrics": {
            "swing_ratio": 0.1,
            "backbeat_strength": 0.6,
        },
        "label": {
            "emotion": "mystery",
            "genre": "unknown",
            "technique": ["ghost"],
            "key": None,
            "grid_class": None,
            "caption": None,
            "license_origin": "invalid",
        },
    }

    validated, failures = validate_records(
        [record],
        schema,
        contract_schema,
        engine,
        apply_fix=False,
    )
    assert not validated
    assert failures
    assert any("emotion" in message for message in failures)
    assert any("license_origin" in message for message in failures)
