from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, cast

import yaml


@dataclass(frozen=True)
class LabelSchema:
    version: str
    emotions: List[str]
    genres: List[str]
    techniques: Set[str]
    license_origins: Set[str]
    default_emotion: Optional[str]
    default_genre: Optional[str]


class LabelRuleEngine:
    """Utility that normalizes label metadata according to project schema."""

    def __init__(self, schema: LabelSchema) -> None:
        self.schema = schema

    @classmethod
    def from_file(cls, path: Path) -> "LabelRuleEngine":
        schema = load_label_schema(path)
        return cls(schema)

    def apply(self, record: Dict[str, Any]) -> Dict[str, Any]:
        label_any = record.setdefault("label", {})
        if not isinstance(label_any, dict):
            label_any = {}
            record["label"] = label_any
        label = cast(Dict[str, Any], label_any)

        metrics_any = record.get("metrics")
        metrics_dict = cast(Dict[str, Any], metrics_any) if isinstance(metrics_any, dict) else {}

        label["emotion"] = self._normalize_scalar(
            label.get("emotion"),
            self.schema.emotions,
            self.schema.default_emotion,
        )
        label["genre"] = self._normalize_scalar(
            label.get("genre"),
            self.schema.genres,
            self.schema.default_genre,
        )

        label["technique"] = self._normalize_techniques(
            label.get("technique"),
        )

        label["grid_class"] = self._infer_grid_class(
            metrics_dict,
            label.get("grid_class"),
        )
        label["key"] = label.get("key") or None
        label["caption"] = self._normalize_caption(label.get("caption"))
        label["license_origin"] = self._normalize_license(
            label.get("license_origin"),
        )
        return record

    def _normalize_scalar(
        self,
        value: Any,
        allowed: Iterable[str],
        fallback: Optional[str],
    ) -> Optional[str]:
        if isinstance(value, str) and value in allowed:
            return value
        return fallback

    def _normalize_techniques(self, value: Any) -> List[str]:
        techniques: List[str] = []
        if isinstance(value, list):
            for item in cast(List[Any], value):
                if isinstance(item, str) and item in self.schema.techniques:
                    techniques.append(item)
        techniques = list(dict.fromkeys(techniques))  # preserve order + dedupe
        return techniques

    def _normalize_license(self, value: Any) -> str:
        if isinstance(value, str) and value in self.schema.license_origins:
            return value
        if "research_only" in self.schema.license_origins:
            return "research_only"
        return next(iter(self.schema.license_origins))

    def _normalize_caption(self, value: Any) -> Optional[str]:
        if isinstance(value, str):
            text = value.strip()
            if text:
                return text[:280]
        return None

    def _infer_grid_class(
        self,
        metrics: Dict[str, Any],
        existing: Any,
    ) -> Optional[str]:
        if isinstance(existing, str):
            return existing
        swing_ratio = _coerce_float(metrics.get("swing_ratio"))
        syncopation = _coerce_float(metrics.get("syncopation_rate"))
        if swing_ratio is None:
            return None
        if swing_ratio >= 0.3:
            return "swing"
        if syncopation is not None and syncopation >= 0.35:
            return "shuffle"
        return "straight"


def load_label_schema(path: Path) -> LabelSchema:
    raw_any: Any = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    raw = cast(Dict[str, Any], raw_any) if isinstance(raw_any, dict) else {}
    version = str(raw.get("version", "unknown"))
    schema_root = cast(Dict[str, Any], raw.get("schema", {}))

    emotions = _extract_list(schema_root, "emotion", "classes")
    genres = _extract_list(schema_root, "genre", "classes")
    technique_source = cast(Dict[str, Any], schema_root.get("technique", {}))
    technique_set = set(_gather_techniques(technique_source))
    license_list = _extract_list(schema_root, "license", "origin") or ["research_only"]
    license_origins = set(license_list)

    default_emotion = emotions[0] if emotions else None
    default_genre = genres[0] if genres else None

    return LabelSchema(
        version=version,
        emotions=emotions,
        genres=genres,
        techniques=technique_set,
        license_origins=license_origins,
        default_emotion=default_emotion,
        default_genre=default_genre,
    )


def _extract_list(root: Dict[str, Any], *keys: str) -> List[str]:
    node: Any = root
    for key in keys:
        if not isinstance(node, dict):
            return []
        node = cast(Dict[str, Any], node).get(key)
    if isinstance(node, list):
        return [str(item) for item in cast(List[Any], node)]
    return []


def _gather_techniques(tree: Dict[str, Any]) -> List[str]:
    values: List[str] = []
    for _, value in tree.items():
        if isinstance(value, list):
            values.extend(
                str(item) for item in cast(List[Any], value) if isinstance(item, (str, int, float))
            )
        elif isinstance(value, dict):
            values.extend(_gather_techniques(cast(Dict[str, Any], value)))
        elif isinstance(value, str):
            values.append(value)
    return values


def _coerce_float(value: Any) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
