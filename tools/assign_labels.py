from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, cast

try:  # pragma: no cover - runtime import convenience
    from .label_rules import LabelRuleEngine
except ImportError:  # pragma: no cover
    import sys

    sys.path.append(str(Path(__file__).resolve().parent))
    from label_rules import LabelRuleEngine  # type: ignore

DEFAULT_SCHEMA_PATH = Path("configs/labels/labels_schema.yaml")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Assign normalized labels to loop metadata.",
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Input file containing loop summaries (JSONL or CSV)",
    )
    parser.add_argument(
        "output",
        type=Path,
        help="Output JSONL file with normalized labels",
    )
    parser.add_argument(
        "--schema",
        type=Path,
        default=DEFAULT_SCHEMA_PATH,
        help="Path to labels_schema.yaml (default: %(default)s)",
    )
    return parser.parse_args()


def _parse_cell(value: str | None) -> Any:
    if value is None:
        return None
    text = value.strip()
    if not text:
        return None
    lowered = text.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if text[0] in '[{"':
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
    try:
        if any(ch in text for ch in [".", "e", "E"]):
            return float(text)
        return int(text)
    except ValueError:
        return text


def _load_from_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if isinstance(payload, dict):
                records.append(cast(Dict[str, Any], payload))
    return records


def _load_from_csv(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            record: Dict[str, Any] = {}
            metrics: Dict[str, Any] = {}
            for key, value in row.items():
                parsed = _parse_cell(value)
                if key.startswith("metrics."):
                    metric_key = key.split(".", 1)[1]
                    metrics[metric_key] = parsed
                else:
                    record[key] = parsed
            if metrics:
                record["metrics"] = metrics
            if "loop_id" in record and "id" not in record:
                record["id"] = record["loop_id"]
            record.setdefault("label", {})
            records.append(record)
    return records


def load_records(path: Path) -> List[Dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return _load_from_csv(path)
    return _load_from_jsonl(path)


def write_records(path: Path, records: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def assign_labels() -> None:
    args = parse_args()
    engine = LabelRuleEngine.from_file(args.schema)
    records = load_records(args.input)

    normalized: List[Dict[str, Any]] = []
    for record in records:
        if "loop_id" in record and "id" not in record:
            record["id"] = record["loop_id"]
        record.setdefault("label", {})
        normalized.append(engine.apply(record))

    write_records(args.output, normalized)


if __name__ == "__main__":
    assign_labels()
