from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple, cast

from jsonschema import Draft7Validator

try:  # pragma: no cover - allow running as a script
    from .label_rules import LabelRuleEngine, LabelSchema, load_label_schema
except ImportError:  # pragma: no cover
    import sys

    sys.path.append(str(Path(__file__).resolve().parent))
    from label_rules import (  # type: ignore
        LabelRuleEngine,
        LabelSchema,
        load_label_schema,
    )

DEFAULT_CONTRACT_PATH = Path("configs/contracts/loop_summary.v2025_10.json")
DEFAULT_SCHEMA_PATH = Path("configs/labels/labels_schema.yaml")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate loop summary records against the data contract.",
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Input JSONL file to validate",
    )
    parser.add_argument(
        "--contract",
        type=Path,
        default=DEFAULT_CONTRACT_PATH,
        help="Path to loop summary JSON Schema (default: %(default)s)",
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=DEFAULT_SCHEMA_PATH,
        help="Path to labels_schema.yaml (default: %(default)s)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        help="Optional path to write validated records (JSONL)",
    )
    parser.add_argument(
        "--fix",
        action="store_true",
        help="Apply automatic normalization before validating",
    )
    return parser.parse_args()


def load_records(path: Path) -> List[Dict[str, Any]]:
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


def write_records(path: Path, records: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def validate_records(
    records: List[Dict[str, Any]],
    schema: LabelSchema,
    contract_schema: Dict[str, Any],
    engine: LabelRuleEngine,
    apply_fix: bool,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    validator: Draft7Validator = Draft7Validator(contract_schema)
    validated: List[Dict[str, Any]] = []
    failures: List[str] = []

    for idx, record in enumerate(records):
        candidate = engine.apply(record) if apply_fix else record
        error_list = list(validator.iter_errors(candidate))  # type: ignore[misc]
        json_errors = sorted(error_list, key=lambda error: error.path)
        vocab_errors = _check_label_vocab(candidate, schema)
        if json_errors or vocab_errors:
            failure_lines = [
                "Record {idx}: {msg} (path: {path})".format(
                    idx=idx,
                    msg=error.message,
                    path="/".join(str(p) for p in error.path),
                )
                for error in json_errors
            ]
            failure_lines.extend(f"Record {idx}: {msg}" for msg in vocab_errors)
            failures.extend(failure_lines)
        else:
            validated.append(candidate)
    return validated, failures


def _check_label_vocab(
    record: Dict[str, Any],
    schema: LabelSchema,
) -> List[str]:
    label_obj = record.get("label")
    if not isinstance(label_obj, dict):
        return ["missing label object"]
    label = cast(Dict[str, Any], label_obj)

    errors: List[str] = []
    emotion = label.get("emotion")
    if emotion is not None and emotion not in schema.emotions:
        errors.append(f"invalid emotion '{emotion}'")

    genre = label.get("genre")
    if genre is not None and genre not in schema.genres:
        errors.append(f"invalid genre '{genre}'")

    techniques = label.get("technique")
    if isinstance(techniques, list):
        for technique in cast(List[Any], techniques):
            if technique not in schema.techniques:
                errors.append(f"invalid technique '{technique}'")
    elif techniques not in (None, []):
        errors.append("technique must be a list")

    license_origin = label.get("license_origin")
    if license_origin not in schema.license_origins:
        errors.append(f"invalid license_origin '{license_origin}'")

    return errors


def main() -> None:
    args = parse_args()
    records = load_records(args.input)
    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    schema = load_label_schema(args.labels)
    engine = LabelRuleEngine(schema)

    validated, failures = validate_records(
        records,
        schema,
        contract,
        engine,
        args.fix,
    )

    if failures:
        for failure in failures:
            print(failure)
        print(f"Validation failed for {len(failures)} issue(s)")
    else:
        print(f"Validated {len(validated)} records successfully")

    if args.out and validated:
        write_records(args.out, validated)

    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
