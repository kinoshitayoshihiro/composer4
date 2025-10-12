#!/usr/bin/env python3
"""Stage3 → Stage2 quick evaluation pipeline with KPI reporting.

This script wires the full closed loop:

1. (Optional) Generate MIDI samples from Stage3 given prompts
2. Evaluate each sample with the real Stage2 extractor
3. Compute aggregate KPIs and stratified metrics
4. Persist a schema-validated JSON report

It is designed to be interactive and CI friendly.  When a --midi-dir is
provided the generation step is skipped so that previously rendered material
can be re-evaluated quickly.
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ml.generation_logger import GenerationLogger
from ml.stage3_infer import Stage3Tokenizer, decode_to_midi, generate_sequences

try:  # pragma: no cover - optional heavyweight dependencies
    import torch
    from transformers import GPT2LMHeadModel
except Exception:  # pragma: no cover - optional dependency guard
    torch = None  # type: ignore
    GPT2LMHeadModel = None  # type: ignore

Report = Dict[str, Any]
Record = Dict[str, Any]

SCHEMA_PATH_DEFAULT = PROJECT_ROOT / "outputs" / "eval" / "schema" / "quick_eval_v1.json"
STAGE2_SCRIPT = PROJECT_ROOT / "scripts" / "lamda_stage2_extractor.py"
DEFAULT_PROMPTS = PROJECT_ROOT / "configs" / "stage3" / "prompts_eval.yaml"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "eval_stage2"

LOGGER = logging.getLogger("quick_eval_stage2")


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def load_prompts(path: Path) -> List[Dict[str, Any]]:
    import yaml

    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}")

    with path.open() as f:
        data = yaml.safe_load(f) or {}

    prompts = data.get("prompts", [])
    if not isinstance(prompts, list):
        raise ValueError("Expected 'prompts' key to be a list in prompts file")
    return prompts


def load_model_and_tokenizer(
    model_dir: Path,
    tokenizer_path: Path,
    device: str = "cpu",
) -> tuple[Any, Stage3Tokenizer]:
    if torch is None or GPT2LMHeadModel is None:
        raise RuntimeError("torch and transformers are required for generation")

    LOGGER.info("Loading tokenizer from %s", tokenizer_path)
    tokenizer = Stage3Tokenizer(tokenizer_path)

    LOGGER.info("Loading model from %s", model_dir)
    model = GPT2LMHeadModel.from_pretrained(str(model_dir))
    model.to(device)
    model.eval()
    return model, tokenizer


def percentile(values: Iterable[float], q: float) -> Optional[float]:
    vals = list(values)
    if not vals:
        return None
    return float(np.percentile(np.asarray(vals, dtype=float), q))


def mean(values: Iterable[float]) -> Optional[float]:
    vals = list(values)
    if not vals:
        return None
    return float(np.mean(np.asarray(vals, dtype=float)))


def tempo_bin(tempo: Optional[float]) -> str:
    if tempo is None:
        return "unknown"
    tempo = float(tempo)
    if tempo < 80:
        return "slow"
    if tempo < 120:
        return "medium"
    return "fast"


def safe_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def ensure_schema(path: Path) -> None:
    if path.exists():
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    schema = {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "title": "Stage3 quick evaluation report",
        "type": "object",
        "required": ["meta", "overall", "errors", "stratified", "items"],
        "properties": {
            "meta": {
                "type": "object",
                "required": [
                    "created_at",
                    "prompt_file",
                    "model_commit",
                    "tokenizer_hash",
                    "n",
                    "stage2_version",
                ],
                "properties": {
                    "created_at": {"type": "string"},
                    "prompt_file": {"type": "string"},
                    "model_commit": {"type": "string"},
                    "tokenizer_hash": {"type": "string"},
                    "n": {"type": "integer"},
                    "stage2_version": {"type": "string"},
                },
            },
            "overall": {
                "type": "object",
                "required": ["pass_rate", "p50", "p90", "mean", "bar_beat_violation_rate"],
                "properties": {
                    "pass_rate": {"type": "number"},
                    "p50": {"type": "number"},
                    "p90": {"type": "number"},
                    "mean": {"type": "number"},
                    "bar_beat_violation_rate": {"type": "number"},
                },
            },
            "errors": {
                "type": "object",
                "required": ["total", "by_reason"],
                "properties": {
                    "total": {"type": "integer"},
                    "by_reason": {
                        "type": "object",
                        "additionalProperties": {"type": "integer"},
                    },
                },
            },
            "stratified": {
                "type": "object",
                "properties": {
                    "time_sig": {"type": "object"},
                    "tempo_bin": {"type": "object"},
                    "genre": {"type": "object"},
                    "emotion": {"type": "object"},
                    "audio_adaptive": {"type": "object"},
                },
                "additionalProperties": False,
            },
            "items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["gen_id", "status", "score", "passed", "diagnostics"],
                    "properties": {
                        "gen_id": {"type": "string"},
                        "file": {"type": "string"},
                        "status": {"type": "string"},
                        "error_reason": {"type": "string"},
                        "score": {"type": "number"},
                        "passed": {"type": "boolean"},
                        "axes_raw": {"type": "object"},
                        "diagnostics": {"type": "object"},
                    },
                },
            },
        },
    }
    path.write_text(json.dumps(schema, indent=2))


def validate_report_schema(report: Report, schema_path: Path) -> None:
    """Validate report against JSON schema. Raise on failure with diagnostic logging."""
    try:
        import jsonschema
    except ImportError:  # pragma: no cover - optional dep
        LOGGER.warning(
            "jsonschema package not available, schema validation SKIPPED. "
            "Install via: pip install jsonschema"
        )
        return

    if not schema_path.exists():
        LOGGER.error("Schema file not found: %s", schema_path)
        raise FileNotFoundError(f"Schema file missing: {schema_path}")

    schema = json.loads(schema_path.read_text())
    try:
        jsonschema.validate(report, schema)
        LOGGER.info("✅ Report schema validation passed: %s", schema_path.name)
    except jsonschema.ValidationError as exc:
        LOGGER.error("❌ Schema validation FAILED at path: %s", list(exc.path))
        LOGGER.error("Validation error: %s", exc.message)
        LOGGER.error("Failed value: %s", exc.instance)
        raise RuntimeError(
            f"Report does not conform to schema {schema_path.name}: {exc.message}"
        ) from exc


def compute_tokenizer_hash(tokenizer_path: Path) -> str:
    data = json.loads(tokenizer_path.read_text())
    logger = GenerationLogger(auto_commit_hash=True)
    return logger.compute_tokenizer_hash(data)


def build_generation_diagnostics(prompt: Dict[str, Any]) -> Dict[str, Any]:
    tempo = safe_float(prompt.get("tempo"))
    tsig = str(prompt.get("time_signature", prompt.get("time_sig", "4/4")))
    genre = str(prompt.get("genre", "unknown"))
    emotion = str(prompt.get("emotion", "unknown"))
    adaptive_enabled = bool(
        prompt.get("audio_clap") is not None or prompt.get("audio_mert") is not None
    )

    return {
        "bar_beat_violation": False,
        "time_sig": tsig,
        "tempo_bin": tempo_bin(tempo),
        "genre": genre,
        "emotion": emotion,
        "audio": {
            "adaptive_enabled": adaptive_enabled,
            "failsafe_reason": None,
        },
    }


def generate_sequences_to_midi(
    *,
    model_dir: Path,
    tokenizer_path: Path,
    prompts: List[Dict[str, Any]],
    num_samples: int,
    output_dir: Path,
    device: str,
    logger: GenerationLogger,
) -> List[Dict[str, Any]]:
    model, tokenizer = load_model_and_tokenizer(model_dir, tokenizer_path, device)

    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer_hash = compute_tokenizer_hash(tokenizer_path)
    generation_records: List[Dict[str, Any]] = []
    for idx, prompt in enumerate(prompts[:num_samples]):
        try:
            sequences = generate_sequences(
                model=model,
                tokenizer=tokenizer,
                prompts=[prompt],
                num_samples=1,
                device=device,
                enforce_bar_constraint=True,
            )
        except Exception as exc:  # pragma: no cover - defensive
            LOGGER.exception("Generation failed for prompt %d", idx)
            generation_records.append(
                {
                    "prompt": prompt,
                    "midi_path": None,
                    "status": "error",
                    "error_reason": f"gen_failed:{exc}",
                    "diagnostics": build_generation_diagnostics(prompt),
                    "gen_id": None,
                }
            )
            continue

        if not sequences:
            generation_records.append(
                {
                    "prompt": prompt,
                    "midi_path": None,
                    "status": "error",
                    "error_reason": "gen_empty",
                    "diagnostics": build_generation_diagnostics(prompt),
                    "gen_id": None,
                }
            )
            continue

        sequence = sequences[0]
        midi_path = output_dir / f"sample_{idx:04d}.mid"
        decode_to_midi(sequence, tokenizer, midi_path)

        gen_id = logger.log_generation(
            prompt=prompt,
            output_file=str(midi_path),
            model_checkpoint=str(model_dir),
            tokenizer_hash=tokenizer_hash,
            generation_params={"device": device},
        )
        logger.embed_metadata_in_midi(str(midi_path), gen_id)

        generation_records.append(
            {
                "prompt": prompt,
                "midi_path": midi_path,
                "status": "ok",
                "error_reason": None,
                "diagnostics": build_generation_diagnostics(prompt),
                "gen_id": gen_id,
            }
        )

    return generation_records


def evaluate_midi_with_stage2(
    midi_path: Path,
    timeout: int = 60,
) -> tuple[str, Optional[Dict[str, Any]], Optional[str]]:
    if not STAGE2_SCRIPT.exists():
        return "missing_stage2", None, f"Stage2 extractor missing at {STAGE2_SCRIPT}"

    cmd = [
        sys.executable,
        str(STAGE2_SCRIPT),
        "--input",
        str(midi_path),
        "--output",
        str(midi_path.with_suffix(".stage2.json")),
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return "stage2_timeout", None, "Stage2 timeout"
    except Exception as exc:  # pragma: no cover - defensive
        return "stage2_error", None, f"Stage2 invocation error: {exc}"

    if result.returncode != 0:
        return "stage2_failed", None, result.stderr.strip() or "Stage2 failed"

    out_path = midi_path.with_suffix(".stage2.json")
    if not out_path.exists():
        return "stage2_output_missing", None, "Stage2 output missing"

    try:
        data = json.loads(out_path.read_text())
    except json.JSONDecodeError as exc:
        return "stage2_invalid_json", None, f"Invalid Stage2 JSON: {exc}"

    return "ok", data, None


def fallback_evaluate(midi_path: Path) -> Dict[str, Any]:
    try:
        import pretty_midi
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("pretty_midi required for fallback evaluation") from exc

    pm = pretty_midi.PrettyMIDI(str(midi_path))
    num_notes = sum(len(inst.notes) for inst in pm.instruments)
    duration = pm.get_end_time()
    score = 50
    if num_notes > 20:
        score += 10
    if duration > 10:
        score += 10
    if len(pm.instruments) > 0:
        score += 10

    return {
        "total_score": float(score),
        "axes": {
            "notes": num_notes,
            "duration": duration,
        },
        "stage2_version": "fallback",
        "diagnostics": {
            "tempo": pm.estimate_tempo() if pm.get_end_time() else None,
            "time_signature": "4/4",
        },
    }


def build_report(records: List[Record], meta: Dict[str, Any]) -> Report:
    ok_records = [r for r in records if r["status"] == "ok" and r.get("score") is not None]

    MIN_SAMPLE_COUNT = 5  # Minimum samples for reliable stratified KPI

    def overall_kpi(rows: List[Record]) -> Dict[str, Any]:
        scores = [float(r["score"]) for r in rows if r.get("score") is not None]
        passes = [r for r in rows if r.get("passed")]
        violations = [r for r in rows if r.get("diagnostics", {}).get("bar_beat_violation")]

        total = len(rows) or 1
        return {
            "pass_rate": round(len(passes) / total, 6),
            "p50": percentile(scores, 50) or 0.0,
            "p90": percentile(scores, 90) or 0.0,
            "mean": mean(scores) or 0.0,
            "bar_beat_violation_rate": round(len(violations) / total, 6),
        }

    def collect_errors(rows: List[Record]) -> Dict[str, Any]:
        by_reason: Dict[str, int] = defaultdict(int)
        for r in rows:
            if r["status"] == "error":
                by_reason[r.get("error_reason", "unknown")] += 1
        return {"total": sum(by_reason.values()), "by_reason": dict(by_reason)}

    def bucketize(rows: List[Record], path: Sequence[str]) -> Dict[str, List[Record]]:
        buckets: Dict[str, List[Record]] = defaultdict(list)
        for row in rows:
            node: Any = row
            valid = True
            for key in path:
                node = node.get(key) if isinstance(node, dict) else None
                if node is None:
                    valid = False
                    break
            if not valid:
                continue
            buckets[str(node)].append(row)
        return buckets

    def kpi_per_bucket(buckets: Dict[str, List[Record]]) -> Dict[str, Any]:
        result = {}
        for key, rows in buckets.items():
            if not rows:
                continue
            kpi = overall_kpi(rows)
            kpi["n"] = len(rows)
            # Mark low sample count buckets with a warning flag for CI gating
            if len(rows) < MIN_SAMPLE_COUNT:
                kpi["_warning"] = f"low_sample_count (n={len(rows)} < {MIN_SAMPLE_COUNT})"
                LOGGER.warning(
                    "Stratified bucket '%s' has insufficient samples: n=%d (threshold=%d)",
                    key,
                    len(rows),
                    MIN_SAMPLE_COUNT,
                )
            result[key] = kpi
        return result

    report: Report = {
        "meta": meta,
        "overall": overall_kpi(ok_records),
        "errors": collect_errors(records),
        "stratified": {
            "time_sig": kpi_per_bucket(bucketize(ok_records, ["diagnostics", "time_sig"])),
            "tempo_bin": kpi_per_bucket(bucketize(ok_records, ["diagnostics", "tempo_bin"])),
            "genre": kpi_per_bucket(bucketize(ok_records, ["diagnostics", "genre"])),
            "emotion": kpi_per_bucket(bucketize(ok_records, ["diagnostics", "emotion"])),
            "audio_adaptive": kpi_per_bucket(
                bucketize(ok_records, ["diagnostics", "audio", "adaptive_enabled"])
            ),
        },
        "items": records,
    }
    return report


def save_report(report: Report, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")
    json_path = output_dir / f"eval_report_{timestamp}.json"
    json_path.write_text(json.dumps(report, indent=2))
    return json_path


def build_meta(
    *,
    prompt_file: Optional[Path],
    generation_count: int,
    model_commit: Optional[str],
    tokenizer_hash: Optional[str],
    stage2_version: str,
) -> Dict[str, Any]:
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "prompt_file": str(prompt_file) if prompt_file else None,
        "model_commit": model_commit or "unknown",
        "tokenizer_hash": tokenizer_hash or "unknown",
        "n": generation_count,
        "stage2_version": stage2_version,
    }


def evaluate_records(
    generations: List[Dict[str, Any]],
    timeout: int,
    use_fallback: bool,
) -> List[Record]:
    records: List[Record] = []
    for entry in generations:
        midi_path = entry.get("midi_path")
        if midi_path is None:
            records.append(
                {
                    "gen_id": entry.get("gen_id") or "unknown",
                    "file": None,
                    "status": "error",
                    "error_reason": entry.get("error_reason", "gen_failed"),
                    "score": None,
                    "passed": False,
                    "axes_raw": None,
                    "diagnostics": entry.get("diagnostics", {}),
                }
            )
            continue

        status, stage2_data, error = evaluate_midi_with_stage2(Path(midi_path), timeout=timeout)
        if status != "ok" and use_fallback:
            stage2_data = fallback_evaluate(Path(midi_path))
            status = "ok"
            error = None

        if status != "ok" or stage2_data is None:
            records.append(
                {
                    "gen_id": entry.get("gen_id") or "unknown",
                    "file": str(midi_path),
                    "status": "error",
                    "error_reason": error or status,
                    "score": None,
                    "passed": False,
                    "axes_raw": None,
                    "diagnostics": entry.get("diagnostics", {}),
                }
            )
            continue

        score = safe_float(stage2_data.get("total_score"))
        axes_raw = stage2_data.get("axes_raw") or stage2_data.get("axes")
        diagnostics = entry.get("diagnostics", {}).copy()

        stage2_diag = stage2_data.get("diagnostics", {})
        if isinstance(stage2_diag, dict):
            diagnostics.setdefault(
                "time_sig", str(stage2_diag.get("time_signature", diagnostics.get("time_sig")))
            )
            diagnostics.setdefault("tempo_bin", tempo_bin(stage2_diag.get("tempo")))
            diagnostics.setdefault("tempo", stage2_diag.get("tempo"))

        bar_violation = bool(stage2_data.get("bar_violations") or stage2_data.get("bar_violation"))
        diagnostics["bar_beat_violation"] = (
            diagnostics.get("bar_beat_violation", False) or bar_violation
        )

        records.append(
            {
                "gen_id": entry.get("gen_id") or "unknown",
                "file": str(midi_path),
                "status": "ok",
                "error_reason": None,
                "score": score or 0.0,
                "passed": bool(score and score >= 50),
                "axes_raw": axes_raw,
                "diagnostics": diagnostics,
            }
        )

    return records


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage3 → Stage2 quick evaluation")
    parser.add_argument("--model", type=Path, help="Stage3 model checkpoint directory")
    parser.add_argument("--tokenizer", type=Path, help="Stage3 tokenizer JSON path")
    parser.add_argument(
        "--prompts", type=Path, default=DEFAULT_PROMPTS, help="Prompt YAML for generation"
    )
    parser.add_argument(
        "--midi-dir", type=Path, help="Existing MIDI directory to evaluate (skip generation)"
    )
    parser.add_argument(
        "--num-samples", type=int, default=32, help="Number of samples to generate/evaluate"
    )
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--schema", type=Path, default=SCHEMA_PATH_DEFAULT)
    parser.add_argument(
        "--timeout", type=int, default=60, help="Stage2 evaluation timeout (seconds)"
    )
    parser.add_argument(
        "--fallback", action="store_true", help="Fallback to heuristic evaluation if Stage2 fails"
    )
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    setup_logging(args.verbose)

    ensure_schema(args.schema)

    generation_logger = GenerationLogger()
    tokenizer_hash: Optional[str] = None
    model_commit: Optional[str] = None
    generations: List[Dict[str, Any]]

    if args.midi_dir and args.midi_dir.exists():
        LOGGER.info("Using existing MIDI directory: %s", args.midi_dir)
        midi_files = sorted(p for p in args.midi_dir.glob("*.mid"))
        generations = [
            {
                "prompt": {},
                "midi_path": midi_file,
                "status": "ok",
                "error_reason": None,
                "diagnostics": {
                    "time_sig": "unknown",
                    "tempo_bin": "unknown",
                    "genre": "unknown",
                    "emotion": "unknown",
                    "audio": {"adaptive_enabled": False, "failsafe_reason": None},
                    "bar_beat_violation": False,
                },
                "gen_id": Path(midi_file).stem,
            }
            for midi_file in midi_files[: args.num_samples]
        ]
    else:
        if not args.model or not args.tokenizer:
            LOGGER.error("Model and tokenizer paths are required when generation is enabled")
            return 2
        tokenizer_hash = compute_tokenizer_hash(args.tokenizer)
        prompts = load_prompts(args.prompts)
        if not prompts:
            LOGGER.error("No prompts found in %s", args.prompts)
            return 2

        generations = generate_sequences_to_midi(
            model_dir=args.model,
            tokenizer_path=args.tokenizer,
            prompts=prompts,
            num_samples=args.num_samples,
            output_dir=args.output_dir / "midi",
            device=args.device,
            logger=generation_logger,
        )

        if generations:
            first_ok = next((g for g in generations if g.get("gen_id")), None)
            if first_ok:
                meta = generation_logger.get_generation_metadata(first_ok["gen_id"])
                if meta:
                    model_commit = meta.get("model_commit")

    records = evaluate_records(generations, timeout=args.timeout, use_fallback=args.fallback)

    stage2_version = "unknown"
    for rec in records:
        axes = rec.get("axes_raw")
        if isinstance(axes, dict) and "stage2_version" in axes:
            stage2_version = str(axes["stage2_version"])
            break

    meta = build_meta(
        prompt_file=args.prompts if args.prompts else None,
        generation_count=len(generations),
        model_commit=model_commit,
        tokenizer_hash=tokenizer_hash,
        stage2_version=stage2_version,
    )

    report = build_report(records, meta)
    validate_report_schema(report, args.schema)
    json_path = save_report(report, args.output_dir)
    LOGGER.info("Report saved to %s", json_path)

    LOGGER.info(
        "Summary: pass_rate=%.2f%%, p50=%.2f, p90=%.2f",
        report["overall"].get("pass_rate", 0.0) * 100,
        report["overall"].get("p50", 0.0),
        report["overall"].get("p90", 0.0),
    )

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
    print(f"P50:           {kpi.get('p50', 0):.1f}")
    print(f"P90:           {kpi.get('p90', 0):.1f}")
    print(f"Violations:    {kpi.get('bar_violation_rate', 0):.1%}")
    print(f"Gate:          {'✅ PASS' if kpi.get('gate_pass') else '❌ FAIL'}")
    print("=" * 70)


if __name__ == "__main__":
    main()
