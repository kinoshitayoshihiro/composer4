from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, cast

import pandas as pd


CONTRACT_VERSION = "2025.10"


@dataclass(frozen=True)
class LoopMeta:
    loop_id: str
    time_signature: str
    ticks_per_beat: Optional[float]
    base_step: Optional[float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=("Transform Stage 2 loop summaries into contract-compliant records."),
    )
    parser.add_argument(
        "labeled_input",
        type=Path,
        help="Path to loop_summary_labeled.jsonl (Stage 2 output)",
    )
    parser.add_argument(
        "canonical_events",
        type=Path,
        help="Path to canonical_events.parquet file",
    )
    parser.add_argument(
        "output",
        type=Path,
        help="Destination path for contract-compliant JSONL",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
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


def write_jsonl(path: Path, records: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def parse_time_signature(value: Optional[str]) -> Optional[tuple[int, int]]:
    if not value:
        return None
    try:
        numerator_str, denominator_str = value.split("/", 1)
        numerator = int(numerator_str)
        denominator = int(denominator_str)
        if numerator <= 0 or denominator <= 0:
            return None
        return numerator, denominator
    except (ValueError, AttributeError):
        return None


def compute_tempo_bpm(record: Dict[str, Any]) -> Optional[float]:
    tempo_summary = record.get("tempo.summary")
    if isinstance(tempo_summary, dict):
        tempo_summary_dict = cast(Dict[str, Any], tempo_summary)
        bpm = tempo_summary_dict.get("initial_bpm") or tempo_summary_dict.get("initial")
        if isinstance(bpm, (int, float)) and bpm > 0:
            return float(bpm)
    bpm_value = record.get("bpm")
    if isinstance(bpm_value, (int, float)) and bpm_value > 0:
        return float(bpm_value)
    tempo_events = record.get("tempo.events")
    if isinstance(tempo_events, list) and tempo_events:
        events_seq = cast(Sequence[Any], tempo_events)
        first = events_seq[0]
        if isinstance(first, (int, float)) and first > 0:
            return 60_000_000.0 / float(first)
    return None


def safe_div(numerator: float, denominator: float) -> Optional[float]:
    if denominator == 0:
        return None
    return numerator / denominator


def summarize_backbeat(
    events: Optional[pd.DataFrame],
    meta: LoopMeta,
) -> Optional[float]:
    if events is None or events.empty:
        return 0.0

    snare_events = events[events["instrument_role"] == "snare"]
    if snare_events.empty:
        return 0.0

    time_sig = parse_time_signature(meta.time_signature)
    if not time_sig:
        return 0.0
    numerator, _ = time_sig

    base_step = meta.base_step or 0.0
    ticks_per_beat = meta.ticks_per_beat or 0.0
    if base_step <= 0 or ticks_per_beat <= 0:
        return 0.0

    subdivisions_per_beat = max(1, int(round(ticks_per_beat / base_step)))
    steps_per_bar = subdivisions_per_beat * max(1, numerator)

    if steps_per_bar <= 0:
        return 0.0

    # Determine canonical backbeat indices (0-based positions within a bar)
    backbeat_indices: set[int] = set()
    if numerator >= 4:
        backbeat_indices.update({1, 3})
        if numerator > 4:
            # Include every other beat after the second for longer meters
            # (e.g., 5/4, 7/4)
            for idx in range(5, numerator, 2):
                backbeat_indices.add(idx)
    elif numerator == 3:
        backbeat_indices.add(1)
    elif numerator == 2:
        backbeat_indices.add(1)

    if not backbeat_indices:
        return 0.0

    # Compute quantized step within the bar using grid_onset values
    grid_steps = cast(
        pd.Series,
        (snare_events["grid_onset"] / base_step).round().astype(int),
    )
    step_in_bar = grid_steps % steps_per_bar

    beat_indices = cast(
        pd.Series,
        (step_in_bar // subdivisions_per_beat).astype(int),
    )
    subdivision_offsets = cast(
        pd.Series,
        (step_in_bar % subdivisions_per_beat).astype(int),
    )

    # Count snare hits that land on the first subdivision of a backbeat beat
    is_backbeat = beat_indices.isin(  # type: ignore[attr-defined]
        list(backbeat_indices),
    )
    hits_on_backbeat = (is_backbeat & (subdivision_offsets == 0)).sum()

    total_snare_hits = len(snare_events)
    if total_snare_hits == 0:
        return 0.0

    strength = hits_on_backbeat / total_snare_hits
    return float(max(0.0, min(1.0, strength)))


def build_contract_record(
    record: Dict[str, Any],
    events: Optional[pd.DataFrame],
) -> Dict[str, Any]:
    loop_id = str(record.get("id") or record.get("loop_id") or "")
    tempo_bpm = compute_tempo_bpm(record)
    ticks_per_beat = record.get("ticks_per_beat")
    duration_ticks = record.get("duration_ticks")

    tempo_bpm = tempo_bpm if tempo_bpm and math.isfinite(tempo_bpm) else None

    loop_length_beats: Optional[float] = None
    duration_seconds: Optional[float] = None

    if isinstance(duration_ticks, (int, float)) and isinstance(
        ticks_per_beat,
        (int, float),
    ):
        if ticks_per_beat > 0:
            loop_length_beats = duration_ticks / ticks_per_beat
            if tempo_bpm:
                seconds = loop_length_beats * 60.0 / tempo_bpm
                duration_seconds = seconds

    metrics = record.get("metrics")
    metrics_obj: Dict[str, Any] = {}
    if isinstance(metrics, dict):
        metrics_obj = dict(cast(Dict[str, Any], metrics))
    swing_ratio = metrics_obj.get("swing_ratio")
    if isinstance(swing_ratio, (int, float)):
        metrics_obj["swing_ratio"] = float(max(0.0, min(2.0, swing_ratio)))

    base_step = metrics_obj.get("base_step")
    base_step_value = float(base_step) if isinstance(base_step, (int, float)) else None
    ticks_value = float(ticks_per_beat) if isinstance(ticks_per_beat, (int, float)) else None

    meta = LoopMeta(
        loop_id=loop_id,
        time_signature=str(record.get("time_signature") or ""),
        ticks_per_beat=ticks_value,
        base_step=base_step_value,
    )
    backbeat_strength = summarize_backbeat(events, meta)
    if backbeat_strength is not None:
        metrics_obj["backbeat_strength"] = backbeat_strength

    label = record.get("label")
    label_obj: Dict[str, Any] = {}
    if isinstance(label, dict):
        label_obj = dict(cast(Dict[str, Any], label))

    # Ensure the label payload conforms to expected shapes
    technique = label_obj.get("technique")
    if isinstance(technique, list):
        technique_items = cast(Sequence[Any], technique)
        label_obj["technique"] = [str(item) for item in technique_items]
    else:
        label_obj["technique"] = [] if technique in (None, "") else [str(technique)]

    label_obj.setdefault("emotion", None)
    label_obj.setdefault("genre", None)
    label_obj.setdefault("key", None)
    label_obj.setdefault("grid_class", None)
    label_obj.setdefault("caption", None)
    label_obj.setdefault("license_origin", "research_only")

    contract_record: Dict[str, Any] = {
        "contract_version": CONTRACT_VERSION,
        "id": loop_id,
        "source": str(record.get("source") or ""),
        "origin_dataset": str(record.get("source") or ""),
        "path": str(record.get("filename") or ""),
        "tempo_bpm": tempo_bpm,
        "time_signature": str(record.get("time_signature") or ""),
        "duration_seconds": duration_seconds,
        "loop_length_beats": loop_length_beats,
        "metrics": metrics_obj,
        "label": label_obj,
        "checksum_md5": str(record.get("file_digest") or ""),
    }

    return {key: value for key, value in contract_record.items() if value is not None}


def main() -> None:
    args = parse_args()

    labeled_records = load_jsonl(args.labeled_input)
    if not labeled_records:
        raise SystemExit("No records found in labeled input")

    events_df = pd.read_parquet(  # type: ignore[call-arg]
        args.canonical_events,
    )
    events_by_loop: Dict[str, pd.DataFrame] = {}
    for loop_id, group in events_df.groupby(  # type: ignore[attr-defined]
        "loop_id",
    ):
        events_by_loop[str(loop_id)] = group.copy()

    transformed: List[Dict[str, Any]] = []
    for record in labeled_records:
        loop_id = str(record.get("id") or record.get("loop_id") or "")
        events = events_by_loop.get(loop_id)
        transformed.append(build_contract_record(record, events))

    write_jsonl(args.output, transformed)
    print(f"Wrote {len(transformed)} contract records to {args.output}")


if __name__ == "__main__":
    main()
