#!/usr/bin/env python3
"""Generate CREPE-based vocal F0 parquet with bar-aware metadata."""

from __future__ import annotations

import argparse
import getpass
import socket
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def write_parquet_with_meta(df: pd.DataFrame, path: Path, meta: dict[str, str]) -> None:
    """Persist dataframe while preserving/merging schema metadata."""

    table = pa.Table.from_pandas(df, preserve_index=False)
    metadata = dict(table.schema.metadata or {})
    metadata.update({str(k).encode(): str(v).encode() for k, v in meta.items()})
    table = table.replace_schema_metadata(metadata)
    pq.write_table(table, str(path))


def hz_to_midi(freq_hz: np.ndarray) -> np.ndarray:
    safe = np.maximum(freq_hz, 1e-6)
    return 69.0 + 12.0 * np.log2(safe / 440.0)


def median_filter(values: np.ndarray, kernel: int = 5) -> np.ndarray:
    try:
        from scipy.signal import medfilt
    except Exception:
        return values

    kernel = max(3, (kernel // 2) * 2 + 1)
    return medfilt(values, kernel_size=kernel)


def extract_f0_crepe(audio_path: Path, hop_ms: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    try:
        import crepe
        import soundfile as sf
    except Exception as exc:  # pragma: no cover - handled via runtime guard
        raise SystemExit("Install dependencies: pip install crepe soundfile") from exc

    audio, sample_rate = sf.read(str(audio_path), always_2d=False)
    if getattr(audio, "ndim", 1) == 2:
        audio = audio.mean(axis=1)

    times, f0_hz, confidence, _ = crepe.predict(
        audio, sample_rate, step_size=int(hop_ms), viterbi=True
    )
    return np.asarray(times), np.asarray(f0_hz), np.asarray(confidence)


def attach_bar_context(
    times_s: np.ndarray, bars_path: str | None
) -> tuple[np.ndarray | None, np.ndarray | None]:
    if not bars_path:
        return None, None

    try:
        bars = pd.read_parquet(bars_path)
    except Exception as exc:
        print(f"[WARN] Failed to load bars parquet {bars_path}: {exc}")
        return None, None

    required = {"bar_index", "start_sec", "end_sec", "tempo_bpm", "start_beat"}
    if not required.issubset(bars.columns):
        missing = required - set(bars.columns)
        print(f"[WARN] bars parquet missing columns {missing}; skipping bar context")
        return None, None

    intervals = pd.IntervalIndex.from_arrays(
        bars["start_sec"].to_numpy(), bars["end_sec"].to_numpy(), closed="left"
    )
    frame_idx = intervals.get_indexer(times_s)
    bar_index = np.full(len(times_s), -1, dtype=int)
    time_ql = np.full(len(times_s), np.nan, dtype=float)

    valid = frame_idx >= 0
    if not np.any(valid):
        return bar_index, time_ql

    bars_valid = bars.iloc[frame_idx[valid]]
    bar_index[valid] = bars_valid["bar_index"].to_numpy(dtype=int, copy=False)
    tempo = bars_valid["tempo_bpm"].to_numpy(dtype=float, copy=False)
    rel = times_s[valid] - bars_valid["start_sec"].to_numpy(dtype=float, copy=False)
    beats = bars_valid["start_beat"].to_numpy(dtype=float, copy=False) + (tempo / 60.0) * rel
    time_ql[valid] = beats
    return bar_index, time_ql


def build_dataframe(
    times_s: np.ndarray,
    f0_hz: np.ndarray,
    confidence: np.ndarray,
    bars_path: str | None,
) -> pd.DataFrame:
    voiced = confidence > 0.5
    f0_midi = hz_to_midi(np.where(voiced, f0_hz, np.nan))
    filtered = median_filter(np.where(voiced, f0_hz, np.nan_to_num(f0_hz, nan=0.0)))

    if len(times_s) > 1:
        slope = np.gradient(filtered, np.median(np.diff(times_s)))
    else:
        slope = np.zeros_like(filtered)

    register = np.full(len(f0_hz), "mid", dtype=object)
    register[f0_hz >= 440.0] = "high"
    register[f0_hz < 196.0] = "low"

    bar_index, time_ql = attach_bar_context(times_s, bars_path)

    data: dict[str, np.ndarray] = {
        "time_s": times_s,
        "f0_hz": f0_hz,
        "f0_midi": f0_midi,
        "voicing_prob": confidence,
        "f0_smooth_hz": filtered,
        "f0_slope_hz_per_s": slope,
        "vibrato_rate_hz": np.zeros_like(f0_hz),
        "vibrato_depth_cents": np.zeros_like(f0_hz),
        "register_band": register,
    }

    if time_ql is not None:
        data["time_ql"] = time_ql
    if bar_index is not None:
        data["bar_index"] = bar_index

    return pd.DataFrame(data)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CREPE vocal F0 extractor")
    parser.add_argument("--audio", required=True, help="Input vocal stem (wav)")
    parser.add_argument("--out", required=True, help="Output parquet path")
    parser.add_argument(
        "--bars", default=None, help="bars_with_slots.parquet for bar index mapping"
    )
    parser.add_argument("--hop_ms", type=float, default=10.0, help="CREPE hop size in milliseconds")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    times_s, f0_hz, confidence = extract_f0_crepe(Path(args.audio), args.hop_ms)
    df = build_dataframe(times_s, f0_hz, confidence, args.bars)
    meta = {
        "generator": "CREPE",
        "hop_ms": args.hop_ms,
        "host": socket.gethostname(),
        "user": getpass.getuser(),
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    write_parquet_with_meta(df, Path(args.out), meta)
    print(f"Wrote {args.out} rows: {len(df)}")


if __name__ == "__main__":
    main()
