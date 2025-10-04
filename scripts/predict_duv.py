from __future__ import annotations

"""Predict velocity and duration from feature CSV and apply to MIDI.

The command mirrors ``predict_pedal.py`` in spirit but is intentionally
minimal.  Input is a note‑wise CSV with feature columns matching the
statistics saved alongside the checkpoint.  The resulting MIDI will have
predicted velocities and durations applied.
"""

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import pretty_midi as pm
import torch

from utilities.csv_io import coerce_columns
from utilities.duv_infer import (
    CSV_FLOAT32_COLUMNS,
    CSV_INT32_COLUMNS,
    OPTIONAL_COLUMNS,
    OPTIONAL_FLOAT32_COLUMNS,
    OPTIONAL_INT32_COLUMNS,
    REQUIRED_COLUMNS,
    mask_any,
    duv_sequence_predict,
    load_duv_dataframe,
    duv_verbose,
)
from utilities.ml_velocity import MLVelocityModel

from .eval_duv import (  # reuse helpers
    _ensure_int,
    _duration_predict,
    _get_device,
    _load_duration_model,
    _load_stats,
    _parse_quant,
    _NoopDurationModel,
    load_stats_and_normalize,
)


_duv_sequence_predict = duv_sequence_predict
_mask_any = mask_any


def _median_smooth(x: np.ndarray, k: int) -> np.ndarray:
    if k <= 1:
        return x
    if k % 2 == 0:
        raise ValueError("median window must be odd")
    pad = k // 2
    x_pad = np.pad(x, (pad, pad), mode="edge")
    out = np.empty_like(x)
    for i in range(out.size):
        out[i] = np.median(x_pad[i : i + k])
    return out


def _first_group_length(df: pd.DataFrame) -> int:
    if df.empty:
        return 0
    group_cols = [c for c in ("track_id", "bar") if c in df.columns]
    if group_cols:
        for _, group in df.groupby(group_cols, sort=False):
            return int(len(group))
    return int(len(df))


def run(args: argparse.Namespace) -> int:
    stats = _load_stats(args.stats_json, args.ckpt)
    if stats is None:
        raise SystemExit("missing or invalid stats json")

    feat_cols = list(stats[0] or [])

    limit = max(0, int(getattr(args, "limit", 0) or 0))

    df, program_hist = load_duv_dataframe(
        args.csv,
        feature_columns=feat_cols,
        filter_expr=getattr(args, "filter_program", None),
        limit=limit,
        collect_program_hist=getattr(args, "verbose", False),
    )

    if getattr(args, "verbose", False):
        print({"rows": int(len(df)), "limit": (limit if limit > 0 else None)}, file=sys.stderr)
        if program_hist is not None and not program_hist.empty:
            top = program_hist.head(10)
            print({"program_top": top.to_dict()}, file=sys.stderr)
    if "track_id" not in df.columns and "file" in df.columns:
        df["track_id"] = pd.factorize(df["file"])[0].astype("int32")
    float_cols = (
        set(stats[0] or [])
        | (set(CSV_FLOAT32_COLUMNS) - {"velocity"})
        | set(OPTIONAL_FLOAT32_COLUMNS)
    )
    int_cols = set(CSV_INT32_COLUMNS) | (
        {c for c in REQUIRED_COLUMNS if c not in {"velocity", "duration"}}
    ) | set(OPTIONAL_INT32_COLUMNS)
    int_cols.discard("pitch")
    int_cols.discard("program")
    df = coerce_columns(df, float32=float_cols, int32=int_cols)

    print({"rows": int(len(df)), "limit": limit}, file=sys.stderr)

    device = _get_device(args.device)
    vel_model = MLVelocityModel.load(str(args.ckpt))
    loader_type = getattr(vel_model, "_duv_loader", "ts" if str(args.ckpt).endswith((".ts", ".torchscript")) else "ckpt")
    core = getattr(vel_model, "core", vel_model)
    d_model = _ensure_int(getattr(vel_model, "d_model", getattr(core, "d_model", 0)), 0)
    max_len = _ensure_int(getattr(vel_model, "max_len", getattr(core, "max_len", 0)), 0)
    heads = getattr(
        vel_model,
        "heads",
        {
            "vel_reg": bool(getattr(vel_model, "has_vel_head", getattr(core, "head_vel_reg", None))),
            "dur_reg": bool(getattr(vel_model, "has_dur_head", getattr(core, "head_dur_reg", None))),
        },
    )
    extras = {
        "use_bar_beat": bool(getattr(core, "use_bar_beat", False)),
        "section_emb": getattr(core, "section_emb", None) is not None,
        "mood_emb": getattr(core, "mood_emb", None) is not None,
        "vel_bucket_emb": getattr(core, "vel_bucket_emb", None) is not None,
        "dur_bucket_emb": getattr(core, "dur_bucket_emb", None) is not None,
    }
    info = {
        "ckpt": str(args.ckpt),
        "loader": loader_type,
        "d_model": d_model or None,
        "max_len": max_len or None,
        "heads": {k: bool(v) for k, v in heads.items()},
        "extras": extras,
    }
    if getattr(args, "verbose", False):
        print(json.dumps(info), file=sys.stderr)

    vel_model = vel_model.to(device).eval()
    heads = info["heads"]
    need_duration = bool(heads.get("dur_reg") or heads.get("dur_cls"))
    duration_model: _NoopDurationModel | torch.nn.Module = _NoopDurationModel()
    if need_duration:
        duration_model = _load_duration_model(args.ckpt).to(device)
    verbose = duv_verbose(getattr(args, "verbose", False))
    try:
        duv_preds = _duv_sequence_predict(df, vel_model, device, verbose=verbose)
    except Exception as exc:
        raise RuntimeError(f"DUV inference failed (rows={len(df)}): {exc}") from exc

    vel_pred: np.ndarray | None
    vel_mask: np.ndarray | None = None
    if duv_preds is not None and _mask_any(duv_preds["velocity_mask"]):
        vel_mask = duv_preds["velocity_mask"]
        base = df.get("velocity")
        if base is not None:
            vel_pred = base.to_numpy(dtype="float32", copy=False).copy()
        else:
            vel_pred = np.zeros(len(df), dtype=np.float32)
        vel_pred[vel_mask] = duv_preds["velocity"].astype("float32", copy=False)[vel_mask]
    else:
        if getattr(vel_model, "requires_duv_feats", False):
            missing = sorted(REQUIRED_COLUMNS - set(df.columns))
            detail = f"; missing columns: {', '.join(missing)}" if missing else ""
            raise RuntimeError(
                "DUV checkpoint requires phrase-level features (pitch, velocity, duration, position) "
                "for inference and cannot fall back to dense feature tensors"
                f"{detail}."
            )
        preds: list[np.ndarray] = []
        X, _ = load_stats_and_normalize(df, stats, strict=True)
        with torch.no_grad():
            for start in range(0, X.shape[0], args.batch):
                xb = torch.from_numpy(X[start : start + args.batch]).to(device)
                out = vel_model(xb)
                preds.append(out.cpu().numpy())
        if preds:
            vel_pred = np.concatenate(preds, axis=0).astype("float32")
        else:
            vel_pred = np.zeros(len(df), dtype=np.float32)

    if args.vel_smooth > 1:
        smoothed = _median_smooth(vel_pred.copy(), args.vel_smooth)
        if args.smooth_pred_only and _mask_any(vel_mask):
            vel_pred[vel_mask] = smoothed[vel_mask]
        else:
            vel_pred = smoothed
    vel_pred = np.clip(vel_pred, 1, 127)

    dur_pred: np.ndarray | None = None
    if duv_preds is not None and _mask_any(duv_preds["duration_mask"]):
        base = df.get("duration")
        if base is not None:
            dur_pred = base.to_numpy(dtype="float32", copy=False).copy()
        else:
            dur_pred = np.zeros(len(df), dtype=np.float32)
        mask = duv_preds["duration_mask"]
        dur_pred[mask] = duv_preds["duration"].astype("float32", copy=False)[mask]
    elif (
        need_duration
        and not isinstance(duration_model, _NoopDurationModel)
        and "duration" in df.columns
        and "bar" in df.columns
        and "position" in df.columns
        and "pitch" in df.columns
    ):
        dur_pred, _ = _duration_predict(df, duration_model)
    grid = _parse_quant(args.dur_quant, stats[3])
    if grid > 0 and dur_pred is not None:
        dur_pred = np.maximum(grid, np.round(dur_pred / grid) * grid)
    elif grid <= 0:
        print({"dur_quant": "skipped", "grid": float(grid)}, file=sys.stderr)

    if getattr(args, "verbose", False) and vel_pred is not None:
        seq_len = _first_group_length(df)
        preview_vals = vel_pred[:8].astype("float32", copy=False).tolist()
        source = (
            "dense_fallback"
            if duv_preds is None or not _mask_any(duv_preds["velocity_mask"])
            else "duv_sequence"
            )
        print(
            {
                "preview": {
                    "seq_len": int(seq_len) if seq_len is not None else None,
                    "d_model_effective": d_model or None,
                    "source": source,
                    "velocity": [float(v) for v in preview_vals],
                }
            },
            file=sys.stderr,
        )

    pm_obj = pm.PrettyMIDI()
    inst = pm.Instrument(program=0)
    start_col = "start" if "start" in df.columns else "onset"
    for i, row in df.reset_index(drop=True).iterrows():
        start = float(row.get(start_col, 0.0))
        pitch = int(row.get("pitch", 60))
        dur = float(dur_pred[i]) if dur_pred is not None and i < len(dur_pred) else float(row.get("duration", 0.5))
        vel = int(vel_pred[i])
        end = start + max(dur, 0.0)
        inst.notes.append(pm.Note(velocity=vel, pitch=pitch, start=start, end=end))
    inst.notes.sort(key=lambda n: (n.start, n.pitch))
    pm_obj.instruments.append(inst)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pm_obj.write(str(args.out))
    return 0


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - CLI
    p = argparse.ArgumentParser(prog="predict_duv.py")
    p.add_argument("--csv", type=Path, required=True)
    p.add_argument(
        "--ckpt",
        type=Path,
        required=True,
        help="Checkpoint (.ckpt state_dict or .ts TorchScript)",
    )
    p.add_argument("--out", type=Path, required=True, help="Output MIDI path")
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--device", default="cpu")
    p.add_argument("--stats-json", type=Path)
    p.add_argument("--num-workers", dest="num_workers", type=int)
    p.add_argument(
        "--vel-smooth",
        type=int,
        default=1,
        dest="vel_smooth",
        choices=(1, 3, 5),
        help="Velocity median smoothing window; 1 disables, 3/5 apply a median filter",
    )
    p.add_argument(
        "--smooth-pred-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When smoothing velocities, only adjust bins predicted by the model",
    )
    p.add_argument(
        "--dur-quant",
        type=str,
        dest="dur_quant",
        help="Quantise durations to the given fraction (e.g. 1/16); omit to keep predictions",
    )
    p.add_argument(
        "--filter-program",
        dest="filter_program",
        help=(
            "Optional pandas query string applied immediately after loading the CSV (before --limit); "
            "the DataFrame index is reset (e.g. program == 0 and position >= 0)"
        ),
    )
    p.add_argument(
        "--limit",
        type=int,
        default=0,
        help=(
            "Optional maximum number of rows to load; applied during CSV read and after filtering "
            "(0 keeps all rows)"
        ),
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Emit additional DUV diagnostics including optional zero-filled feature usage",
    )
    args = p.parse_args(argv)
    return run(args)


if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(main())
