from __future__ import annotations

"""Shared helpers for DUV (Duration/Velocity) inference."""

import os
from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np
import pandas as pd
import torch
from pandas.errors import UndefinedVariableError

DUV_BASE_COLUMNS: set[str] = {"pitch", "velocity", "duration", "position", "bar", "program"}
DUV_BUCKET_COLUMNS: set[str] = {"velocity_bucket", "duration_bucket"}

REQUIRED_COLUMNS: set[str] = {"pitch", "velocity", "duration", "position"}
OPTIONAL_COLUMNS: set[str] = {
    "bar",
    "bar_phase",
    "beat_phase",
    "section",
    "mood",
    "vel_bucket",
    "dur_bucket",
    "track_id",
    "program",
    "file",
    "start",
    "onset",
}

OPTIONAL_FLOAT32_COLUMNS: set[str] = {"bar_phase", "beat_phase", "start", "onset"}
OPTIONAL_INT32_COLUMNS: set[str] = {
    "bar",
    "track_id",
    "section",
    "mood",
    "vel_bucket",
    "dur_bucket",
    "program",
}
CSV_FLOAT32_COLUMNS: set[str] = {
    "velocity",
    "duration",
    "bar_phase",
    "beat_phase",
    "start",
    "onset",
    "q_onset",
    "q_duration",
}
CSV_INT32_COLUMNS: set[str] = {
    "pitch",
    "position",
    "bar",
    "track_id",
    "vel_bucket",
    "dur_bucket",
    "section",
    "mood",
    "program",
}


def _mask_any(x: object):
    """Reduce *x* to a single truthy scalar with robust semantics.
    Always returns a numpy.bool_ so callers may safely call .item()."""

    _np = np if "np" in globals() else None

    if x is None:
        return _np.bool_(False) if _np is not None else False

    try:
        t = torch.as_tensor(x)
        val = bool(t.any().item())
        return _np.bool_(val) if _np is not None else val
    except Exception:
        pass

    if _np is not None:
        try:
            return _np.bool_(_np.any(x))
        except Exception:
            pass

    try:
        val = any(bool(v) for v in x)  # type: ignore[arg-type]
        return _np.bool_(val) if _np is not None else val
    except Exception:
        val = bool(x)
        return _np.bool_(val) if _np is not None else val


def mask_any(mask: object):
    """Backward compatible wrapper for :func:`_mask_any`.
    Intentionally returns numpy.bool_ (when available) to support ``.item()``."""

    return _mask_any(mask)


def _missing_required(columns: Iterable[str]) -> list[str]:
    return sorted(REQUIRED_COLUMNS - set(columns))


def _coerce_numeric(
    df: pd.DataFrame,
    columns: Iterable[str],
    dtype: str,
    *,
    fill_value: int | float | None = 0,
) -> None:
    for column in columns:
        if column in df.columns:
            series = pd.to_numeric(df[column], errors="coerce")
            if fill_value is not None:
                series = series.fillna(fill_value)
            df[column] = series.astype(dtype, copy=False)


def load_duv_dataframe(
    csv_path: str | os.PathLike[str],
    *,
    feature_columns: Sequence[str] | None = None,
    filter_expr: str | None = None,
    limit: int = 0,
    collect_program_hist: bool = False,
) -> tuple[pd.DataFrame, pd.Series | None]:
    """Load a DUV CSV with consistent dtypes and optional filtering."""

    header = pd.read_csv(csv_path, nrows=0)
    available = list(header.columns)
    available_set = set(available)

    usecols: set[str] = set(feature_columns or ())
    usecols.update(DUV_BASE_COLUMNS & available_set)
    usecols.update(DUV_BUCKET_COLUMNS & available_set)
    usecols.update(OPTIONAL_COLUMNS & available_set)
    usecols.update(CSV_FLOAT32_COLUMNS & available_set)
    usecols.update(CSV_INT32_COLUMNS & available_set)

    ordered_cols = [col for col in available if col in usecols]

    df = pd.read_csv(
        csv_path,
        low_memory=False,
        usecols=ordered_cols if ordered_cols else None,
    )

    if "program" not in df.columns:
        df["program"] = -1

    _coerce_numeric(df, ["program"], "int16", fill_value=-1)
    _coerce_numeric(df, ["pitch"], "int16")
    _coerce_numeric(df, ["velocity"], "int8")
    _coerce_numeric(df, ["duration"], "float32", fill_value=0.0)
    _coerce_numeric(df, ["bar", "position"], "int32")

    extra_int32 = (OPTIONAL_INT32_COLUMNS | CSV_INT32_COLUMNS) - {
        "pitch",
        "program",
        "position",
        "bar",
    }
    _coerce_numeric(df, extra_int32, "int32")

    float_targets = (
        set(feature_columns or ()) | CSV_FLOAT32_COLUMNS | OPTIONAL_FLOAT32_COLUMNS
    ) - {"duration", "velocity"}
    _coerce_numeric(df, float_targets, "float32", fill_value=0.0)

    if filter_expr:
        try:
            df = df.query(filter_expr, engine="python")
        except UndefinedVariableError as exc:
            raise SystemExit(f"--filter-program referenced missing column: {exc}") from exc

    df = df.reset_index(drop=True)

    if limit > 0 and len(df) > limit:
        df = df.iloc[:limit].reset_index(drop=True)

    program_hist: pd.Series | None = None
    if collect_program_hist and "program" in df.columns:
        program_hist = df["program"].value_counts().sort_values(ascending=False)

    return df, program_hist


def duv_verbose(flag: bool | None) -> bool:
    env = os.getenv("COMPOSER2_VERBOSE", "")
    if env.lower() not in {"", "0", "false", "no"}:
        return True
    return bool(flag)


def duv_sequence_predict(
    df: pd.DataFrame,
    model: torch.nn.Module,
    device: torch.device,
    *,
    verbose: bool = False,
) -> dict[str, np.ndarray] | None:
    """Run phrase-level DUV inference when the checkpoint requires it."""

    if not getattr(model, "requires_duv_feats", False):
        return None

    df = df.reset_index(drop=True)

    missing = _missing_required(df.columns)
    if missing:
        raise RuntimeError(
            "DUV checkpoint requires phrase-level features for inference but the CSV is missing: "
            + ", ".join(missing)
        )

    if "bar" not in df.columns:
        import warnings

        warnings.warn("DUV requires bar segmentation for best results", RuntimeWarning)

    core: Any = getattr(model, "core", model)
    max_len = int(getattr(core, "max_len", getattr(model, "max_len", 16)))
    d_model = int(getattr(core, "d_model", getattr(model, "d_model", 0)))
    has_vel = bool(getattr(model, "has_vel_head", getattr(core, "head_vel_reg", None)))
    has_dur = bool(getattr(model, "has_dur_head", getattr(core, "head_dur_reg", None)))
    if not has_vel and not has_dur:
        return None

    n = len(df)
    vel_pred = np.zeros(n, dtype=np.float32)
    dur_pred = np.zeros(n, dtype=np.float32)
    vel_mask = np.zeros(n, dtype=bool)
    dur_mask = np.zeros(n, dtype=bool)

    group_cols = [c for c in ("track_id", "bar") if c in df.columns]
    groups: Iterable[tuple[Any, pd.DataFrame]]
    if group_cols:
        groups = df.groupby(group_cols, sort=False)
    else:
        groups = ((None, df),)

    missing_optional: set[str] = set()

    preview: dict[str, object] | None = None

    stderr = None
    if verbose:
        import sys

        stderr = sys.stderr

    optional_summary_printed = False

    for _, group in groups:
        if group.empty:
            continue
        g = group.sort_values("position")
        length = min(len(g), max_len)
        if length == 0:
            continue
        g = g.iloc[:length]
        idx = g.index.to_numpy(dtype=np.int64, copy=False)

        pitch_vals = torch.as_tensor(
            g["pitch"].to_numpy(dtype="int64", copy=False)[:length],
            dtype=torch.long,
            device=device,
        ).clamp_(min=0, max=127)
        pos_vals = torch.as_tensor(
            g["position"].to_numpy(dtype="int64", copy=False)[:length],
            dtype=torch.long,
            device=device,
        ).clamp_(max=max_len - 1)
        vel_vals = torch.as_tensor(
            g["velocity"].to_numpy(dtype="float32", copy=False)[:length],
            dtype=torch.float32,
            device=device,
        )
        dur_vals = torch.as_tensor(
            g["duration"].to_numpy(dtype="float32", copy=False)[:length],
            dtype=torch.float32,
            device=device,
        )

        pitch = torch.zeros(1, max_len, dtype=torch.long, device=device)
        pitch[:, :length] = pitch_vals
        pos = torch.zeros(1, max_len, dtype=torch.long, device=device)
        pos[:, :length] = pos_vals
        vel_in = torch.zeros(1, max_len, dtype=torch.float32, device=device)
        vel_in[:, :length] = vel_vals
        dur_in = torch.zeros(1, max_len, dtype=torch.float32, device=device)
        dur_in[:, :length] = dur_vals

        feat_dict: dict[str, torch.Tensor] = {
            "pitch": pitch,
            "pitch_class": pitch % 12,
            "position": pos,
            "velocity": vel_in,
            "duration": dur_in,
        }

        use_bar_beat = bool(getattr(core, "use_bar_beat", False))
        has_section = getattr(core, "section_emb", None) is not None
        has_mood = getattr(core, "mood_emb", None) is not None
        has_vel_bucket = getattr(core, "vel_bucket_emb", None) is not None
        has_dur_bucket = getattr(core, "dur_bucket_emb", None) is not None

        summary: dict[str, str] = {}

        if use_bar_beat:
            bar_phase = torch.zeros(1, max_len, dtype=torch.float32, device=device)
            beat_phase = torch.zeros(1, max_len, dtype=torch.float32, device=device)
            if "bar_phase" in g.columns:
                bar_phase[:, :length] = torch.as_tensor(
                    g["bar_phase"].to_numpy(dtype="float32", copy=False)[:length],
                    dtype=torch.float32,
                    device=device,
                )
                summary["bar_phase"] = "csv"
            else:
                missing_optional.add("bar_phase")
                summary["bar_phase"] = "zero_fill"
            if "beat_phase" in g.columns:
                beat_phase[:, :length] = torch.as_tensor(
                    g["beat_phase"].to_numpy(dtype="float32", copy=False)[:length],
                    dtype=torch.float32,
                    device=device,
                )
                summary["beat_phase"] = "csv"
            else:
                missing_optional.add("beat_phase")
                summary["beat_phase"] = "zero_fill"
            feat_dict["bar_phase"] = bar_phase
            feat_dict["beat_phase"] = beat_phase
        if has_section:
            section = torch.zeros(1, max_len, dtype=torch.long, device=device)
            if "section" in g.columns:
                section[:, :length] = torch.as_tensor(
                    g["section"].to_numpy(dtype="int64", copy=False)[:length],
                    dtype=torch.long,
                    device=device,
                )
                summary["section"] = "csv"
            else:
                missing_optional.add("section")
                summary["section"] = "zero_fill"
            feat_dict["section"] = section
        if has_mood:
            mood = torch.zeros(1, max_len, dtype=torch.long, device=device)
            if "mood" in g.columns:
                mood[:, :length] = torch.as_tensor(
                    g["mood"].to_numpy(dtype="int64", copy=False)[:length],
                    dtype=torch.long,
                    device=device,
                )
                summary["mood"] = "csv"
            else:
                missing_optional.add("mood")
                summary["mood"] = "zero_fill"
            feat_dict["mood"] = mood
        if has_vel_bucket:
            vel_bucket = torch.zeros(1, max_len, dtype=torch.long, device=device)
            if "vel_bucket" in g.columns:
                vel_bucket[:, :length] = torch.as_tensor(
                    g["vel_bucket"].to_numpy(dtype="int64", copy=False)[:length],
                    dtype=torch.long,
                    device=device,
                )
                summary["vel_bucket"] = "csv"
            else:
                missing_optional.add("vel_bucket")
                summary["vel_bucket"] = "zero_fill"
            feat_dict["vel_bucket"] = vel_bucket
        if has_dur_bucket:
            dur_bucket = torch.zeros(1, max_len, dtype=torch.long, device=device)
            if "dur_bucket" in g.columns:
                dur_bucket[:, :length] = torch.as_tensor(
                    g["dur_bucket"].to_numpy(dtype="int64", copy=False)[:length],
                    dtype=torch.long,
                    device=device,
                )
                summary["dur_bucket"] = "csv"
            else:
                missing_optional.add("dur_bucket")
                summary["dur_bucket"] = "zero_fill"
            feat_dict["dur_bucket"] = dur_bucket

        mask = torch.zeros(1, max_len, dtype=torch.bool, device=device)
        mask[:, :length] = True

        # Verbose診断ログ: mask, features distribution
        if verbose and stderr is not None:
            mask_sum = mask.sum().item()
            print(
                {
                    "duv_debug": {
                        "mask_sum": int(mask_sum),
                        "length": int(length),
                        "max_len": int(max_len),
                        "position_range": [int(pos_vals.min().item()), int(pos_vals.max().item())],
                        "position_unique": int(len(pos_vals.unique())),
                        "velocity_range": [
                            float(vel_vals.min().item()),
                            float(vel_vals.max().item()),
                        ],
                        "velocity_var": float(vel_vals.var().item()),
                        "duration_range": [
                            float(dur_vals.min().item()),
                            float(dur_vals.max().item()),
                        ],
                    }
                },
                file=stderr,
            )

        # Debug: feature statistics for verbose mode
        if verbose and stderr is not None:
            print("=== Feature Statistics ===", file=stderr)
            feature_order = sorted(feat_dict.keys())
            print(f"Feature order: {feature_order}", file=stderr)
            for key in feature_order:
                tensor = feat_dict[key]
                if tensor.dtype.is_floating_point:
                    print(
                        f"{key}: shape={tuple(tensor.shape)}, "
                        f"min={tensor.min().item():.6f}, "
                        f"max={tensor.max().item():.6f}, "
                        f"mean={tensor.mean().item():.6f}",
                        file=stderr,
                    )
                else:
                    print(
                        f"{key}: shape={tuple(tensor.shape)}, dtype={tensor.dtype}, "
                        f"min={tensor.min().item()}, "
                        f"max={tensor.max().item()}",
                        file=stderr,
                    )

            # Enable debug mode for PhraseTransformer
            if hasattr(model, "core"):
                setattr(model.core, "_debug", True)
            else:
                setattr(model, "_debug", True)

        with torch.no_grad():
            outputs = model(feat_dict, mask=mask)

        if isinstance(outputs, dict):
            # NOTE: Tensor に ``or`` を使うと真偽値変換で失敗するため、キーが無い場合だけフォールバック
            vel_out = outputs.get("velocity", None)
            if vel_out is None:
                vel_out = outputs.get("vel_reg", None)
            dur_out = outputs.get("duration", None)
            if dur_out is None:
                dur_out = outputs.get("dur_reg", None)
        elif isinstance(outputs, (tuple, list)):
            vel_out = outputs[0] if len(outputs) > 0 else None
            dur_out = outputs[1] if len(outputs) > 1 else None
        else:
            vel_out = outputs
            dur_out = None

        if verbose and preview is None:
            preview = {
                "seq_len": int(length),
                "d_model_effective": d_model or None,
                "features": sorted(feat_dict),
            }

        if verbose and not optional_summary_printed and summary and stderr is not None:
            print({"duv_optional_features": summary}, file=stderr)
            optional_summary_printed = True

        # Verbose診断ログ: model output distribution
        if verbose and stderr is not None:
            output_debug = {}
            if vel_out is not None:
                vel_raw = vel_out.squeeze(0)[:length]
                output_debug["vel_raw"] = {
                    "min": float(vel_raw.min().item()),
                    "max": float(vel_raw.max().item()),
                    "mean": float(vel_raw.mean().item()),
                    "std": float(vel_raw.std().item()),
                    "nan_count": int(torch.isnan(vel_raw).sum().item()),
                    "inf_count": int(torch.isinf(vel_raw).sum().item()),
                }
            if dur_out is not None:
                dur_raw = dur_out.squeeze(0)[:length]
                output_debug["dur_raw"] = {
                    "min": float(dur_raw.min().item()),
                    "max": float(dur_raw.max().item()),
                    "mean": float(dur_raw.mean().item()),
                    "std": float(dur_raw.std().item()),
                    "nan_count": int(torch.isnan(dur_raw).sum().item()),
                    "inf_count": int(torch.isinf(dur_raw).sum().item()),
                }
            if output_debug:
                print({"duv_model_output": output_debug}, file=stderr)

        if vel_out is not None:
            v = torch.clamp(vel_out.squeeze(0), 0.0, 1.0).mul(127.0).round().clamp(1.0, 127.0)
            vel_pred[idx] = v.detach().cpu().numpy()[:length]
            vel_mask[idx] = True
        if dur_out is not None:
            d = torch.expm1(dur_out.squeeze(0)).clamp(min=0.0)
            dur_pred[idx] = d.detach().cpu().numpy()[:length]
            dur_mask[idx] = True

    if missing_optional and verbose:
        import warnings

        warnings.warn(
            "DUV checkpoint requested optional features missing from CSV: "
            + ", ".join(sorted(missing_optional))
            + "; using zero-filled tensors for inference",
            RuntimeWarning,
        )

    if verbose and preview is not None and stderr is not None and mask_any(vel_mask):
        head = [float(v) for v in vel_pred[:8].tolist()]
        print(
            {
                "duv_preview": {
                    **preview,
                    "velocity_head": head,
                }
            },
            file=stderr,
        )

    return {
        "velocity": vel_pred,
        "velocity_mask": vel_mask,
        "duration": dur_pred,
        "duration_mask": dur_mask,
    }


__all__ = [
    "OPTIONAL_FLOAT32_COLUMNS",
    "OPTIONAL_INT32_COLUMNS",
    "OPTIONAL_COLUMNS",
    "CSV_FLOAT32_COLUMNS",
    "CSV_INT32_COLUMNS",
    "REQUIRED_COLUMNS",
    "mask_any",
    "_mask_any",
    "duv_sequence_predict",
    "duv_verbose",
]
