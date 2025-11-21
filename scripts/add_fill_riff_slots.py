#!/usr/bin/env python3
"""
bars.parquet に fill_slot / riff_slot 列を追加

目的:
  セクション境界・エネルギー上昇箇所に「ここで何か起こす」フラグを立てる。
  
  - fill_slot (bool): ドラム・フィルを発火すべき小節
  - riff_slot (bool): ギター/ピアノ/ストリングスのリフ/装飾を発火すべき小節

発火条件:
  1. セクション終端 (end_bar - 1): 境界フィル保証
  2. エネルギー急上昇 (energy_curve の差分 > threshold)
  3. fill_likelihood > threshold
  4. セクション種別 (pre_chorus, chorus, bridge は優先)

使用例:
  python scripts/add_fill_riff_slots.py \
    --bars data/suno_ai/suno_themesong/song_004/analysis/bars.parquet \
    --sections data/suno_ai/suno_themesong/song_004/analysis/sections.json \
    --out data/suno_ai/suno_themesong/song_004/analysis/bars_with_slots.parquet \
    --energy-jump-thresh 0.06 \
    --fill-likelihood-thresh 0.15 \
    --boundary-fill always

参照:
  ChatGPT guidance (2025-11-12)
  「位置決め（スロット）は bars/sections。表現の造形は楽器別レンダラ。」
"""

import argparse
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Sequence, Tuple


def load_sections(sections_path: Path) -> List[Dict[str, Any]]:
    """Load sections.json"""
    with open(sections_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("sections", data)


def load_lyric_anchors(anchors_path: Path) -> pd.DataFrame:
    """Load lyric/vocal anchor annotations (stress, sibilant, etc.)."""
    with open(anchors_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    anchors = payload.get("anchors", payload)
    records: List[Dict[str, Any]] = []
    for item in anchors:
        time_val = item.get("time")
        if time_val is None:
            continue
        classes = item.get("classes") or []
        records.append(
            {
                "time": float(time_val),
                "classes": [str(cls).lower() for cls in classes],
            }
        )

    if not records:
        return pd.DataFrame(columns=["time", "classes"])
    return pd.DataFrame.from_records(records)


def sample_voiced_duration_from_crepe(
    bars_df: pd.DataFrame,
    crepe_path: Path | None,
    voicing_threshold: float,
) -> pd.Series:
    """Return per-bar voiced seconds sampled from crepe_f0.parquet."""

    voiced = pd.Series(np.zeros(len(bars_df), dtype=float), index=bars_df.index, dtype=float)
    if not crepe_path:
        return voiced
    crepe_path = crepe_path.expanduser()
    if not crepe_path.exists():
        print(f"⚠️  CREPE file not found: {crepe_path}")
        return voiced

    try:
        crepe_df = pd.read_parquet(crepe_path)
    except Exception as exc:  # pragma: no cover - defensive I/O
        print(f"⚠️  Failed to read {crepe_path}: {exc}")
        return voiced

    if crepe_df.empty or "time_s" not in crepe_df.columns:
        return voiced

    times = crepe_df["time_s"].to_numpy(dtype=float, copy=False)
    valid_times = times[np.isfinite(times)]
    if len(valid_times) <= 1:
        frame_sec = 0.01
    else:
        diffs = np.diff(np.sort(valid_times))
        frame_sec = float(np.median(diffs[diffs > 0])) if np.any(diffs > 0) else 0.01

    voicing_series = crepe_df.get("voicing_prob")
    if voicing_series is not None:
        voiced_mask = voicing_series.fillna(0.0) >= voicing_threshold
    else:
        voiced_mask = crepe_df.get("f0_hz", pd.Series([], dtype=float)) > 0.0

    if voiced_mask.empty or not voiced_mask.any():
        return voiced

    if "bar_index" in crepe_df.columns:
        counts = crepe_df.loc[voiced_mask, "bar_index"].dropna().astype(int).value_counts()
        for bar_idx, frames in counts.items():
            if 0 <= bar_idx < len(voiced):
                voiced.iloc[bar_idx] += frames * frame_sec
        return voiced

    if not {"start_sec", "end_sec"}.issubset(bars_df.columns):
        return voiced

    starts = bars_df["start_sec"].to_numpy(dtype=float, copy=False)
    ends = bars_df["end_sec"].to_numpy(dtype=float, copy=False)
    voiced_times = times[voiced_mask.to_numpy()]  # type: ignore[arg-type]
    for t in voiced_times:
        idx = np.searchsorted(starts, t, side="right") - 1
        if 0 <= idx < len(bars_df) and t < ends[idx]:
            voiced.iloc[idx] += frame_sec
    return voiced


def _density_bucket_from_value(value: Any) -> str:
    if isinstance(value, str):
        token = value.lower()
        if token in {"sparse", "medium", "dense", "wall"}:
            return token
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "medium"
    if numeric < 0.35:
        return "sparse"
    if numeric < 0.6:
        return "medium"
    if numeric < 0.85:
        return "dense"
    return "wall"


def annotate_vocal_accents(
    df: pd.DataFrame,
    anchors: pd.DataFrame,
    boundary_ms: float,
) -> Dict[str, pd.Series]:
    n_bars = len(df)
    zeros_int = np.zeros(n_bars, dtype=np.int64)
    zeros_bool = np.zeros(n_bars, dtype=bool)
    result = {
        "vocal_accent_count": pd.Series(zeros_int.copy(), index=df.index),
        "vocal_accent_near_boundary": pd.Series(zeros_bool.copy(), index=df.index),
        "vocal_plosive_hits": pd.Series(zeros_int.copy(), index=df.index),
        "vocal_sibilant_hits": pd.Series(zeros_int.copy(), index=df.index),
    }

    if anchors is None or anchors.empty:
        return result
    if not {"start_sec", "end_sec"}.issubset(df.columns):
        return result

    starts = df["start_sec"].to_numpy()
    ends = df["end_sec"].to_numpy()
    boundary_window = max(boundary_ms, 0.0) / 1000.0

    accent_count = zeros_int.copy()
    boundary_hits = zeros_bool.copy()
    plosive_hits = zeros_int.copy()
    sibilant_hits = zeros_int.copy()

    for _, anchor in anchors.iterrows():
        time_val = anchor.get("time")
        if time_val is None:
            continue
        t = float(time_val)
        idx = np.searchsorted(starts, t, side="right") - 1
        if idx < 0 or idx >= n_bars:
            continue
        if t >= ends[idx]:
            continue
        classes = [str(cls).lower() for cls in (anchor.get("classes") or [])]
        if "stress" in classes:
            accent_count[idx] += 1
            dist_start = abs(t - starts[idx])
            dist_end = abs(ends[idx] - t)
            if min(dist_start, dist_end) <= boundary_window:
                boundary_hits[idx] = True
        if "plosive" in classes:
            plosive_hits[idx] += 1
        if "sibilant" in classes:
            sibilant_hits[idx] += 1

    result["vocal_accent_count"] = pd.Series(accent_count, index=df.index)
    result["vocal_accent_near_boundary"] = pd.Series(boundary_hits, index=df.index)
    result["vocal_plosive_hits"] = pd.Series(plosive_hits, index=df.index)
    result["vocal_sibilant_hits"] = pd.Series(sibilant_hits, index=df.index)
    return result


def compute_vocal_profile(
    df: pd.DataFrame,
    accent_data: Dict[str, pd.Series],
    sync_onset_pct: float,
    counter_onset_pct: float,
    sync_rms_pct: float,
    counter_rms_pct: float,
    voiced_ratio: pd.Series,
    sync_ratio_thresh: float,
    counter_ratio_thresh: float,
) -> Tuple[pd.Series, pd.Series]:
    onset_rate = df.get("vocal_onset_rate", pd.Series([0.0] * len(df), index=df.index)).fillna(0)
    vocal_rms = df.get("vocal_rms_db", pd.Series([-32.0] * len(df), index=df.index)).copy()
    vocal_rms = vocal_rms.ffill().fillna(-28.0)
    phrases = df.get("phrase_boundary", pd.Series([0] * len(df), index=df.index)).fillna(0)
    stress = accent_data["vocal_accent_count"].fillna(0)
    boundary_hit = accent_data["vocal_accent_near_boundary"].fillna(False)
    ratio = voiced_ratio.reindex(df.index).fillna(0.0)

    onset_high = float(onset_rate.quantile(sync_onset_pct)) if onset_rate.notna().any() else 8.0
    onset_low = float(onset_rate.quantile(counter_onset_pct)) if onset_rate.notna().any() else 2.0
    rms_high = float(vocal_rms.quantile(sync_rms_pct)) if vocal_rms.notna().any() else -16.0
    rms_low = float(vocal_rms.quantile(counter_rms_pct)) if vocal_rms.notna().any() else -24.0

    ratio_window = max(sync_ratio_thresh - counter_ratio_thresh, 1e-3)
    ratio_score = ((ratio - counter_ratio_thresh) / ratio_window).clip(0.0, 1.0)

    sync_score = (
        0.35 * (stress > 0).astype(float)
        + 0.2 * boundary_hit.astype(float)
        + 0.2 * (onset_rate >= onset_high).astype(float)
        + 0.15 * (vocal_rms >= rms_high).astype(float)
        + 0.4 * ratio_score
    )
    counter_score = (
        0.3 * (stress == 0).astype(float)
        + 0.2 * (phrases > 0).astype(float)
        + 0.2 * (onset_rate <= onset_low).astype(float)
        + 0.15 * (vocal_rms <= rms_low).astype(float)
        + 0.45 * (1.0 - ratio_score)
    )

    sync_force = ratio >= sync_ratio_thresh
    counter_force = ratio <= counter_ratio_thresh

    dominance = sync_score - counter_score
    profile = np.full(len(df), "neutral", dtype=object)
    profile[sync_force] = "sync"
    profile[~sync_force & counter_force] = "counter"

    neutral_mask = profile == "neutral"
    sync_strong = neutral_mask & (dominance >= 0.15) & (sync_score >= 0.55)
    counter_strong = neutral_mask & (dominance <= -0.15) & (counter_score >= 0.55)
    profile[sync_strong] = "sync"
    profile[counter_strong] = "counter"

    profile_series = pd.Series(profile, index=df.index)
    confidence = dominance.abs().clip(0.0, 1.0)
    confidence = confidence.mask(profile_series == "sync", sync_score.clip(0.0, 1.0))
    confidence = confidence.mask(profile_series == "counter", counter_score.clip(0.0, 1.0))
    confidence = confidence.fillna(0.0).clip(0.0, 1.0)
    confidence.loc[sync_force | counter_force] = confidence.loc[sync_force | counter_force].clip(
        lower=0.7
    )

    return profile_series, confidence


def compute_density_bounds(
    df: pd.DataFrame,
    vocal_profile: pd.Series,
    sync_min_events: int,
    counter_max_events: int,
) -> Tuple[pd.Series, pd.Series]:
    ranges = {
        "sparse": (0, 10),
        "medium": (4, 18),
        "dense": (8, 30),
        "wall": (14, 48),
    }
    floors = []
    ceilings = []
    for idx, row in df.iterrows():
        bucket = _density_bucket_from_value(row.get("density_target"))
        floor, ceil = ranges.get(bucket, (6, 22))
        if idx in vocal_profile.index:
            profile = str(vocal_profile.loc[idx])
        else:
            profile = "neutral"
        if profile == "sync":
            floor = max(floor, sync_min_events)
            ceil = max(ceil, sync_min_events + 12)
        elif profile == "counter":
            floor = 0
            ceil = min(ceil, counter_max_events)
        floors.append(int(max(0, floor)))
        ceilings.append(int(max(floors[-1], ceil)))
    return pd.Series(floors, index=df.index), pd.Series(ceilings, index=df.index)


def derive_style_hint(fill_slot: bool, groove_slot: bool, vocal_profile: str) -> str:
    base = "fill" if fill_slot else ("groove" if groove_slot else "neutral")
    if vocal_profile in {"sync", "counter"}:
        if base != "neutral":
            return f"{base}_{vocal_profile}"
        return vocal_profile
    return base


def compute_fill_slots(
    df: pd.DataFrame,
    sections: List[Dict[str, Any]],
    energy_jump_thresh: float = 0.06,
    fill_likelihood_thresh: float = 0.15,
    boundary_fill: str = "always",
) -> pd.Series:
    """
    fill_slot を計算

    条件:
      1. boundary_fill="always": セクション終端-1 で必ず True
      2. energy_curve の差分 > energy_jump_thresh
      3. fill_likelihood > fill_likelihood_thresh
    """
    n_bars = len(df)
    fill_slot = pd.Series([False] * n_bars, index=df.index)

    # 1. セクション境界フィル（end_bar - 1）
    if boundary_fill == "always":
        for sec in sections:
            end_bar = sec.get("end_bar", sec.get("bar_end"))
            if end_bar is not None and end_bar > 0:
                # 終端の直前
                boundary_idx = end_bar - 1
                if 0 <= boundary_idx < n_bars:
                    fill_slot.iloc[boundary_idx] = True

    # 2. エネルギー急上昇
    if "energy_curve" in df.columns:
        energy_diff = df["energy_curve"].diff().fillna(0)
        fill_slot |= energy_diff > energy_jump_thresh

    # 3. fill_likelihood 高
    if "fill_likelihood" in df.columns:
        fill_slot |= df["fill_likelihood"] > fill_likelihood_thresh

    return fill_slot


def suppress_fill_interior_sections(
    df: pd.DataFrame,
    fill_slot: pd.Series,
    sections: List[Dict[str, Any]],
    suppress_labels: Sequence[str],
) -> pd.Series:
    if not suppress_labels:
        return fill_slot

    suppress_lower = [label.lower() for label in suppress_labels]
    updated = fill_slot.copy()
    if "bar_index" not in df.columns:
        return updated

    for sec in sections:
        label = str(sec.get("label") or sec.get("section_label") or "").lower()
        if not any(key in label for key in suppress_lower):
            continue
        start_bar = sec.get("start_bar", sec.get("bar_start"))
        end_bar = sec.get("end_bar", sec.get("bar_end"))
        if start_bar is None or end_bar is None:
            continue
        # Keep boundary fills (end_bar - 1) while clearing interior bars
        interior_end = max(start_bar, end_bar - 1)
        mask = (df["bar_index"] >= start_bar) & (df["bar_index"] < interior_end)
        updated.loc[mask] = False
    return updated


def compute_riff_slots(
    df: pd.DataFrame,
    sections: List[Dict[str, Any]],
    riff_sections: List[str] = None,
    min_activity: float = 0.2,
) -> pd.Series:
    """
    riff_slot を計算

    条件:
      1. セクション種別が riff_sections に含まれる
      2. guitar_activity / piano_activity / strings_activity > min_activity
      3. セクション終端-1（境界装飾）
    """
    if riff_sections is None:
        riff_sections = ["pre_chorus", "chorus", "bridge"]

    n_bars = len(df)
    riff_slot = pd.Series([False] * n_bars, index=df.index)

    # 1. セクション種別フィルタ
    if "section_label" in df.columns:
        for label in riff_sections:
            riff_slot |= df["section_label"].str.contains(label, case=False, na=False)

    # 2. アクティビティ高
    for col in ["guitar_activity", "piano_activity", "strings_activity"]:
        if col in df.columns:
            riff_slot |= df[col] > min_activity

    # 3. セクション境界装飾（end_bar - 1）
    for sec in sections:
        label = sec.get("label", "")
        if any(key in label.lower() for key in riff_sections):
            end_bar = sec.get("end_bar", sec.get("bar_end"))
            if end_bar is not None and end_bar > 0:
                boundary_idx = end_bar - 1
                if 0 <= boundary_idx < n_bars:
                    riff_slot.iloc[boundary_idx] = True

    return riff_slot


def compute_groove_slots(
    df: pd.DataFrame,
    fill_slot: pd.Series,
    groove_sections: Sequence[str],
    min_activity: float = 0.35,
) -> pd.Series:
    n_bars = len(df)
    groove_slot = pd.Series([False] * n_bars, index=df.index)
    if "section_label" in df.columns and groove_sections:
        for label in groove_sections:
            groove_slot |= df["section_label"].str.contains(label, case=False, na=False)

    activity = None
    if "drums_active" in df.columns:
        activity = df["drums_active"].fillna(0.5)
    elif "energy_curve" in df.columns:
        activity = df["energy_curve"].fillna(0.5)
    else:
        activity = pd.Series([0.5] * n_bars, index=df.index)

    groove_slot &= activity >= min_activity
    groove_slot &= ~fill_slot
    return groove_slot


def main():
    ap = argparse.ArgumentParser(description="Add fill_slot / riff_slot to bars.parquet")
    ap.add_argument("--bars", required=True, help="Input bars.parquet")
    ap.add_argument("--sections", required=True, help="sections.json")
    ap.add_argument("--out", required=True, help="Output bars_with_slots.parquet")
    ap.add_argument(
        "--energy-jump-thresh",
        type=float,
        default=0.06,
        help="Energy jump threshold for fill_slot (default: 0.06)",
    )
    ap.add_argument(
        "--fill-likelihood-thresh",
        type=float,
        default=0.15,
        help="fill_likelihood threshold (default: 0.15)",
    )
    ap.add_argument(
        "--boundary-fill",
        choices=["never", "auto", "always"],
        default="always",
        help="Boundary fill mode (default: always)",
    )
    ap.add_argument(
        "--suppress-fill-sections",
        nargs="+",
        default=["chorus"],
        help="Section labels whose interior bars should keep fill_slot=False (default: chorus)",
    )
    ap.add_argument(
        "--riff-sections",
        nargs="+",
        default=["pre_chorus", "chorus", "bridge"],
        help="Section labels for riff priority (default: pre_chorus chorus bridge)",
    )
    ap.add_argument(
        "--min-riff-activity",
        type=float,
        default=0.2,
        help="Minimum activity for riff_slot (default: 0.2)",
    )
    ap.add_argument(
        "--groove-sections",
        nargs="+",
        default=["verse", "chorus"],
        help="Section labels eligible for groove_slot tagging (default: verse chorus)",
    )
    ap.add_argument(
        "--min-groove-activity",
        type=float,
        default=0.35,
        help="Minimum drums_active/energy needed for groove_slot (default: 0.35)",
    )
    ap.add_argument(
        "--crepe-f0",
        help="Optional crepe_f0.parquet to measure voiced duration",
    )
    ap.add_argument(
        "--voicing-threshold",
        type=float,
        default=0.6,
        help="Voicing probability threshold when counting CREPE frames (default: 0.6)",
    )
    ap.add_argument(
        "--lyric-anchors",
        help="Optional lyric_anchors.json used to derive vocal accents",
    )
    ap.add_argument(
        "--accent-boundary-ms",
        type=float,
        default=120.0,
        help="Window (ms) around bar boundaries considered 'sync' accents (default: 120)",
    )
    ap.add_argument(
        "--sync-onset-pct",
        type=float,
        default=0.65,
        help="Quantile for high vocal onset rate when tagging sync bars (default: 0.65)",
    )
    ap.add_argument(
        "--counter-onset-pct",
        type=float,
        default=0.35,
        help="Quantile for low vocal onset rate when tagging counter bars (default: 0.35)",
    )
    ap.add_argument(
        "--sync-rms-pct",
        type=float,
        default=0.65,
        help="Quantile for loud vocal RMS when tagging sync bars (default: 0.65)",
    )
    ap.add_argument(
        "--counter-rms-pct",
        type=float,
        default=0.35,
        help="Quantile for quiet vocal RMS when tagging counter bars (default: 0.35)",
    )
    ap.add_argument(
        "--sync-voiced-ratio",
        type=float,
        default=0.55,
        help="Minimum vocal_voiced_ratio to force sync profile (default: 0.55)",
    )
    ap.add_argument(
        "--counter-voiced-ratio",
        type=float,
        default=0.2,
        help="Maximum vocal_voiced_ratio to force counter profile (default: 0.20)",
    )
    ap.add_argument(
        "--sync-min-events",
        type=int,
        default=8,
        help="Minimum events per bar demanded for sync vocal moments (default: 8)",
    )
    ap.add_argument(
        "--counter-max-events",
        type=int,
        default=14,
        help="Maximum events per bar allowed for counter vocal moments (default: 14)",
    )
    args = ap.parse_args()

    # Load
    bars_path = Path(args.bars)
    sections_path = Path(args.sections)
    out_path = Path(args.out)

    df = pd.read_parquet(bars_path)
    sections = load_sections(sections_path)

    print(f"📊 Input: {len(df)} bars")

    # Compute slots
    fill_slot = compute_fill_slots(
        df,
        sections,
        energy_jump_thresh=args.energy_jump_thresh,
        fill_likelihood_thresh=args.fill_likelihood_thresh,
        boundary_fill=args.boundary_fill,
    )
    suppressed = suppress_fill_interior_sections(
        df,
        fill_slot,
        sections,
        suppress_labels=args.suppress_fill_sections,
    )
    if not suppressed.equals(fill_slot):
        print(
            f"   ℹ️  Suppressed fill slots inside sections: {(fill_slot.sum() - suppressed.sum())} bars"
        )
    fill_slot = suppressed

    riff_slot = compute_riff_slots(
        df,
        sections,
        riff_sections=args.riff_sections,
        min_activity=args.min_riff_activity,
    )

    groove_slot = compute_groove_slots(
        df,
        fill_slot,
        groove_sections=args.groove_sections,
        min_activity=args.min_groove_activity,
    )

    voiced_duration = sample_voiced_duration_from_crepe(
        df,
        Path(args.crepe_f0) if args.crepe_f0 else None,
        voicing_threshold=args.voicing_threshold,
    )
    if {"start_sec", "end_sec"}.issubset(df.columns):
        bar_len = (df["end_sec"] - df["start_sec"]).replace(0, np.nan)
        voiced_ratio = voiced_duration.divide(bar_len).fillna(0.0)
    else:
        voiced_ratio = pd.Series([0.0] * len(df), index=df.index)

    anchors_df = (
        load_lyric_anchors(Path(args.lyric_anchors))
        if args.lyric_anchors
        else pd.DataFrame(columns=["time", "classes"])
    )
    accent_data = annotate_vocal_accents(df, anchors_df, args.accent_boundary_ms)
    vocal_profile, vocal_profile_conf = compute_vocal_profile(
        df,
        accent_data,
        sync_onset_pct=args.sync_onset_pct,
        counter_onset_pct=args.counter_onset_pct,
        sync_rms_pct=args.sync_rms_pct,
        counter_rms_pct=args.counter_rms_pct,
        voiced_ratio=voiced_ratio,
        sync_ratio_thresh=args.sync_voiced_ratio,
        counter_ratio_thresh=args.counter_voiced_ratio,
    )
    density_floor, density_ceiling = compute_density_bounds(
        df,
        vocal_profile,
        sync_min_events=args.sync_min_events,
        counter_max_events=args.counter_max_events,
    )

    # Add columns
    df["fill_slot"] = fill_slot
    df["riff_slot"] = riff_slot
    df["groove_slot"] = groove_slot
    df["vocal_accent_count"] = accent_data["vocal_accent_count"]
    df["vocal_accent_near_boundary"] = accent_data["vocal_accent_near_boundary"]
    df["vocal_voiced_sec"] = voiced_duration
    df["vocal_voiced_ratio"] = voiced_ratio.clip(0.0, 1.2)
    df["vocal_profile"] = vocal_profile
    df["vocal_profile_confidence"] = vocal_profile_conf.clip(0.0, 1.0)
    df["vocal_density_floor"] = density_floor
    df["vocal_density_ceiling"] = density_ceiling
    df["drum_style_hint"] = [
        derive_style_hint(bool(f), bool(g), str(p))
        for f, g, p in zip(df["fill_slot"], df["groove_slot"], df["vocal_profile"])
    ]

    # Save
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)

    # Report
    fill_count = fill_slot.sum()
    riff_count = riff_slot.sum()

    print(f"\n✅ Output: {out_path}")
    groove_count = groove_slot.sum()
    print(f"   fill_slot: {fill_count} bars ({fill_count/len(df)*100:.1f}%)")
    print(f"   riff_slot: {riff_count} bars ({riff_count/len(df)*100:.1f}%)")
    print(f"   groove_slot: {groove_count} bars ({groove_count/len(df)*100:.1f}%)")
    sync_count = (df["vocal_profile"] == "sync").sum()
    counter_count = (df["vocal_profile"] == "counter").sum()
    print(
        f"   vocal sync/counter: {sync_count}/{counter_count} bars "
        f"({sync_count/len(df)*100:.1f}% / {counter_count/len(df)*100:.1f}%)"
    )
    if voiced_duration.sum() > 0:
        avg_ratio = df["vocal_voiced_ratio"].mean()
        print(f"   avg vocal voiced ratio: {avg_ratio*100:.1f}%")

    # Section boundary check
    boundary_fills = 0
    for sec in sections:
        end_bar = sec.get("end_bar", sec.get("bar_end"))
        if end_bar is not None and end_bar > 0:
            boundary_idx = end_bar - 1
            if 0 <= boundary_idx < len(df) and df["fill_slot"].iloc[boundary_idx]:
                boundary_fills += 1

    print(f"\n🎯 Section boundary fills: {boundary_fills}/{len(sections)} sections")

    # Sample
    if fill_count > 0:
        print("\nSample fill_slot bars:")
        fill_bars = df[df["fill_slot"]][:5]
        for idx, row in fill_bars.iterrows():
            print(
                f"  bar {row['bar_index']}: {row.get('section_label', 'N/A')} "
                f"(energy={row.get('energy_curve', 0):.2f}, "
                f"fill_likelihood={row.get('fill_likelihood', 0):.2f})"
            )

    if riff_count > 0:
        print("\nSample riff_slot bars:")
        riff_bars = df[df["riff_slot"]][:5]
        for idx, row in riff_bars.iterrows():
            print(
                f"  bar {row['bar_index']}: {row.get('section_label', 'N/A')} "
                f"(guitar={row.get('guitar_activity', 0):.2f}, "
                f"piano={row.get('piano_activity', 0):.2f})"
            )


if __name__ == "__main__":
    main()
