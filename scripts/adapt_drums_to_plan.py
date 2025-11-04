#!/usr/bin/env python3
# Adapt drums recommendations → plan
# Extended: stem MIDI (weak labels) + stems_features/bars (activity/density) + lyric anchors (ducking)
# Backward compatible: if --stem-midi/--stems-features are omitted, behaves like the classic adapter.

import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import math

import numpy as np
import pandas as pd
from random import random

try:
    import pretty_midi
except Exception:
    pretty_midi = None


GM_KICK = {35, 36}
GM_SNARE = {38, 40}
GM_HH_C = {42}  # closed
GM_HH_P = {44}  # pedal
GM_HH_O = {46}  # open
GM_TOMS = {41, 43, 45, 47, 48, 50}
GM_RIDES = {51, 53, 59}
GM_CRASH = {49, 52, 57}
GM_ALL = set().union(GM_KICK, GM_SNARE, GM_HH_C, GM_HH_P, GM_HH_O, GM_TOMS, GM_RIDES, GM_CRASH)

TOMS_LOW = [41, 43, 45]  # floor / low
TOMS_MID = [47, 48, 50]
TOMS_HIGH = [50, 48, 47]  # descending color


def _sec_to_beats(t_sec: float, tempo_bpm: float) -> float:
    return (t_sec * tempo_bpm) / 60.0


def _beats_to_sec(b: float, tempo_bpm: float) -> float:
    return (b * 60.0) / tempo_bpm


def _safe_col(df: pd.DataFrame, name: str, default) -> pd.Series:
    return df[name] if name in df.columns else pd.Series([default] * len(df), index=df.index)


def _load_bars(bars_path: Optional[str], tempo_bpm: float) -> pd.DataFrame:
    if not bars_path:
        # minimal 1 bar fallback
        return pd.DataFrame(
            [
                {
                    "bar_index": 0,
                    "start_beat": 0.0,
                    "end_beat": 4.0,
                    "drums_active": 1,
                    "density_target": 0.5,
                    "hat_density": 0.0,
                    "swing_target": 0.0,
                    "energy_curve": 0.5,
                }
            ]
        )
    df = pd.read_parquet(bars_path)
    # normalize column names
    if "bar" in df.columns and "bar_index" not in df.columns:
        df["bar_index"] = df["bar"]
    if "start_beats" in df.columns and "start_beat" not in df.columns:
        df["start_beat"] = df["start_beats"]
    if "end_beats" in df.columns and "end_beat" not in df.columns:
        df["end_beat"] = df["end_beats"]
    # required with defaults
    if "drums_active" not in df.columns:
        df["drums_active"] = 1
    if "density_target" not in df.columns:
        # gentle mapping from energy to target density
        energy = df["energy_curve"] if "energy_curve" in df.columns else 0.5
        if not isinstance(energy, pd.Series):
            energy = pd.Series([energy] * len(df))
        df["density_target"] = energy.clip(0, 1) * 0.6 + 0.2
    if "swing_target" not in df.columns:
        df["swing_target"] = 0.0
    if "energy_curve" not in df.columns:
        df["energy_curve"] = 0.5
    # start/end beats
    if "start_beat" not in df.columns or "end_beat" not in df.columns:
        # synthesize from bar_index
        df["start_beat"] = df["bar_index"] * 4.0
        df["end_beat"] = df["start_beat"] + 4.0
    return df


def _load_stem_features(path: Optional[str]) -> Optional[pd.DataFrame]:
    if not path:
        return None
    try:
        df = pd.read_parquet(path)
        if "bar" in df.columns and "bar_index" not in df.columns:
            df["bar_index"] = df["bar"]
        return df
    except Exception:
        return None


def _load_anchors(path: Optional[str], tempo_bpm: float) -> List[Tuple[float, float]]:
    """Return list of (start_beats, end_beats) windows to duck hi-hats around vocal activity."""
    if not path or not Path(path).exists():
        return []
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return []
    wins = []
    # accept formats: [{"time_sec":..,"dur_sec":..}] or {"anchors":[...]}
    anchors = data.get("anchors", data)
    for a in anchors:
        t = a.get("time_sec", a.get("time", 0.0))
        d = a.get("dur_sec", a.get("duration", 0.08))
        s_b = _sec_to_beats(float(t), tempo_bpm)
        e_b = _sec_to_beats(float(t) + float(d), tempo_bpm)
        wins.append((s_b, e_b))
    return wins


def _hh_grid_positions(bar_start: float, density_target: float) -> List[float]:
    """
    Return preferred HH onsets (beats) inside the bar according to target density.
    density_target roughly maps to 8th/16th density.
    """
    # Map 0..1 to 8..16th grid count
    n = int(round(8 + density_target * 8))  # 8..16
    step = 4.0 / n  # beats per HH
    return [bar_start + i * step for i in range(n)]


def _is_in_windows(t_beats: float, wins: List[Tuple[float, float]], pad: float = 0.05) -> bool:
    for s, e in wins:
        if (t_beats >= s - pad) and (t_beats <= e + pad):
            return True
    return False


def _label_from_pitch(p: int) -> str:
    if p in GM_KICK:
        return "kick"
    if p in GM_SNARE:
        return "snare"
    if p in GM_HH_C:
        return "hh_c"
    if p in GM_HH_P:
        return "hh_p"
    if p in GM_HH_O:
        return "hh_o"
    if p in GM_TOMS:
        return "tom"
    if p in GM_RIDES:
        return "ride"
    if p in GM_CRASH:
        return "crash"
    return "other"


def _load_stem_midi(path: str, tempo_bpm: float) -> List[Dict[str, Any]]:
    """Return list of drum notes dicts from stem MIDI (onset/offset in beats, pitch, velocity)."""
    if not path or not Path(path).exists() or pretty_midi is None:
        return []
    pm = pretty_midi.PrettyMIDI(path)
    evts: List[Dict[str, Any]] = []
    for inst in pm.instruments:
        if not inst.is_drum:
            continue
        for n in inst.notes:
            pitch = int(n.pitch)
            if pitch not in GM_ALL:
                continue
            start_b = _sec_to_beats(float(n.start), tempo_bpm)
            end_b = _sec_to_beats(float(n.end), tempo_bpm)
            evts.append(
                {
                    "start_beats": start_b,
                    "end_beats": end_b if end_b > start_b else start_b + 0.05,
                    "pitch": pitch,
                    "velocity": int(np.clip(n.velocity, 1, 127)),
                    "label": _label_from_pitch(pitch),
                }
            )
    # sort by time
    evts.sort(key=lambda x: x["start_beats"])
    return evts


def _collect_by_bar(
    events: List[Dict[str, Any]], bars: pd.DataFrame
) -> Dict[int, List[Dict[str, Any]]]:
    out: Dict[int, List[Dict[str, Any]]] = {int(b): [] for b in bars["bar_index"].tolist()}
    for e in events:
        t = e["start_beats"]
        # find bar index quickly by floor on 4/4 assumption using start_beat
        # robust: scan bars where start_beat<=t<end_beat
        hit = bars[(bars["start_beat"] <= t) & (t < bars["end_beat"])]
        if len(hit) == 0:
            # if at the very end, glue to last bar
            bidx = int(bars["bar_index"].iloc[-1])
        else:
            bidx = int(hit["bar_index"].iloc[0])
        out.setdefault(bidx, []).append(e)
    return out


def _add_ghost_hats(
    bar_events: List[Dict[str, Any]],
    bar_start: float,
    density_target: float,
    vocal_windows: List[Tuple[float, float]],
    hh_pitch: int = 42,
    base_vel: int = 46,
    vel_jit: int = 8,
    min_gap_beat: float = 0.08,
):
    """
    Add missing HH according to density target on a 8~16th grid,
    skipping vocal windows and too-close collisions.
    """
    # current HH onsets
    curr = sorted(
        [e["start_beats"] for e in bar_events if _label_from_pitch(e["pitch"]).startswith("hh")]
    )
    desired = _hh_grid_positions(bar_start, density_target)
    for t in desired:
        # skip near vocal
        if _is_in_windows(t, vocal_windows, pad=0.05):
            continue
        # skip if too close to existing HH
        if any(abs(t - c) < min_gap_beat for c in curr):
            continue
        v = int(np.clip(np.random.normal(base_vel, vel_jit), 18, 72))
        bar_events.append(
            {
                "start_beats": t,
                "end_beats": t + 0.05,
                "pitch": hh_pitch,
                "velocity": v,
                "label": "hh_c",
            }
        )
        curr.append(t)


def _apply_ducking(
    bar_events: List[Dict[str, Any]], vocal_windows: List[Tuple[float, float]], duck_vel: int = 12
):
    """Lower HH/Ride/Crash velocity around vocal anchors to avoid harshness."""
    for e in bar_events:
        lab = _label_from_pitch(e["pitch"])
        if lab in ("hh_c", "hh_p", "hh_o", "ride", "crash"):
            if _is_in_windows(e["start_beats"], vocal_windows, pad=0.06):
                e["velocity"] = max(1, e["velocity"] - duck_vel)


def _ensure_backbeat(
    events: List[Dict[str, Any]],
    bar_idx: int,
    min_vel: int = 86,
    snare_pitch: int = 38,
    flam: bool = False,
    flam_ms: float = 12.0,
    tempo_bpm: float = 120.0,
) -> int:
    """Guarantee snare on beats 2 & 4; boost velocity; optional tiny flam."""
    bar_start = bar_idx * 4.0
    target_beats = [bar_start + 1.0, bar_start + 3.0]  # beats 2 & 4
    added = 0

    for tb in target_beats:
        # 既存スネアを探索（±0.08拍以内）
        sn = [e for e in events if e["pitch"] == snare_pitch and abs(e["start_beats"] - tb) < 0.08]
        if sn:
            # velocity boosting
            if sn[0]["velocity"] < min_vel:
                sn[0]["velocity"] = min_vel
        else:
            # スネア追加
            events.append(
                {
                    "pitch": snare_pitch,
                    "velocity": min_vel,
                    "start_beats": float(tb),
                    "end_beats": float(tb) + 0.12,
                    "channel": 9,
                    "label": "snare",
                }
            )
            added += 1

        # フラム追加（オプション）
        if flam:
            flam_offset = -(flam_ms / 1000.0) * (tempo_bpm / 60.0)  # msをbeatsに変換
            flam_vel = max(40, int(min_vel * 0.55))
            events.append(
                {
                    "pitch": snare_pitch,
                    "velocity": flam_vel,
                    "start_beats": float(tb + flam_offset),
                    "end_beats": float(tb + flam_offset + 0.06),
                    "channel": 9,
                    "label": "snare_flam",
                }
            )

    return added


def _two_stage_fill(
    events: List[Dict[str, Any]],
    bar_idx: int,
    kit: Dict[str, int],
) -> None:
    """Ride→Tom fill: last 1 bar。小節後半で ride 連打→中低 tom ラン."""
    bar_start = bar_idx * 4.0
    start_b = bar_start + 2.5  # beat 2.5から開始

    # ride 3連打
    for i in range(3):
        events.append(
            {
                "pitch": kit.get("ride", 51),
                "velocity": 82 + i * 3,
                "start_beats": float(start_b + i * 0.25),
                "end_beats": float(start_b + i * 0.25 + 0.08),
                "channel": 9,
                "label": "ride",
            }
        )

    # tom run（ハイ→ミッド→ロー）
    tom_pitches = [
        kit.get("tom_hi", 48),
        kit.get("tom_mid", 47),
        kit.get("tom_low", 45),
    ]
    for i, p in enumerate(tom_pitches):
        events.append(
            {
                "pitch": p,
                "velocity": 88 + i * 4,
                "start_beats": float(start_b + 0.9 + i * 0.15),
                "end_beats": float(start_b + 0.9 + i * 0.15 + 0.12),
                "channel": 9,
                "label": f"tom_{i}",
            }
        )


def _apply_open_hat_policy(
    bar_events: List[Dict[str, Any]],
    bar_slice: pd.Series,
    vocal_windows: List[Tuple[float, float]],
    open_prob: float = 0.20,
    close_delay_beats: float = 0.18,
    min_gap_beats: float = 0.10,
    open_vel_boost: int = 10,
    avoid_vocal: bool = True,
):
    """
    一部のクローズHHをオープンHHに置き換え、直後にペダル閉鎖を置く。
    ・energy_curve / accent_score_target が高い箇所を優先
    ・vocal window 付近は回避
    """
    energy = float(bar_slice.get("energy_curve", 0.5))
    accent = float(bar_slice.get("accent_score_target", energy))
    prob = open_prob * (0.6 + 0.8 * energy) * (0.7 + 0.6 * accent)

    # HH候補を抽出（クローズのみ）
    hh_idx = [i for i, e in enumerate(bar_events) if _label_from_pitch(e["pitch"]) == "hh_c"]
    if not hh_idx:
        return
    last_open_t = -1e9
    result = []
    for i in range(len(bar_events)):
        e = bar_events[i]
        if i in hh_idx and random() < prob:
            t = e["start_beats"]
            if (t - last_open_t) < min_gap_beats:
                result.append(e)
                continue
            if avoid_vocal and _is_in_windows(t, vocal_windows, pad=0.06):
                result.append(e)
                continue
            # 置換：open(46) にして vel を少しブースト
            v = int(np.clip(e.get("velocity", e.get("vel", 64)) + open_vel_boost, 1, 110))
            open_evt = dict(e)
            open_evt["pitch"] = 46  # open hat
            open_evt["velocity"] = v
            open_evt["vel"] = v
            # エンドは少し長め（ただし小節内にクランプは後段で行う）
            open_evt["end_beats"] = max(
                open_evt.get("end_beats", t + 0.10), t + close_delay_beats * 0.9
            )
            result.append(open_evt)
            # クローズ：pedal(44) を close_delay で
            close_evt = {
                "start_beats": t + close_delay_beats,
                "end_beats": t + close_delay_beats + 0.05,
                "pitch": 44,  # pedal close
                "velocity": int(v * 0.85),
                "vel": int(v * 0.85),
                "label": "hh_p",
            }
            result.append(close_evt)
            last_open_t = t
        else:
            result.append(e)
    # 反映
    bar_events.clear()
    bar_events.extend(result)


def _pick_tom_palette(name: str) -> List[int]:
    name = (name or "mid").lower()
    if name.startswith("low"):
        return TOMS_LOW
    if name.startswith("high"):
        return TOMS_HIGH
    return TOMS_MID


def _inject_tom_fill(
    bar_events: List[Dict[str, Any]],
    bar_slice: pd.Series,
    next_bar_slice: Optional[pd.Series],
    palette: List[int],
    max_notes: int = 8,
    strength: float = 0.9,
    add_crash_on_next: bool = True,
):
    """
    小節末尾にタム主体のフィルを注入。既存のHHを少し間引き。
    - 16分（または12/16の混合）に並べる短いラチェット
    - 終端クラッシュ（次小節の頭）を追加（任意）
    """
    bar_start = float(bar_slice["start_beat"])
    bar_end = float(bar_slice["end_beat"])
    bar_len = bar_end - bar_start
    if bar_len <= 0.1:
        return

    # 既存HHを軽く間引き（最後の1拍付近）
    trimmed = []
    for e in bar_events:
        if _label_from_pitch(e["pitch"]).startswith("hh") and (e["start_beats"] > bar_end - 1.0):
            # 60%で削除
            if random() < 0.60:
                continue
        trimmed.append(e)
    bar_events.clear()
    bar_events.extend(trimmed)

    # Fill パターン生成
    n = max(3, min(max_notes, 12))
    step = min(0.25, (bar_len * 0.9) / n)  # 16分基準
    t0 = max(bar_start + bar_len * 0.5, bar_end - n * step - 0.05)  # 後半に寄せる
    vel_base = int(80 + 32 * strength)  # 80..112
    pitches = []
    # palette を上昇→下降で並べる簡易階段
    up = palette
    dn = list(reversed(palette))
    seq = (up + dn)[0:n]
    for i in range(n):
        pitches.append(seq[i % len(seq)])
    for i in range(n):
        t = t0 + i * step
        p = pitches[i]
        v = int(np.clip(np.random.normal(vel_base, 6), 40, 120))
        bar_events.append(
            {
                "start_beats": t,
                "end_beats": t + max(0.05, step * 0.7),
                "pitch": p,
                "velocity": v,
                "vel": v,
                "label": "tom",
            }
        )

    # 次小節頭にクラッシュ
    if add_crash_on_next and next_bar_slice is not None:
        t_downbeat = float(next_bar_slice["start_beat"])
        bar_events.append(
            {
                "start_beats": min(bar_end, t_downbeat),
                "end_beats": min(bar_end, t_downbeat) + 0.10,
                "pitch": 49,  # crash1
                "velocity": int(np.clip(vel_base + 6, 1, 127)),
                "vel": int(np.clip(vel_base + 6, 1, 127)),
                "label": "crash",
            }
        )


def _ensure_channel_and_vel(evts: List[Dict[str, Any]], channel: int = 9):
    for e in evts:
        # provide both vel and velocity (schema dual)
        v = int(np.clip(e.get("velocity", e.get("vel", 64)), 1, 127))
        e["velocity"] = v
        e["vel"] = v
        e["channel"] = channel
        # enforce minimal duration
        if (e.get("end_beats", e["start_beats"] + 0.05) - e["start_beats"]) < 0.02:
            e["end_beats"] = e["start_beats"] + 0.05


def _merge_sources_for_bar(
    bar_idx: int,
    bar_slice: pd.Series,
    stem_bar_events: List[Dict[str, Any]],
    rec_bar_pattern: Optional[List[Dict[str, Any]]],
    vocal_windows: List[Tuple[float, float]],
) -> List[Dict[str, Any]]:
    """
    Fusion policy (lightweight, safe-by-default):
      1) keep KICK/SNARE from stemMIDI if present; otherwise take from rec pattern.
      2) keep existing HH/ride/crash from stemMIDI if present; then fill missing HH to density_target.
      3) if drums_active==0: thin to sparse HH only (no ride/crash), keep backbeat snare if exists.
      4) apply vocal ducking.
    """
    out: List[Dict[str, Any]] = []
    # 1) Kick/Snare from stem first
    if stem_bar_events:
        ks = [e for e in stem_bar_events if _label_from_pitch(e["pitch"]) in ("kick", "snare")]
        cym = [
            e
            for e in stem_bar_events
            if _label_from_pitch(e["pitch"]) in ("hh_c", "hh_p", "hh_o", "ride", "crash")
        ]
        toms = [e for e in stem_bar_events if _label_from_pitch(e["pitch"]) == "tom"]
        out.extend(ks + cym + toms)
    # 1b) fallback to recommendations for kick/snare if missing
    if rec_bar_pattern:
        have_kick = any(_label_from_pitch(e["pitch"]) == "kick" for e in out)
        have_snare = any(_label_from_pitch(e["pitch"]) == "snare" for e in out)
        if not have_kick:
            out.extend([e for e in rec_bar_pattern if _label_from_pitch(e["pitch"]) == "kick"])
        if not have_snare:
            out.extend([e for e in rec_bar_pattern if _label_from_pitch(e["pitch"]) == "snare"])
        # take HH from rec only if stem was empty
        if not any(_label_from_pitch(e["pitch"]).startswith("hh") for e in out):
            out.extend(
                [e for e in rec_bar_pattern if _label_from_pitch(e["pitch"]).startswith("hh")]
            )

    # 2) ghost HH fill-up to density target
    density_target = float(bar_slice.get("density_target", 0.5))
    _add_ghost_hats(out, bar_slice["start_beat"], density_target, vocal_windows)

    # 3) break bars thinning
    if int(bar_slice.get("drums_active", 1)) == 0:
        # keep only sparse closed HH (quarter notes) and backbeat snare if any
        out = [e for e in out if _label_from_pitch(e["pitch"]) in ("snare", "hh_c")]
        # thin HH to 4 notes
        hh = [e for e in out if _label_from_pitch(e["pitch"]) == "hh_c"]
        others = [e for e in out if e not in hh]
        if len(hh) > 4:
            hh = sorted(hh, key=lambda x: x["start_beats"])
            step = max(1, len(hh) // 4)
            hh = hh[::step][:4]
        for e in hh:
            e["velocity"] = max(1, int(e["velocity"] * 0.8))
        out = others + hh

    # 4) vocal ducking
    _apply_ducking(out, vocal_windows)
    return out


def _should_trigger_fill(
    bars: pd.DataFrame, idx: int, modes: List[str], min_spacing_bars: int = 2
) -> bool:
    """
    フィルトリガー条件：
      - 'section' : 次小節が別セクション
      - 'cadence' : 4または8小節周期の終端
      - 'energy_rise' : 次小節で energy_curve が上がる
    過密防止のため、最低間隔 min_spacing_bars を守る（簡易：偶数バーのみ等）。
    """
    # 境界情報
    if idx >= len(bars) - 1:
        return False
    cur = bars.iloc[idx]
    nxt = bars.iloc[idx + 1]
    # 最低間隔（ざっくり：奇数バーのみ or 2小節以上空ける）
    if (idx % min_spacing_bars) != 0:
        return False
    vec = []
    if "section" in modes:
        vec.append(cur.get("section_label", "") != nxt.get("section_label", ""))
    if "cadence" in modes:
        vec.append(((idx + 1) % 4 == 0) or ((idx + 1) % 8 == 0))
    if "energy_rise" in modes:
        vec.append(float(nxt.get("energy_curve", 0.5)) > float(cur.get("energy_curve", 0.5)) + 0.07)
    return any(vec)


def _is_last_bar_before_section_change(bars: pd.DataFrame, idx: int) -> bool:
    """セクション変更直前の最終バーかどうか判定"""
    if idx >= len(bars) - 1:
        return False
    cur = bars.iloc[idx]
    nxt = bars.iloc[idx + 1]
    return cur.get("section_label", "") != nxt.get("section_label", "")


def _read_recommendations(path: str) -> Dict[str, Any]:
    if not path:
        return {}
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return {}


def _pattern_to_events(
    pattern: Dict[str, Any], bar_start: float, bar_end: float
) -> List[Dict[str, Any]]:
    """Very light translation of a recommended pattern item to events (expects GM pitches)."""
    evts: List[Dict[str, Any]] = []
    for e in pattern.get("events", []):
        s = float(e.get("start_beats", bar_start))
        d = float(e.get("dur_beats", e.get("duration_beats", 0.25)))
        p = int(e.get("pitch", 42))
        v = int(np.clip(e.get("velocity", e.get("vel", 64)), 1, 127))
        evts.append(
            {
                "start_beats": s,
                "end_beats": s + d,
                "pitch": p,
                "velocity": v,
                "label": _label_from_pitch(p),
            }
        )
    # clamp to bar
    for e in evts:
        e["start_beats"] = max(bar_start, e["start_beats"])
        e["end_beats"] = min(bar_end, e["end_beats"])
        if e["end_beats"] <= e["start_beats"]:
            e["end_beats"] = e["start_beats"] + 0.05
    return evts


def _recommendations_by_bar(
    recs: Dict[str, Any], bars: pd.DataFrame
) -> Dict[int, List[Dict[str, Any]]]:
    """Make a simple per-bar fallback pattern set if recommendations file is given."""
    by_bar: Dict[int, List[Dict[str, Any]]] = {}
    if not recs:
        return by_bar
    # heuristic: find a best pattern blob in recs (implementation-agnostic)
    patterns = recs.get("patterns", recs.get("best", []))
    # Allow a single 'pattern' with events to be used across bars
    template = None
    if isinstance(patterns, list) and patterns:
        template = patterns[0]
    elif isinstance(patterns, dict) and "events" in patterns:
        template = patterns
    for _, row in bars.iterrows():
        bar_idx = int(row["bar_index"])
        if template:
            by_bar[bar_idx] = _pattern_to_events(template, row["start_beat"], row["end_beat"])
        else:
            by_bar[bar_idx] = []
    return by_bar


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--recommendations",
        required=False,
        help="recommend_drums.py output (optional when using stem-midi)",
    )
    ap.add_argument("--out", required=True)
    ap.add_argument("--tempo-bpm", type=float, required=True)
    ap.add_argument("--bars", default=None)
    # NEW: hybrid sources
    ap.add_argument("--stem-midi", default=None, help="Path to drums stem MIDI (weak labels)")
    ap.add_argument(
        "--stems-features", default=None, help="Parquet with hat_density/fill_likelihood, etc."
    )
    ap.add_argument("--lyric-anchors", default=None, help="JSON with anchors to duck cymbals")
    # Tunables
    ap.add_argument("--duck-vel", type=int, default=12)
    ap.add_argument("--seed", type=int, default=7)
    # NEW: open-hat policy
    ap.add_argument(
        "--oh-open-prob",
        type=float,
        default=0.22,
        help="Base probability for turning 42→46 (+44 close)",
    )
    ap.add_argument(
        "--oh-close-delay", type=float, default=0.18, help="Beats until pedal-close (44)"
    )
    ap.add_argument(
        "--oh-min-gap", type=float, default=0.10, help="Minimum beats between two open-hats"
    )
    ap.add_argument("--oh-avoid-vocal", action="store_true", help="Avoid open-hat in vocal windows")
    # NEW: tom fill
    ap.add_argument("--enable-fills", action="store_true", help="Enable tom-based fills")
    ap.add_argument(
        "--fill-when", default="section,cadence", help="Comma list: section,cadence,energy_rise"
    )
    ap.add_argument("--fill-strength", type=float, default=0.9, help="0..1 intensity for fills")
    ap.add_argument("--fill-max-notes", type=int, default=8)
    ap.add_argument("--fill-palette", default="mid", help="low|mid|high palette for toms")
    ap.add_argument(
        "--fill-crash-next", action="store_true", help="Add crash on next-bar downbeat after fill"
    )

    # NEW: ride→tom 2段構え
    ap.add_argument("--fill-ride", action="store_true", help="Insert ride lead-in before tom fills")
    ap.add_argument(
        "--fill-ride-window", type=float, default=1.0, help="Lead-in window in beats (0.25..2.0)"
    )
    ap.add_argument(
        "--fill-ride-rate", type=float, default=0.5, help="Ride spacing in beats (0.125..0.5)"
    )
    ap.add_argument("--fill-ride-pitch", type=int, default=51, help="Ride pitch (51/53/59 etc.)")
    # NEW: snare flam
    ap.add_argument("--flam-enable", action="store_true", help="Enable light snare flam/drag")
    ap.add_argument("--flam-ms", type=float, default=18.0, help="Flam pre-delay in ms (5..28)")
    ap.add_argument(
        "--flam-vel-ratio", type=float, default=0.65, help="Flam velocity ratio (0.3..0.9)"
    )
    ap.add_argument(
        "--flam-prob", type=float, default=0.55, help="Per-hit probability to add flam (0..1)"
    )
    ap.add_argument("--flam-avoid-vocal", action="store_true", help="Avoid flam in vocal windows")

    # NEW: backbeat保障とKPI強化
    ap.add_argument(
        "--enforce-backbeat",
        action="store_true",
        help="Ensure snare hits exist and accent on 2/4 where groove active",
    )
    ap.add_argument(
        "--min-backbeat-vel", type=int, default=86, help="Minimum snare velocity on beats 2/4"
    )
    ap.add_argument(
        "--light-flam", action="store_true", help="Inject tiny snare flam around 2/4 (+/-12ms)"
    )
    ap.add_argument(
        "--fill-l2",
        action="store_true",
        help="Ride→tom two-stage fill on phrase ends (last 1 bar before section change)",
    )

    args = ap.parse_args()

    np.random.seed(args.seed)
    tempo_bpm = float(args.tempo_bpm)
    bars = _load_bars(args.bars, tempo_bpm)

    # stem features → merge into bars if present
    stems_df = _load_stem_features(args.stems_features)
    if stems_df is not None and len(stems_df):
        # select useful columns if exist
        cols = [
            c
            for c in [
                "bar_index",
                "hat_density",
                "fill_likelihood",
                "density_target",
                "drums_active",
            ]
            if c in stems_df.columns
        ]
        if "bar_index" not in cols and "bar" in stems_df.columns:
            stems_df["bar_index"] = stems_df["bar"]
            cols.append("bar_index")
        if "density_target" not in cols and "hat_density" in stems_df.columns:
            tmp = stems_df["hat_density"].clip(0, 6) / 8.0 + 0.2
            stems_df["density_target"] = tmp.clip(0.2, 0.95)
            cols.append("density_target")
        m = bars.merge(
            stems_df[cols].drop_duplicates("bar_index"),
            on="bar_index",
            how="left",
            suffixes=("", "_stem"),
        )
        # prefer stems' drums_active if provided
        if "drums_active_stem" in m.columns:
            m["drums_active"] = m["drums_active_stem"].fillna(m["drums_active"])
            m = m.drop(
                columns=[c for c in m.columns if c.endswith("_stem") and c != "drums_active_stem"],
                errors="ignore",
            )
            m = m.drop(columns=["drums_active_stem"], errors="ignore")
        bars = m

    # recommendations (fallback patterns)
    recs = _read_recommendations(args.recommendations) if args.recommendations else {}
    rec_by_bar = _recommendations_by_bar(recs, bars)

    # stem MIDI (weak labels)
    stem_evts = _load_stem_midi(args.stem_midi, tempo_bpm) if args.stem_midi else []
    stem_by_bar = _collect_by_bar(stem_evts, bars) if stem_evts else {}

    # vocal windows (beats)
    vocal_windows = _load_anchors(args.lyric_anchors, tempo_bpm)

    # fuse per bar
    all_events: List[Dict[str, Any]] = []
    fill_modes = [s.strip() for s in str(args.fill_when).split(",") if s.strip()]
    palette = _pick_tom_palette(args.fill_palette)
    bars = bars.sort_values("bar_index").reset_index(drop=True)
    for i, row in bars.iterrows():
        b = int(row["bar_index"])
        stem_bar = stem_by_bar.get(b, [])
        rec_bar = rec_by_bar.get(b, [])
        bar_evts = _merge_sources_for_bar(b, row, stem_bar, rec_bar, vocal_windows)

        # open-hat policy（任意）
        _apply_open_hat_policy(
            bar_evts,
            row,
            vocal_windows,
            open_prob=args.oh_open_prob,
            close_delay_beats=args.oh_close_delay,
            min_gap_beats=args.oh_min_gap,
            avoid_vocal=args.oh_avoid_vocal,
        )

        # tom-based fill（任意）
        if args.enable_fills and _should_trigger_fill(
            bars, i, modes=fill_modes, min_spacing_bars=2
        ):
            next_row = bars.iloc[i + 1] if i < len(bars) - 1 else None
            _inject_tom_fill(
                bar_evts,
                row,
                next_row,
                palette=palette,
                max_notes=args.fill_max_notes,
                strength=float(args.fill_strength),
                add_crash_on_next=bool(args.fill_crash_next),
            )

        # --- KPI向けの最終小技 ---
        # Backbeat保障（スネア2拍4拍の存在確保とvelocity強化）
        if args.enforce_backbeat:
            drums_active = row.get("drums_active", 1)
            if drums_active:
                kit = {"snare": 38}
                _ensure_backbeat(
                    bar_evts,
                    b,
                    min_vel=args.min_backbeat_vel,
                    snare_pitch=kit["snare"],
                    flam=args.light_flam,
                    flam_ms=12.0,
                    tempo_bpm=tempo_bpm,
                )

        # ライド→タム2段フィル（セクション変更直前のみ）
        if args.fill_l2 and _is_last_bar_before_section_change(bars, i):
            kit = {
                "ride": 51,
                "tom_hi": 48,
                "tom_mid": 47,
                "tom_low": 45,
            }
            _two_stage_fill(bar_evts, b, kit)

        _ensure_channel_and_vel(bar_evts, channel=9)
        all_events.extend(bar_evts)

    plan = {
        "meta": {
            "role": "Drums",
            "version": "hybrid_v2",
            "tempo_bpm": tempo_bpm,
            "source": {
                "recommendations": bool(recs),
                "stem_midi": bool(stem_evts),
                "stems_features": bool(stems_df is not None and len(stems_df)),
                "lyric_anchors": bool(vocal_windows),
                "tom_fills": bool(args.enable_fills),
                "open_hat": True,
            },
        },
        "tracks": [{"name": "Drums", "is_drum": 1, "channel": 9, "events": all_events}],
    }

    # final vel/velocity sync & clamp
    for e in plan["tracks"][0]["events"]:
        v = int(np.clip(e.get("velocity", e.get("vel", 64)), 1, 127))
        e["velocity"] = v
        e["vel"] = v

    Path(args.out).write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
