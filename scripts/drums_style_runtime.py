# -*- coding: utf-8 -*-
"""
drums_style_runtime.py
 - 空ドラム検知と最小オートリジェネ（バックビート＋ハイハット）
 - スタイルYAMLの読み込み（密度・ブレイク率・フィル頻度/セクション別）
 - アクセント計画（drum_accent_plan.json）とセクション切替に応じたクラッシュ/フィル
想定イベント形式:
  {"bar": int, "start_beats": float, "end_beats": float, "pitch": int, "velocity": int}
"""

from __future__ import annotations
import json, math, random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np
import yaml

# General MIDI percussion note numbers (channel 10 / pitch map)
KICK = 36
SNARE = 38
CLAP = 39
HAT_C = 42
HAT_O = 46
CRASH = 49
RIDE = 51
TOM_H = 50
TOM_M = 47
TOM_L = 45

def jload(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))

def try_jload(p: Optional[Path]):
    try:
        if p and Path(p).exists():
            return jload(Path(p))
    except Exception:
        pass
    return None

def load_style_yaml(p: Optional[str]) -> Dict:
    """スタイルYAMLを読み込み。無指定/不在時は安全な既定値。"""
    default = {
        "profile": "pop_standard",
        "density_by_section": {"intro":0.5, "verse":0.6, "bridge":0.7, "chorus":0.9, "outro":0.5},
        "break_rate_by_section": {"intro":0.10, "verse":0.05, "bridge":0.06, "chorus":0.02, "outro":0.08},
        "fill_freq_by_section": {"intro":0.10, "verse":0.12, "bridge":0.18, "chorus":0.20, "outro":0.14},
        "hat_open_ratio": {"low":0.05, "mid":0.10, "high":0.18},
        "ghost_snare": True,
        "swing": 0.0,
        "crash_on_transitions": True
    }
    if not p:
        return default
    path = Path(p)
    if not path.exists():
        return default
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        # merge shallowly with defaults
        for k,v in default.items():
            if k not in data:
                data[k] = v
        return data
    except Exception:
        return default

def _secname(label: str) -> str:
    s = (label or "").lower()
    for key in ["intro","verse","pre","bridge","chorus","outro"]:
        if s.startswith(key):
            return key
    return "verse"

def _section_density(sec_label: str, style: Dict) -> float:
    m = style.get("density_by_section", {})
    return float(m.get(_secname(sec_label), 0.6))

def _section_break_rate(sec_label: str, style: Dict) -> float:
    m = style.get("break_rate_by_section", {})
    return float(m.get(_secname(sec_label), 0.06))

def _section_fill_freq(sec_label: str, style: Dict) -> float:
    m = style.get("fill_freq_by_section", {})
    return float(m.get(_secname(sec_label), 0.12))

def _energy_band(e: float) -> str:
    if e >= 0.7: return "high"
    if e >= 0.5: return "mid"
    return "low"

def _is_transition_bar(b: int, bars: pd.DataFrame) -> bool:
    r = bars[bars["bar_index"]==b]
    if r.empty: 
        return False
    s = str(r.iloc[0].get("section_label",""))
    if (b+1) in set(bars["bar_index"].values):
        n = str(bars[bars["bar_index"]==b+1].iloc[0].get("section_label",""))
        return s != n
    return False

def _density_to_hat_step(density: float) -> float:
    """密度からハイハット間隔(拍)を決める。0.5以上で8分、0.8以上で16分寄り。"""
    if density >= 0.85: return 0.25   # 16th
    if density >= 0.60: return 0.5    # 8th
    return 1.0                         # quarter

def _emit(events: List[Dict], bar:int, on:float, off:float, pitch:int, vel:int):
    events.append({"bar":bar, "start_beats":on, "end_beats":off, "pitch":pitch, "velocity":vel})

def _tom_fill(events: List[Dict], bar:int, start_b: float, end_b: float, base_vel:int=92):
    """簡易フィル（16分3発＋最後クラッシュ）"""
    t = max(start_b, end_b-1.0)
    step = 0.25
    seq = [TOM_L, TOM_M, TOM_H]
    vels = [base_vel, base_vel-4, base_vel-6]
    for i,p in enumerate(seq):
        on = t + i*step
        if on < end_b-0.05:
            _emit(events, bar, on, min(end_b, on+0.20), p, vels[i])
    # 直後クラッシュは呼び出し側（遷移バー）で入れることが多い

def _break_bar(events: List[Dict], bar:int, start_b:float, end_b:float, hat_step:float):
    """休符を多めにしたブレイク。頭に軽いキックのみ、ハットは1-2発。"""
    _emit(events, bar, start_b, start_b+0.20, KICK, 76)
    # sparse hats
    _emit(events, bar, start_b+hat_step, start_b+hat_step+0.15, HAT_C, 62)

def auto_regen_minimal(
    bars: pd.DataFrame,
    energy_s: Optional[pd.Series],
    style_yaml: Optional[str],
    accent_plan_path: Optional[str],
    rng_seed: int = 777
) -> Tuple[List[Dict], Dict]:
    """
    最小ドラム自動生成（バックビート＋可変ハイハット＋遷移クラッシュ＋簡易フィル）
    - sections/energy から密度/開ハット率を決定
    - accent_plan（任意）を見て break_bars / crash_points を尊重
    戻り値: (events, meta)
    """
    style = load_style_yaml(style_yaml)
    accent = try_jload(Path(accent_plan_path) if accent_plan_path else None) or {}
    # accent例: {"break_bars":[12,28], "crash_bars":[33], "fills":{"bars":[15,31]}}
    break_bars = set(accent.get("break_bars", []))
    crash_bars = set(accent.get("crash_bars", []))
    fill_bars = set(accent.get("fills", {}).get("bars", []))

    rng = random.Random(rng_seed)

    events: List[Dict] = []
    for _, row in bars.sort_values("bar_index").iterrows():
        b = int(row["bar_index"])
        s = float(row.get("start_beats", b*4.0))
        e = float(row.get("end_beats", s+4.0))
        sec = str(row.get("section_label",""))
        energy = float(row.get("energy_curve", row.get("energy", 0.6)))
        if energy_s is not None and b in energy_s.index and not math.isnan(energy_s[b]):
            energy = float(energy_s[b])

        density = _section_density(sec, style)
        hat_step = _density_to_hat_step(density)
        open_ratio = float(style.get("hat_open_ratio", {}).get(_energy_band(energy), 0.10))
        do_break = (b in break_bars) or (rng.random() < _section_break_rate(sec, style))

        # ハット
        if do_break:
            _break_bar(events, b, s, e, hat_step)
        else:
            t = s
            while t < e - 0.05:
                is_open = rng.random() < open_ratio
                pit = HAT_O if is_open else HAT_C
                vel = 66 if not is_open else 72
                _emit(events, b, t, min(e, t+0.18), pit, vel)
                t += hat_step

            # バックビート
            _emit(events, b, s, s+0.15, KICK, 90)
            beat2 = s + 2.0
            _emit(events, b, beat2, beat2+0.15, SNARE, 96)

            # 追加キック（密度によって 3拍目に入れる）
            if density >= 0.6:
                beat3 = s + 2.95 if hat_step <= 0.5 else s + 3.0
                _emit(events, b, beat3, beat3+0.12, KICK, 84)

        # セクション遷移クラッシュ
        if style.get("crash_on_transitions", True):
            if (b in crash_bars) or _is_transition_bar(b, bars):
                _emit(events, b, s, s+0.40, CRASH, 110)

        # フィル（小節末）
        if (b in fill_bars) or (rng.random() < _section_fill_freq(sec, style)):
            _tom_fill(events, b, s, e, base_vel=92)
            # 小節頭クラッシュ
            _emit(events, b, e-0.01, e, CRASH, 105)  # 次小節頭へ繋がる意図

    meta = {"auto_regen": True, "style_profile": style.get("profile","pop_standard")}
    return events, meta

def ensure_nonempty_drums(events: List[Dict], min_events: int, bars: pd.DataFrame,
                          energy_s: Optional[pd.Series], style_yaml: Optional[str],
                          accent_plan_path: Optional[str], seed: int) -> Tuple[List[Dict], Dict]:
    """
    既存eventsが少なすぎる場合、最小自動生成に切替。
    """
    if events is None or len(events) < int(min_events):
        auto_events, meta = auto_regen_minimal(bars, energy_s, style_yaml, accent_plan_path, seed)
        return auto_events, {"auto_regen": True, **meta}
    return events, {"auto_regen": False}
